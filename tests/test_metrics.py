from datetime import timedelta
import json
import math
from pathlib import Path
import pytest

from TwinTurbo.ai.agents.critic import Critic
from TwinTurbo.ai.agents.forecaster import Forecaster
from TwinTurbo.ai.evaluate import (TemporalFold, compare_forecasts, interval_metrics, pinball_loss,
    point_metrics, residuals_from_forecasts, walk_forward, write_report)
from TwinTurbo.ai.models.baseline import ConstantBaseline
from TwinTurbo.ai.models.bias import update_bias
from TwinTurbo.ai.schemas import ForecastResult, PredictionBatch, PredictionRow, digest
from .test_models import T0, TURBINES, history, observation, prepared_experiment, snapshot
from .test_bias import residual


def forecast(origin=T0, bias=None, model=None, observations=()):
    model = model or ConstantBaseline.fit(snapshot(origin-timedelta(days=1)))
    snap = snapshot(origin, observations=observations)
    trace = Forecaster(model).predict_with_trace(snap,bias)
    result = ForecastResult(forecast_id=digest({"origin":origin.isoformat(),"bias":bias.bias_id if bias else None}),
        origin_time=origin,predictions=trace.predictions,run_id=snap.weather_run_metadata.run_id,
        model_id=model.state.model_id,bias_id=bias.bias_id if bias else None,provenance="synthetic",
        mode="fixture",release_kind="scheduled",manifest={"model":model.state.model_dump(mode="json")})
    return result,trace.base_predictions


def test_known_metrics_and_zero_targets():
    metrics = point_metrics([0,0.5,1],[0.1,0.5,0.8])
    assert metrics["mae"] == pytest.approx(0.1)
    assert metrics["rmse"] == pytest.approx(math.sqrt(0.05/3))
    assert metrics["mean_error"] == pytest.approx(-0.1/3)
    assert pinball_loss([0,1],[0.5,0.5],0.1) == pytest.approx(0.25)
    assert point_metrics([],[])["mae"] is None
    with pytest.raises(ValueError):
        point_metrics([float("nan")],[0])


def test_interval_coverage_width_and_pinball():
    metrics = interval_metrics([0.2,0.5,0.9],[0.1,0.4,0.6],[0.2,0.5,0.7],[0.3,0.6,0.8])
    assert metrics["coverage"] == pytest.approx(2/3)
    assert metrics["mean_width"] == pytest.approx(0.2)
    assert metrics["pinball_q50"] == pytest.approx(0.1/3)


def test_comparison_uses_common_keys_and_reports_failed_rows():
    f,_ = forecast()
    limited = f.model_copy(update={"forecast_id":"limited","predictions":PredictionBatch(rows=f.predictions.rows[:24])})
    actuals = tuple(observation(T0+timedelta(hours=h),power=0,turbine=t) for t in TURBINES for h in range(1,49))
    expected = {(T0,r.turbine_id,r.target_start,r.target_end) for r in f.predictions.rows}
    report = compare_forecasts({"full":[f],"partial":[limited]},actuals,as_of=T0+timedelta(days=3),expected_keys=expected)
    assert report["matched_sample_count"] == 24
    assert report["reports"]["partial"]["forecast_coverage"] == 0.25
    assert report["coverage"]["partial"]["release_coverage"] == 0
    assert report["reports"]["full"]["sample_count"] == report["reports"]["partial"]["sample_count"]
    assert report["reports"]["full"]["metrics_by_turbine_and_lead"]["turbine_2"]["all"]["mae"] is None


def test_missing_and_immature_actuals_not_scored_as_zero():
    f,_ = forecast()
    actual = observation(T0+timedelta(hours=1),power=0.3)
    report = compare_forecasts({"base":[f]},[actual],as_of=T0+timedelta(hours=1))
    assert report["matched_sample_count"] == 0
    report = compare_forecasts({"base":[f]},[actual],as_of=actual.available_at)
    assert report["matched_sample_count"] == 1
    assert report["missing_or_immature_actual_rows"] == 95


def test_critic_requires_original_base_after_clipping_and_is_idempotent():
    model = ConstantBaseline.fit(snapshot(T0-timedelta(days=1)))
    bias = update_bias([residual(i,model=model.state.model_id,actual=1,base=0) for i in range(5)],
                     model_id=model.state.model_id,as_of=T0)
    f,base = forecast(bias=bias,model=model)
    actual = observation(T0+timedelta(hours=1),0.8)
    at = T0+timedelta(hours=3)
    with pytest.raises(ValueError,match="BASE_PREDICTIONS_REQUIRED"):
        residuals_from_forecasts([f],[actual],as_of=at)
    errors = residuals_from_forecasts([f],[actual],as_of=at,base_batches={f.forecast_id:base})
    assert errors[0].p_base == pytest.approx(0.38)
    assert errors[0].p_issued == 1.0
    critic = Critic()
    first = critic.review([f],[actual],model_id=model.state.model_id,as_of=at,base_batches={f.forecast_id:base})
    assert first.action == "propose_bias"
    second = critic.review([f],[actual],model_id=model.state.model_id,as_of=at,
        base_batches={f.forecast_id:base},previous=first.proposed_bias)
    assert second.action == "skip" and "NO_NEW_ACTUALS" in second.reasons


def test_critic_no_actual_has_explicit_reason():
    f,_ = forecast()
    decision = Critic().review([f],[],model_id=f.model_id,as_of=T0)
    assert decision.action == "skip"
    assert decision.reasons == ("NO_MATURE_ERRORS",)
    assert decision.proposed_bias is None


def test_time_ordered_folds_reject_random_origin_order():
    fold = TemporalFold(snapshot(T0-timedelta(days=1)),(snapshot(T0+timedelta(days=1)),snapshot(T0)))
    with pytest.raises(ValueError,match="MUST_INCREASE"):
        walk_forward([fold],[],as_of=T0+timedelta(days=3))


def test_online_correction_cannot_use_future_poisoned_labels():
    train = snapshot(T0-timedelta(days=1))
    first,second = snapshot(T0),snapshot(T0+timedelta(days=1))
    actuals = tuple(observation(T0+timedelta(hours=h),0.6,turbine=t) for t in TURBINES for h in range(1,73))
    folds = [TemporalFold(train,(first,second))]
    at = T0+timedelta(days=4)
    _, original = walk_forward(folds,actuals,as_of=at)
    poisoned = tuple(o.model_copy(update={"power_norm":0.0}) if o.available_at > second.origin_time else o for o in actuals)
    _, changed = walk_forward(folds,poisoned,as_of=at)
    assert original == changed


def run_fixture_report(path):
    folds,observations,as_of = prepared_experiment()
    report,forecasts = walk_forward(folds,observations,as_of=as_of)
    report["fixture"] = {"generator":"tests.test_models.prepared_experiment", "seed":42,
                         "warning":"SYNTHETIC TEST DATA ONLY; NOT REAL TURBINE PERFORMANCE",
                         "fold_count":len(folds), "validation_origins":sum(len(f.validation) for f in folds)}
    report["input_sha256"] = digest({"folds":[{"training":f.training.model_dump(mode="json"),
        "validation":[s.model_dump(mode="json") for s in f.validation]} for f in folds],
        "observations":[o.model_dump(mode="json") for o in observations],"as_of":as_of.isoformat()})
    if path is not None:
        write_report(report,path)
    return report,forecasts


def test_reproducible_fixture_report(tmp_path):
    path = tmp_path/"fixture.json"
    report,forecasts = run_fixture_report(path)
    assert not report["failures"]
    assert report["fixture_only"] is True
    assert report["matched_sample_count"] == 14*96
    assert all(len(fs) == 14 for fs in forecasts.values())
    assert json.loads(path.read_text()) == report
    for t in TURBINES:
        m = report["reports"]
        baseline = m["baseline"]["metrics_by_turbine_and_lead"][t]["all"]["mae"]
        curve = m["curve"]["metrics_by_turbine_and_lead"][t]["all"]["mae"]
        corrected = m["curve_bias"]["metrics_by_turbine_and_lead"][t]["all"]["mae"]
        assert corrected < curve < baseline
        assert m["curve_bias"]["interval_metrics"][t]["all"]["sample_count"] > 0


def run_real_cache_report(path, config_path, origins):
    """Read already prepared data/cache using participant 1 APIs, with no download."""
    from TwinTurbo.ai.config import load_config
    from TwinTurbo.ai.service import ForecastService
    from TwinTurbo.ai.store import Store
    from TwinTurbo.ai.weather.archive import GFSArchive
    from TwinTurbo.ai.schemas import ForecastRequest
    from TwinTurbo.ai.models.power_curve import PowerCurvePredictor
    from TwinTurbo.ai.models.registry import save_predictor
    config = load_config(config_path)
    if not Path(config.storage.database).is_file():
        raise ValueError("PREPARED_DATABASE_REQUIRED: run participant 1's ingest command first")
    store = Store(config.storage.database)
    service = ForecastService(config, store, GFSArchive(config))
    snapshots = tuple(service.snapshot(ForecastRequest(origin_time=origin,
        turbine_ids=tuple(t.id for t in config.site.turbines), mode="replay")) for origin in origins)
    as_of = max(origins)+timedelta(hours=50)
    observations = store.observations_as_of(as_of, tuple(t.id for t in config.site.turbines))
    report, forecasts = walk_forward([TemporalFold(snapshots[0], snapshots)], observations, as_of=as_of)
    curve = PowerCurvePredictor.fit(snapshots[0])
    artifact = Path('artifacts/models')/(curve.state.model_id+'.json')
    save_predictor(curve, artifact)
    report['model_artifact'] = str(artifact)
    report['training_counts'] = {t:c.train_count for t,c in curve.curves.items()}
    report['curve_bin_counts'] = {t:len(c.wind) for t,c in curve.curves.items()}
    report['config'] = config.model_dump(mode='json')
    report['source_revisions'] = sorted({o.revision for o in observations})
    report['weather_runs'] = [s.weather_run_metadata.model_dump(mode='json') for s in snapshots]
    report['warnings'] = sorted({w for s in snapshots for w in s.quality_flags})
    report['limitations'] = [
        'Short chronological cache experiment, not a full-month or final competition score.',
        'Source SCADA timezone remains assumed in the supplied configuration; alignment is unconfirmed.',
        'Grid forecast wind is used without site calibration; measured-wind power curves may underperform the baseline.',
        'No ML or ensemble trained on this short archive; no claim of calibrated 80% coverage.',
        'Training/activation is synchronous at the first origin; no simulated compute delay.'
    ]
    report['input_sha256'] = digest({'snapshots':[s.model_dump(mode='json') for s in snapshots],
                                    'observations':[o.model_dump(mode='json') for o in observations]})
    write_report(report,path)
    return report, forecasts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TwinTurbo.ai synthetic fixture comparison; never real quality")
    parser.add_argument("--output",default="reports/forecast-models/fixture-comparison.json")
    parser.add_argument("--real-cache", action="store_true", help="Use participant 1's existing DB and GFS cache")
    parser.add_argument("--config", default="configs/site.example.yaml")
    parser.add_argument("--origins", nargs="+", help="Explicit aware UTC origins, chronological")
    args = parser.parse_args()
    if args.real_cache:
        from datetime import datetime
        if not args.origins:
            parser.error("--real-cache requires --origins")
        report,_ = run_real_cache_report(args.output,args.config,
            [datetime.fromisoformat(x.replace('Z','+00:00')) for x in args.origins])
    else:
        report,_ = run_fixture_report(args.output)
    print(json.dumps({"output":args.output,"fixture_only":report["fixture_only"],
                      "matched_rows":report["matched_sample_count"],"failures":report["failures"]}))


def test_training_failure_is_reported_in_coverage_not_hidden():
    train = snapshot(T0-timedelta(days=1))
    validation = snapshot(T0)
    def unavailable(_):
        raise ValueError('insufficient archive history')
    report,_ = walk_forward([TemporalFold(train,(validation,))],[],as_of=T0+timedelta(days=3),
        fitters={'baseline':ConstantBaseline.fit,'unavailable':unavailable},bias_models=())
    assert report['coverage']['baseline']['forecast_rows'] == 96
    assert report['coverage']['unavailable']['forecast_rows'] == 0
    assert report['matched_sample_count'] == 0
    assert report['failures'][0]['reason'].startswith('MODEL_TRAINING_FAILED')
