from datetime import timedelta
import importlib
import json

import pytest

from windoracle.agents.twin_builder import TwinBuilder
from windoracle.backtest import TemporalFold, walk_forward_snapshots
from windoracle.features import latest_observations, supervised_examples
from windoracle.models.baseline import PersistenceBaseline
from windoracle.models.registry import load_predictor, save_predictor
from windoracle.schemas import ForecastRequest, Observation
from .conftest import ORIGIN
from .test_models import snapshot


def actual(turbine, start, power=.4):
    return Observation(turbine_id=turbine, event_start=start, event_end=start + timedelta(hours=1),
        available_at=start + timedelta(hours=1, minutes=15), power_norm=power, wind_ms=5,
        temperature_c=12, n_samples=6, coverage=1, quality_flag="complete", revision="test")


def test_namespaces_share_contract_and_model_identity():
    for name in ("schemas", "service", "models.registry", "models.power_curve"):
        assert importlib.import_module("TwinTurbo.ai." + name) is importlib.import_module("windoracle." + name)
    import TwinTurbo.ai.schemas
    assert TwinTurbo.ai.schemas.ForecastRequest is ForecastRequest


def test_persistence_rejects_stale_history_and_roundtrips(tmp_path):
    source = snapshot()
    base = TwinBuilder(provenance="synthetic").build_baseline(source)
    predictor = PersistenceBaseline(base.state, base.means)
    assert len(predictor.predict(source).rows) == 4
    stale = source.model_copy(update={"observations": source.observations[:4]})
    with pytest.raises(ValueError, match="STALE_PERSISTENCE"):
        predictor.predict(stale)
    path = save_predictor(predictor, tmp_path / "persistence.json")
    assert load_predictor(path).predict(source) == predictor.predict(source)


def test_revision_selection_precedes_quality_filter():
    old = snapshot().observations[0]
    new = old.model_copy(update={"available_at": ORIGIN, "revision": "corrected", "quality_flag": "invalid", "power_norm": None})
    assert latest_observations((old, new), ORIGIN) == (new,)
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        latest_observations((old, old.model_copy(update={"revision": "other"})), ORIGIN)


def test_display_context_freezes_weather_and_hides_future_actual(setup):
    setup.store.ingest(snapshot().observations, {})
    req = ForecastRequest(origin_time=ORIGIN, turbine_ids=("turbine_1", "turbine_2"), mode="fixture")
    setup.predictor = TwinBuilder(provenance="synthetic").build(setup.snapshot(req))
    result = setup.create_forecast(req)
    row = actual("turbine_1", ORIGIN + timedelta(hours=1))
    setup.store.ingest((row,), {})
    early = setup.get_display_context(result.forecast_id, as_of=ORIGIN)
    assert len(early["weather"]) == 96
    assert early["actuals"] == []
    late = setup.get_display_context(result.forecast_id, as_of=row.available_at)
    assert len(late["actuals"]) == 1
    assert late["weather"] == early["weather"]
    assert len(result.manifest["base_predictions"]["rows"]) == 96
    with pytest.raises(ValueError, match="FUTURE_DATA"):
        setup.get_display_context(result.forecast_id, as_of=ORIGIN - timedelta(hours=1))


def test_boosting_uses_only_mature_forecast_labels_and_requires_trust(tmp_path):
    pytest.importorskip("sklearn")
    from windoracle.models.boosting import GradientBoostingPredictor
    history = snapshot()
    labels = tuple(actual(v.turbine_id, v.valid_time) for v in history.weather_values)
    cutoff = ORIGIN + timedelta(hours=4)
    training = history.model_copy(update={"origin_time": cutoff, "observations": history.observations + labels,
        "target_intervals": tuple(t.model_copy(update={"target_start": t.target_start + timedelta(hours=5),
             "target_end": t.target_end + timedelta(hours=5)}) for t in history.target_intervals),
        "weather_values": tuple(v.model_copy(update={"valid_time": v.valid_time + timedelta(hours=5)}) for v in history.weather_values)})
    assert supervised_examples((history,), labels, training_cutoff=ORIGIN, allow_synthetic=True) == ()
    with pytest.raises(ValueError, match="ML_REQUIRES_OPERATIONAL"):
        supervised_examples((history,), labels, training_cutoff=cutoff)
    model = GradientBoostingPredictor.fit(training, (history,), allow_synthetic=True,
        min_samples=2, max_iter=2, min_samples_leaf=1)
    assert model.state.max_label_available_at <= cutoff
    assert model.state.provenance == "synthetic"
    path = save_predictor(model, tmp_path / "boosting.json")
    with pytest.raises(ValueError, match="TRUSTED"):
        load_predictor(path)
    assert load_predictor(path, trusted=True).predict(training) == model.predict(training)


def test_prepared_backtest_scores_common_keys_and_rejects_reverse_origins():
    source = snapshot()
    labels = tuple(actual(v.turbine_id, v.valid_time) for v in source.weather_values)
    report, outputs = walk_forward_snapshots((TemporalFold(source, (source,)),), labels,
        as_of=ORIGIN + timedelta(hours=4))
    assert report["matched_sample_count"] == 4
    assert report["fixture_only"]
    assert report["failures"] == []
    assert set(outputs) == {"baseline", "curve", "curve_bias"}
    assert all(value == 1 for value in report["forecast_coverage"].values())
    with pytest.raises(ValueError, match="VALIDATION_ORIGINS"):
        walk_forward_snapshots((TemporalFold(source, (source, source)),), labels,
            as_of=ORIGIN + timedelta(hours=4))


def test_prepared_backtest_cli(tmp_path):
    from windoracle.backtest import main
    source = snapshot()
    labels = tuple(actual(v.turbine_id, v.valid_time) for v in source.weather_values)
    data = {"folds": [{"training": source.model_dump(mode="json"), "validation": [source.model_dump(mode="json")]}],
            "observations": [o.model_dump(mode="json") for o in labels],
            "as_of": (ORIGIN + timedelta(hours=4)).isoformat()}
    source_path, report_path = tmp_path / "input.json", tmp_path / "report.json"
    source_path.write_text(json.dumps(data), encoding="utf-8")
    assert main(["--input", str(source_path), "--output", str(report_path)]) == 0
    assert json.loads(report_path.read_text())["matched_sample_count"] == 4


def test_backtest_updates_bias_only_after_actual_arrival(setup):
    setup.store.ingest(snapshot().observations, {})
    facts = tuple(actual(t, ORIGIN + timedelta(hours=h))
                  for t in ("turbine_1", "turbine_2") for h in range(1, 28))
    setup.store.ingest(facts, {})
    snapshots = tuple(setup.snapshot(ForecastRequest(origin_time=ORIGIN + timedelta(hours=h),
        turbine_ids=("turbine_1", "turbine_2"), horizon_hours=24, mode="fixture")) for h in (0, 3))
    report, forecasts = walk_forward_snapshots([TemporalFold(snapshots[0], snapshots)], facts,
        as_of=ORIGIN + timedelta(hours=30))
    assert report["failures"] == []
    assert forecasts["curve_bias"][0].bias_id is None
    corrected = forecasts["curve_bias"][1]
    assert corrected.bias_id is not None
    from windoracle.evaluate import residuals_from_forecasts
    residuals = residuals_from_forecasts(forecasts["curve_bias"], facts, as_of=ORIGIN + timedelta(hours=30))
    assert residuals
    assert len(corrected.manifest["base_predictions"]["rows"]) == 48
