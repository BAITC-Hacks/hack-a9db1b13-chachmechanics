"""Participant 2's explicit synthetic fixtures; never production weather/data."""
from datetime import datetime, timedelta, timezone
from dataclasses import replace
import json
import math
import numpy as np
import pytest

from TwinTurbo.ai.features import (features_from_snapshot, latest_observations,
                                   supervised_examples, training_observations)
from TwinTurbo.ai.models.baseline import ConstantBaseline, PersistenceBaseline
from TwinTurbo.ai.models.power_curve import PowerCurvePredictor
from TwinTurbo.ai.models.registry import load_predictor, save_predictor
from TwinTurbo.ai.schemas import (AsOfSnapshot, ForecastRequest, Observation, TargetInterval,
                                 WeatherRunMetadata, WeatherValue)

T0 = datetime(2025, 6, 1, 18, tzinfo=timezone.utc)
TURBINES = ("turbine_1", "turbine_2")


def observation(start, power=0.4, wind=5.0, turbine="turbine_1", delay=0, revision="fixture-v1"):
    return Observation(turbine_id=turbine, event_start=start, event_end=start+timedelta(hours=1),
        available_at=start+timedelta(hours=1, minutes=delay), power_norm=power, wind_ms=wind,
        temperature_c=10, n_samples=6, coverage=1, quality_flag="complete", revision=revision)


def history(origin=T0, count=240):
    return tuple(observation(origin-timedelta(hours=count-i+1),
        power=(i % 20)/25 * scale, wind=(i % 20)/2, turbine=t)
        for t, scale in zip(TURBINES, (1.0, 0.5)) for i in range(count))


def snapshot(origin=T0, observations=None, wind=5.0, horizon=48):
    obs = history(origin) if observations is None else tuple(observations)
    return AsOfSnapshot(origin_time=origin, observations=obs,
        weather_run_metadata=WeatherRunMetadata(run_id="fixture-"+origin.isoformat(), provider="fixture",
            model="synthetic", run_init_time=origin-timedelta(hours=6), available_at=origin,
            availability_basis="synthetic", provenance="synthetic", retrieved_at=T0,
            sha256="0"*64),
        target_intervals=tuple(TargetInterval(target_start=origin+timedelta(hours=h),
            target_end=origin+timedelta(hours=h+1)) for h in range(1, horizon+1)),
        weather_values=tuple(WeatherValue(turbine_id=t, valid_time=origin+timedelta(hours=h),
            wind_ms=wind, temperature_c=10, u_ms=wind, v_ms=0, grid_latitude=0, grid_longitude=0)
            for t in TURBINES for h in range(1,horizon+1)))


def test_independent_curves_96_rows_and_endpoint_extension():
    model = PowerCurvePredictor.fit(snapshot())
    batch = model.predict(snapshot(T0+timedelta(hours=1), observations=(), wind=5))
    assert len(batch.rows) == 96
    assert batch.rows[0].prediction_norm == pytest.approx(0.4)
    assert batch.rows[48].prediction_norm == pytest.approx(0.2)
    outside = model.predict(snapshot(T0+timedelta(hours=1), observations=(), wind=40))
    assert all("OUT_OF_DOMAIN" in r.status and 0 <= r.prediction_norm <= 1 for r in outside.rows)
    assert outside.rows[0].prediction_norm == pytest.approx(0.76)  # no invented cut-out


def test_baseline_is_train_mean_and_never_reads_future_actuals():
    train = snapshot()
    model = ConstantBaseline.fit(train)
    later = snapshot(T0+timedelta(days=1), observations=history(T0+timedelta(days=1)))
    changed = later.model_copy(update={"observations": tuple(o.model_copy(update={"power_norm": 1.0}) for o in later.observations)})
    assert model.predict(later) == model.predict(changed)
    assert model.means["turbine_1"] == pytest.approx(0.38)


def test_training_excludes_incomplete_and_resolves_revisions():
    train = snapshot()
    first = train.observations[0]
    invalid = first.model_copy(update={"quality_flag": "incomplete", "power_norm": None,
                                       "n_samples": 5, "coverage": 5/6,
                                       "available_at": first.available_at+timedelta(minutes=1), "revision": "v2"})
    changed = train.model_copy(update={"observations": train.observations+(invalid,)})
    model = ConstantBaseline.fit(changed)
    assert model.counts["turbine_1"] == 239
    assert first not in training_observations(changed)


def test_rejects_future_model_and_tampered_snapshot():
    model = ConstantBaseline.fit(snapshot(), activated_at=T0+timedelta(hours=2))
    with pytest.raises(ValueError, match="FUTURE_MODEL"):
        model.predict(snapshot())
    forged = snapshot().model_copy(update={"observations": (observation(T0),)})
    with pytest.raises(ValueError, match="FUTURE_OBSERVATION"):
        ConstantBaseline.fit(forged)


def test_features_have_weather_and_run_leads_no_actual_lags():
    rows = features_from_snapshot(snapshot())
    assert rows[0].values[4:7] == (1.0, 7.0, 6.0)
    a = snapshot().model_copy(update={"observations": ()})
    assert features_from_snapshot(a) == rows
    with pytest.raises(ValueError, match="WEATHER_COVERAGE"):
        features_from_snapshot(a.model_copy(update={"weather_values": a.weather_values[:-1]}))


def test_persistence_requires_fresh_actual():
    model = PersistenceBaseline.fit(snapshot())
    fresh = snapshot(T0+timedelta(hours=1), observations=(observation(T0, 0.7),
        observation(T0, 0.2, turbine="turbine_2")))
    assert model.predict(fresh).rows[0].prediction_norm == pytest.approx(0.7)
    with pytest.raises(ValueError, match="STALE_PERSISTENCE"):
        model.predict(snapshot(T0+timedelta(days=30), observations=history()))


def test_training_labels_after_cutoff_are_excluded():
    snap = snapshot(observations=())
    cutoff = T0+timedelta(hours=25)
    labels = tuple(observation(T0+timedelta(hours=h), 0.1*h/48)
                   for h in range(1,49))
    examples = supervised_examples([snap], labels, training_cutoff=cutoff, allow_synthetic=True)
    assert len(examples) == 24  # target starting at +25 has not ended yet
    poisoned = labels[:24]+tuple(o.model_copy(update={"power_norm":1.0}) for o in labels[24:])
    assert examples == supervised_examples([snap], poisoned, training_cutoff=cutoff, allow_synthetic=True)
    with pytest.raises(ValueError, match="ML_REQUIRES_OPERATIONAL"):
        supervised_examples([snap], labels, training_cutoff=cutoff)


def test_registry_roundtrip_checksum_and_immutable_version(tmp_path):
    model = PowerCurvePredictor.fit(snapshot())
    path = save_predictor(model, tmp_path/"curve.json")
    assert save_predictor(model, path) == path
    assert load_predictor(path).predict(snapshot()) == model.predict(snapshot())
    with pytest.raises(ValueError, match="IMMUTABLE"):
        save_predictor(ConstantBaseline.fit(snapshot()), path)
    body = json.loads(path.read_text())
    body["payload"]["curves"]["turbine_1"]["power"][0] = 0.9
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="CHECKSUM"):
        load_predictor(path)


def test_service_accepts_real_predictor_interface(setup):
    model = PowerCurvePredictor.fit(snapshot(T0-timedelta(days=1)))
    setup.predictor = model
    result = setup.create_forecast(ForecastRequest(origin_time=T0, turbine_ids=TURBINES, mode="fixture"))
    assert len(result.predictions.rows) == 96
    assert result == setup.create_forecast(ForecastRequest(origin_time=T0, turbine_ids=TURBINES, mode="fixture"))
    with pytest.raises(ValueError, match="SYNTHETIC_MODEL"):
        setup.create_forecast(ForecastRequest(origin_time=T0, turbine_ids=TURBINES, mode="replay"))


def prepared_experiment():
    """Deterministic fixture generator, seed 42, two chronological training folds."""
    from TwinTurbo.ai.evaluate import TemporalFold
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(42)
    observations = []
    total_hours = 40*24
    winds = {}
    for ti, turbine in enumerate(TURBINES):
        for hour in range(total_hours):
            wind = max(0.1, 6 + 3.2*math.sin(hour/17) + 1.1*math.sin(hour/5+ti))
            winds[turbine, hour] = wind
            power = float(np.clip((wind/11)**3*(1-0.2*ti) + rng.normal(0,0.015),0,1))
            observations.append(observation(start+timedelta(hours=hour),power,wind,turbine,delay=15))
    def prepared(day):
        origin = start+timedelta(days=day)
        allowed = tuple(o for o in observations if o.available_at <= origin)
        snap = snapshot(origin, allowed)
        values = tuple(w.model_copy(update={"wind_ms": max(0, winds[w.turbine_id,
            int((w.valid_time-start).total_seconds()/3600)]-0.7), "u_ms": max(0, winds[w.turbine_id,
            int((w.valid_time-start).total_seconds()/3600)]-0.7)}) for w in snap.weather_values)
        return snap.model_copy(update={"weather_values":values})
    folds = (TemporalFold(prepared(20), tuple(prepared(d) for d in range(21,28))),
             TemporalFold(prepared(28), tuple(prepared(d) for d in range(29,36))))
    return folds, tuple(observations), start+timedelta(days=39)


def test_optional_ml_no_random_validation_and_serialization(tmp_path):
    pytest.importorskip("sklearn")
    from TwinTurbo.ai.models.ml import MLPredictor
    folds, observations, _ = prepared_experiment()
    train = folds[1].training
    earlier = folds[0].validation
    model = MLPredictor.fit(train, earlier, allow_synthetic=True, min_samples=50, max_iter=5)
    assert all(not e.early_stopping and e.loss == "absolute_error" for e in model.estimators.values())
    snap = folds[1].validation[0]
    batch = model.predict(snap)
    assert len(batch.rows) == 96
    path = save_predictor(model, tmp_path/"ml.json")
    with pytest.raises(ValueError, match="TRUSTED"):
        load_predictor(path)
    assert load_predictor(path, trusted=True).predict(snap) == batch


def test_activation_is_part_of_immutable_model_identity():
    now = ConstantBaseline.fit(snapshot())
    later = ConstantBaseline.fit(snapshot(),activated_at=T0+timedelta(hours=1))
    assert now.state.model_id != later.state.model_id


def test_ensemble_weights_use_matured_later_validation_only():
    from TwinTurbo.ai.models.ensemble import EnsemblePredictor
    curve = ConstantBaseline.fit(snapshot(T0-timedelta(days=2)))
    ml = ConstantBaseline.fit(snapshot(T0-timedelta(days=2)))
    # Controlled competing model values make the grid optimum independently known.
    ml.means = {t:0.8 for t in TURBINES}
    origins = [snapshot(T0-timedelta(days=1),observations=())]
    target = origins[0].target_intervals[0].target_start
    labels = [observation(target,0.8,turbine=t) for t in TURBINES]
    model = EnsemblePredictor.fit(curve,ml,origins,labels,as_of=T0)
    assert model.weights['1-6'] == 1
    assert model.weights['25-48'] == 0  # no mature validation for this group
    with pytest.raises(ValueError,match='FUTURE_MODEL'):
        model.predict(snapshot(T0-timedelta(hours=1)))
    assert len(model.predict(snapshot(T0+timedelta(hours=1))).rows) == 96
    later = labels + [observation(T0+timedelta(hours=1),0)]
    assert EnsemblePredictor.fit(curve,ml,origins,later,as_of=T0).state == model.state
