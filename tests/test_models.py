from datetime import datetime, timedelta, timezone

import pytest

from windoracle.agents.twin_builder import TwinBuilder
from windoracle.features import (
    FEATURE_NAMES,
    LEAD_GROUPS,
    TrainingExample,
    build_features,
    feature_vector,
)
from windoracle.models.baseline import fit_turbine_means
from windoracle.models.ensemble import (
    EnsembleExample,
    build_ensemble_predictor,
    select_lead_weights,
)
from windoracle.models.ml import fit_ridge_predictor
from windoracle.models.power_curve import fit_power_curve, fit_power_curves
from windoracle.models.registry import load_predictor, save_predictor
from windoracle.schemas import (
    AsOfSnapshot,
    BiasState,
    ForecastRequest,
    Observation,
    TargetInterval,
    WeatherRunMetadata,
    WeatherValue,
)


ORIGIN = datetime(2025, 6, 1, 18, tzinfo=timezone.utc)


def observation(turbine, start_hour, available_hour, wind, power, revision):
    start = ORIGIN.replace(hour=start_hour)
    return Observation(
        turbine_id=turbine,
        event_start=start,
        event_end=start + timedelta(hours=1),
        available_at=ORIGIN.replace(hour=available_hour, minute=15),
        power_norm=power,
        wind_ms=wind,
        temperature_c=10,
        n_samples=6,
        coverage=1,
        quality_flag="complete",
        revision=revision,
    )


def snapshot():
    observations = (
        observation("turbine_1", 10, 11, 4, .2, "a"),
        observation("turbine_1", 11, 12, 8, .6, "b"),
        observation("turbine_2", 10, 11, 4, .1, "a"),
        observation("turbine_2", 11, 12, 8, .3, "b"),
        # Available at origin, but deliberately after the training cutoff used below.
        observation("turbine_1", 16, 17, 12, 1, "late"),
        observation("turbine_2", 16, 17, 12, .9, "late"),
    )
    intervals = tuple(
        TargetInterval(
            target_start=ORIGIN + timedelta(hours=lead),
            target_end=ORIGIN + timedelta(hours=lead + 1),
        )
        for lead in (1, 2)
    )
    weather = tuple(
        WeatherValue(
            turbine_id=turbine,
            valid_time=interval.target_start,
            wind_ms=100,
            temperature_c=12,
            u_ms=5,
            v_ms=0,
            grid_latitude=0,
            grid_longitude=0,
        )
        for turbine in ("turbine_1", "turbine_2")
        for interval in intervals
    )
    metadata = WeatherRunMetadata(
        run_id="run",
        provider="test",
        model="test",
        run_init_time=ORIGIN - timedelta(hours=6),
        available_at=ORIGIN,
        availability_basis="synthetic",
        provenance="synthetic",
        retrieved_at=ORIGIN,
        sha256="0" * 64,
    )
    return AsOfSnapshot(
        origin_time=ORIGIN,
        observations=observations,
        weather_values=weather,
        weather_run_metadata=metadata,
        target_intervals=intervals,
    )


def test_features_are_pure_sorted_and_require_exact_weather_grid():
    source = snapshot()
    rows = build_features(source)
    assert [(row.turbine_id, row.target_start) for row in rows] == sorted(
        (value.turbine_id, value.valid_time) for value in source.weather_values
    )
    assert len(feature_vector(rows[0])) == len(FEATURE_NAMES)

    incomplete = source.model_copy(update={"weather_values": source.weather_values[:-1]})
    with pytest.raises(ValueError, match="WEATHER_FEATURE_COVERAGE"):
        build_features(incomplete)
    duplicated = source.model_copy(update={"weather_values": source.weather_values + source.weather_values[:1]})
    with pytest.raises(ValueError, match="duplicate"):
        build_features(duplicated)


def test_power_curves_are_per_turbine_binned_medians_and_clamp_domain():
    source = snapshot()
    curves = fit_power_curves(source.observations, turbine_ids=("turbine_1", "turbine_2"), bin_width=5)
    assert {curve.turbine_id for curve in curves} == {"turbine_1", "turbine_2"}
    first = next(curve for curve in curves if curve.turbine_id == "turbine_1")
    second = next(curve for curve in curves if curve.turbine_id == "turbine_2")
    assert first.predict(0) == first.median_power[0]
    assert first.predict(1_000) == first.median_power[-1]
    assert first.predict(1_000) == pytest.approx(1.0)
    assert second.predict(1_000) == pytest.approx(.9)
    assert all(0 <= first.predict(wind) <= 1 for wind in (0, 4, 6, 12, 1_000))

    only_first = fit_power_curve(source.observations, turbine_id="turbine_1", bin_width=20)
    assert only_first.median_power == (pytest.approx(.6),)


def test_constant_baseline_is_independent_train_mean_per_turbine():
    source = snapshot()
    cutoff = ORIGIN - timedelta(hours=3)
    means = fit_turbine_means(
        source.observations,
        turbine_ids=("turbine_1", "turbine_2"),
        cutoff=cutoff,
    )
    values = {item.turbine_id: item.prediction_norm for item in means}
    assert values == {"turbine_1": pytest.approx(.4), "turbine_2": pytest.approx(.2)}


def test_twin_builder_respects_availability_cutoff_and_is_deterministic():
    source = snapshot()
    cutoff = ORIGIN - timedelta(hours=3)
    builder = TwinBuilder(bin_width=2, provenance="synthetic")
    first = builder.build(source, cutoff=cutoff)
    second = builder.build(source, cutoff=cutoff)
    assert first.state == second.state
    assert first.state.max_label_available_at <= first.state.training_cutoff <= first.state.activated_at
    assert first.state.training_cutoff == cutoff
    assert first.state.artifact_ref.startswith("inline-sha256:")

    batch = first.predict(source)
    assert len(batch.rows) == len(source.weather_values)
    assert {(row.turbine_id, row.target_start) for row in batch.rows} == {
        (value.turbine_id, value.valid_time) for value in source.weather_values
    }
    # The later power=1/.9 labels are unavailable at cutoff.  High-wind
    # predictions therefore clamp to the earlier endpoint (.6/.3), not to them.
    by_turbine = {}
    for row in batch.rows:
        by_turbine.setdefault(row.turbine_id, row.prediction_norm)
    assert by_turbine == {"turbine_1": pytest.approx(.6), "turbine_2": pytest.approx(.3)}
    assert all("out_of_domain" in row.status for row in batch.rows)

    groups = {
        name: {"bias": .1, "count": 1, "status": "calibrated"}
        for name in LEAD_GROUPS
    }
    bias = BiasState(
        bias_id="test-bias",
        model_id=first.state.model_id,
        created_as_of=ORIGIN,
        last_actual_available_at=ORIGIN - timedelta(hours=1),
        parameters={
            "schema": "rolling-bias-v1",
            "window_days": 21,
            "turbines": {
                "turbine_1": {"groups": groups},
                "turbine_2": {"groups": groups},
            },
            "intervals": {},
        },
    )
    corrected = first.predict(source, bias)
    corrected_values = {
        row.turbine_id: row.prediction_norm for row in corrected.rows
    }
    assert corrected_values == {
        "turbine_1": pytest.approx(.7),
        "turbine_2": pytest.approx(.4),
    }
    assert all("bias:calibrated" in row.status for row in corrected.rows)

    baseline = builder.build(source, kind="baseline", cutoff=cutoff)
    baseline_rows = baseline.predict(source).rows
    baseline_values = {row.turbine_id: row.prediction_norm for row in baseline_rows}
    assert baseline_values == {"turbine_1": pytest.approx(.4), "turbine_2": pytest.approx(.2)}


def test_twin_builder_rejects_cutoff_after_snapshot_origin():
    with pytest.raises(ValueError, match="cutoff"):
        TwinBuilder().build(snapshot(), cutoff=ORIGIN + timedelta(seconds=1))


def training_examples(count=16):
    rows = []
    for turbine, factor in (("turbine_1", .08), ("turbine_2", .04)):
        for index in range(count):
            origin = ORIGIN - timedelta(days=count - index + 2)
            wind = 2.0 + index * .5
            target = origin + timedelta(hours=(index % 4) + 1)
            rows.append(TrainingExample(
                turbine_id=turbine,
                origin_time=origin,
                target_start=target,
                weather_run_init_time=origin - timedelta(hours=6),
                wind_ms=wind,
                temperature_c=5 + index,
                u_ms=wind,
                v_ms=0,
                actual_norm=min(1, factor * wind),
                actual_available_at=target + timedelta(hours=1, minutes=15),
            ))
    return tuple(rows)


def test_ridge_ml_is_per_turbine_deterministic_and_leakage_safe():
    examples = training_examples() + (
        TrainingExample(
            turbine_id="turbine_1",
            origin_time=ORIGIN - timedelta(hours=3),
            target_start=ORIGIN - timedelta(hours=1),
            weather_run_init_time=ORIGIN - timedelta(hours=9),
            wind_ms=20,
            temperature_c=0,
            u_ms=20,
            v_ms=0,
            actual_norm=1,
            actual_available_at=ORIGIN + timedelta(minutes=15),
        ),
    )
    first = fit_ridge_predictor(
        examples,
        training_cutoff=ORIGIN,
        activated_at=ORIGIN,
        artifact_ref="inline:test-ridge",
        provenance="synthetic",
    )
    second = fit_ridge_predictor(
        examples,
        training_cutoff=ORIGIN,
        activated_at=ORIGIN,
        artifact_ref="inline:test-ridge",
        provenance="synthetic",
    )
    assert first == second
    assert first.state.max_label_available_at < ORIGIN
    source = snapshot()
    source = source.model_copy(update={
        "weather_values": tuple(
            value.model_copy(update={"wind_ms": 6, "u_ms": 6})
            for value in source.weather_values
        )
    })
    batch = first.predict(source)
    assert len(batch.rows) == 4
    assert all(0 <= row.prediction_norm <= 1 for row in batch.rows)
    by_turbine = {row.turbine_id: row.prediction_norm for row in batch.rows}
    assert by_turbine["turbine_1"] > by_turbine["turbine_2"]


def test_ensemble_weights_use_only_available_validation_labels():
    validation = tuple(
        EnsembleExample(
            turbine_id="turbine_1",
            origin_time=ORIGIN - timedelta(days=3),
            target_start=ORIGIN - timedelta(days=3) + timedelta(hours=lead),
            actual_available_at=ORIGIN - timedelta(hours=12),
            actual_norm=.8,
            twin_prediction=.2,
            ml_prediction=.8,
        )
        for lead in (1, 7, 13, 25)
    ) + (
        EnsembleExample(
            turbine_id="turbine_1",
            origin_time=ORIGIN - timedelta(hours=3),
            target_start=ORIGIN - timedelta(hours=2),
            actual_available_at=ORIGIN + timedelta(hours=1),
            actual_norm=0,
            twin_prediction=0,
            ml_prediction=1,
        ),
    )
    selection = select_lead_weights(validation, as_of=ORIGIN)
    assert selection.sample_count == 4
    assert all(weight.ml_weight == 1 for weight in selection.weights)

    twin = TwinBuilder(provenance="synthetic").build(
        snapshot(), cutoff=ORIGIN - timedelta(hours=3))
    ml = fit_ridge_predictor(
        training_examples(),
        training_cutoff=ORIGIN,
        activated_at=ORIGIN,
        artifact_ref="inline:test-ridge",
        provenance="synthetic",
    )
    ensemble = build_ensemble_predictor(
        twin, ml, selection, activated_at=ORIGIN, artifact_ref="inline:test-ensemble")
    ensemble_values = [row.prediction_norm for row in ensemble.predict_base(snapshot()).rows]
    ml_values = [row.prediction_norm for row in ml.predict_base(snapshot()).rows]
    assert ensemble_values == ml_values


def test_json_registry_round_trip_and_zero_arg_factory(tmp_path, monkeypatch):
    predictor = TwinBuilder(provenance="synthetic").build(
        snapshot(), cutoff=ORIGIN - timedelta(hours=3))
    artifact = tmp_path / "curve.json"
    save_predictor(predictor, artifact)
    assert save_predictor(predictor, artifact) == artifact.resolve()
    loaded = load_predictor(artifact)
    assert loaded.state == predictor.state
    assert loaded.predict(snapshot()) == predictor.predict(snapshot())

    monkeypatch.setenv("TWINTURBO_MODEL_ARTIFACT", str(artifact))
    assert load_predictor().state == predictor.state

    artifact.write_text(
        artifact.read_text(encoding="utf-8").replace(
            '"domain_max_wind_ms":8.0', '"domain_max_wind_ms":9.0'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="CHECKSUM"):
        load_predictor(artifact)


def test_power_curve_predictor_integrates_with_service_for_96_rows(setup):
    setup.predictor = TwinBuilder(provenance="synthetic").build(
        snapshot(), cutoff=ORIGIN - timedelta(hours=3))
    result = setup.create_forecast(ForecastRequest(
        origin_time=ORIGIN,
        turbine_ids=("turbine_1", "turbine_2"),
        horizon_hours=48,
        mode="fixture",
    ))
    assert len(result.predictions.rows) == 96
    assert all(0 <= row.prediction_norm <= 1 for row in result.predictions.rows)
    assert all(
        (row.q10, row.q50, row.q90) == (None, None, None)
        for row in result.predictions.rows
    )
