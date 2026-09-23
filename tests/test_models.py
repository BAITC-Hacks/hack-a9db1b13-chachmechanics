from datetime import datetime, timedelta, timezone

import pytest

from windoracle.agents.twin_builder import TwinBuilder
from windoracle.features import FEATURE_NAMES, LEAD_GROUPS, build_features, feature_vector
from windoracle.models.baseline import fit_turbine_means
from windoracle.models.power_curve import fit_power_curve, fit_power_curves
from windoracle.schemas import (
    AsOfSnapshot,
    BiasState,
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
