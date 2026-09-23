from copy import deepcopy
from datetime import timedelta

import pytest

from windoracle.models.bias import apply_bias, update_bias
from windoracle.models.intervals import (
    apply_intervals,
    interval_coverage,
    mean_interval_width,
)
from windoracle.schemas import PredictionBatch, PredictionRow
from .test_bias import T0, residual


def point_batch(power=0.5, lead=1, *, origin=T0, turbine="turbine_1"):
    return PredictionBatch(
        rows=(
            PredictionRow(
                turbine_id=turbine,
                target_start=origin + timedelta(hours=lead),
                target_end=origin + timedelta(hours=lead + 1),
                prediction_norm=power,
            ),
        )
    )


def test_quantiles_use_issued_error_and_do_not_add_bias_twice():
    rows = [
        residual(index, actual=0.7, base=0.3, issued=0.6)
        for index in range(30)
    ]
    state = update_bias(rows, model_id="curve-v1", as_of=T0)
    corrected = apply_bias(
        point_batch(0.3), state, model_id="curve-v1", origin_time=T0
    )
    result = apply_intervals(corrected, state, origin_time=T0).rows[0]
    assert result.prediction_norm == pytest.approx(0.7)
    assert (result.q10, result.q50, result.q90) == pytest.approx((0.8, 0.8, 0.8))


def test_sparse_group_produces_nulls_instead_of_fake_point_copies():
    state = update_bias(
        [residual(index) for index in range(29)],
        model_id="curve-v1",
        as_of=T0,
    )
    row = apply_intervals(point_batch(), state, origin_time=T0).rows[0]
    assert (row.q10, row.q50, row.q90) == (None, None, None)
    assert "interval:insufficient_history" in row.status


def test_same_turbine_fallback_is_explicit_and_does_not_cross_turbines():
    rows = [residual(index, lead=8) for index in range(30)]
    strict = update_bias(rows, model_id="curve-v1", as_of=T0)
    fallback = update_bias(
        rows,
        model_id="curve-v1",
        as_of=T0,
        interval_fallback=True,
    )
    assert apply_intervals(point_batch(), strict, origin_time=T0).rows[0].q10 is None
    row = apply_intervals(point_batch(), fallback, origin_time=T0).rows[0]
    assert row.q10 is not None
    assert "interval:turbine_pool" in row.status

    other = apply_intervals(
        point_batch(turbine="turbine_2"), fallback, origin_time=T0
    ).rows[0]
    assert (other.q10, other.q50, other.q90) == (None, None, None)


def test_quantile_order_and_bounds_survive_clipping():
    rows = [
        residual(index, actual=index / 39, base=0.5, issued=0.5)
        for index in range(40)
    ]
    state = update_bias(rows, model_id="curve-v1", as_of=T0)
    for power in (0.0, 0.5, 1.0):
        row = apply_intervals(point_batch(power), state, origin_time=T0).rows[0]
        assert 0 <= row.q10 <= row.q50 <= row.q90 <= 1


def test_coverage_and_width_helpers_have_known_values_and_inclusive_edges():
    assert interval_coverage(
        [0.0, 0.5, 1.0], [0.0, 0.4, 0.9], [0.1, 0.6, 1.0]
    ) == 1.0
    assert mean_interval_width([0.0, 0.4, 0.9], [0.1, 0.6, 1.0]) == pytest.approx(
        0.4 / 3
    )
    assert interval_coverage([], [], []) is None
    assert mean_interval_width([], []) is None
    with pytest.raises(ValueError, match="INVALID_INTERVAL_BOUNDS"):
        interval_coverage([0.5], [0.7], [0.6])


def test_interval_calibration_expires_without_minting_a_new_bias_state():
    rows = [residual(index) for index in range(30)]
    state = update_bias(rows, model_id="curve-v1", as_of=T0)
    origin = T0 + timedelta(days=22)
    row = apply_intervals(
        point_batch(origin=origin), state, origin_time=origin
    ).rows[0]
    assert (row.q10, row.q50, row.q90) == (None, None, None)
    assert "interval:expired_history" in row.status


def test_corrupt_calibration_and_double_application_are_rejected():
    rows = [residual(index) for index in range(30)]
    state = update_bias(rows, model_id="curve-v1", as_of=T0)
    parameters = deepcopy(state.parameters)
    parameters["intervals"]["turbines"]["turbine_1"]["1-6"]["offsets"] = [
        0.2,
        0.1,
        0.3,
    ]
    corrupt = state.model_copy(update={"parameters": parameters})
    with pytest.raises(ValueError, match="INVALID_INTERVAL_CALIBRATION"):
        apply_intervals(point_batch(), corrupt, origin_time=T0)

    result = apply_intervals(point_batch(), state, origin_time=T0)
    with pytest.raises(ValueError, match="INTERVALS_ALREADY_APPLIED"):
        apply_intervals(result, state, origin_time=T0)


def test_future_bias_cannot_supply_intervals():
    state = update_bias(
        [residual(index) for index in range(30)],
        model_id="curve-v1",
        as_of=T0,
    )
    with pytest.raises(ValueError, match="FUTURE_BIAS"):
        apply_intervals(
            point_batch(origin=T0 - timedelta(hours=1)),
            state,
            origin_time=T0 - timedelta(hours=1),
        )
