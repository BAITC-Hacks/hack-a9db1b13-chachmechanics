from datetime import timedelta
import pytest
from TwinTurbo.ai.models.bias import apply_bias, update_bias
from TwinTurbo.ai.models.intervals import apply_intervals
from TwinTurbo.ai.schemas import PredictionBatch, PredictionRow
from .test_models import T0
from .test_bias import residual


def batch(power=0.5, lead=1):
    return PredictionBatch(rows=(PredictionRow(turbine_id="turbine_1",target_start=T0+timedelta(hours=lead),
        target_end=T0+timedelta(hours=lead+1),prediction_norm=power),))


def test_quantiles_from_issued_errors_bias_not_added_twice():
    rows = [residual(i,actual=0.7,base=0.3,issued=0.6) for i in range(30)]
    state = update_bias(rows,model_id="curve",as_of=T0)
    corrected = apply_bias(batch(0.3),state,model_id="curve",origin_time=T0)
    result = apply_intervals(corrected,state,origin_time=T0).rows[0]
    assert result.prediction_norm == pytest.approx(0.7)
    assert (result.q10,result.q50,result.q90) == pytest.approx((0.8,0.8,0.8))


def test_sparse_group_has_null_quantiles_not_three_point_copies():
    state = update_bias([residual(i) for i in range(29)],model_id="curve",as_of=T0)
    row = apply_intervals(batch(),state,origin_time=T0).rows[0]
    assert (row.q10,row.q50,row.q90) == (None,None,None)


def test_pool_fallback_requires_explicit_validation_choice():
    rows = [residual(i,lead=8) for i in range(30)]
    a = update_bias(rows,model_id="curve",as_of=T0)
    b = update_bias(rows,model_id="curve",as_of=T0,interval_fallback=True)
    assert apply_intervals(batch(),a,origin_time=T0).rows[0].q10 is None
    row = apply_intervals(batch(),b,origin_time=T0).rows[0]
    assert row.q10 is not None and "turbine_pool" in row.status


def test_order_and_bounds_near_zero_and_one():
    state = update_bias([residual(i,actual=i/39,base=0.5,issued=0.5) for i in range(40)],model_id="curve",as_of=T0)
    for power in (0,0.5,1):
        row = apply_intervals(batch(power),state,origin_time=T0).rows[0]
        assert 0 <= row.q10 <= row.q50 <= row.q90 <= 1
