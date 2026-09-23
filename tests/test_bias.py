from datetime import timedelta
import pytest
from TwinTurbo.ai.models.bias import Residual, apply_bias, select_residuals, update_bias
from TwinTurbo.ai.schemas import PredictionBatch, PredictionRow
from .test_models import T0


def residual(i=0, *, model="curve", turbine="turbine_1", lead=1, actual=0.7, base=0.4, issued=0.6):
    target = T0-timedelta(hours=i+2)
    return Residual(forecast_id=f"f-{i}-{lead}", model_id=model, turbine_id=turbine,
        origin_time=target-timedelta(hours=lead), target_start=target, target_end=target+timedelta(hours=1),
        training_cutoff=T0-timedelta(days=50), actual_available_at=target+timedelta(hours=1),
        actual_revision="v1", actual=actual, p_base=base, p_issued=issued)


def test_shrinkage_uses_base_not_corrected_error():
    records = [residual(0,lead=1,actual=0.8), residual(1,lead=8,actual=0.6)]
    state = update_bias(records, model_id="curve", as_of=T0)
    group = state.parameters["turbines"]["turbine_1"]["groups"]["1-6"]
    assert group["bias"] == pytest.approx((0.4+48*0.3)/49)
    assert group["count"] == 1


def test_bias_idempotent_duplicates_and_order_independent():
    rows = [residual(i) for i in range(10)]
    state = update_bias(rows, model_id="curve", as_of=T0)
    assert state == update_bias(list(reversed(rows))*3, model_id="curve", as_of=T0)
    assert state == update_bias(rows, model_id="curve", as_of=T0+timedelta(hours=1), previous=state)


def test_future_errors_and_other_models_never_change_bias():
    current = residual()
    late = residual(1).model_copy(update={"actual_available_at":T0+timedelta(hours=1)})
    a = update_bias([current], model_id="curve", as_of=T0)
    b = update_bias([current,late,residual(2,model="other")], model_id="curve", as_of=T0)
    assert a == b
    assert update_bias([], model_id="new-model", as_of=T0, previous=a) is None


def test_revision_supersedes_old_fact_and_updates_never_reweight():
    first = residual()
    revised = first.model_copy(update={"actual_available_at":T0,"actual_revision":"v2","actual":0.9})
    update = residual(1).model_copy(update={"release_kind":"update","actual":0.1})
    selected = select_residuals([first,revised,update],model_id="curve",as_of=T0)
    assert selected == (revised,)
    earlier = select_residuals([first,revised],model_id="curve",as_of=T0-timedelta(minutes=1))
    assert earlier == (first,)


def test_latest_designated_scheduled_origin_per_target_group():
    first = residual(lead=3)
    second = first.model_copy(update={"forecast_id":"later","origin_time":first.origin_time+timedelta(hours=1)})
    assert select_residuals([first,second],model_id="curve",as_of=T0) == (second,)


def test_expired_window_resets_without_new_learning():
    state = update_bias([residual()],model_id="curve",as_of=T0)
    expired = update_bias([residual()],model_id="curve",as_of=T0+timedelta(days=22),previous=state)
    assert expired.parameters["status"] == "expired_history"
    assert expired.parameters["turbines"] == {}
    assert expired.last_actual_available_at == state.last_actual_available_at


def test_clipping_and_model_compatibility():
    state = update_bias([residual(actual=1,base=0)],model_id="curve",as_of=T0)
    batch = PredictionBatch(rows=(PredictionRow(turbine_id="turbine_1", target_start=T0+timedelta(hours=1),
        target_end=T0+timedelta(hours=2), prediction_norm=0.8),))
    assert apply_bias(batch,state,model_id="curve",origin_time=T0).rows[0].prediction_norm == 1
    with pytest.raises(ValueError,match="INADMISSIBLE"):
        apply_bias(batch,state,model_id="another",origin_time=T0)
    with pytest.raises(ValueError,match="INADMISSIBLE"):
        apply_bias(batch,state,model_id="curve",origin_time=T0-timedelta(hours=1))


def test_no_fact_no_state():
    assert update_bias([],model_id="curve",as_of=T0) is None
