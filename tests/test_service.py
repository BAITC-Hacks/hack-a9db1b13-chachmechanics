import pytest
from windoracle.schemas import ForecastRequest, PredictionBatch
from .conftest import ORIGIN


def test_96_rows_idempotency_and_persistence(setup):
    req = ForecastRequest(origin_time=ORIGIN, turbine_ids=("turbine_1", "turbine_2"), mode="fixture")
    result = setup.create_forecast(req)
    assert len(result.predictions.rows) == 96
    assert setup.create_forecast(req) == result
    assert setup.get_forecast(result.forecast_id) == result
    assert len(setup.list_forecasts()) == 1
    assert setup.predictor.calls == 1
    assert all(r.q10 is None and r.q50 is None and r.q90 is None for r in result.predictions.rows)


def test_model_cannot_return_incomplete_batch(setup):
    setup.predictor.predict = lambda snapshot, bias: PredictionBatch(rows=())
    req = ForecastRequest(origin_time=ORIGIN, turbine_ids=("turbine_1", "turbine_2"), mode="fixture")
    with pytest.raises(ValueError, match="MODEL_OUTPUT_COVERAGE"):
        setup.create_forecast(req)
    assert setup.list_forecasts() == []


def test_ui_can_read_without_model(setup):
    from windoracle.service import ForecastService
    req = ForecastRequest(origin_time=ORIGIN, turbine_ids=("turbine_1", "turbine_2"), mode="fixture")
    result = setup.create_forecast(req)
    reader = ForecastService(setup.config, setup.store, setup.weather)
    assert reader.get_forecast(result.forecast_id) == result
    assert reader.list_forecasts() == [result]
    assert len(reader.data_summary()["turbines"]) == 2
    with pytest.raises(ValueError, match="MODEL_REQUIRED"):
        reader.create_forecast(req)


def test_model_versions_cannot_be_overwritten(setup):
    from datetime import timedelta
    state = setup.predictor.state
    setup.store.save_model(state)
    setup.store.save_model(state)
    assert setup.store.models_as_of(ORIGIN) == (state,)
    assert setup.store.models_as_of(ORIGIN - timedelta(days=3)) == ()
    with pytest.raises(ValueError, match="immutable"):
        setup.store.save_model(state.model_copy(update={"artifact_ref": "changed"}))


def test_bias_versions_are_available_only_after_creation(setup):
    from datetime import timedelta
    from windoracle.schemas import BiasState
    setup.store.save_model(setup.predictor.state)
    bias = BiasState(bias_id="b1", model_id=setup.predictor.state.model_id,
        created_as_of=ORIGIN, last_actual_available_at=ORIGIN - timedelta(hours=1), parameters={"offset": 0.1})
    setup.store.save_bias(bias)
    setup.store.save_bias(bias)
    assert setup.store.bias_as_of(bias.model_id, ORIGIN - timedelta(seconds=1)) is None
    assert setup.store.bias_as_of(bias.model_id, ORIGIN) == bias
    with pytest.raises(ValueError, match="immutable"):
        setup.store.save_bias(bias.model_copy(update={"parameters": {"offset": .2}}))
