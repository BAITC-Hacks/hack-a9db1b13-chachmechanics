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
