from datetime import timedelta
import pytest
from windoracle.schemas import ForecastRequest, Observation, BiasState
from .conftest import ORIGIN, bundle


def test_mutating_future_weather_and_actuals_cannot_change_result(setup):
    req = ForecastRequest(origin_time=ORIGIN, turbine_ids=("turbine_1", "turbine_2"), mode="fixture")
    first = setup.create_forecast(req)
    setup.weather.cache.save(bundle("future", init=ORIGIN, available=ORIGIN + timedelta(hours=6), wind=18))
    obs = Observation(turbine_id="turbine_1", event_start=ORIGIN, event_end=ORIGIN + timedelta(hours=1),
        available_at=ORIGIN + timedelta(hours=2), power_norm=1, wind_ms=30, temperature_c=20,
        n_samples=6, coverage=1, quality_flag="complete", revision="new")
    setup.store.ingest([obs], {"test": "future"})
    second = setup.create_forecast(req)
    assert first == second
    assert setup.predictor.calls == 1


def test_future_model_and_bias_are_rejected(setup):
    req = ForecastRequest(origin_time=ORIGIN, turbine_ids=("turbine_1", "turbine_2"), mode="fixture")
    setup.predictor.state = setup.predictor.state.model_copy(update={"activated_at": ORIGIN + timedelta(seconds=1)})
    with pytest.raises(ValueError, match="FUTURE_MODEL"):
        setup.create_forecast(req)
    with pytest.raises(ValueError, match="future actual"):
        BiasState(bias_id="b", model_id="m", created_as_of=ORIGIN,
                  last_actual_available_at=ORIGIN + timedelta(hours=1))
