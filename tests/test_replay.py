from datetime import timedelta
from windoracle.replay import replay
from .conftest import ORIGIN, bundle


def test_new_run_preserves_previous_version(setup):
    setup.weather.cache.save(bundle("run-2", init=ORIGIN, available=ORIGIN + timedelta(hours=6), wind=8))
    result = replay(setup, [ORIGIN, ORIGIN + timedelta(hours=7)], mode="fixture", include_updates=True)
    assert len(result["forecast_ids"]) == 3
    first, update, _ = [setup.get_forecast(i) for i in result["forecast_ids"]]
    assert first.predictions.rows[0].prediction_norm == .25
    assert update.predictions.rows[0].prediction_norm == .4
    assert update.parent_forecast_id == first.forecast_id
    assert setup.get_forecast(first.forecast_id) == first
    again = replay(setup, [ORIGIN, ORIGIN + timedelta(hours=7)], mode="fixture", include_updates=True)
    assert again == result
    assert len(setup.list_forecasts()) == 3


def test_same_cycle_with_different_cached_window_is_not_new_run(setup):
    from windoracle.agents.orchestrator import Orchestrator
    from windoracle.clock import VirtualClock
    from windoracle.schemas import ForecastRequest
    req = ForecastRequest(origin_time=ORIGIN, turbine_ids=("turbine_1", "turbine_2"), mode="fixture")
    first = setup.create_forecast(req)
    copy = bundle("same-cycle-other-window", count=75)
    setup.weather.cache.save(copy)
    orchestrator = Orchestrator(setup, VirtualClock(ORIGIN + timedelta(hours=1)))
    assert orchestrator.new_run(copy, first) == first
    assert len(setup.list_forecasts()) == 1
