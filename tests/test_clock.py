from datetime import timedelta, date
import pytest
from TwinTurbo.ai.clock import VirtualClock, targets
from TwinTurbo.ai.schemas import ForecastRequest
from TwinTurbo.ai.replay import scheduled_origins
from .conftest import ORIGIN


def test_clock_and_48_intervals():
    clock = VirtualClock(ORIGIN)
    clock.advance(ORIGIN + timedelta(hours=1))
    with pytest.raises(ValueError):
        clock.advance(ORIGIN)
    req = ForecastRequest(origin_time=ORIGIN, turbine_ids=("turbine_1",))
    hours = targets(req)
    assert len(hours) == 48
    assert hours[0].target_start == ORIGIN + timedelta(hours=1)
    assert hours[-1].target_end == ORIGIN + timedelta(hours=49)


def test_naive_time_rejected():
    with pytest.raises(ValueError, match="explicit UTC offset"):
        ForecastRequest(origin_time="2026-01-01T00:00:00", turbine_ids=("turbine_1",))


def test_schedule_is_configurable():
    values = list(scheduled_origins(date(2025, 6, 1), date(2025, 6, 2), "Asia/Almaty", "23:00"))
    assert values[0] == ORIGIN
    assert len(values) == 2
