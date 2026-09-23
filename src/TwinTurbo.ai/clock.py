from datetime import datetime, timedelta
from .schemas import ForecastRequest, TargetInterval, utc


class VirtualClock:
    def __init__(self, now: datetime):
        self._now = utc(now)

    @property
    def now(self):
        return self._now

    def advance(self, value: datetime):
        value = utc(value)
        if value < self._now:
            raise ValueError("Virtual clock cannot move backwards")
        self._now = value


def targets(request: ForecastRequest) -> tuple[TargetInterval, ...]:
    start = request.target_start or request.origin_time + timedelta(hours=1)
    return tuple(TargetInterval(target_start=start + timedelta(hours=i),
                                target_end=start + timedelta(hours=i + 1))
                 for i in range(request.horizon_hours))
