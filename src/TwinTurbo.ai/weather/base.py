from typing import Protocol
from datetime import datetime
from ..schemas import ForecastRequest, WeatherBundle


class WeatherUnavailable(RuntimeError):
    pass


class WeatherProvider(Protocol):
    def select_run(self, request: ForecastRequest) -> WeatherBundle: ...
    def fetch_run(self, initialized_at: datetime, request: ForecastRequest) -> WeatherBundle: ...
