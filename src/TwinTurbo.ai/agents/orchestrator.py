"""Agent decisions are explicit and recorded; numerical forecasting is injected."""
from ..schemas import ForecastRequest
from ..weather.audit import audit_bundle
from .weather_archivist import WeatherArchivist


class Orchestrator:
    def __init__(self, service, clock):
        self.service, self.clock = service, clock

    def run(self, request, bias=None, parent_forecast_id=None, *, online=False):
        if request.origin_time != self.clock.now:
            raise ValueError("REQUEST_CLOCK_MISMATCH")
        try:
            WeatherArchivist(self.service.weather, self.service.store,
                self.service.config.weather.max_run_age_hours).acquire(request, online=online)
            result = self.service.create_forecast(request, bias, parent_forecast_id)
            self.service.store.event(self.clock.now, "calculate", "VALID_INPUTS",
                                     forecast_id=result.forecast_id, status="ok")
            return result
        except (ValueError, RuntimeError) as exc:
            self.service.store.event(self.clock.now, "reject", type(exc).__name__, message=str(exc), status="error")
            raise

    def new_run(self, bundle, previous, bias=None):
        if bundle.metadata.available_at > self.clock.now:
            self.service.store.event(self.clock.now, "reject_run", "FUTURE_WEATHER", run_id=bundle.metadata.run_id)
            raise ValueError("FUTURE_WEATHER")
        previous_weather = previous.manifest["weather"]
        from datetime import datetime
        previous_init = datetime.fromisoformat(previous_weather["run_init_time"].replace("Z", "+00:00"))
        if (bundle.metadata.run_id == previous.run_id or
            (bundle.metadata.model == previous_weather["model"] and bundle.metadata.run_init_time <= previous_init)):
            self.service.store.event(self.clock.now, "skip", "RUN_UNCHANGED", run_id=previous.run_id)
            return previous
        # Each update forecasts the next complete future window. Comparisons use
        # intersecting target hours, not matching row positions.
        original = ForecastRequest.model_validate(previous.manifest["request"])
        request = ForecastRequest(origin_time=self.clock.now, turbine_ids=original.turbine_ids,
            horizon_hours=original.horizon_hours, mode=original.mode, release_kind="update",
            target_start=None)
        audit_bundle(bundle, request, self.service.config.weather.max_run_age_hours)
        selected = self.service.weather.select_run(request)
        if selected.metadata.run_id != bundle.metadata.run_id:
            self.service.store.event(self.clock.now, "skip", "NEWER_RUN_ALREADY_AVAILABLE", run_id=bundle.metadata.run_id)
            return previous
        return self.run(request, bias, previous.forecast_id)
