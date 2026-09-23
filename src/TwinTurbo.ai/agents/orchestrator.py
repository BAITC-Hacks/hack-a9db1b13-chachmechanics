"""Agent decisions are explicit and recorded; numerical forecasting is injected."""
from ..schemas import ForecastRequest
from ..weather.audit import audit_bundle


class Orchestrator:
    def __init__(self, service, clock):
        self.service, self.clock = service, clock

    def run(self, request, bias=None, parent_forecast_id=None):
        if request.origin_time != self.clock.now:
            raise ValueError("REQUEST_CLOCK_MISMATCH")
        try:
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
        if bundle.metadata.run_id == previous.run_id:
            self.service.store.event(self.clock.now, "skip", "RUN_UNCHANGED", run_id=previous.run_id)
            return previous
        # An update retains the same still-future target window; once that window
        # has started, the replay schedules the next complete future window.
        original = ForecastRequest.model_validate(previous.manifest["request"])
        request = ForecastRequest(origin_time=self.clock.now, turbine_ids=original.turbine_ids,
            horizon_hours=original.horizon_hours, mode=original.mode, release_kind="update",
            target_start=original.target_start if original.target_start and original.target_start > self.clock.now else None)
        audit_bundle(bundle, request, self.service.config.weather.max_run_age_hours)
        selected = self.service.weather.select_run(request)
        if selected.metadata.run_id != bundle.metadata.run_id:
            self.service.store.event(self.clock.now, "skip", "NEWER_RUN_ALREADY_AVAILABLE", run_id=bundle.metadata.run_id)
            return previous
        return self.run(request, bias, previous.forecast_id)
