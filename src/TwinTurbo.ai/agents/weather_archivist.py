"""Weather acquisition decisions; the provider owns networking and decoding."""
from ..weather.audit import audit_bundle


class WeatherArchivist:
    def __init__(self, provider, store, max_age_hours=24):
        self.provider, self.store, self.max_age_hours = provider, store, max_age_hours

    def acquire(self, request, *, online=False):
        try:
            if online:
                bundle = self.provider.fetch_latest(request)
                for event in getattr(self.provider, "last_fetch_events", ()):
                    self.store.event(request.origin_time, "fetch_weather", event["reason"], **event["details"])
            else:
                bundle = self.provider.select_run(request)
            audit_bundle(bundle, request, self.max_age_hours)
            self.store.event(request.origin_time, "select_weather", "LATEST_ADMISSIBLE_RUN",
                             run_id=bundle.metadata.run_id, source="online" if online else "cache")
            return bundle
        except (ValueError, RuntimeError) as exc:
            self.store.event(request.origin_time, "reject_weather", "WEATHER_UNAVAILABLE", message=str(exc))
            raise
