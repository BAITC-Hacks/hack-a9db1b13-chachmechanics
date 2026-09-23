from datetime import datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo
from .clock import VirtualClock
from .schemas import ForecastRequest
from .agents.orchestrator import Orchestrator


def scheduled_origins(start, end, site_timezone, issue_local_time):
    day, last = start, end
    if day > last:
        raise ValueError("Start must not be after end")
    zone = ZoneInfo(site_timezone)
    hour = time.fromisoformat(issue_local_time)
    while day <= last:
        local = datetime.combine(day, hour, tzinfo=zone)
        if local.replace(fold=0).utcoffset() != local.replace(fold=1).utcoffset():
            raise ValueError("Ambiguous/nonexistent local issue time")
        yield local.astimezone(timezone.utc)
        day += timedelta(days=1)


def replay(service, origins, *, mode="replay", include_updates=False, continue_on_error=False):
    origins = sorted(set(origins))
    if not origins:
        return {"forecast_ids": [], "failures": []}
    clock = VirtualClock(origins[0])
    orchestrator = Orchestrator(service, clock)
    events = [(o, 1, "scheduled", None) for o in origins]
    if include_updates:
        for bundle in service.weather.cache.bundles():
            available = bundle.metadata.available_at
            if origins[0] <= available <= origins[-1]:
                events.append((available, 0, bundle.metadata.run_id, bundle))
    results, failures = [], []
    previous = None
    for at, _, name, bundle in sorted(events, key=lambda e: (e[0], e[1], e[2])):
        clock.advance(at)
        try:
            if bundle is None:
                request = ForecastRequest(origin_time=at, turbine_ids=tuple(t.id for t in service.config.site.turbines),
                    horizon_hours=service.config.forecast.horizon_hours, mode=mode)
                previous = orchestrator.run(request)
            elif previous:
                previous = orchestrator.new_run(bundle, previous)
            else:
                continue
            if previous.forecast_id not in results:
                results.append(previous.forecast_id)
        except (ValueError, RuntimeError) as exc:
            failures.append({"origin_time": at.isoformat(), "event": name, "error": str(exc)})
            if not continue_on_error:
                raise
    return {"forecast_ids": results, "failures": failures}
