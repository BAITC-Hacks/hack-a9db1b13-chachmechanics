from datetime import timedelta
from ..clock import targets
from ..schemas import ForecastRequest, WeatherBundle


def audit_bundle(bundle: WeatherBundle, request: ForecastRequest, max_age_hours=24):
    m = bundle.metadata
    if m.available_at > request.origin_time:
        raise ValueError("FUTURE_WEATHER")
    if request.origin_time - m.run_init_time > timedelta(hours=max_age_hours):
        raise ValueError("STALE_WEATHER")
    if request.mode != "fixture" and m.provenance != "operational_archive":
        raise ValueError("NON_OPERATIONAL_WEATHER")
    required = {(t, h.target_start) for t in request.turbine_ids for h in targets(request)}
    actual = [(v.turbine_id, v.valid_time) for v in bundle.values]
    if len(actual) != len(set(actual)):
        raise ValueError("DUPLICATE_WEATHER")
    if not required <= set(actual):
        raise ValueError("INCOMPLETE_WEATHER")
    return {"run_id": m.run_id, "required_points": len(required),
            "available_at": m.available_at.isoformat(), "provenance": m.provenance,
            "availability_basis": m.availability_basis}
