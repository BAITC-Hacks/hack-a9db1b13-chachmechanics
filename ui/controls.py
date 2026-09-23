"""Validate choices against the catalogue supplied by the service."""
from frontend.contracts import UIError, utc


def initial_selection(catalog):
    origins = catalog.get("origins", [])
    turbines = catalog.get("turbines", [])
    if not origins or not turbines:
        raise UIError("NO_DATA")
    preferred = next((o for o in origins if o["forecast_id"] == catalog.get("default_forecast_id")), origins[0])
    origins = [preferred]
    return {"origin_time": origins[0]["origin_time"], "forecast_id": origins[0]["forecast_id"], "turbine_id": turbines[0]["id"],
            "horizon_hours": 48, "compare": False, "as_of": origins[0]["origin_time"]}


def validate_selection(selection, catalog):
    if selection["horizon_hours"] not in (24, 48):
        raise UIError("INVALID_SELECTION")
    if selection["turbine_id"] not in [row["id"] for row in catalog["turbines"]]:
        raise UIError("INVALID_SELECTION")
    origins = {row["forecast_id"]: row for row in catalog["origins"]}
    if selection["forecast_id"] not in origins or origins[selection["forecast_id"]]["origin_time"] != selection["origin_time"]:
        raise UIError("INVALID_SELECTION")
    allowed_clocks = [selection["origin_time"], *origins[selection["forecast_id"]].get("inspection_times", [])]
    if selection["as_of"] not in allowed_clocks or utc(selection["as_of"]) < utc(selection["origin_time"]):
        raise UIError("INVALID_SELECTION")
