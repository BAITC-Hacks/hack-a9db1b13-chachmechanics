import csv
import io
from .schemas import BiasState, ForecastRequest, ForecastResult, ModelState, WeatherRunMetadata, digest, utc
from .clock import targets

COLUMNS = ["forecast_id", "turbine_id", "origin_time", "target_start", "target_end", "lead_hours",
           "prediction_norm", "q10", "q50", "q90", "weather_run_id", "model_version", "bias_version",
           "status", "mode", "provenance", "release_kind", "prediction_unit"]


def verify_result(result: ForecastResult):
    """Validate portable results without requiring the originating SQLite DB."""
    result = ForecastResult.model_validate(result.model_dump())
    request = ForecastRequest.model_validate(result.manifest["request"])
    state = ModelState.model_validate(result.manifest["model"])
    weather = WeatherRunMetadata.model_validate(result.manifest["weather"])
    if (request.origin_time != result.origin_time or request.mode != result.mode
            or request.release_kind != result.release_kind):
        raise ValueError("RESULT_REQUEST_MISMATCH")
    if state.model_id != result.model_id or state.activated_at > result.origin_time:
        raise ValueError("INADMISSIBLE_MODEL")
    if (weather.run_id != result.run_id or weather.available_at > result.origin_time
            or weather.provenance != result.provenance):
        raise ValueError("INADMISSIBLE_WEATHER")
    bias = BiasState.model_validate(result.manifest["bias"]) if result.manifest.get("bias") else None
    if (bias.bias_id if bias else None) != result.bias_id:
        raise ValueError("BIAS_ID_MISMATCH")
    if bias and (bias.created_as_of > result.origin_time or bias.model_id != result.model_id):
        raise ValueError("INADMISSIBLE_BIAS")
    if result.mode != "fixture" and (state.provenance == "synthetic" or result.provenance != "operational_archive"):
        raise ValueError("NON_OPERATIONAL_FORECAST")
    if result.manifest.get("last_actual_available_at"):
        from datetime import datetime
        actual_available = datetime.fromisoformat(result.manifest["last_actual_available_at"].replace("Z", "+00:00"))
        if utc(actual_available) > result.origin_time:
            raise ValueError("FUTURE_ACTUAL")
    expected = {(t, h.target_start, h.target_end) for t in request.turbine_ids for h in targets(request)}
    actual = [(r.turbine_id, r.target_start, r.target_end) for r in result.predictions.rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("MODEL_OUTPUT_COVERAGE")
    keys = ("request", "model", "bias", "snapshot_hash", "config_hash")
    identity = {k: result.manifest[k] for k in keys}
    if "parent_forecast_id" in result.manifest:
        identity["parent_forecast_id"] = result.manifest["parent_forecast_id"]
        if identity["parent_forecast_id"] != result.parent_forecast_id:
            raise ValueError("PARENT_ID_MISMATCH")
    if digest(identity) != result.forecast_id:
        raise ValueError("FORECAST_ID_MISMATCH")
    return {"forecast_id": result.forecast_id, "rows": len(actual), "mode": result.mode,
            "provenance": result.provenance}


def export_csv(results, *, strict=True, target_start=None, target_end=None, release_policy="all"):
    if release_policy not in ("all", "scheduled", "update"):
        raise ValueError("Unknown release policy")
    if target_start:
        target_start = utc(target_start)
    if target_end:
        target_end = utc(target_end)
    if target_start and target_end and target_start >= target_end:
        raise ValueError("Invalid export interval")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
    writer.writeheader()
    for result in sorted(results, key=lambda r: (r.origin_time, r.forecast_id)):
        verify_result(result)
        if strict and (result.mode == "fixture" or result.provenance != "operational_archive"
                       or result.manifest["model"].get("provenance") == "synthetic"):
            raise ValueError("STRICT_EXPORT_REQUIRES_OPERATIONAL_FORECAST")
        if release_policy != "all" and result.release_kind != release_policy:
            continue
        for row in result.predictions.rows:
            if target_start and row.target_start < target_start:
                continue
            if target_end and row.target_start >= target_end:
                continue
            writer.writerow({**row.model_dump(mode="json"), "forecast_id": result.forecast_id,
                "origin_time": result.origin_time.isoformat(),
                "lead_hours": (row.target_start - result.origin_time).total_seconds() / 3600,
                "weather_run_id": result.run_id, "model_version": result.model_id,
                "bias_version": result.bias_id, "mode": result.mode, "provenance": result.provenance,
                "release_kind": result.release_kind, "prediction_unit": "normalized_power"})
    return buffer.getvalue()
