import csv
import io
from .schemas import utc

COLUMNS = ["forecast_id", "turbine_id", "origin_time", "target_start", "target_end", "lead_hours",
           "prediction_norm", "q10", "q50", "q90", "weather_run_id", "model_version", "bias_version",
           "status", "mode", "provenance", "release_kind", "prediction_unit"]


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
