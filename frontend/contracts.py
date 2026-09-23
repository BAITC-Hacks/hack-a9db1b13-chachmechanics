"""Presentation-side checks, not a replacement for windoracle.schemas.

Wire methods are documented in docs/UI_INTEGRATION.md. UTC is mandatory.
"""
from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite


class UIError(Exception):
    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(code)


def utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone missing")
        return parsed.astimezone(timezone.utc)
    except (ValueError, AttributeError, TypeError) as exc:
        raise UIError("INVALID_TIME") from exc


def wire(value):
    """Accept JSON mappings or backend Pydantic objects without importing ML."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def validate_result(result: dict, *, mode: str, as_of: str) -> dict:
    required = ("forecast_id", "origin_time", "run_id", "model_id", "predictions", "provenance")
    if not isinstance(result, dict) or any(key not in result for key in required):
        raise UIError("CONTRACT_MISMATCH")
    origin, clock = utc(result["origin_time"]), utc(as_of)
    if origin > clock:
        raise UIError("FUTURE_DATA")
    provenance = result["provenance"]
    if not isinstance(provenance, dict):
        raise UIError("CONTRACT_MISMATCH")
    if mode != "fixture" and (result.get("synthetic") or provenance.get("kind") != "operational_archive"):
        raise UIError("UNVERIFIED_SOURCE")
    for field in ("weather_available_at", "model_activated_at", "training_cutoff"):
        if provenance.get(field) is None:
            raise UIError("CONTRACT_MISMATCH")
        if utc(provenance[field]) > origin:
            raise UIError("FUTURE_DATA")
    if utc(provenance["training_cutoff"]) > utc(provenance["model_activated_at"]):
        raise UIError("CONTRACT_MISMATCH")
    if not isinstance(result["predictions"], list):
        raise UIError("CONTRACT_MISMATCH")
    seen = set()
    for row in result["predictions"]:
        if not all(key in row for key in ("turbine_id", "target_start", "target_end", "prediction_norm")):
            raise UIError("CONTRACT_MISMATCH")
        start, end = utc(row["target_start"]), utc(row["target_end"])
        if start <= origin or (end - start).total_seconds() != 3600:
            raise UIError("CONTRACT_MISMATCH")
        key = (row["turbine_id"], start)
        if key in seen:
            raise UIError("CONTRACT_MISMATCH")
        seen.add(key)
        for field in ("prediction_norm", "q10", "q50", "q90"):
            value = row.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 <= value <= 1):
                raise UIError("CONTRACT_MISMATCH")
        quantiles = [row.get(field) for field in ("q10", "q50", "q90")]
        if any(v is not None for v in quantiles) and (any(v is None for v in quantiles) or quantiles != sorted(quantiles)):
            raise UIError("CONTRACT_MISMATCH")
    return result


def visible_actuals(rows: list, as_of: str) -> list:
    """Defence in depth: never display unavailable or unfinished observations."""
    clock = utc(as_of)
    visible = []
    for row in rows:
        if not row.get("available_at") or not row.get("target_end"):
            continue
        if utc(row["available_at"]) <= clock and utc(row["target_end"]) <= clock:
            value = row.get("power_norm")
            if value is not None and isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) and 0 <= value <= 1:
                visible.append(row)
    return visible
