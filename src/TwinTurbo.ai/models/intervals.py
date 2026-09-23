"""Empirical forecast intervals from matured, already-corrected errors."""
from __future__ import annotations

from datetime import timedelta
import math

import numpy as np

from ..features import LEAD_GROUPS, lead_group
from ..schemas import BiasState, PredictionBatch, PredictionRow, utc


INTERVAL_SCHEMA = "empirical-residual-quantiles-v1"


def _finite_vector(values, code: str) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(code)
    return array


def interval_coverage(actual, q10, q90) -> float | None:
    """Inclusive empirical coverage of Q10--Q90; ``None`` for no samples."""

    actual = _finite_vector(actual, "INVALID_INTERVAL_INPUT")
    lower = _finite_vector(q10, "INVALID_INTERVAL_INPUT")
    upper = _finite_vector(q90, "INVALID_INTERVAL_INPUT")
    if not (actual.shape == lower.shape == upper.shape):
        raise ValueError("INVALID_INTERVAL_INPUT")
    if np.any((actual < 0) | (actual > 1) | (lower < 0) | (upper > 1) | (lower > upper)):
        raise ValueError("INVALID_INTERVAL_BOUNDS")
    if not len(actual):
        return None
    return float(np.mean((lower <= actual) & (actual <= upper)))


def mean_interval_width(q10, q90) -> float | None:
    """Mean normalized Q10--Q90 width; ``None`` for no samples."""

    lower = _finite_vector(q10, "INVALID_INTERVAL_INPUT")
    upper = _finite_vector(q90, "INVALID_INTERVAL_INPUT")
    if lower.shape != upper.shape:
        raise ValueError("INVALID_INTERVAL_INPUT")
    if np.any((lower < 0) | (upper > 1) | (lower > upper)):
        raise ValueError("INVALID_INTERVAL_BOUNDS")
    if not len(lower):
        return None
    return float(np.mean(upper - lower))


def fit_intervals(
    records,
    *,
    min_samples: int = 30,
    allow_turbine_fallback: bool = False,
) -> dict:
    """Fit error quantiles by turbine and lead group.

    Errors are ``actual - p_issued``.  Thus a caller applies the offsets to an
    already bias-corrected point prediction and never adds the bias twice.  A
    broader same-turbine pool is used only when explicitly enabled.
    """

    from .bias import Residual

    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 2:
        raise ValueError("INTERVAL_MIN_SAMPLES")
    rows = []
    seen = {}
    for raw in records:
        if not isinstance(raw, Residual) and hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        row = Residual.model_validate(raw.model_dump() if isinstance(raw, Residual) else raw)
        key = (
            row.forecast_id,
            row.turbine_id,
            row.target_start,
            row.target_end,
            row.actual_revision,
        )
        previous = seen.get(key)
        if previous is not None and previous != row:
            raise ValueError("CONFLICTING_INTERVAL_RESIDUAL")
        if previous is None:
            seen[key] = row
            rows.append(row)

    result = {}
    for turbine in sorted({row.turbine_id for row in rows}):
        pool = tuple(row for row in rows if row.turbine_id == turbine)
        # A fallback pool counts an actual target once even when it appeared at
        # several lead groups.  The latest designated origin wins.
        fallback_by_target = {}
        for row in sorted(pool, key=lambda item: (item.origin_time, item.forecast_id)):
            fallback_by_target[(row.target_start, row.target_end)] = row
        fallback_pool = tuple(fallback_by_target.values())
        groups = {}
        for group in LEAD_GROUPS:
            group_rows = tuple(row for row in pool if lead_group(row.lead_hours) == group)
            calibration_rows = group_rows
            status = "group"
            if len(group_rows) < min_samples and allow_turbine_fallback:
                calibration_rows = fallback_pool
                status = "turbine_pool"

            offsets = None
            if len(calibration_rows) >= min_samples:
                offsets = [
                    float(value)
                    for value in np.quantile(
                        [row.issued_error for row in calibration_rows],
                        (0.1, 0.5, 0.9),
                        method="linear",
                    )
                ]
            else:
                status = "insufficient_history"
            groups[group] = {
                "offsets": offsets,
                "group_count": len(group_rows),
                "calibration_count": len(calibration_rows),
                "status": status,
            }
        result[turbine] = groups
    return {
        "schema": INTERVAL_SCHEMA,
        "target_coverage": 0.8,
        "min_samples": min_samples,
        "fallback_enabled": bool(allow_turbine_fallback),
        "error": "actual_minus_issued_prediction",
        "turbines": result,
    }


def _append_status(status: str, value: str) -> str:
    tokens = [token for token in status.split("|") if token]
    if value not in tokens:
        tokens.append(value)
    return "|".join(tokens)


def apply_intervals(
    batch: PredictionBatch,
    bias: BiasState | None,
    *,
    origin_time,
) -> PredictionBatch:
    """Attach Q10/Q50/Q90 exactly once to corrected point predictions."""

    origin_time = utc(origin_time)
    batch = PredictionBatch.model_validate(batch.model_dump())
    if any(
        any(token.startswith("interval:") for token in row.status.split("|"))
        or any(value is not None for value in (row.q10, row.q50, row.q90))
        for row in batch.rows
    ):
        raise ValueError("INTERVALS_ALREADY_APPLIED")

    calibration = {}
    expired = False
    if bias is not None:
        bias = BiasState.model_validate(bias.model_dump())
        if bias.created_as_of > origin_time:
            raise ValueError("FUTURE_BIAS")
        calibration = bias.parameters.get("intervals", {})
        if calibration and calibration.get("schema") != INTERVAL_SCHEMA:
            raise ValueError("UNKNOWN_INTERVAL_SCHEMA")
        window_days = bias.parameters.get("window_days")
        if (
            isinstance(window_days, bool)
            or not isinstance(window_days, (int, float))
            or not math.isfinite(float(window_days))
            or window_days <= 0
        ):
            raise ValueError("INVALID_ERROR_WINDOW")
        expired = origin_time - bias.last_actual_available_at > timedelta(
            days=float(window_days)
        )

    rows = []
    for row in batch.rows:
        group = lead_group(
            (row.target_start - origin_time).total_seconds() / 3600.0
        )
        info = (
            calibration.get("turbines", {})
            .get(row.turbine_id, {})
            .get(group)
        )
        quantiles = {"q10": None, "q50": None, "q90": None}
        status = "expired_history" if expired else "insufficient_history"
        if info is not None and not expired and info.get("offsets") is not None:
            offsets = np.asarray(info["offsets"], dtype=float)
            if (
                offsets.shape != (3,)
                or not np.isfinite(offsets).all()
                or np.any(np.diff(offsets) < 0)
                or np.any((offsets < -1) | (offsets > 1))
            ):
                raise ValueError("INVALID_INTERVAL_CALIBRATION")
            values = np.clip(row.prediction_norm + offsets, 0.0, 1.0)
            quantiles = dict(
                zip(("q10", "q50", "q90"), (float(value) for value in values))
            )
            status = info.get("status", "group")
        rows.append(
            PredictionRow(
                **{
                    **row.model_dump(),
                    **quantiles,
                    "status": _append_status(row.status, "interval:" + status),
                }
            )
        )
    return PredictionBatch(rows=tuple(rows))
