"""Empirical Q10/Q50/Q90 of already-issued errors. Not a coverage guarantee."""
import numpy as np
from ..features import LEAD_GROUPS, lead_group
from ..schemas import PredictionBatch, PredictionRow


def fit_intervals(records, *, min_samples=30, allow_turbine_fallback=False):
    if min_samples < 2:
        raise ValueError("INTERVAL_MIN_SAMPLES: at least two required")
    result = {}
    for turbine in sorted({r.turbine_id for r in records}):
        pool = [r for r in records if r.turbine_id == turbine]
        # A fallback pool counts each actual hour once, even across lead groups.
        unique = {}
        for r in sorted(pool, key=lambda r: (r.origin_time, r.forecast_id)):
            unique[(r.target_start, r.target_end)] = r
        groups = {}
        for group in LEAD_GROUPS:
            rows = [r for r in pool if lead_group(r.lead_hours) == group]
            status = "group"
            if len(rows) < min_samples and allow_turbine_fallback:
                rows, status = list(unique.values()), "turbine_pool"
            if len(rows) >= min_samples:
                groups[group] = {"offsets": np.quantile([r.issued_error for r in rows],
                    [0.1, 0.5, 0.9]).tolist(), "count": len(rows), "status": status}
        result[turbine] = groups
    return {"target_coverage": 0.8, "min_samples": min_samples,
            "fallback_enabled": allow_turbine_fallback, "turbines": result}


def apply_intervals(batch, bias, *, origin_time):
    calibration = bias.parameters.get("intervals", {}) if bias else {}
    rows = []
    for row in batch.rows:
        group = lead_group((row.target_start - origin_time).total_seconds() / 3600)
        info = calibration.get("turbines", {}).get(row.turbine_id, {}).get(group)
        quantiles = dict(q10=None, q50=None, q90=None)
        status = "insufficient_history"
        if info:
            offsets = np.asarray(info["offsets"], dtype=float)
            if offsets.shape != (3,) or not np.isfinite(offsets).all() or np.any(np.diff(offsets) < 0):
                raise ValueError("INVALID_INTERVAL_CALIBRATION")
            q = np.clip(row.prediction_norm + offsets, 0, 1)
            quantiles = dict(zip(("q10", "q50", "q90"), map(float, q)))
            status = info["status"]
        rows.append(PredictionRow(**{**row.model_dump(), **quantiles,
            "status": row.status + "|interval:" + status}))
    return PredictionBatch(rows=tuple(rows))
