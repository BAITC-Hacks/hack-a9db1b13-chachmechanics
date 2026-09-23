"""Versioned rolling bias from saved out-of-sample, uncorrected forecast errors."""
from __future__ import annotations

from datetime import timedelta
import math
from typing import Literal
import numpy as np
from pydantic import model_validator

from ..features import LEAD_GROUPS, lead_group
from ..schemas import (BiasState, Contract, Power, PredictionBatch, PredictionRow,
                       UTCDateTime, digest, utc)


class Residual(Contract):
    forecast_id: str
    model_id: str
    turbine_id: str
    origin_time: UTCDateTime
    target_start: UTCDateTime
    target_end: UTCDateTime
    training_cutoff: UTCDateTime
    actual_available_at: UTCDateTime
    actual_revision: str
    actual: Power
    p_base: Power
    p_issued: Power
    release_kind: Literal["scheduled", "update"] = "scheduled"

    @model_validator(mode="after")
    def check(self):
        if not self.training_cutoff <= self.origin_time < self.target_start < self.target_end <= self.actual_available_at:
            raise ValueError("RESIDUAL_NOT_OUT_OF_SAMPLE")
        if self.target_end - self.target_start != timedelta(hours=1):
            raise ValueError("RESIDUAL_INTERVAL")
        lead_group(self.lead_hours)
        return self

    @property
    def lead_hours(self):
        return (self.target_start - self.origin_time).total_seconds() / 3600

    @property
    def base_error(self):
        return self.actual - self.p_base

    @property
    def issued_error(self):
        return self.actual - self.p_issued


def select_residuals(records, *, model_id, as_of, window_days=21):
    """Latest designated scheduled origin per target/lead group, latest known revision.

    Updates never add weight. Caller supplies saved forecasts, not fitted residuals.
    Revisions supersede earlier facts only in a newly computed state.
    """
    as_of = utc(as_of)
    if not math.isfinite(window_days) or window_days <= 0:
        raise ValueError("INVALID_ERROR_WINDOW")
    versions = {}
    for raw in records:
        r = Residual.model_validate(raw.model_dump())
        if r.model_id != model_id or r.release_kind != "scheduled" or r.actual_available_at > as_of:
            continue
        if not as_of - timedelta(days=window_days) < r.target_end <= as_of:
            continue
        key = r.forecast_id, r.turbine_id, r.target_start, r.target_end
        old = versions.get(key)
        if old is not None and old.actual_available_at == r.actual_available_at and old != r:
            raise ValueError("CONFLICTING_RESIDUAL")
        if old is None or old.actual_available_at < r.actual_available_at:
            versions[key] = r
    designated = {}
    for r in versions.values():
        key = r.turbine_id, r.target_start, r.target_end, lead_group(r.lead_hours)
        old = designated.get(key)
        if old is None or (r.origin_time, r.forecast_id) > (old.origin_time, old.forecast_id):
            designated[key] = r
    return tuple(designated[k] for k in sorted(designated))


def update_bias(records, *, model_id, as_of, previous=None, window_days=21,
                shrinkage=48.0, min_interval_samples=30, interval_fallback=False):
    """Recompute rather than increment: repeats/revisions cannot double-count errors.

    The returned immutable state is a proposal. Participant 1 persists/activates it.
    """
    from .intervals import fit_intervals
    as_of = utc(as_of)
    if not math.isfinite(shrinkage) or shrinkage < 0:
        raise ValueError("INVALID_SHRINKAGE")
    if previous is not None:
        previous = BiasState.model_validate(previous.model_dump())
        if previous.created_as_of > as_of:
            raise ValueError("FUTURE_BIAS")
        if previous.model_id != model_id:
            previous = None
    selected = select_residuals(records, model_id=model_id, as_of=as_of, window_days=window_days)
    if not selected and previous is None:
        return None
    options = {"window_days": window_days, "shrinkage": shrinkage,
               "min_interval_samples": min_interval_samples, "interval_fallback": interval_fallback}
    source_hash = digest({"rows": [r.model_dump(mode="json") for r in selected], "options": options})
    if previous and previous.parameters.get("source_hash") == source_hash:
        return previous
    turbines = {}
    for turbine in sorted({r.turbine_id for r in selected}):
        pool = [r for r in selected if r.turbine_id == turbine]
        global_bias = float(np.mean([r.base_error for r in pool]))
        groups = {}
        for group in LEAD_GROUPS:
            errors = [r.base_error for r in pool if lead_group(r.lead_hours) == group]
            n = len(errors)
            value = ((sum(errors) + shrinkage * global_bias) / (n + shrinkage)
                     if n + shrinkage else global_bias)
            groups[group] = {"bias": value, "count": n,
                             "status": "calibrated" if n else "turbine_pool"}
        turbines[turbine] = {"global_bias": global_bias, "count": len(pool), "groups": groups}
    params = {"schema": "rolling-bias-v1", "source_hash": source_hash, **options,
              "status": "calibrated" if selected else "expired_history", "turbines": turbines,
              "selection": "latest scheduled origin per target and lead group; latest known actual revision",
              "intervals": fit_intervals(selected, min_samples=min_interval_samples,
                                         allow_turbine_fallback=interval_fallback)}
    last = max((r.actual_available_at for r in selected), default=previous.last_actual_available_at if previous else as_of)
    identity = {"model_id": model_id, "as_of": as_of.isoformat(), "parameters": params}
    return BiasState(bias_id="bias-" + digest(identity)[:24], model_id=model_id,
                     created_as_of=as_of, last_actual_available_at=last, parameters=params)


def apply_bias(batch, bias, *, model_id, origin_time):
    origin_time = utc(origin_time)
    if bias is None:
        return PredictionBatch(rows=tuple(PredictionRow(**{**r.model_dump(),
            "status": r.status + "|bias:insufficient_history"}) for r in batch.rows))
    bias = BiasState.model_validate(bias.model_dump())
    if bias.model_id != model_id or bias.created_as_of > origin_time:
        raise ValueError("INADMISSIBLE_BIAS")
    if bias.parameters.get("schema") != "rolling-bias-v1":
        raise ValueError("UNKNOWN_BIAS_SCHEMA")
    rows = []
    for row in batch.rows:
        group = lead_group((row.target_start - origin_time).total_seconds() / 3600)
        entry = bias.parameters.get("turbines", {}).get(row.turbine_id, {}).get("groups", {}).get(group)
        value, status = (entry["bias"], entry["status"]) if entry else (0.0, "insufficient_history")
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("INVALID_BIAS_VALUE")
        rows.append(PredictionRow(**{**row.model_dump(),
            "prediction_norm": float(np.clip(row.prediction_norm + value, 0, 1)),
            "status": row.status + "|bias:" + status}))
    return PredictionBatch(rows=tuple(rows))


def drift_signal(records, *, threshold=0.1, min_samples=48):
    """Per-turbine signed issued error, so opposite turbine errors do not cancel."""
    for turbine in {r.turbine_id for r in records}:
        errors = [r.issued_error for r in records if r.turbine_id == turbine]
        if len(errors) >= min_samples and abs(float(np.mean(errors))) > threshold:
            return True
    return False
