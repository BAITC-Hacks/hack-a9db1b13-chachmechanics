"""Leakage-safe rolling bias built from saved out-of-sample forecasts.

The functions in this module are deliberately pure.  They neither query the
store nor persist the proposed :class:`~windoracle.schemas.BiasState`; the
integration layer owns both operations.
"""
from __future__ import annotations

from datetime import timedelta
import math
from typing import Literal

import numpy as np
from pydantic import model_validator

from ..features import LEAD_GROUPS, lead_group
from ..schemas import (
    BiasState,
    Contract,
    Power,
    PredictionBatch,
    PredictionRow,
    UTCDateTime,
    digest,
    utc,
)


BIAS_SCHEMA = "rolling-bias-v1"
BiasEstimator = Literal["mean", "median"]


class Residual(Contract):
    """One matured error from a forecast that was actually issued.

    ``base_error`` is used to learn bias.  ``issued_error`` is used for
    interval calibration, because it already contains the effect of whatever
    bias was active for that historical issue.
    """

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
    def check_temporal_order(self):
        if not all((self.forecast_id, self.model_id, self.turbine_id, self.actual_revision)):
            raise ValueError("RESIDUAL_IDENTIFIERS_REQUIRED")
        if self.target_end - self.target_start != timedelta(hours=1):
            raise ValueError("RESIDUAL_INTERVAL_MUST_BE_ONE_HOUR")
        if not self.training_cutoff <= self.origin_time < self.target_start:
            raise ValueError("RESIDUAL_NOT_OUT_OF_SAMPLE")
        if self.actual_available_at < self.target_end:
            raise ValueError("ACTUAL_AVAILABLE_BEFORE_TARGET_END")
        lead_group(self.lead_hours)
        return self

    @property
    def lead_hours(self) -> float:
        return (self.target_start - self.origin_time).total_seconds() / 3600.0

    @property
    def base_error(self) -> float:
        """Signed error used by bias: actual minus uncorrected prediction."""

        return self.actual - self.p_base

    @property
    def issued_error(self) -> float:
        """Signed error after the correction active for the saved issue."""

        return self.actual - self.p_issued


def _residual(value) -> Residual:
    if isinstance(value, Residual):
        # Revalidation also catches unsafe ``model_copy(update=...)`` values.
        return Residual.model_validate(value.model_dump())
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return Residual.model_validate(value)


def _positive_number(value, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(code)
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(code)
    return value


def _bias_estimator(value: object) -> BiasEstimator:
    if value == "mean":
        return "mean"
    if value == "median":
        return "median"
    raise ValueError("INVALID_BIAS_ESTIMATOR")


def select_residuals(records, *, model_id: str, as_of, window_days: float = 21):
    """Return deterministic, mature residuals for one model and rolling window.

    Fact revisions supersede earlier revisions.  Update releases do not add
    statistical weight.  If several scheduled origins predict the same target
    within one lead group, the latest origin is the designated release.
    """

    if not model_id:
        raise ValueError("MODEL_ID_REQUIRED")
    as_of = utc(as_of)
    window_days = _positive_number(window_days, "INVALID_ERROR_WINDOW")
    window_start = as_of - timedelta(days=window_days)

    revisions: dict[tuple, Residual] = {}
    for raw in records:
        row = _residual(raw)
        if row.model_id != model_id or row.release_kind != "scheduled":
            continue
        if row.actual_available_at > as_of:
            continue
        if not window_start < row.target_end <= as_of:
            continue
        key = (row.forecast_id, row.turbine_id, row.target_start, row.target_end)
        previous = revisions.get(key)
        if previous is not None and previous.actual_available_at == row.actual_available_at:
            if previous != row:
                raise ValueError("CONFLICTING_RESIDUAL_REVISION")
            continue
        if previous is None or row.actual_available_at > previous.actual_available_at:
            revisions[key] = row

    designated: dict[tuple, Residual] = {}
    for row in revisions.values():
        key = (row.turbine_id, row.target_start, row.target_end, lead_group(row.lead_hours))
        previous = designated.get(key)
        if previous is None or (row.origin_time, row.forecast_id) > (
            previous.origin_time,
            previous.forecast_id,
        ):
            designated[key] = row
    return tuple(designated[key] for key in sorted(designated))


def residual_summary(records) -> dict:
    """Summarize base and issued errors without changing their sign convention."""

    rows = tuple(_residual(row) for row in records)

    def metrics(values: tuple[float, ...]) -> dict:
        if not values:
            return {"sample_count": 0, "mean_error": None, "mae": None, "rmse": None}
        array = np.asarray(values, dtype=float)
        return {
            "sample_count": len(values),
            "mean_error": float(np.mean(array)),
            "mae": float(np.mean(np.abs(array))),
            "rmse": float(np.sqrt(np.mean(np.square(array)))),
        }

    def one(pool: tuple[Residual, ...]) -> dict:
        return {
            "base": metrics(tuple(row.base_error for row in pool)),
            "issued": metrics(tuple(row.issued_error for row in pool)),
        }

    return {
        "error_sign": "actual_minus_prediction",
        "all": one(rows),
        "by_turbine": {
            turbine: one(tuple(row for row in rows if row.turbine_id == turbine))
            for turbine in sorted({row.turbine_id for row in rows})
        },
    }


def update_bias(
    records,
    *,
    model_id: str,
    as_of,
    previous: BiasState | None = None,
    window_days: float = 21,
    shrinkage: float = 48.0,
    estimator: BiasEstimator = "mean",
    min_interval_samples: int = 30,
    interval_fallback: bool = False,
) -> BiasState | None:
    """Propose an immutable bias state from matured OOS base errors.

    Recalculation starts from the selected saved residuals instead of updating
    aggregates in-place.  Exact retries therefore cannot double-count facts.
    With no newly available fact and unchanged options, the previous state is
    returned unchanged.

    ``mean`` preserves the original correction.  ``median`` estimates the
    robust residual centre aligned with MAE; lead-group centres are shrunk
    toward the corresponding turbine-wide centre in either mode.
    """

    from .intervals import fit_intervals

    if not model_id:
        raise ValueError("MODEL_ID_REQUIRED")
    as_of = utc(as_of)
    window_days = _positive_number(window_days, "INVALID_ERROR_WINDOW")
    if isinstance(shrinkage, bool) or not isinstance(shrinkage, (int, float)):
        raise ValueError("INVALID_SHRINKAGE")
    shrinkage = float(shrinkage)
    if not math.isfinite(shrinkage) or shrinkage < 0:
        raise ValueError("INVALID_SHRINKAGE")
    estimator = _bias_estimator(estimator)

    compatible_previous = None
    if previous is not None:
        previous = BiasState.model_validate(previous.model_dump())
        if previous.created_as_of > as_of:
            raise ValueError("FUTURE_BIAS")
        if previous.model_id == model_id:
            compatible_previous = previous

    selected = select_residuals(
        records, model_id=model_id, as_of=as_of, window_days=window_days
    )
    if not selected:
        return compatible_previous

    options = {
        "window_days": window_days,
        "shrinkage": shrinkage,
        "estimator": estimator,
        "min_interval_samples": min_interval_samples,
        "interval_fallback": bool(interval_fallback),
    }
    source_hash = digest(
        {
            "rows": [row.model_dump(mode="json") for row in selected],
            "options": options,
        }
    )
    if compatible_previous is not None:
        if compatible_previous.parameters.get("source_hash") == source_hash:
            return compatible_previous
        # States written before the estimator option was introduced cannot
        # reproduce the current source hash even when their selected rows and
        # effective options are unchanged.  Keep that one migration retry
        # idempotent.  Current states must rely on the complete source hash:
        # an equal latest timestamp does not imply an equal source set (a new
        # row may share it), and advancing the rolling boundary may remove
        # rows while leaving the newest timestamp unchanged.
        if "estimator" not in compatible_previous.parameters:
            newest_fact = max(row.actual_available_at for row in selected)
            same_options = all(
                (
                    compatible_previous.parameters.get(name, "mean")
                    if name == "estimator"
                    else compatible_previous.parameters.get(name)
                )
                == value
                for name, value in options.items()
            )
            if (
                estimator == "mean"
                and newest_fact <= compatible_previous.last_actual_available_at
                and same_options
            ):
                return compatible_previous

    turbines = {}
    for turbine in sorted({row.turbine_id for row in selected}):
        pool = tuple(row for row in selected if row.turbine_id == turbine)
        pool_errors = tuple(row.base_error for row in pool)
        global_bias = float(
            np.mean(pool_errors) if estimator == "mean" else np.median(pool_errors)
        )
        groups = {}
        for group in LEAD_GROUPS:
            errors = tuple(
                row.base_error for row in pool if lead_group(row.lead_hours) == group
            )
            count = len(errors)
            denominator = count + shrinkage
            if not denominator:
                value = global_bias
            elif estimator == "mean":
                # Preserve the original/default estimator exactly.
                value = (sum(errors) + shrinkage * global_bias) / denominator
            else:
                group_center = float(np.median(errors)) if errors else global_bias
                value = (
                    count * group_center + shrinkage * global_bias
                ) / denominator
            groups[group] = {
                "bias": float(value),
                "count": count,
                "status": "calibrated" if count else "turbine_pool",
            }
        turbines[turbine] = {
            "global_bias": global_bias,
            "count": len(pool),
            "groups": groups,
        }

    parameters = {
        "schema": BIAS_SCHEMA,
        "source_hash": source_hash,
        **options,
        "status": "calibrated",
        "turbines": turbines,
        "selection": (
            "latest scheduled origin per target and lead group; "
            "latest actual revision available as-of"
        ),
        "intervals": fit_intervals(
            selected,
            min_samples=min_interval_samples,
            allow_turbine_fallback=interval_fallback,
        ),
    }
    last_actual = max(row.actual_available_at for row in selected)
    identity = {
        "model_id": model_id,
        "created_as_of": as_of.isoformat(),
        "last_actual_available_at": last_actual.isoformat(),
        "parameters": parameters,
    }
    return BiasState(
        bias_id="bias-" + digest(identity)[:24],
        model_id=model_id,
        created_as_of=as_of,
        last_actual_available_at=last_actual,
        parameters=parameters,
    )


def _append_status(status: str, value: str) -> str:
    tokens = [token for token in status.split("|") if token]
    if value not in tokens:
        tokens.append(value)
    return "|".join(tokens)


def _has_status(status: str, prefix: str) -> bool:
    return any(token.startswith(prefix) for token in status.split("|"))


def apply_bias(
    batch: PredictionBatch,
    bias: BiasState | None,
    *,
    model_id: str,
    origin_time,
) -> PredictionBatch:
    """Apply one compatible bias state exactly once to a point batch."""

    origin_time = utc(origin_time)
    batch = PredictionBatch.model_validate(batch.model_dump())
    if any(_has_status(row.status, "bias:") for row in batch.rows):
        raise ValueError("BIAS_ALREADY_APPLIED")
    if any(any(value is not None for value in (row.q10, row.q50, row.q90)) for row in batch.rows):
        raise ValueError("BIAS_REQUIRES_POINT_PREDICTIONS")

    if bias is None:
        return PredictionBatch(
            rows=tuple(
                PredictionRow(
                    **{
                        **row.model_dump(),
                        "status": _append_status(
                            row.status, "bias:insufficient_history"
                        ),
                    }
                )
                for row in batch.rows
            )
        )

    bias = BiasState.model_validate(bias.model_dump())
    if bias.model_id != model_id or bias.created_as_of > origin_time:
        raise ValueError("INADMISSIBLE_BIAS")
    if bias.parameters.get("schema") != BIAS_SCHEMA:
        raise ValueError("UNKNOWN_BIAS_SCHEMA")

    window_days = _positive_number(
        bias.parameters.get("window_days"), "INVALID_ERROR_WINDOW"
    )
    expired = origin_time - bias.last_actual_available_at > timedelta(days=window_days)
    rows = []
    for row in batch.rows:
        group = lead_group(
            (row.target_start - origin_time).total_seconds() / 3600.0
        )
        entry = (
            bias.parameters.get("turbines", {})
            .get(row.turbine_id, {})
            .get("groups", {})
            .get(group)
        )
        if expired:
            value, status = 0.0, "expired_history"
        elif entry is None:
            value, status = 0.0, "insufficient_history"
        else:
            value = entry.get("bias")
            status = entry.get("status", "calibrated")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not -1 <= float(value) <= 1
            ):
                raise ValueError("INVALID_BIAS_VALUE")
            value = float(value)
        rows.append(
            PredictionRow(
                **{
                    **row.model_dump(),
                    "prediction_norm": float(
                        np.clip(row.prediction_norm + value, 0.0, 1.0)
                    ),
                    "status": _append_status(row.status, "bias:" + status),
                }
            )
        )
    return PredictionBatch(rows=tuple(rows))


def drift_signal(records, *, threshold: float = 0.1, min_samples: int = 48) -> bool:
    """Flag persistent per-turbine issued error without causal attribution."""

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or threshold < 0
        or isinstance(min_samples, bool)
        or not isinstance(min_samples, int)
        or min_samples < 1
    ):
        raise ValueError("INVALID_DRIFT_THRESHOLDS")
    rows = tuple(_residual(row) for row in records)
    for turbine in {row.turbine_id for row in rows}:
        errors = [row.issued_error for row in rows if row.turbine_id == turbine]
        if len(errors) >= min_samples and abs(float(np.mean(errors))) > threshold:
            return True
    return False
