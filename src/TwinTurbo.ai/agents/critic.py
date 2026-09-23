"""Read-only quality review and bias-state proposal.

Critic never writes to the store and never mutates an already published
forecast.  It reports the evidence and leaves activation to the integration
layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math

from ..models.bias import (
    BiasEstimator,
    Residual,
    drift_signal,
    residual_summary,
    select_residuals,
    update_bias,
)
from ..schemas import BiasState, utc


@dataclass(frozen=True)
class CriticDecision:
    action: str
    reasons: tuple[str, ...]
    proposed_bias: BiasState | None
    evaluation: dict | None
    sample_count: int
    last_actual_available_at: str | None
    actual_age_hours: float | None
    retrain_recommended: bool


class Critic:
    """Calculate errors and propose, but never persist, a correction state."""

    def review_residuals(
        self,
        records,
        *,
        model_id: str,
        as_of,
        previous: BiasState | None = None,
        window_days: float = 21,
        shrinkage: float = 48.0,
        estimator: BiasEstimator = "mean",
        min_interval_samples: int = 30,
        drift_threshold: float = 0.1,
        min_drift_samples: int = 48,
        interval_fallback: bool = False,
    ) -> CriticDecision:
        as_of = utc(as_of)
        if (
            isinstance(drift_threshold, bool)
            or not isinstance(drift_threshold, (int, float))
            or not math.isfinite(float(drift_threshold))
            or drift_threshold < 0
            or isinstance(min_drift_samples, bool)
            or not isinstance(min_drift_samples, int)
            or min_drift_samples < 1
        ):
            raise ValueError("INVALID_CRITIC_THRESHOLDS")

        rows = tuple(
            Residual.model_validate(
                row.model_dump() if hasattr(row, "model_dump") else row
            )
            for row in records
        )
        selected = select_residuals(
            rows, model_id=model_id, as_of=as_of, window_days=window_days
        )
        candidate = update_bias(
            rows,
            model_id=model_id,
            as_of=as_of,
            previous=previous,
            window_days=window_days,
            shrinkage=shrinkage,
            estimator=estimator,
            min_interval_samples=min_interval_samples,
            interval_fallback=interval_fallback,
        )

        compatible_previous = previous if previous and previous.model_id == model_id else None
        changed = candidate is not None and (
            compatible_previous is None
            or candidate.bias_id != compatible_previous.bias_id
        )
        reasons = []
        if previous is not None and previous.model_id != model_id:
            reasons.append("MODEL_CHANGED_RESET")
        estimator_changed = (
            compatible_previous is not None
            and compatible_previous.parameters.get("estimator", "mean")
            != estimator
        )
        if not selected:
            reasons.append("NO_MATURE_ERRORS")
        elif changed and estimator_changed:
            reasons.append("BIAS_ESTIMATOR_CHANGED")
        elif changed:
            reasons.append("MATURE_OUT_OF_SAMPLE_ERRORS")
        else:
            reasons.append("NO_NEW_ACTUALS")

        effective = candidate or compatible_previous
        if changed:
            interval_groups = (
                effective.parameters.get("intervals", {}).get("turbines", {})
                if effective
                else {}
            )
            interval_ready = any(
                info.get("offsets") is not None
                for groups in interval_groups.values()
                for info in groups.values()
            )
            reasons.append(
                "INTERVALS_CALIBRATED"
                if interval_ready
                else "INTERVALS_INSUFFICIENT_HISTORY"
            )

        drift = drift_signal(
            selected,
            threshold=float(drift_threshold),
            min_samples=min_drift_samples,
        )
        if drift:
            reasons.append("PERSISTENT_SIGNED_ERROR_REVIEW_REQUIRED")

        last_actual = effective.last_actual_available_at if effective else None
        if last_actual is not None and as_of - last_actual > timedelta(days=window_days):
            reasons.append("BIAS_HISTORY_STALE")
        return CriticDecision(
            action="propose_bias" if changed else "skip",
            reasons=tuple(reasons),
            proposed_bias=candidate if changed else None,
            evaluation=residual_summary(selected) if selected else None,
            sample_count=len(selected),
            last_actual_available_at=last_actual.isoformat() if last_actual else None,
            actual_age_hours=(as_of - last_actual).total_seconds() / 3600.0
            if last_actual
            else None,
            retrain_recommended=drift,
        )

    def review(
        self,
        forecasts_or_residuals,
        observations=None,
        *,
        model_id: str,
        as_of,
        base_batches=None,
        **kwargs,
    ) -> CriticDecision:
        """Review residuals directly or derive them from saved forecast inputs.

        When forecasts are supplied, their base batches must accompany any
        corrected issue; inversion after clipping is intentionally forbidden by
        :func:`windoracle.evaluate.residuals_from_forecasts`.
        """

        if observations is None:
            records = forecasts_or_residuals
        else:
            from ..evaluate import residuals_from_forecasts

            records = residuals_from_forecasts(
                forecasts_or_residuals,
                observations,
                as_of=as_of,
                base_batches=base_batches,
            )
        return self.review_residuals(
            records, model_id=model_id, as_of=as_of, **kwargs
        )
