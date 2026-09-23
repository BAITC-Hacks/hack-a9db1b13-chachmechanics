"""Read-only quality decision; the integrator owns saving/applying the proposal."""
from dataclasses import dataclass
import math

from ..evaluate import compare_forecasts, residuals_from_forecasts
from ..models.bias import select_residuals, update_bias
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
    def review(self, forecasts, observations, *, model_id, as_of, base_batches=None,
               previous=None, min_interval_samples=30, drift_threshold=0.1,
               min_drift_samples=48, interval_fallback=False):
        as_of = utc(as_of)
        if not math.isfinite(drift_threshold) or drift_threshold < 0 or min_drift_samples < 1:
            raise ValueError("INVALID_CRITIC_THRESHOLDS")
        forecasts = tuple(f for f in forecasts if f.model_id == model_id and f.origin_time <= as_of)
        errors = residuals_from_forecasts(forecasts, observations, as_of=as_of, base_batches=base_batches)
        selected = select_residuals(errors, model_id=model_id, as_of=as_of)
        bias = update_bias(errors, model_id=model_id, as_of=as_of, previous=previous,
                           min_interval_samples=min_interval_samples, interval_fallback=interval_fallback)
        evaluation = compare_forecasts({"issued": forecasts}, observations, as_of=as_of) if forecasts else None
        changed = bias is not None and (previous is None or bias.bias_id != previous.bias_id)
        reasons = []
        if previous and previous.model_id != model_id:
            reasons.append("MODEL_CHANGED_RESET")
        if not selected:
            reasons.append("NO_MATURE_ERRORS" if bias is None else "ERROR_WINDOW_EXPIRED")
        elif not changed:
            reasons.append("NO_NEW_ACTUALS")
        else:
            reasons.append("MATURE_OUT_OF_SAMPLE_ERRORS")
        # This is a signal only, never automatic retraining or causal attribution.
        from ..models.bias import drift_signal
        drift = drift_signal(selected, threshold=drift_threshold, min_samples=min_drift_samples)
        if drift:
            reasons.append("PERSISTENT_SIGNED_ERROR_REVIEW_REQUIRED")
        last = bias.last_actual_available_at if bias else None
        return CriticDecision("propose_bias" if changed else "skip", tuple(reasons), bias, evaluation,
            len(selected), last.isoformat() if last else None,
            (as_of-last).total_seconds()/3600 if last else None, drift)
