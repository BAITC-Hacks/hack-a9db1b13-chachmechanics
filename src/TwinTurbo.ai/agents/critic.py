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


CAUSAL_LIMITATION = (
    "Ошибки прогноза показывают статистическое отклонение, но не доказывают "
    "неисправность или ее причину: возможны ошибка модели погоды, датчика, "
    "ограничение станции или изменение доступности оборудования."
)


@dataclass(frozen=True)
class CriticRecommendation:
    """Conservative, machine-readable next step based on issued errors only.

    Recommendations deliberately concern evidence collection, model review and
    inspection.  They never prescribe a dispatch set-point and never diagnose
    a component failure from forecast residuals.
    """

    code: str
    category: str
    severity: str
    next_step: str
    message: str
    turbine_id: str | None
    evidence: dict
    causal_limitation: str = CAUSAL_LIMITATION

    def as_dict(self) -> dict:
        """Return a JSON-friendly representation for service/UI adapters."""

        return {
            "code": self.code,
            "category": self.category,
            "severity": self.severity,
            "next_step": self.next_step,
            "message": self.message,
            "turbine_id": self.turbine_id,
            "evidence": dict(self.evidence),
            "causal_limitation": self.causal_limitation,
        }


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
    recommendations: tuple[CriticRecommendation, ...] = ()


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
        min_recommendation_samples: int = 48,
        maintenance_error_threshold: float = 0.25,
        min_maintenance_samples: int = 96,
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
            or isinstance(min_recommendation_samples, bool)
            or not isinstance(min_recommendation_samples, int)
            or min_recommendation_samples < 1
            or isinstance(maintenance_error_threshold, bool)
            or not isinstance(maintenance_error_threshold, (int, float))
            or not math.isfinite(float(maintenance_error_threshold))
            or not 0 < float(maintenance_error_threshold) <= 1
            or isinstance(min_maintenance_samples, bool)
            or not isinstance(min_maintenance_samples, int)
            or min_maintenance_samples < 1
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
        mature_model_rows = tuple(
            row
            for row in rows
            if row.model_id == model_id
            and row.release_kind == "scheduled"
            and row.actual_available_at <= as_of
        )
        evidence_last_actual = (
            max(row.actual_available_at for row in mature_model_rows)
            if mature_model_rows
            else None
        )
        freshness_reference = last_actual or evidence_last_actual
        history_stale = (
            freshness_reference is not None
            and as_of - freshness_reference > timedelta(days=window_days)
        )
        if history_stale:
            reasons.append("BIAS_HISTORY_STALE")

        evaluation = residual_summary(selected) if selected else None
        recommendations: list[CriticRecommendation] = []
        if len(selected) < min_recommendation_samples:
            recommendations.append(
                CriticRecommendation(
                    code="INSUFFICIENT_MATURE_ERROR_EVIDENCE",
                    category="evidence",
                    severity="info",
                    next_step="collect_mature_actuals",
                    message=(
                        "Недостаточно зрелых ошибок выпущенных прогнозов для "
                        "надежной операционной или технической рекомендации."
                    ),
                    turbine_id=None,
                    evidence={
                        "sample_count": len(selected),
                        "required_sample_count": min_recommendation_samples,
                        "error_sign": "actual_minus_prediction",
                    },
                )
            )
        if history_stale:
            age_hours = (as_of - freshness_reference).total_seconds() / 3600.0
            recommendations.append(
                CriticRecommendation(
                    code="STALE_ERROR_HISTORY",
                    category="data_freshness",
                    severity="warning",
                    next_step="refresh_actuals_before_decision",
                    message=(
                        "История доступного факта устарела; дождитесь или "
                        "восстановите свежие измерения до принятия решения."
                    ),
                    turbine_id=None,
                    evidence={
                        "last_actual_available_at": freshness_reference.isoformat(),
                        "actual_age_hours": age_hours,
                        "maximum_age_hours": float(window_days) * 24.0,
                    },
                )
            )

        # Use the issued (post-correction) error for operational review.  A
        # positive value means the model underpredicted actual generation; a
        # negative value means actual generation was below its prediction.
        if evaluation is not None:
            for turbine_id, summary in evaluation["by_turbine"].items():
                issued = summary["issued"]
                count = int(issued["sample_count"])
                mean_error = issued["mean_error"]
                if mean_error is None:
                    continue
                mean_error = float(mean_error)
                direction = "underprediction" if mean_error > 0 else "overprediction"
                if count >= min_drift_samples and abs(mean_error) > float(
                    drift_threshold
                ):
                    recommendations.append(
                        CriticRecommendation(
                            code="REVIEW_MODEL_PERSISTENT_SIGNED_ERROR",
                            category="model_quality",
                            severity="warning",
                            next_step="review_model_and_inputs",
                            message=(
                                f"Для {turbine_id} сохраняется знаковая ошибка; "
                                "проверьте калибровку модели и входные данные."
                            ),
                            turbine_id=turbine_id,
                            evidence={
                                "sample_count": count,
                                "mean_issued_error": mean_error,
                                "issued_mae": issued["mae"],
                                "threshold": float(drift_threshold),
                                "direction": direction,
                                "error_sign": "actual_minus_prediction",
                            },
                        )
                    )

                # Only persistent *underperformance* can justify an inspection
                # prompt.  It still cannot justify a repair diagnosis: actual
                # below prediction may be curtailment, sensor error or weather
                # mismatch.  Positive error therefore never emits this alert.
                if (
                    count >= min_maintenance_samples
                    and mean_error <= -float(maintenance_error_threshold)
                ):
                    recommendations.append(
                        CriticRecommendation(
                            code="INSPECT_DATA_AND_ASSET_PERSISTENT_UNDERPERFORMANCE",
                            category="maintenance_screening",
                            severity="warning",
                            next_step="inspect_measurements_availability_and_asset",
                            message=(
                                f"Для {turbine_id} факт устойчиво ниже выпущенного "
                                "прогноза: проверьте датчики, ограничения, журнал "
                                "доступности и выполните осмотр. Не назначайте "
                                "ремонт только по этому сигналу."
                            ),
                            turbine_id=turbine_id,
                            evidence={
                                "sample_count": count,
                                "mean_issued_error": mean_error,
                                "issued_mae": issued["mae"],
                                "threshold": float(maintenance_error_threshold),
                                "direction": "actual_below_prediction",
                                "error_sign": "actual_minus_prediction",
                            },
                        )
                    )
        return CriticDecision(
            action="propose_bias" if changed else "skip",
            reasons=tuple(reasons),
            proposed_bias=candidate if changed else None,
            evaluation=evaluation,
            sample_count=len(selected),
            last_actual_available_at=last_actual.isoformat() if last_actual else None,
            actual_age_hours=(as_of - last_actual).total_seconds() / 3600.0
            if last_actual
            else None,
            retrain_recommended=drift,
            recommendations=tuple(recommendations),
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
