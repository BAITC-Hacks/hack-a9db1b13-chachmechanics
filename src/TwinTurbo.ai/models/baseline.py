"""Per-turbine constant-mean baseline and shared prediction post-processing."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Iterable, Mapping

from ..features import build_features, latest_observations
from ..schemas import (
    AsOfSnapshot,
    BiasState,
    ModelState,
    Observation,
    PredictionBatch,
    PredictionRow,
    utc,
)


@dataclass(frozen=True, slots=True)
class TurbineMean:
    turbine_id: str
    prediction_norm: float
    sample_count: int

    def __post_init__(self) -> None:
        value = float(self.prediction_norm)
        if not self.turbine_id:
            raise ValueError("turbine_id must not be empty")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Baseline prediction must be finite and in [0, 1]")
        if self.sample_count <= 0:
            raise ValueError("Baseline requires at least one sample")
        object.__setattr__(self, "prediction_norm", value)

    def to_dict(self) -> dict[str, object]:
        return {
            "turbine_id": self.turbine_id,
            "prediction_norm": self.prediction_norm,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TurbineMean":
        return cls(
            turbine_id=str(value["turbine_id"]),
            prediction_norm=float(value["prediction_norm"]),
            sample_count=int(value["sample_count"]),
        )


def fit_turbine_means(
    observations: Iterable[Observation],
    *,
    turbine_ids: Iterable[str] | None = None,
    cutoff: datetime | None = None,
) -> tuple[TurbineMean, ...]:
    """Fit arithmetic train means independently for every requested turbine."""

    limit = utc(cutoff) if cutoff is not None else None
    values: dict[str, list[float]] = {}
    for observation in observations:
        if limit is not None and observation.available_at > limit:
            continue
        if observation.quality_flag != "complete" or observation.power_norm is None:
            continue
        values.setdefault(observation.turbine_id, []).append(float(observation.power_norm))
    requested = set(turbine_ids) if turbine_ids is not None else set(values)
    if not requested:
        raise ValueError("BASELINE_NO_TURBINES")
    missing = requested - values.keys()
    if missing:
        raise ValueError("BASELINE_NO_TRAINING_DATA: " + ",".join(sorted(missing)))
    return tuple(
        TurbineMean(
            turbine_id=turbine_id,
            prediction_norm=math.fsum(values[turbine_id]) / len(values[turbine_id]),
            sample_count=len(values[turbine_id]),
        )
        for turbine_id in sorted(requested)
    )


def postprocess_predictions(
    batch: PredictionBatch,
    bias: BiasState | None,
    *,
    model_id: str,
    origin_time: datetime,
) -> PredictionBatch:
    """Apply the optional bias and interval components at one shared boundary."""

    try:
        from .bias import apply_bias
    except ImportError as exc:
        if bias is not None:
            raise ValueError("BIAS_COMPONENT_UNAVAILABLE") from exc
        corrected = batch
    else:
        corrected = apply_bias(batch, bias, model_id=model_id, origin_time=origin_time)

    try:
        from .intervals import apply_intervals
    except ImportError:
        return corrected
    return apply_intervals(corrected, bias, origin_time=origin_time)


@dataclass(frozen=True, slots=True)
class ConstantBaselinePredictor:
    """A deterministic Predictor using each turbine's training mean."""

    state: ModelState
    means: tuple[TurbineMean, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", ModelState.model_validate(self.state.model_dump()))
        if not self.means:
            raise ValueError("Baseline predictor requires turbine means")
        ids = [item.turbine_id for item in self.means]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate turbine baseline")
        object.__setattr__(self, "means", tuple(sorted(self.means, key=lambda item: item.turbine_id)))

    @property
    def means_by_turbine(self) -> dict[str, float]:
        return {item.turbine_id: item.prediction_norm for item in self.means}

    def predict_base(self, snapshot: AsOfSnapshot) -> PredictionBatch:
        features = build_features(snapshot)
        means = self.means_by_turbine
        missing = {row.turbine_id for row in features} - means.keys()
        if missing:
            raise ValueError("BASELINE_UNKNOWN_TURBINE: " + ",".join(sorted(missing)))
        return PredictionBatch(rows=tuple(
            PredictionRow(
                turbine_id=row.turbine_id,
                target_start=row.target_start,
                target_end=row.target_end,
                prediction_norm=means[row.turbine_id],
            )
            for row in features
        ))

    def predict(
        self, snapshot: AsOfSnapshot, bias: BiasState | None = None
    ) -> PredictionBatch:
        return postprocess_predictions(
            self.predict_base(snapshot),
            bias,
            model_id=self.state.model_id,
            origin_time=snapshot.origin_time,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "constant_baseline",
            "state": self.state.model_dump(mode="json"),
            "means": [item.to_dict() for item in self.means],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ConstantBaselinePredictor":
        return cls(
            state=ModelState.model_validate(value["state"]),
            means=tuple(TurbineMean.from_dict(item) for item in value["means"]),
        )


def fit_constant_baseline(
    observations: Iterable[Observation],
    *,
    state: ModelState,
    turbine_ids: Iterable[str] | None = None,
    cutoff: datetime | None = None,
) -> ConstantBaselinePredictor:
    return ConstantBaselinePredictor(
        state=state,
        means=fit_turbine_means(observations, turbine_ids=turbine_ids, cutoff=cutoff),
    )


# Concise aliases used in reports and by callers that prefer the statistical name.
MeanBaselinePredictor = ConstantBaselinePredictor


@dataclass(frozen=True, slots=True)
class PersistenceBaseline(ConstantBaselinePredictor):
    """Last complete hour, with an explicit one-hour freshness requirement."""

    def predict_base(self, snapshot):
        features = build_features(snapshot)
        latest = {}
        for obs in latest_observations(snapshot.observations, snapshot.origin_time):
            if obs.quality_flag == "complete" and (obs.turbine_id not in latest or
                    obs.event_end > latest[obs.turbine_id].event_end):
                latest[obs.turbine_id] = obs
        rows = []
        for row in features:
            actual = latest.get(row.turbine_id)
            if actual is None or snapshot.origin_time - actual.event_end > timedelta(hours=1):
                raise ValueError("STALE_PERSISTENCE:" + row.turbine_id)
            rows.append(PredictionRow(turbine_id=row.turbine_id, target_start=row.target_start,
                target_end=row.target_end, prediction_norm=actual.power_norm))
        return PredictionBatch(rows=tuple(rows))

    def to_dict(self):
        return {**ConstantBaselinePredictor.to_dict(self), "kind": "persistence"}
fit_mean_baseline = fit_constant_baseline
