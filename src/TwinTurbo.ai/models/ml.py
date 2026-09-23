"""Deterministic per-turbine ridge model for the optional P1 forecast.

The project lock intentionally has no scikit-learn dependency.  Ridge
regression gives us a real, reproducible ML candidate with the existing NumPy
runtime and, unlike a hidden fallback, can be evaluated honestly against the
power-curve baseline in walk-forward validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Iterable, Literal, Mapping

import numpy as np

from ..features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    TrainingExample,
    build_features,
    feature_vector,
    select_training_examples,
)
from ..schemas import (
    AsOfSnapshot,
    BiasState,
    ModelState,
    PredictionBatch,
    PredictionRow,
    digest,
    utc,
)
from .baseline import postprocess_predictions


@dataclass(frozen=True, slots=True)
class RidgeTurbineModel:
    """Standardised linear model fitted for exactly one turbine."""

    turbine_id: str
    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    sample_count: int
    alpha: float

    def __post_init__(self) -> None:
        size = len(self.feature_names)
        if not self.turbine_id:
            raise ValueError("turbine_id must not be empty")
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("ML feature schema does not match the runtime")
        if not size or any(
            len(values) != size
            for values in (self.feature_means, self.feature_scales, self.coefficients)
        ):
            raise ValueError("Ridge coefficient dimensions do not match features")
        numeric = (*self.feature_means, *self.feature_scales, *self.coefficients,
                   self.intercept, self.alpha)
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("Ridge model contains a non-finite value")
        if any(float(value) <= 0 for value in self.feature_scales):
            raise ValueError("Feature scales must be positive")
        if self.sample_count <= 0 or self.alpha < 0:
            raise ValueError("Invalid ridge training metadata")
        for name in ("feature_means", "feature_scales", "coefficients"):
            object.__setattr__(self, name, tuple(float(v) for v in getattr(self, name)))
        object.__setattr__(self, "intercept", float(self.intercept))
        object.__setattr__(self, "alpha", float(self.alpha))

    def predict_value(self, row) -> tuple[float, str]:
        if row.turbine_id != self.turbine_id:
            raise ValueError("ML_TURBINE_MISMATCH")
        vector = np.asarray(feature_vector(row), dtype=float)
        means = np.asarray(self.feature_means, dtype=float)
        scales = np.asarray(self.feature_scales, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
        raw = self.intercept + float(((vector - means) / scales) @ coefficients)
        clipped = min(1.0, max(0.0, raw))
        if raw < 0:
            status = "clipped_low"
        elif raw > 1:
            status = "clipped_high"
        else:
            status = "ok"
        return clipped, status

    def to_dict(self) -> dict[str, object]:
        return {
            "turbine_id": self.turbine_id,
            "feature_names": list(self.feature_names),
            "feature_means": list(self.feature_means),
            "feature_scales": list(self.feature_scales),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "sample_count": self.sample_count,
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RidgeTurbineModel":
        return cls(
            turbine_id=str(value["turbine_id"]),
            feature_names=tuple(str(v) for v in value["feature_names"]),
            feature_means=tuple(float(v) for v in value["feature_means"]),
            feature_scales=tuple(float(v) for v in value["feature_scales"]),
            coefficients=tuple(float(v) for v in value["coefficients"]),
            intercept=float(value["intercept"]),
            sample_count=int(value["sample_count"]),
            alpha=float(value["alpha"]),
        )


@dataclass(frozen=True, slots=True)
class RidgePredictor:
    """A versioned Predictor containing one ridge model per turbine."""

    state: ModelState
    models: tuple[RidgeTurbineModel, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", ModelState.model_validate(self.state.model_dump()))
        if self.state.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("MODEL_FEATURE_SCHEMA_MISMATCH")
        if not self.models:
            raise ValueError("RidgePredictor requires at least one turbine model")
        ids = [model.turbine_id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate turbine ridge model")
        object.__setattr__(self, "models", tuple(sorted(self.models, key=lambda m: m.turbine_id)))

    @property
    def models_by_turbine(self) -> dict[str, RidgeTurbineModel]:
        return {model.turbine_id: model for model in self.models}

    def predict_base(self, snapshot: AsOfSnapshot) -> PredictionBatch:
        features = build_features(snapshot)
        models = self.models_by_turbine
        missing = {row.turbine_id for row in features} - models.keys()
        if missing:
            raise ValueError("ML_UNKNOWN_TURBINE: " + ",".join(sorted(missing)))
        rows = []
        for row in features:
            value, status = models[row.turbine_id].predict_value(row)
            rows.append(PredictionRow(
                turbine_id=row.turbine_id,
                target_start=row.target_start,
                target_end=row.target_end,
                prediction_norm=value,
                status=status,
            ))
        return PredictionBatch(rows=tuple(rows))

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
            "kind": "ridge",
            "state": self.state.model_dump(mode="json"),
            "models": [model.to_dict() for model in self.models],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RidgePredictor":
        return cls(
            state=ModelState.model_validate(value["state"]),
            models=tuple(RidgeTurbineModel.from_dict(item) for item in value["models"]),
        )


def _fit_one(
    turbine_id: str,
    examples: list[TrainingExample],
    *,
    alpha: float,
) -> RidgeTurbineModel:
    matrix = np.asarray([feature_vector(item) for item in examples], dtype=float)
    target = np.asarray([item.actual_norm for item in examples], dtype=float)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    design = (matrix - means) / scales
    target_mean = float(target.mean())
    centered_target = target - target_mean
    penalty = float(alpha) * np.eye(design.shape[1], dtype=float)
    lhs = design.T @ design + penalty
    rhs = design.T @ centered_target
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return RidgeTurbineModel(
        turbine_id=turbine_id,
        feature_names=FEATURE_NAMES,
        feature_means=tuple(means.tolist()),
        feature_scales=tuple(scales.tolist()),
        coefficients=tuple(coefficients.tolist()),
        intercept=target_mean,
        sample_count=len(examples),
        alpha=alpha,
    )


def fit_ridge_predictor(
    examples: Iterable[TrainingExample],
    *,
    training_cutoff: datetime,
    activated_at: datetime,
    artifact_ref: str,
    turbine_ids: Iterable[str] | None = None,
    alpha: float = 1.0,
    min_samples_per_turbine: int = 12,
    provenance: Literal["trained", "synthetic"] = "trained",
) -> RidgePredictor:
    """Fit a leakage-safe ridge candidate from prepared historical examples.

    Only labels whose ``actual_available_at`` is not later than
    ``training_cutoff`` are considered.  Callers remain responsible for
    constructing each historical weather feature from the run available at its
    own ``origin_time``.
    """

    cutoff = utc(training_cutoff)
    activation = utc(activated_at)
    if activation < cutoff:
        raise ValueError("activated_at cannot precede training_cutoff")
    if not artifact_ref:
        raise ValueError("artifact_ref must not be empty")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if min_samples_per_turbine <= 0:
        raise ValueError("min_samples_per_turbine must be positive")

    eligible = select_training_examples(examples, cutoff)
    grouped: dict[str, list[TrainingExample]] = {}
    for example in eligible:
        grouped.setdefault(example.turbine_id, []).append(example)
    requested = set(turbine_ids) if turbine_ids is not None else set(grouped)
    if not requested:
        raise ValueError("ML_NO_TURBINES")
    missing = [
        turbine_id for turbine_id in sorted(requested)
        if len(grouped.get(turbine_id, ())) < min_samples_per_turbine
    ]
    if missing:
        raise ValueError("ML_INSUFFICIENT_TRAINING_DATA: " + ",".join(missing))
    selected = [item for item in eligible if item.turbine_id in requested]
    models = tuple(
        _fit_one(turbine_id, grouped[turbine_id], alpha=alpha)
        for turbine_id in sorted(requested)
    )
    max_label_available_at = max(item.actual_available_at for item in selected)
    content = {
        "algorithm": "per_turbine_standardized_ridge_v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "training_cutoff": cutoff.isoformat(),
        "max_label_available_at": max_label_available_at.isoformat(),
        "activated_at": activation.isoformat(),
        "artifact_ref": artifact_ref,
        "provenance": provenance,
        "alpha": alpha,
        "models": [model.to_dict() for model in models],
    }
    state = ModelState(
        model_id="twinturbo-ridge-" + digest(content)[:16],
        training_cutoff=cutoff,
        max_label_available_at=max_label_available_at,
        activated_at=activation,
        artifact_ref=artifact_ref,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        provenance=provenance,
    )
    return RidgePredictor(state=state, models=models)


# Friendly aliases for callers and reports.
MLPredictor = RidgePredictor
fit_ml_predictor = fit_ridge_predictor
