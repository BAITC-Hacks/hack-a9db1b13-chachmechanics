"""Lead-group ensemble selected on out-of-sample validation predictions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Iterable, Mapping, Protocol

from ..features import FEATURE_SCHEMA_VERSION, LEAD_GROUPS, lead_group
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


class BasePredictor(Protocol):
    state: ModelState

    def predict(self, snapshot: AsOfSnapshot, bias: BiasState | None = None) -> PredictionBatch: ...


@dataclass(frozen=True, slots=True)
class EnsembleExample:
    """Already out-of-sample predictions used only to select a blend weight."""

    turbine_id: str
    origin_time: datetime
    target_start: datetime
    actual_available_at: datetime
    actual_norm: float
    twin_prediction: float
    ml_prediction: float

    def __post_init__(self) -> None:
        for name in ("origin_time", "target_start", "actual_available_at"):
            object.__setattr__(self, name, utc(getattr(self, name)))
        if not self.turbine_id or self.target_start <= self.origin_time:
            raise ValueError("Invalid ensemble validation key")
        if self.actual_available_at < self.target_start + timedelta(hours=1):
            raise ValueError("Validation actual cannot be available before target end")
        for name in ("actual_norm", "twin_prediction", "ml_prediction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class LeadWeight:
    lead_group: str
    ml_weight: float
    sample_count: int
    validation_mae: float | None

    def __post_init__(self) -> None:
        if self.lead_group not in LEAD_GROUPS:
            raise ValueError("Unknown lead group")
        if not math.isfinite(float(self.ml_weight)) or not 0 <= self.ml_weight <= 1:
            raise ValueError("ml_weight must be in [0, 1]")
        if self.sample_count < 0:
            raise ValueError("sample_count cannot be negative")
        if self.validation_mae is not None and (
            not math.isfinite(float(self.validation_mae)) or self.validation_mae < 0
        ):
            raise ValueError("validation_mae must be finite and non-negative")
        object.__setattr__(self, "ml_weight", float(self.ml_weight))
        if self.validation_mae is not None:
            object.__setattr__(self, "validation_mae", float(self.validation_mae))


@dataclass(frozen=True, slots=True)
class WeightSelection:
    weights: tuple[LeadWeight, ...]
    selected_as_of: datetime
    max_label_available_at: datetime
    sample_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_as_of", utc(self.selected_as_of))
        object.__setattr__(self, "max_label_available_at", utc(self.max_label_available_at))
        if self.max_label_available_at > self.selected_as_of:
            raise ValueError("Weight selection uses a future label")
        if (len(self.weights) != len(LEAD_GROUPS)
                or {item.lead_group for item in self.weights} != set(LEAD_GROUPS)):
            raise ValueError("Weight selection must cover all lead groups")
        if self.sample_count != sum(item.sample_count for item in self.weights):
            raise ValueError("Weight selection sample count mismatch")
        order = {name: index for index, name in enumerate(LEAD_GROUPS)}
        object.__setattr__(
            self, "weights", tuple(sorted(self.weights, key=lambda item: order[item.lead_group]))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "weights": [
                {
                    "lead_group": item.lead_group,
                    "ml_weight": item.ml_weight,
                    "sample_count": item.sample_count,
                    "validation_mae": item.validation_mae,
                }
                for item in self.weights
            ],
            "selected_as_of": self.selected_as_of.isoformat(),
            "max_label_available_at": self.max_label_available_at.isoformat(),
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "WeightSelection":
        return cls(
            weights=tuple(
                LeadWeight(
                    lead_group=str(item["lead_group"]),
                    ml_weight=float(item["ml_weight"]),
                    sample_count=int(item["sample_count"]),
                    validation_mae=(
                        None if item.get("validation_mae") is None
                        else float(item["validation_mae"])
                    ),
                )
                for item in value["weights"]
            ),
            selected_as_of=datetime.fromisoformat(str(value["selected_as_of"])),
            max_label_available_at=datetime.fromisoformat(
                str(value["max_label_available_at"])
            ),
            sample_count=int(value["sample_count"]),
        )


def select_lead_weights(
    examples: Iterable[EnsembleExample],
    *,
    as_of: datetime,
    grid: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    empty_group_weight: float = 0.0,
) -> WeightSelection:
    """Choose frozen ML weights by minimum MAE on labels available at ``as_of``."""

    cutoff = utc(as_of)
    candidates = tuple(sorted({float(value) for value in grid}))
    if not candidates or any(not math.isfinite(v) or not 0 <= v <= 1 for v in candidates):
        raise ValueError("Weight grid must contain finite values in [0, 1]")
    if not math.isfinite(float(empty_group_weight)) or not 0 <= empty_group_weight <= 1:
        raise ValueError("empty_group_weight must be in [0, 1]")
    eligible = sorted(
        (item for item in examples if item.actual_available_at <= cutoff),
        key=lambda item: (item.origin_time, item.target_start, item.turbine_id),
    )
    if not eligible:
        raise ValueError("ENSEMBLE_NO_AVAILABLE_VALIDATION_LABELS")
    grouped: dict[str, list[EnsembleExample]] = {name: [] for name in LEAD_GROUPS}
    for item in eligible:
        hours = (item.target_start - item.origin_time).total_seconds() / 3600
        grouped[lead_group(hours)].append(item)
    selected = []
    for name in LEAD_GROUPS:
        rows = grouped[name]
        if not rows:
            selected.append(LeadWeight(name, float(empty_group_weight), 0, None))
            continue
        scored = []
        for weight in candidates:
            absolute_errors = [
                abs(((1 - weight) * row.twin_prediction + weight * row.ml_prediction)
                    - row.actual_norm)
                for row in rows
            ]
            scored.append((math.fsum(absolute_errors) / len(rows), weight))
        mae, weight = min(scored)  # sorted candidates make ties deterministic.
        selected.append(LeadWeight(name, weight, len(rows), mae))
    return WeightSelection(
        weights=tuple(selected),
        selected_as_of=cutoff,
        max_label_available_at=max(item.actual_available_at for item in eligible),
        sample_count=len(eligible),
    )


@dataclass(frozen=True, slots=True)
class EnsemblePredictor:
    """Blend curve and ML point forecasts; apply bias exactly once afterwards."""

    state: ModelState
    twin: BasePredictor
    ml: BasePredictor
    selection: WeightSelection

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", ModelState.model_validate(self.state.model_dump()))
        if self.state.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("MODEL_FEATURE_SCHEMA_MISMATCH")
        if self.selection.selected_as_of > self.state.training_cutoff:
            raise ValueError("Ensemble weights postdate its training cutoff")
        if any(child.state.activated_at > self.state.activated_at for child in (self.twin, self.ml)):
            raise ValueError("Ensemble activated before a child model")

    @property
    def weights_by_group(self) -> dict[str, float]:
        return {item.lead_group: item.ml_weight for item in self.selection.weights}

    @staticmethod
    def _base(predictor: BasePredictor, snapshot: AsOfSnapshot) -> PredictionBatch:
        method = getattr(predictor, "predict_base", None)
        return method(snapshot) if callable(method) else predictor.predict(snapshot, None)

    def predict_base(self, snapshot: AsOfSnapshot) -> PredictionBatch:
        twin_batch = self._base(self.twin, snapshot)
        ml_batch = self._base(self.ml, snapshot)
        twin_rows = {(r.turbine_id, r.target_start, r.target_end): r for r in twin_batch.rows}
        ml_rows = {(r.turbine_id, r.target_start, r.target_end): r for r in ml_batch.rows}
        if (len(twin_rows) != len(twin_batch.rows) or len(ml_rows) != len(ml_batch.rows)
                or twin_rows.keys() != ml_rows.keys()):
            raise ValueError("ENSEMBLE_CHILD_COVERAGE_MISMATCH")
        weights = self.weights_by_group
        rows = []
        for key in sorted(twin_rows, key=lambda item: (item[0], item[1])):
            twin_row, ml_row = twin_rows[key], ml_rows[key]
            hours = (twin_row.target_start - snapshot.origin_time).total_seconds() / 3600
            weight = weights[lead_group(hours)]
            value = min(1.0, max(0.0,
                (1 - weight) * twin_row.prediction_norm + weight * ml_row.prediction_norm))
            statuses = sorted({s for s in (twin_row.status, ml_row.status) if s != "ok"})
            rows.append(PredictionRow(
                turbine_id=twin_row.turbine_id,
                target_start=twin_row.target_start,
                target_end=twin_row.target_end,
                prediction_norm=value,
                status="ok" if not statuses else "ensemble:" + "+".join(statuses),
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
        if not callable(getattr(self.twin, "to_dict", None)) or not callable(
            getattr(self.ml, "to_dict", None)
        ):
            raise TypeError("Ensemble children must support JSON serialization")
        return {
            "kind": "ensemble",
            "state": self.state.model_dump(mode="json"),
            "twin": self.twin.to_dict(),
            "ml": self.ml.to_dict(),
            "selection": self.selection.to_dict(),
        }


def build_ensemble_predictor(
    twin: BasePredictor,
    ml: BasePredictor,
    selection: WeightSelection,
    *,
    activated_at: datetime,
    artifact_ref: str,
) -> EnsemblePredictor:
    activation = utc(activated_at)
    if not artifact_ref:
        raise ValueError("artifact_ref must not be empty")
    training_cutoff = max(
        twin.state.training_cutoff,
        ml.state.training_cutoff,
        selection.selected_as_of,
    )
    max_label = max(
        twin.state.max_label_available_at,
        ml.state.max_label_available_at,
        selection.max_label_available_at,
    )
    if activation < training_cutoff:
        raise ValueError("activated_at cannot precede ensemble selection")
    provenance = (
        "trained"
        if twin.state.provenance == ml.state.provenance == "trained"
        else "synthetic"
    )
    weights_payload = [
        (item.lead_group, item.ml_weight, item.sample_count, item.validation_mae)
        for item in selection.weights
    ]
    content = {
        "algorithm": "lead_group_fixed_blend_v1",
        "twin_model_id": twin.state.model_id,
        "ml_model_id": ml.state.model_id,
        "weights": weights_payload,
        "selection_as_of": selection.selected_as_of.isoformat(),
        "training_cutoff": training_cutoff.isoformat(),
        "max_label_available_at": max_label.isoformat(),
        "activated_at": activation.isoformat(),
        "artifact_ref": artifact_ref,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "provenance": provenance,
    }
    state = ModelState(
        model_id="twinturbo-ensemble-" + digest(content)[:16],
        training_cutoff=training_cutoff,
        max_label_available_at=max_label,
        activated_at=activation,
        artifact_ref=artifact_ref,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        provenance=provenance,
    )
    return EnsemblePredictor(state=state, twin=twin, ml=ml, selection=selection)


# Short report-friendly alias.
select_weights = select_lead_weights
