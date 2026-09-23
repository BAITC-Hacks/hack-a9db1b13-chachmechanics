"""Build deterministic per-turbine predictors from an allowed as-of snapshot."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

from ..features import FEATURE_SCHEMA_VERSION, build_features, eligible_observations
from ..models.baseline import ConstantBaselinePredictor, fit_turbine_means
from ..models.power_curve import PowerCurvePredictor, fit_power_curves
from ..schemas import AsOfSnapshot, ModelState, Observation, digest, utc


ModelKind = Literal["power_curve", "baseline"]
ModelProvenance = Literal["trained", "synthetic"]


@dataclass(frozen=True, slots=True)
class TwinBuilder:
    """Fit simple reproducible models without reading external data.

    ``cutoff`` is a label-availability cutoff, not merely an event-time cutoff.
    Observations arriving after it are never included in the fitted artifact.
    """

    bin_width: float = 1.0
    min_samples_per_bin: int = 1
    provenance: ModelProvenance = "trained"

    def _training_data(
        self, snapshot: AsOfSnapshot, cutoff: datetime | None
    ) -> tuple[datetime, tuple[str, ...], tuple[Observation, ...]]:
        snapshot = AsOfSnapshot.model_validate(snapshot.model_dump())
        cutoff = utc(cutoff or snapshot.origin_time)
        feature_rows = build_features(snapshot)
        turbine_ids = tuple(sorted({row.turbine_id for row in feature_rows}))
        observations = eligible_observations(snapshot, cutoff)
        present = {observation.turbine_id for observation in observations}
        missing = set(turbine_ids) - present
        if missing:
            raise ValueError("MODEL_NO_TRAINING_DATA: " + ",".join(sorted(missing)))
        return cutoff, turbine_ids, observations

    def _state(
        self,
        *,
        kind: ModelKind,
        cutoff: datetime,
        activated_at: datetime,
        observations: tuple[Observation, ...],
        learned_parameters: object,
        model_id: str | None,
        artifact_ref: str | None,
        provenance: ModelProvenance,
    ) -> ModelState:
        activated_at = utc(activated_at)
        if activated_at < cutoff:
            raise ValueError("Model activation cannot precede its training cutoff")
        max_available = max(observation.available_at for observation in observations)
        content = {
            "algorithm": kind,
            "algorithm_version": "1",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "training_cutoff": cutoff.isoformat(),
            "max_label_available_at": max_available.isoformat(),
            "activated_at": activated_at.isoformat(),
            "provenance": provenance,
            "training_rows": [observation.model_dump(mode="json") for observation in observations],
            "learned_parameters": learned_parameters,
        }
        content_hash = digest(content)
        reference = artifact_ref or f"inline-sha256:{content_hash}"
        identity_hash = digest({"content_hash": content_hash, "artifact_ref": reference})
        identity = model_id or f"{kind}-v1-{identity_hash[:20]}"
        return ModelState(
            model_id=identity,
            training_cutoff=cutoff,
            max_label_available_at=max_available,
            activated_at=activated_at,
            artifact_ref=reference,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            provenance=provenance,
        )

    def build_power_curve(
        self,
        snapshot: AsOfSnapshot,
        *,
        cutoff: datetime | None = None,
        activated_at: datetime | None = None,
        model_id: str | None = None,
        artifact_ref: str | None = None,
        provenance: ModelProvenance | None = None,
    ) -> PowerCurvePredictor:
        cutoff, turbine_ids, observations = self._training_data(snapshot, cutoff)
        activation = utc(activated_at or snapshot.origin_time)
        curves = fit_power_curves(
            observations,
            turbine_ids=turbine_ids,
            bin_width=self.bin_width,
            min_samples_per_bin=self.min_samples_per_bin,
            cutoff=cutoff,
        )
        selected_provenance = provenance or self.provenance
        state = self._state(
            kind="power_curve",
            cutoff=cutoff,
            activated_at=activation,
            observations=observations,
            learned_parameters=[asdict(curve) for curve in curves],
            model_id=model_id,
            artifact_ref=artifact_ref,
            provenance=selected_provenance,
        )
        return PowerCurvePredictor(state=state, curves=curves)

    def build_baseline(
        self,
        snapshot: AsOfSnapshot,
        *,
        cutoff: datetime | None = None,
        activated_at: datetime | None = None,
        model_id: str | None = None,
        artifact_ref: str | None = None,
        provenance: ModelProvenance | None = None,
    ) -> ConstantBaselinePredictor:
        cutoff, turbine_ids, observations = self._training_data(snapshot, cutoff)
        activation = utc(activated_at or snapshot.origin_time)
        means = fit_turbine_means(observations, turbine_ids=turbine_ids, cutoff=cutoff)
        selected_provenance = provenance or self.provenance
        state = self._state(
            kind="baseline",
            cutoff=cutoff,
            activated_at=activation,
            observations=observations,
            learned_parameters=[asdict(value) for value in means],
            model_id=model_id,
            artifact_ref=artifact_ref,
            provenance=selected_provenance,
        )
        return ConstantBaselinePredictor(state=state, means=means)

    def build(
        self,
        snapshot: AsOfSnapshot,
        *,
        kind: ModelKind = "power_curve",
        cutoff: datetime | None = None,
        activated_at: datetime | None = None,
        model_id: str | None = None,
        artifact_ref: str | None = None,
        provenance: ModelProvenance | None = None,
    ) -> PowerCurvePredictor | ConstantBaselinePredictor:
        options = {
            "cutoff": cutoff,
            "activated_at": activated_at,
            "model_id": model_id,
            "artifact_ref": artifact_ref,
            "provenance": provenance,
        }
        if kind == "power_curve":
            return self.build_power_curve(snapshot, **options)
        if kind == "baseline":
            return self.build_baseline(snapshot, **options)
        raise ValueError(f"Unknown model kind: {kind}")


def build_twin(
    snapshot: AsOfSnapshot, **kwargs
) -> PowerCurvePredictor | ConstantBaselinePredictor:
    """Convenience wrapper around :class:`TwinBuilder` for the default curve."""

    return TwinBuilder().build(snapshot, **kwargs)
