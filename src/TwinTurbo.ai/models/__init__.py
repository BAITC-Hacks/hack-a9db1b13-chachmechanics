"""Shared numerical predictor boundary. All predict methods are free of I/O."""
from datetime import datetime
import numpy as np

from ..features import FEATURE_SCHEMA, features_from_snapshot, training_observations
from ..schemas import BiasState, ModelState, PredictionBatch, PredictionRow, digest, utc


def make_state(kind, snapshot, parameters, *, activated_at=None):
    observations = training_observations(snapshot)
    identity = {"kind": kind, "cutoff": snapshot.origin_time.isoformat(), "parameters": parameters,
                "data": [o.model_dump(mode="json") for o in observations], "schema": FEATURE_SCHEMA,
                "provenance": snapshot.weather_run_metadata.provenance,
                "activated_at": (utc(activated_at) if activated_at is not None else snapshot.origin_time).isoformat()}
    model_id = kind + "-" + digest(identity)[:24]
    return ModelState(model_id=model_id, training_cutoff=snapshot.origin_time,
        max_label_available_at=max(o.available_at for o in observations),
        activated_at=utc(activated_at) if activated_at is not None else snapshot.origin_time,
        artifact_ref="model://" + model_id, feature_schema_version=FEATURE_SCHEMA,
        provenance="synthetic" if snapshot.weather_run_metadata.provenance == "synthetic" else "trained")


class NumericalPredictor:
    calendar_timezone = "UTC"

    def predict_base(self, snapshot):
        state = ModelState.model_validate(self.state.model_dump())
        if state.activated_at > snapshot.origin_time:
            raise ValueError("FUTURE_MODEL")
        features = features_from_snapshot(snapshot, calendar_timezone=self.calendar_timezone)
        values = self._predict_values(features)
        if len(values) != len(features):
            raise ValueError("MODEL_OUTPUT_COVERAGE")
        return PredictionBatch(rows=tuple(PredictionRow(turbine_id=f.turbine_id,
            target_start=f.target_start, target_end=f.target_end,
            prediction_norm=float(np.clip(p, 0, 1)), status=status)
            for f, (p, status) in zip(features, values)))

    def predict(self, snapshot, bias: BiasState | None = None):
        from .bias import apply_bias
        from .intervals import apply_intervals
        base = self.predict_base(snapshot)
        corrected = apply_bias(base, bias, model_id=self.state.model_id, origin_time=snapshot.origin_time)
        return apply_intervals(corrected, bias, origin_time=snapshot.origin_time)
