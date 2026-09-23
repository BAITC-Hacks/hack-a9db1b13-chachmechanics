"""Optional histogram gradient boosting adapted from feat/forecast-models.

Ridge remains available separately. Both use the same validated snapshot features.
"""
from dataclasses import asdict
import base64
import pickle

import numpy as np

from ..features import FEATURE_SCHEMA_VERSION, build_features, feature_vector, supervised_examples
from ..schemas import ModelState, PredictionBatch, PredictionRow, digest, utc
from .baseline import postprocess_predictions


class GradientBoostingPredictor:
    def __init__(self, state, estimators, counts, parameters):
        self.state = ModelState.model_validate(state.model_dump())
        self.estimators, self.counts, self.parameters = estimators, counts, parameters

    @classmethod
    def fit(cls, snapshot, historical_snapshots, *, activated_at=None,
            min_samples=100, max_iter=100, max_leaf_nodes=15, min_samples_leaf=20,
            allow_synthetic=False):
        from sklearn.ensemble import HistGradientBoostingRegressor
        history = tuple(historical_snapshots)
        examples = supervised_examples(history, snapshot.observations,
            training_cutoff=snapshot.origin_time, allow_synthetic=allow_synthetic)
        turbines = {v.turbine_id for v in build_features(snapshot)}
        if min_samples < 2 or not turbines:
            raise ValueError("INVALID_ML_TRAINING_REQUEST")
        parameters = dict(loss="absolute_error", max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
                          min_samples_leaf=min_samples_leaf, early_stopping=False, random_state=42)
        estimators, counts = {}, {}
        for turbine in sorted(turbines):
            rows = [e for e in examples if e.turbine_id == turbine]
            if len(rows) < min_samples:
                raise ValueError("INSUFFICIENT_ML_HISTORY:" + turbine)
            estimator = HistGradientBoostingRegressor(**parameters)
            estimator.fit(np.asarray([feature_vector(e) for e in rows]), [e.actual_norm for e in rows])
            estimators[turbine], counts[turbine] = estimator, len(rows)
        used = [e for e in examples if e.turbine_id in turbines]
        activation = utc(activated_at or snapshot.origin_time)
        identity = {"kind": "hist_gradient_boosting", "schema": FEATURE_SCHEMA_VERSION,
                    "parameters": parameters, "cutoff": snapshot.origin_time.isoformat(),
                    "activation": activation.isoformat(),
                    "examples": [{k: v.isoformat() if hasattr(v, "isoformat") else v
                                  for k, v in asdict(e).items()} for e in used]}
        model_id = "boosting-" + digest(identity)[:24]
        synthetic = snapshot.weather_run_metadata.provenance == "synthetic" or any(
            s.weather_run_metadata.provenance == "synthetic" for s in history)
        state = ModelState(model_id=model_id, training_cutoff=snapshot.origin_time,
            max_label_available_at=max(e.actual_available_at for e in used), activated_at=activation,
            artifact_ref="model://" + model_id, feature_schema_version=FEATURE_SCHEMA_VERSION,
            provenance="synthetic" if synthetic else "trained")
        return cls(state, estimators, counts, parameters)

    def predict_base(self, snapshot):
        if self.state.activated_at > snapshot.origin_time:
            raise ValueError("FUTURE_MODEL")
        features = build_features(snapshot)
        rows = []
        for turbine in sorted({f.turbine_id for f in features}):
            if turbine not in self.estimators:
                raise ValueError("UNTRAINED_TURBINE:" + turbine)
            selected = [f for f in features if f.turbine_id == turbine]
            predictions = self.estimators[turbine].predict(np.asarray([feature_vector(f) for f in selected]))
            for feature, prediction in zip(selected, predictions):
                rows.append(PredictionRow(turbine_id=turbine, target_start=feature.target_start,
                    target_end=feature.target_end, prediction_norm=float(np.clip(prediction, 0, 1))))
        return PredictionBatch(rows=tuple(rows))

    def predict(self, snapshot, bias=None):
        return postprocess_predictions(self.predict_base(snapshot), bias,
            model_id=self.state.model_id, origin_time=snapshot.origin_time)

    def to_dict(self):
        import sklearn
        return {"kind": "hist_gradient_boosting", "state": self.state.model_dump(mode="json"),
                "estimators": base64.b64encode(pickle.dumps(self.estimators, protocol=5)).decode(),
                "sklearn_version": sklearn.__version__, "counts": self.counts, "parameters": self.parameters}

    @classmethod
    def from_dict(cls, payload, *, trusted=False):
        if not trusted:
            raise ValueError("TRUSTED_ML_ARTIFACT_REQUIRED")
        import sklearn
        if payload["sklearn_version"] != sklearn.__version__:
            raise ValueError("SKLEARN_VERSION_MISMATCH")
        return cls(ModelState.model_validate(payload["state"]),
                   pickle.loads(base64.b64decode(payload["estimators"], validate=True)),
                   payload["counts"], payload["parameters"])
