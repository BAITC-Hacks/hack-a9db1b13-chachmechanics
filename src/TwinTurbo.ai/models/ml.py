"""Optional CPU gradient boosting, trained only on historical forecast snapshots."""
from dataclasses import asdict
import numpy as np
from ..features import FEATURE_SCHEMA, supervised_examples
from ..schemas import ModelState, digest, utc
from . import NumericalPredictor


class MLPredictor(NumericalPredictor):
    kind = "hist_gradient_boosting"

    def __init__(self, state, estimators, counts, *, calendar_timezone="UTC", parameters=None):
        self.state, self.estimators, self.counts = state, estimators, counts
        self.calendar_timezone = calendar_timezone
        self.parameters = parameters or {}

    @classmethod
    def fit(cls, snapshot, historical_snapshots, *, activated_at=None, calendar_timezone="UTC",
            min_samples=100, max_iter=100, max_leaf_nodes=15, min_samples_leaf=20,
            allow_synthetic=False):
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except ImportError as exc:
            raise RuntimeError("ML_DEPENDENCY_REQUIRED: install scikit-learn; P0 works without it") from exc
        examples = supervised_examples(historical_snapshots, snapshot.observations,
            training_cutoff=snapshot.origin_time, calendar_timezone=calendar_timezone,
            allow_synthetic=allow_synthetic)
        turbines = {v.turbine_id for v in snapshot.weather_values}
        if min_samples < 2 or not turbines:
            raise ValueError("INVALID_ML_TRAINING_REQUEST")
        parameters = dict(loss="absolute_error", max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
                          min_samples_leaf=min_samples_leaf, early_stopping=False, random_state=42)
        estimators, counts = {}, {}
        for turbine in sorted(turbines):
            rows = [e for e in examples if e.feature.turbine_id == turbine]
            if len(rows) < min_samples:
                raise ValueError("INSUFFICIENT_ML_HISTORY:" + turbine)
            estimator = HistGradientBoostingRegressor(**parameters)
            estimator.fit(np.asarray([e.feature.values for e in rows]), [e.power_norm for e in rows])
            estimators[turbine], counts[turbine] = estimator, len(rows)
        used = [e for e in examples if e.feature.turbine_id in turbines]
        identity = {"kind": cls.kind, "schema": FEATURE_SCHEMA, "parameters": parameters,
                    "calendar_timezone": calendar_timezone, "cutoff": snapshot.origin_time.isoformat(),
                    "activated_at": (utc(activated_at) if activated_at else snapshot.origin_time).isoformat(),
                    "examples": [{"feature": {**asdict(e.feature),
                        "origin_time": e.feature.origin_time.isoformat(),
                        "target_start": e.feature.target_start.isoformat(),
                        "target_end": e.feature.target_end.isoformat()}, "power": e.power_norm,
                        "available_at": e.label_available_at.isoformat(), "revision": e.label_revision} for e in used]}
        model_id = "ml-" + digest(identity)[:24]
        state = ModelState(model_id=model_id, training_cutoff=snapshot.origin_time,
            max_label_available_at=max(e.label_available_at for e in used),
            activated_at=utc(activated_at) if activated_at else snapshot.origin_time,
            artifact_ref="model://" + model_id, feature_schema_version=FEATURE_SCHEMA,
            provenance="synthetic" if any(e.feature.provenance == "synthetic" for e in used) else "trained")
        return cls(state, estimators, counts, calendar_timezone=calendar_timezone, parameters=parameters)

    def _predict_values(self, features):
        if {f.turbine_id for f in features} - self.estimators.keys():
            raise ValueError("UNTRAINED_TURBINE")
        result = [None] * len(features)
        for turbine, estimator in self.estimators.items():
            indices = [i for i, f in enumerate(features) if f.turbine_id == turbine]
            if indices:
                values = estimator.predict(np.asarray([features[i].values for i in indices]))
                for i, value in zip(indices, values):
                    result[i] = float(np.clip(value, 0, 1)), "ok"
        return result
