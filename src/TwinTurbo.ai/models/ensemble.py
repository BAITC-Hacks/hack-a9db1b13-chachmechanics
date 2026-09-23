"""Frozen lead-group weights selected on a later, already-matured validation period."""
import numpy as np
from ..features import LEAD_GROUPS, latest_observations, lead_group
from ..schemas import ModelState, digest, utc
from . import NumericalPredictor


class EnsemblePredictor(NumericalPredictor):
    kind = "ensemble"

    def __init__(self, state, curve, ml, weights, validation_counts):
        self.state, self.curve, self.ml = state, curve, ml
        self.weights, self.validation_counts = weights, validation_counts
        self.calendar_timezone = ml.calendar_timezone

    @classmethod
    def fit(cls, curve, ml, snapshots, observations, *, as_of, activated_at=None,
            grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
        as_of = utc(as_of)
        grid = tuple(sorted(set(grid)))
        if not grid or any(not np.isfinite(w) or not 0 <= w <= 1 for w in grid):
            raise ValueError("INVALID_ENSEMBLE_GRID")
        labels = {(o.turbine_id, o.event_start, o.event_end): o
                  for o in latest_observations(observations, as_of) if o.quality_flag == "complete"}
        groups, seen, used = {g: [] for g in LEAD_GROUPS}, set(), []
        for snapshot in sorted(snapshots, key=lambda s: s.origin_time):
            if snapshot.origin_time > as_of:
                continue
            a, b = curve.predict_base(snapshot), ml.predict_base(snapshot)
            bm = {(r.turbine_id, r.target_start, r.target_end): r for r in b.rows}
            for row in a.rows:
                key = row.turbine_id, row.target_start, row.target_end
                origin_key = (snapshot.origin_time, *key)
                if origin_key in seen:
                    raise ValueError("DUPLICATE_VALIDATION_ORIGIN")
                seen.add(origin_key)
                if key not in labels:
                    continue
                other, actual = bm[key], labels[key]
                group = lead_group((row.target_start - snapshot.origin_time).total_seconds() / 3600)
                groups[group].append((row.prediction_norm, other.prediction_norm, actual.power_norm))
                used.append(actual)
        if not used:
            raise ValueError("NO_MATURE_VALIDATION_TARGETS")
        weights = {}
        for group, rows in groups.items():
            # No validation for a group: keep the P0 curve, explicitly count zero.
            weights[group] = min(grid, key=lambda w: np.mean([
                abs(y - ((1-w)*a+w*b)) for a, b, y in rows])) if rows else 0.0
        identity = {"curve": curve.state.model_id, "ml": ml.state.model_id,
                    "as_of": as_of.isoformat(), "weights": weights, "validation": groups,
                    "activated_at": (utc(activated_at) if activated_at else as_of).isoformat()}
        model_id = "ensemble-" + digest(identity)[:24]
        state = ModelState(model_id=model_id, training_cutoff=as_of,
            max_label_available_at=max(curve.state.max_label_available_at, ml.state.max_label_available_at,
                                       max(o.available_at for o in used)),
            activated_at=utc(activated_at) if activated_at else as_of,
            artifact_ref="model://" + model_id, feature_schema_version=ml.state.feature_schema_version,
            provenance="synthetic" if "synthetic" in (curve.state.provenance, ml.state.provenance) else "trained")
        return cls(state, curve, ml, weights, {g: len(rows) for g, rows in groups.items()})

    def _predict_values(self, features):
        a, b = self.curve._predict_values(features), self.ml._predict_values(features)
        output = []
        for feature, (pa, status), (pb, _) in zip(features, a, b):
            w = self.weights[lead_group(feature.lead_hours)]
            output.append(((1-w)*pa+w*pb, status))
        return output
