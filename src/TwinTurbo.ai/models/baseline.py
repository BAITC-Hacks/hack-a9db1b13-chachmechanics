"""Training-only turbine means and an explicitly freshness-gated persistence baseline."""
from datetime import timedelta
import numpy as np
from ..features import latest_observations, training_observations
from ..schemas import PredictionBatch, PredictionRow
from . import NumericalPredictor, make_state


class ConstantBaseline(NumericalPredictor):
    kind = "constant"

    def __init__(self, state, means, counts):
        self.state, self.means, self.counts = state, dict(means), dict(counts)

    @classmethod
    def fit(cls, snapshot, *, activated_at=None):
        observations = training_observations(snapshot)
        groups = {t: [o.power_norm for o in observations if o.turbine_id == t]
                  for t in sorted({o.turbine_id for o in observations})}
        means = {t: float(np.mean(values)) for t, values in groups.items()}
        counts = {t: len(values) for t, values in groups.items()}
        return cls(make_state(cls.kind, snapshot, {"means": means, "counts": counts},
                              activated_at=activated_at), means, counts)

    def _predict_values(self, features):
        if {f.turbine_id for f in features} - self.means.keys():
            raise ValueError("UNTRAINED_TURBINE")
        return [(self.means[f.turbine_id], "ok") for f in features]


class PersistenceBaseline(ConstantBaseline):
    kind = "persistence"

    def predict_base(self, snapshot):
        # Reuse boundary validation, but do not use a stale training mean as fallback.
        batch = super().predict_base(snapshot)
        latest = {}
        for o in latest_observations(snapshot.observations, snapshot.origin_time):
            if o.quality_flag == "complete" and (o.turbine_id not in latest or
                    o.event_end > latest[o.turbine_id].event_end):
                latest[o.turbine_id] = o
        rows = []
        for row in batch.rows:
            o = latest.get(row.turbine_id)
            if o is None or snapshot.origin_time - o.event_end > timedelta(hours=1):
                raise ValueError("STALE_PERSISTENCE:" + row.turbine_id)
            rows.append(PredictionRow(**{**row.model_dump(), "prediction_norm": o.power_norm}))
        return PredictionBatch(rows=tuple(rows))
