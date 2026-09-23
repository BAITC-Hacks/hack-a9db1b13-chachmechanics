"""Independent empirical median curves; no invented cut-out or global monotonicity."""
from dataclasses import asdict, dataclass
import math
import numpy as np

from ..features import training_observations
from . import NumericalPredictor, make_state


@dataclass(frozen=True)
class TurbineCurve:
    wind: tuple[float, ...]
    power: tuple[float, ...]
    counts: tuple[int, ...]
    train_count: int
    min_wind: float
    max_wind: float

    def predict(self, wind_ms):
        outside = wind_ms < self.min_wind or wind_ms > self.max_wind
        # Constant endpoint extension is explicit; high-wind shutdown is unknown.
        value = float(np.interp(wind_ms, self.wind, self.power))
        return value, "OUT_OF_DOMAIN" if outside else "ok"


class PowerCurvePredictor(NumericalPredictor):
    kind = "power_curve"

    def __init__(self, state, curves, *, bin_width=0.5, min_bin_count=6):
        self.state, self.curves = state, dict(curves)
        self.bin_width, self.min_bin_count = bin_width, min_bin_count

    @classmethod
    def fit(cls, snapshot, *, bin_width=0.5, min_bin_count=6, activated_at=None):
        if not math.isfinite(bin_width) or bin_width <= 0 or min_bin_count < 1:
            raise ValueError("INVALID_CURVE_PARAMETERS")
        observations = training_observations(snapshot)
        curves = {}
        for turbine in sorted({o.turbine_id for o in observations}):
            data = [o for o in observations if o.turbine_id == turbine]
            bins = {}
            for o in data:
                bins.setdefault(math.floor(o.wind_ms / bin_width), []).append(o)
            accepted = [rows for _, rows in sorted(bins.items()) if len(rows) >= min_bin_count]
            if len(accepted) < 2:
                raise ValueError("INSUFFICIENT_CURVE_BINS:" + turbine)
            curves[turbine] = TurbineCurve(
                tuple(float(np.median([o.wind_ms for o in rows])) for rows in accepted),
                tuple(float(np.median([o.power_norm for o in rows])) for rows in accepted),
                tuple(len(rows) for rows in accepted), len(data),
                min(o.wind_ms for rows in accepted for o in rows),
                max(o.wind_ms for rows in accepted for o in rows))
        params = {"bin_width": bin_width, "min_bin_count": min_bin_count,
                  "curves": {t: asdict(c) for t, c in curves.items()}, "extension": "constant_endpoint"}
        state = make_state(cls.kind, snapshot, params, activated_at=activated_at)
        return cls(state, curves, bin_width=bin_width, min_bin_count=min_bin_count)

    def _predict_values(self, features):
        if {f.turbine_id for f in features} - self.curves.keys():
            raise ValueError("UNTRAINED_TURBINE")
        return [self.curves[f.turbine_id].predict(f.values[0]) for f in features]
