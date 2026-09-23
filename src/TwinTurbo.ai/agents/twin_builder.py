"""Training orchestration only; the numerical curve lives in models/power_curve.py."""
from ..models.power_curve import PowerCurvePredictor


class TwinBuilder:
    def build(self, snapshot, **parameters):
        return PowerCurvePredictor.fit(snapshot, **parameters)
