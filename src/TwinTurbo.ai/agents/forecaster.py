"""Thin predictor adapter; capture uncorrected values before clipping for Critic."""
from dataclasses import dataclass
from ..schemas import PredictionBatch


@dataclass(frozen=True)
class ForecastTrace:
    base_predictions: PredictionBatch
    predictions: PredictionBatch
    model_id: str
    bias_id: str | None


class Forecaster:
    def __init__(self, predictor):
        self.predictor = predictor

    @property
    def state(self):
        return self.predictor.state

    def predict(self, snapshot, bias=None):
        return self.predictor.predict(snapshot, bias)

    def predict_with_trace(self, snapshot, bias=None):
        return ForecastTrace(self.predictor.predict_base(snapshot), self.predict(snapshot, bias),
                             self.state.model_id, bias.bias_id if bias else None)
