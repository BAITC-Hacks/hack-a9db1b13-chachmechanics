"""Thin orchestration boundary around a numerical predictor.

The agent deliberately contains no forecasting mathematics: model code lives in
``windoracle.models`` and receives an already authorised :class:`AsOfSnapshot`.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..schemas import AsOfSnapshot, BiasState, ModelState, PredictionBatch, Predictor


@dataclass(frozen=True)
class Forecaster:
    """Delegate a forecast to an immutable, versioned predictor.

    Keeping this adapter small makes the trust boundary explicit.  It neither
    selects weather runs nor reads observations outside the supplied snapshot.
    """

    predictor: Predictor

    @property
    def state(self) -> ModelState:
        return self.predictor.state

    def predict(
        self, snapshot: AsOfSnapshot, bias: BiasState | None = None
    ) -> PredictionBatch:
        return self.predictor.predict(snapshot, bias)

    def run(
        self, snapshot: AsOfSnapshot, bias: BiasState | None = None
    ) -> PredictionBatch:
        """Agent-style alias used by orchestration code."""

        return self.predict(snapshot, bias)
