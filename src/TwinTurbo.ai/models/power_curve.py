"""Robust, per-turbine binned-median power curves."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
import math
from statistics import median
from typing import Iterable, Mapping

from ..features import build_features
from ..schemas import (
    AsOfSnapshot,
    BiasState,
    ModelState,
    Observation,
    PredictionBatch,
    PredictionRow,
    utc,
)
from .baseline import postprocess_predictions


@dataclass(frozen=True, slots=True)
class BinnedPowerCurve:
    """One turbine curve using occupied wind bins and linear interpolation.

    Values outside the observed wind domain are explicitly clamped to the
    closest learned endpoint.  In particular, the implementation does not
    invent a high-wind cut-out that is absent from the training data.
    """

    turbine_id: str
    bin_width: float
    wind_points: tuple[float, ...]
    median_power: tuple[float, ...]
    sample_counts: tuple[int, ...]
    domain_min_wind_ms: float
    domain_max_wind_ms: float

    def __post_init__(self) -> None:
        width = float(self.bin_width)
        if not self.turbine_id:
            raise ValueError("turbine_id must not be empty")
        if not math.isfinite(width) or width <= 0:
            raise ValueError("bin_width must be a positive finite number")
        object.__setattr__(self, "bin_width", width)
        size = len(self.wind_points)
        if size == 0 or len(self.median_power) != size or len(self.sample_counts) != size:
            raise ValueError("Power curve arrays must have the same non-zero length")
        winds = tuple(float(value) for value in self.wind_points)
        powers = tuple(float(value) for value in self.median_power)
        if any(not math.isfinite(value) or value < 0 for value in winds):
            raise ValueError("Curve wind points must be finite and non-negative")
        if any(left >= right for left, right in zip(winds, winds[1:])):
            raise ValueError("Curve wind points must be strictly increasing")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in powers):
            raise ValueError("Curve power values must be finite and in [0, 1]")
        if any(count <= 0 for count in self.sample_counts):
            raise ValueError("Every retained wind bin needs a sample")
        low = float(self.domain_min_wind_ms)
        high = float(self.domain_max_wind_ms)
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
            raise ValueError("Invalid learned wind domain")
        object.__setattr__(self, "wind_points", winds)
        object.__setattr__(self, "median_power", powers)
        object.__setattr__(self, "domain_min_wind_ms", low)
        object.__setattr__(self, "domain_max_wind_ms", high)

    def predict(self, wind_ms: float) -> float:
        wind = float(wind_ms)
        if not math.isfinite(wind) or wind < 0:
            raise ValueError("wind_ms must be finite and non-negative")
        wind = min(max(wind, self.domain_min_wind_ms), self.domain_max_wind_ms)
        if len(self.wind_points) == 1 or wind <= self.wind_points[0]:
            return self.median_power[0]
        if wind >= self.wind_points[-1]:
            return self.median_power[-1]
        right = bisect_right(self.wind_points, wind)
        left = right - 1
        x0, x1 = self.wind_points[left], self.wind_points[right]
        y0, y1 = self.median_power[left], self.median_power[right]
        weight = (wind - x0) / (x1 - x0)
        return min(1.0, max(0.0, y0 + weight * (y1 - y0)))

    __call__ = predict

    def predict_with_status(self, wind_ms: float) -> tuple[float, str]:
        """Return the bounded prediction and explicit learned-domain status."""

        wind = float(wind_ms)
        value = self.predict(wind)
        status = (
            "out_of_domain"
            if wind < self.domain_min_wind_ms or wind > self.domain_max_wind_ms
            else "ok"
        )
        return value, status

    def to_dict(self) -> dict[str, object]:
        return {
            "turbine_id": self.turbine_id,
            "bin_width": self.bin_width,
            "wind_points": list(self.wind_points),
            "median_power": list(self.median_power),
            "sample_counts": list(self.sample_counts),
            "domain_min_wind_ms": self.domain_min_wind_ms,
            "domain_max_wind_ms": self.domain_max_wind_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "BinnedPowerCurve":
        return cls(
            turbine_id=str(value["turbine_id"]),
            bin_width=float(value["bin_width"]),
            wind_points=tuple(float(item) for item in value["wind_points"]),
            median_power=tuple(float(item) for item in value["median_power"]),
            sample_counts=tuple(int(item) for item in value["sample_counts"]),
            domain_min_wind_ms=float(value["domain_min_wind_ms"]),
            domain_max_wind_ms=float(value["domain_max_wind_ms"]),
        )


def fit_power_curve(
    observations: Iterable[Observation],
    *,
    turbine_id: str | None = None,
    bin_width: float = 1.0,
    min_samples_per_bin: int = 1,
    cutoff: datetime | None = None,
) -> BinnedPowerCurve:
    """Fit a median power value in each occupied wind-speed bin."""

    width = float(bin_width)
    if not math.isfinite(width) or width <= 0:
        raise ValueError("bin_width must be a positive finite number")
    if min_samples_per_bin <= 0:
        raise ValueError("min_samples_per_bin must be positive")
    limit = utc(cutoff) if cutoff is not None else None
    prepared = [
        observation for observation in observations
        if (turbine_id is None or observation.turbine_id == turbine_id)
        and (limit is None or observation.available_at <= limit)
        and observation.quality_flag == "complete"
        and observation.wind_ms is not None
        and observation.power_norm is not None
    ]
    turbine_ids = {observation.turbine_id for observation in prepared}
    if turbine_id is None:
        if len(turbine_ids) != 1:
            raise ValueError("fit_power_curve requires observations from exactly one turbine")
        turbine_id = next(iter(turbine_ids))
    if not prepared:
        raise ValueError(f"POWER_CURVE_NO_TRAINING_DATA: {turbine_id}")

    bins: dict[int, list[Observation]] = {}
    for observation in prepared:
        wind = float(observation.wind_ms)
        power = float(observation.power_norm)
        if not math.isfinite(wind) or wind < 0 or not math.isfinite(power) or not 0 <= power <= 1:
            raise ValueError("Invalid power-curve training value")
        index = math.floor(wind / width)
        bins.setdefault(index, []).append(observation)

    retained = [(index, values) for index, values in sorted(bins.items())
                if len(values) >= min_samples_per_bin]
    if not retained:
        raise ValueError(f"POWER_CURVE_NO_SUPPORTED_BINS: {turbine_id}")
    retained_values = tuple(value for _, values in retained for value in values)
    wind_points = tuple(float(median(float(value.wind_ms) for value in values))
                        for _, values in retained)
    powers = tuple(float(median(float(value.power_norm) for value in values))
                   for _, values in retained)
    return BinnedPowerCurve(
        turbine_id=turbine_id,
        bin_width=width,
        wind_points=wind_points,
        median_power=powers,
        sample_counts=tuple(len(values) for _, values in retained),
        domain_min_wind_ms=min(float(value.wind_ms) for value in retained_values),
        domain_max_wind_ms=max(float(value.wind_ms) for value in retained_values),
    )


def fit_power_curves(
    observations: Iterable[Observation],
    *,
    turbine_ids: Iterable[str] | None = None,
    bin_width: float = 1.0,
    min_samples_per_bin: int = 1,
    cutoff: datetime | None = None,
) -> tuple[BinnedPowerCurve, ...]:
    prepared = tuple(observations)
    requested = set(turbine_ids) if turbine_ids is not None else {
        observation.turbine_id for observation in prepared
    }
    if not requested:
        raise ValueError("POWER_CURVE_NO_TURBINES")
    return tuple(
        fit_power_curve(
            prepared,
            turbine_id=turbine_id,
            bin_width=bin_width,
            min_samples_per_bin=min_samples_per_bin,
            cutoff=cutoff,
        )
        for turbine_id in sorted(requested)
    )


@dataclass(frozen=True, slots=True)
class PowerCurvePredictor:
    state: ModelState
    curves: tuple[BinnedPowerCurve, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", ModelState.model_validate(self.state.model_dump()))
        if not self.curves:
            raise ValueError("PowerCurvePredictor requires at least one curve")
        ids = [curve.turbine_id for curve in self.curves]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate turbine power curve")
        object.__setattr__(self, "curves", tuple(sorted(self.curves, key=lambda curve: curve.turbine_id)))

    @property
    def curves_by_turbine(self) -> dict[str, BinnedPowerCurve]:
        return {curve.turbine_id: curve for curve in self.curves}

    def predict_base(self, snapshot: AsOfSnapshot) -> PredictionBatch:
        features = build_features(snapshot)
        curves = self.curves_by_turbine
        missing = {row.turbine_id for row in features} - curves.keys()
        if missing:
            raise ValueError("POWER_CURVE_UNKNOWN_TURBINE: " + ",".join(sorted(missing)))
        rows = []
        for row in features:
            value, status = curves[row.turbine_id].predict_with_status(row.wind_ms)
            rows.append(PredictionRow(
                turbine_id=row.turbine_id,
                target_start=row.target_start,
                target_end=row.target_end,
                prediction_norm=value,
                status=status,
            ))
        return PredictionBatch(rows=tuple(rows))

    def predict(
        self, snapshot: AsOfSnapshot, bias: BiasState | None = None
    ) -> PredictionBatch:
        return postprocess_predictions(
            self.predict_base(snapshot),
            bias,
            model_id=self.state.model_id,
            origin_time=snapshot.origin_time,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "power_curve",
            "state": self.state.model_dump(mode="json"),
            "curves": [curve.to_dict() for curve in self.curves],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PowerCurvePredictor":
        return cls(
            state=ModelState.model_validate(value["state"]),
            curves=tuple(BinnedPowerCurve.from_dict(item) for item in value["curves"]),
        )


# Public names kept terse for notebooks and reports.
PowerCurve = BinnedPowerCurve
TurbinePowerCurve = BinnedPowerCurve
