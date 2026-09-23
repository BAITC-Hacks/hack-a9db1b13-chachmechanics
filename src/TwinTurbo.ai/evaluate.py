"""Leakage-safe, deterministic evaluation helpers.

This module deliberately performs no I/O.  Callers pass saved forecasts and
already prepared actuals; the data/store layer remains responsible for deciding
which revisions exist.  The evaluator can then freeze that choice with an
``evaluation_as_of`` timestamp and report only facts available by that time.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import json
from math import fsum, isclose, isfinite, sqrt
from typing import Any, Generic, TypeVar

from .schemas import (
    EvaluationReport,
    ForecastResult,
    ModelState,
    Observation,
    PredictionBatch,
    utc,
)


DEFAULT_LEAD_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("lead_01_06", 1, 6),
    ("lead_07_12", 7, 12),
    ("lead_13_24", 13, 24),
    ("lead_25_48", 25, 48),
)


def _as_utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    try:
        return utc(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be timezone-aware") from exc


def _finite(value: Any, name: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _power(value: Any, name: str, *, nullable: bool = False) -> float | None:
    number = _finite(value, name, nullable=nullable)
    if number is not None and not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


@dataclass(frozen=True, order=True, slots=True)
class ForecastKey:
    """The fair-comparison key required by the project quality protocol."""

    origin_time: datetime
    target_start: datetime
    turbine_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_time", _as_utc(self.origin_time, "origin_time"))
        object.__setattr__(self, "target_start", _as_utc(self.target_start, "target_start"))
        if not self.turbine_id:
            raise ValueError("turbine_id must not be empty")
        if self.target_start <= self.origin_time:
            raise ValueError("target_start must be after origin_time")


@dataclass(frozen=True, slots=True)
class EvaluationPoint:
    """One immutable forecast/actual pair prepared by the integration layer."""

    turbine_id: str
    origin_time: datetime
    target_start: datetime
    prediction_norm: float | None
    actual_norm: float | None
    actual_available_at: datetime | None
    target_end: datetime | None = None
    q10: float | None = None
    q50: float | None = None
    q90: float | None = None
    status: str = "ok"

    def __post_init__(self) -> None:
        origin = _as_utc(self.origin_time, "origin_time")
        start = _as_utc(self.target_start, "target_start")
        end = _as_utc(self.target_end or start + timedelta(hours=1), "target_end")
        available = (None if self.actual_available_at is None else
                     _as_utc(self.actual_available_at, "actual_available_at"))
        prediction = _power(self.prediction_norm, "prediction_norm", nullable=True)
        actual = _power(self.actual_norm, "actual_norm", nullable=True)
        quantiles = tuple(_power(value, name, nullable=True) for value, name in (
            (self.q10, "q10"), (self.q50, "q50"), (self.q90, "q90")))

        if not self.turbine_id:
            raise ValueError("turbine_id must not be empty")
        if start <= origin:
            raise ValueError("target_start must be after origin_time")
        if end - start != timedelta(hours=1):
            raise ValueError("target interval must be exactly one hour")
        if actual is not None and available is None:
            raise ValueError("actual_available_at is required when actual_norm is present")
        if available is not None and available < end:
            raise ValueError("an actual cannot be available before target_end")
        if any(value is not None for value in quantiles):
            if any(value is None for value in quantiles):
                raise ValueError("q10, q50 and q90 must be all present or all absent")
            if not quantiles[0] <= quantiles[1] <= quantiles[2]:
                raise ValueError("quantiles must satisfy q10 <= q50 <= q90")

        object.__setattr__(self, "origin_time", origin)
        object.__setattr__(self, "target_start", start)
        object.__setattr__(self, "target_end", end)
        object.__setattr__(self, "actual_available_at", available)
        object.__setattr__(self, "prediction_norm", prediction)
        object.__setattr__(self, "actual_norm", actual)
        object.__setattr__(self, "q10", quantiles[0])
        object.__setattr__(self, "q50", quantiles[1])
        object.__setattr__(self, "q90", quantiles[2])

    @property
    def key(self) -> ForecastKey:
        return ForecastKey(self.origin_time, self.target_start, self.turbine_id)

    @property
    def lead_hours(self) -> int:
        hours = (self.target_start - self.origin_time).total_seconds() / 3600
        rounded = round(hours)
        if not isclose(hours, rounded, abs_tol=1e-12):
            raise ValueError("forecast lead must be a whole number of hours")
        return int(rounded)

    @property
    def has_interval(self) -> bool:
        return self.q10 is not None


_MISSING = object()


def _field(value: Any, names: Sequence[str], default: Any = _MISSING) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    if default is not _MISSING:
        return default
    raise TypeError(f"missing required field {names[0]}")


def as_evaluation_point(value: EvaluationPoint | Mapping[str, Any] | Any) -> EvaluationPoint:
    """Normalize a mapping/object without depending on pandas or a data store."""

    if isinstance(value, EvaluationPoint):
        return value
    return EvaluationPoint(
        turbine_id=_field(value, ("turbine_id",)),
        origin_time=_field(value, ("origin_time",)),
        target_start=_field(value, ("target_start",)),
        target_end=_field(value, ("target_end",), None),
        prediction_norm=_field(value, ("prediction_norm", "prediction", "y_pred"), None),
        actual_norm=_field(value, ("actual_norm", "actual", "power_norm", "y_true"), None),
        actual_available_at=_field(value, ("actual_available_at", "available_at"), None),
        q10=_field(value, ("q10",), None),
        q50=_field(value, ("q50",), None),
        q90=_field(value, ("q90",), None),
        status=_field(value, ("status",), "ok"),
    )


def _as_key(value: ForecastKey | EvaluationPoint | Sequence[Any] | Mapping[str, Any] | Any) -> ForecastKey:
    if isinstance(value, ForecastKey):
        return value
    if isinstance(value, EvaluationPoint):
        return value.key
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) == 3:
        if isinstance(value[0], datetime):
            return ForecastKey(value[0], value[1], value[2])
        return ForecastKey(value[1], value[2], value[0])
    return ForecastKey(
        _field(value, ("origin_time",)),
        _field(value, ("target_start",)),
        _field(value, ("turbine_id",)),
    )


def _numbers(values: Iterable[Any], name: str) -> tuple[float, ...]:
    result = tuple(_finite(value, name) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result  # type: ignore[return-value]


def _pairs(actual: Iterable[Any], prediction: Iterable[Any]) -> tuple[tuple[float, float], ...]:
    left = _numbers(actual, "actual")
    right = _numbers(prediction, "prediction")
    if len(left) != len(right):
        raise ValueError("actual and prediction must have equal lengths")
    return tuple(zip(left, right))


def mae(actual: Iterable[Any], prediction: Iterable[Any]) -> float:
    pairs = _pairs(actual, prediction)
    return fsum(abs(predicted - observed) for observed, predicted in pairs) / len(pairs)


def rmse(actual: Iterable[Any], prediction: Iterable[Any]) -> float:
    pairs = _pairs(actual, prediction)
    return sqrt(fsum((predicted - observed) ** 2 for observed, predicted in pairs) / len(pairs))


def bias(actual: Iterable[Any], prediction: Iterable[Any]) -> float:
    """Return signed forecast bias, defined explicitly as prediction - actual."""

    pairs = _pairs(actual, prediction)
    return fsum(predicted - observed for observed, predicted in pairs) / len(pairs)


mean_absolute_error = mae
root_mean_squared_error = rmse
mean_bias = bias


def pinball_loss(actual: Iterable[Any], quantile_prediction: Iterable[Any], quantile: float) -> float:
    q = _finite(quantile, "quantile")
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
    pairs = _pairs(actual, quantile_prediction)
    losses = []
    for observed, predicted in pairs:
        residual = observed - predicted
        losses.append(max(q * residual, (q - 1.0) * residual))
    return fsum(losses) / len(losses)


def _triples(actual: Iterable[Any], lower: Iterable[Any], upper: Iterable[Any]) -> tuple[tuple[float, float, float], ...]:
    observed = _numbers(actual, "actual")
    lows = _numbers(lower, "lower")
    highs = _numbers(upper, "upper")
    if len(observed) != len(lows) or len(observed) != len(highs):
        raise ValueError("actual, lower and upper must have equal lengths")
    result = tuple(zip(observed, lows, highs))
    if any(lo > hi for _, lo, hi in result):
        raise ValueError("interval lower bound exceeds upper bound")
    return result


def interval_coverage(actual: Iterable[Any], lower: Iterable[Any], upper: Iterable[Any]) -> float:
    values = _triples(actual, lower, upper)
    return sum(lo <= observed <= hi for observed, lo, hi in values) / len(values)


def mean_interval_width(lower: Iterable[Any], upper: Iterable[Any]) -> float:
    lows = _numbers(lower, "lower")
    highs = _numbers(upper, "upper")
    if len(lows) != len(highs):
        raise ValueError("lower and upper must have equal lengths")
    if any(lo > hi for lo, hi in zip(lows, highs)):
        raise ValueError("interval lower bound exceeds upper bound")
    return fsum(hi - lo for lo, hi in zip(lows, highs)) / len(lows)


def lead_group(lead_hours: int, groups: Sequence[tuple[str, int, int]] = DEFAULT_LEAD_GROUPS) -> str:
    for name, first, last in groups:
        if first <= lead_hours <= last:
            return name
    return "lead_other"


def _validate_groups(groups: Sequence[tuple[str, int, int]]) -> tuple[tuple[str, int, int], ...]:
    result = tuple(groups)
    if not result:
        raise ValueError("at least one lead group is required")
    names: set[str] = set()
    occupied: set[int] = set()
    for name, first, last in result:
        if not name or name in names or first <= 0 or last < first:
            raise ValueError("lead groups require unique names and positive ordered bounds")
        hours = set(range(first, last + 1))
        if occupied & hours:
            raise ValueError("lead groups must not overlap")
        names.add(name)
        occupied |= hours
    return tuple(sorted(result, key=lambda value: (value[1], value[2], value[0])))


def _index(points: Iterable[EvaluationPoint | Mapping[str, Any] | Any]) -> dict[ForecastKey, EvaluationPoint]:
    result: dict[ForecastKey, EvaluationPoint] = {}
    for raw in points:
        point = as_evaluation_point(raw)
        if point.key in result:
            raise ValueError(f"duplicate forecast key: {point.key}")
        result[point.key] = point
    return result


def forecast_coverage(
    points: Iterable[EvaluationPoint | Mapping[str, Any] | Any],
    expected_keys: Iterable[ForecastKey | EvaluationPoint | Sequence[Any] | Mapping[str, Any] | Any] | None = None,
) -> float:
    indexed = _index(points)
    expected = set(indexed) if expected_keys is None else {_as_key(value) for value in expected_keys}
    if not expected:
        return 0.0
    present = sum(key in indexed and indexed[key].prediction_norm is not None for key in expected)
    return present / len(expected)


def _point_metrics(points: Sequence[EvaluationPoint]) -> dict[str, int | float]:
    actual = [point.actual_norm for point in points]
    prediction = [point.prediction_norm for point in points]
    # Scored points are selected before this helper, so Optional values cannot occur.
    return {
        "count": len(points),
        "mae": mae(actual, prediction),
        "rmse": rmse(actual, prediction),
        "bias": bias(actual, prediction),
    }


def _interval_values(points: Sequence[EvaluationPoint]) -> dict[str, Any]:
    actual = [point.actual_norm for point in points]
    q10 = [point.q10 for point in points]
    q50 = [point.q50 for point in points]
    q90 = [point.q90 for point in points]
    return {
        "count": len(points),
        "coverage_q10_q90": interval_coverage(actual, q10, q90),
        "mean_width_q10_q90": mean_interval_width(q10, q90),
        "pinball_loss": {
            "q10": pinball_loss(actual, q10, 0.1),
            "q50": pinball_loss(actual, q50, 0.5),
            "q90": pinball_loss(actual, q90, 0.9),
        },
    }


def _by_turbine_and_lead(
    points: Sequence[EvaluationPoint],
    groups: Sequence[tuple[str, int, int]],
    metric: Callable[[Sequence[EvaluationPoint]], dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for turbine in sorted({point.turbine_id for point in points}):
        turbine_points = [point for point in points if point.turbine_id == turbine]
        grouped: dict[str, Any] = {"all": metric(turbine_points)}
        names = sorted({lead_group(point.lead_hours, groups) for point in turbine_points})
        for name in names:
            subset = [point for point in turbine_points if lead_group(point.lead_hours, groups) == name]
            grouped[name] = metric(subset)
        result[turbine] = grouped
    return result


def evaluate(
    points: Iterable[EvaluationPoint | Mapping[str, Any] | Any],
    *,
    expected_keys: Iterable[ForecastKey | EvaluationPoint | Sequence[Any] | Mapping[str, Any] | Any] | None = None,
    period: tuple[datetime, datetime] | None = None,
    evaluation_as_of: datetime | None = None,
    lead_groups: Sequence[tuple[str, int, int]] = DEFAULT_LEAD_GROUPS,
) -> EvaluationReport:
    """Build an :class:`EvaluationReport` from immutable prepared pairs.

    ``forecast_coverage`` has an operationally honest denominator only when the
    caller supplies every planned ``expected_key``, including failed origins.
    Without that argument it is coverage of the supplied key set.
    """

    groups = _validate_groups(lead_groups)
    indexed = _index(points)
    all_points = tuple(indexed[key] for key in sorted(indexed))
    normalized_expected = (None if expected_keys is None else
                           {_as_key(value) for value in expected_keys})
    if period is None:
        starts = ([point.target_start for point in all_points]
                  + [key.target_start for key in normalized_expected or ()])
        ends = ([point.target_end for point in all_points]
                + [key.target_start + timedelta(hours=1) for key in normalized_expected or ()])
        if not starts:
            raise ValueError("period is required when no points or expected keys are supplied")
        period_start = min(starts)
        period_end = max(ends)
    else:
        period_start = _as_utc(period[0], "period start")
        period_end = _as_utc(period[1], "period end")
        if period_end <= period_start:
            raise ValueError("period end must be after its start")
    as_of = None if evaluation_as_of is None else _as_utc(evaluation_as_of, "evaluation_as_of")

    selected = tuple(point for point in all_points
                     if period_start <= point.target_start < period_end)
    expected = (set(point.key for point in selected) if normalized_expected is None else
                {key for key in normalized_expected
                 if period_start <= key.target_start < period_end})
    if expected:
        coverage = sum(key in indexed and indexed[key].prediction_norm is not None
                       for key in expected) / len(expected)
    else:
        coverage = 0.0

    scored = tuple(point for point in selected
                   if point.prediction_norm is not None
                   and point.actual_norm is not None
                   and (as_of is None or point.actual_available_at <= as_of))
    metrics = _by_turbine_and_lead(scored, groups, _point_metrics) if scored else {}

    interval_points = tuple(point for point in scored if point.has_interval)
    interval_metrics = None
    if interval_points:
        interval_metrics = {
            "target_coverage_q10_q90": 0.8,
            "eligible_sample_count": len(scored),
            "interval_sample_count": len(interval_points),
            "interval_availability": len(interval_points) / len(scored),
            **_interval_values(interval_points),
            "by_turbine_and_lead": _by_turbine_and_lead(interval_points, groups, _interval_values),
        }

    return EvaluationReport(
        period=(period_start, period_end),
        metrics_by_turbine_and_lead=metrics,
        sample_count=len(scored),
        forecast_coverage=coverage,
        interval_metrics=interval_metrics,
    )


evaluate_records = evaluate


def align_forecasts(
    baseline: Iterable[EvaluationPoint | Mapping[str, Any] | Any],
    candidate: Iterable[EvaluationPoint | Mapping[str, Any] | Any],
) -> tuple[tuple[EvaluationPoint, ...], tuple[EvaluationPoint, ...]]:
    """Align two models on identical ``(origin, target, turbine)`` keys."""

    base_index = _index(baseline)
    candidate_index = _index(candidate)
    common = sorted(set(base_index) & set(candidate_index))
    base_result: list[EvaluationPoint] = []
    candidate_result: list[EvaluationPoint] = []
    for key in common:
        base_point = base_index[key]
        candidate_point = candidate_index[key]
        if base_point.actual_norm is not None and candidate_point.actual_norm is not None:
            if (base_point.actual_norm != candidate_point.actual_norm or
                    base_point.actual_available_at != candidate_point.actual_available_at):
                raise ValueError(f"actual mismatch for common key: {key}")
        elif base_point.actual_norm is None and candidate_point.actual_norm is not None:
            base_point = replace(base_point, actual_norm=candidate_point.actual_norm,
                                 actual_available_at=candidate_point.actual_available_at)
        elif candidate_point.actual_norm is None and base_point.actual_norm is not None:
            candidate_point = replace(candidate_point, actual_norm=base_point.actual_norm,
                                      actual_available_at=base_point.actual_available_at)
        base_result.append(base_point)
        candidate_result.append(candidate_point)
    return tuple(base_result), tuple(candidate_result)


align_on_common_keys = align_forecasts


@dataclass(frozen=True, slots=True)
class ModelComparison:
    common_keys: tuple[ForecastKey, ...]
    baseline: EvaluationReport
    candidate: EvaluationReport

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-safe comparison report document."""

        return {
            "schema": "twinturbo.model-comparison.v1",
            "common_keys": [
                {
                    "origin_time": key.origin_time.isoformat(),
                    "target_start": key.target_start.isoformat(),
                    "turbine_id": key.turbine_id,
                }
                for key in self.common_keys
            ],
            "baseline": self.baseline.model_dump(mode="json"),
            "candidate": self.candidate.model_dump(mode="json"),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministically; suitable for a reproducible report."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )


def compare_models(
    baseline: Iterable[EvaluationPoint | Mapping[str, Any] | Any],
    candidate: Iterable[EvaluationPoint | Mapping[str, Any] | Any],
    *,
    expected_keys: Iterable[ForecastKey | EvaluationPoint | Sequence[Any] | Mapping[str, Any] | Any] | None = None,
    period: tuple[datetime, datetime] | None = None,
    evaluation_as_of: datetime | None = None,
    lead_groups: Sequence[tuple[str, int, int]] = DEFAULT_LEAD_GROUPS,
) -> ModelComparison:
    """Compare metrics on common keys while retaining per-model coverage."""

    base_all = tuple(as_evaluation_point(value) for value in baseline)
    candidate_all = tuple(as_evaluation_point(value) for value in candidate)
    base_common, candidate_common = align_forecasts(base_all, candidate_all)
    if period is None:
        combined = base_all + candidate_all
        if not combined:
            raise ValueError("period is required when both model inputs are empty")
        period = (min(point.target_start for point in combined),
                  max(point.target_end for point in combined))
    expected = (tuple(expected_keys) if expected_keys is not None else
                tuple(sorted({point.key for point in base_all + candidate_all})))
    baseline_report = evaluate(base_common, expected_keys=expected, period=period,
                               evaluation_as_of=evaluation_as_of, lead_groups=lead_groups)
    candidate_report = evaluate(candidate_common, expected_keys=expected, period=period,
                                evaluation_as_of=evaluation_as_of, lead_groups=lead_groups)
    period_start = _as_utc(period[0], "period start")
    period_end = _as_utc(period[1], "period end")
    expected_in_period = tuple(
        key for key in expected if period_start <= _as_key(key).target_start < period_end
    )
    base_in_period = tuple(
        point for point in base_all if period_start <= point.target_start < period_end
    )
    candidate_in_period = tuple(
        point for point in candidate_all if period_start <= point.target_start < period_end
    )
    baseline_report = baseline_report.model_copy(update={
        "forecast_coverage": forecast_coverage(base_in_period, expected_in_period),
    })
    candidate_report = candidate_report.model_copy(update={
        "forecast_coverage": forecast_coverage(candidate_in_period, expected_in_period),
    })
    return ModelComparison(
        common_keys=tuple(point.key for point in base_common),
        baseline=baseline_report,
        candidate=candidate_report,
    )


def _latest_complete_actuals(
    actuals: Iterable[Observation | Mapping[str, Any] | Any], as_of: datetime
) -> dict[tuple[str, datetime, datetime], Observation]:
    facts: dict[tuple[str, datetime, datetime], Observation] = {}
    for raw in actuals:
        actual = Observation.model_validate(
            raw.model_dump() if hasattr(raw, "model_dump") else raw
        )
        if (actual.available_at > as_of or actual.quality_flag != "complete"
                or actual.power_norm is None):
            continue
        key = (actual.turbine_id, actual.event_start, actual.event_end)
        previous = facts.get(key)
        if previous and previous.available_at == actual.available_at and previous != actual:
            raise ValueError("ambiguous actual revisions at the same availability time")
        if previous is None or actual.available_at > previous.available_at:
            facts[key] = actual
    return facts


def points_from_forecasts(
    forecasts: Iterable[ForecastResult],
    actuals: Iterable[Observation],
    *,
    evaluation_as_of: datetime,
    release_kind: str = "scheduled",
) -> tuple[EvaluationPoint, ...]:
    """Join saved forecasts to the latest allowed actual revision, without I/O."""

    as_of = _as_utc(evaluation_as_of, "evaluation_as_of")
    if release_kind not in {"scheduled", "update", "all"}:
        raise ValueError("release_kind must be scheduled, update or all")
    facts = _latest_complete_actuals(actuals, as_of)

    points: list[EvaluationPoint] = []
    seen: set[ForecastKey] = set()
    for forecast in sorted(forecasts, key=lambda value: (value.origin_time, value.forecast_id)):
        if release_kind != "all" and forecast.release_kind != release_kind:
            continue
        for row in forecast.predictions.rows:
            actual = facts.get((row.turbine_id, row.target_start, row.target_end))
            point = EvaluationPoint(
                turbine_id=row.turbine_id,
                origin_time=forecast.origin_time,
                target_start=row.target_start,
                target_end=row.target_end,
                prediction_norm=row.prediction_norm,
                actual_norm=actual.power_norm if actual else None,
                actual_available_at=actual.available_at if actual else None,
                q10=row.q10,
                q50=row.q50,
                q90=row.q90,
                status=row.status,
            )
            if point.key in seen:
                raise ValueError("multiple selected releases share one evaluation key")
            seen.add(point.key)
            points.append(point)
    return tuple(sorted(points, key=lambda point: point.key))


def residuals_from_forecasts(
    forecasts: Iterable[ForecastResult],
    observations: Iterable[Observation],
    *,
    as_of: datetime,
    base_batches: Mapping[str, PredictionBatch | Mapping[str, Any] | Any] | None = None,
):
    """Build bias/interval residuals from saved out-of-sample releases.

    A corrected and clipped issue cannot be inverted to recover its base
    prediction.  Therefore every forecast carrying ``bias_id`` must have its
    original pre-correction batch supplied in ``base_batches[forecast_id]``.
    The function is pure: it neither queries nor updates the shared store.
    """

    # Lazy import keeps general metric evaluation independent of the bias
    # implementation and avoids a models -> evaluate import cycle.
    from .models.bias import Residual

    cutoff = _as_utc(as_of, "as_of")
    facts = _latest_complete_actuals(observations, cutoff)
    if base_batches is not None and not isinstance(base_batches, Mapping):
        raise TypeError("base_batches must map forecast_id to PredictionBatch")
    supplied = base_batches or {}
    residuals = []

    prepared = []
    for raw in forecasts:
        forecast = ForecastResult.model_validate(
            raw.model_dump() if hasattr(raw, "model_dump") else raw
        )
        prepared.append(forecast)
    for forecast in sorted(prepared, key=lambda value: (value.origin_time, value.forecast_id)):
        raw_state = forecast.manifest.get("model")
        if raw_state is None:
            raise ValueError("FORECAST_MODEL_STATE_REQUIRED")
        state = ModelState.model_validate(raw_state)
        if state.model_id != forecast.model_id:
            raise ValueError("FORECAST_MODEL_ID_MISMATCH")

        issued_rows = {}
        for row in forecast.predictions.rows:
            key = (row.turbine_id, row.target_start, row.target_end)
            if key in issued_rows:
                raise ValueError("DUPLICATE_FORECAST_ROW")
            issued_rows[key] = row

        # With no mature matching fact there is nothing to calibrate, so a
        # historical base batch is not needed yet.  This preserves the core
        # "no actual -> no update" behaviour of Critic.
        if not set(issued_rows) & set(facts):
            continue

        base_rows = None
        if forecast.bias_id is not None:
            if forecast.forecast_id not in supplied:
                raise ValueError("BASE_BATCH_REQUIRED_FOR_CORRECTED_FORECAST")
            raw_batch = supplied[forecast.forecast_id]
            batch = PredictionBatch.model_validate(
                raw_batch.model_dump() if hasattr(raw_batch, "model_dump") else raw_batch
            )
            base_rows = {}
            for row in batch.rows:
                key = (row.turbine_id, row.target_start, row.target_end)
                if key in base_rows:
                    raise ValueError("DUPLICATE_BASE_FORECAST_ROW")
                base_rows[key] = row
            if set(base_rows) != set(issued_rows):
                raise ValueError("BASE_BATCH_COVERAGE_MISMATCH")

        for key, issued in issued_rows.items():
            actual = facts.get(key)
            if actual is None:
                continue
            base = issued if base_rows is None else base_rows[key]
            residuals.append(Residual(
                forecast_id=forecast.forecast_id,
                model_id=forecast.model_id,
                turbine_id=issued.turbine_id,
                origin_time=forecast.origin_time,
                target_start=issued.target_start,
                target_end=issued.target_end,
                training_cutoff=state.training_cutoff,
                actual_available_at=actual.available_at,
                actual_revision=actual.revision,
                actual=actual.power_norm,
                p_base=base.prediction_norm,
                p_issued=issued.prediction_norm,
                release_kind=forecast.release_kind,
            ))
    return tuple(sorted(residuals, key=lambda row: (
        row.origin_time, row.turbine_id, row.target_start, row.forecast_id,
    )))


LabelT = TypeVar("LabelT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class WalkForwardFold(Generic[LabelT]):
    origin_time: datetime
    train: tuple[LabelT, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_time", _as_utc(self.origin_time, "origin_time"))


def walk_forward_folds(
    origins: Iterable[datetime],
    labels: Iterable[LabelT],
    *,
    get_actual_available_at: Callable[[LabelT], datetime] | None = None,
) -> tuple[WalkForwardFold[LabelT], ...]:
    """Return expanding folds whose train labels were available by each origin."""

    ordered_origins = tuple(sorted(_as_utc(value, "origin") for value in origins))
    if len(set(ordered_origins)) != len(ordered_origins):
        raise ValueError("walk-forward origins must be unique")
    get_available = (get_actual_available_at or
                     (lambda item: _field(item, ("actual_available_at", "available_at"))))
    prepared = [(_as_utc(get_available(label), "actual_available_at"), index, label)
                for index, label in enumerate(labels)]
    prepared.sort(key=lambda value: (value[0], value[1]))
    return tuple(WalkForwardFold(
        origin_time=origin,
        train=tuple(label for available, _, label in prepared if available <= origin),
    ) for origin in ordered_origins)


sequential_folds = walk_forward_folds


def walk_forward(
    origins: Iterable[datetime],
    labels: Iterable[LabelT],
    run_fold: Callable[[tuple[LabelT, ...], datetime], ResultT],
    *,
    get_actual_available_at: Callable[[LabelT], datetime] | None = None,
) -> tuple[ResultT, ...]:
    """Run a generic deterministic expanding-window evaluation.

    ``run_fold`` receives ``(train, origin)``.  The train tuple is constructed
    centrally, so no label with ``actual_available_at > origin`` can enter it.
    """

    return tuple(run_fold(fold.train, fold.origin_time) for fold in walk_forward_folds(
        origins, labels, get_actual_available_at=get_actual_available_at))
