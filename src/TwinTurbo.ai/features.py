"""Pure feature preparation for participant 2 models.

This module deliberately knows nothing about CSV files, the store, or weather
providers.  Its only forecasting input is the already validated as-of snapshot
prepared by :mod:`windoracle.service`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Iterable

from .schemas import AsOfSnapshot, Observation, utc


FEATURE_SCHEMA_VERSION = "1"
FEATURE_NAMES = (
    "wind_ms",
    "temperature_c",
    "u_ms",
    "v_ms",
    "lead_hours",
    "weather_run_age_hours",
    "target_hour_sin",
    "target_hour_cos",
    "target_year_sin",
    "target_year_cos",
)
LEAD_GROUPS = ("1-6", "7-12", "13-24", "25-48")


def _aware_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} requires an explicit UTC offset")
    return value.astimezone(timezone.utc)


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """One future turbine/hour row derived from an :class:`AsOfSnapshot`."""

    turbine_id: str
    origin_time: datetime
    target_start: datetime
    target_end: datetime
    weather_run_init_time: datetime
    wind_ms: float
    temperature_c: float
    u_ms: float
    v_ms: float

    def __post_init__(self) -> None:
        if not self.turbine_id:
            raise ValueError("turbine_id must not be empty")
        for name in ("origin_time", "target_start", "target_end", "weather_run_init_time"):
            object.__setattr__(self, name, _aware_utc(getattr(self, name), name))
        if self.target_end - self.target_start != timedelta(hours=1):
            raise ValueError("FeatureRow must describe exactly one hour")
        if self.target_start <= self.origin_time:
            raise ValueError("FeatureRow target must be in the future")
        if self.weather_run_init_time > self.origin_time:
            raise ValueError("Weather run cannot initialize after the forecast origin")
        for name in ("wind_ms", "temperature_c", "u_ms", "v_ms"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.wind_ms < 0:
            raise ValueError("wind_ms must be non-negative")


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """Prepared historical-origin example for optional ML models.

    The actual is accompanied by its availability timestamp so callers can
    perform sequential validation without accidentally selecting future labels.
    """

    turbine_id: str
    origin_time: datetime
    target_start: datetime
    weather_run_init_time: datetime
    wind_ms: float
    temperature_c: float
    u_ms: float
    v_ms: float
    actual_norm: float
    actual_available_at: datetime

    def __post_init__(self) -> None:
        if not self.turbine_id:
            raise ValueError("turbine_id must not be empty")
        for name in ("origin_time", "target_start", "weather_run_init_time", "actual_available_at"):
            object.__setattr__(self, name, _aware_utc(getattr(self, name), name))
        if self.target_start <= self.origin_time:
            raise ValueError("Training target must be after its historical origin")
        if self.weather_run_init_time > self.origin_time:
            raise ValueError("Weather run cannot initialize after the historical origin")
        if self.actual_available_at < self.target_start + timedelta(hours=1):
            raise ValueError("Actual cannot be available before its target hour ends")
        for name in ("wind_ms", "temperature_c", "u_ms", "v_ms", "actual_norm"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.wind_ms < 0:
            raise ValueError("wind_ms must be non-negative")
        if not 0 <= self.actual_norm <= 1:
            raise ValueError("actual_norm must be in [0, 1]")


def lead_group(lead_hours: float) -> str:
    """Return the shared evaluation/calibration group for a 1--48 h lead."""

    lead = _finite(lead_hours, "lead_hours")
    if not 0 < lead <= 48:
        raise ValueError("lead_hours must be in (0, 48]")
    if lead <= 6:
        return LEAD_GROUPS[0]
    if lead <= 12:
        return LEAD_GROUPS[1]
    if lead <= 24:
        return LEAD_GROUPS[2]
    return LEAD_GROUPS[3]


def feature_vector(row: FeatureRow | TrainingExample) -> tuple[float, ...]:
    """Return a deterministic numeric vector in :data:`FEATURE_NAMES` order."""

    origin = _aware_utc(row.origin_time, "origin_time")
    target = _aware_utc(row.target_start, "target_start")
    run_init = _aware_utc(row.weather_run_init_time, "weather_run_init_time")
    lead_hours = (target - origin).total_seconds() / 3600
    run_age = (origin - run_init).total_seconds() / 3600
    if lead_hours <= 0 or run_age < 0:
        raise ValueError("Feature times violate as-of ordering")
    hour_angle = 2 * math.pi * (
        target.hour + target.minute / 60 + target.second / 3600
    ) / 24
    year_start = datetime(target.year, 1, 1, tzinfo=timezone.utc)
    next_year = datetime(target.year + 1, 1, 1, tzinfo=timezone.utc)
    year_fraction = (target - year_start).total_seconds() / (next_year - year_start).total_seconds()
    year_angle = 2 * math.pi * year_fraction
    result = (
        _finite(row.wind_ms, "wind_ms"),
        _finite(row.temperature_c, "temperature_c"),
        _finite(row.u_ms, "u_ms"),
        _finite(row.v_ms, "v_ms"),
        lead_hours,
        run_age,
        math.sin(hour_angle),
        math.cos(hour_angle),
        math.sin(year_angle),
        math.cos(year_angle),
    )
    if len(result) != len(FEATURE_NAMES) or not all(math.isfinite(v) for v in result):
        raise ValueError("Feature vector contains an invalid value")
    return result


def build_features(snapshot: AsOfSnapshot) -> tuple[FeatureRow, ...]:
    """Build one row per turbine/target and reject non-exact weather coverage.

    The service normally guarantees this Cartesian coverage.  Rechecking at the
    model boundary keeps direct users and tests fail-closed instead of silently
    forecasting a shortened horizon.
    """

    snapshot = AsOfSnapshot.model_validate(snapshot.model_dump())
    if not snapshot.target_intervals:
        raise ValueError("WEATHER_FEATURE_COVERAGE: no target intervals")
    intervals = {item.target_start: item for item in snapshot.target_intervals}
    if len(intervals) != len(snapshot.target_intervals):
        raise ValueError("WEATHER_FEATURE_COVERAGE: duplicate target intervals")

    weather_by_key = {}
    for value in snapshot.weather_values:
        key = (value.turbine_id, value.valid_time)
        if key in weather_by_key:
            raise ValueError("WEATHER_FEATURE_COVERAGE: duplicate turbine/hour")
        weather_by_key[key] = value

    # Observations retain requested turbine identity even when a weather point
    # is missing.  The upstream service additionally knows the request itself.
    turbine_ids = {value.turbine_id for value in snapshot.weather_values}
    turbine_ids.update(value.turbine_id for value in snapshot.observations)
    if not turbine_ids:
        raise ValueError("WEATHER_FEATURE_COVERAGE: no turbines")
    expected = {(turbine_id, start) for turbine_id in turbine_ids for start in intervals}
    actual = set(weather_by_key)
    if actual != expected:
        missing = len(expected - actual)
        extra = len(actual - expected)
        raise ValueError(
            f"WEATHER_FEATURE_COVERAGE: expected exact turbine/hour grid; missing={missing}, extra={extra}"
        )

    metadata = snapshot.weather_run_metadata
    return tuple(
        FeatureRow(
            turbine_id=turbine_id,
            origin_time=snapshot.origin_time,
            target_start=start,
            target_end=intervals[start].target_end,
            weather_run_init_time=metadata.run_init_time,
            wind_ms=weather_by_key[(turbine_id, start)].wind_ms,
            temperature_c=weather_by_key[(turbine_id, start)].temperature_c,
            u_ms=weather_by_key[(turbine_id, start)].u_ms,
            v_ms=weather_by_key[(turbine_id, start)].v_ms,
        )
        for turbine_id in sorted(turbine_ids)
        for start in sorted(intervals)
    )


def eligible_observations(snapshot: AsOfSnapshot, cutoff: datetime) -> tuple[Observation, ...]:
    """Return complete labels that were actually available by ``cutoff``."""

    cutoff = utc(cutoff)
    if cutoff > snapshot.origin_time:
        raise ValueError("Training cutoff cannot be after the snapshot origin")
    result = (
        observation
        for observation in snapshot.observations
        if observation.available_at <= cutoff
        and observation.quality_flag == "complete"
        and observation.power_norm is not None
        and observation.wind_ms is not None
    )
    return tuple(sorted(result, key=lambda value: (
        value.turbine_id, value.event_start, value.available_at, value.revision
    )))


def select_training_examples(
    examples: Iterable[TrainingExample], cutoff: datetime
) -> tuple[TrainingExample, ...]:
    """Select labels available by a sequential-training cutoff."""

    cutoff = utc(cutoff)
    return tuple(sorted(
        (example for example in examples if example.actual_available_at <= cutoff),
        key=lambda value: (value.turbine_id, value.origin_time, value.target_start),
    ))
