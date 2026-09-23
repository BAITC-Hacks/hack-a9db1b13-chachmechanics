"""Pure transformations of participant 1's prepared, timezone-aware snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from zoneinfo import ZoneInfo

from .schemas import AsOfSnapshot, Observation, digest, utc

FEATURE_SCHEMA = "weather-calendar-v1"
FEATURE_NAMES = ("wind_ms", "temperature_c", "direction_sin", "direction_cos",
                 "lead_hours", "weather_lead_hours", "run_age_hours",
                 "hour_sin", "hour_cos", "year_sin", "year_cos")
LEAD_GROUPS = ("1-6", "7-12", "13-24", "25-48")


def lead_group(hours: float) -> str:
    if not math.isfinite(hours) or not 0 < hours <= 48:
        raise ValueError("LEAD_OUT_OF_RANGE: expected 0 < lead_hours <= 48")
    return LEAD_GROUPS[next(i for i, edge in enumerate((6, 12, 24, 48)) if hours <= edge)]


@dataclass(frozen=True)
class FeatureRow:
    turbine_id: str
    origin_time: datetime
    target_start: datetime
    target_end: datetime
    run_id: str
    provenance: str
    values: tuple[float, ...]

    @property
    def key(self):
        return self.origin_time, self.turbine_id, self.target_start, self.target_end

    @property
    def lead_hours(self):
        return (self.target_start - self.origin_time).total_seconds() / 3600


def features_from_snapshot(snapshot: AsOfSnapshot, *, calendar_timezone: str = "UTC") -> tuple[FeatureRow, ...]:
    """No imputation, telemetry lag, future observation or weather-run selection."""
    snapshot = AsOfSnapshot.model_validate(snapshot.model_dump())
    zone = ZoneInfo(calendar_timezone)
    targets = {t.target_start: t for t in snapshot.target_intervals}
    if not targets or len(targets) != len(snapshot.target_intervals):
        raise ValueError("INVALID_TARGET_GRID")
    values = {(v.turbine_id, v.valid_time): v for v in snapshot.weather_values}
    turbines = {v.turbine_id for v in snapshot.weather_values}
    expected = {(t, h) for t in turbines for h in targets}
    if not turbines or len(values) != len(snapshot.weather_values) or set(values) != expected:
        raise ValueError("WEATHER_COVERAGE: one supplied value per turbine/target required")
    meta = snapshot.weather_run_metadata
    rows = []
    for (turbine, target), weather in sorted(values.items()):
        lead = (target - snapshot.origin_time).total_seconds() / 3600
        lead_group(lead)
        dt = target.astimezone(zone)
        hour = 2 * math.pi * (dt.hour + dt.minute / 60) / 24
        year = 2 * math.pi * (dt.timetuple().tm_yday - 1) / 365.25
        norm = math.hypot(weather.u_ms, weather.v_ms)
        vals = (weather.wind_ms, weather.temperature_c,
                weather.u_ms / norm if norm else 0.0, weather.v_ms / norm if norm else 0.0,
                lead, (target - meta.run_init_time).total_seconds() / 3600,
                (snapshot.origin_time - meta.run_init_time).total_seconds() / 3600,
                math.sin(hour), math.cos(hour), math.sin(year), math.cos(year))
        rows.append(FeatureRow(turbine, snapshot.origin_time, target, targets[target].target_end,
                               meta.run_id, meta.provenance, vals))
    return tuple(rows)


def latest_observations(observations, as_of: datetime) -> tuple[Observation, ...]:
    """Select label revisions at an explicit training/evaluation cutoff, never a clock."""
    as_of = utc(as_of)
    selected = {}
    for raw in observations:
        obs = Observation.model_validate(raw.model_dump())
        if obs.available_at > as_of:
            continue
        key = obs.turbine_id, obs.event_start, obs.event_end
        old = selected.get(key)
        if old and old.available_at == obs.available_at and old != obs:
            raise ValueError("AMBIGUOUS_OBSERVATION_REVISION")
        if old is None or obs.available_at > old.available_at:
            selected[key] = obs
    return tuple(selected[key] for key in sorted(selected))


def training_observations(snapshot: AsOfSnapshot) -> tuple[Observation, ...]:
    snapshot = AsOfSnapshot.model_validate(snapshot.model_dump())
    turbines = {v.turbine_id for v in snapshot.weather_values}
    rows = tuple(o for o in latest_observations(snapshot.observations, snapshot.origin_time)
                 if o.turbine_id in turbines and o.quality_flag == "complete")
    if not turbines or turbines - {o.turbine_id for o in rows}:
        raise ValueError("INSUFFICIENT_TRAINING_HISTORY")
    return rows


@dataclass(frozen=True)
class TrainingExample:
    feature: FeatureRow
    power_norm: float
    label_available_at: datetime
    label_revision: str


def supervised_examples(snapshots, observations, *, training_cutoff: datetime,
                        calendar_timezone: str = "UTC", allow_synthetic: bool = False):
    """Join archived forecast features to matured labels; crossing 48h labels excluded."""
    cutoff = utc(training_cutoff)
    labels = {(o.turbine_id, o.event_start, o.event_end): o
              for o in latest_observations(observations, cutoff) if o.quality_flag == "complete"}
    examples = {}
    for snapshot in snapshots:
        if snapshot.origin_time > cutoff:
            continue
        for row in features_from_snapshot(snapshot, calendar_timezone=calendar_timezone):
            allowed = {"operational_archive", "synthetic"} if allow_synthetic else {"operational_archive"}
            if row.provenance not in allowed:
                raise ValueError("ML_REQUIRES_OPERATIONAL_FORECASTS")
            obs = labels.get((row.turbine_id, row.target_start, row.target_end))
            if obs is None:
                continue
            item = TrainingExample(row, obs.power_norm, obs.available_at, obs.revision)
            if row.key in examples and examples[row.key] != item:
                raise ValueError("DUPLICATE_TRAINING_ORIGIN: supply one designated run")
            examples[row.key] = item
    return tuple(examples[key] for key in sorted(examples))
