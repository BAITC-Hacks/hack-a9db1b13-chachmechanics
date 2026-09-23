"""Public immutable contracts shared by the data, model and UI components."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from typing import Annotated, Any, Literal, Protocol

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp requires an explicit UTC offset")
    return value.astimezone(timezone.utc)


UTCDateTime = Annotated[datetime, AfterValidator(utc)]
Power = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Mode = Literal["replay", "submission", "fixture"]
Provenance = Literal["operational_archive", "hindcast", "reanalysis", "synthetic"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def digest(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class ForecastRequest(Contract):
    origin_time: UTCDateTime
    turbine_ids: tuple[str, ...]
    horizon_hours: Literal[24, 48] = 48
    mode: Mode = "replay"
    release_kind: Literal["scheduled", "update"] = "scheduled"
    target_start: UTCDateTime | None = None

    @model_validator(mode="after")
    def check(self):
        if not self.turbine_ids or len(set(self.turbine_ids)) != len(self.turbine_ids):
            raise ValueError("Provide unique turbine_ids")
        start = self.target_start or self.origin_time + timedelta(hours=1)
        if start <= self.origin_time or start.minute or start.second or start.microsecond:
            raise ValueError("Targets must start at a full future hour")
        return self


class TargetInterval(Contract):
    target_start: UTCDateTime
    target_end: UTCDateTime

    @model_validator(mode="after")
    def check(self):
        if self.target_end - self.target_start != timedelta(hours=1):
            raise ValueError("Target interval must be exactly one hour")
        return self


class Observation(Contract):
    turbine_id: str
    event_start: UTCDateTime
    event_end: UTCDateTime
    available_at: UTCDateTime
    power_norm: Power | None
    wind_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    temperature_c: Finite | None
    n_samples: int = Field(ge=0, le=6)
    coverage: float = Field(ge=0, le=1)
    quality_flag: Literal["complete", "incomplete", "missing", "invalid"]
    revision: str

    @model_validator(mode="after")
    def check(self):
        if self.event_end - self.event_start != timedelta(hours=1):
            raise ValueError("Observation must cover one hour")
        if self.available_at < self.event_end:
            raise ValueError("Observation cannot be available before its end")
        if self.quality_flag == "complete" and (self.n_samples != 6 or self.coverage != 1
                or any(v is None for v in (self.power_norm, self.wind_ms, self.temperature_c))):
            raise ValueError("Complete hour requires six valid observations")
        return self


class WeatherRunMetadata(Contract):
    run_id: str
    provider: str
    model: str
    run_init_time: UTCDateTime
    available_at: UTCDateTime
    availability_basis: Literal["archive_last_modified_plus_delay", "estimated", "synthetic"]
    provenance: Provenance
    retrieved_at: UTCDateTime
    sha256: str
    source_urls: tuple[str, ...] = ()
    wind_height_m: int = 100
    interpolation: str = "linear u/v and temperature within a single run; nearest grid point"
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check(self):
        if self.available_at < self.run_init_time:
            raise ValueError("Run cannot be available before initialization")
        return self


class WeatherValue(Contract):
    turbine_id: str
    valid_time: UTCDateTime
    wind_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    temperature_c: Finite
    u_ms: Finite
    v_ms: Finite
    grid_latitude: Finite
    grid_longitude: Finite


class WeatherBundle(Contract):
    metadata: WeatherRunMetadata
    values: tuple[WeatherValue, ...]


class AsOfSnapshot(Contract):
    origin_time: UTCDateTime
    observations: tuple[Observation, ...]
    weather_values: tuple[WeatherValue, ...]
    weather_run_metadata: WeatherRunMetadata
    target_intervals: tuple[TargetInterval, ...]
    quality_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def check(self):
        if self.weather_run_metadata.available_at > self.origin_time:
            raise ValueError("FUTURE_WEATHER")
        if any(o.available_at > self.origin_time for o in self.observations):
            raise ValueError("FUTURE_OBSERVATION")
        if any(t.target_start <= self.origin_time for t in self.target_intervals):
            raise ValueError("Target is not in the future")
        return self


class ModelState(Contract):
    model_id: str
    training_cutoff: UTCDateTime
    max_label_available_at: UTCDateTime
    activated_at: UTCDateTime
    artifact_ref: str
    feature_schema_version: str = "1"
    provenance: Literal["trained", "synthetic"] = "trained"

    @model_validator(mode="after")
    def check(self):
        if not self.max_label_available_at <= self.training_cutoff <= self.activated_at:
            raise ValueError("Model training/activation time invariant violated")
        return self


class BiasState(Contract):
    bias_id: str
    model_id: str
    created_as_of: UTCDateTime
    last_actual_available_at: UTCDateTime
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check(self):
        if self.last_actual_available_at > self.created_as_of:
            raise ValueError("Bias uses a future actual")
        return self


class PredictionRow(TargetInterval):
    turbine_id: str
    prediction_norm: Power
    q10: Power | None = None
    q50: Power | None = None
    q90: Power | None = None
    status: str = "ok"

    @model_validator(mode="after")
    def check_quantiles(self):
        q = (self.q10, self.q50, self.q90)
        if any(v is not None for v in q) and (any(v is None for v in q)
                                            or not self.q10 <= self.q50 <= self.q90):
            raise ValueError("Quantiles must be all null or ordered")
        return self


class PredictionBatch(Contract):
    rows: tuple[PredictionRow, ...]


class ForecastResult(Contract):
    forecast_id: str
    origin_time: UTCDateTime
    predictions: PredictionBatch
    run_id: str
    model_id: str
    bias_id: str | None = None
    parent_forecast_id: str | None = None
    warnings: tuple[str, ...] = ()
    provenance: Provenance
    mode: Mode
    release_kind: Literal["scheduled", "update"]
    manifest: dict[str, Any]


class EvaluationReport(Contract):
    period: tuple[UTCDateTime, UTCDateTime]
    metrics_by_turbine_and_lead: dict[str, Any]
    sample_count: int = Field(ge=0)
    forecast_coverage: float = Field(ge=0, le=1)
    interval_metrics: dict[str, Any] | None = None


class Predictor(Protocol):
    """Participant 2 supplies this object; no I/O is permitted in predict."""
    state: ModelState

    def predict(self, snapshot: AsOfSnapshot, bias: BiasState | None = None) -> PredictionBatch: ...
