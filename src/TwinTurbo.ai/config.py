from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo
import yaml
from pydantic import Field
from .schemas import Contract, digest


class TurbineConfig(Contract):
    id: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    rated_power_mw: float | None = Field(default=None, gt=0)


class SiteConfig(Contract):
    timezone: str
    timestamp_semantics: Literal["interval_start", "interval_end", "instant"]
    time_basis: Literal["confirmed", "assumed"] = "assumed"
    observation_delay_minutes: int = Field(default=15, ge=0)
    turbines: tuple[TurbineConfig, ...]


class WeatherConfig(Contract):
    provider: Literal["noaa_gfs_s3"] = "noaa_gfs_s3"
    publication_delay_hours: int = Field(default=6, ge=0, le=24)
    max_run_age_hours: int = Field(default=24, ge=6, le=120)
    wind_height_m: Literal[10, 100] = 100
    sample_step_hours: Literal[1, 3] = 3
    max_download_mb_per_run: int = Field(default=250, ge=1, le=1000)
    cache_dir: str = "data/weather"


class ForecastConfig(Contract):
    issue_local_time: str = "23:00"
    horizon_hours: Literal[24, 48] = 48
    mode: Literal["replay", "submission", "fixture"] = "replay"


class StorageConfig(Contract):
    database: str = "artifacts/TwinTurbo.ai.sqlite"


class Config(Contract):
    site: SiteConfig
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @property
    def config_hash(self):
        return digest(self)


def load_config(path: str | Path) -> Config:
    config = Config.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    ZoneInfo(config.site.timezone)
    from datetime import time
    time.fromisoformat(config.forecast.issue_local_time)
    if len({t.id for t in config.site.turbines}) != len(config.site.turbines) or not config.site.turbines:
        raise ValueError("Unique turbine ids are required")
    return config
