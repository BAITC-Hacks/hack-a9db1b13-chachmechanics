from datetime import datetime, timedelta, timezone
import pytest
from windoracle.config import Config, SiteConfig, TurbineConfig
from windoracle.schemas import (ModelState, PredictionBatch, PredictionRow, WeatherBundle,
    WeatherRunMetadata, WeatherValue)
from windoracle.store import Store
from windoracle.weather.cache import WeatherCache
from windoracle.weather.archive import GFSArchive
from windoracle.service import ForecastService

ORIGIN = datetime(2025, 6, 1, 18, tzinfo=timezone.utc)


class FixturePredictor:
    """Test double only; never loaded as a production model."""
    def __init__(self):
        self.state = ModelState(model_id="test-only", training_cutoff=ORIGIN - timedelta(days=2),
            max_label_available_at=ORIGIN - timedelta(days=2), activated_at=ORIGIN - timedelta(days=1),
            artifact_ref="test fixture", provenance="synthetic")
        self.calls = 0

    def predict(self, snapshot, bias=None):
        self.calls += 1
        rows = []
        for v in snapshot.weather_values:
            rows.append(PredictionRow(turbine_id=v.turbine_id, target_start=v.valid_time,
                target_end=v.valid_time + timedelta(hours=1), prediction_norm=min(1, v.wind_ms / 20)))
        return PredictionBatch(rows=tuple(rows))


def bundle(name="run-1", init=None, available=None, wind=5, count=70):
    init = init or ORIGIN - timedelta(hours=6)
    return WeatherBundle(metadata=WeatherRunMetadata(run_id=name, provider="test", model="test",
        run_init_time=init, available_at=available or ORIGIN, availability_basis="synthetic",
        provenance="synthetic", retrieved_at=ORIGIN, sha256="0" * 64),
        values=tuple(WeatherValue(turbine_id=t, valid_time=ORIGIN + timedelta(hours=h), wind_ms=wind,
            temperature_c=10, u_ms=wind, v_ms=0, grid_latitude=0, grid_longitude=0)
            for t in ("turbine_1", "turbine_2") for h in range(1, count + 1)))


@pytest.fixture
def setup(tmp_path):
    config = Config(site=SiteConfig(timezone="UTC", timestamp_semantics="interval_start",
        turbines=tuple(TurbineConfig(id=t, latitude=0, longitude=0) for t in ("turbine_1", "turbine_2"))))
    store = Store(tmp_path / "test.sqlite")
    cache = WeatherCache(tmp_path / "weather")
    cache.save(bundle())
    provider = GFSArchive(config, cache)
    predictor = FixturePredictor()
    return ForecastService(config, store, provider, predictor)
