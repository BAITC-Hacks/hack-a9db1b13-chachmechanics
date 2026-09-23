"""Explicit SYNTHETIC integration smoke test; this does not train a wind model.

python scripts/prepare_demo.py --fixture --origin 2025-06-01T18:00:00Z
python -m TwinTurbo.ai verify --input outputs/integration-smoke
"""
import argparse
from datetime import timedelta
from pathlib import Path

from TwinTurbo.ai.cli import output, persist_outputs, timestamp
from TwinTurbo.ai.config import load_config
from TwinTurbo.ai.replay import replay
from TwinTurbo.ai.schemas import (ModelState, PredictionBatch, PredictionRow,
                               WeatherBundle, WeatherRunMetadata, WeatherValue)
from TwinTurbo.ai.service import ForecastService
from TwinTurbo.ai.store import Store
from TwinTurbo.ai.weather.archive import GFSArchive
from TwinTurbo.ai.weather.cache import WeatherCache, atomic_write


class SmokePredictor:
    """A constant test double, deliberately marked synthetic and unscored."""
    def __init__(self, origin):
        self.state = ModelState(model_id="integration-smoke-only", provenance="synthetic",
            training_cutoff=origin - timedelta(days=2), max_label_available_at=origin - timedelta(days=2),
            activated_at=origin - timedelta(days=1), artifact_ref="scripts/prepare_demo.py")

    def predict(self, snapshot, bias=None):
        return PredictionBatch(rows=tuple(PredictionRow(turbine_id=v.turbine_id,
            target_start=v.valid_time, target_end=v.valid_time + timedelta(hours=1),
            prediction_norm=0.5, status="SYNTHETIC_INTEGRATION_TEST") for v in snapshot.weather_values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true", required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--config", default="configs/site.example.yaml")
    parser.add_argument("--output", default="outputs/integration-smoke")
    parser.add_argument("--real-weather", action="store_true", help="Use cached NOAA inputs; predictions still synthetic")
    args = parser.parse_args()
    origin = timestamp(args.origin)
    config = load_config(args.config)
    directory = Path(args.output)
    store = Store(directory / "fixture.sqlite")
    if args.real_weather:
        archive = GFSArchive(config)
        origins, updates = [origin], False
    else:
        cache = WeatherCache(directory / "fixture-weather")
        for i in range(2):
            init = origin - timedelta(hours=6) + timedelta(hours=6 * i)
            values = tuple(WeatherValue(turbine_id=t.id, valid_time=origin + timedelta(hours=h),
                wind_ms=5 + i, temperature_c=10, u_ms=5 + i, v_ms=0,
                grid_latitude=t.latitude, grid_longitude=t.longitude)
                for t in config.site.turbines for h in range(1, 80))
            cache.save(WeatherBundle(metadata=WeatherRunMetadata(run_id=f"synthetic-{init:%Y%m%dT%HZ}",
                provider="integration fixture", model="synthetic", run_init_time=init,
                available_at=init + timedelta(hours=6), retrieved_at=origin,
                availability_basis="synthetic", provenance="synthetic", sha256="0" * 64), values=values))
        archive = GFSArchive(config, cache)
        origins, updates = [origin, origin + timedelta(hours=7)], True
    service = ForecastService(config, store, archive, SmokePredictor(origin))
    report = replay(service, origins, mode="fixture", include_updates=updates)
    persist_outputs(service, report["forecast_ids"], directory)
    atomic_write(directory / "fixture.csv", service.export(report["forecast_ids"], strict=False).encode())
    output({**report, "mode": "fixture", "model": "synthetic; no quality claim", "events": service.events()},
           directory / "smoke-report.json")


if __name__ == "__main__":
    main()
