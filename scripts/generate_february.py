"""Generate the reproducible February 2026 replay from prepared artifacts.

This script is deliberately offline.  It reads the imported SQLite store,
the immutable model/bias artifacts, and already cached admissible GFS bundles.
It never downloads weather and never opens the raw turbine CSV files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Literal

from windoracle.config import load_config
from windoracle.export import export_csv, verify_result
from windoracle.models.registry import load_predictor
from windoracle.schemas import (
    BiasState,
    ForecastRequest,
    ForecastResult,
    PredictionBatch,
    digest,
    utc,
)
from windoracle.service import ForecastService
from windoracle.store import Store
from windoracle.weather.audit import audit_bundle
from windoracle.weather.cache import WeatherCache, atomic_write


UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "site.yaml"
DEFAULT_MODEL = ROOT / "artifacts" / "models" / "february-production.json"
DEFAULT_BIAS = ROOT / "artifacts" / "bias" / "february-production-bias.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "replay-february"
DEFAULT_START = datetime(2026, 1, 31, 18, tzinfo=UTC)
DEFAULT_END = datetime(2026, 2, 27, 18, tzinfo=UTC)


class CachedWeatherProvider:
    """Select only from bundles loaded and audited from the local cache.

    Loading the cache once avoids re-hashing every raw GRIB range for every
    forecast origin.  Selection remains equivalent to ``GFSArchive.select_run``
    and retains its coordinate, height, publication-delay, age, and coverage
    checks.  There is intentionally no network/fetch method on this provider.
    """

    def __init__(self, config, bundles):
        self.config = config
        self.bundles = tuple(bundles)
        self.expected_points = {
            turbine.id: (turbine.latitude, turbine.longitude)
            for turbine in config.site.turbines
        }

    def select_run(self, request: ForecastRequest):
        candidates = []
        for bundle in self.bundles:
            metadata = bundle.metadata
            context = metadata.evidence.get("context", {})
            if metadata.provenance == "operational_archive":
                points = {
                    turbine["id"]: (turbine["latitude"], turbine["longitude"])
                    for turbine in context.get("turbines", [])
                }
                if any(
                    points.get(turbine_id)
                    != self.expected_points.get(turbine_id)
                    for turbine_id in request.turbine_ids
                ):
                    continue
                if metadata.wind_height_m != self.config.weather.wind_height_m:
                    continue
                if request.origin_time < metadata.run_init_time + timedelta(
                    hours=self.config.weather.publication_delay_hours
                ):
                    continue
            try:
                audit_bundle(
                    bundle, request, self.config.weather.max_run_age_hours
                )
            except ValueError:
                continue
            candidates.append(bundle)
        if not candidates:
            raise RuntimeError(
                "WEATHER_UNAVAILABLE: no complete, admissible cached run for "
                + request.origin_time.isoformat()
            )
        return max(
            candidates,
            key=lambda bundle: (
                bundle.metadata.run_init_time,
                bundle.metadata.available_at,
                bundle.metadata.run_id,
            ),
        )


def timestamp(value: str) -> datetime:
    try:
        return utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "timestamp must be ISO 8601 with an explicit UTC offset"
        ) from exc


def resolve_from_root(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def daily_origins(start: datetime, end: datetime) -> tuple[datetime, ...]:
    start, end = utc(start), utc(end)
    if start > end:
        raise ValueError("START_AFTER_END")
    if any(
        value.minute or value.second or value.microsecond for value in (start, end)
    ):
        raise ValueError("ORIGINS_MUST_BE_FULL_HOURS")
    span = end - start
    if span % timedelta(days=1):
        raise ValueError("ORIGIN_RANGE_MUST_USE_WHOLE_DAYS")
    return tuple(
        start + timedelta(days=index)
        for index in range(span.days + 1)
    )


def load_bias(path: Path) -> BiasState:
    if not path.is_file():
        raise FileNotFoundError(f"Bias artifact not found: {path}")
    try:
        return BiasState.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid bias artifact: {path}") from exc


def bias_lifecycle(
    bias: BiasState, origin: datetime
) -> tuple[BiasState | None, Literal["active", "future_skipped", "expired_passthrough"]]:
    """Return a temporally safe bias input and its operational state.

    An expired state is passed through solely so model post-processing can
    emit truthful ``expired_history`` statuses.  ``apply_bias`` then uses a
    zero adjustment and ``apply_intervals`` emits null quantiles.  The replay
    validates both properties before exporting the result.
    """

    origin = utc(origin)
    if bias.created_as_of > origin:
        return None, "future_skipped"
    window_days = bias.parameters.get("window_days")
    if (
        isinstance(window_days, bool)
        or not isinstance(window_days, (int, float))
        or not math.isfinite(float(window_days))
        or float(window_days) <= 0
    ):
        raise ValueError("INVALID_BIAS_WINDOW")
    if origin - bias.last_actual_available_at > timedelta(days=float(window_days)):
        return bias, "expired_passthrough"
    return bias, "active"


def validate_bias_result(
    result: ForecastResult,
    lifecycle: Literal["active", "future_skipped", "expired_passthrough"],
    bias: BiasState,
) -> None:
    if lifecycle == "future_skipped":
        if result.bias_id is not None:
            raise ValueError("FUTURE_BIAS_WAS_PUBLISHED")
        return
    if result.bias_id != bias.bias_id:
        raise ValueError("BIAS_ID_NOT_PUBLISHED")
    if lifecycle != "expired_passthrough":
        return

    base_payload = result.manifest.get("base_predictions")
    if base_payload is None:
        raise ValueError("EXPIRED_BIAS_REQUIRES_BASE_TRACE")
    base_rows = {
        (row.turbine_id, row.target_start, row.target_end): row
        for row in PredictionBatch.model_validate(base_payload).rows
    }
    for row in result.predictions.rows:
        tokens = set(row.status.split("|"))
        if "bias:expired_history" not in tokens:
            raise ValueError("STALE_BIAS_STATUS_MISSING")
        if "interval:expired_history" not in tokens:
            raise ValueError("STALE_INTERVAL_STATUS_MISSING")
        if any(value is not None for value in (row.q10, row.q50, row.q90)):
            raise ValueError("STALE_INTERVAL_WAS_PUBLISHED")
        key = (
            row.turbine_id,
            row.target_start,
            row.target_end,
        )
        base = base_rows.get(key)
        if base is None or row.prediction_norm != base.prediction_norm:
            raise ValueError("STALE_BIAS_CHANGED_POINT_FORECAST")


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_outputs(
    service: ForecastService,
    results: tuple[ForecastResult, ...],
    directory: Path,
    *,
    report: dict[str, object],
) -> None:
    ids = []
    checksums = {}
    for result in results:
        verify_result(result)
        identity = result.forecast_id
        ids.append(identity)
        checksums[identity] = digest(result)
        atomic_write(
            directory / f"{identity}.json",
            result.model_dump_json(indent=2).encode("utf-8"),
        )
    index = {
        "schema": "twinturbo.replay-index.v1",
        "forecast_ids": ids,
        "checksums": checksums,
        "config_hash": service.config.config_hash,
        "model_id": service.predictor.state.model_id,
    }
    atomic_write(
        directory / "index.json",
        json.dumps(
            index, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8"),
    )
    atomic_write(
        directory / "forecasts.csv",
        export_csv(results, strict=True, release_policy="scheduled").encode("utf-8"),
    )
    atomic_write(
        directory / "replay-report.json",
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8"),
    )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    command.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    command.add_argument("--bias", type=Path, default=DEFAULT_BIAS)
    command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument(
        "--database",
        type=Path,
        help="Override the configured SQLite store (relative paths use repository root).",
    )
    command.add_argument(
        "--start",
        type=timestamp,
        default=DEFAULT_START,
        help="First daily origin, inclusive (default: 2026-01-31T18:00:00Z).",
    )
    command.add_argument(
        "--end",
        type=timestamp,
        default=DEFAULT_END,
        help="Last daily origin, inclusive (default: 2026-02-27T18:00:00Z).",
    )
    command.add_argument(
        "--horizon-hours",
        type=int,
        choices=(24, 48),
        help="Override the configured 24/48-hour forecast horizon.",
    )
    command.add_argument(
        "--mode", choices=("replay", "submission"), default="replay"
    )
    return command


def run(args: argparse.Namespace) -> dict[str, object]:
    config_path = resolve_from_root(args.config)
    model_path = resolve_from_root(args.model)
    bias_path = resolve_from_root(args.bias)
    output_path = resolve_from_root(args.output)
    config = load_config(config_path)
    predictor = load_predictor(model_path)
    bias = load_bias(bias_path)
    if bias.model_id != predictor.state.model_id:
        raise ValueError("BIAS_MODEL_MISMATCH")

    origins = daily_origins(args.start, args.end)
    if predictor.state.activated_at > origins[0]:
        raise ValueError("FUTURE_MODEL")
    turbine_ids = tuple(turbine.id for turbine in config.site.turbines)
    horizon = args.horizon_hours or config.forecast.horizon_hours
    requests = tuple(
        ForecastRequest(
            origin_time=origin,
            turbine_ids=turbine_ids,
            horizon_hours=horizon,
            mode=args.mode,
        )
        for origin in origins
    )

    cache_path = resolve_from_root(Path(config.weather.cache_dir))
    bundles = tuple(WeatherCache(cache_path).bundles())
    if not bundles:
        raise RuntimeError(f"WEATHER_CACHE_EMPTY: {cache_path}")
    provider = CachedWeatherProvider(config, bundles)

    # Complete the read-only weather preflight before mutating the forecast
    # store or output directory.  A partial month must never look successful.
    selected_runs = tuple(provider.select_run(request) for request in requests)
    database_path = resolve_from_root(
        args.database if args.database is not None else Path(config.storage.database)
    )
    store = Store(database_path)
    service = ForecastService(config, store, provider, predictor)

    results = []
    lifecycle_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for request in requests:
        selected_bias, lifecycle = bias_lifecycle(bias, request.origin_time)
        result = service.create_forecast(request, bias=selected_bias)
        validate_bias_result(result, lifecycle, bias)
        results.append(result)
        lifecycle_counts[lifecycle] += 1
        for row in result.predictions.rows:
            status_counts.update(row.status.split("|"))

    result_tuple = tuple(results)
    report = {
        "schema": "twinturbo.february-replay.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "offline_only": True,
        "raw_csv_read": False,
        "config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
            "config_hash": config.config_hash,
            "source_timezone": config.site.timezone,
        },
        "model": {
            "path": str(model_path),
            "sha256": file_sha256(model_path),
            "model_id": predictor.state.model_id,
            "training_cutoff": predictor.state.training_cutoff.isoformat(),
        },
        "bias": {
            "path": str(bias_path),
            "sha256": file_sha256(bias_path),
            "bias_id": bias.bias_id,
            "created_as_of": bias.created_as_of.isoformat(),
            "last_actual_available_at": bias.last_actual_available_at.isoformat(),
            "window_days": bias.parameters["window_days"],
            "lifecycle_counts": dict(sorted(lifecycle_counts.items())),
            "expiry_policy": (
                "expired state is retained only to emit expired_history; "
                "point correction is zero and quantiles are null"
            ),
        },
        "period": {
            "first_origin": origins[0].isoformat(),
            "last_origin": origins[-1].isoformat(),
            "origin_count": len(origins),
            "horizon_hours": horizon,
            "expected_rows": len(origins) * len(turbine_ids) * horizon,
            "published_rows": sum(len(result.predictions.rows) for result in results),
        },
        "weather": {
            "cache_path": str(cache_path),
            "cached_bundle_count": len(bundles),
            "selected_runs": [
                {
                    "origin_time": request.origin_time.isoformat(),
                    "run_id": bundle.metadata.run_id,
                    "run_init_time": bundle.metadata.run_init_time.isoformat(),
                    "available_at": bundle.metadata.available_at.isoformat(),
                    "sha256": bundle.metadata.sha256,
                }
                for request, bundle in zip(requests, selected_runs, strict=True)
            ],
        },
        "store": str(database_path),
        "output": str(output_path),
        "forecast_ids": [result.forecast_id for result in results],
        "status_counts": dict(sorted(status_counts.items())),
    }
    write_outputs(service, result_tuple, output_path, report=report)
    return report


def main() -> None:
    report = run(parser().parse_args())
    print(
        json.dumps(
            {
                "output": report["output"],
                "origin_count": report["period"]["origin_count"],
                "published_rows": report["period"]["published_rows"],
                "model_id": report["model"]["model_id"],
                "bias_id": report["bias"]["bias_id"],
                "bias_lifecycle": report["bias"]["lifecycle_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
