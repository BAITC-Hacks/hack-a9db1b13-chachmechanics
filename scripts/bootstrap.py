"""Validate bundled TwinTurbo.ai assets and prepare the local SQLite store.

The script is intentionally safe to run more than once.  It never replaces an
existing database and never downloads weather unless ``--fetch-january`` is
explicitly supplied.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/site.yaml")
DEFAULT_MODEL = Path("artifacts/models/february-production.json")
DEFAULT_BIAS = Path("artifacts/bias/february-production-bias.json")
DEFAULT_REPORT = Path("reports/january-model-comparison.json")
DEFAULT_INGEST_REPORT = Path("reports/ingest-utc-plus05.json")
DEFAULT_TURBINE_1 = Path("data/raw/turbine_1.csv")
DEFAULT_TURBINE_2 = Path("data/raw/turbine_2.csv")
REPLAY_SEED = Path("outputs/replay-february")


class BootstrapError(RuntimeError):
    """An actionable local packaging or data error."""


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _required_file(path: Path, label: str, action: str) -> Path:
    if not path.is_file():
        raise BootstrapError(f"{label} not found: {path}. {action}")
    return path


def _json_file(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"{label} is not valid UTF-8 JSON: {path}: {exc}") from exc


def _load_project_modules():
    try:
        from windoracle.config import load_config
        from windoracle.cli import read_outputs
        from windoracle.ingest import audit_csv, ingest_csv
        from windoracle.models.registry import load_predictor
        from windoracle.schemas import BiasState, digest
        from windoracle.store import Store
    except ImportError as exc:
        raise BootstrapError(
            "TwinTurbo.ai is not installed. Run "
            "`python -m pip install -r requirements.lock` and "
            "`python -m pip install --no-deps -e .`, then retry."
        ) from exc
    return (
        load_config,
        read_outputs,
        audit_csv,
        ingest_csv,
        load_predictor,
        BiasState,
        digest,
        Store,
    )


def _verify_config(config_path: Path, load_config):
    _required_file(
        config_path,
        "Confirmed site config",
        "Use the included configs/site.yaml; do not deploy the unconfirmed example config.",
    )
    try:
        config = load_config(config_path)
    except Exception as exc:
        raise BootstrapError(f"Site config is invalid: {config_path}: {exc}") from exc
    zone = ZoneInfo(config.site.timezone)
    offsets = {
        datetime(2026, month, 1, tzinfo=zone).utcoffset()
        for month in (1, 7)
    }
    if offsets != {timedelta(hours=5)}:
        raise BootstrapError(
            "This dataset is confirmed as fixed GMT+05:00, but the configured "
            f"timezone {config.site.timezone!r} does not stay at +05:00."
        )
    if config.site.time_basis != "confirmed":
        raise BootstrapError(
            "configs/site.yaml must use time_basis: confirmed for the supplied CSV timestamps."
        )
    return config


def _verify_model(model_path: Path, load_predictor):
    _required_file(
        model_path,
        "Trained model artifact",
        "Restore artifacts/models/february-production.json from the repository.",
    )
    try:
        predictor = load_predictor(model_path)
    except Exception as exc:
        raise BootstrapError(f"Model artifact failed validation: {model_path}: {exc}") from exc
    if predictor.state.provenance != "trained":
        raise BootstrapError("The deployment model must have provenance='trained'.")
    return predictor


def _verify_bias(bias_path: Path, model_id: str, BiasState) -> dict[str, object]:
    if not bias_path.exists():
        return {"status": "not_present", "path": _relative_or_absolute(bias_path)}
    document = _json_file(bias_path, "Bias artifact")
    try:
        bias = BiasState.model_validate(document)
    except Exception as exc:
        raise BootstrapError(f"Bias artifact failed validation: {bias_path}: {exc}") from exc
    if bias.model_id != model_id:
        raise BootstrapError(
            f"Bias {bias.bias_id} belongs to {bias.model_id}, not deployment model {model_id}."
        )
    return {
        "status": "validated",
        "path": _relative_or_absolute(bias_path),
        "bias_id": bias.bias_id,
    }


def _verify_selection_report(report_path: Path, config, model_id: str) -> dict[str, object]:
    if not report_path.exists():
        return {"status": "not_present", "path": _relative_or_absolute(report_path)}
    document = _json_file(report_path, "January comparison report")
    if not isinstance(document, dict) or document.get("schema") != "twinturbo.january-model-selection.v1":
        raise BootstrapError(f"Unsupported January comparison report schema: {report_path}")
    report_hash = document.get("as_of_contract", {}).get("config_hash")
    if report_hash != config.config_hash:
        raise BootstrapError(
            "January report was produced with a different site configuration: "
            f"report={report_hash}, current={config.config_hash}."
        )
    selection = document.get("selection", {})
    if selection.get("selected_model_id") != model_id:
        raise BootstrapError(
            "Deployment model does not match the model selected by the January report."
        )
    mae = selection.get("selected_test_mae")
    if not isinstance(mae, (int, float)) or isinstance(mae, bool) or not math.isfinite(mae):
        raise BootstrapError("January report has no finite held-out MAE.")
    return {
        "status": "validated",
        "path": _relative_or_absolute(report_path),
        "selected": selection.get("selected"),
        "held_out_mae": float(mae),
    }


def _load_replay_seed(
    directory: Path,
    model_id: str,
    bias_id: str | None,
    config,
    read_outputs,
):
    """Load only the checksummed, non-fixture replay shipped by the team."""
    if not directory.exists():
        return (), {"status": "not_present", "path": _relative_or_absolute(directory)}
    if directory.resolve() != (ROOT / REPLAY_SEED).resolve():
        raise BootstrapError("Replay seed directory is not on the deployment whitelist.")
    if not (directory / "index.json").is_file():
        raise BootstrapError(
            f"Replay seed exists without its checksummed index: {directory / 'index.json'}"
        )
    try:
        results = tuple(read_outputs(directory))
    except Exception as exc:
        raise BootstrapError(f"Replay seed failed checksum/contract validation: {exc}") from exc
    if not results:
        raise BootstrapError("Replay seed index contains no forecasts.")
    if bias_id is None:
        raise BootstrapError("A production bias artifact is required for the real replay seed.")
    expected_origins = {
        datetime(2026, 1, 31, 18, tzinfo=timezone.utc) + timedelta(days=index)
        for index in range(28)
    }
    actual_origins = [result.origin_time for result in results]
    if len(results) != 28 or len(set(actual_origins)) != 28 or set(actual_origins) != expected_origins:
        raise BootstrapError(
            "Replay seed must contain exactly one daily forecast for each origin from "
            "2026-01-31T18:00:00Z through 2026-02-27T18:00:00Z."
        )
    turbine_ids = tuple(turbine.id for turbine in config.site.turbines)
    for result in results:
        if result.mode != "replay" or result.provenance != "operational_archive":
            raise BootstrapError(
                f"Replay seed forecast is not a real operational-archive replay: {result.forecast_id}"
            )
        if result.release_kind != "scheduled" or result.parent_forecast_id is not None:
            raise BootstrapError(
                f"Replay seed must contain scheduled releases only: {result.forecast_id}"
            )
        if result.model_id != model_id:
            raise BootstrapError(
                f"Replay seed forecast {result.forecast_id} uses model {result.model_id}, "
                f"not deployment model {model_id}."
            )
        if result.bias_id != bias_id:
            raise BootstrapError(
                f"Replay seed forecast {result.forecast_id} does not use production bias {bias_id}."
            )
        manifest = result.manifest
        if manifest.get("config_hash") != config.config_hash:
            raise BootstrapError(
                f"Replay seed forecast {result.forecast_id} has a different config hash."
            )
        request = manifest.get("request", {})
        if (
            request.get("horizon_hours") != 48
            or request.get("mode") != "replay"
            or tuple(request.get("turbine_ids", ())) != turbine_ids
        ):
            raise BootstrapError(
                f"Replay seed forecast {result.forecast_id} has an unexpected request contract."
            )
        if manifest.get("model", {}).get("model_id") != model_id:
            raise BootstrapError(
                f"Replay seed manifest model differs from its result: {result.forecast_id}"
            )
        if manifest.get("bias", {}).get("bias_id") != bias_id:
            raise BootstrapError(
                f"Replay seed manifest bias differs from its result: {result.forecast_id}"
            )
        rows = result.predictions.rows
        if len(rows) != 96:
            raise BootstrapError(
                f"Replay seed forecast {result.forecast_id} has {len(rows)} rows, expected 96."
            )
        expected_targets = {
            result.origin_time + timedelta(hours=lead) for lead in range(1, 49)
        }
        keys = {(row.turbine_id, row.target_start) for row in rows}
        expected_keys = {
            (turbine_id, target)
            for turbine_id in turbine_ids
            for target in expected_targets
        }
        if keys != expected_keys or len(keys) != len(rows):
            raise BootstrapError(
                f"Replay seed forecast {result.forecast_id} does not have complete unique 48h coverage."
            )
        if any(row.target_end != row.target_start + timedelta(hours=1) for row in rows):
            raise BootstrapError(
                f"Replay seed forecast {result.forecast_id} contains a non-hourly interval."
            )
    prediction_rows = sum(len(result.predictions.rows) for result in results)
    if prediction_rows != 2688:
        raise BootstrapError(
            f"Replay seed has {prediction_rows} prediction rows, expected exactly 2688."
        )
    return results, {
        "status": "validated",
        "path": _relative_or_absolute(directory),
        "forecast_count": len(results),
        "prediction_rows": prediction_rows,
        "first_origin": min(actual_origins).isoformat(),
        "last_origin": max(actual_origins).isoformat(),
    }


def _verify_raw_csvs(paths, turbine_ids, audit_csv, ingest_report_path: Path):
    present = tuple(path.is_file() for path in paths)
    if any(present) and not all(present):
        missing = paths[present.index(False)]
        raise BootstrapError(
            f"Only one raw turbine CSV is present; restore the pair. Missing: {missing}"
        )
    if not all(present):
        return None, {"status": "not_present"}

    audits = {}
    for path, turbine_id in zip(paths, turbine_ids):
        try:
            report = audit_csv(path)
        except Exception as exc:
            raise BootstrapError(f"Raw CSV failed validation: {path}: {exc}") from exc
        rejected = {
            key: report[key]
            for key in (
                "duplicate_timestamps",
                "invalid_numeric_values",
                "invalid_power_rows",
                "negative_wind_rows",
            )
            if report.get(key)
        }
        if rejected:
            raise BootstrapError(f"Raw CSV {path} contains rejected rows: {rejected}")
        if report.get("complete_hours", 0) <= 0:
            raise BootstrapError(f"Raw CSV has no complete hourly observations: {path}")
        audits[turbine_id] = report

    if ingest_report_path.exists():
        expected = _json_file(ingest_report_path, "Ingest audit report")
        if not isinstance(expected, list):
            raise BootstrapError("Ingest audit report must contain a JSON list.")
        expected_by_turbine = {
            item.get("turbine_id"): item for item in expected if isinstance(item, dict)
        }
        for turbine_id, report in audits.items():
            recorded = expected_by_turbine.get(turbine_id)
            if recorded is None or recorded.get("sha256") != report["sha256"]:
                raise BootstrapError(
                    f"Raw CSV checksum for {turbine_id} does not match {ingest_report_path}."
                )

    summary = {
        turbine_id: {
            "path": _relative_or_absolute(path),
            "sha256": audits[turbine_id]["sha256"],
            "rows": audits[turbine_id]["rows"],
            "complete_hours": audits[turbine_id]["complete_hours"],
            "hourly_rows": (
                int(
                    (
                        datetime.fromisoformat(audits[turbine_id]["end"])
                        - datetime.fromisoformat(audits[turbine_id]["start"])
                    ).total_seconds()
                    // 3600
                )
                + 1
            ),
        }
        for path, turbine_id in zip(paths, turbine_ids)
    }
    return audits, {"status": "validated", "turbines": summary}


def _database_counts(path: Path) -> dict[str, int]:
    try:
        # sqlite3.Connection's context manager commits/rolls back but does not
        # close the handle.  Explicit closing is required before an atomic
        # rename on Windows.
        with closing(sqlite3.connect(path)) as database:
            quick_check = database.execute("PRAGMA quick_check").fetchone()
            if quick_check != ("ok",):
                raise BootstrapError(f"SQLite integrity check failed: {quick_check}")
            rows = database.execute(
                "SELECT turbine, COUNT(*) FROM observations GROUP BY turbine ORDER BY turbine"
            ).fetchall()
    except (sqlite3.Error, OSError) as exc:
        raise BootstrapError(f"Cannot validate SQLite database {path}: {exc}") from exc
    return {str(turbine): int(count) for turbine, count in rows}


def _prepare_database(
    database_path: Path,
    config,
    csv_paths,
    audits,
    replay_seed,
    ingest_csv,
    Store,
):
    expected_ids = tuple(turbine.id for turbine in config.site.turbines)
    created = False
    if not database_path.exists():
        if audits is None:
            raise BootstrapError(
                f"Database is absent ({database_path}) and raw CSV files are absent. "
                "Restore data/raw/turbine_1.csv and data/raw/turbine_2.csv, then rerun bootstrap."
            )
        database_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = database_path.with_name(
            f".{database_path.name}.bootstrap-{os.getpid()}.tmp"
        )
        if temporary.exists():
            temporary.unlink()
        try:
            store = Store(temporary)
            for turbine, csv_path in zip(config.site.turbines, csv_paths):
                observations, report = ingest_csv(csv_path, turbine.id, config.site)
                store.ingest(observations, report)
            for result in replay_seed:
                store.save_forecast(result)
            counts = _database_counts(temporary)
            if set(counts) != set(expected_ids):
                raise BootstrapError(f"Temporary database is missing a turbine: {counts}")
            temporary.replace(database_path)
            created = True
        finally:
            temporary.unlink(missing_ok=True)

    if not created and replay_seed:
        store = Store(database_path)
        for result in replay_seed:
            store.save_forecast(result)

    counts = _database_counts(database_path)
    missing = sorted(set(expected_ids) - set(counts))
    if missing or any(counts.get(turbine_id, 0) <= 0 for turbine_id in expected_ids):
        raise BootstrapError(
            f"Database exists but is incomplete ({counts}); missing={missing}. "
            "Move this database aside and rerun bootstrap to rebuild it from the raw CSV pair."
        )
    if audits is not None:
        for turbine_id in expected_ids:
            expected = int(
                (
                    datetime.fromisoformat(audits[turbine_id]["end"])
                    - datetime.fromisoformat(audits[turbine_id]["start"])
                ).total_seconds()
                // 3600
            ) + 1
            if counts[turbine_id] != expected:
                raise BootstrapError(
                    f"Database row count for {turbine_id} is {counts[turbine_id]}, "
                    f"but the included CSV spans {expected} hourly rows."
                )
    try:
        with closing(sqlite3.connect(database_path)) as database:
            forecast_count = int(database.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0])
    except sqlite3.Error as exc:
        raise BootstrapError(f"Cannot count seeded forecasts in {database_path}: {exc}") from exc
    return {
        "status": "created" if created else "reused",
        "path": _relative_or_absolute(database_path),
        "rows_by_turbine": counts,
        "forecast_count": forecast_count,
    }


def _expected_january_runs(config, year: int) -> set[datetime]:
    zone = ZoneInfo(config.site.timezone)
    issue_time = time.fromisoformat(config.forecast.issue_local_time)
    first_target = date(year, 1, 1)
    result = set()
    for day_index in range(31):
        target_day = first_target + timedelta(days=day_index)
        release_day = target_day - timedelta(days=1)
        origin = datetime.combine(release_day, issue_time, tzinfo=zone).astimezone(timezone.utc)
        latest = origin - timedelta(hours=config.weather.publication_delay_hours)
        result.add(latest.replace(hour=latest.hour // 6 * 6, minute=0, second=0, microsecond=0))
    return result


def _cached_run_initializations(cache_dir: Path, digest) -> set[datetime]:
    initialized = set()
    for path in sorted((cache_dir / "runs").glob("*.json")):
        document = _json_file(path, "Weather run manifest")
        if not isinstance(document, dict) or not isinstance(document.get("bundle"), dict):
            raise BootstrapError(f"Invalid weather run manifest: {path}")
        if document.get("checksum") != digest(document["bundle"]):
            raise BootstrapError(f"Weather run manifest checksum mismatch: {path}")
        metadata = document["bundle"].get("metadata", {})
        if metadata.get("provenance") != "operational_archive":
            continue
        try:
            initialized.add(datetime.fromisoformat(metadata["run_init_time"]).astimezone(timezone.utc))
        except (KeyError, TypeError, ValueError) as exc:
            raise BootstrapError(f"Weather run manifest has invalid initialization time: {path}") from exc
    return initialized


def _fetch_january(config_path: Path, config, year: int, max_download_mb: int, digest):
    if not 2000 <= max_download_mb <= 10000:
        raise BootstrapError("--max-download-mb must be between 2000 and 10000 for a full January.")
    cache_dir = _resolve(config.weather.cache_dir)
    expected = _expected_january_runs(config, year)
    cached = _cached_run_initializations(cache_dir, digest)
    missing = expected - cached
    if not missing:
        return {
            "status": "reused",
            "expected_runs": len(expected),
            "cached_runs": len(expected),
            "cache_dir": _relative_or_absolute(cache_dir),
        }

    try:
        import eccodes

        eccodes_version = eccodes.codes_get_api_version()
    except Exception as exc:
        raise BootstrapError(
            "ecCodes native runtime is unavailable. Use Python 3.13 and install "
            "requirements.lock (the Docker image also installs libeccodes0)."
        ) from exc

    report_path = ROOT / "reports" / f"weather-january-{year}-bootstrap.json"
    command = [
        sys.executable,
        "-m",
        "windoracle",
        "weather",
        "fetch",
        "--config",
        str(config_path),
        "--start",
        f"{year:04d}-01-01",
        "--end",
        f"{year:04d}-01-31",
        "--max-runs",
        "31",
        "--max-download-mb",
        str(max_download_mb),
        "--progress",
        "--output",
        str(report_path),
    ]
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise BootstrapError(
            "January GFS download did not finish. Cached fragments were retained; "
            "check network/free disk space and rerun the same --fetch-january command."
        ) from exc
    cached_after = _cached_run_initializations(cache_dir, digest)
    remaining = expected - cached_after
    if remaining:
        first = min(remaining).isoformat()
        raise BootstrapError(
            f"January GFS cache remains incomplete ({len(remaining)} runs; first missing {first})."
        )
    return {
        "status": "downloaded",
        "expected_runs": len(expected),
        "cached_runs": len(expected),
        "cache_dir": _relative_or_absolute(cache_dir),
        "eccodes": eccodes_version,
        "report": _relative_or_absolute(report_path),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Validate included assets and create the local TwinTurbo.ai database."
    )
    result.add_argument(
        "--config",
        default=os.environ.get("TWINTURBO_CONFIG", str(DEFAULT_CONFIG)),
        help="Confirmed fixed-GMT+05 site config.",
    )
    result.add_argument(
        "--model",
        default=os.environ.get("TWINTURBO_MODEL_ARTIFACT", str(DEFAULT_MODEL)),
        help="Selected trained predictor artifact.",
    )
    result.add_argument(
        "--bias",
        default=os.environ.get("TWINTURBO_BIAS_ARTIFACT", str(DEFAULT_BIAS)),
        help="Optional calibrated bias/quantile artifact; validated when present.",
    )
    result.add_argument("--report", default=str(DEFAULT_REPORT))
    result.add_argument("--ingest-report", default=str(DEFAULT_INGEST_REPORT))
    result.add_argument("--turbine-1", default=str(DEFAULT_TURBINE_1))
    result.add_argument("--turbine-2", default=str(DEFAULT_TURBINE_2))
    result.add_argument(
        "--fetch-january",
        action="store_true",
        help="Explicitly fetch/verify all 31 archived GFS runs for January (about 2 GB).",
    )
    result.add_argument("--january-year", type=int, default=2026)
    result.add_argument(
        "--max-download-mb",
        type=int,
        default=4000,
        help="Network safety budget used only with --fetch-january.",
    )
    return result


def bootstrap(args: argparse.Namespace) -> dict[str, object]:
    os.chdir(ROOT)
    (
        load_config,
        read_outputs,
        audit_csv,
        ingest_csv,
        load_predictor,
        BiasState,
        digest,
        Store,
    ) = _load_project_modules()
    config_path = _resolve(args.config)
    model_path = _resolve(args.model)
    bias_path = _resolve(args.bias)
    report_path = _resolve(args.report)
    ingest_report_path = _resolve(args.ingest_report)
    csv_paths = (_resolve(args.turbine_1), _resolve(args.turbine_2))

    config = _verify_config(config_path, load_config)
    predictor = _verify_model(model_path, load_predictor)
    bias_summary = _verify_bias(bias_path, predictor.state.model_id, BiasState)
    replay_seed, replay_summary = _load_replay_seed(
        (ROOT / REPLAY_SEED).resolve(),
        predictor.state.model_id,
        bias_summary.get("bias_id"),
        config,
        read_outputs,
    )
    turbine_ids = tuple(turbine.id for turbine in config.site.turbines)
    if len(turbine_ids) != 2:
        raise BootstrapError(
            f"The packaged two-file dataset requires exactly two turbines, found {turbine_ids}."
        )
    audits, raw_summary = _verify_raw_csvs(
        csv_paths, turbine_ids, audit_csv, ingest_report_path
    )
    database_path = _resolve(config.storage.database)
    summary = {
        "config": {
            "status": "validated",
            "path": _relative_or_absolute(config_path),
            "config_hash": config.config_hash,
            "timezone": config.site.timezone,
            "fixed_utc_offset": "+05:00",
        },
        "raw_csv": raw_summary,
        "database": _prepare_database(
            database_path,
            config,
            csv_paths,
            audits,
            replay_seed,
            ingest_csv,
            Store,
        ),
        "model": {
            "status": "validated",
            "path": _relative_or_absolute(model_path),
            "model_id": predictor.state.model_id,
            "training_cutoff": predictor.state.training_cutoff.isoformat(),
        },
        "bias": bias_summary,
        "report": _verify_selection_report(
            report_path, config, predictor.state.model_id
        ),
        "replay_seed": replay_summary,
        "weather": {"status": "not_requested"},
    }
    if args.fetch_january:
        summary["weather"] = _fetch_january(
            config_path,
            config,
            args.january_year,
            args.max_download_mb,
            digest,
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        summary = bootstrap(args)
    except BootstrapError as exc:
        print(f"BOOTSTRAP_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
