"""CLI adapter; uses exactly the same service as the UI."""
import argparse
from datetime import date, datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
import sys

from .config import load_config
from .ingest import audit_csv, ingest_csv
from .replay import replay, scheduled_origins
from .schemas import ForecastRequest, ForecastResult, ModelState, digest, utc
from .service import ForecastService
from .store import Store
from .weather.archive import GFSArchive
from .weather.audit import audit_bundle
from .weather.cache import atomic_write
from .export import export_csv, verify_result


def timestamp(value):
    return utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def output(value, path=None):
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False)
    if path:
        atomic_write(Path(path), text.encode("utf-8"))
    print(text)


def load_predictor(spec):
    if not spec or ":" not in spec:
        raise ValueError("MODEL_REQUIRED: pass --predictor module:factory from participant 2")
    module, factory = spec.split(":", 1)
    predictor = getattr(importlib.import_module(module), factory)()
    if not callable(getattr(predictor, "predict", None)) or not isinstance(getattr(predictor, "state", None), ModelState):
        raise ValueError("Predictor must have ModelState state and predict(snapshot, bias=None)")
    return predictor


def parser():
    root = argparse.ArgumentParser(prog="TwinTurbo.ai")
    commands = root.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-csv", help="Audit the supplied dataset without assuming timezone")
    audit.add_argument("paths", nargs="+")
    audit.add_argument("--output")
    doctor = commands.add_parser("doctor")
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--turbine-1", required=True)
    ingest.add_argument("--turbine-2", required=True)
    ingest.add_argument("--output", default="reports/ingest.json")
    weather = commands.add_parser("weather")
    wc = weather.add_subparsers(dest="weather_command", required=True)
    fetch = wc.add_parser("fetch")
    fetch.add_argument("--origin", help="Explicit timezone-aware decision time")
    fetch.add_argument("--run", help="Explicit timezone-aware GFS initialization")
    fetch.add_argument("--start", type=date.fromisoformat, help="First target date, inclusive")
    fetch.add_argument("--end", type=date.fromisoformat, help="Last target date, inclusive")
    fetch.add_argument("--max-runs", type=int, default=2, help="Safety limit on origins fetched")
    fetch.add_argument("--output", default="reports/weather-fetch.json")
    wa = wc.add_parser("audit")
    wa.add_argument("--origin")
    wa.add_argument("--output", default="reports/weather-audit.json")
    predict = commands.add_parser("predict")
    predict.add_argument("--origin", required=True)
    predict.add_argument("--output", default="outputs/forecast")
    rp = commands.add_parser("replay")
    rp.add_argument("--start", type=date.fromisoformat, help="First release date, inclusive; defaults to dataset end date")
    rp.add_argument("--end", type=date.fromisoformat)
    rp.add_argument("--updates", action="store_true")
    rp.add_argument("--continue-on-error", action="store_true")
    rp.add_argument("--output", default="outputs/replay")
    for p in (predict, rp):
        p.add_argument("--predictor", required=True, help="Trusted local module:factory")
        p.add_argument("--mode", choices=["replay", "submission", "fixture"])
        p.add_argument("--online", action="store_true", help="Fetch a real archive before calculating")
    export = commands.add_parser("export")
    export.add_argument("--input", help="Optional replay output directory (otherwise database)")
    export.add_argument("--output", required=True)
    export.add_argument("--target-start", help="Local date, inclusive")
    export.add_argument("--target-end", help="Local date, exclusive")
    export.add_argument("--release-policy", choices=["all", "scheduled", "update"], default="scheduled")
    export.add_argument("--allow-fixture", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--input", help="Optional saved output directory")
    for p in (doctor, ingest, fetch, wa, predict, rp, export, verify):
        p.add_argument("--config", default="configs/site.example.yaml")
    return root


def request_for(config, origin, mode=None):
    return ForecastRequest(origin_time=origin, turbine_ids=tuple(t.id for t in config.site.turbines),
                           horizon_hours=config.forecast.horizon_hours, mode=mode or config.forecast.mode)


def persist_outputs(service, ids, directory):
    path = Path(directory)
    checksums = {}
    for identity in ids:
        result = service.get_forecast(identity)
        verify_result(result)
        atomic_write(path / (identity + ".json"), result.model_dump_json(indent=2).encode())
        checksums[identity] = digest(result)
    output({"forecast_ids": ids, "checksums": checksums}, path / "index.json")


def read_outputs(directory):
    path = Path(directory)
    index = json.loads((path / "index.json").read_text(encoding="utf-8"))
    results = []
    for identity in index["forecast_ids"]:
        if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("Invalid forecast filename in index")
        result = ForecastResult.model_validate_json((path / (identity + ".json")).read_text(encoding="utf-8"))
        if result.forecast_id != identity:
            raise ValueError("Forecast file does not match index")
        if index.get("checksums", {}).get(identity) != digest(result):
            raise ValueError("OUTPUT_CHECKSUM_MISMATCH: use a checksummed export from persist_outputs")
        verify_result(result)
        results.append(result)
    return results


def execute(args):
    if args.command == "audit-csv":
        return output([audit_csv(path) for path in args.paths], args.output)
    config = load_config(args.config)
    store = Store(config.storage.database)
    archive = GFSArchive(config)
    if args.command == "doctor":
        import importlib.metadata
        return output({"config_hash": config.config_hash, "time_basis": config.site.time_basis,
                       "warning": "Source timestamps use an unconfirmed processing assumption" if config.site.time_basis == "assumed" else None,
                       "data": store.bounds(), "cached_runs": len(list(archive.cache.bundles())),
                       "model": "Injected with --predictor; no model is silently substituted",
                       "versions": {p: importlib.metadata.version(p) for p in ("pandas", "numpy", "pydantic", "PyYAML")}})
    if args.command == "ingest":
        if len(config.site.turbines) != 2:
            raise ValueError("The two-file CLI expects two configured turbines; Python ingest supports any id")
        reports = []
        for turbine, path in zip(config.site.turbines, [args.turbine_1, args.turbine_2]):
            observations, report = ingest_csv(path, turbine.id, config.site)
            store.ingest(observations, report)
            reports.append({"turbine_id": turbine.id, **report})
        return output(reports, args.output)
    if args.command == "weather":
        if args.weather_command == "audit":
            reports = []
            for bundle in archive.cache.bundles():
                item = {"run_id": bundle.metadata.run_id, "values": len(bundle.values),
                        "metadata": bundle.metadata.model_dump(mode="json", exclude={"evidence", "source_urls"})}
                if args.origin:
                    try:
                        item["audit"] = audit_bundle(bundle, request_for(config, timestamp(args.origin)), config.weather.max_run_age_hours)
                    except ValueError as exc:
                        item["rejected"] = str(exc)
                reports.append(item)
            return output(reports, args.output)
        if args.origin:
            origins = [timestamp(args.origin)]
        elif args.start and args.end:
            # fetch start/end describe target dates, as in the original README.
            origins = list(scheduled_origins(args.start - timedelta(days=1), args.end - timedelta(days=1),
                           config.site.timezone, config.forecast.issue_local_time))
        else:
            raise ValueError("Provide --origin or both --start and --end")
        if len(origins) > args.max_runs or args.max_runs <= 0:
            raise ValueError(f"Download plan has {len(origins)} origins, exceeds --max-runs {args.max_runs}")
        reports = []
        for origin in origins:
            req = request_for(config, origin)
            bundle = archive.fetch_run(timestamp(args.run), req) if args.run else archive.fetch_latest(req)
            reports.append(audit_bundle(bundle, req, config.weather.max_run_age_hours))
        return output(reports, args.output)
    if args.command in ("predict", "replay"):
        service = ForecastService(config, store, archive, load_predictor(args.predictor))
        mode = args.mode or config.forecast.mode
        if args.command == "predict":
            req = request_for(config, timestamp(args.origin), mode)
            from .agents.orchestrator import Orchestrator
            from .clock import VirtualClock
            result = Orchestrator(service, VirtualClock(req.origin_time)).run(req, online=args.online)
            return persist_outputs(service, [result.forecast_id], args.output)
        from zoneinfo import ZoneInfo
        default_end = store.bounds()["end"]
        if not args.start and not default_end:
            raise ValueError("Import a dataset or specify release dates")
        start = args.start or timestamp(default_end).astimezone(ZoneInfo(config.site.timezone)).date()
        end = args.end or start
        origins = list(scheduled_origins(start, end, config.site.timezone, config.forecast.issue_local_time))
        if args.online:
            if len(origins) > 2:
                raise ValueError("Use bounded weather fetch to prepare larger online replays")
            for origin in origins:
                archive.fetch_latest(request_for(config, origin, mode))
        summary = replay(service, origins, mode=mode, include_updates=args.updates, continue_on_error=args.continue_on_error)
        persist_outputs(service, summary["forecast_ids"], args.output)
        output(summary, Path(args.output) / "replay-report.json")
        if summary["failures"]:
            raise RuntimeError("REPLAY_INCOMPLETE: see replay-report.json")
        return
    results = store.forecasts()
    if args.input:
        results = read_outputs(args.input)
    if args.command == "verify":
        if not results:
            raise ValueError("No forecasts to verify")
        for result in results:
            verify_result(result)
        return output({"verified_forecasts": len(results), "rows": sum(len(r.predictions.rows) for r in results)})
    if args.command == "export":
        from zoneinfo import ZoneInfo
        def boundary(value):
            return datetime.combine(date.fromisoformat(value), datetime.min.time(), tzinfo=ZoneInfo(config.site.timezone)) if value else None
        content = export_csv(results, strict=not args.allow_fixture, target_start=boundary(args.target_start),
                             target_end=boundary(args.target_end), release_policy=args.release_policy)
        atomic_write(Path(args.output), content.encode("utf-8"))
        output({"exported": args.output, "rows": len(content.splitlines()) - 1})


def main():
    args = parser().parse_args()
    try:
        execute(args)
    except (ValueError, RuntimeError, KeyError, OSError, ImportError) as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from exc
