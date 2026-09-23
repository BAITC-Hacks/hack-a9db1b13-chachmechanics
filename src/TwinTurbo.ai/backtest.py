"""Sequential prepared-snapshot experiment from the forecast-models branch.

The generic evaluation API stays in evaluate.py. This runner never downloads
weather or reads raw telemetry and never fits on validation targets.
"""
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from .agents.twin_builder import TwinBuilder
from .evaluate import ForecastKey, evaluate, points_from_forecasts, residuals_from_forecasts
from .models.bias import update_bias
from .schemas import AsOfSnapshot, ForecastResult, Observation, PredictionBatch, digest, utc
from .weather.cache import atomic_write


@dataclass(frozen=True)
class TemporalFold:
    training: AsOfSnapshot
    validation: tuple[AsOfSnapshot, ...]


def walk_forward_snapshots(folds, observations, *, as_of, fitters=None, min_interval_samples=30):
    as_of = utc(as_of)
    observations = tuple(observations)
    if fitters is None:
        def build(snapshot, kind):
            provenance = "synthetic" if snapshot.weather_run_metadata.provenance == "synthetic" else "trained"
            return TwinBuilder(provenance=provenance).build(snapshot, kind=kind)
        fitters = {"baseline": lambda s: build(s, "baseline"), "curve": lambda s: build(s, "power_curve")}
    output = {name: [] for name in fitters}
    if "curve" in fitters:
        output["curve_bias"] = []
    expected, failures, training, states = set(), [], [], {}
    previous_origin = None
    period_start = period_end = None
    for fold in folds:
        snapshot_train = AsOfSnapshot.model_validate(fold.training.model_dump())
        if snapshot_train.origin_time > as_of:
            raise ValueError("FUTURE_TRAINING_FOLD")
        models = {name: fitter(snapshot_train) for name, fitter in fitters.items()}
        training.append({name: model.state.model_dump(mode="json") for name, model in models.items()})
        for raw in fold.validation:
            snapshot = AsOfSnapshot.model_validate(raw.model_dump())
            if (snapshot.origin_time < snapshot_train.origin_time or snapshot.origin_time > as_of
                    or previous_origin is not None and snapshot.origin_time <= previous_origin):
                raise ValueError("VALIDATION_ORIGINS_MUST_INCREASE_AFTER_TRAINING")
            previous_origin = snapshot.origin_time
            turbines = {v.turbine_id for v in snapshot.weather_values}
            keys = {(t, h.target_start, h.target_end) for t in turbines for h in snapshot.target_intervals}
            expected.update(ForecastKey(snapshot.origin_time, start, turbine) for turbine, start, end in keys)
            start, end = min(h.target_start for h in snapshot.target_intervals), max(h.target_end for h in snapshot.target_intervals)
            period_start = min(period_start, start) if period_start else start
            period_end = max(period_end, end) if period_end else end
            for name in output:
                model = models["curve" if name == "curve_bias" else name]
                try:
                    if model.state.activated_at > snapshot.origin_time:
                        raise ValueError("FUTURE_MODEL")
                    bias = None
                    if name == "curve_bias":
                        residuals = residuals_from_forecasts(output[name], observations, as_of=snapshot.origin_time)
                        bias = update_bias(residuals, model_id=model.state.model_id, as_of=snapshot.origin_time,
                            previous=states.get(name), min_interval_samples=min_interval_samples)
                        states[name] = bias
                    base = PredictionBatch.model_validate(model.predict_base(snapshot))
                    batch = PredictionBatch.model_validate(model.predict(snapshot, bias))
                    for values in (base, batch):
                        actual = [(r.turbine_id, r.target_start, r.target_end) for r in values.rows]
                        if len(actual) != len(keys) or set(actual) != keys:
                            raise ValueError("MODEL_OUTPUT_COVERAGE")
                    identity = {"snapshot_hash": digest(snapshot), "model": model.state.model_dump(mode="json"),
                                "bias": bias.model_dump(mode="json") if bias else None, "experiment": name}
                    fixture = model.state.provenance == "synthetic" or snapshot.weather_run_metadata.provenance == "synthetic"
                    output[name].append(ForecastResult(forecast_id=digest(identity), origin_time=snapshot.origin_time,
                        predictions=batch, run_id=snapshot.weather_run_metadata.run_id, model_id=model.state.model_id,
                        bias_id=bias.bias_id if bias else None, provenance=snapshot.weather_run_metadata.provenance,
                        mode="fixture" if fixture else "replay", release_kind="scheduled", manifest={**identity,
                            "base_predictions": base.model_dump(mode="json")}))
                except (ValueError, RuntimeError) as exc:
                    failures.append({"model": name, "origin_time": snapshot.origin_time.isoformat(), "reason": str(exc)})
    if not expected:
        raise ValueError("NO_VALIDATION_ORIGINS")
    points = {name: points_from_forecasts(values, observations, evaluation_as_of=as_of) for name, values in output.items()}
    common = set.intersection(*(set(p.key for p in values) for values in points.values()))
    metrics = {name: evaluate([p for p in values if p.key in common], expected_keys=expected,
                             period=(period_start, period_end), evaluation_as_of=as_of).model_dump(mode="json")
               for name, values in points.items()}
    report = {"schema": "twinturbo.prepared-backtest.v1", "as_of": as_of.isoformat(),
              "matched_sample_count": len(common), "expected_sample_count": len(expected),
              "fixture_only": any(f.mode == "fixture" for values in output.values() for f in values),
              "models": metrics, "training": training, "failures": failures,
              "forecast_coverage": {name: sum(len(f.predictions.rows) for f in values) / len(expected)
                                    for name, values in output.items()},
              "protocol": "fixed model per fold; only mature scheduled errors update bias; common evaluation keys"}
    return report, output


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate prepared chronological snapshots")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    folds = [TemporalFold(AsOfSnapshot.model_validate(f["training"]),
             tuple(AsOfSnapshot.model_validate(s) for s in f["validation"])) for f in data["folds"]]
    report, _ = walk_forward_snapshots(folds, [Observation.model_validate(o) for o in data["observations"]],
        as_of=datetime.fromisoformat(data["as_of"].replace("Z", "+00:00")))
    report["input_sha256"] = digest(data)
    atomic_write(Path(args.output), json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode())
    print(json.dumps({"report": args.output, "matched_rows": report["matched_sample_count"],
                      "fixture_only": report["fixture_only"], "failure_count": len(report["failures"])}))
    return 2 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
