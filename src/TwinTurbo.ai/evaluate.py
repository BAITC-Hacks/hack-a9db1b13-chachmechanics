"""Time-ordered validation and honest matched-sample reports; no raw data/network I/O.

Errors in reports use prediction - actual. Bias residuals use actual - p_base.
Only complete matured actual hours are scored. Missing hours are never zeros.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import numpy as np

from .features import LEAD_GROUPS, latest_observations, lead_group
from .schemas import (AsOfSnapshot, EvaluationReport, ForecastResult, ModelState,
                      PredictionBatch, digest, utc)
from .models.bias import Residual, update_bias


def point_metrics(actual, predicted):
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if a.ndim != 1 or a.shape != p.shape or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("INVALID_METRIC_INPUT")
    if not len(a):
        return {"sample_count": 0, "mae": None, "rmse": None, "mean_error": None}
    error = p - a
    return {"sample_count": len(a), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))), "mean_error": float(np.mean(error))}


def pinball_loss(actual, predicted, quantile):
    if not 0 < quantile < 1:
        raise ValueError("INVALID_QUANTILE")
    point_metrics(actual, predicted)
    if len(actual) == 0:
        return None
    error = np.asarray(actual) - np.asarray(predicted)
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def interval_metrics(actual, q10, q50, q90):
    for q in (q10, q50, q90):
        point_metrics(actual, q)
    low, mid, high = map(lambda x: np.asarray(x, dtype=float), (q10, q50, q90))
    if np.any(low > mid) or np.any(mid > high) or np.any(low < 0) or np.any(high > 1):
        raise ValueError("INVALID_QUANTILE_ORDER_OR_BOUNDS")
    if not len(actual):
        return {"sample_count": 0, "coverage": None, "mean_width": None,
                "pinball_q10": None, "pinball_q50": None, "pinball_q90": None, "target_coverage": 0.8}
    a = np.asarray(actual)
    return {"sample_count": len(a), "coverage": float(np.mean((low <= a) & (a <= high))),
            "mean_width": float(np.mean(high - low)), "target_coverage": 0.8,
            "pinball_q10": pinball_loss(a, low, 0.1), "pinball_q50": pinball_loss(a, mid, 0.5),
            "pinball_q90": pinball_loss(a, high, 0.9)}


def _forecast_rows(forecasts):
    result = {}
    for raw in forecasts:
        f = ForecastResult.model_validate(raw.model_dump())
        if f.release_kind != "scheduled":
            continue
        state = ModelState.model_validate(f.manifest["model"])
        if state.model_id != f.model_id or state.activated_at > f.origin_time:
            raise ValueError("INADMISSIBLE_FORECAST_MODEL")
        for row in f.predictions.rows:
            if row.target_start <= f.origin_time:
                raise ValueError("NON_FUTURE_FORECAST_TARGET")
            lead_group((row.target_start - f.origin_time).total_seconds() / 3600)
            key = f.origin_time, row.turbine_id, row.target_start, row.target_end
            value = (f, row)
            if key in result and result[key] != value:
                raise ValueError("DUPLICATE_EVALUATION_KEY: supply one designated scheduled release")
            result[key] = value
    return result


def residuals_from_forecasts(forecasts, observations, *, as_of, base_batches=None):
    """A corrected/clipped forecast cannot be inverted; supply its saved base batch.

    The base batches should be captured by Forecaster.predict_with_trace, then
    persisted by participant 1. Never recompute them with a newer model or run.
    """
    as_of = utc(as_of)
    labels = {(o.turbine_id, o.event_start, o.event_end): o for o in latest_observations(observations, as_of)
              if o.quality_flag == "complete"}
    base_batches = base_batches or {}
    result = []
    for key, (forecast, row) in _forecast_rows(forecasts).items():
        if forecast.origin_time > as_of:
            continue
        obs = labels.get(key[1:])
        if obs is None:
            continue
        state = ModelState.model_validate(forecast.manifest["model"])
        base = base_batches.get(forecast.forecast_id)
        if base is None and forecast.bias_id is not None:
            raise ValueError("BASE_PREDICTIONS_REQUIRED: never subtract bias after clipping")
        if base is not None:
            base = PredictionBatch.model_validate(base.model_dump())
            mapped = {(r.turbine_id, r.target_start, r.target_end): r for r in base.rows}
            expected = {(r.turbine_id, r.target_start, r.target_end) for r in forecast.predictions.rows}
            if len(mapped) != len(base.rows) or set(mapped) != expected:
                raise ValueError("BASE_PREDICTION_COVERAGE")
            p_base = mapped[key[1:]].prediction_norm
        else:
            p_base = row.prediction_norm
        result.append(Residual(forecast_id=forecast.forecast_id, model_id=forecast.model_id,
            turbine_id=row.turbine_id, origin_time=forecast.origin_time,
            target_start=row.target_start, target_end=row.target_end, training_cutoff=state.training_cutoff,
            actual_available_at=obs.available_at, actual_revision=obs.revision,
            actual=obs.power_norm, p_base=p_base, p_issued=row.prediction_norm))
    return tuple(result)


def compare_forecasts(models, observations, *, as_of, expected_keys=None):
    """Score every model on identical (origin,turbine,target) pairs, report omissions.

    expected_keys must come from the planned origins (including failed releases).
    Without it, coverage is relative to the union of supplied releases only.
    """
    if not models:
        raise ValueError("NO_MODELS_TO_COMPARE")
    as_of = utc(as_of)
    maps = {name: {k: v for k, v in _forecast_rows(fs).items() if k[0] <= as_of}
            for name, fs in models.items()}
    universe = set(expected_keys) if expected_keys is not None else set().union(*(set(m) for m in maps.values()))
    if not universe:
        raise ValueError("NO_EXPECTED_FORECAST_ROWS")
    if any(set(m) - universe for m in maps.values()):
        raise ValueError("FORECAST_OUTSIDE_EVALUATION_PLAN")
    labels = {(o.turbine_id, o.event_start, o.event_end): o for o in latest_observations(observations, as_of)
              if o.quality_flag == "complete"}
    scoreable = {k for k in universe if k[1:] in labels}
    common = scoreable.intersection(*(set(m) for m in maps.values()))
    period = min(k[2] for k in universe), max(k[3] for k in universe)
    turbines = sorted({k[1] for k in universe})
    reports, coverage = {}, {}
    origins = {k[0] for k in universe}
    for name, rows in maps.items():
        grouped, intervals = {}, {}
        for turbine in turbines:
            grouped[turbine], intervals[turbine] = {}, {}
            for group in (*LEAD_GROUPS, "all"):
                keys = sorted(k for k in common if k[1] == turbine and
                    (group == "all" or lead_group((k[2]-k[0]).total_seconds()/3600) == group))
                grouped[turbine][group] = point_metrics([labels[k[1:]].power_norm for k in keys],
                                                      [rows[k][1].prediction_norm for k in keys])
                qkeys = [k for k in keys if rows[k][1].q10 is not None]
                intervals[turbine][group] = interval_metrics([labels[k[1:]].power_norm for k in qkeys],
                    *[[getattr(rows[k][1], q) for k in qkeys] for q in ("q10", "q50", "q90")])
                intervals[turbine][group]["interval_availability"] = len(qkeys)/len(keys) if keys else 0.0
        reports[name] = EvaluationReport(period=period, metrics_by_turbine_and_lead=grouped,
            sample_count=len(common), forecast_coverage=len(rows)/len(universe), interval_metrics=intervals)
        complete = sum(all(k in rows for k in universe if k[0] == origin) for origin in origins)
        coverage[name] = {"forecast_rows": len(rows), "expected_rows": len(universe),
            "complete_origins": complete, "expected_origins": len(origins),
            "release_coverage": complete/len(origins), "own_scoreable_rows": len(set(rows) & scoreable),
            "matched_rows": len(common)}
    all_provenance = sorted({f.provenance for rows in maps.values() for f, _ in rows.values()})
    model_provenance = sorted({f.manifest["model"]["provenance"] for rows in maps.values() for f, _ in rows.values()})
    return {"reports": {n: r.model_dump(mode="json") for n, r in reports.items()},
            "coverage": coverage, "matched_sample_count": len(common), "eligible_actual_rows": len(scoreable),
            "missing_or_immature_actual_rows": len(universe)-len(scoreable), "as_of": as_of.isoformat(),
            "coverage_basis": "planned_origins" if expected_keys is not None else "supplied_release_union",
            "weather_provenance": all_provenance, "model_provenance": model_provenance,
            "fixture_only": "synthetic" in all_provenance or "synthetic" in model_provenance,
            "error_sign": "prediction_minus_actual", "selection": "scheduled releases only",
            "comparison_keys_sha256": digest([[x.isoformat() if isinstance(x, datetime) else x for x in k]
                                               for k in sorted(common)])}


@dataclass(frozen=True)
class TemporalFold:
    training: AsOfSnapshot
    validation: tuple[AsOfSnapshot, ...]


def walk_forward(folds, observations, *, as_of, fitters=None, bias_models=("curve",),
                 min_interval_samples=30, interval_fallback=False):
    """Fit once at each fold cutoff; advance origins strictly; labels mature gradually.

    Input snapshots are prepared by participant 1. No weather, CSV, Store, or
    virtual clock is created here. Training/activation is synchronous by default;
    factories can provide a later activated_at to model actual compute latency.
    """
    from .models.baseline import ConstantBaseline
    from .models.power_curve import PowerCurvePredictor
    fitters = fitters or {"baseline": ConstantBaseline.fit, "curve": PowerCurvePredictor.fit}
    if set(bias_models) - set(fitters):
        raise ValueError("UNKNOWN_BIAS_MODEL")
    as_of = utc(as_of)
    output = {n: [] for n in fitters}
    output.update({n+"_bias": [] for n in bias_models})
    traces = {n: {} for n in output}
    states, expected, failures, training = {}, set(), [], []
    previous_origin = None
    for fold in folds:
        fitted, fit_errors = {}, {}
        for name, fit in fitters.items():
            try:
                fitted[name] = fit(fold.training)
            except (ValueError, RuntimeError) as exc:
                fit_errors[name] = str(exc)
        training.append({n: m.state.model_dump(mode="json") for n, m in fitted.items()})
        for snapshot in fold.validation:
            if previous_origin is not None and snapshot.origin_time <= previous_origin:
                raise ValueError("VALIDATION_ORIGINS_MUST_INCREASE")
            if snapshot.origin_time < fold.training.origin_time or snapshot.origin_time > as_of:
                raise ValueError("INVALID_TEMPORAL_FOLD")
            previous_origin = snapshot.origin_time
            turbines = {w.turbine_id for w in snapshot.weather_values}
            expected.update((snapshot.origin_time, t, h.target_start, h.target_end)
                            for t in turbines for h in snapshot.target_intervals)
            for name in output:
                parent = name[:-5] if name.endswith("_bias") and name[:-5] in bias_models else name
                if parent in fit_errors:
                    failures.append({"model": name, "origin": snapshot.origin_time.isoformat(),
                                     "reason": "MODEL_TRAINING_FAILED: " + fit_errors[parent]})
                    continue
                model = fitted[parent]
                try:
                    bias = None
                    if name != parent:
                        errors = residuals_from_forecasts(output[name], observations,
                            as_of=snapshot.origin_time, base_batches=traces[name])
                        bias = update_bias(errors, model_id=model.state.model_id, as_of=snapshot.origin_time,
                            previous=states.get(name), min_interval_samples=min_interval_samples,
                            interval_fallback=interval_fallback)
                        states[name] = bias
                    base = model.predict_base(snapshot)
                    batch = model.predict(snapshot, bias)
                    identity = {"snapshot": digest(snapshot), "model": model.state.model_id,
                                "bias": bias.bias_id if bias else None, "name": name}
                    forecast = ForecastResult(forecast_id=digest(identity), origin_time=snapshot.origin_time,
                        predictions=batch, run_id=snapshot.weather_run_metadata.run_id,
                        model_id=model.state.model_id, bias_id=bias.bias_id if bias else None,
                        provenance=snapshot.weather_run_metadata.provenance,
                        mode="fixture" if model.state.provenance == "synthetic" or
                            snapshot.weather_run_metadata.provenance == "synthetic" else "replay",
                        release_kind="scheduled", warnings=snapshot.quality_flags,
                        manifest={"model": model.state.model_dump(mode="json"), "snapshot_hash": digest(snapshot),
                                  "bias": bias.model_dump(mode="json") if bias else None})
                    output[name].append(forecast)
                    traces[name][forecast.forecast_id] = base
                except (ValueError, RuntimeError) as exc:
                    failures.append({"model": name, "origin": snapshot.origin_time.isoformat(), "reason": str(exc)})
    result = compare_forecasts(output, observations, as_of=as_of, expected_keys=expected)
    result.update(training=training, failures=failures,
        protocol={"split": "expanding/rolling chronological folds supplied by integration layer",
                  "adaptation": "matured scheduled out-of-sample residuals only",
                  "activation_delay": "synchronous unless supplied by model factory",
                  "wind_calibration": "none: measured turbine wind curve applied to supplied grid wind",
                  "interval_min_samples": min_interval_samples, "interval_fallback": interval_fallback})
    return result, output


def write_report(report, path):
    """Write a standalone result supplied by the caller, without loading raw inputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def main(argv=None):
    """Evaluate participant 1's prepared JSON snapshots; never load original CSVs."""
    import argparse
    from .schemas import Observation
    parser = argparse.ArgumentParser(description="TwinTurbo.ai chronological model comparison")
    parser.add_argument("--input", required=True, help="Prepared JSON: folds, observations, as_of")
    parser.add_argument("--output", required=True, help="Evaluation report JSON")
    args = parser.parse_args(argv)
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    folds = tuple(TemporalFold(AsOfSnapshot.model_validate(f["training"]),
        tuple(AsOfSnapshot.model_validate(s) for s in f["validation"])) for f in data["folds"])
    observations = tuple(Observation.model_validate(o) for o in data["observations"])
    report, _ = walk_forward(folds, observations, as_of=datetime.fromisoformat(data["as_of"].replace("Z", "+00:00")))
    report["input_sha256"] = digest(data)
    write_report(report, args.output)
    print(json.dumps({"report": args.output, "matched_rows": report["matched_sample_count"],
                      "fixture_only": report["fixture_only"], "failure_count": len(report["failures"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
