from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import numpy as np

from windoracle.agents.critic import Critic
from windoracle.clock import targets
from windoracle.config import load_config
from windoracle.evaluate import (
    EvaluationPoint,
    compare_models,
    interval_coverage,
    mae,
    mean_interval_width,
    pinball_loss,
)
from windoracle.features import build_training_examples
from windoracle.models.baseline import ConstantBaselinePredictor, fit_turbine_means
from windoracle.models.bias import Residual
from windoracle.models.ml import fit_ridge_predictor
from windoracle.models.power_curve import fit_forecast_power_curve_predictor
from windoracle.models.registry import save_predictor
from windoracle.schemas import (
    AsOfSnapshot,
    BiasState,
    ForecastRequest,
    ModelState,
    PredictionBatch,
    digest,
)
from windoracle.store import Store
from windoracle.weather.audit import audit_bundle
from windoracle.weather.cache import WeatherCache


UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "site.yaml"
REPORT_JSON = ROOT / "reports" / "january-model-comparison.json"
REPORT_MD = ROOT / "reports" / "january-model-comparison.md"
MODEL_DIR = ROOT / "artifacts" / "models"
BIAS_DIR = ROOT / "artifacts" / "bias"

TURBINES = ("turbine_1", "turbine_2")
FINAL_TRAIN_CUTOFF = datetime(2025, 12, 31, 18, tzinfo=UTC)
TUNE_TRAIN_CUTOFF = datetime(2025, 12, 28, 18, tzinfo=UTC)
TUNE_EVALUATION_AS_OF = FINAL_TRAIN_CUTOFF
JAN_START = datetime(2025, 12, 31, 19, tzinfo=UTC)
JAN_SPLIT = datetime(2026, 1, 15, 19, tzinfo=UTC)
JAN_END = datetime(2026, 1, 31, 19, tzinfo=UTC)
JAN_EVALUATION_AS_OF = datetime(2026, 1, 31, 19, 15, tzinfo=UTC)
PRODUCTION_BIAS_AS_OF = datetime(2026, 1, 31, 18, tzinfo=UTC)
BIAS_TUNE_ORIGIN_END = datetime(2026, 1, 13, 18, tzinfo=UTC)
BIAS_TUNE_EVALUATION_AS_OF = datetime(2026, 1, 14, 19, 15, tzinfo=UTC)
TEST_ORIGIN_START = datetime(2026, 1, 15, 18, tzinfo=UTC)


def utc_origins(first: datetime, count: int) -> tuple[datetime, ...]:
    return tuple(first + timedelta(days=index) for index in range(count))


DEC_ORIGINS = utc_origins(datetime(2025, 12, 23, 18, tzinfo=UTC), 8)
JAN_ORIGINS = utc_origins(datetime(2025, 12, 31, 18, tzinfo=UTC), 31)


class MemoryWeatherProvider:
    """Exact in-memory equivalent of GFSArchive.select_run after one cache audit."""

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
                    points.get(turbine_id) != self.expected_points.get(turbine_id)
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
                f"WEATHER_UNAVAILABLE: no admissible run for {request.origin_time.isoformat()}"
            )
        return max(
            candidates,
            key=lambda bundle: (
                bundle.metadata.run_init_time,
                bundle.metadata.available_at,
                bundle.metadata.run_id,
            ),
        )


def make_snapshot(provider: MemoryWeatherProvider, origin: datetime) -> AsOfSnapshot:
    request = ForecastRequest(
        origin_time=origin,
        turbine_ids=TURBINES,
        horizon_hours=48,
        mode="replay",
    )
    bundle = provider.select_run(request)
    intervals = targets(request)
    required = {
        (turbine_id, interval.target_start)
        for turbine_id in request.turbine_ids
        for interval in intervals
    }
    values = tuple(
        value
        for value in bundle.values
        if (value.turbine_id, value.valid_time) in required
    )
    return AsOfSnapshot(
        origin_time=origin,
        observations=(),
        weather_values=values,
        weather_run_metadata=bundle.metadata,
        target_intervals=intervals,
        quality_flags=(),
    )


def actual_index(actuals, as_of: datetime):
    result = {}
    for actual in actuals:
        if (
            actual.available_at <= as_of
            and actual.quality_flag == "complete"
            and actual.power_norm is not None
        ):
            key = (actual.turbine_id, actual.event_start, actual.event_end)
            previous = result.get(key)
            if previous is None or actual.available_at > previous.available_at:
                result[key] = actual
    return result


def points_from_batches(
    snapshots: tuple[AsOfSnapshot, ...],
    batches: tuple[PredictionBatch, ...],
    actuals,
    *,
    evaluation_as_of: datetime,
) -> tuple[EvaluationPoint, ...]:
    facts = actual_index(actuals, evaluation_as_of)
    points = []
    for snapshot, batch in zip(snapshots, batches, strict=True):
        for row in batch.rows:
            actual = facts.get((row.turbine_id, row.target_start, row.target_end))
            points.append(
                EvaluationPoint(
                    turbine_id=row.turbine_id,
                    origin_time=snapshot.origin_time,
                    target_start=row.target_start,
                    target_end=row.target_end,
                    prediction_norm=row.prediction_norm,
                    actual_norm=actual.power_norm if actual else None,
                    actual_available_at=actual.available_at if actual else None,
                    q10=row.q10,
                    q50=row.q50,
                    q90=row.q90,
                    status=row.status,
                )
            )
    return tuple(sorted(points, key=lambda point: point.key))


def period_score(
    points,
    start: datetime,
    end: datetime,
    *,
    origin_start: datetime | None = None,
    origin_end: datetime | None = None,
    evaluation_as_of: datetime | None = None,
) -> dict[str, object]:
    selected = tuple(
        point
        for point in points
        if start <= point.target_start < end
        and (origin_start is None or point.origin_time >= origin_start)
        and (origin_end is None or point.origin_time < origin_end)
        and point.actual_norm is not None
        and (
            evaluation_as_of is None
            or point.actual_available_at <= evaluation_as_of
        )
        and point.prediction_norm is not None
    )
    by_turbine = {}
    for turbine_id in TURBINES:
        pool = tuple(point for point in selected if point.turbine_id == turbine_id)
        by_turbine[turbine_id] = {
            "sample_count": len(pool),
            "mae": mae(
                (point.actual_norm for point in pool),
                (point.prediction_norm for point in pool),
            )
            if pool
            else None,
        }
    interval_points = tuple(point for point in selected if point.has_interval)
    interval = {
        "available_count": len(interval_points),
        "availability": len(interval_points) / len(selected) if selected else 0.0,
        "coverage_q10_q90": interval_coverage(
            (point.actual_norm for point in interval_points),
            (point.q10 for point in interval_points),
            (point.q90 for point in interval_points),
        )
        if interval_points
        else None,
        "mean_width_q10_q90": mean_interval_width(
            (point.q10 for point in interval_points),
            (point.q90 for point in interval_points),
        )
        if interval_points
        else None,
        "q50_mae": mae(
            (point.actual_norm for point in interval_points),
            (point.q50 for point in interval_points),
        )
        if interval_points
        else None,
        "pinball_q10": pinball_loss(
            (point.actual_norm for point in interval_points),
            (point.q10 for point in interval_points),
            0.1,
        )
        if interval_points
        else None,
        "pinball_q50": pinball_loss(
            (point.actual_norm for point in interval_points),
            (point.q50 for point in interval_points),
            0.5,
        )
        if interval_points
        else None,
        "pinball_q90": pinball_loss(
            (point.actual_norm for point in interval_points),
            (point.q90 for point in interval_points),
            0.9,
        )
        if interval_points
        else None,
    }
    return {
        "period": [start.isoformat(), end.isoformat()],
        "sample_count": len(selected),
        "mae": mae(
            (point.actual_norm for point in selected),
            (point.prediction_norm for point in selected),
        )
        if selected
        else None,
        "by_turbine": by_turbine,
        "intervals": interval,
    }


def make_baseline(actuals) -> ConstantBaselinePredictor:
    eligible = tuple(
        actual
        for actual in actuals
        if actual.available_at <= FINAL_TRAIN_CUTOFF
        and actual.quality_flag == "complete"
        and actual.power_norm is not None
    )
    means = fit_turbine_means(
        eligible, turbine_ids=TURBINES, cutoff=FINAL_TRAIN_CUTOFF
    )
    content = {
        "algorithm": "per_turbine_historical_mean_v1",
        "training_cutoff": FINAL_TRAIN_CUTOFF.isoformat(),
        "means": [item.to_dict() for item in means],
    }
    path = MODEL_DIR / "mean-baseline-final-prejan-2026.json"
    state = ModelState(
        model_id="twinturbo-mean-" + digest(content)[:16],
        training_cutoff=FINAL_TRAIN_CUTOFF,
        max_label_available_at=max(actual.available_at for actual in eligible),
        activated_at=FINAL_TRAIN_CUTOFF,
        artifact_ref=str(path.relative_to(ROOT)).replace("\\", "/"),
        provenance="trained",
    )
    predictor = ConstantBaselinePredictor(state=state, means=means)
    save_predictor(predictor, path)
    return predictor


@dataclass(frozen=True)
class BiasSimulation:
    points: tuple[EvaluationPoint, ...]
    final_bias: BiasState | None
    decisions: tuple[dict[str, object], ...]


def simulate_bias(
    predictor,
    snapshots: tuple[AsOfSnapshot, ...],
    actuals,
    *,
    estimator: str,
    shrinkage: float,
    activation_origin: datetime,
    capture_decisions: bool = False,
) -> BiasSimulation:
    facts = actual_index(actuals, JAN_EVALUATION_AS_OF)
    history: list[Residual] = []
    state = None
    batches = []
    decision_log = []
    critic = Critic()
    for snapshot in snapshots:
        base = predictor.predict_base(snapshot)
        decision = None
        if snapshot.origin_time >= activation_origin:
            decision = critic.review_residuals(
                history,
                model_id=predictor.state.model_id,
                as_of=snapshot.origin_time,
                previous=state,
                window_days=21,
                shrinkage=shrinkage,
                estimator=estimator,
                min_interval_samples=30,
                interval_fallback=False,
            )
            if decision.proposed_bias is not None:
                state = decision.proposed_bias
            issued = predictor.predict(snapshot, state)
        else:
            # Before the candidate is activated, the actually issued stream is
            # the uncorrected base.  Its mature errors remain valid evidence
            # when Critic is first activated on the untouched test origin.
            issued = predictor.predict(snapshot, None)
        batches.append(issued)
        forecast_id = "simulation-" + digest(
            {
                "model_id": predictor.state.model_id,
                "origin_time": snapshot.origin_time.isoformat(),
                "estimator": estimator,
                "shrinkage": shrinkage,
            }
        )[:24]
        base_rows = {
            (row.turbine_id, row.target_start, row.target_end): row
            for row in base.rows
        }
        for row in issued.rows:
            key = (row.turbine_id, row.target_start, row.target_end)
            actual = facts.get(key)
            if actual is None:
                continue
            history.append(
                Residual(
                    forecast_id=forecast_id,
                    model_id=predictor.state.model_id,
                    turbine_id=row.turbine_id,
                    origin_time=snapshot.origin_time,
                    target_start=row.target_start,
                    target_end=row.target_end,
                    training_cutoff=predictor.state.training_cutoff,
                    actual_available_at=actual.available_at,
                    actual_revision=actual.revision,
                    actual=actual.power_norm,
                    p_base=base_rows[key].prediction_norm,
                    p_issued=row.prediction_norm,
                )
            )
        if capture_decisions and decision is not None:
            decision_log.append(
                {
                    "origin_time": snapshot.origin_time.isoformat(),
                    "action": decision.action,
                    "reasons": list(decision.reasons),
                    "sample_count": decision.sample_count,
                    "last_actual_available_at": decision.last_actual_available_at,
                    "actual_age_hours": decision.actual_age_hours,
                    "retrain_recommended": decision.retrain_recommended,
                    "bias_id": state.bias_id if state else None,
                }
            )
    final_decision = critic.review_residuals(
        history,
        model_id=predictor.state.model_id,
        as_of=PRODUCTION_BIAS_AS_OF,
        previous=state,
        window_days=21,
        shrinkage=shrinkage,
        estimator=estimator,
        min_interval_samples=30,
        interval_fallback=False,
    )
    if final_decision.proposed_bias is not None:
        state = final_decision.proposed_bias
    if capture_decisions:
        decision_log.append(
            {
                "origin_time": PRODUCTION_BIAS_AS_OF.isoformat(),
                "action": final_decision.action,
                "reasons": list(final_decision.reasons),
                "sample_count": final_decision.sample_count,
                "last_actual_available_at": final_decision.last_actual_available_at,
                "actual_age_hours": final_decision.actual_age_hours,
                "retrain_recommended": final_decision.retrain_recommended,
                "bias_id": state.bias_id if state else None,
                "production_refresh": True,
            }
        )
    return BiasSimulation(
        points=points_from_batches(
            snapshots,
            tuple(batches),
            actuals,
            evaluation_as_of=JAN_EVALUATION_AS_OF,
        ),
        final_bias=state,
        decisions=tuple(decision_log),
    )


def tune_bias(predictor, snapshots, actuals):
    trials = []
    cache = {}
    for estimator in ("median", "mean"):
        for shrinkage in (0.0, 12.0, 24.0, 48.0, 96.0):
            simulation = simulate_bias(
                predictor,
                snapshots,
                actuals,
                estimator=estimator,
                shrinkage=shrinkage,
                activation_origin=JAN_ORIGINS[0],
            )
            calibration = period_score(
                simulation.points,
                JAN_START,
                JAN_SPLIT,
                origin_end=BIAS_TUNE_ORIGIN_END,
                evaluation_as_of=BIAS_TUNE_EVALUATION_AS_OF,
            )
            trial = {
                "estimator": estimator,
                "shrinkage": shrinkage,
                "calibration": calibration,
            }
            trials.append(trial)
            cache[(estimator, shrinkage)] = simulation
    best = min(
        trials,
        key=lambda trial: (
            trial["calibration"]["mae"],
            0 if trial["estimator"] == "median" else 1,
            trial["shrinkage"],
        ),
    )
    selected = simulate_bias(
        predictor,
        snapshots,
        actuals,
        estimator=best["estimator"],
        shrinkage=best["shrinkage"],
        activation_origin=TEST_ORIGIN_START,
        capture_decisions=True,
    )
    return {
        "selection_rule": (
            "lowest overall MAE on Jan 1-15 targets; median wins exact estimator ties; "
            "smaller shrinkage wins remaining ties"
        ),
        "selected": {
            "estimator": best["estimator"],
            "shrinkage": best["shrinkage"],
            "calibration": best["calibration"],
            "test": period_score(
                selected.points,
                JAN_SPLIT,
                JAN_END,
                origin_start=TEST_ORIGIN_START,
                evaluation_as_of=JAN_EVALUATION_AS_OF,
            ),
        },
        "trials": trials,
    }, selected


def hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "Reproduce TwinTurbo.ai's leakage-safe January model selection "
            "from the imported observation store and cached GFS runs."
        )
    )
    command.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Site configuration (default: configs/site.yaml).",
    )
    return command


def main(config_path: Path = CONFIG_PATH) -> None:
    config = load_config(config_path)
    store = Store(ROOT / config.storage.database)
    actuals = store.observations_as_of(JAN_EVALUATION_AS_OF, TURBINES)
    cached_bundles = tuple(WeatherCache(ROOT / config.weather.cache_dir).bundles())
    provider = MemoryWeatherProvider(config, cached_bundles)
    dec_snapshots = tuple(make_snapshot(provider, origin) for origin in DEC_ORIGINS)
    jan_snapshots = tuple(make_snapshot(provider, origin) for origin in JAN_ORIGINS)
    selected_run_ids = {
        snapshot.weather_run_metadata.run_id
        for snapshot in (*dec_snapshots, *jan_snapshots)
    }
    bundle_by_id = {
        bundle.metadata.run_id: bundle for bundle in cached_bundles
    }
    missing_selected_runs = selected_run_ids - set(bundle_by_id)
    if missing_selected_runs:
        raise RuntimeError(
            "SELECTED_WEATHER_RUN_MISSING_FROM_CACHE: "
            + ",".join(sorted(missing_selected_runs))
        )
    bundles = tuple(
        sorted(
            (bundle_by_id[run_id] for run_id in selected_run_ids),
            key=lambda item: item.metadata.run_init_time,
        )
    )
    expected_run_count = len(DEC_ORIGINS) + len(JAN_ORIGINS)
    if len(bundles) != expected_run_count or any(
        bundle.metadata.provenance != "operational_archive" for bundle in bundles
    ):
        raise RuntimeError(
            "JANUARY_WEATHER_PROVENANCE_INCOMPLETE: expected "
            f"{expected_run_count} unique operational runs, selected {len(bundles)}"
        )
    examples = build_training_examples(
        dec_snapshots, actuals, label_as_of=FINAL_TRAIN_CUTOFF
    )

    validation_snapshots = tuple(
        snapshot
        for snapshot in dec_snapshots
        if snapshot.origin_time >= TUNE_TRAIN_CUTOFF
    )
    curve_trials = []
    for bin_width in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        for min_samples in (3, 6, 12, 24):
            try:
                predictor = fit_forecast_power_curve_predictor(
                    examples,
                    training_cutoff=TUNE_TRAIN_CUTOFF,
                    activated_at=TUNE_TRAIN_CUTOFF,
                    artifact_ref="candidate://forecast-power-curve",
                    turbine_ids=TURBINES,
                    bin_width=bin_width,
                    min_samples_per_bin=min_samples,
                )
                batches = tuple(
                    predictor.predict_base(snapshot)
                    for snapshot in validation_snapshots
                )
                points = points_from_batches(
                    validation_snapshots,
                    batches,
                    actuals,
                    evaluation_as_of=TUNE_EVALUATION_AS_OF,
                )
                score = period_score(
                    points,
                    validation_snapshots[0].target_intervals[0].target_start,
                    datetime(2026, 1, 1, 19, tzinfo=UTC),
                )
                curve_trials.append(
                    {
                        "bin_width": bin_width,
                        "min_samples_per_bin": min_samples,
                        "validation": score,
                        "status": "ok",
                    }
                )
            except ValueError as exc:
                curve_trials.append(
                    {
                        "bin_width": bin_width,
                        "min_samples_per_bin": min_samples,
                        "validation": None,
                        "status": "rejected",
                        "reason": str(exc),
                    }
                )
    valid_curve_trials = [
        trial for trial in curve_trials if trial["status"] == "ok"
    ]
    best_curve = min(
        valid_curve_trials,
        key=lambda trial: (
            trial["validation"]["mae"],
            trial["bin_width"],
            trial["min_samples_per_bin"],
        ),
    )
    curve_path = MODEL_DIR / "forecast-power-curve-prejan-2026.json"
    curve = fit_forecast_power_curve_predictor(
        examples,
        training_cutoff=FINAL_TRAIN_CUTOFF,
        activated_at=FINAL_TRAIN_CUTOFF,
        artifact_ref=str(curve_path.relative_to(ROOT)).replace("\\", "/"),
        turbine_ids=TURBINES,
        bin_width=best_curve["bin_width"],
        min_samples_per_bin=best_curve["min_samples_per_bin"],
    )
    save_predictor(curve, curve_path)

    ridge_trials = []
    for alpha in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        predictor = fit_ridge_predictor(
            examples,
            training_cutoff=TUNE_TRAIN_CUTOFF,
            activated_at=TUNE_TRAIN_CUTOFF,
            artifact_ref="candidate://ridge",
            turbine_ids=TURBINES,
            alpha=alpha,
            min_samples_per_turbine=24,
        )
        batches = tuple(
            predictor.predict_base(snapshot) for snapshot in validation_snapshots
        )
        points = points_from_batches(
            validation_snapshots,
            batches,
            actuals,
            evaluation_as_of=TUNE_EVALUATION_AS_OF,
        )
        ridge_trials.append(
            {
                "alpha": alpha,
                "validation": period_score(
                    points,
                    validation_snapshots[0].target_intervals[0].target_start,
                    datetime(2026, 1, 1, 19, tzinfo=UTC),
                ),
            }
        )
    best_ridge = min(
        ridge_trials,
        key=lambda trial: (trial["validation"]["mae"], trial["alpha"]),
    )
    ridge_path = MODEL_DIR / "ridge-prejan-2026.json"
    ridge = fit_ridge_predictor(
        examples,
        training_cutoff=FINAL_TRAIN_CUTOFF,
        activated_at=FINAL_TRAIN_CUTOFF,
        artifact_ref=str(ridge_path.relative_to(ROOT)).replace("\\", "/"),
        turbine_ids=TURBINES,
        alpha=best_ridge["alpha"],
        min_samples_per_turbine=24,
    )
    save_predictor(ridge, ridge_path)

    baseline = make_baseline(actuals)
    models = {"mean_baseline": baseline, "power_curve": curve, "ridge": ridge}
    base_points = {}
    for name, predictor in models.items():
        batches = tuple(
            predictor.predict_base(snapshot) for snapshot in jan_snapshots
        )
        base_points[name] = points_from_batches(
            jan_snapshots,
            batches,
            actuals,
            evaluation_as_of=JAN_EVALUATION_AS_OF,
        )

    expected_keys = tuple(point.key for point in base_points["mean_baseline"])
    comparisons = {
        "mean_vs_power_curve": compare_models(
            base_points["mean_baseline"],
            base_points["power_curve"],
            expected_keys=expected_keys,
            period=(JAN_START, JAN_END),
            evaluation_as_of=JAN_EVALUATION_AS_OF,
            min_selection_samples=2928,
        ).to_dict(),
        "power_curve_vs_ridge": compare_models(
            base_points["power_curve"],
            base_points["ridge"],
            expected_keys=expected_keys,
            period=(JAN_START, JAN_END),
            evaluation_as_of=JAN_EVALUATION_AS_OF,
            min_selection_samples=2928,
        ).to_dict(),
    }

    base_scores = {
        name: {
            "full_january": period_score(
                points,
                JAN_START,
                JAN_END,
                evaluation_as_of=JAN_EVALUATION_AS_OF,
            ),
            "bias_tuning_period": period_score(
                points,
                JAN_START,
                JAN_SPLIT,
                origin_end=BIAS_TUNE_ORIGIN_END,
                evaluation_as_of=BIAS_TUNE_EVALUATION_AS_OF,
            ),
            "untouched_selection_period": period_score(
                points,
                JAN_SPLIT,
                JAN_END,
                origin_start=TEST_ORIGIN_START,
                evaluation_as_of=JAN_EVALUATION_AS_OF,
            ),
        }
        for name, points in base_points.items()
    }

    bias_tuning = {}
    bias_simulations = {}
    for name, predictor in models.items():
        tuning, simulation = tune_bias(predictor, jan_snapshots, actuals)
        bias_tuning[name] = tuning
        bias_simulations[name] = simulation

    # Complexity is gated in the requested order. Each extra component must
    # beat the current simpler choice on the untouched Jan 16-31 targets.
    selection_log = []
    selected_name = "mean_baseline"
    selected_predictor = baseline
    selected_points = base_points[selected_name]
    selected_bias = None
    selected_mae = base_scores[selected_name]["untouched_selection_period"]["mae"]

    curve_mae = base_scores["power_curve"]["untouched_selection_period"]["mae"]
    curve_enabled = curve_mae < selected_mae
    selection_log.append(
        {
            "candidate": "power_curve",
            "reference": selected_name,
            "candidate_mae": curve_mae,
            "reference_mae": selected_mae,
            "enabled": curve_enabled,
            "rule": "strictly lower overall MAE; simpler model wins ties",
        }
    )
    if curve_enabled:
        selected_name = "power_curve"
        selected_predictor = curve
        selected_points = base_points[selected_name]
        selected_mae = curve_mae

    for name in ("mean_baseline", "power_curve"):
        corrected_mae = bias_tuning[name]["selected"]["test"]["mae"]
        base_mae = base_scores[name]["untouched_selection_period"]["mae"]
        improves_own_base = corrected_mae < base_mae
        improves_current = corrected_mae < selected_mae
        enabled = improves_own_base and improves_current
        selection_log.append(
            {
                "candidate": name + "+bias",
                "reference": name,
                "candidate_mae": corrected_mae,
                "reference_mae": base_mae,
                "improves_own_base": improves_own_base,
                "improves_current_selection": improves_current,
                "enabled": enabled,
                "rule": "bias must lower held-out MAE of its own base and current selection",
            }
        )
        if enabled:
            selected_name = name + "+bias"
            selected_predictor = models[name]
            selected_points = bias_simulations[name].points
            selected_bias = bias_simulations[name].final_bias
            selected_mae = corrected_mae

    ridge_mae = base_scores["ridge"]["untouched_selection_period"]["mae"]
    ridge_enabled = ridge_mae < selected_mae
    selection_log.append(
        {
            "candidate": "ridge",
            "reference": selected_name,
            "candidate_mae": ridge_mae,
            "reference_mae": selected_mae,
            "enabled": ridge_enabled,
            "rule": "Ridge must strictly lower untouched overall MAE",
        }
    )
    if ridge_enabled:
        selected_name = "ridge"
        selected_predictor = ridge
        selected_points = base_points["ridge"]
        selected_bias = None
        selected_mae = ridge_mae

    ridge_bias_mae = bias_tuning["ridge"]["selected"]["test"]["mae"]
    ridge_bias_improves_base = ridge_bias_mae < ridge_mae
    ridge_bias_enabled = ridge_bias_improves_base and ridge_bias_mae < selected_mae
    selection_log.append(
        {
            "candidate": "ridge+bias",
            "reference": "ridge",
            "candidate_mae": ridge_bias_mae,
            "reference_mae": ridge_mae,
            "improves_own_base": ridge_bias_improves_base,
            "improves_current_selection": ridge_bias_mae < selected_mae,
            "enabled": ridge_bias_enabled,
            "rule": "Ridge bias must lower held-out Ridge MAE and current selection",
        }
    )
    if ridge_bias_enabled:
        selected_name = "ridge+bias"
        selected_predictor = ridge
        selected_points = bias_simulations["ridge"].points
        selected_bias = bias_simulations["ridge"].final_bias
        selected_mae = ridge_bias_mae

    production_model_path = MODEL_DIR / "february-production.json"
    save_predictor(selected_predictor, production_model_path)
    production_bias_path = None
    if selected_bias is not None:
        BIAS_DIR.mkdir(parents=True, exist_ok=True)
        production_bias_path = BIAS_DIR / "february-production-bias.json"
        production_bias_path.write_text(
            selected_bias.model_dump_json(indent=2), encoding="utf-8"
        )

    selected_base_name = selected_name.removesuffix("+bias")
    selected_simulation = (
        bias_simulations.get(selected_base_name) if selected_bias is not None else None
    )
    interval_score = period_score(
        selected_points,
        JAN_SPLIT,
        JAN_END,
        origin_start=TEST_ORIGIN_START,
        evaluation_as_of=JAN_EVALUATION_AS_OF,
    )["intervals"]

    code_files = (
        Path(__file__).resolve(),
        ROOT / "src" / "TwinTurbo.ai" / "features.py",
        ROOT / "src" / "TwinTurbo.ai" / "evaluate.py",
        ROOT / "src" / "TwinTurbo.ai" / "models" / "baseline.py",
        ROOT / "src" / "TwinTurbo.ai" / "models" / "power_curve.py",
        ROOT / "src" / "TwinTurbo.ai" / "models" / "ml.py",
        ROOT / "src" / "TwinTurbo.ai" / "models" / "bias.py",
        ROOT / "src" / "TwinTurbo.ai" / "models" / "intervals.py",
        ROOT / "src" / "TwinTurbo.ai" / "agents" / "critic.py",
    )
    report = {
        "schema": "twinturbo.january-model-selection.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "objective": {
            "primary": "minimize overall MAE",
            "secondary": "report peak-hour MAE without overriding primary selection",
            "peak_definition": "actual normalized power >= 0.8",
            "forecast_horizon_hours": 48,
            "quantiles": [0.1, 0.5, 0.9],
        },
        "as_of_contract": {
            "source_timezone": config.site.timezone,
            "fixed_utc_offset": "+05:00",
            "issue_local_time": config.forecast.issue_local_time,
            "issue_utc_time": "18:00",
            "observation_delay_minutes": config.site.observation_delay_minutes,
            "weather_publication_delay_hours": config.weather.publication_delay_hours,
            "config_hash": config.config_hash,
        },
        "data": {
            "actual_rows_visible_at_evaluation": len(actuals),
            "training_example_count": len(examples),
            "training_cutoff": FINAL_TRAIN_CUTOFF.isoformat(),
            "tuning_training_cutoff": TUNE_TRAIN_CUTOFF.isoformat(),
            "january_evaluation_as_of": JAN_EVALUATION_AS_OF.isoformat(),
            "january_period": [JAN_START.isoformat(), JAN_END.isoformat()],
            "bias_tuning_period": [JAN_START.isoformat(), JAN_SPLIT.isoformat()],
            "untouched_selection_period": [JAN_SPLIT.isoformat(), JAN_END.isoformat()],
        },
        "weather_archive": {
            "provider": config.weather.provider,
            "run_count": len(bundles),
            "network_payload_bytes": sum(
                int(bundle.metadata.evidence.get("download_bytes", 0))
                for bundle in bundles
            ),
            "runs": [
                {
                    "run_id": bundle.metadata.run_id,
                    "run_init_time": bundle.metadata.run_init_time.isoformat(),
                    "available_at": bundle.metadata.available_at.isoformat(),
                    "sha256": bundle.metadata.sha256,
                    "value_count": len(bundle.values),
                    "provenance": bundle.metadata.provenance,
                    "availability_basis": bundle.metadata.availability_basis,
                }
                for bundle in sorted(
                    bundles, key=lambda item: item.metadata.run_init_time
                )
            ],
        },
        "tuning": {
            "power_curve": {
                "selection_rule": "lowest validation MAE before January; narrower bin then lower min-samples wins ties",
                "selected": best_curve,
                "trials": curve_trials,
            },
            "ridge": {
                "selection_rule": "lowest validation MAE before January; lower alpha wins ties",
                "selected": best_ridge,
                "trials": ridge_trials,
            },
            "bias": bias_tuning,
        },
        "model_states": {
            name: predictor.state.model_dump(mode="json")
            for name, predictor in models.items()
        },
        "base_scores": base_scores,
        "comparisons": comparisons,
        "selection": {
            "rule": "ordered complexity gates on untouched Jan 16-31 target MAE; strict improvement required",
            "steps": selection_log,
            "selected": selected_name,
            "selected_model_id": selected_predictor.state.model_id,
            "selected_bias_id": selected_bias.bias_id if selected_bias else None,
            "selected_test_mae": selected_mae,
            "selected_test_metrics": period_score(
                selected_points,
                JAN_SPLIT,
                JAN_END,
                origin_start=TEST_ORIGIN_START,
                evaluation_as_of=JAN_EVALUATION_AS_OF,
            ),
            "ensemble": {
                "enabled": False,
                "reason": "not attempted after a single validated winner; avoids an extra uncalibrated degree of freedom",
            },
        },
        "uncertainty": {
            "test_metrics": interval_score,
            "policy": "null quantiles and explicit interval:insufficient_history until each turbine/lead group has 30 mature residuals",
        },
        "critic": {
            "selected_model_decisions": list(
                selected_simulation.decisions if selected_simulation else ()
            ),
            "final_bias_state": selected_bias.model_dump(mode="json")
            if selected_bias
            else None,
        },
        "artifacts": {
            "baseline": str(
                (MODEL_DIR / "mean-baseline-final-prejan-2026.json").relative_to(ROOT)
            ).replace("\\", "/"),
            "power_curve": str(curve_path.relative_to(ROOT)).replace("\\", "/"),
            "ridge_candidate": str(ridge_path.relative_to(ROOT)).replace("\\", "/"),
            "production_model": str(production_model_path.relative_to(ROOT)).replace("\\", "/"),
            "production_bias": str(production_bias_path.relative_to(ROOT)).replace("\\", "/")
            if production_bias_path
            else None,
        },
        "reproducibility": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "code_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"): hash_file(path)
                for path in code_files
            },
        },
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    full_scores = {
        name: values["full_january"]["mae"]
        for name, values in base_scores.items()
    }
    lines = [
        "# TwinTurbo.ai — последовательная январская проверка",
        "",
        "Главный критерий выбора: минимальный общий MAE. Пиковый MAE — только диагностический показатель.",
        "",
        f"- Часовой пояс исходных меток: фиксированный UTC+05 (`{config.site.timezone}`).",
        f"- Архив: {len(bundles)} реальных выпуска GFS, {report['weather_archive']['network_payload_bytes']:,} байт полезной нагрузки.",
        f"- Контрольный январский набор: {base_scores['mean_baseline']['full_january']['sample_count']} forecast-target строк.",
        f"- Mean baseline MAE: {full_scores['mean_baseline']:.6f}.",
        f"- Отдельная GFS power curve MAE: {full_scores['power_curve']:.6f}.",
        f"- Ridge MAE: {full_scores['ridge']:.6f}.",
        f"- Выбранный вариант на нетронутом 16–31 января: **{selected_name}**, MAE {selected_mae:.6f}.",
        "",
        "## Неопределённость",
        "",
        f"- Доля строк с Q10/Q50/Q90: {interval_score['availability']:.3%}.",
        f"- Покрытие Q10–Q90: {interval_score['coverage_q10_q90'] if interval_score['coverage_q10_q90'] is not None else 'недостаточно истории'}.",
        f"- Средняя ширина Q10–Q90: {interval_score['mean_width_q10_q90'] if interval_score['mean_width_q10_q90'] is not None else 'недостаточно истории'}.",
        "- До накопления 30 зрелых ошибок в конкретной паре «турбина × группа горизонта» квантили равны null, а status содержит `interval:insufficient_history`.",
        "",
        "## Воспроизводимость",
        "",
        "Полный JSON содержит временные границы, все гиперпараметры, результаты каждого кандидата, GFS run_id/хэши, состояния Critic и SHA-256 кода.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(REPORT_JSON),
        "selected": selected_name,
        "selected_test_mae": selected_mae,
        "full_january_mae": full_scores,
        "intervals": interval_score,
        "production_model": str(production_model_path),
        "production_bias": str(production_bias_path) if production_bias_path else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(parser().parse_args().config)
