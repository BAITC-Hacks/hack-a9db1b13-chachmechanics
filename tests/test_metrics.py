from datetime import datetime, timedelta, timezone
from math import sqrt

import pytest

from windoracle.evaluate import (
    EvaluationPoint,
    ForecastKey,
    align_forecasts,
    bias,
    compare_models,
    evaluate,
    interval_coverage,
    mae,
    mean_interval_width,
    pinball_loss,
    residuals_from_forecasts,
    rmse,
    walk_forward,
    walk_forward_folds,
)
from windoracle.schemas import (
    ForecastResult,
    ModelState,
    Observation,
    PredictionBatch,
    PredictionRow,
)


UTC = timezone.utc
ORIGIN = datetime(2026, 1, 1, tzinfo=UTC)


def point(*, lead=1, turbine="turbine_1", prediction=.4, actual=.5,
          available=None, q10=None, q50=None, q90=None):
    target = ORIGIN + timedelta(hours=lead)
    end = target + timedelta(hours=1)
    return EvaluationPoint(
        turbine_id=turbine,
        origin_time=ORIGIN,
        target_start=target,
        target_end=end,
        prediction_norm=prediction,
        actual_norm=actual,
        actual_available_at=available or end + timedelta(minutes=15),
        q10=q10,
        q50=q50,
        q90=q90,
    )


def test_deterministic_point_metrics_and_explicit_bias_direction():
    actual = [0.0, 1.0, .5]
    prediction = [.1, .8, .4]

    assert mae(actual, prediction) == pytest.approx(2 / 15)
    assert rmse(actual, prediction) == pytest.approx(sqrt(.02))
    # Forecast bias is prediction - actual.  Correction residuals use its negative.
    assert bias(actual, prediction) == pytest.approx(-1 / 15)
    assert mae(iter(actual), iter(prediction)) == pytest.approx(2 / 15)


def test_pinball_and_interval_metrics_have_standard_definitions():
    assert pinball_loss([0.0, 1.0], [.2, .8], .1) == pytest.approx(.1)
    assert pinball_loss([0.0, 1.0], [.2, .8], .9) == pytest.approx(.1)
    assert interval_coverage([0.0, 1.0, .5], [0.0, .2, .6], [.2, 1.0, .8]) == pytest.approx(2 / 3)
    assert mean_interval_width([0.0, .2, .6], [.2, 1.0, .8]) == pytest.approx(.4)


def test_metric_inputs_reject_empty_nonfinite_and_misaligned_values():
    with pytest.raises(ValueError, match="must not be empty"):
        mae([], [])
    with pytest.raises(ValueError, match="equal lengths"):
        rmse([0], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        bias([0], [float("nan")])
    with pytest.raises(ValueError, match="strictly between"):
        pinball_loss([0], [0], 1)
    with pytest.raises(ValueError, match="lower bound"):
        interval_coverage([.5], [.8], [.2])


def test_evaluation_report_groups_leads_and_uses_planned_coverage_denominator():
    points = (
        point(lead=1, prediction=.2, actual=.1, q10=0, q50=.15, q90=.3),
        point(lead=7, prediction=.6, actual=.8, q10=.5, q50=.7, q90=.9),
    )
    expected = tuple(item.key for item in points) + (
        ForecastKey(ORIGIN, ORIGIN + timedelta(hours=8), "turbine_1"),
    )

    report = evaluate(points, expected_keys=expected)

    assert report.sample_count == 2
    assert report.forecast_coverage == pytest.approx(2 / 3)
    grouped = report.metrics_by_turbine_and_lead["turbine_1"]
    assert grouped["all"] == pytest.approx({
        "count": 2, "mae": .15, "rmse": sqrt(.025), "bias": -.05,
    })
    assert grouped["lead_01_06"]["count"] == 1
    assert grouped["lead_07_12"]["count"] == 1
    assert report.interval_metrics["target_coverage_q10_q90"] == .8
    assert report.interval_metrics["coverage_q10_q90"] == 1.0
    assert report.interval_metrics["mean_width_q10_q90"] == pytest.approx(.35)
    assert report.period == (ORIGIN + timedelta(hours=1), ORIGIN + timedelta(hours=9))


def test_evaluation_as_of_excludes_unavailable_actual_but_not_forecast_coverage():
    early = point(lead=1)
    late = point(lead=2, available=ORIGIN + timedelta(days=2))
    cutoff = ORIGIN + timedelta(hours=4)

    report = evaluate((early, late), evaluation_as_of=cutoff)

    assert report.sample_count == 1
    assert report.forecast_coverage == 1.0
    assert report.metrics_by_turbine_and_lead["turbine_1"]["all"]["count"] == 1


def test_evaluation_point_enforces_hourly_and_actual_availability_contracts():
    with pytest.raises(ValueError, match="actual_available_at"):
        EvaluationPoint("t", ORIGIN, ORIGIN + timedelta(hours=1), .2, .3, None)
    with pytest.raises(ValueError, match="before target_end"):
        point(available=ORIGIN + timedelta(hours=1, minutes=30))
    with pytest.raises(ValueError, match="all present"):
        point(q10=.1)
    with pytest.raises(ValueError, match="q10 <= q50 <= q90"):
        point(q10=.8, q50=.5, q90=.9)


def test_models_are_scored_only_on_common_keys_but_keep_separate_coverage():
    base_a = point(lead=1, prediction=.2)
    base_b = point(lead=2, prediction=.3)
    candidate_b = point(lead=2, prediction=.4)
    candidate_c = point(lead=3, prediction=.5)

    aligned_base, aligned_candidate = align_forecasts(
        (base_a, base_b), (candidate_b, candidate_c))
    assert tuple(value.key for value in aligned_base) == (base_b.key,)
    assert tuple(value.key for value in aligned_candidate) == (candidate_b.key,)

    expected = (base_a.key, base_b.key, candidate_c.key)
    comparison = compare_models(
        (base_a, base_b), (candidate_b, candidate_c), expected_keys=expected)
    assert comparison.common_keys == (base_b.key,)
    assert comparison.baseline.sample_count == comparison.candidate.sample_count == 1
    assert comparison.baseline.forecast_coverage == pytest.approx(2 / 3)
    assert comparison.candidate.forecast_coverage == pytest.approx(2 / 3)
    assert comparison.baseline.metrics_by_turbine_and_lead["turbine_1"]["all"]["mae"] == pytest.approx(.2)
    assert comparison.candidate.metrics_by_turbine_and_lead["turbine_1"]["all"]["mae"] == pytest.approx(.1)
    document = comparison.to_dict()
    assert document["schema"] == "twinturbo.model-comparison.v1"
    assert len(document["expected_keys"]) == 3
    assert document["lead_groups"][0] == ["lead_01_06", 1, 6]
    assert document["selection"]["selected_model"] == "candidate"
    assert document["selection"]["overall"] == pytest.approx({
        "sample_count": 1,
        "baseline_mae": .2,
        "candidate_mae": .1,
        "delta_candidate_minus_baseline": -.1,
        "skill_vs_baseline": .5,
    })
    assert document["selection"]["peak_period"]["sample_count"] == 0
    assert document["selection"]["peak_period"]["baseline_mae"] is None
    assert comparison.to_json() == comparison.to_json()
    assert '"baseline"' in comparison.to_json()


def test_model_selection_prioritizes_overall_mae_and_reports_peak_mae():
    actuals = (.1, .2, .3, .9)
    baseline_predictions = (.3, .4, .5, .85)
    candidate_predictions = (.1, .2, .3, .6)
    baseline = tuple(
        point(
            lead=index,
            turbine="turbine_1" if index % 2 else "turbine_2",
            prediction=prediction,
            actual=actual,
        )
        for index, (prediction, actual) in enumerate(
            zip(baseline_predictions, actuals), start=1
        )
    )
    candidate = tuple(
        point(
            lead=index,
            turbine="turbine_1" if index % 2 else "turbine_2",
            prediction=prediction,
            actual=actual,
        )
        for index, (prediction, actual) in enumerate(
            zip(candidate_predictions, actuals), start=1
        )
    )

    comparison = compare_models(
        baseline,
        candidate,
        peak_actual_threshold=.8,
    )
    selection = comparison.selection

    assert selection.selected_model == "candidate"
    assert selection.overall.sample_count == 4
    assert selection.overall.baseline_mae == pytest.approx(.1625)
    assert selection.overall.candidate_mae == pytest.approx(.075)
    assert selection.overall.delta_candidate_minus_baseline == pytest.approx(-.0875)
    assert selection.overall.skill_vs_baseline == pytest.approx(7 / 13)
    # The candidate wins the user's primary overall-MAE objective even though
    # it is worse on the separately disclosed high-output hour.
    assert selection.peak_period.sample_count == 1
    assert selection.peak_period.baseline_mae == pytest.approx(.05)
    assert selection.peak_period.candidate_mae == pytest.approx(.3)
    assert selection.peak_period.delta_candidate_minus_baseline == pytest.approx(.25)
    assert selection.peak_period.skill_vs_baseline == pytest.approx(-5)
    assert comparison.to_dict()["selection"]["peak_definition"] == (
        "actual_norm >= peak_actual_threshold"
    )


def test_model_selection_uses_only_mature_pairs_and_baseline_wins_ties():
    mature = point(lead=1, prediction=.5, actual=.5)
    late = point(
        lead=2,
        prediction=.4,
        actual=.5,
        available=ORIGIN + timedelta(days=2),
    )
    cutoff = ORIGIN + timedelta(hours=4)
    comparison = compare_models(
        (mature, late),
        (
            point(lead=1, prediction=.5, actual=.5),
            point(
                lead=2,
                prediction=.5,
                actual=.5,
                available=ORIGIN + timedelta(days=2),
            ),
        ),
        evaluation_as_of=cutoff,
    )

    assert comparison.selection.overall.sample_count == 1
    assert comparison.selection.overall.baseline_mae == 0
    assert comparison.selection.overall.candidate_mae == 0
    assert comparison.selection.overall.skill_vs_baseline is None
    assert comparison.selection.selected_model == "baseline"

    no_mature_samples = compare_models(
        (late,),
        (late,),
        evaluation_as_of=cutoff,
    )
    assert no_mature_samples.selection.overall.sample_count == 0
    assert no_mature_samples.selection.selected_model == "baseline"
    assert no_mature_samples.selection.sample_count_gate_passed is False

    with pytest.raises(ValueError, match="peak_actual_threshold"):
        compare_models((mature,), (mature,), peak_actual_threshold=1.1)


def test_model_selection_rejects_better_mae_from_lower_forecast_coverage():
    baseline = tuple(
        point(lead=lead, prediction=.8, actual=.5)
        for lead in range(1, 5)
    )
    # Perfect on its one forecast, but absent for three of the four planned
    # keys.  Comparing only the common row must not reward that sparsity.
    candidate = (point(lead=1, prediction=.5, actual=.5),)

    comparison = compare_models(baseline, candidate)
    selection = comparison.selection
    gates = comparison.to_dict()["selection"]["gates"]

    assert comparison.baseline.forecast_coverage == 1.0
    assert comparison.candidate.forecast_coverage == .25
    assert selection.overall.baseline_mae == pytest.approx(.3)
    assert selection.overall.candidate_mae == 0
    assert selection.selected_model == "baseline"
    assert gates["coverage"] == {
        "passed": False,
        "baseline_forecast_coverage": 1.0,
        "candidate_forecast_coverage": .25,
        "requirement": "candidate >= baseline",
    }
    assert gates["sample_count"]["passed"] is True
    assert gates["mae_improvement"]["passed"] is True


def test_model_selection_requires_minimum_samples_and_mae_beyond_tolerance():
    baseline = tuple(
        point(lead=lead, prediction=.75, actual=.5)
        for lead in range(1, 5)
    )
    candidate = tuple(
        point(lead=lead, prediction=.625, actual=.5)
        for lead in range(1, 5)
    )

    too_few = compare_models(
        baseline,
        candidate,
        min_selection_samples=5,
    )
    assert too_few.selection.selected_model == "baseline"
    assert too_few.selection.sample_count_gate_passed is False
    assert too_few.to_dict()["selection"]["gates"]["sample_count"] == {
        "passed": False,
        "actual": 4,
        "minimum": 5,
    }

    exact_threshold = compare_models(
        baseline,
        candidate,
        mae_tolerance=.125,
    )
    assert exact_threshold.selection.mae_improvement == pytest.approx(.125)
    assert exact_threshold.selection.mae_gate_passed is False
    assert exact_threshold.selection.selected_model == "baseline"

    beyond_threshold = compare_models(
        baseline,
        candidate,
        mae_tolerance=.124,
    )
    assert beyond_threshold.selection.mae_gate_passed is True
    assert beyond_threshold.selection.selected_model == "candidate"


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    (
        ({"min_selection_samples": 0}, ValueError, "at least 1"),
        ({"min_selection_samples": True}, TypeError, "integer"),
        ({"min_selection_samples": 1.5}, TypeError, "integer"),
        ({"mae_tolerance": -0.01}, ValueError, "nonnegative"),
        ({"mae_tolerance": float("nan")}, ValueError, "finite"),
    ),
)
def test_model_selection_gate_arguments_are_validated(kwargs, error, message):
    sample = point(lead=1)
    with pytest.raises(error, match=message):
        compare_models((sample,), (sample,), **kwargs)


def test_alignment_rejects_different_actual_revisions_for_same_key():
    baseline = point(lead=1, actual=.4)
    candidate = point(lead=1, actual=.5)
    with pytest.raises(ValueError, match="actual mismatch"):
        align_forecasts((baseline,), (candidate,))


def test_residual_join_uses_saved_base_batch_and_only_mature_actuals():
    target = ORIGIN + timedelta(hours=1)
    end = target + timedelta(hours=1)
    model = ModelState(
        model_id="model-1",
        training_cutoff=ORIGIN - timedelta(days=2),
        max_label_available_at=ORIGIN - timedelta(days=2),
        activated_at=ORIGIN - timedelta(days=1),
        artifact_ref="test-only",
        provenance="synthetic",
    )
    issued = PredictionRow(
        turbine_id="turbine_1", target_start=target, target_end=end,
        prediction_norm=.5,
    )
    forecast = ForecastResult(
        forecast_id="forecast-1",
        origin_time=ORIGIN,
        predictions=PredictionBatch(rows=(issued,)),
        run_id="run-1",
        model_id=model.model_id,
        bias_id="bias-1",
        provenance="synthetic",
        mode="fixture",
        release_kind="scheduled",
        manifest={"model": model.model_dump(mode="json")},
    )
    actual = Observation(
        turbine_id="turbine_1",
        event_start=target,
        event_end=end,
        available_at=end + timedelta(minutes=15),
        power_norm=.7,
        wind_ms=8,
        temperature_c=2,
        n_samples=6,
        coverage=1,
        quality_flag="complete",
        revision="actual-v1",
    )
    base = PredictionBatch(rows=(issued.model_copy(update={"prediction_norm": .3}),))

    with pytest.raises(ValueError, match="BASE_BATCH_REQUIRED"):
        residuals_from_forecasts((forecast,), (actual,), as_of=actual.available_at)
    # No mature fact means no residual and does not require a base batch yet.
    assert residuals_from_forecasts((forecast,), (actual,), as_of=end) == ()

    rows = residuals_from_forecasts(
        (forecast,), (actual,), as_of=actual.available_at,
        base_batches={forecast.forecast_id: base},
    )
    assert len(rows) == 1
    assert rows[0].p_base == .3
    assert rows[0].p_issued == .5
    assert rows[0].base_error == pytest.approx(.4)
    assert rows[0].issued_error == pytest.approx(.2)


def test_walk_forward_never_exposes_labels_that_arrive_after_origin():
    labels = (
        {"id": "first", "actual_available_at": ORIGIN + timedelta(hours=1)},
        {"id": "equal", "actual_available_at": ORIGIN + timedelta(hours=2)},
        {"id": "future", "actual_available_at": ORIGIN + timedelta(hours=3)},
    )
    origins = (ORIGIN + timedelta(hours=2), ORIGIN)

    folds = walk_forward_folds(origins, labels)
    assert [fold.origin_time for fold in folds] == sorted(origins)
    assert folds[0].train == ()
    assert [label["id"] for label in folds[1].train] == ["first", "equal"]
    assert all(label["actual_available_at"] <= fold.origin_time
               for fold in folds for label in fold.train)

    result = walk_forward(
        origins,
        labels,
        lambda train, origin: (origin, tuple(label["id"] for label in train)),
    )
    assert result[1][1] == ("first", "equal")


def test_walk_forward_requires_aware_unique_origins():
    with pytest.raises(ValueError, match="timezone-aware"):
        walk_forward_folds((datetime(2026, 1, 1),), ())
    with pytest.raises(ValueError, match="unique"):
        walk_forward_folds((ORIGIN, ORIGIN), ())
