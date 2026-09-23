from datetime import datetime, timedelta, timezone

import pytest

from windoracle.agents.critic import Critic
from windoracle.models.bias import (
    Residual,
    apply_bias,
    select_residuals,
    update_bias,
)
from windoracle.schemas import (
    ForecastResult,
    ModelState,
    Observation,
    PredictionBatch,
    PredictionRow,
)


T0 = datetime(2025, 6, 30, 0, tzinfo=timezone.utc)


def residual(
    index=0,
    *,
    model="curve-v1",
    turbine="turbine_1",
    lead=1,
    actual=0.7,
    base=0.4,
    issued=0.6,
):
    target = T0 - timedelta(hours=index + 2)
    origin = target - timedelta(hours=lead)
    return Residual(
        forecast_id=f"forecast-{index}-{lead}",
        model_id=model,
        turbine_id=turbine,
        origin_time=origin,
        target_start=target,
        target_end=target + timedelta(hours=1),
        training_cutoff=origin - timedelta(days=1),
        actual_available_at=target + timedelta(hours=1),
        actual_revision="revision-1",
        actual=actual,
        p_base=base,
        p_issued=issued,
    )


def point_batch(origin=T0, *, turbine="turbine_1", power=0.5, lead=1):
    return PredictionBatch(
        rows=(
            PredictionRow(
                turbine_id=turbine,
                target_start=origin + timedelta(hours=lead),
                target_end=origin + timedelta(hours=lead + 1),
                prediction_norm=power,
            ),
        )
    )


def test_bias_uses_base_error_and_documented_shrinkage():
    rows = [
        residual(0, lead=1, actual=0.8, base=0.4),
        residual(1, lead=8, actual=0.6, base=0.4),
    ]
    state = update_bias(rows, model_id="curve-v1", as_of=T0)
    group = state.parameters["turbines"]["turbine_1"]["groups"]["1-6"]
    assert group["bias"] == pytest.approx((0.4 + 48 * 0.3) / 49)
    assert group["count"] == 1
    # The issued error is deliberately different and must not drive bias.
    assert rows[0].issued_error == pytest.approx(0.2)


def test_bias_estimator_supports_mae_aligned_median_and_keeps_mean_default():
    rows = [
        residual(0, actual=0.5, base=0.5),
        residual(1, actual=0.5, base=0.5),
        residual(2, actual=1.0, base=0.4),
    ]
    default = update_bias(
        rows, model_id="curve-v1", as_of=T0, shrinkage=0
    )
    explicit_mean = update_bias(
        rows,
        model_id="curve-v1",
        as_of=T0,
        shrinkage=0,
        estimator="mean",
    )
    median = update_bias(
        rows,
        model_id="curve-v1",
        as_of=T0,
        shrinkage=0,
        estimator="median",
    )

    assert default == explicit_mean
    assert default.parameters["estimator"] == "mean"
    assert median.parameters["estimator"] == "median"
    assert default.parameters["source_hash"] != median.parameters["source_hash"]
    assert default.bias_id != median.bias_id
    mean_group = default.parameters["turbines"]["turbine_1"]["groups"]["1-6"]
    median_group = median.parameters["turbines"]["turbine_1"]["groups"]["1-6"]
    assert mean_group["bias"] == pytest.approx(.2)
    assert median_group["bias"] == pytest.approx(0)


def test_bias_estimator_change_is_versioned_and_legacy_state_defaults_to_mean():
    rows = [
        residual(0, actual=.5, base=.5),
        residual(1, actual=.5, base=.5),
        residual(2, actual=1, base=.4),
    ]
    mean_state = update_bias(
        rows, model_id="curve-v1", as_of=T0, shrinkage=0
    )
    decision = Critic().review_residuals(
        rows,
        model_id="curve-v1",
        as_of=T0 + timedelta(hours=1),
        previous=mean_state,
        shrinkage=0,
        estimator="median",
    )
    assert decision.action == "propose_bias"
    assert "BIAS_ESTIMATOR_CHANGED" in decision.reasons
    assert decision.proposed_bias.parameters["estimator"] == "median"
    assert decision.proposed_bias.bias_id != mean_state.bias_id

    legacy_parameters = dict(mean_state.parameters)
    legacy_parameters.pop("estimator")
    legacy_parameters["source_hash"] = "legacy-source-hash"
    legacy = mean_state.model_copy(update={"parameters": legacy_parameters})
    assert update_bias(
        rows,
        model_id="curve-v1",
        as_of=T0 + timedelta(hours=1),
        previous=legacy,
        shrinkage=0,
    ) == legacy

    with pytest.raises(ValueError, match="INVALID_BIAS_ESTIMATOR"):
        update_bias(rows, model_id="curve-v1", as_of=T0, estimator="mode")
    with pytest.raises(ValueError, match="INVALID_BIAS_ESTIMATOR"):
        Critic().review_residuals(
            rows, model_id="curve-v1", as_of=T0, estimator="mode"
        )


def test_bias_is_order_independent_and_exact_retries_are_idempotent():
    rows = [residual(index) for index in range(10)]
    state = update_bias(rows, model_id="curve-v1", as_of=T0)
    repeated = update_bias(
        list(reversed(rows)) * 3, model_id="curve-v1", as_of=T0
    )
    assert repeated == state
    assert (
        update_bias(
            rows,
            model_id="curve-v1",
            as_of=T0 + timedelta(hours=1),
            previous=state,
        )
        == state
    )


def test_same_timestamp_new_row_mints_a_new_bias_state():
    first_row = residual(turbine="turbine_1")
    first = update_bias([first_row], model_id="curve-v1", as_of=T0)
    same_timestamp_row = residual(turbine="turbine_2")

    updated = update_bias(
        [first_row, same_timestamp_row],
        model_id="curve-v1",
        as_of=T0 + timedelta(hours=1),
        previous=first,
    )

    assert updated is not first
    assert updated.bias_id != first.bias_id
    assert updated.parameters["source_hash"] != first.parameters["source_hash"]
    assert updated.last_actual_available_at == first.last_actual_available_at
    assert set(updated.parameters["turbines"]) == {"turbine_1", "turbine_2"}


def test_nonempty_rolling_window_eviction_mints_a_new_bias_state():
    newest = residual(0, actual=0.8, base=0.4)
    expiring = residual(20, actual=0.2, base=0.4)
    first = update_bias(
        [newest, expiring],
        model_id="curve-v1",
        as_of=T0,
        window_days=1,
        shrinkage=0,
    )

    updated = update_bias(
        [newest, expiring],
        model_id="curve-v1",
        as_of=T0 + timedelta(hours=4),
        previous=first,
        window_days=1,
        shrinkage=0,
    )

    assert updated is not first
    assert updated.bias_id != first.bias_id
    assert updated.parameters["source_hash"] != first.parameters["source_hash"]
    assert updated.last_actual_available_at == first.last_actual_available_at
    turbine = updated.parameters["turbines"]["turbine_1"]
    assert turbine["count"] == 1
    assert turbine["global_bias"] == pytest.approx(0.4)


def test_future_actual_and_other_model_do_not_change_state():
    current = residual()
    future = residual(1).model_copy(
        update={"actual_available_at": T0 + timedelta(hours=1)}
    )
    first = update_bias([current], model_id="curve-v1", as_of=T0)
    second = update_bias(
        [current, future, residual(2, model="another-model")],
        model_id="curve-v1",
        as_of=T0,
    )
    assert second == first
    assert (
        update_bias([], model_id="new-model", as_of=T0, previous=first) is None
    )


def test_latest_fact_revision_wins_and_update_release_has_no_extra_weight():
    first = residual()
    revised = first.model_copy(
        update={
            "actual_available_at": T0,
            "actual_revision": "revision-2",
            "actual": 0.9,
        }
    )
    update = residual(1).model_copy(update={"release_kind": "update", "actual": 0.1})
    assert select_residuals(
        [first, revised, update], model_id="curve-v1", as_of=T0
    ) == (revised,)
    assert select_residuals(
        [first, revised], model_id="curve-v1", as_of=T0 - timedelta(minutes=30)
    ) == (first,)


def test_latest_scheduled_origin_is_designated_within_lead_group():
    first = residual(lead=3)
    later = first.model_copy(
        update={
            "forecast_id": "later-forecast",
            "origin_time": first.origin_time + timedelta(hours=1),
        }
    )
    assert select_residuals(
        [first, later], model_id="curve-v1", as_of=T0
    ) == (later,)


def test_conflicting_revision_at_same_availability_is_rejected():
    first = residual()
    conflict = first.model_copy(update={"actual": 0.9, "actual_revision": "revision-2"})
    with pytest.raises(ValueError, match="CONFLICTING_RESIDUAL_REVISION"):
        select_residuals([first, conflict], model_id="curve-v1", as_of=T0)


def test_no_fact_mints_no_state_and_clock_advance_does_not_mint_expiry_state():
    assert update_bias([], model_id="curve-v1", as_of=T0) is None
    state = update_bias([residual()], model_id="curve-v1", as_of=T0)
    later = update_bias(
        [residual()],
        model_id="curve-v1",
        as_of=T0 + timedelta(days=22),
        previous=state,
    )
    assert later == state
    assert later.bias_id == state.bias_id

    origin = T0 + timedelta(days=22)
    row = apply_bias(
        point_batch(origin, power=0.4),
        state,
        model_id="curve-v1",
        origin_time=origin,
    ).rows[0]
    assert row.prediction_norm == 0.4
    assert "bias:expired_history" in row.status


def test_bias_clips_output_and_cannot_be_applied_twice():
    state = update_bias(
        [residual(actual=1.0, base=0.0)], model_id="curve-v1", as_of=T0
    )
    corrected = apply_bias(
        point_batch(power=0.8), state, model_id="curve-v1", origin_time=T0
    )
    assert corrected.rows[0].prediction_norm == 1.0
    with pytest.raises(ValueError, match="BIAS_ALREADY_APPLIED"):
        apply_bias(corrected, state, model_id="curve-v1", origin_time=T0)
    with pytest.raises(ValueError, match="INADMISSIBLE_BIAS"):
        apply_bias(
            point_batch(), state, model_id="another-model", origin_time=T0
        )
    with pytest.raises(ValueError, match="INADMISSIBLE_BIAS"):
        apply_bias(
            point_batch(T0 - timedelta(hours=1)),
            state,
            model_id="curve-v1",
            origin_time=T0 - timedelta(hours=1),
        )


def test_missing_bias_is_explicit_and_keeps_point_value():
    row = apply_bias(
        point_batch(power=0.25), None, model_id="curve-v1", origin_time=T0
    ).rows[0]
    assert row.prediction_norm == 0.25
    assert "bias:insufficient_history" in row.status


def test_critic_proposes_only_on_new_fact_and_reports_error_basis():
    rows = [residual(index, actual=0.8, base=0.4, issued=0.6) for index in range(4)]
    critic = Critic()
    first = critic.review_residuals(
        rows,
        model_id="curve-v1",
        as_of=T0,
        min_drift_samples=3,
        drift_threshold=0.1,
    )
    assert first.action == "propose_bias"
    assert first.proposed_bias is not None
    assert "MATURE_OUT_OF_SAMPLE_ERRORS" in first.reasons
    assert first.retrain_recommended is True
    assert first.evaluation["error_sign"] == "actual_minus_prediction"
    assert first.evaluation["all"]["base"]["mean_error"] == pytest.approx(0.4)
    model_review = next(
        item
        for item in first.recommendations
        if item.code == "REVIEW_MODEL_PERSISTENT_SIGNED_ERROR"
    )
    assert model_review.turbine_id == "turbine_1"
    assert model_review.next_step == "review_model_and_inputs"
    assert model_review.evidence["direction"] == "underprediction"
    assert "не доказывают неисправность" in model_review.causal_limitation

    second = critic.review_residuals(
        rows,
        model_id="curve-v1",
        as_of=T0 + timedelta(hours=1),
        previous=first.proposed_bias,
        min_drift_samples=3,
        drift_threshold=0.1,
    )
    assert second.action == "skip"
    assert second.proposed_bias is None
    assert "NO_NEW_ACTUALS" in second.reasons


def test_critic_recommendations_report_insufficient_and_stale_evidence():
    state = update_bias([residual()], model_id="curve-v1", as_of=T0)
    decision = Critic().review_residuals(
        [residual()],
        model_id="curve-v1",
        as_of=T0 + timedelta(days=22),
        previous=state,
    )

    by_code = {item.code: item for item in decision.recommendations}
    assert "INSUFFICIENT_MATURE_ERROR_EVIDENCE" in by_code
    assert "STALE_ERROR_HISTORY" in by_code
    assert by_code["INSUFFICIENT_MATURE_ERROR_EVIDENCE"].evidence == {
        "sample_count": 0,
        "required_sample_count": 48,
        "error_sign": "actual_minus_prediction",
    }
    assert by_code["STALE_ERROR_HISTORY"].next_step == (
        "refresh_actuals_before_decision"
    )
    assert "BIAS_HISTORY_STALE" in decision.reasons
    assert by_code["STALE_ERROR_HISTORY"].as_dict()["severity"] == "warning"


def test_critic_maintenance_screening_requires_persistent_underperformance():
    underperformance = [
        residual(index, actual=0.2, base=0.7, issued=0.7)
        for index in range(4)
    ]
    decision = Critic().review_residuals(
        underperformance,
        model_id="curve-v1",
        as_of=T0,
        min_drift_samples=3,
        min_recommendation_samples=3,
        min_maintenance_samples=3,
        maintenance_error_threshold=0.25,
    )
    maintenance = [
        item
        for item in decision.recommendations
        if item.code == "INSPECT_DATA_AND_ASSET_PERSISTENT_UNDERPERFORMANCE"
    ]
    assert len(maintenance) == 1
    assert maintenance[0].evidence["mean_issued_error"] == pytest.approx(-0.5)
    assert "Не назначайте ремонт" in maintenance[0].message
    assert "не доказывают неисправность" in maintenance[0].causal_limitation

    overperformance = [
        residual(index, actual=0.8, base=0.2, issued=0.2)
        for index in range(4)
    ]
    positive = Critic().review_residuals(
        overperformance,
        model_id="curve-v1",
        as_of=T0,
        min_drift_samples=3,
        min_recommendation_samples=3,
        min_maintenance_samples=3,
        maintenance_error_threshold=0.25,
    )
    assert not any(
        item.code == "INSPECT_DATA_AND_ASSET_PERSISTENT_UNDERPERFORMANCE"
        for item in positive.recommendations
    )


def test_critic_recommendation_thresholds_are_validated():
    critic = Critic()
    for kwargs in (
        {"min_recommendation_samples": 0},
        {"min_maintenance_samples": True},
        {"maintenance_error_threshold": 0},
        {"maintenance_error_threshold": -0.1},
        {"maintenance_error_threshold": 1.1},
    ):
        with pytest.raises(ValueError, match="INVALID_CRITIC_THRESHOLDS"):
            critic.review_residuals(
                [], model_id="curve-v1", as_of=T0, **kwargs
            )


def test_critic_joins_saved_forecast_to_only_mature_complete_actual():
    source = residual()
    state = ModelState(
        model_id="curve-v1",
        training_cutoff=source.training_cutoff,
        max_label_available_at=source.training_cutoff,
        activated_at=source.origin_time,
        artifact_ref="sha256:test-artifact",
    )
    forecast = ForecastResult(
        forecast_id=source.forecast_id,
        origin_time=source.origin_time,
        predictions=PredictionBatch(
            rows=(
                PredictionRow(
                    turbine_id=source.turbine_id,
                    target_start=source.target_start,
                    target_end=source.target_end,
                    prediction_norm=source.p_base,
                ),
            )
        ),
        run_id="run-1",
        model_id=state.model_id,
        provenance="operational_archive",
        mode="replay",
        release_kind="scheduled",
        manifest={"model": state.model_dump(mode="json")},
    )
    actual = Observation(
        turbine_id=source.turbine_id,
        event_start=source.target_start,
        event_end=source.target_end,
        available_at=source.actual_available_at,
        power_norm=source.actual,
        wind_ms=8.0,
        temperature_c=10.0,
        n_samples=6,
        coverage=1.0,
        quality_flag="complete",
        revision=source.actual_revision,
    )
    decision = Critic().review(
        [forecast], [actual], model_id=state.model_id, as_of=T0
    )
    assert decision.action == "propose_bias"
    assert decision.sample_count == 1

    corrected_without_fact = forecast.model_copy(update={"bias_id": "old-bias"})
    no_fact = Critic().review(
        [corrected_without_fact], [], model_id=state.model_id, as_of=T0
    )
    assert no_fact.action == "skip"
    assert "NO_MATURE_ERRORS" in no_fact.reasons


def test_residual_rejects_in_sample_or_over_horizon_record():
    invalid = residual().model_dump()
    invalid["training_cutoff"] = invalid["target_start"]
    with pytest.raises(ValueError, match="RESIDUAL_NOT_OUT_OF_SAMPLE"):
        Residual.model_validate(invalid)
    with pytest.raises(ValueError, match="lead_hours"):
        residual(lead=49)
