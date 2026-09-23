from datetime import datetime, timedelta, timezone

import pytest

from scripts.generate_february import bias_lifecycle, daily_origins
from windoracle.schemas import BiasState


UTC = timezone.utc


def _bias() -> BiasState:
    created = datetime(2026, 1, 31, 18, tzinfo=UTC)
    return BiasState(
        bias_id="bias-test",
        model_id="model-test",
        created_as_of=created,
        last_actual_available_at=created - timedelta(minutes=45),
        parameters={"window_days": 21.0},
    )


def test_daily_origins_are_inclusive_and_require_whole_days():
    start = datetime(2026, 1, 31, 18, tzinfo=UTC)
    end = datetime(2026, 2, 27, 18, tzinfo=UTC)
    origins = daily_origins(start, end)

    assert len(origins) == 28
    assert origins[0] == start
    assert origins[-1] == end
    assert all(right - left == timedelta(days=1) for left, right in zip(origins, origins[1:]))

    with pytest.raises(ValueError, match="ORIGIN_RANGE_MUST_USE_WHOLE_DAYS"):
        daily_origins(start, end + timedelta(hours=1))


def test_bias_lifecycle_never_uses_future_or_stale_correction():
    bias = _bias()

    selected, lifecycle = bias_lifecycle(
        bias, bias.created_as_of - timedelta(seconds=1)
    )
    assert selected is None
    assert lifecycle == "future_skipped"

    selected, lifecycle = bias_lifecycle(bias, bias.created_as_of)
    assert selected == bias
    assert lifecycle == "active"

    # The state is passed through only to let model post-processing publish
    # expired_history with zero correction and null intervals.
    selected, lifecycle = bias_lifecycle(
        bias, bias.last_actual_available_at + timedelta(days=21, seconds=1)
    )
    assert selected == bias
    assert lifecycle == "expired_passthrough"
