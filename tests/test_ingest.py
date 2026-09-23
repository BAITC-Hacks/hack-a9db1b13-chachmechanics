from datetime import datetime, timezone
import pandas as pd
import pytest
from TwinTurbo.ai.ingest import SOURCE_COLUMNS, audit_csv, ingest_csv


def csv_file(tmp_path, timestamps, powers=None):
    path = tmp_path / "data.csv"
    rows = [[i + 1, t, 5, powers[i] if powers else .4, 12] for i, t in enumerate(timestamps)]
    pd.DataFrame(rows, columns=list(SOURCE_COLUMNS)).to_csv(path, index=False)
    return path


def test_full_partial_missing_and_availability(tmp_path, setup):
    stamps = list(pd.date_range("2025-06-01", periods=6, freq="10min")) + [pd.Timestamp("2025-06-01 02:00")]
    path = csv_file(tmp_path, stamps)
    obs, report = ingest_csv(path, "turbine_1", setup.config.site)
    assert [o.quality_flag for o in obs] == ["complete", "missing", "incomplete"]
    assert obs[0].power_norm == pytest.approx(.4)
    assert obs[1].power_norm is None and obs[2].power_norm is None
    setup.store.ingest(obs, report)
    before = datetime(2025, 6, 1, 1, 14, tzinfo=timezone.utc)
    assert setup.store.observations_as_of(before, ("turbine_1",)) == ()
    after = before.replace(minute=15)
    assert len(setup.store.observations_as_of(after, ("turbine_1",))) == 1


def test_duplicates_rejected(tmp_path, setup):
    path = csv_file(tmp_path, ["2025-06-01", "2025-06-01"])
    assert audit_csv(path)["duplicate_timestamps"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        ingest_csv(path, "turbine_1", setup.config.site)


def test_invalid_power_does_not_make_partial_target(tmp_path, setup):
    path = csv_file(tmp_path, pd.date_range("2025-06-01", periods=6, freq="10min"), [.4] * 5 + [2])
    obs, _ = ingest_csv(path, "turbine_1", setup.config.site)
    assert obs[0].quality_flag == "invalid"
    assert obs[0].power_norm is None


def test_interval_end_and_timezone(tmp_path, setup):
    path = csv_file(tmp_path, pd.date_range("2025-06-01 05:10", periods=6, freq="10min"))
    site = setup.config.site.model_copy(update={"timezone": "Asia/Almaty", "timestamp_semantics": "interval_end"})
    obs, _ = ingest_csv(path, "turbine_1", site)
    assert obs[0].event_start == datetime(2025, 6, 1, tzinfo=timezone.utc)
    assert obs[0].n_samples == 6
