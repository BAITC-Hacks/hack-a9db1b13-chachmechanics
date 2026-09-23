import csv
import io
from datetime import timedelta
import pytest
from windoracle.schemas import ForecastRequest
from windoracle.export import export_csv
from .conftest import ORIGIN


def test_fixture_export_guard_and_halfopen_window(setup):
    req = ForecastRequest(origin_time=ORIGIN, turbine_ids=("turbine_1", "turbine_2"), mode="fixture")
    result = setup.create_forecast(req)
    with pytest.raises(ValueError, match="STRICT_EXPORT"):
        export_csv([result])
    content = export_csv([result], strict=False, target_start=ORIGIN + timedelta(hours=1),
                         target_end=ORIGIN + timedelta(hours=25))
    rows = list(csv.DictReader(io.StringIO(content)))
    assert len(rows) == 48
    assert {r["prediction_unit"] for r in rows} == {"normalized_power"}
    assert all(r["q10"] == "" for r in rows)


def test_portable_output_and_tamper_check(setup, tmp_path):
    import json
    from windoracle.cli import persist_outputs, read_outputs
    req = ForecastRequest(origin_time=ORIGIN, turbine_ids=("turbine_1", "turbine_2"), mode="fixture")
    result = setup.create_forecast(req)
    directory = tmp_path / "portable"
    persist_outputs(setup, [result.forecast_id], directory)
    assert read_outputs(directory) == [result]
    path = directory / (result.forecast_id + ".json")
    data = json.loads(path.read_text())
    data["predictions"]["rows"][0]["prediction_norm"] = .9
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="CHECKSUM"):
        read_outputs(directory)
