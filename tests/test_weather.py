from datetime import timedelta
import json
import pytest
from TwinTurbo.ai.schemas import ForecastRequest
from TwinTurbo.ai.weather.audit import audit_bundle
from TwinTurbo.ai.weather.archive import index_ranges
from TwinTurbo.ai.weather.base import WeatherUnavailable
from .conftest import ORIGIN, bundle


def request(mode="fixture"):
    return ForecastRequest(origin_time=ORIGIN, turbine_ids=("turbine_1", "turbine_2"), mode=mode)


def test_future_run_is_not_selected(setup):
    setup.weather.cache.save(bundle("late", init=ORIGIN, available=ORIGIN + timedelta(hours=6), wind=10))
    assert setup.weather.select_run(request()).metadata.run_id == "run-1"
    with pytest.raises(ValueError, match="FUTURE_WEATHER"):
        audit_bundle(bundle("late", available=ORIGIN + timedelta(seconds=1)), request())


def test_incomplete_and_nonoperational_rejected(setup):
    with pytest.raises(ValueError, match="INCOMPLETE_WEATHER"):
        audit_bundle(bundle(count=47), request())
    with pytest.raises(WeatherUnavailable):
        setup.weather.select_run(request("submission"))


def test_corrupt_raw_and_manifest_fail_closed(setup):
    cache = setup.weather.cache
    checksum = cache.put_object(b"original")
    (cache.root / "objects" / checksum).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="CHECKSUM"):
        cache.get_object(checksum)
    path = next((cache.root / "runs").glob("*.json"))
    value = json.loads(path.read_text())
    value["bundle"]["values"][0]["wind_ms"] = 100
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="CHECKSUM"):
        list(cache.bundles())


def test_index_selects_only_required_fields():
    idx = "1:0:d=2026010112:TMP:2 m above ground:6 hour fcst:\n2:100:d=2026010112:UGRD:100 m above ground:6 hour fcst:\n3:200:d=2026010112:VGRD:100 m above ground:6 hour fcst:\n4:300:d=2026010112:OTHER:surface:6 hour fcst:"
    assert index_ranges(idx, 100) == [("TMP", 0, 99), ("UGRD", 100, 199), ("VGRD", 200, 299)]
