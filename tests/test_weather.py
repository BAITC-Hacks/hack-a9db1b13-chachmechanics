from datetime import timedelta
import json
import pytest
from windoracle.schemas import ForecastRequest
from windoracle.weather.audit import audit_bundle
from windoracle.weather.archive import index_ranges
from windoracle.weather.base import WeatherUnavailable
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


def test_resume_reuses_completed_fragments_after_failure(setup, monkeypatch):
    from windoracle.weather import archive
    calls = []
    def network(url, start, end, max_bytes, budget=None):
        calls.append((url, start, end))
        if url.endswith("bad"):
            raise WeatherUnavailable("interrupted")
        return b"abc", {"Last-Modified": "Thu, 15 Jan 2026 15:00:00 GMT"}
    monkeypatch.setattr(archive, "get_bytes", network)
    setup.weather._get("https://example.test/good", 0, 2)
    with pytest.raises(WeatherUnavailable):
        setup.weather._get("https://example.test/bad", 0, 2)
    resumed = archive.GFSArchive(setup.config, setup.weather.cache)
    assert resumed._get("https://example.test/good", 0, 2)[0] == b"abc"
    assert len(calls) == 2
    assert resumed.transfer_stats["cache_hits"] == 1
    assert resumed.download_budget.used_bytes == 0


def test_resume_rejects_corrupted_object(setup, monkeypatch):
    cache = setup.weather.cache
    cache.save_request("https://example.test/field", 0, 2, b"abc", {})
    checksum = cache.put_object(b"abc")
    (cache.root / "objects" / checksum).write_bytes(b"xyz")
    with pytest.raises(ValueError, match="CHECKSUM"):
        setup.weather._get("https://example.test/field", 0, 2)


def test_budget_stops_before_network_for_oversized_range(monkeypatch):
    from windoracle.weather import archive
    def forbidden(*args, **kwargs):
        pytest.fail("Network must not be called beyond budget")
    monkeypatch.setattr(archive, "urlopen", forbidden)
    with pytest.raises(archive.DownloadLimitExceeded):
        archive.get_bytes("https://example.test", 0, 10, budget=archive.DownloadBudget(10))


def test_budget_accounts_network_bytes_across_requests(monkeypatch):
    import io
    from windoracle.weather import archive
    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 0-2/20"}
    monkeypatch.setattr(archive, "urlopen", lambda *a, **kw: Response(b"abc"))
    budget = archive.DownloadBudget(6)
    assert archive.get_bytes("https://example.test/a", 0, 2, budget=budget)[0] == b"abc"
    assert archive.get_bytes("https://example.test/b", 0, 2, budget=budget)[0] == b"abc"
    assert budget.used_bytes == 6
    with pytest.raises(archive.DownloadLimitExceeded):
        archive.get_bytes("https://example.test/c", 0, 2, budget=budget)
