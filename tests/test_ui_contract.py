"""Run with unittest (stdlib) or pytest; no network, model or weather needed."""
import base64
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import unittest

from frontend.contracts import UIError, validate_result, visible_actuals
from frontend.controller import Controller
from frontend.fixture_service import FixtureService
from frontend.gateway import BackendAdapter, ServiceGateway, display_result
from ui.charts import chart_view
from ui.controls import calculation_request
from windoracle.agents.advisor import explain_forecast


class UIContractTests(unittest.TestCase):
    def setUp(self):
        self.controller = Controller()
        self.fixture = FixtureService()
        self.result = self.fixture.data["forecasts"]["demo-0715-r1"]

    def test_initial_screen_has_two_turbines_and_48_rows(self):
        view = self.controller.view()
        self.assertIsNone(view["error"])
        self.assertEqual(len(view["catalog"]["turbines"]), 2)
        self.assertEqual(len(view["chart"]["rows"]), 48)
        self.assertEqual(view["mode"], "fixture")

    def test_turbine_and_horizon_change(self):
        view = self.controller.dispatch({"type": "select", "turbine_id": "turbine_2", "horizon_hours": 24})
        self.assertEqual(len(view["chart"]["rows"]), 24)
        self.assertTrue(all(row["turbine_id"] == "turbine_2" for row in view["chart"]["rows"]))

    def test_future_actuals_hidden_until_available(self):
        self.assertEqual(self.controller.view()["chart"]["actuals"], [])
        view = self.controller.dispatch({"type": "select", "forecast_id": "demo-0715-r1", "as_of": "2026-07-15T12:00:00Z"})
        self.assertEqual(len(view["chart"]["actuals"]), 10)
        self.assertTrue(all(datetime.fromisoformat(a["available_at"].replace("Z", "+00:00")) <= datetime(2026,7,15,12,tzinfo=timezone.utc) for a in view["chart"]["actuals"]))

    def test_future_weather_is_rejected(self):
        bad = deepcopy(self.result)
        bad["provenance"]["weather_available_at"] = "2026-07-16T00:00:00Z"
        with self.assertRaisesRegex(UIError, "FUTURE_DATA"):
            validate_result(bad, mode="fixture", as_of=bad["origin_time"])

    def test_missing_intervals_are_not_synthesized(self):
        view = self.controller.dispatch({"type":"select", "forecast_id":"demo-0201-r1"})
        self.assertTrue(all(row["q10"] is None for row in view["chart"]["rows"]))
        self.assertEqual(view["chart"]["actuals"], [])

    def test_null_and_zero_are_distinct(self):
        result = deepcopy(self.result)
        result["predictions"][0]["prediction_norm"] = 0
        result["predictions"][1]["prediction_norm"] = None
        view = chart_view(result, "turbine_1", 24, result["origin_time"])
        self.assertEqual(view["rows"][0]["prediction_norm"], 0)
        self.assertIsNone(view["rows"][1]["prediction_norm"])

    def test_synthetic_data_cannot_enter_live_mode(self):
        with self.assertRaisesRegex(UIError, "UNVERIFIED_SOURCE"):
            validate_result(self.result, mode="submission", as_of=self.result["origin_time"])

    def test_ready_export_bytes_and_label(self):
        view = self.controller.dispatch({"type":"export"})
        content = base64.b64decode(view["download"]["base64"]).decode("utf-8-sig")
        expected = (self.fixture.root / "exports/demo-0715-r1.csv").read_text(encoding="utf-8")
        self.assertEqual(content, expected)
        self.assertTrue(view["download"]["filename"].startswith("DEMO_ONLY_"))
        self.assertIn("synthetic", content)

    def test_fixture_submission_export_blocked(self):
        view = self.controller.dispatch({"type":"export", "kind":"submission"})
        self.assertEqual(view["error"]["code"], "FIXTURE_EXPORT_BLOCKED")
        self.assertIsNone(view["download"])

    def test_comparison_contains_only_matching_target_hours(self):
        before = deepcopy(self.fixture.data["forecasts"]["demo-0715-r1"])
        view = self.controller.dispatch({"type":"select", "forecast_id":"demo-0715-r2", "compare":True})
        self.assertEqual(len(view["chart"]["comparison"]), 42)
        self.assertEqual(before, self.fixture.data["forecasts"]["demo-0715-r1"])
        old_keys = {r["target_start"] for r in before["predictions"]}
        self.assertTrue(all(r["target_start"] in old_keys for r in view["chart"]["comparison"]))

    def test_failure_clears_chart_and_retry_restores_it(self):
        view = self.controller.dispatch({"type":"simulate_failure"})
        self.assertEqual(view["error"]["code"], "WEATHER_UNAVAILABLE")
        self.assertIsNone(view["result"])
        self.assertIsNone(view["chart"])
        self.assertIsNotNone(self.controller.dispatch({"type":"refresh"})["chart"])

    def test_invalid_selection_does_not_change_state(self):
        before = deepcopy(self.controller.selection)
        view = self.controller.dispatch({"type":"select", "turbine_id":"unknown"})
        self.assertEqual(view["error"]["code"], "INVALID_SELECTION")
        self.assertEqual(self.controller.selection, before)

    def test_invalid_timestamp_and_quantiles(self):
        bad=deepcopy(self.result)
        bad["origin_time"]="2026-07-15T00:00:00"
        with self.assertRaisesRegex(UIError,"INVALID_TIME"):
            validate_result(bad, mode="fixture", as_of=self.result["origin_time"])
        bad=deepcopy(self.result)
        bad["predictions"][0]["q10"]=.99
        with self.assertRaisesRegex(UIError,"CONTRACT_MISMATCH"):
            validate_result(bad, mode="fixture", as_of=bad["origin_time"])

    def test_advisor_does_not_diagnose_unsupported_causes(self):
        text = " ".join(explain_forecast(self.result, turbine_id="turbine_1"))
        self.assertIn("не подтверждены", text)
        self.assertIn("не рассчитана", text)
        self.assertNotIn("из-за", text)

    def test_no_live_service_never_falls_back_to_fixture(self):
        def absent():
            raise UIError("SERVICE_UNAVAILABLE")
        controller = Controller(gateway_factory=absent)
        view = controller.dispatch({"type":"mode", "value":"replay"})
        self.assertEqual(view["mode"], "replay")
        self.assertIsNone(view["result"])
        self.assertEqual(view["catalog"]["origins"], [])

    def test_actual_end_must_be_available_too(self):
        row = {"available_at":"2026-07-15T00:00:00Z", "target_end":"2026-07-16T00:00:00Z", "power_norm":.5}
        self.assertEqual(visible_actuals([row], "2026-07-15T12:00:00Z"), [])

    def test_backend_mapping_uses_existing_schema_shape(self):
        native={"forecast_id":"real", "origin_time":"2026-07-15T00:00:00Z", "run_id":"gfs", "model_id":"v1", "mode":"replay",
                "provenance":"operational_archive", "predictions":{"rows":self.result["predictions"]},
                "manifest":{"weather":{"available_at":"2026-07-14T22:00:00Z"}, "model":{"activated_at":"2026-07-01T00:00:00Z", "training_cutoff":"2026-06-30T00:00:00Z"}}}
        mapped=display_result(native)
        self.assertIs(validate_result(mapped, mode="replay", as_of=native["origin_time"]), mapped)
        self.assertFalse(mapped["synthetic"])
        self.assertEqual(mapped["weather"], [])

    def test_backend_exceptions_are_sanitized(self):
        class Broken:
            def list_forecasts(self, **kwargs):
                raise RuntimeError("SECRET_API_KEY=not-for-ui")
        gateway=ServiceGateway(Broken())
        with self.assertRaisesRegex(UIError,"SERVICE_ERROR") as error:
            gateway.call("get_catalog",mode="replay")
        self.assertNotIn("SECRET", str(error.exception))

    def test_all_demo_releases_satisfy_temporal_contract(self):
        for result in self.fixture.data["forecasts"].values():
            validate_result(result, mode="fixture", as_of=result["origin_time"])
            self.assertEqual(len(result["predictions"]), 96)


class UIReadinessTests(unittest.TestCase):
    def test_empty_catalog_keeps_configured_turbines_and_service_inventory(self):
        service = SimpleNamespace(
            list_forecasts=lambda **kwargs: [],
            data_summary=lambda: {"rows": 1200, "turbines": [{"id": "turbine_1"}]},
            predictor=SimpleNamespace(state=SimpleNamespace(provenance="trained")),
        )
        catalog = BackendAdapter(service).get_catalog()
        self.assertEqual(catalog["origins"], [])
        self.assertEqual(catalog["turbines"][0]["id"], "turbine_1")
        self.assertTrue(catalog["readiness"]["can_calculate"])
        states = {item["label"]: item for item in catalog["readiness"]["items"]}
        self.assertEqual(states["Датасеты"]["value"], "1 200 записей")
        self.assertEqual(states["Погода"]["state"], "unknown")
        service.predictor = None
        service.data_summary = lambda: {"rows": 0, "turbines": [{"id": "turbine_1"}]}
        readiness = BackendAdapter(service).get_catalog()["readiness"]
        self.assertFalse(readiness["can_calculate"])
        self.assertEqual([i["state"] for i in readiness["items"]], ["ready", "missing", "missing", "unknown"])

    def test_demo_readiness_does_not_claim_real_connections_or_calculate(self):
        controller = Controller()
        before = deepcopy(controller.catalog)
        view = controller.dispatch({"type": "calculate"})
        self.assertEqual(view["error"]["code"], "DEMO_CALCULATION_DISABLED")
        self.assertFalse(view["readiness"]["can_calculate"])
        self.assertTrue(all(i["state"] == "demo" for i in view["readiness"]["items"]))
        self.assertEqual(controller.catalog, before)

    def test_failed_refresh_discards_previous_readiness(self):
        service = SimpleNamespace(list_forecasts=lambda **kwargs: [],
                                  data_summary=lambda: {"rows": 500, "turbines": []}, predictor=None)
        controller = Controller("replay", gateway_factory=lambda: ServiceGateway(service))
        self.assertEqual(controller.view()["readiness"]["items"][0]["state"], "ready")
        def unavailable():
            raise UIError("SERVICE_UNAVAILABLE")
        controller.gateway_factory = unavailable
        view = controller.dispatch({"type": "refresh"})
        self.assertEqual(view["error"]["code"], "SERVICE_UNAVAILABLE")
        self.assertEqual(view["readiness"]["items"][0]["state"], "missing")
        self.assertIsNone(view["selection"])
        self.assertIsNone(view["result"])

    def test_calculation_validates_time_and_horizon_and_uses_catalog_turbines(self):
        catalog = {"turbines": [{"id": "turbine_1"}, {"id": "turbine_2"}]}
        action = {"origin_time": "2026-07-15T06:00:00Z", "horizon_hours": 24, "turbine_ids": ["unknown"]}
        request = calculation_request(action, None, catalog, "replay")
        self.assertEqual(request["turbine_ids"], ["turbine_1", "turbine_2"])
        self.assertEqual(request["mode"], "replay")
        for change, code in [({"origin_time": "2026-07-15T06:00:00"}, "INVALID_TIME"),
                             ({"horizon_hours": 12}, "INVALID_SELECTION")]:
            with self.assertRaisesRegex(UIError, code):
                calculation_request({**action, **change}, None, catalog, "replay")

    def test_first_calculation_works_without_existing_release_and_selects_result(self):
        # Mock the service boundary; this does not certify a real model or weather archive.
        fixture = FixtureService()
        result = deepcopy(fixture.data["forecasts"]["demo-0715-r1"])
        result.update(synthetic=False, mode="replay")
        result["provenance"]["kind"] = "operational_archive"
        catalog = {"origins": [], "turbines": fixture.get_catalog()["turbines"], "timezone": "UTC"}
        calls = []
        class Gateway:
            def call(self, method, **kwargs):
                calls.append((method, kwargs))
                if method == "get_catalog":
                    return deepcopy(catalog)
                if method == "create_forecast":
                    catalog["origins"].append({"forecast_id": result["forecast_id"], "origin_time": result["origin_time"]})
                    return result
                if method == "get_forecast":
                    return deepcopy(result)
                if method == "get_events":
                    return []
                raise AssertionError(method)
        controller = Controller("replay", gateway_factory=Gateway)
        self.assertEqual(controller.view()["error"]["code"], "NO_DATA")
        view = controller.dispatch({"type": "calculate", "origin_time": result["origin_time"], "horizon_hours": 24})
        self.assertIsNone(view["error"])
        self.assertEqual(view["selection"]["forecast_id"], result["forecast_id"])
        self.assertEqual(len(view["chart"]["rows"]), 24)
        request = next(kwargs["request"] for method, kwargs in calls if method == "create_forecast")
        self.assertEqual(request["origin_time"], result["origin_time"])
        self.assertEqual(request["horizon_hours"], 24)


@unittest.skipUnless(importlib.util.find_spec("pydantic"), "Optional integration check requires backend pydantic")
class BackendSchemaIntegrationTests(unittest.TestCase):
    def test_adapter_with_actual_forecastresult_and_backend_export(self):
        from windoracle.schemas import ForecastResult, ForecastRequest, ModelState, WeatherRunMetadata, digest
        from windoracle.export import export_csv
        fixtures=FixtureService().data["forecasts"]
        native={}
        for key in ("demo-0715-r1", "demo-0715-r2"):
            row=fixtures[key]
            fields={k:row[k] for k in ("forecast_id","origin_time","run_id","model_id","parent_forecast_id","mode","release_kind")}
            provenance = row["provenance"]
            request = ForecastRequest(origin_time=row["origin_time"], turbine_ids=("turbine_1", "turbine_2"),
                                      mode="fixture", release_kind=row["release_kind"])
            model = ModelState(model_id=row["model_id"], provenance="synthetic", artifact_ref="UI test fixture",
                               activated_at=provenance["model_activated_at"], training_cutoff=provenance["training_cutoff"],
                               max_label_available_at=provenance["training_cutoff"])
            weather = WeatherRunMetadata(run_id=row["run_id"], provider="UI fixture", model="DEMO-WX",
                run_init_time=provenance["run_init_time"], available_at=provenance["weather_available_at"],
                retrieved_at=row["origin_time"], availability_basis="synthetic", provenance="synthetic", sha256="0" * 64)
            parent = native[row["parent_forecast_id"]].forecast_id if row["parent_forecast_id"] else None
            identity = {"request": request.model_dump(mode="json"), "model": model.model_dump(mode="json"),
                        "bias": None, "snapshot_hash": "UI-test-only", "config_hash": "UI-test-only", "parent_forecast_id": parent}
            native[key]=ForecastResult.model_validate({**fields, "forecast_id": digest(identity), "parent_forecast_id": parent,
                "predictions":{"rows":row["predictions"]}, "provenance":"synthetic",
                "manifest":{**identity, "weather":weather.model_dump(mode="json")}})
        class Service:
            predictor=None
            def get_forecast(self, forecast_id): return native[forecast_id]
            def list_forecasts(self, **kwargs): return list(native.values())
            def export(self, ids, **kwargs): return export_csv([native[i] for i in ids], **kwargs)
            def compare_forecasts(self, first_id, second_id):
                previous={(r.turbine_id,r.target_start):r for r in native[first_id].predictions.rows}
                return [{"turbine_id":r.turbine_id,"target_start":r.target_start.isoformat(),"before":previous[(r.turbine_id,r.target_start)].prediction_norm,
                         "after":r.prediction_norm,"delta":r.prediction_norm-previous[(r.turbine_id,r.target_start)].prediction_norm}
                        for r in native[second_id].predictions.rows if (r.turbine_id,r.target_start) in previous]
        adapter=BackendAdapter(Service())
        catalog=adapter.get_catalog("fixture")
        self.assertEqual(len(catalog["origins"]),2)
        result=adapter.get_forecast("demo-0715-r2","2026-07-15T06:00:00Z")
        comparison=adapter.compare_forecasts("demo-0715-r1","demo-0715-r2","2026-07-15T06:00:00Z")
        self.assertEqual(len(chart_view(result,"turbine_1",48,"2026-07-15T06:00:00Z",comparison)["comparison"]),42)
        exported=adapter.export_forecast("demo-0715-r2","demo","2026-07-15T06:00:00Z")
        self.assertEqual(exported["content"],export_csv([native["demo-0715-r2"]],strict=False,release_policy="all"))
        with self.assertRaisesRegex(UIError,"FIXTURE_EXPORT_BLOCKED"):
            adapter.export_forecast("demo-0715-r2","submission","2026-07-15T06:00:00Z")


def test_adapter_with_persisted_service_releases(setup):
    """Exercise the real service/store boundary with explicitly synthetic inputs."""
    from datetime import timedelta
    import pytest
    from windoracle.schemas import ForecastRequest
    from windoracle.service import ForecastService
    from tests.conftest import ORIGIN, bundle

    request = ForecastRequest(origin_time=ORIGIN, turbine_ids=("turbine_1", "turbine_2"), mode="fixture")
    first = setup.create_forecast(request)
    later = ORIGIN + timedelta(hours=6)
    setup.weather.cache.save(bundle("run-2", init=ORIGIN, available=later, wind=8))
    second = setup.create_forecast(request.model_copy(update={"origin_time": later, "release_kind": "update"}),
                                   parent_forecast_id=first.forecast_id)
    reader = ForecastService(setup.config, setup.store, setup.weather)
    adapter = BackendAdapter(reader)
    assert len(adapter.get_catalog("fixture")["origins"]) == 2
    result = adapter.get_forecast(second.forecast_id, later.isoformat())
    validate_result(result, mode="fixture", as_of=later.isoformat())
    comparison = adapter.compare_forecasts(first.forecast_id, second.forecast_id, later.isoformat())
    assert len(comparison) == 84
    assert all(abs(row["delta"] - .15) < 1e-10 for row in comparison)
    assert adapter.get_events(second.forecast_id, later.isoformat())
    exported = adapter.export_forecast(second.forecast_id, "demo", later.isoformat())
    assert exported["content"] == reader.export([second.forecast_id], strict=False, release_policy="all")
    with pytest.raises(UIError, match="FIXTURE_EXPORT_BLOCKED"):
        adapter.export_forecast(second.forecast_id, "submission", later.isoformat())
    with pytest.raises(UIError, match="FUTURE_DATA"):
        adapter.get_forecast(second.forecast_id, ORIGIN.isoformat())
    with pytest.raises(UIError, match="MODEL_UNAVAILABLE"):
        adapter.create_forecast(request.model_dump(mode="json"))
    assert reader.get_forecast(first.forecast_id) == first


def test_streamlit_host_starts_without_exception():
    import pytest
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    assert app.session_state["tt_controller"].view()["error"] is None
    before = json.loads(app.get("bidi_component")[0].proto.json)
    app.session_state["tt_controller"].dispatch({"type": "export"})
    app.session_state["tt_last_action"] = "export-1"
    app.run(timeout=20)
    assert not app.exception
    assert len(app.get("download_button")) == 1
    after = json.loads(app.get("bidi_component")[0].proto.json)
    # Export is delivered by Streamlit, but its completion must still notify
    # the component or the unchanged view leaves subsequent controls blocked.
    assert after.pop("action_ack") == "export-1"
    before.pop("action_ack")
    assert after == before


if __name__ == "__main__":
    unittest.main()
