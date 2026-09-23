"""Run with unittest (stdlib) or pytest; no network, model or weather needed."""
import base64
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import unittest

from frontend.contracts import UIError, validate_result, visible_actuals
from frontend.controller import Controller
from frontend.fixture_service import FixtureService
from frontend.gateway import BackendAdapter, ServiceGateway, display_result
from ui.charts import chart_view
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


@unittest.skipUnless(importlib.util.find_spec("pydantic"), "Optional integration check requires backend pydantic")
class BackendSchemaIntegrationTests(unittest.TestCase):
    def test_adapter_with_actual_forecastresult_and_backend_export(self):
        from windoracle.schemas import ForecastResult
        from windoracle.export import export_csv
        fixtures=FixtureService().data["forecasts"]
        native={}
        for key in ("demo-0715-r1", "demo-0715-r2"):
            row=fixtures[key]
            fields={k:row[k] for k in ("forecast_id","origin_time","run_id","model_id","parent_forecast_id","mode","release_kind")}
            native[key]=ForecastResult.model_validate({**fields,"predictions":{"rows":row["predictions"]},"provenance":"synthetic",
                "manifest":{"weather":{"available_at":row["provenance"]["weather_available_at"]},
                            "model":{"provenance":"synthetic","activated_at":row["provenance"]["model_activated_at"],"training_cutoff":row["provenance"]["training_cutoff"]}}})
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


if __name__ == "__main__":
    unittest.main()
