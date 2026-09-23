"""Shared state machine for Streamlit and dependency-free local preview."""
from __future__ import annotations

import base64
import re

from frontend.bootstrap import ensure_package
from frontend.contracts import UIError, validate_result
from frontend.fixture_service import FixtureService
from frontend.gateway import ServiceGateway
from ui.charts import chart_view
from ui.controls import initial_selection, validate_selection
from ui.events import error_view
from ui.provenance import provenance_view

ensure_package()
from windoracle.agents.advisor import explain_forecast


class Controller:
    def __init__(self, mode="fixture", gateway_factory=ServiceGateway):
        self.mode, self.gateway_factory = mode, gateway_factory
        self.selection = None
        self.catalog = {"origins": [], "turbines": [], "timezone": "UTC"}
        self.error = self.download = self.gateway = None
        self.reload()

    def call(self, method, **kwargs):
        if self.gateway is None:
            raise UIError("SERVICE_UNAVAILABLE")
        if self.mode == "fixture":
            return getattr(self.gateway, method)(**kwargs)
        return self.gateway.call(method, **kwargs)

    def reload(self):
        self.error, self.download = None, None
        try:
            self.gateway = FixtureService() if self.mode == "fixture" else self.gateway_factory()
            self.catalog = self.call("get_catalog", mode=self.mode)
            if self.selection:
                try:
                    validate_selection(self.selection, self.catalog)
                except UIError:
                    self.selection = None
            if self.selection is None:
                self.selection = initial_selection(self.catalog)
        except UIError as exc:
            self.error = error_view(exc.code)

    def dispatch(self, action):
        self.error, self.download = None, None
        try:
            kind = action.get("type")
            if kind == "mode":
                if action.get("value") not in ("fixture", "replay", "submission"):
                    raise UIError("INVALID_SELECTION")
                self.mode, self.selection = action["value"], None
                self.catalog = {"origins": [], "turbines": [], "timezone": "UTC"}
                self.gateway = None
                self.reload()
            elif kind == "refresh":
                self.reload()
            elif kind == "simulate_failure" and self.mode == "fixture":
                raise UIError("WEATHER_UNAVAILABLE")
            elif self.selection is None:
                raise UIError("NO_DATA")
            elif kind == "select":
                updated = dict(self.selection)
                for key in ("turbine_id", "horizon_hours", "compare", "as_of"):
                    if key in action:
                        updated[key] = action[key]
                if "forecast_id" in action and action["forecast_id"] != self.selection["forecast_id"]:
                    origin = next((o for o in self.catalog["origins"] if o["forecast_id"] == action["forecast_id"]), None)
                    if origin is None:
                        raise UIError("INVALID_SELECTION")
                    updated.update(forecast_id=origin["forecast_id"], origin_time=origin["origin_time"], as_of=origin["origin_time"])
                validate_selection(updated, self.catalog)
                self.selection = updated
            elif kind == "calculate":
                request = {"origin_time": self.selection["origin_time"], "turbine_ids": [t["id"] for t in self.catalog["turbines"]],
                           "horizon_hours": self.selection["horizon_hours"], "mode": self.mode}
                created = self.call("create_forecast", request=request)
                self.reload()
                if not self.error:
                    self.selection.update(forecast_id=created["forecast_id"], origin_time=created["origin_time"],
                                          as_of=created["origin_time"])
            elif kind == "export":
                export_kind = "demo" if self.mode == "fixture" else "submission"
                if action.get("kind") == "submission" and self.mode == "fixture":
                    raise UIError("FIXTURE_EXPORT_BLOCKED")
                artifact = self.call("export_forecast", forecast_id=self.selection["forecast_id"], kind=export_kind, as_of=self.selection["as_of"])
                if not isinstance(artifact.get("content"), (bytes, str)) or (self.mode != "fixture" and artifact.get("synthetic")):
                    raise UIError("EXPORT_UNAVAILABLE")
                content = artifact["content"].encode("utf-8-sig") if isinstance(artifact["content"], str) else artifact["content"]
                filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", artifact.get("filename", "forecast.csv"))
                self.download = {"filename": filename, "mime": "text/csv", "base64": base64.b64encode(content).decode()}
            else:
                raise UIError("INVALID_SELECTION")
        except UIError as exc:
            self.error = error_view(exc.code)
        return self.view()

    def view(self):
        view = {"mode": self.mode, "catalog": self.catalog, "selection": self.selection, "error": self.error,
                "download": self.download, "result": None, "chart": None, "events": [], "advisor": []}
        if self.error or not self.selection:
            return view
        try:
            s = self.selection
            result = validate_result(self.call("get_forecast", forecast_id=s["forecast_id"], as_of=s["as_of"]), mode=self.mode, as_of=s["as_of"])
            compare = []
            if s["compare"] and result.get("parent_forecast_id"):
                compare = self.call("compare_forecasts", previous_id=result["parent_forecast_id"], current_id=result["forecast_id"], as_of=s["as_of"])
            chart = chart_view(result, s["turbine_id"], s["horizon_hours"], s["as_of"], compare)
            result["actuals"] = chart["actuals"]
            view.update(result=result, chart=chart, provenance=provenance_view(result, s["as_of"]),
                        events=self.call("get_events", forecast_id=result["forecast_id"], as_of=s["as_of"]),
                        advisor=explain_forecast(result, turbine_id=s["turbine_id"], has_actuals=bool(chart["actuals"]), comparison=compare))
        except UIError as exc:
            view["error"] = error_view(exc.code)
        return view
