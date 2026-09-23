"""Explicit synthetic UI fixture. Reads saved values; performs no prediction.

This adapter is never a fallback for an unavailable production service.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from frontend.contracts import UIError, utc, visible_actuals


class FixtureService:
    def __init__(self):
        self.root = Path(__file__).parent / "fixtures"
        self.data = json.loads((self.root / "demo.json").read_text(encoding="utf-8"))

    def get_catalog(self, mode="fixture"):
        if mode != "fixture":
            raise UIError("FIXTURE_EXPORT_BLOCKED")
        catalog = copy.deepcopy(self.data["catalog"])
        demo_coordinates = {
            "turbine_1": (43.645150, 78.535604),
            "turbine_2": (43.643198, 78.538828),
        }
        for turbine in catalog.get("turbines", []):
            latitude, longitude = demo_coordinates.get(
                turbine.get("id"), (None, None)
            )
            turbine.setdefault("latitude", latitude)
            turbine.setdefault("longitude", longitude)
            turbine.setdefault("rated_power_mw", None)
        catalog["readiness"] = {"can_calculate": False, "items": [
            {"label": label, "state": "demo", "value": value,
             "detail": "Готовый синтетический пример. Подключение реальных данных в деморежиме не проверяется."}
            for label, value in (("Сервис", "Демо"), ("Датасеты", "Примеры"), ("Модель", "Готовые прогнозы"), ("Погода", "Пример"))]}
        return catalog

    def create_forecast(self, request):
        if request.get("mode") != "fixture":
            raise UIError("FIXTURE_EXPORT_BLOCKED")
        # An existing immutable example is selected, never computed.
        for result in self.data["forecasts"].values():
            if result["origin_time"] == request["origin_time"]:
                return copy.deepcopy(result)
        raise UIError("NO_DATA")

    def get_forecast(self, forecast_id, as_of):
        if forecast_id not in self.data["forecasts"]:
            raise UIError("NO_DATA")
        result = copy.deepcopy(self.data["forecasts"][forecast_id])
        if utc(result["origin_time"]) > utc(as_of):
            raise UIError("FUTURE_DATA")
        result["actuals"] = visible_actuals(result.get("actuals", []), as_of)
        return result

    def compare_forecasts(self, previous_id, current_id, as_of):
        self.get_forecast(previous_id, as_of)
        self.get_forecast(current_id, as_of)
        return copy.deepcopy(self.data["comparisons"].get(f"{previous_id}|{current_id}", []))

    def get_events(self, forecast_id, as_of):
        return [copy.deepcopy(event) for event in self.data["events"].get(forecast_id, []) if utc(event["event_time"]) <= utc(as_of)]

    def export_forecast(self, forecast_id, kind, as_of):
        self.get_forecast(forecast_id, as_of)
        if kind != "demo":
            raise UIError("FIXTURE_EXPORT_BLOCKED")
        # Download a ready artifact. CSV is not assembled by UI code.
        return {"filename": f"DEMO_ONLY_{forecast_id}.csv", "mime": "text/csv", "synthetic": True,
                "content": (self.root / "exports" / f"{forecast_id}.csv").read_text(encoding="utf-8")}
