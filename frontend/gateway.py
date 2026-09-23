"""Presentation adapter for ForecastService; never reads raw CSV or SQL."""
from __future__ import annotations

import importlib
import os
from datetime import timedelta

from frontend.bootstrap import ensure_package
from frontend.contracts import UIError, utc, wire


def display_result(native):
    raw = wire(native)
    manifest = raw.get("manifest", {})
    weather, model = manifest.get("weather", {}), manifest.get("model", {})
    return {**raw, "synthetic": raw.get("mode") == "fixture" or raw.get("provenance") == "synthetic",
            "predictions": raw["predictions"]["rows"],
            "provenance": {"kind": raw["provenance"], "provider": weather.get("provider"),
                "weather_model": weather.get("model"), "run_init_time": weather.get("run_init_time"),
                "weather_available_at": weather.get("available_at"),
                "availability_basis": weather.get("availability_basis"), "sha256": weather.get("sha256"),
                "model_activated_at": model.get("activated_at"), "training_cutoff": model.get("training_cutoff"),
                "last_observation_available_at": manifest.get("last_actual_available_at")},
            "weather": [], "actuals": [], "metrics": None}


class BackendAdapter:
    def __init__(self, service=None):
        if service is None:
            ensure_package()
            factory_name = os.environ.get("TWINTURBO_SERVICE_FACTORY")
            if factory_name:
                module, name = factory_name.split(":", 1)
                service = getattr(importlib.import_module(module), name)()
            else:
                config_path = os.environ.get("TWINTURBO_CONFIG")
                if not config_path:
                    raise UIError("SERVICE_UNAVAILABLE")
                from windoracle.config import load_config
                from windoracle.service import ForecastService
                from windoracle.store import Store
                from windoracle.weather.archive import GFSArchive
                config = load_config(config_path)
                predictor = None
                if os.environ.get("TWINTURBO_PREDICTOR"):
                    from windoracle.cli import load_predictor
                    predictor = load_predictor(os.environ["TWINTURBO_PREDICTOR"])
                service = ForecastService(config, Store(config.storage.database), GFSArchive(config), predictor)
        self.service = service

    def get_catalog(self, mode="replay"):
        saved = sorted(self.service.list_forecasts(mode=mode), key=lambda r: (r.origin_time, r.forecast_id))
        summary = self.service.data_summary() if callable(getattr(self.service, "data_summary", None)) else {}
        origins, turbine_ids = [], set()
        turbine_ids.update(t["id"] for t in summary.get("turbines", []))
        for result in saved:
            turbine_ids.update(row.turbine_id for row in result.predictions.rows)
            final_hour = max(row.target_end for row in result.predictions.rows)
            origins.append({"origin_time": result.origin_time.isoformat(), "forecast_id": result.forecast_id,
                            "label": "Обновление" if result.parent_forecast_id else "Основной выпуск",
                            "inspection_times": [(result.origin_time + timedelta(hours=12)).isoformat(),
                                                 (final_hour + timedelta(hours=1)).isoformat()]})
        predictor = getattr(self.service, "predictor", None)
        trained = predictor is not None and getattr(predictor.state, "provenance", None) == "trained"
        rows = summary.get("rows")
        weather_evidence = any(r.provenance == "operational_archive" for r in saved)
        readiness = {"can_calculate": trained and bool(turbine_ids), "items": [
            {"label": "Сервис", "state": "ready", "value": "Подключён", "detail": "Сохранённые выпуски доступны для просмотра."},
            {"label": "Датасеты", "state": "ready" if rows else "missing" if rows == 0 else "unknown",
             "value": f"{rows:,} записей".replace(",", " ") if rows else "Не загружены" if rows == 0 else "Не проверены",
             "detail": "Число импортированных записей по данным сервиса. Доступность к моменту выпуска проверяется при расчёте."},
            {"label": "Модель", "state": "ready" if trained else "missing",
             "value": "Подключена" if trained else "Тестовая" if predictor else "Не подключена",
             "detail": "Время активации модели проверяется при расчёте." if trained else "Для расчёта подключите обученную модель. Сохранённые выпуски можно просматривать без неё."},
            {"label": "Погода", "state": "unknown", "value": "Есть в выпусках" if weather_evidence else "Не проверена",
             "detail": "Происхождение погоды сохранено в выпусках. Покрытие нового момента проверяется при расчёте; наличие кэша здесь не подтверждается." if weather_evidence else "Для нового расчёта нужен допустимый архивный прогноз погоды. Сервис проверит его для выбранного момента."},
        ]}
        return {"turbines": [{"id": t, "name": t.replace("turbine_", "Турбина ")} for t in sorted(turbine_ids)],
                "origins": origins, "timezone": "UTC", "site_name": "Ветровая площадка", "mode": mode,
                "readiness": readiness}

    def create_forecast(self, request):
        if self.service.predictor is None:
            raise UIError("MODEL_UNAVAILABLE")
        from windoracle.schemas import ForecastRequest
        allowed = {k: request[k] for k in ("origin_time", "turbine_ids", "horizon_hours", "mode")}
        return display_result(self.service.create_forecast(ForecastRequest.model_validate(allowed)))

    def get_forecast(self, forecast_id, as_of):
        result = self.service.get_forecast(forecast_id)
        if result.origin_time > utc(as_of):
            raise UIError("FUTURE_DATA")
        view = display_result(result)
        # Optional additive endpoint. No reaching into service.store/weather.
        if callable(getattr(self.service, "get_display_context", None)):
            context = wire(self.service.get_display_context(forecast_id, as_of=utc(as_of)))
            for field in ("weather", "actuals", "metrics"):
                view[field] = context.get(field, view[field])
        return view

    def compare_forecasts(self, previous_id, current_id, as_of):
        self.get_forecast(previous_id, as_of)
        current = self.get_forecast(current_id, as_of)
        ends = {(r["turbine_id"], utc(r["target_start"])): (r["target_start"], r["target_end"]) for r in current["predictions"]}
        return [{**row, "target_start": ends[(row["turbine_id"], utc(row["target_start"]))][0],
                 "target_end": ends[(row["turbine_id"], utc(row["target_start"]))][1]}
                for row in wire(self.service.compare_forecasts(previous_id, current_id))
                if (row["turbine_id"], utc(row["target_start"])) in ends]

    def get_events(self, forecast_id, as_of):
        return [row for row in wire(self.service.events(as_of=utc(as_of))) if row.get("forecast_id") in (None, forecast_id)]

    def export_forecast(self, forecast_id, kind, as_of):
        result = self.get_forecast(forecast_id, as_of)
        if result["synthetic"] and kind != "demo":
            raise UIError("FIXTURE_EXPORT_BLOCKED")
        content = self.service.export([forecast_id], strict=kind != "demo", release_policy="all")
        return {"filename": f"{'DEMO_ONLY_' if kind == 'demo' else ''}{forecast_id}.csv", "mime": "text/csv",
                "content": content, "synthetic": result["synthetic"]}


class ServiceGateway:
    def __init__(self, service=None):
        try:
            self.service = BackendAdapter(service)
        except UIError:
            raise
        except Exception as exc:
            raise UIError("SERVICE_UNAVAILABLE")

    def call(self, method: str, **kwargs):
        try:
            return wire(getattr(self.service, method)(**kwargs))
        except UIError:
            raise
        except Exception as exc:
            known = ("WEATHER_UNAVAILABLE", "FUTURE_MODEL", "FUTURE_WEATHER", "FUTURE_OBSERVATION", "MODEL_OUTPUT_COVERAGE")
            code = next((code for code in known if code in str(exc)), "SERVICE_ERROR")
            code = {"FUTURE_MODEL": "FUTURE_DATA", "FUTURE_WEATHER": "FUTURE_DATA", "FUTURE_OBSERVATION": "FUTURE_DATA", "MODEL_OUTPUT_COVERAGE": "CONTRACT_MISMATCH"}.get(code, code)
            raise UIError(code) from exc
