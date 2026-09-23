"""Formatting metadata only; source/run selection belongs to service.py."""
from frontend.contracts import utc


def age_label(timestamp, reference):
    if not timestamp:
        return "Нет данных"
    seconds = (utc(reference) - utc(timestamp)).total_seconds()
    if seconds < 0:
        return "Ещё недоступно"
    minutes = int(seconds // 60)
    return f"{minutes // 60} ч {minutes % 60:02d} мин" if minutes >= 60 else f"{minutes} мин"


def provenance_view(result, as_of):
    meta = dict(result["provenance"])
    meta.update(run_id=result["run_id"], model_id=result["model_id"], bias_id=result.get("bias_id"),
                forecast_id=result["forecast_id"], parent_forecast_id=result.get("parent_forecast_id"),
                weather_age=age_label(meta.get("weather_available_at"), result["origin_time"]),
                telemetry_age=age_label(meta.get("last_observation_available_at"), result["origin_time"]),
                view_as_of=as_of)
    return meta
