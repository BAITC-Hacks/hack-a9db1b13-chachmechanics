"""Deterministic presentation of supplied evidence. No diagnosis or model math."""
from __future__ import annotations


def explain_forecast(result, *, turbine_id, has_actuals=False, comparison=None):
    lines = []
    if result.get("synthetic"):
        lines.append("Это синтетический пример интерфейса. Значения не описывают реальную ВЭС.")
    if any(w.startswith("SOURCE_TIME_ASSUMED") for w in result.get("warnings", [])):
        lines.append("Часовой пояс и смысл меток исходного CSV приняты как явное допущение; совмещение с погодой требует подтверждения.")
    lines.append(f"Показан сохранённый выпуск {result['forecast_id']} для {turbine_id}. Модель: {result['model_id']}.")
    if not has_actuals:
        lines.append("На выбранный момент факт для этих часов ещё недоступен. Ошибка прогноза не рассчитана.")
    if not any(row.get("q10") is not None for row in result["predictions"] if row["turbine_id"] == turbine_id):
        lines.append("Оценка интервала отсутствует; показан только точечный прогноз.")
    if comparison:
        lines.append("Сравниваются готовые значения двух выпусков на совпадающих часах. Изменение погоды или версии модели само по себе не доказывает причину отклонения.")
    lines.append("Без статусов оборудования нельзя установить причину отклонения: неисправность, обледенение или ограничение мощности не подтверждены.")
    return lines
