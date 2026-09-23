"""Operational presentation built from already supplied forecast evidence."""
from __future__ import annotations

from math import atan2, degrees, hypot

from windoracle.evaluate import (
    capacity_factor_summary,
    deviation_diagnostics,
    interval_coverage,
    mean_interval_width,
)


def operational_summary(result, turbine_id: str, horizon_hours: int) -> dict:
    """Return KIUM, accuracy, uncertainty and cautious operator guidance.

    The function never diagnoses a component failure and never issues a power
    set-point.  Those actions require equipment limits and condition-monitoring
    signals that are not present in the hackathon dataset.
    """

    rows = sorted(
        (
            row
            for row in result.get("predictions", [])
            if row.get("turbine_id") == turbine_id
        ),
        key=lambda row: row["target_start"],
    )[:horizon_hours]
    actuals = {
        (row.get("target_start"), row.get("target_end")): row
        for row in result.get("actuals", [])
        if row.get("turbine_id") == turbine_id and row.get("power_norm") is not None
    }
    matched = [
        (row, actuals[(row["target_start"], row["target_end"])])
        for row in rows
        if (row["target_start"], row["target_end"]) in actuals
    ]
    interval_rows = [
        (row, actual)
        for row, actual in matched
        if row.get("q10") is not None and row.get("q90") is not None
    ]
    forecast_kium = capacity_factor_summary(
        (row["prediction_norm"] for row in rows)
    ).to_dict() if rows else None
    actual_kium = capacity_factor_summary(
        (actual["power_norm"] for _, actual in matched)
    ).to_dict() if matched else None
    accuracy = deviation_diagnostics(
        (actual["power_norm"] for _, actual in matched),
        (row["prediction_norm"] for row, _ in matched),
        deadband=.05,
    ).to_dict() if matched else None
    uncertainty = {
        "available_count": len(interval_rows),
        "matched_count": len(matched),
        "coverage_q10_q90": interval_coverage(
            (actual["power_norm"] for _, actual in interval_rows),
            (row["q10"] for row, _ in interval_rows),
            (row["q90"] for row, _ in interval_rows),
        ) if interval_rows else None,
        "mean_width_q10_q90": mean_interval_width(
            (row["q10"] for row, _ in interval_rows),
            (row["q90"] for row, _ in interval_rows),
        ) if interval_rows else None,
    }

    status_counts: dict[str, int] = {}
    for row in rows:
        for token in str(row.get("status") or "").split("|"):
            if token:
                status_counts[token] = status_counts.get(token, 0) + 1
    out_of_domain_count = status_counts.get("out_of_domain", 0)

    recommendations = []
    if out_of_domain_count:
        recommendations.append({
            "severity": "warning",
            "title": "Ветер вне обученного диапазона кривой",
            "text": (
                f"{out_of_domain_count} из {len(rows)} часов помечены "
                "out_of_domain. Прогноз экстраполируется к границе кривой; "
                "учитывайте это как отдельный признак неуверенности."
            ),
        })
    widths = [
        row["q90"] - row["q10"]
        for row in rows
        if row.get("q10") is not None and row.get("q90") is not None
    ]
    if not widths:
        recommendations.append({
            "severity": "warning",
            "title": "Неопределённость ещё не откалибрована",
            "text": "Не использовать точечный прогноз как гарантированный график; дождаться зрелых ошибок или держать резерв.",
        })
    elif max(widths) >= .40:
        recommendations.append({
            "severity": "warning",
            "title": "Широкий прогнозный диапазон",
            "text": "Системному оператору стоит предусмотреть резерв/гибкость. Это не команда менять уставку турбины.",
        })
    if rows and max(row["prediction_norm"] for row in rows) >= .90:
        recommendations.append({
            "severity": "info",
            "title": "Ожидаются часы высокой выработки",
            "text": "Проверить сетевые ограничения и готовность приёма мощности; повышение или ограничение выполняется только по регламенту оператора.",
        })
    if accuracy and accuracy["sample_count"] >= 6 and accuracy["signed_mean_error"] >= .20:
        recommendations.append({
            "severity": "warning",
            "title": "Факт устойчиво ниже прогноза",
            "text": "Проверить доступность, ограничения, качество датчиков и состояние установки. По одной ошибке нельзя назначать ремонт или называть неисправный узел.",
        })
    recommendations.append({
        "severity": "info",
        "title": "Ремонт — только после диагностики",
        "text": "Для рекомендации конкретного ремонта нужны SCADA-аварии, вибрация, температуры подшипников/редуктора, токи и паспортные пределы; в текущем датасете их нет.",
    })
    return {
        "forecast_kium": forecast_kium,
        "actual_kium": actual_kium,
        "accuracy": accuracy,
        "uncertainty": uncertainty,
        "quality": {
            "row_count": len(rows),
            "out_of_domain_count": out_of_domain_count,
            "status_counts": status_counts,
        },
        "recommendations": recommendations,
    }


def wind_map_rows(result, turbines: list[dict], turbine_id: str) -> list[dict]:
    """Build point-map rows at turbine coordinates from supplied weather."""

    coordinates = {
        turbine["id"]: (turbine.get("latitude"), turbine.get("longitude"))
        for turbine in turbines
    }
    weather_by_turbine = {}
    for row in sorted(result.get("weather", []), key=lambda item: item.get("valid_time", "")):
        weather_by_turbine.setdefault(row.get("turbine_id"), row)
    result_rows = []
    for current_id, (latitude, longitude) in coordinates.items():
        if latitude is None or longitude is None:
            continue
        weather = weather_by_turbine.get(current_id, {})
        u_ms, v_ms = weather.get("u_ms"), weather.get("v_ms")
        wind = weather.get("wind_ms")
        if wind is None and isinstance(u_ms, (int, float)) and isinstance(v_ms, (int, float)):
            wind = hypot(u_ms, v_ms)
        direction = weather.get("direction")
        if (
            direction is None
            and isinstance(u_ms, (int, float))
            and isinstance(v_ms, (int, float))
            and (u_ms != 0 or v_ms != 0)
        ):
            # Meteorological direction: degrees clockwise from north from
            # which the wind arrives.  GFS u/v describe where it travels.
            direction = degrees(atan2(-u_ms, -v_ms)) % 360.0
        result_rows.append({
            "turbine_id": current_id,
            "latitude": latitude,
            "longitude": longitude,
            "wind_ms": wind,
            "temperature_c": weather.get("temperature_c"),
            "direction": direction,
            "selected": current_id == turbine_id,
        })
    return result_rows
