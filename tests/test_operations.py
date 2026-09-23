from __future__ import annotations

import pytest

from ui.operations import operational_summary, wind_map_rows


def test_operational_summary_reports_kium_error_and_honest_guidance():
    result = {
        "predictions": [
            {
                "turbine_id": "turbine_1",
                "target_start": "2026-01-16T00:00:00+05:00",
                "target_end": "2026-01-16T01:00:00+05:00",
                "prediction_norm": 0.4,
                "q10": 0.1,
                "q90": 0.7,
            },
            {
                "turbine_id": "turbine_1",
                "target_start": "2026-01-16T01:00:00+05:00",
                "target_end": "2026-01-16T02:00:00+05:00",
                "prediction_norm": 0.8,
                "q10": 0.5,
                "q90": 1.0,
            },
        ],
        "actuals": [
            {
                "turbine_id": "turbine_1",
                "target_start": "2026-01-16T00:00:00+05:00",
                "target_end": "2026-01-16T01:00:00+05:00",
                "power_norm": 0.5,
            },
            {
                "turbine_id": "turbine_1",
                "target_start": "2026-01-16T01:00:00+05:00",
                "target_end": "2026-01-16T02:00:00+05:00",
                "power_norm": 0.7,
            },
        ],
    }

    summary = operational_summary(result, "turbine_1", 48)

    assert summary["forecast_kium"]["capacity_factor"] == pytest.approx(0.6)
    assert summary["actual_kium"]["capacity_factor"] == pytest.approx(0.6)
    assert summary["accuracy"]["mae"] == pytest.approx(0.1)
    assert summary["accuracy"]["underforecast_rate"] == pytest.approx(0.5)
    assert summary["accuracy"]["overforecast_rate"] == pytest.approx(0.5)
    assert summary["uncertainty"]["coverage_q10_q90"] == pytest.approx(1.0)
    codes = " ".join(item["text"].lower() for item in summary["recommendations"])
    assert "резерв" in codes
    assert "только после диагностики" not in codes
    assert any(item["title"] == "Ремонт — только после диагностики" for item in summary["recommendations"])


def test_operational_summary_tells_the_truth_when_intervals_are_missing():
    result = {
        "predictions": [
            {
                "turbine_id": "turbine_2",
                "target_start": "2026-02-22T00:00:00+05:00",
                "target_end": "2026-02-22T01:00:00+05:00",
                "prediction_norm": 0.3,
                "q10": None,
                "q90": None,
            }
        ],
        "actuals": [],
    }

    summary = operational_summary(result, "turbine_2", 24)

    assert summary["actual_kium"] is None
    assert summary["accuracy"] is None
    assert summary["uncertainty"]["available_count"] == 0
    assert summary["uncertainty"]["coverage_q10_q90"] is None
    assert summary["recommendations"][0]["title"] == "Неопределённость ещё не откалибрована"


def test_wind_map_uses_only_supplied_weather_and_coordinates():
    result = {
        "weather": [
            {
                "turbine_id": "turbine_1",
                "valid_time": "2026-02-01T00:00:00+05:00",
                "u_ms": 3.0,
                "v_ms": 4.0,
                "temperature_c": -2.0,
            }
        ]
    }
    turbines = [
        {"id": "turbine_1", "latitude": 43.64515, "longitude": 78.535604},
        {"id": "turbine_2", "latitude": None, "longitude": 78.538828},
    ]

    rows = wind_map_rows(result, turbines, "turbine_1")

    assert rows == [
        {
            "turbine_id": "turbine_1",
            "latitude": 43.64515,
            "longitude": 78.535604,
            "wind_ms": pytest.approx(5.0),
            "temperature_c": -2.0,
            "direction": pytest.approx(216.86989764584402),
            "selected": True,
        }
    ]
