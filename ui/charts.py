"""Select rows for display. Never calculate forecast, metrics or intervals."""
from frontend.contracts import utc, visible_actuals


def chart_view(result, turbine_id, horizon_hours, as_of, comparison=()):
    rows = [row for row in result["predictions"] if row["turbine_id"] == turbine_id]
    rows = sorted(rows, key=lambda row: utc(row["target_start"]))[:horizon_hours]
    keys = {(utc(row["target_start"]), utc(row["target_end"])) for row in rows}
    labels = {(utc(row["target_start"]), utc(row["target_end"])): (row["target_start"], row["target_end"]) for row in rows}
    actuals = [row for row in visible_actuals(result.get("actuals", []), as_of)
               if row["turbine_id"] == turbine_id and (utc(row["target_start"]), utc(row["target_end"])) in keys]
    actuals = [{**row, "target_start": labels[(utc(row["target_start"]), utc(row["target_end"]))][0],
                "target_end": labels[(utc(row["target_start"]), utc(row["target_end"]))][1]} for row in actuals]
    compared = [row for row in comparison if row["turbine_id"] == turbine_id
                and (utc(row["target_start"]), utc(row["target_end"])) in keys]
    return {"rows": rows, "actuals": actuals, "comparison": compared,
            "coverage": f"{len(rows)} / {horizon_hours}", "unit": "норм. ед."}
