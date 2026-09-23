"""CSV audit and hourly ingestion. Raw files are never changed or backfilled."""
from hashlib import sha256
from pathlib import Path
import numpy as np
import pandas as pd
from .config import SiteConfig
from .schemas import Observation, digest

SOURCE_COLUMNS = {
    "ID": "id", "Статистическое время": "timestamp",
    "Средняя скорость ветра(m/s)": "wind_ms",
    "Нормализованная активная мощность": "power_norm",
    "Средняя температура окружающей среды(°C)": "temperature_c",
}
VALUE_COLUMNS = ["wind_ms", "power_norm", "temperature_c"]


def read_source(path):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if set(frame.columns) != set(SOURCE_COLUMNS):
        raise ValueError(f"Unexpected CSV columns: {list(frame.columns)}")
    frame = frame.rename(columns=SOURCE_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, errors="raise")
    if frame.empty or frame.timestamp.isna().any():
        raise ValueError("CSV has no usable timestamps")
    if frame.timestamp.dt.tz is not None:
        raise ValueError("Expected original naive CSV timestamps; configure their timezone")
    if ((frame.timestamp.dt.minute % 10 != 0) | (frame.timestamp.dt.second != 0)
            | (frame.timestamp.dt.microsecond != 0)).any():
        raise ValueError("Timestamp outside the 10-minute grid")
    for c in VALUE_COLUMNS:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame.sort_values("timestamp")


def audit_csv(path) -> dict:
    d = read_source(path)
    unique = d.drop_duplicates("timestamp")
    expected = int((d.timestamp.max() - d.timestamp.min()).total_seconds() / 600) + 1
    hours = unique.set_index("timestamp").resample("1h").size()
    numeric = np.isfinite(d[VALUE_COLUMNS].to_numpy(dtype=float))
    return {
        "source": str(Path(path)), "sha256": sha256(Path(path).read_bytes()).hexdigest(),
        "rows": len(d), "start": str(d.timestamp.min()), "end": str(d.timestamp.max()),
        "duplicate_timestamps": int(d.timestamp.duplicated().sum()),
        "invalid_numeric_values": int((~numeric).sum()),
        "invalid_power_rows": int((~d.power_norm.between(0, 1)).sum()),
        "negative_wind_rows": int((d.wind_ms < 0).sum()),
        "missing_10minute_rows": expected - len(unique),
        "complete_hours": int((hours == 6).sum()),
        "partial_hours": int(((hours > 0) & (hours < 6)).sum()),
        "missing_hours": int((hours == 0).sum()),
        "timezone": "unspecified in source", "february_2026_rows": int(
            ((d.timestamp >= "2026-02-01") & (d.timestamp < "2026-03-01")).sum()),
    }


def ingest_csv(path, turbine_id: str, site: SiteConfig):
    report = audit_csv(path)
    if report["duplicate_timestamps"]:
        raise ValueError("Duplicate timestamps need an explicit source correction")
    d = read_source(path)
    if turbine_id not in {t.id for t in site.turbines}:
        raise ValueError(f"Unknown turbine: {turbine_id}")
    # Ambiguous/repeated civil-clock hours are never guessed.
    d["timestamp"] = d.timestamp.dt.tz_localize(site.timezone, ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")
    if site.timestamp_semantics == "interval_end":
        d["timestamp"] -= pd.Timedelta(minutes=10)
    valid = np.isfinite(d[VALUE_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    valid &= d.power_norm.between(0, 1).to_numpy() & (d.wind_ms >= 0).to_numpy()
    d["valid"] = valid.astype(int)
    d.loc[~valid, VALUE_COLUMNS] = np.nan
    d = d.set_index("timestamp")
    hourly = d.resample("1h")[VALUE_COLUMNS].mean()
    hourly["n_samples"] = d.resample("1h").size()
    hourly["n_valid"] = d["valid"].resample("1h").sum()
    revision = digest({"source": report["sha256"], "site": site.model_dump(mode="json"), "schema": 1})
    observations = []
    for start, row in hourly.iterrows():
        n, nv = int(row.n_samples), int(row.n_valid)
        quality = "missing" if n == 0 else "invalid" if nv < n else "complete" if n == 6 else "incomplete"
        # An incomplete mean is not a full-hour target. Preserve coverage instead.
        values = {c: float(row[c]) if quality == "complete" else None for c in VALUE_COLUMNS}
        end = start + pd.Timedelta(hours=1)
        observations.append(Observation(turbine_id=turbine_id, event_start=start.to_pydatetime(),
            event_end=end.to_pydatetime(), available_at=(end + pd.Timedelta(minutes=site.observation_delay_minutes)).to_pydatetime(),
            n_samples=n, coverage=nv / 6, quality_flag=quality, revision=revision, **values))
    report.update({"processing_timezone": site.timezone, "time_basis": site.time_basis,
                   "timestamp_semantics": site.timestamp_semantics, "revision": revision,
                   "hourly_rows": len(observations), "valid_complete_hours": sum(o.quality_flag == "complete" for o in observations)})
    return tuple(observations), report
