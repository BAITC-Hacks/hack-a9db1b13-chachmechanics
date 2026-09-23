"""SQLite store with immutable observations, forecast versions and event records."""
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from .schemas import ForecastResult, Observation, digest, utc


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS observations (
                    turbine TEXT NOT NULL, start TEXT NOT NULL, available TEXT NOT NULL,
                    revision TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(turbine, start, revision));
                CREATE INDEX IF NOT EXISTS obs_asof ON observations(turbine, available);
                CREATE TABLE IF NOT EXISTS forecasts (
                    id TEXT PRIMARY KEY, origin TEXT NOT NULL, mode TEXT NOT NULL,
                    payload TEXT NOT NULL, checksum TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, event_time TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS imports (
                    id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def ingest(self, observations, report):
        with self.connect() as db:
            for o in observations:
                key = (o.turbine_id, o.event_start.isoformat(), o.revision)
                existing = db.execute("SELECT payload FROM observations WHERE turbine=? AND start=? AND revision=?", key).fetchone()
                body = o.model_dump_json()
                if existing and existing[0] != body:
                    raise ValueError("Observation revision is immutable")
                db.execute("INSERT OR IGNORE INTO observations VALUES(?,?,?,?,?)",
                    (o.turbine_id, key[1], o.available_at.isoformat(), o.revision, body))
            db.execute("INSERT OR IGNORE INTO imports VALUES(?,?)", (digest(report), json.dumps(report)))

    def observations_as_of(self, origin: datetime, turbine_ids):
        placeholders = ",".join("?" for _ in turbine_ids)
        if not placeholders:
            return ()
        with self.connect() as db:
            rows = db.execute(f"SELECT payload FROM observations WHERE turbine IN ({placeholders}) AND available <= ? ORDER BY turbine,start,available,revision",
                              (*turbine_ids, utc(origin).isoformat())).fetchall()
        result = {}
        for (body,) in rows:
            o = Observation.model_validate_json(body)
            key = (o.turbine_id, o.event_start)
            previous = result.get(key)
            if previous and previous.available_at == o.available_at and previous != o:
                raise ValueError("AMBIGUOUS_OBSERVATION_REVISION: specify a corrected availability time")
            result[key] = o
        return tuple(result.values())

    def bounds(self):
        with self.connect() as db:
            row = db.execute("SELECT MIN(start),MAX(start),COUNT(*) FROM observations").fetchone()
        return {"start": row[0], "end": row[1], "rows": row[2]}

    def get_forecast(self, forecast_id):
        with self.connect() as db:
            row = db.execute("SELECT payload,checksum FROM forecasts WHERE id=?", (forecast_id,)).fetchone()
        if row is None:
            raise KeyError(forecast_id)
        result = ForecastResult.model_validate_json(row[0])
        if digest(result) != row[1]:
            raise ValueError("FORECAST_CHECKSUM_MISMATCH")
        return result

    def forecasts(self, as_of=None, mode=None):
        with self.connect() as db:
            rows = db.execute("SELECT id FROM forecasts ORDER BY origin,id").fetchall()
        results = [self.get_forecast(row[0]) for row in rows]
        return [r for r in results if (as_of is None or r.origin_time <= utc(as_of))
                and (mode is None or r.mode == mode)]

    def save_forecast(self, result: ForecastResult):
        event = {"event_time": result.origin_time.isoformat(), "agent": "Orchestrator",
                 "action": "publish", "reason_code": "NEW_RUN" if result.parent_forecast_id else "SCHEDULED",
                 "forecast_id": result.forecast_id, "run_id": result.run_id, "status": "ok"}
        with self.connect() as db:
            row = db.execute("SELECT checksum FROM forecasts WHERE id=?", (result.forecast_id,)).fetchone()
            if row and row[0] != digest(result):
                raise ValueError("Forecast versions are immutable")
            db.execute("INSERT OR IGNORE INTO forecasts VALUES(?,?,?,?,?)",
                       (result.forecast_id, result.origin_time.isoformat(), result.mode,
                        result.model_dump_json(), digest(result)))
            db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?)",
                       (digest(event), event["event_time"], json.dumps(event)))
        return self.get_forecast(result.forecast_id)

    def event(self, time, action, reason_code, **details):
        event = {"event_time": utc(time).isoformat(), "agent": "Orchestrator",
                 "action": action, "reason_code": reason_code, **details}
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?)", (digest(event), event["event_time"], json.dumps(event)))

    def events(self, as_of=None):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM events ORDER BY event_time,id").fetchall()
        events = [json.loads(row[0]) for row in rows]
        return [e for e in events if as_of is None or datetime.fromisoformat(e["event_time"]) <= utc(as_of)]
