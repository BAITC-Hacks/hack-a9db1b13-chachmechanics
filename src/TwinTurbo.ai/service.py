"""Single entry point for CLI/UI; model receives only the validated as-of snapshot."""
from datetime import timedelta
from .clock import targets
from .config import Config
from .schemas import AsOfSnapshot, BiasState, ForecastRequest, ForecastResult, ModelState, PredictionBatch, Predictor, digest
from .store import Store
from .weather.audit import audit_bundle
from .weather.base import WeatherProvider


class ForecastService:
    def __init__(self, config: Config, store: Store, weather: WeatherProvider, predictor: Predictor | None = None):
        self.config, self.store, self.weather, self.predictor = config, store, weather, predictor

    def snapshot(self, request: ForecastRequest):
        if not set(request.turbine_ids) <= {t.id for t in self.config.site.turbines}:
            raise ValueError("Unknown turbine")
        weather = self.weather.select_run(request)
        audit_bundle(weather, request, self.config.weather.max_run_age_hours)
        intervals = targets(request)
        required = {(t, h.target_start) for t in request.turbine_ids for h in intervals}
        observations = self.store.observations_as_of(request.origin_time, request.turbine_ids)
        flags = []
        if self.config.site.time_basis == "assumed":
            flags.append("SOURCE_TIME_ASSUMED: " + self.config.site.timezone + "; " + self.config.site.timestamp_semantics)
        if any(o.quality_flag != "complete" for o in observations):
            flags.append("INCOMPLETE_OBSERVATIONS")
        for turbine in request.turbine_ids:
            valid = [o for o in observations if o.turbine_id == turbine and o.quality_flag == "complete"]
            if not valid:
                flags.append("NO_ACTUALS:" + turbine)
            elif request.origin_time - max(o.event_end for o in valid) > timedelta(hours=1):
                flags.append("STALE_ACTUALS:" + turbine)
        return AsOfSnapshot(origin_time=request.origin_time, observations=observations,
            weather_values=tuple(v for v in weather.values if (v.turbine_id, v.valid_time) in required),
            weather_run_metadata=weather.metadata, target_intervals=intervals, quality_flags=tuple(flags))

    def create_forecast(self, request: ForecastRequest, bias: BiasState | None = None,
                        parent_forecast_id: str | None = None):
        if self.predictor is None:
            raise ValueError("MODEL_REQUIRED: attach participant 2's predictor to create forecasts")
        # Revalidate at the boundary, including model_copy() objects from plugins.
        state = ModelState.model_validate(self.predictor.state.model_dump())
        if state.activated_at > request.origin_time:
            raise ValueError("FUTURE_MODEL")
        if state.provenance == "synthetic" and request.mode != "fixture":
            raise ValueError("SYNTHETIC_MODEL_NOT_ALLOWED")
        if bias:
            bias = BiasState.model_validate(bias.model_dump())
            if bias.model_id != state.model_id or bias.created_as_of > request.origin_time:
                raise ValueError("INADMISSIBLE_BIAS")
        if parent_forecast_id:
            parent = self.store.get_forecast(parent_forecast_id)
            parent_request = ForecastRequest.model_validate(parent.manifest["request"])
            if (parent.origin_time > request.origin_time or parent.mode != request.mode
                    or set(parent_request.turbine_ids) != set(request.turbine_ids)
                    or request.release_kind != "update"):
                raise ValueError("Invalid parent forecast")
        snapshot = self.snapshot(request)
        identity = {"request": request.model_dump(mode="json"), "model": state.model_dump(mode="json"),
                    "bias": bias.model_dump(mode="json") if bias else None,
                    "snapshot_hash": digest(snapshot), "config_hash": self.config.config_hash,
                    "parent_forecast_id": parent_forecast_id}
        forecast_id = digest(identity)
        try:
            return self.store.get_forecast(forecast_id)
        except KeyError:
            pass
        batch = PredictionBatch.model_validate(self.predictor.predict(snapshot, bias))
        expected = {(t, h.target_start, h.target_end) for t in request.turbine_ids for h in snapshot.target_intervals}
        actual = [(r.turbine_id, r.target_start, r.target_end) for r in batch.rows]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("MODEL_OUTPUT_COVERAGE: exactly one prediction per turbine/hour required")
        batch = PredictionBatch(rows=tuple(sorted(batch.rows, key=lambda r: (r.turbine_id, r.target_start))))
        metadata = snapshot.weather_run_metadata
        result = ForecastResult(forecast_id=forecast_id, origin_time=request.origin_time,
            predictions=batch, run_id=metadata.run_id, model_id=state.model_id,
            bias_id=bias.bias_id if bias else None, parent_forecast_id=parent_forecast_id,
            warnings=snapshot.quality_flags, provenance=metadata.provenance,
            mode=request.mode, release_kind=request.release_kind,
            manifest={**identity, "weather": metadata.model_dump(mode="json"),
                      "source_revisions": sorted({o.revision for o in snapshot.observations}),
                      "observation_count": len(snapshot.observations),
                      "last_actual_available_at": max((o.available_at.isoformat() for o in snapshot.observations), default=None)})
        self.store.save_model(state)
        if bias:
            self.store.save_bias(bias)
        return self.store.save_forecast(result)

    def get_forecast(self, forecast_id):
        return self.store.get_forecast(forecast_id)

    def list_forecasts(self, as_of=None, mode=None):
        return self.store.forecasts(as_of, mode)

    def events(self, as_of=None):
        return self.store.events(as_of)

    def data_summary(self):
        """UI can inspect dataset bounds without accessing the raw dataframe."""
        return {**self.store.bounds(), "time_basis": self.config.site.time_basis,
                "timezone": self.config.site.timezone,
                "turbines": [t.model_dump() for t in self.config.site.turbines]}

    def compare_forecasts(self, first_id, second_id):
        a, b = self.get_forecast(first_id), self.get_forecast(second_id)
        before = {(r.turbine_id, r.target_start): r for r in a.predictions.rows}
        return [{"turbine_id": r.turbine_id, "target_start": r.target_start.isoformat(),
                 "before": before[(r.turbine_id, r.target_start)].prediction_norm,
                 "after": r.prediction_norm,
                 "delta": r.prediction_norm - before[(r.turbine_id, r.target_start)].prediction_norm}
                for r in b.predictions.rows if (r.turbine_id, r.target_start) in before]

    def export(self, forecast_ids, **kwargs):
        from .export import export_csv
        return export_csv([self.get_forecast(i) for i in forecast_ids], **kwargs)
