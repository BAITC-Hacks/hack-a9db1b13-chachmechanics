"""Operational NOAA GFS 0.25-degree S3 archive. Bounded HTTP Range requests only."""
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import math
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from ..clock import targets
from ..config import Config
from ..schemas import WeatherBundle, WeatherRunMetadata, WeatherValue, digest, utc
from .audit import audit_bundle
from .base import WeatherUnavailable
from .cache import WeatherCache

BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"


def get_bytes(url, start=None, end=None, max_bytes=5_000_000):
    headers = {"User-Agent": "TwinTurbo.ai/0.1 (NOAA archive research)"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers), timeout=40) as response:
                if start is not None:
                    expected_range = f"bytes {start}-{end}/"
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected_range):
                        raise ValueError("Server ignored or changed bounded Range request")
                data = response.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise ValueError("Response exceeds download limit")
                if start is not None and len(data) != end - start + 1:
                    raise ValueError("Truncated GRIB range")
                return data, dict(response.headers)
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise WeatherUnavailable(f"HTTP {exc.code}: {url}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            if attempt == 2:
                raise WeatherUnavailable(f"NETWORK_ERROR: {url}: {exc}") from exc
        time.sleep(0.5 * 2 ** attempt)


def index_ranges(text, wind_height):
    entries = [line.split(":") for line in text.splitlines() if line]
    wanted = {("UGRD", f"{wind_height} m above ground"),
              ("VGRD", f"{wind_height} m above ground"), ("TMP", "2 m above ground")}
    result = []
    for i, fields in enumerate(entries):
        if len(fields) >= 6 and (fields[3], fields[4]) in wanted:
            if i + 1 == len(entries):
                raise ValueError("Missing bounded end offset in index")
            result.append((fields[3], int(fields[1]), int(entries[i + 1][1]) - 1))
    if {v[0] for v in result} != {"UGRD", "VGRD", "TMP"} or len(result) != 3:
        raise ValueError("Required wind/temperature fields not found exactly once")
    return result


def decode_points(payload, turbines, initialized_at, lead, variable, height):
    import eccodes as ec
    handle = ec.codes_new_from_message(payload)
    try:
        actual_init = str(ec.codes_get(handle, "dataDate")) + f"{int(ec.codes_get(handle, 'dataTime')):04d}"
        if actual_init != initialized_at.strftime("%Y%m%d%H%M") or int(ec.codes_get(handle, "endStep")) != lead:
            raise ValueError("GRIB run/lead differs from requested archive")
        if int(ec.codes_get(handle, "stepUnits")) != 1:
            raise ValueError("GRIB step is not in hours")
        if int(ec.codes_get(handle, "level")) != (2 if variable == "TMP" else height):
            raise ValueError("Unexpected GRIB level")
        expected_short = {"TMP": {"2t", "t"}, "UGRD": {"100u", "10u", "u"}, "VGRD": {"100v", "10v", "v"}}
        if ec.codes_get(handle, "shortName") not in expected_short[variable]:
            raise ValueError("Unexpected GRIB variable")
        units = ec.codes_get(handle, "units")
        if units != ("K" if variable == "TMP" else "m s**-1"):
            raise ValueError(f"Unexpected GRIB units: {units}")
        values = {}
        for turbine in turbines:
            point = ec.codes_grib_find_nearest(handle, turbine.latitude, turbine.longitude)[0]
            value = float(point["value"])
            if not math.isfinite(value) or abs(value) > 1e6:
                raise ValueError("Invalid GRIB point value")
            values[turbine.id] = (value, float(point["lat"]), float(point["lon"]))
        return values
    finally:
        ec.codes_release(handle)


class GFSArchive:
    def __init__(self, config: Config, cache: WeatherCache | None = None):
        self.config = config
        self.cache = cache or WeatherCache(config.weather.cache_dir)
        self.last_fetch_events = []

    def fetch_run(self, initialized_at, request):
        init = utc(initialized_at)
        if init.hour % 6 or init.minute or init.second or init.microsecond:
            raise ValueError("GFS initialization must be 00, 06, 12 or 18 UTC")
        site_ids = {t.id for t in self.config.site.turbines}
        if not set(request.turbine_ids) <= site_ids:
            raise ValueError("Unknown turbine")
        chosen = tuple(t for t in self.config.site.turbines if t.id in request.turbine_ids)
        cfg = self.config.weather
        tt = targets(request)
        # Include the final interval boundary as evidence of full coverage.
        first = math.floor((tt[0].target_start - init).total_seconds() / 3600 / cfg.sample_step_hours) * cfg.sample_step_hours
        last = math.ceil((tt[-1].target_end - init).total_seconds() / 3600 / cfg.sample_step_hours) * cfg.sample_step_hours
        if first < 1 or last > 120:
            raise ValueError("This adapter supports positive forecast leads through 120h")
        context = {"init": init.isoformat(), "turbines": [t.model_dump() for t in chosen],
                   "first": first, "last": last, "weather": cfg.model_dump(), "adapter": 1}
        key = "gfs-" + init.strftime("%Y%m%dT%HZ") + "-" + digest(context)[:16]
        for existing in self.cache.bundles():
            if existing.metadata.run_id == key:
                return existing
        sample = {}
        records, modified = [], []
        total = 0
        for lead in range(first, last + 1, cfg.sample_step_hours):
            url = f"{BASE_URL}/gfs.{init:%Y%m%d}/{init:%H}/atmos/gfs.t{init:%H}z.pgrb2.0p25.f{lead:03d}"
            idx, headers = get_bytes(url + ".idx", max_bytes=200_000)
            total += len(idx)
            records.append({"url": url + ".idx", "sha256": self.cache.put_object(idx), "bytes": len(idx)})
            fields = {}
            for variable, start, end in index_ranges(idx.decode(), cfg.wind_height_m):
                if total + end - start + 1 > cfg.max_download_mb_per_run * 1_000_000:
                    raise ValueError("Per-run download budget exceeded")
                payload, headers = get_bytes(url, start, end, max_bytes=10_000_000)
                total += len(payload)
                last_modified = next((v for k, v in headers.items() if k.lower() == "last-modified"), None)
                if last_modified:
                    modified.append(utc(parsedate_to_datetime(last_modified)))
                records.append({"url": url, "range": [start, end], "variable": variable,
                                "sha256": self.cache.put_object(payload), "bytes": len(payload),
                                "last_modified": last_modified})
                fields[variable] = decode_points(payload, chosen, init, lead, variable, cfg.wind_height_m)
            sample[lead] = fields
        values = []
        for hour in range(first, last + 1):
            lower = hour // cfg.sample_step_hours * cfg.sample_step_hours
            upper = min(lower + cfg.sample_step_hours, last)
            fraction = (hour - lower) / cfg.sample_step_hours
            for t in chosen:
                interpolated = {}
                grids = set()
                for name in ("TMP", "UGRD", "VGRD"):
                    lo, la, ln = sample[lower][name][t.id]
                    hi, la2, ln2 = sample[upper][name][t.id]
                    grids.update([(la, ln), (la2, ln2)])
                    interpolated[name] = lo + fraction * (hi - lo)
                if len(grids) != 1:
                    raise ValueError("GRIB variables use different grid points")
                lat, lon = grids.pop()
                u, v = interpolated["UGRD"], interpolated["VGRD"]
                values.append(WeatherValue(turbine_id=t.id, valid_time=init + timedelta(hours=hour),
                    wind_ms=math.hypot(u, v), u_ms=u, v_ms=v, temperature_c=interpolated["TMP"] - 273.15,
                    grid_latitude=lat, grid_longitude=lon))
        # S3 Last-Modified is archive evidence, not a claim about the original dissemination time.
        conservative = init + timedelta(hours=cfg.publication_delay_hours)
        available = max([conservative] + modified)
        basis = "archive_last_modified_plus_delay" if modified else "estimated"
        metadata = WeatherRunMetadata(run_id=key, provider="NOAA GFS on AWS", model="gfs_0p25",
            run_init_time=init, available_at=available, availability_basis=basis,
            provenance="operational_archive", retrieved_at=datetime.now(timezone.utc),
            sha256=digest(records), source_urls=tuple(sorted({r["url"] for r in records})),
            wind_height_m=cfg.wind_height_m, evidence={"files": records, "download_bytes": total,
                "context": context, "publication_delay_hours": cfg.publication_delay_hours,
                "availability_note": "max(archive Last-Modified, initialization + configured delay); not an original publication log",
                "provider_documentation": "https://registry.opendata.aws/noaa-gfs-bdp-pds/"})
        bundle = WeatherBundle(metadata=metadata, values=tuple(values))
        self.cache.save(bundle)
        return bundle

    def select_run(self, request):
        candidates = []
        expected_points = {t.id: (t.latitude, t.longitude) for t in self.config.site.turbines}
        for bundle in self.cache.bundles():
            context = bundle.metadata.evidence.get("context", {})
            if bundle.metadata.provenance == "operational_archive":
                points = {t["id"]: (t["latitude"], t["longitude"]) for t in context.get("turbines", [])}
                if any(points.get(t) != expected_points.get(t) for t in request.turbine_ids):
                    continue
                if bundle.metadata.wind_height_m != self.config.weather.wind_height_m:
                    continue
                if request.origin_time < bundle.metadata.run_init_time + timedelta(hours=self.config.weather.publication_delay_hours):
                    continue
            try:
                audit_bundle(bundle, request, self.config.weather.max_run_age_hours)
            except ValueError:
                continue
            candidates.append(bundle)
        if not candidates:
            raise WeatherUnavailable("WEATHER_UNAVAILABLE: no complete, admissible cached run")
        return max(candidates, key=lambda b: (b.metadata.run_init_time, b.metadata.available_at, b.metadata.run_id))

    def fetch_latest(self, request):
        self.last_fetch_events = []
        delay = timedelta(hours=self.config.weather.publication_delay_hours)
        latest = request.origin_time - delay
        latest = latest.replace(hour=latest.hour // 6 * 6, minute=0, second=0, microsecond=0)
        errors = []
        for age in range(0, self.config.weather.max_run_age_hours + 1, 6):
            init = latest - timedelta(hours=age)
            if request.origin_time - init > timedelta(hours=self.config.weather.max_run_age_hours):
                break
            try:
                bundle = self.fetch_run(init, request)
                audit_bundle(bundle, request, self.config.weather.max_run_age_hours)
                self.last_fetch_events.append({"reason": "FALLBACK_RUN" if errors else "RUN_AVAILABLE",
                    "details": {"run_id": bundle.metadata.run_id, "run_init_time": init.isoformat()}})
                return self.select_run(request)
            except (WeatherUnavailable, ValueError) as exc:
                errors.append(str(exc))
                self.last_fetch_events.append({"reason": "RUN_REJECTED", "details": {
                    "run_init_time": init.isoformat(), "message": str(exc)}})
        raise WeatherUnavailable("WEATHER_UNAVAILABLE: " + "; ".join(errors))
