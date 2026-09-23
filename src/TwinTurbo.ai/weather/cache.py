"""Content-addressed raw data plus immutable, validated run manifests."""
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from ..schemas import WeatherBundle, digest


def atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class WeatherCache:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "objects").mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        (self.root / "requests").mkdir(exist_ok=True)

    def request_key(self, url, start=None, end=None):
        return {"url": url, "start": start, "end": end}

    def get_request(self, url, start=None, end=None):
        key = self.request_key(url, start, end)
        path = self.root / "requests" / (digest(key) + ".json")
        if not path.exists():
            return None
        envelope = json.loads(path.read_text(encoding="utf-8"))
        record = envelope["record"]
        if digest(record) != envelope["checksum"] or record["request"] != key:
            raise ValueError("CACHE_REQUEST_CHECKSUM_MISMATCH")
        payload = self.get_object(record["sha256"])
        if len(payload) != record["bytes"]:
            raise ValueError("CACHE_REQUEST_LENGTH_MISMATCH")
        if start is not None and len(payload) != end - start + 1:
            raise ValueError("CACHE_RANGE_LENGTH_MISMATCH")
        return payload, record["headers"]

    def save_request(self, url, start, end, payload, headers):
        key = self.request_key(url, start, end)
        existing = self.get_request(url, start, end)
        if existing is not None:
            if existing[0] != payload:
                raise ValueError("Cached archive response is immutable")
            return
        record = {"request": key, "sha256": self.put_object(payload), "bytes": len(payload),
                  "headers": {k: v for k, v in headers.items()
                              if k.lower() in ("last-modified", "etag", "content-range", "content-length")}}
        atomic_write(self.root / "requests" / (digest(key) + ".json"),
                     json.dumps({"record": record, "checksum": digest(record)}).encode())

    def put_object(self, payload: bytes) -> str:
        checksum = sha256(payload).hexdigest()
        path = self.root / "objects" / checksum
        if path.exists():
            self.get_object(checksum)
        else:
            atomic_write(path, payload)
        return checksum

    def get_object(self, checksum: str) -> bytes:
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("Invalid object checksum")
        data = (self.root / "objects" / checksum).read_bytes()
        if sha256(data).hexdigest() != checksum:
            raise ValueError("CACHE_CHECKSUM_MISMATCH")
        return data

    def save(self, bundle: WeatherBundle):
        path = self.root / "runs" / (digest(bundle.metadata.run_id) + ".json")
        if path.exists():
            existing = self.load(path)
            if existing != bundle:
                raise ValueError("Existing run is immutable")
            return
        body = bundle.model_dump(mode="json")
        envelope = {"bundle": body, "checksum": digest(body)}
        atomic_write(path, json.dumps(envelope, ensure_ascii=False, allow_nan=False).encode())

    def load(self, path: Path) -> WeatherBundle:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if digest(envelope["bundle"]) != envelope["checksum"]:
            raise ValueError("CACHE_MANIFEST_CHECKSUM_MISMATCH")
        bundle = WeatherBundle.model_validate(envelope["bundle"])
        files = bundle.metadata.evidence.get("files", [])
        for record in files:
            self.get_object(record["sha256"])
        if files and digest(files) != bundle.metadata.sha256:
            raise ValueError("CACHE_EVIDENCE_CHECKSUM_MISMATCH")
        if bundle.metadata.provenance == "operational_archive" and not files:
            raise ValueError("Operational cache requires raw evidence")
        return bundle

    def bundles(self):
        for path in sorted((self.root / "runs").glob("*.json")):
            yield self.load(path)
