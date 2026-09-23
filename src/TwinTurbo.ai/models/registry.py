"""Safe JSON model artifacts and the zero-argument CLI predictor factory."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from ..schemas import ModelState, Predictor, digest
from .baseline import ConstantBaselinePredictor
from .ensemble import EnsemblePredictor, WeightSelection
from .ml import RidgePredictor
from .power_curve import PowerCurvePredictor


ARTIFACT_FORMAT = "twinturbo.predictor.v1"
MODEL_ARTIFACT_ENV = "TWINTURBO_MODEL_ARTIFACT"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def predictor_to_dict(predictor: Predictor) -> dict[str, object]:
    serializer = getattr(predictor, "to_dict", None)
    if not callable(serializer):
        raise TypeError("Predictor does not support TwinTurbo.ai JSON artifacts")
    payload = serializer()
    if not isinstance(payload, dict) or not payload.get("kind"):
        raise ValueError("Predictor serializer returned an invalid payload")
    # Round-trip through JSON now so unsupported values and NaN never reach an
    # artifact that appears valid on disk.
    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


def predictor_from_dict(value: Mapping[str, Any]) -> Predictor:
    payload = _mapping(value, "predictor payload")
    kind = payload.get("kind")
    if kind == "power_curve":
        predictor = PowerCurvePredictor.from_dict(payload)
    elif kind == "constant_baseline":
        predictor = ConstantBaselinePredictor.from_dict(payload)
    elif kind == "ridge":
        predictor = RidgePredictor.from_dict(payload)
    elif kind == "ensemble":
        predictor = EnsemblePredictor(
            state=ModelState.model_validate(payload.get("state")),
            twin=predictor_from_dict(_mapping(payload.get("twin"), "ensemble twin")),
            ml=predictor_from_dict(_mapping(payload.get("ml"), "ensemble ml")),
            selection=WeightSelection.from_dict(
                _mapping(payload.get("selection"), "ensemble selection")
            ),
        )
    else:
        raise ValueError(f"Unknown predictor artifact kind: {kind!r}")
    # The CLI requires a concrete ModelState, not merely a look-alike mapping.
    if not isinstance(predictor.state, ModelState):
        raise ValueError("Artifact predictor has no valid ModelState")
    return predictor


def artifact_document(predictor: Predictor) -> dict[str, object]:
    payload = predictor_to_dict(predictor)
    return {
        "format": ARTIFACT_FORMAT,
        "payload_sha256": digest(payload),
        "payload": payload,
    }


def save_predictor(predictor: Predictor, path: str | Path) -> Path:
    """Persist an immutable, content-checked JSON artifact.

    Existing identical content is an idempotent success.  Existing different
    content is never overwritten, matching the shared store's version rules.
    """

    target = Path(path).expanduser().resolve()
    if target.exists() and not target.is_file():
        raise ValueError("Model artifact target is not a file")
    document = artifact_document(predictor)
    body = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") == body:
            return target
        raise FileExistsError(f"Model artifact is immutable: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=target.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(body)
            temporary.flush()
            temporary_name = temporary.name
        Path(temporary_name).replace(target)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return target


def load_predictor(path: str | Path | None = None) -> Predictor:
    """Load a predictor, defaulting to ``TWINTURBO_MODEL_ARTIFACT``.

    This signature is intentionally callable with no arguments so it can be
    passed to ``--predictor windoracle.models.registry:load_predictor``.
    There is no silent model or random-number fallback.
    """

    configured = path if path is not None else os.environ.get(MODEL_ARTIFACT_ENV)
    if configured is None or not str(configured).strip():
        raise ValueError(
            f"MODEL_ARTIFACT_REQUIRED: set {MODEL_ARTIFACT_ENV} to a trained JSON artifact"
        )
    source = Path(configured).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Model artifact not found: {source}")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid model artifact: {source}") from exc
    document = _mapping(document, "artifact")
    if document.get("format") != ARTIFACT_FORMAT:
        raise ValueError("Unsupported model artifact format")
    payload = _mapping(document.get("payload"), "artifact payload")
    expected = document.get("payload_sha256")
    if not isinstance(expected, str) or digest(payload) != expected:
        raise ValueError("MODEL_ARTIFACT_CHECKSUM_MISMATCH")
    return predictor_from_dict(payload)


# Explicit aliases used by notebooks and integration scripts.
save_artifact = save_predictor
load_artifact = load_predictor
