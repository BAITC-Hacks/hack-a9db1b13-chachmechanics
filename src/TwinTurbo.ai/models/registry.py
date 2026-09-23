"""Model artifact I/O, kept outside predict and participant 1's immutable store.

P0 artifacts are JSON. Optional sklearn estimators require an explicitly trusted
pickle payload and the same sklearn version. A checksum detects corruption only.
"""
from dataclasses import asdict
import base64
import json
import os
from pathlib import Path
import pickle
import platform

from ..schemas import ModelState, digest
from .baseline import ConstantBaseline, PersistenceBaseline
from .power_curve import PowerCurvePredictor, TurbineCurve


def _encode(model):
    payload = {"kind": model.kind, "state": model.state.model_dump(mode="json")}
    if isinstance(model, PowerCurvePredictor):
        payload.update(curves={t: asdict(c) for t, c in model.curves.items()},
                       bin_width=model.bin_width, min_bin_count=model.min_bin_count)
    elif isinstance(model, ConstantBaseline):
        payload.update(means=model.means, counts=model.counts)
    elif model.kind == "hist_gradient_boosting":
        import sklearn
        payload.update(estimators=base64.b64encode(pickle.dumps(model.estimators, protocol=5)).decode(),
            sklearn_version=sklearn.__version__, python_version=platform.python_version(),
            counts=model.counts, parameters=model.parameters, calendar_timezone=model.calendar_timezone)
    elif model.kind == "ensemble":
        payload.update(curve=_encode(model.curve), ml=_encode(model.ml), weights=model.weights,
                       validation_counts=model.validation_counts)
    else:
        raise ValueError("UNSUPPORTED_MODEL_TYPE")
    return payload


def _decode(payload, trusted):
    state = ModelState.model_validate(payload["state"])
    kind = payload["kind"]
    if kind == "power_curve":
        curves = {t: TurbineCurve(**{**c, "wind": tuple(c["wind"]), "power": tuple(c["power"]),
                                     "counts": tuple(c["counts"])}) for t, c in payload["curves"].items()}
        return PowerCurvePredictor(state, curves, bin_width=payload["bin_width"], min_bin_count=payload["min_bin_count"])
    if kind in ("constant", "persistence"):
        cls = ConstantBaseline if kind == "constant" else PersistenceBaseline
        return cls(state, payload["means"], payload["counts"])
    if kind == "hist_gradient_boosting":
        if not trusted:
            raise ValueError("TRUSTED_ML_ARTIFACT_REQUIRED: pass trusted=True for your own artifact")
        import sklearn
        from .ml import MLPredictor
        if payload["sklearn_version"] != sklearn.__version__:
            raise ValueError("SKLEARN_VERSION_MISMATCH")
        return MLPredictor(state, pickle.loads(base64.b64decode(payload["estimators"], validate=True)),
            payload["counts"], parameters=payload["parameters"], calendar_timezone=payload["calendar_timezone"])
    if kind == "ensemble":
        from .ensemble import EnsemblePredictor
        return EnsemblePredictor(state, _decode(payload["curve"], trusted), _decode(payload["ml"], trusted),
                                 payload["weights"], payload["validation_counts"])
    raise ValueError("UNSUPPORTED_MODEL_TYPE")


def save_predictor(model, path):
    """Write a new artifact; do not silently replace an existing model version."""
    path = Path(path)
    payload = _encode(model)
    envelope = {"format": "TwinTurbo.ai-model-v1", "sha256": digest(payload), "payload": payload}
    text = json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != text:
            raise ValueError("IMMUTABLE_MODEL_ARTIFACT") from None
    return path


def load_predictor(path=None, *, trusted=False):
    """No-argument factory for --predictor TwinTurbo.ai.models.registry:load_predictor.

    TWINTURBO_AI_MODEL_PATH selects an artifact, default artifacts/models/model.json.
    Set TWINTURBO_AI_TRUSTED_MODEL=1 only for your own optional ML artifact.
    """
    path = Path(path or os.environ.get("TWINTURBO_AI_MODEL_PATH", "artifacts/models/model.json"))
    if not path.is_file():
        raise ValueError("MODEL_ARTIFACT_REQUIRED: " + str(path))
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if envelope.get("format") != "TwinTurbo.ai-model-v1" or digest(envelope["payload"]) != envelope.get("sha256"):
        raise ValueError("MODEL_ARTIFACT_CHECKSUM")
    return _decode(envelope["payload"], trusted or os.environ.get("TWINTURBO_AI_TRUSTED_MODEL") == "1")
