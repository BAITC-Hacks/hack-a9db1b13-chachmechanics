"""Support the repository's package-dir mapping without an editable install."""
import importlib.util
from pathlib import Path
import sys


def ensure_package():
    if "windoracle" in sys.modules:
        return
    try:
        importlib.import_module("windoracle")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "windoracle":
            raise
    package = Path(__file__).resolve().parents[1] / "src" / "TwinTurbo.ai"
    spec = importlib.util.spec_from_file_location("windoracle", package / "__init__.py", submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["windoracle"] = module
    spec.loader.exec_module(module)
