"""Compatibility name for the shared windoracle implementation.

Alias modules, rather than importing their source twice: Pydantic contracts and
predictor types must remain the same objects across the team's two namespaces.
Executable modules have tiny wrappers for Python's -m loader.
"""
import importlib
import pkgutil
import sys
import windoracle

__version__ = windoracle.__version__
for _entry in pkgutil.walk_packages(windoracle.__path__, "windoracle."):
    if _entry.name in {"windoracle.__main__", "windoracle.cli", "windoracle.evaluate", "windoracle.backtest"}:
        continue
    _module = importlib.import_module(_entry.name)
    _alias = __name__ + _entry.name[len("windoracle"):]
    sys.modules[_alias] = _module
    _parent, _, _attribute = _alias.rpartition(".")
    setattr(sys.modules[_parent], _attribute, _module)
