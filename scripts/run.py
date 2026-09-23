"""Prepare the local data store and launch the TwinTurbo.ai Streamlit UI."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/site.yaml")
DEFAULT_MODEL = Path("artifacts/models/february-production.json")
DEFAULT_BIAS = Path("artifacts/bias/february-production-bias.json")
DEFAULT_PREDICTOR = "windoracle.models.registry:load_predictor"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Launch TwinTurbo.ai on Streamlit.")
    result.add_argument(
        "--config",
        default=os.environ.get("TWINTURBO_CONFIG", str(DEFAULT_CONFIG)),
    )
    result.add_argument(
        "--model",
        default=os.environ.get("TWINTURBO_MODEL_ARTIFACT", str(DEFAULT_MODEL)),
    )
    result.add_argument(
        "--bias",
        default=os.environ.get("TWINTURBO_BIAS_ARTIFACT", str(DEFAULT_BIAS)),
    )
    result.add_argument(
        "--predictor",
        default=os.environ.get("TWINTURBO_PREDICTOR", DEFAULT_PREDICTOR),
    )
    result.add_argument(
        "--default-mode",
        choices=("fixture", "replay", "submission"),
        default=os.environ.get("TWINTURBO_DEFAULT_MODE", "replay"),
        help="Initial UI mode; users can still switch modes in the dashboard.",
    )
    result.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    result.add_argument("--port", type=_port, default=_port(os.environ.get("PORT", "8501")))
    result.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Launch without the idempotent asset/database validation.",
    )
    result.add_argument(
        "--check",
        action="store_true",
        help="Validate/bootstrap and print the launch contract without starting Streamlit.",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config_path = _resolve(args.config)
    model_path = _resolve(args.model)
    bias_path = _resolve(args.bias)
    if not args.predictor or ":" not in args.predictor:
        print("RUN_ERROR: --predictor must be a trusted module:factory value", file=sys.stderr)
        return 2

    environment = os.environ.copy()
    environment.update(
        {
            "TWINTURBO_CONFIG": str(config_path),
            "TWINTURBO_MODEL_ARTIFACT": str(model_path),
            "TWINTURBO_PREDICTOR": args.predictor,
            "TWINTURBO_DEFAULT_MODE": args.default_mode,
            "PYTHONUNBUFFERED": "1",
            "PORT": str(args.port),
        }
    )
    if bias_path.exists() or os.environ.get("TWINTURBO_BIAS_ARTIFACT"):
        environment["TWINTURBO_BIAS_ARTIFACT"] = str(bias_path)
    else:
        environment.pop("TWINTURBO_BIAS_ARTIFACT", None)

    if not args.skip_bootstrap:
        bootstrap_command = [
            sys.executable,
            str(ROOT / "scripts" / "bootstrap.py"),
            "--config",
            str(config_path),
            "--model",
            str(model_path),
            "--bias",
            str(bias_path),
        ]
        try:
            subprocess.run(bootstrap_command, cwd=ROOT, env=environment, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                "RUN_ERROR: bootstrap failed; fix the actionable BOOTSTRAP_ERROR above and retry.",
                file=sys.stderr,
            )
            return exc.returncode or 2

    if importlib.util.find_spec("streamlit") is None:
        print(
            "RUN_ERROR: Streamlit is not installed. Run "
            "`python -m pip install -r requirements.lock`.",
            file=sys.stderr,
        )
        return 2

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        f"--server.address={args.host}",
        f"--server.port={args.port}",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]
    if args.check:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "environment": {
                        key: environment.get(key)
                        for key in (
                            "TWINTURBO_CONFIG",
                            "TWINTURBO_MODEL_ARTIFACT",
                            "TWINTURBO_BIAS_ARTIFACT",
                            "TWINTURBO_PREDICTOR",
                            "TWINTURBO_DEFAULT_MODE",
                            "PORT",
                        )
                    },
                    "command": command,
                    "health": f"http://{args.host}:{args.port}/_stcore/health",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    os.chdir(ROOT)
    if os.name == "nt":
        try:
            return subprocess.call(command, cwd=ROOT, env=environment)
        except KeyboardInterrupt:
            return 130
    os.execvpe(sys.executable, command, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
