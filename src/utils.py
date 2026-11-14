"""Shared helpers: params, seeding, paths, device selection and MLflow setup."""
from __future__ import annotations

import json
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
METRICS = ROOT / "metrics"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


log = get_logger("utils")


def load_params(path: Path | str = ROOT / "params.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(pref: str = "auto") -> str:
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=float)


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def setup_mlflow(experiment_name: str):
    """Point MLflow at the remote (DagsHub) server from .env / environment.

    Required for the course submission: MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME,
    MLFLOW_TRACKING_PASSWORD. Without them we fall back to a local sqlite store so the
    pipeline is still runnable offline (set REQUIRE_REMOTE_TRACKING=1 to forbid that).
    """
    import mlflow

    load_dotenv()
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        if os.environ.get("REQUIRE_REMOTE_TRACKING") == "1":
            raise RuntimeError("MLFLOW_TRACKING_URI is not set; see .env.example")
        uri = f"sqlite:///{ROOT / 'mlflow.db'}"
        log.warning("MLFLOW_TRACKING_URI not set -> using LOCAL tracking at %s", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    if uri.startswith("sqlite"):
        # keep local artifacts inside the repo (gitignored)
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment_name)
    log.info("MLflow tracking URI: %s", uri)
    return mlflow


def common_tags(stage: str) -> dict[str, str]:
    tags = {"pipeline_stage": stage, "git_sha": git_sha(), "project": "DGAHunter"}
    lock = ROOT / "dvc.lock"
    if lock.exists():
        try:
            data = yaml.safe_load(lock.read_text())
            outs = data.get("stages", {}).get("preprocess", {}).get("outs", [])
            for o in outs:
                if o.get("path") == "data/processed":
                    tags["data_version_md5"] = o.get("md5", "")
        except Exception:
            pass
    return tags


def load_splits(names=("train", "val", "test", "holdout")):
    import pandas as pd

    return {n: pd.read_csv(DATA_PROCESSED / f"{n}.csv", keep_default_na=False) for n in names}


def dir_size_mb(path: Path) -> float:
    path = Path(path)
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return round(sum(p.stat().st_size for p in files) / 1e6, 3)
