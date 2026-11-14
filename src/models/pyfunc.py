"""MLflow pyfunc wrappers: registered models take RAW domains and return a DGA probability
plus a Low/Medium/High risk label, so the Phase-2 API can load them straight from the registry
(`mlflow.pyfunc.load_model("models:/dgahunter-detector/Staging")`)."""
from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.utils import ROOT

EXAMPLE = pd.DataFrame({"domain": ["google.com", "wikipedia.org", "xjw9qk2h1vbz3.net", "kqmfcxwztepl.ru"]})


def _domains(model_input) -> list[str]:
    if isinstance(model_input, pd.DataFrame):
        col = "domain" if "domain" in model_input.columns else model_input.columns[0]
        return model_input[col].astype(str).tolist()
    return [str(x) for x in np.asarray(model_input).ravel()]


def _risk(p: np.ndarray, thr: list[float]) -> list[str]:
    lo, hi = thr
    return ["Low" if x < lo else ("Medium" if x < hi else "High") for x in p]


def _pip(extra: list[str]) -> list[str]:
    import sklearn
    import tldextract
    import torch

    base = [
        f"mlflow=={mlflow.__version__}",
        f"torch=={torch.__version__.split('+')[0]}",
        f"pandas=={pd.__version__}",
        f"numpy=={np.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        f"tldextract=={tldextract.__version__}",
    ]
    return base + extra


class CharLMModel(mlflow.pyfunc.PythonModel):
    def __init__(self, risk_thresholds: list[float]):
        self.risk_thresholds = list(risk_thresholds)

    def load_context(self, context):
        from src.models.charlm import load_checkpoint

        self.model, self.calib, self.thr_prob = load_checkpoint(context.artifacts["checkpoint"], "cpu")

    def predict(self, context, model_input, params=None):
        from src.data.preprocess import domain_to_text
        from src.models.charlm import score_texts

        domains = _domains(model_input)
        texts = [domain_to_text(d.strip().lower()) or d for d in domains]
        score = score_texts(self.model, texts, "cpu")
        prob = self.calib(score)
        return pd.DataFrame({
            "domain": domains, "text": texts, "anomaly_score": score, "dga_probability": prob,
            "is_dga": prob >= self.thr_prob, "risk": _risk(prob, self.risk_thresholds),
        })


def log_charlm_pyfunc(checkpoint: Path, risk_thresholds: list[float]):
    return mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CharLMModel(risk_thresholds),
        artifacts={"checkpoint": str(checkpoint)},
        code_paths=[str(ROOT / "src")],
        pip_requirements=_pip([]),
        input_example=EXAMPLE,
    )


class CanineLoraModel(mlflow.pyfunc.PythonModel):
    def __init__(self, risk_thresholds: list[float]):
        self.risk_thresholds = list(risk_thresholds)

    def load_context(self, context):
        from src.models.canine_lora import load_for_inference

        self.model, self.tok, self.meta = load_for_inference(Path(context.artifacts["model_dir"]), "cpu")

    def predict(self, context, model_input, params=None):
        from src.data.preprocess import domain_to_text
        from src.models.canine_lora import predict_proba

        domains = _domains(model_input)
        texts = [domain_to_text(d.strip().lower()) or d for d in domains]
        prob = predict_proba(self.model, self.tok, texts, self.meta["max_len"], "cpu")
        return pd.DataFrame({
            "domain": domains, "text": texts, "dga_probability": prob,
            "is_dga": prob >= self.meta["threshold"], "risk": _risk(prob, self.risk_thresholds),
        })


def log_canine_pyfunc(model_dir: Path, risk_thresholds: list[float]):
    import peft
    import transformers

    return mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CanineLoraModel(risk_thresholds),
        artifacts={"model_dir": str(model_dir)},
        code_paths=[str(ROOT / "src")],
        pip_requirements=_pip([f"transformers=={transformers.__version__}", f"peft=={peft.__version__}"]),
        input_example=EXAMPLE,
    )
