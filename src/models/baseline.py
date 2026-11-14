"""Stage 4a - baseline: character n-gram TF-IDF + Logistic Regression.

Comparison-only baseline (classic, non-generative) used to show the gain of the GenAI models.
Outputs: models/baseline/model.joblib, predictions/baseline.csv, metrics/baseline.json
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.metrics import evaluate_scores, plot_confusion
from src.utils import (
    METRICS, MODELS, ROOT, common_tags, dir_size_mb, get_logger, load_params, load_splits,
    set_seed, setup_mlflow, write_json,
)

log = get_logger("baseline")
NAME = "baseline"


def main() -> None:
    params = load_params()
    p, seed = params[NAME], params["seed"]
    set_seed(seed)
    splits = load_splits()
    train = splits["train"]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(p["ngram_min"], p["ngram_max"]),
                                  max_features=p["max_features"], sublinear_tf=True, lowercase=True)),
        ("clf", LogisticRegression(C=p["C"], max_iter=p["max_iter"], random_state=seed)),
    ])
    t0 = time.time()
    pipe.fit(train["text"], train["label"])
    train_time = time.time() - t0
    log.info("trained in %.1fs", train_time)

    frames = []
    for split in ("val", "test", "holdout"):
        d = splits[split]
        frames.append(d.assign(split=split, score=pipe.predict_proba(d["text"])[:, 1]))
    pred = pd.concat(frames, ignore_index=True)
    metrics, thr = evaluate_scores(pred)

    sample = splits["test"]["text"].head(300).tolist()
    t0 = time.time()
    for t in sample:  # single-domain requests, same protocol as the other models
        pipe.predict_proba([t])
    metrics["latency_ms_per_domain"] = (time.time() - t0) * 1000 / len(sample)
    metrics["train_time_s"] = train_time

    out_dir = MODELS / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "threshold": thr}, out_dir / "model.joblib")
    metrics["model_size_mb"] = dir_size_mb(out_dir / "model.joblib")
    (ROOT / "predictions").mkdir(exist_ok=True)
    pred[["split", "text", "label", "family", "score"]].to_csv(ROOT / "predictions" / f"{NAME}.csv", index=False)

    mlflow = setup_mlflow(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=NAME) as run:
        mlflow.set_tags({**common_tags(NAME), "model_family": "char-ngram-logreg", "role": "baseline"})
        mlflow.log_params({f"{NAME}.{k}": v for k, v in p.items()} | {"seed": seed, "n_train": len(train)})
        mlflow.log_metrics(metrics)
        cm_path = out_dir / "confusion_test.png"
        plot_confusion(pred, thr, "Baseline (test)", cm_path)
        mlflow.log_artifact(str(cm_path))
        mlflow.sklearn.log_model(pipe, artifact_path="model",
                                 input_example=np.array(["google", "xjw9qk2h1v"]))
        run_id = run.info.run_id

    write_json({"run_id": run_id, **{k: v for k, v in metrics.items() if not k.startswith(("test_recall_", "holdout_recall_"))}},
               METRICS / f"{NAME}.json")
    log.info("test F1=%.4f ROC-AUC=%.4f holdout DR=%.4f", metrics["test_f1"], metrics["test_roc_auc"],
             metrics.get("holdout_detection_rate", np.nan))


if __name__ == "__main__":
    main()
