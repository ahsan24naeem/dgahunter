"""Stage 5 - evaluate: compare baseline vs. main models and build the hybrid score.

Hybrid = logistic-regression stack over [char-LM log-perplexity, classifier logit], fitted on
the validation split (out-of-fold predictions on val, so threshold selection stays honest).
Outputs: reports/model_comparison.{json,md}, reports/roc_test.png, reports/roc_holdout.png,
         metrics/evaluation.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from src.models.metrics import evaluate_scores, plot_roc
from src.utils import (
    METRICS, REPORTS, ROOT, common_tags, get_logger, load_params, read_json, set_seed, setup_mlflow,
    write_json,
)

log = get_logger("evaluate")
KEY_METRICS = ["test_roc_auc", "test_pr_auc", "test_precision", "test_recall", "test_f1", "test_fpr",
               "holdout_detection_rate", "holdout_roc_auc", "val_roc_auc"]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build_hybrid(lm: pd.DataFrame, clf: pd.DataFrame, seed: int) -> pd.DataFrame:
    assert (lm["text"].values == clf["text"].values).all(), "prediction files are not aligned"
    X = np.c_[lm["score"].values, _logit(clf["score"].values)]
    y = lm["label"].values
    is_val = (lm["split"] == "val").values
    stack = LogisticRegression(random_state=seed)
    score = np.empty(len(lm))
    score[is_val] = cross_val_predict(stack, X[is_val], y[is_val], cv=5, method="predict_proba")[:, 1]
    stack.fit(X[is_val], y[is_val])
    score[~is_val] = stack.predict_proba(X[~is_val])[:, 1]
    return lm.assign(score=score), stack


def main() -> None:
    params = load_params()
    seed = params["seed"]
    set_seed(seed)
    preds = {}
    for name in ("baseline", "charlm", "canine_lora"):
        df = pd.read_csv(ROOT / "predictions" / f"{name}.csv", keep_default_na=False)
        if len(df):
            preds[name] = df
        else:
            log.warning("%s has no predictions (stage skipped)", name)

    classifier = "canine_lora" if "canine_lora" in preds else "baseline"
    preds["hybrid"], stack = build_hybrid(preds["charlm"], preds[classifier], seed)

    rows, full = [], {}
    for name, pred in preds.items():
        m, _ = evaluate_scores(pred)
        full[name] = m
        rows.append({"model": name, **{k: m.get(k) for k in KEY_METRICS}})
    table = pd.DataFrame(rows).set_index("model")
    for name in preds:
        mf = METRICS / f"{name}.json"
        if mf.exists():
            extra = read_json(mf)
            table.loc[name, "latency_ms"] = extra.get("latency_ms_per_domain")
            table.loc[name, "size_mb"] = extra.get("model_size_mb", extra.get("adapter_size_mb"))
    REPORTS.mkdir(exist_ok=True)
    md = table.round(4).to_markdown()
    (REPORTS / "model_comparison.md").write_text(
        "# Model comparison (test split, threshold tuned on val)\n\n" + md
        + f"\n\nHybrid = LR stack of char-LM log-perplexity + {classifier} logit "
          f"(coef={np.round(stack.coef_[0], 3).tolist()}).\n"
          "holdout_* = unseen DGA families (" + ", ".join(params["preprocess"]["holdout_families"]) + ").\n")
    write_json(full, REPORTS / "model_comparison.json")
    plot_roc(preds, REPORTS / "roc_test.png", "test")
    # hold-out ROC: unseen-family DGAs vs. test benign
    hold = {n: pd.concat([p[(p.split == "test") & (p.label == 0)], p[p.split == "holdout"]]).assign(split="h")
            for n, p in preds.items()}
    plot_roc(hold, REPORTS / "roc_holdout.png", "h")

    best = table["test_roc_auc"].idxmax()
    summary = {"best_model_by_test_roc_auc": best, "hybrid_classifier_component": classifier,
               **{f"{n}_{k}": float(table.loc[n, k]) for n in table.index for k in ("test_roc_auc", "test_f1",
                                                                                      "holdout_detection_rate")}}
    mlflow = setup_mlflow(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="evaluation_hybrid") as run:
        mlflow.set_tags({**common_tags("evaluate"), "role": "comparison", "hybrid_classifier": classifier})
        mlflow.log_params({"hybrid.components": f"charlm+{classifier}", "seed": seed})
        mlflow.log_metrics({f"hybrid_{k}": v for k, v in full["hybrid"].items()})
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, float)})
        for f in ("model_comparison.md", "model_comparison.json", "roc_test.png", "roc_holdout.png"):
            mlflow.log_artifact(str(REPORTS / f))
        summary["run_id"] = run.info.run_id
    write_json(summary, METRICS / "evaluation.json")
    log.info("\n%s", md)


if __name__ == "__main__":
    main()
