"""Metric helpers shared by every model so comparisons are apples-to-apples.

Protocol (identical for all models):
  * decision threshold is chosen on the VALIDATION split (max F1),
  * test metrics are reported at that fixed threshold,
  * unseen-family generalization = detection rate (recall) on hold-out DGA families at
    that threshold + ROC-AUC of hold-out DGA vs. test benign.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[int(np.nanargmax(f1[:-1]))])


def binary_metrics(y: np.ndarray, s: np.ndarray, thr: float, prefix: str) -> dict[str, float]:
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        f"{prefix}_roc_auc": float(roc_auc_score(y, s)),
        f"{prefix}_pr_auc": float(average_precision_score(y, s)),
        f"{prefix}_precision": float(precision_score(y, pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y, pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y, pred, zero_division=0)),
        f"{prefix}_fpr": float(fp / max(fp + tn, 1)),
        f"{prefix}_tp": int(tp), f"{prefix}_fp": int(fp), f"{prefix}_tn": int(tn), f"{prefix}_fn": int(fn),
    }


def evaluate_scores(pred: pd.DataFrame) -> tuple[dict[str, float], float]:
    """`pred` has columns split,label,family,score (higher score = more DGA-like)."""
    val, test, hold = (pred[pred["split"] == s] for s in ("val", "test", "holdout"))
    thr = best_f1_threshold(val["label"].values, val["score"].values)
    m = {"threshold": thr}
    m.update(binary_metrics(val["label"].values, val["score"].values, thr, "val"))
    m.update(binary_metrics(test["label"].values, test["score"].values, thr, "test"))
    if len(hold):
        m["holdout_detection_rate"] = float((hold["score"].values >= thr).mean())
        benign = test[test["label"] == 0]
        y = np.r_[np.zeros(len(benign)), np.ones(len(hold))]
        s = np.r_[benign["score"].values, hold["score"].values]
        m["holdout_roc_auc"] = float(roc_auc_score(y, s))
        for fam, g in hold.groupby("family"):
            m[f"holdout_recall_{fam}"] = float((g["score"].values >= thr).mean())
    # per-family recall on the test split (seen families)
    for fam, g in test[test["label"] == 1].groupby("family"):
        m[f"test_recall_{fam}"] = float((g["score"].values >= thr).mean())
    return m, thr


def plot_confusion(pred: pd.DataFrame, thr: float, title: str, path) -> None:
    test = pred[pred["split"] == "test"]
    cm = confusion_matrix(test["label"], (test["score"] >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["benign", "DGA"])
    ax.set_yticks([0, 1], ["benign", "DGA"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(curves: dict[str, pd.DataFrame], path, split: str = "test") -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for name, pred in curves.items():
        d = pred[pred["split"] == split]
        fpr, tpr, _ = roc_curve(d["label"], d["score"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(d['label'], d['score']):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("false positive rate (log)")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC - {split} split")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def risk_label(prob: float, thresholds: list[float]) -> str:
    lo, hi = thresholds
    return "Low" if prob < lo else ("Medium" if prob < hi else "High")
