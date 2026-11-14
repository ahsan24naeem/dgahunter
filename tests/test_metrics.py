import numpy as np
import pandas as pd
import pytest

from src.models.metrics import (
    best_f1_threshold, binary_metrics, evaluate_scores, plot_confusion, plot_roc, risk_label,
)


def _pred():
    rows = []

    def add(split, label, family, scores):
        rows.extend({"split": split, "label": label, "family": family, "score": s} for s in scores)

    add("val", 0, "benign", [0.1, 0.2, 0.3])
    add("val", 1, "famA", [0.7, 0.8, 0.9])
    add("test", 0, "benign", [0.1, 0.2, 0.75])
    add("test", 1, "famA", [0.8, 0.9])
    add("test", 1, "famB", [0.6])
    add("holdout", 1, "h1", [0.9, 0.8])
    add("holdout", 1, "h2", [0.5, 0.2])
    return pd.DataFrame(rows)


def test_best_f1_threshold_separates_perfectly_separable_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    assert best_f1_threshold(y, s) == pytest.approx(0.6)


def test_binary_metrics_on_a_hand_computed_example():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.4, 0.9])
    m = binary_metrics(y, s, 0.5, "t")  # predictions: 0 1 0 1
    assert (m["t_tn"], m["t_fp"], m["t_fn"], m["t_tp"]) == (1, 1, 1, 1)
    assert m["t_precision"] == pytest.approx(0.5)
    assert m["t_recall"] == pytest.approx(0.5)
    assert m["t_f1"] == pytest.approx(0.5)
    assert m["t_fpr"] == pytest.approx(0.5)
    assert m["t_roc_auc"] == pytest.approx(0.75)


def test_binary_metrics_threshold_is_inclusive():
    m = binary_metrics(np.array([0, 1]), np.array([0.2, 0.5]), 0.5, "t")
    assert m["t_tp"] == 1 and m["t_fn"] == 0


def test_evaluate_scores_threshold_comes_from_val_only():
    pred = _pred()
    m, thr = evaluate_scores(pred)
    assert thr == pytest.approx(0.7) and m["threshold"] == thr
    changed = pred.copy()
    changed.loc[changed["split"] == "test", "score"] = 0.0
    _, thr2 = evaluate_scores(changed)
    assert thr2 == thr


def test_evaluate_scores_test_metrics_at_val_threshold():
    m, _ = evaluate_scores(_pred())
    # at 0.7: benign .1 .2 .75 -> 0 0 1, DGA .8 .9 .6 -> 1 1 0
    assert (m["test_tp"], m["test_fp"], m["test_tn"], m["test_fn"]) == (2, 1, 2, 1)
    assert m["test_precision"] == pytest.approx(2 / 3)
    assert m["test_recall"] == pytest.approx(2 / 3)


def test_holdout_metrics():
    m, _ = evaluate_scores(_pred())
    assert m["holdout_detection_rate"] == pytest.approx(0.5)
    assert m["holdout_recall_h1"] == pytest.approx(1.0)
    assert m["holdout_recall_h2"] == pytest.approx(0.0)
    # holdout DGA vs test benign: 9.5 of 12 pairs ranked correctly (one tie counts half)
    assert m["holdout_roc_auc"] == pytest.approx(9.5 / 12)


def test_per_family_recall_on_seen_test_families():
    m, _ = evaluate_scores(_pred())
    assert m["test_recall_famA"] == pytest.approx(1.0)
    assert m["test_recall_famB"] == pytest.approx(0.0)
    assert "test_recall_benign" not in m  # only malware families


def test_no_holdout_rows_means_no_holdout_keys():
    pred = _pred()
    m, _ = evaluate_scores(pred[pred["split"] != "holdout"])
    assert not any(k.startswith("holdout") for k in m)


@pytest.mark.parametrize("p,expected", [
    (0.0, "Low"), (0.49, "Low"), (0.5, "Medium"), (0.84, "Medium"), (0.85, "High"), (1.0, "High"),
])
def test_risk_label_boundaries(p, expected):
    assert risk_label(p, [0.5, 0.85]) == expected


def test_plots_write_files(tmp_path):
    pred = _pred()
    plot_confusion(pred, 0.7, "t", tmp_path / "cm.png")
    plot_roc({"m": pred[pred["split"].isin(["val", "test"])]}, tmp_path / "roc.png", "test")
    assert (tmp_path / "cm.png").stat().st_size > 0
    assert (tmp_path / "roc.png").stat().st_size > 0
