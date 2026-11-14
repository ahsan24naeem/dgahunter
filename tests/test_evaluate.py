import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluate import _logit, build_hybrid


def _frames(seed=0, n_val=100, n_test=60, n_hold=30):
    rng = np.random.default_rng(seed)
    split = np.array(["val"] * n_val + ["test"] * n_test + ["holdout"] * n_hold)
    label = np.r_[np.arange(n_val) % 2, np.arange(n_test) % 2, np.ones(n_hold, dtype=int)]
    base = pd.DataFrame({"text": [f"d{i}" for i in range(len(split))], "label": label,
                         "split": split, "family": "x"})
    lm = base.assign(score=label * 1.5 + rng.normal(0, 1, len(split)))
    clf = base.assign(score=1 / (1 + np.exp(-(label * 3 - 1.5 + rng.normal(0, 1, len(split))))))
    return lm, clf


def test_logit_basics():
    assert _logit(np.array([0.5]))[0] == pytest.approx(0.0)
    out = _logit(np.array([0.0, 1.0]))
    assert np.isfinite(out).all() and out[0] == pytest.approx(-out[1])
    p = np.linspace(0.01, 0.99, 20)
    assert (np.diff(_logit(p)) > 0).all()


def test_hybrid_keeps_rows_and_returns_probabilities():
    lm, clf = _frames()
    df, stack = build_hybrid(lm, clf, 0)
    assert len(df) == len(lm)
    assert (df["text"].values == lm["text"].values).all()
    assert (df["label"].values == lm["label"].values).all()
    assert ((df["score"] >= 0) & (df["score"] <= 1)).all()
    assert stack.coef_.shape == (1, 2)


def test_hybrid_rejects_misaligned_prediction_files():
    lm, clf = _frames()
    with pytest.raises(AssertionError):
        build_hybrid(lm, clf.iloc[::-1].reset_index(drop=True), 0)


def test_hybrid_is_deterministic_for_a_seed():
    lm, clf = _frames()
    a, _ = build_hybrid(lm, clf, 0)
    b, _ = build_hybrid(lm, clf, 0)
    assert np.array_equal(a["score"].values, b["score"].values)


def test_hybrid_uses_both_signals_with_positive_weight():
    lm, clf = _frames()
    _, stack = build_hybrid(lm, clf, 0)
    assert (stack.coef_[0] > 0).all()  # both features point toward DGA


def test_hybrid_ranks_test_split_well():
    lm, clf = _frames()
    df, _ = build_hybrid(lm, clf, 0)
    t = df[df["split"] == "test"]
    assert roc_auc_score(t["label"], t["score"]) > 0.8


def test_val_scores_are_out_of_fold_and_others_use_the_fitted_stack():
    lm, clf = _frames()
    df, stack = build_hybrid(lm, clf, 0)
    X = np.c_[lm["score"].values, _logit(clf["score"].values)]
    is_val = (lm["split"] == "val").values
    assert np.allclose(df["score"].values[~is_val], stack.predict_proba(X[~is_val])[:, 1])
    assert not np.allclose(df["score"].values[is_val], stack.predict_proba(X[is_val])[:, 1])


def test_test_and_holdout_labels_never_influence_the_stack():
    lm, clf = _frames()
    base, _ = build_hybrid(lm, clf, 0)
    flipped_lm, flipped_clf = lm.copy(), clf.copy()
    non_val = flipped_lm["split"] != "val"
    flipped_lm.loc[non_val, "label"] = 1 - flipped_lm.loc[non_val, "label"]
    flipped_clf.loc[non_val, "label"] = 1 - flipped_clf.loc[non_val, "label"]
    other, _ = build_hybrid(flipped_lm, flipped_clf, 0)
    assert np.allclose(base["score"].values, other["score"].values)
