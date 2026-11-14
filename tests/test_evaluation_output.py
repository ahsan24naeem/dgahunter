import json
import re
from pathlib import Path

import pytest

EVAL_PATH = Path("metrics/evaluation.json")
COMPARISON_PATH = Path("reports/model_comparison.json")

MODELS = ["baseline", "charlm", "canine_lora", "hybrid"]
SUFFIXES = ["test_roc_auc", "test_f1", "holdout_detection_rate"]


@pytest.fixture(scope="module")
def ev():
    assert EVAL_PATH.exists(), f"{EVAL_PATH} missing, run `dvc repro` first"
    return json.loads(EVAL_PATH.read_text())


def test_every_model_has_all_three_metrics(ev):
    missing = [f"{m}_{s}" for m in MODELS for s in SUFFIXES if f"{m}_{s}" not in ev]
    assert not missing, missing


def test_metrics_are_floats_in_unit_range(ev):
    for m in MODELS:
        for s in SUFFIXES:
            v = ev[f"{m}_{s}"]
            assert isinstance(v, float), f"{m}_{s}"
            assert 0.0 <= v <= 1.0, f"{m}_{s}={v}"


def test_models_beat_random_guessing(ev):
    for m in MODELS:
        assert ev[f"{m}_test_roc_auc"] > 0.5, m


def test_best_model_is_the_argmax_over_all_four_models(ev):
    # evaluate.py takes idxmax over the whole table, so the hybrid can win
    best = max(MODELS, key=lambda m: ev[f"{m}_test_roc_auc"])
    assert ev["best_model_by_test_roc_auc"] == best


def test_hybrid_component_is_a_known_classifier(ev):
    assert ev["hybrid_classifier_component"] in {"baseline", "canine_lora"}


def test_run_id_looks_like_an_mlflow_run_id(ev):
    assert re.fullmatch(r"[0-9a-f]{32}", ev["run_id"])


def test_no_unexpected_keys(ev):
    allowed = {f"{m}_{s}" for m in MODELS for s in SUFFIXES}
    allowed |= {"best_model_by_test_roc_auc", "hybrid_classifier_component", "run_id"}
    assert set(ev) == allowed


@pytest.mark.skipif(not COMPARISON_PATH.exists(), reason="run `dvc repro` to create reports/model_comparison.json")
def test_summary_matches_the_full_comparison_report(ev):
    full = json.loads(COMPARISON_PATH.read_text())
    for m in MODELS:
        for s in SUFFIXES:
            assert ev[f"{m}_{s}"] == pytest.approx(full[m][s])
