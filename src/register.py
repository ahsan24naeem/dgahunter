"""Stage 6 - register: promote the best main model to the MLflow Model Registry (Staging).

Selection rule: among `mlflow.candidates`, the model with the highest VALIDATION ROC-AUC
(test data is never used for model selection).
Output: reports/registry.json
"""
from __future__ import annotations

from mlflow.tracking import MlflowClient

from src.utils import METRICS, REPORTS, get_logger, load_params, read_json, setup_mlflow, write_json

log = get_logger("register")


def main() -> None:
    params = load_params()
    cfg = params["mlflow"]
    mlflow = setup_mlflow(cfg["experiment_name"])

    candidates = {}
    for name in cfg["candidates"]:
        m = read_json(METRICS / f"{name}.json")
        if m.get("skipped"):
            continue
        candidates[name] = m
    best = max(candidates, key=lambda n: candidates[n]["val_roc_auc"])
    m = candidates[best]
    log.info("best candidate: %s (val ROC-AUC %.4f)", best, m["val_roc_auc"])

    model_uri = f"runs:/{m['run_id']}/model"
    mv = mlflow.register_model(model_uri, cfg["registered_model_name"])
    client = MlflowClient()
    client.update_model_version(
        cfg["registered_model_name"], mv.version,
        description=(f"{best}: val ROC-AUC={m['val_roc_auc']:.4f}, test F1={m['test_f1']:.4f}, "
                     f"unseen-family detection={m.get('holdout_detection_rate', float('nan')):.4f}. "
                     "Input: DataFrame with 'domain' column; output: dga_probability + risk label."),
    )
    for k, v in {"model_type": best, "val_roc_auc": f"{m['val_roc_auc']:.4f}",
                 "test_f1": f"{m['test_f1']:.4f}"}.items():
        client.set_model_version_tag(cfg["registered_model_name"], mv.version, k, v)
    client.transition_model_version_stage(
        cfg["registered_model_name"], mv.version, cfg["promote_stage"], archive_existing_versions=True
    )
    try:  # aliases are the MLflow>=2.3 replacement for stages; set both when the server supports it
        client.set_registered_model_alias(cfg["registered_model_name"], cfg["promote_stage"].lower(), mv.version)
    except Exception as e:  # noqa: BLE001
        log.warning("alias not set (%s)", e)

    out = {"registered_model": cfg["registered_model_name"], "version": str(mv.version),
           "stage": cfg["promote_stage"], "model_type": best, "source_run_id": m["run_id"],
           "val_roc_auc": m["val_roc_auc"], "tracking_uri": mlflow.get_tracking_uri()}
    write_json(out, REPORTS / "registry.json")
    log.info("registered %s v%s -> %s", cfg["registered_model_name"], mv.version, cfg["promote_stage"])


if __name__ == "__main__":
    main()
