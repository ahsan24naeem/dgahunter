"""Stage 4c - main model: pretrained character-level transformer (CANINE-s) fine-tuned with LoRA.

Tokenizer-free: CANINE reads Unicode code points directly, so no manual features or
sub-word vocabulary. Parameter-efficient fine-tuning (PEFT/LoRA) trains <1% of weights.

Outputs: models/canine_lora/export/{adapter/,meta.json}, predictions/canine_lora.csv,
         metrics/canine_lora.json
Set `lora.enabled: false` in params.yaml to skip on very slow machines (stage still succeeds).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Apple Silicon: CPU fallback for missing ops

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.utils import (  # noqa: E402
    METRICS, MODELS, ROOT, common_tags, dir_size_mb, get_logger, load_params, load_splits,
    read_json, resolve_device, set_seed, setup_mlflow, write_json,
)

log = get_logger("canine_lora")
NAME = "canine_lora"
TINY = "__tiny_random__"  # offline smoke-test model (tests/ and CI only)


def _tokenizer():
    from transformers import CanineTokenizer

    return CanineTokenizer()  # code-point tokenizer, no vocab download needed


def _base_model(model_name: str):
    from transformers import CanineConfig, CanineForSequenceClassification

    if model_name == TINY:
        cfg = CanineConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=128, num_hash_buckets=2048, num_labels=2)
        return CanineForSequenceClassification(cfg)
    return CanineForSequenceClassification.from_pretrained(model_name, num_labels=2)


def _peft(model, p: dict):
    from peft import LoraConfig, TaskType, get_peft_model

    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=p["lora_r"], lora_alpha=p["lora_alpha"],
                     lora_dropout=p["lora_dropout"], target_modules=list(p["target_modules"]))
    return get_peft_model(model, cfg)


@torch.no_grad()
def predict_proba(model, tok, texts: list[str], max_len: int, device: str, batch_size: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i: i + batch_size], padding="longest", truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        logits = model(**enc).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def load_for_inference(export_dir: Path, device: str = "cpu"):
    from peft import PeftModel

    meta = read_json(export_dir / "meta.json")
    base_name = meta["base_model"]
    if base_name == TINY:
        from transformers import CanineForSequenceClassification

        base = CanineForSequenceClassification.from_pretrained(export_dir / "base")
    else:
        base = _base_model(base_name)
    model = PeftModel.from_pretrained(base, export_dir / "adapter").to(device).eval()
    return model, _tokenizer(), meta


def _skip(reason: str) -> None:
    log.warning("Skipping CANINE-LoRA stage: %s", reason)
    out = MODELS / NAME / "export"
    out.mkdir(parents=True, exist_ok=True)
    write_json({"skipped": True, "reason": reason}, out / "meta.json")
    (ROOT / "predictions").mkdir(exist_ok=True)
    pd.DataFrame(columns=["split", "text", "label", "family", "score"]).to_csv(
        ROOT / "predictions" / f"{NAME}.csv", index=False)
    write_json({"skipped": True, "reason": reason}, METRICS / f"{NAME}.json")


def main() -> None:
    from transformers import get_linear_schedule_with_warmup

    from src.models.metrics import evaluate_scores, plot_confusion

    params = load_params()
    p, seed = params["lora"], params["seed"]
    if not p.get("enabled", True):
        _skip("disabled in params.yaml (lora.enabled=false)")
        return
    set_seed(seed)
    device = resolve_device(p["device"])
    if p["device"] == "auto" and device == "mps":
        # PyTorch's MPS backend fails in CANINE's backward pass ("view size is not compatible
        # with input tensor's size and stride"), so 'auto' uses the CPU on Apple Silicon.
        # Set lora.device: mps in params.yaml to force it anyway.
        log.warning("MPS backward pass is unreliable for CANINE -> using CPU (set lora.device=mps to override)")
        device = "cpu"
    splits = load_splits()

    train = splits["train"]
    if p["train_samples"] and p["train_samples"] < len(train):
        frac = p["train_samples"] / len(train)
        train = train.groupby("family", group_keys=False).sample(frac=frac, random_state=seed)
        train = train.sample(frac=1, random_state=seed).reset_index(drop=True)
    log.info("fine-tuning on %d samples, device=%s", len(train), device)

    tok = _tokenizer()
    model = _peft(_base_model(p["model_name"]), p).to(device)
    trainable = sum(t.numel() for t in model.parameters() if t.requires_grad)
    total = sum(t.numel() for t in model.parameters())
    log.info("trainable params %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    opt = torch.optim.AdamW([t for t in model.parameters() if t.requires_grad], lr=p["lr"])
    steps_per_epoch = int(np.ceil(len(train) / p["batch_size"]))
    total_steps = steps_per_epoch * p["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    texts, labels = train["text"].tolist(), torch.tensor(train["label"].values)

    history, step, t0 = [], 0, time.time()
    rng = np.random.default_rng(seed)
    for epoch in range(1, p["epochs"] + 1):
        model.train()
        order = rng.permutation(len(texts))
        run_loss = 0.0
        for b in range(steps_per_epoch):
            idx = order[b * p["batch_size"]: (b + 1) * p["batch_size"]]
            enc = tok([texts[i] for i in idx], padding="longest", truncation=True, max_length=p["max_len"],
                      return_tensors="pt").to(device)
            out = model(**enc, labels=labels[idx].to(device))
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += out.loss.item()
            if step % 25 == 0 or b == steps_per_epoch - 1:
                history.append({"step": step, "train_loss": run_loss / (b + 1)})
                log.info("epoch %d step %d/%d loss %.4f (%.0fs)", epoch, step, total_steps,
                         run_loss / (b + 1), time.time() - t0)
    train_time = time.time() - t0

    frames = []
    for split in ("val", "test", "holdout"):
        d = splits[split]
        frames.append(d.assign(split=split, score=predict_proba(model, tok, d["text"].tolist(), p["max_len"], device)))
    pred = pd.concat(frames, ignore_index=True)
    metrics, thr = evaluate_scores(pred)

    out_dir = MODELS / NAME
    export = out_dir / "export"
    export.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(export / "adapter")
    if p["model_name"] == TINY:
        model.get_base_model().save_pretrained(export / "base")
    write_json({"base_model": p["model_name"], "max_len": p["max_len"], "threshold": thr}, export / "meta.json")

    cpu_model, _, _ = load_for_inference(export, "cpu")
    sample = splits["test"]["text"].head(100).tolist()
    t1 = time.time()
    for t in sample:
        predict_proba(cpu_model, tok, [t], p["max_len"], "cpu")
    metrics.update({
        "latency_ms_per_domain": (time.time() - t1) * 1000 / len(sample),
        "train_time_s": train_time, "trainable_params": trainable, "total_params": total,
        "trainable_pct": 100 * trainable / total, "adapter_size_mb": dir_size_mb(export / "adapter"),
    })
    (ROOT / "predictions").mkdir(exist_ok=True)
    pred[["split", "text", "label", "family", "score"]].to_csv(ROOT / "predictions" / f"{NAME}.csv", index=False)
    cm_path = out_dir / "confusion_test.png"
    plot_confusion(pred, thr, "CANINE-LoRA (test)", cm_path)

    mlflow = setup_mlflow(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=NAME) as run:
        mlflow.set_tags({**common_tags(NAME), "model_family": "canine-s + LoRA", "role": "main",
                         "device": device, "base_model": p["model_name"]})
        mlflow.log_params({f"lora.{k}": v for k, v in p.items()} | {"seed": seed, "n_train": len(train)})
        for h in history:
            mlflow.log_metric("step_train_loss", h["train_loss"], step=h["step"])
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(cm_path))
        from src.models.pyfunc import log_canine_pyfunc

        log_canine_pyfunc(export, params["evaluate"]["risk_thresholds"])
        run_id = run.info.run_id

    write_json({"run_id": run_id, **{k: v for k, v in metrics.items() if not k.startswith(("test_recall_", "holdout_recall_"))}},
               METRICS / f"{NAME}.json")
    log.info("test F1=%.4f ROC-AUC=%.4f holdout DR=%.4f", metrics["test_f1"], metrics["test_roc_auc"],
             metrics.get("holdout_detection_rate", float("nan")))


if __name__ == "__main__":
    main()
