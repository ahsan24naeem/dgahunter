"""Stage 4b - main model: character-level generative language model (GPT-style).

Self-supervised: trained ONLY on benign domains to model "what a normal domain looks like".
A domain's mean per-character negative log-likelihood (log-perplexity) is its anomaly score;
a 1-D Platt calibration fitted on the validation split maps it to a DGA probability.
Per-character surprise is exposed for analyst explanations.

Outputs: models/charlm/model.pt, predictions/charlm.csv, metrics/charlm.json,
         reports/charlm_training_curve.png
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Apple Silicon: CPU fallback for missing ops

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.utils import (  # noqa: E402
    METRICS, MODELS, REPORTS, ROOT, common_tags, dir_size_mb, get_logger, load_params, load_splits,
    resolve_device, set_seed, setup_mlflow, write_json,
)

log = get_logger("charlm")
NAME = "charlm"

PAD, BOS, EOS, UNK = 0, 1, 2, 3
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789.-_"
STOI = {c: i + 4 for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB_SIZE = len(CHARS) + 4


def encode(text: str, max_len: int) -> list[int]:
    ids = [STOI.get(c, UNK) for c in text.lower()[: max_len - 2]]
    return [BOS, *ids, EOS]


def batchify(texts: list[str], max_len: int) -> torch.Tensor:
    seqs = [encode(t, max_len) for t in texts]
    T = max(len(s) for s in seqs)
    out = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out


@dataclass
class CharLMConfig:
    max_len: int = 64
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE


class CharGPT(nn.Module):
    def __init__(self, cfg: CharLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, 4 * cfg.d_model, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:  # GPT-2 style init keeps initial logits small
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        h = self.tok(x) + self.pos(pos)[None]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.blocks(h, mask=causal, src_key_padding_mask=(x == PAD))
        return self.head(self.ln(h))

    def token_nll(self, x: torch.Tensor) -> torch.Tensor:
        """Per-position NLL of predicting x[:,1:] (PAD positions = 0)."""
        logits = self(x[:, :-1])
        tgt = x[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                              ignore_index=PAD, reduction="none").view(tgt.shape)
        return nll * (tgt != PAD)


@torch.no_grad()
def score_texts(model: CharGPT, texts: list[str], device: str, batch_size: int = 1024) -> np.ndarray:
    """Mean per-token NLL (= log-perplexity). Higher = more surprising = more DGA-like."""
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        x = batchify(texts[i: i + batch_size], model.cfg.max_len).to(device)
        nll = model.token_nll(x)
        n = (x[:, 1:] != PAD).sum(1).clamp(min=1)
        out.append((nll.sum(1) / n).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


@torch.no_grad()
def char_surprise(model: CharGPT, text: str, device: str = "cpu") -> list[tuple[str, float]]:
    """Per-character surprise (NLL in nats) - which part of the domain looked unnatural."""
    model.eval()
    x = batchify([text], model.cfg.max_len).to(device)
    nll = model.token_nll(x)[0].cpu().numpy()
    chars = list(text.lower()[: model.cfg.max_len - 2]) + ["<eos>"]
    return [(c, float(v)) for c, v in zip(chars, nll)]


class Calibrator:
    """1-D Platt scaling: anomaly score -> P(DGA)."""

    def __init__(self, coef: float = 1.0, intercept: float = 0.0):
        self.coef, self.intercept = coef, intercept

    def fit(self, s: np.ndarray, y: np.ndarray) -> "Calibrator":
        lr = LogisticRegression().fit(s.reshape(-1, 1), y)
        self.coef, self.intercept = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.coef * np.asarray(s) + self.intercept)))


def load_checkpoint(path, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = CharGPT(CharLMConfig(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, Calibrator(**ck["calibration"]), ck["threshold_prob"]


# ----------------------------------------------------------------------------- training
def train(params: dict, splits: dict, device: str):
    p = params[NAME]
    cfg = CharLMConfig(max_len=p["max_len"], d_model=p["d_model"], n_layers=p["n_layers"],
                       n_heads=p["n_heads"], dropout=p["dropout"])
    model = CharGPT(cfg).to(device)
    n_params = sum(t.numel() for t in model.parameters())
    log.info("CharGPT params: %s on %s", f"{n_params:,}", device)

    benign_train = splits["train"].query("label == 0")["text"].tolist()
    benign_val = splits["val"].query("label == 0")["text"].tolist()
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    steps_per_epoch = math.ceil(len(benign_train) / p["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], total_steps=steps_per_epoch * p["epochs"])
    rng = np.random.default_rng(params["seed"])

    history, best_ppl, best_state = [], float("inf"), None
    for epoch in range(1, p["epochs"] + 1):
        model.train()
        order = rng.permutation(len(benign_train))
        tot, ntok, t0 = 0.0, 0, time.time()
        for b in range(steps_per_epoch):
            idx = order[b * p["batch_size"]: (b + 1) * p["batch_size"]]
            x = batchify([benign_train[i] for i in idx], cfg.max_len).to(device)
            nll = model.token_nll(x)
            n = (x[:, 1:] != PAD).sum()
            loss = nll.sum() / n
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * n.item()
            ntok += n.item()
        train_ppl = math.exp(tot / ntok)
        val_ppl = math.exp(float(np.mean(score_texts(model, benign_val, device))))
        history.append({"epoch": epoch, "train_ppl": train_ppl, "val_benign_ppl": val_ppl,
                        "epoch_time_s": time.time() - t0})
        log.info("epoch %d | train ppl %.3f | val benign ppl %.3f | %.1fs", epoch, train_ppl, val_ppl,
                 time.time() - t0)
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, cfg, history, n_params


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.models.metrics import evaluate_scores, plot_confusion

    params = load_params()
    p, seed = params[NAME], params["seed"]
    set_seed(seed)
    device = resolve_device(p["device"])
    splits = load_splits()

    t0 = time.time()
    model, cfg, history, n_params = train(params, splits, device)
    train_time = time.time() - t0

    frames = []
    for split in ("val", "test", "holdout"):
        d = splits[split]
        frames.append(d.assign(split=split, score=score_texts(model, d["text"].tolist(), device)))
    pred = pd.concat(frames, ignore_index=True)
    metrics, thr = evaluate_scores(pred)  # thresholds on raw anomaly score
    val = pred[pred["split"] == "val"]
    calib = Calibrator().fit(val["score"].values, val["label"].values)
    thr_prob = float(calib(np.array([thr]))[0])

    metrics.update({
        "val_benign_perplexity": history[-1]["val_benign_ppl"] if history else float("nan"),
        "best_val_benign_perplexity": min(h["val_benign_ppl"] for h in history),
        "test_benign_perplexity": math.exp(pred.query("split=='test' and label==0")["score"].mean()),
        "test_dga_perplexity": math.exp(pred.query("split=='test' and label==1")["score"].mean()),
        "n_params": n_params, "train_time_s": train_time, "threshold_prob": thr_prob,
    })

    # CPU latency for a single-domain request (serving scenario)
    cpu_model = CharGPT(cfg)
    cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    sample = splits["test"]["text"].head(300).tolist()
    t1 = time.time()
    for t in sample:
        score_texts(cpu_model, [t], "cpu")
    metrics["latency_ms_per_domain"] = (time.time() - t1) * 1000 / len(sample)

    out_dir = MODELS / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model.pt"
    torch.save({"config": asdict(cfg), "state_dict": cpu_model.state_dict(),
                "calibration": {"coef": calib.coef, "intercept": calib.intercept},
                "threshold_score": thr, "threshold_prob": thr_prob}, ckpt)
    metrics["model_size_mb"] = dir_size_mb(ckpt)
    (ROOT / "predictions").mkdir(exist_ok=True)
    pred[["split", "text", "label", "family", "score"]].to_csv(ROOT / "predictions" / f"{NAME}.csv", index=False)

    REPORTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot([h["epoch"] for h in history], [h["train_ppl"] for h in history], "o-", label="train (benign)")
    ax.plot([h["epoch"] for h in history], [h["val_benign_ppl"] for h in history], "s-", label="val (benign)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("perplexity")
    ax.set_title("Char-LM training curve")
    ax.legend()
    fig.tight_layout()
    curve = REPORTS / "charlm_training_curve.png"
    fig.savefig(curve, dpi=120)
    plt.close(fig)
    cm_path = out_dir / "confusion_test.png"
    plot_confusion(pred, thr, "Char-LM (test)", cm_path)

    examples = {t: char_surprise(cpu_model, t) for t in ["google", "wikipedia", "xjw9qk2h1vbz3", "kqmfcxwztepl"]}
    write_json(examples, out_dir / "char_surprise_examples.json")

    mlflow = setup_mlflow(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=NAME) as run:
        mlflow.set_tags({**common_tags(NAME), "model_family": "char-gpt-lm", "role": "main",
                         "training": "self-supervised on benign only", "device": device})
        mlflow.log_params({f"{NAME}.{k}": v for k, v in p.items()} | {"seed": seed})
        for h in history:
            mlflow.log_metrics({"epoch_train_ppl": h["train_ppl"], "epoch_val_benign_ppl": h["val_benign_ppl"]},
                               step=h["epoch"])
        mlflow.log_metrics(metrics)
        for f in (curve, cm_path, out_dir / "char_surprise_examples.json"):
            mlflow.log_artifact(str(f))
        from src.models.pyfunc import log_charlm_pyfunc

        log_charlm_pyfunc(ckpt, params["evaluate"]["risk_thresholds"])
        run_id = run.info.run_id

    write_json({"run_id": run_id, **{k: v for k, v in metrics.items() if not k.startswith(("test_recall_", "holdout_recall_"))}},
               METRICS / f"{NAME}.json")
    log.info("test F1=%.4f ROC-AUC=%.4f holdout DR=%.4f", metrics["test_f1"], metrics["test_roc_auc"],
             metrics.get("holdout_detection_rate", float("nan")))


if __name__ == "__main__":
    main()
