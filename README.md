# DGAHunter — GenAI detection of algorithmically generated (DGA) malware domains

**Course:** MLOps — Phase 1 (reproducible core ML pipeline)
**Team:** Ahsan Naeem (23L-0517) · Aaiza Iqbal (23L-0616) — Section 7A

Malware uses Domain Generation Algorithms (DGAs) to spin up thousands of random-looking
command-and-control domains, so static blocklists fall behind. DGAHunter learns what a *normal*
domain looks like directly from raw characters and flags domains that do not fit:

| Role | Model | Idea |
|---|---|---|
| Baseline | char n-gram TF-IDF + Logistic Regression | classic, non-generative reference |
| Main (generative) | **Char-level GPT language model** (~0.6M params, PyTorch, trained from scratch on benign domains only) | log-perplexity = anomaly score; per-character surprise explains *which part* looked unnatural |
| Main (fine-tuned LLM) | **CANINE-s + LoRA** (Hugging Face Transformers + PEFT, tokenizer-free) | binary benign/DGA classifier, <1% of weights trained |
| Hybrid | LR stack of LM perplexity + classifier logit | combined score |

All models are evaluated identically: threshold tuned on the validation split, precision /
recall / F1 / ROC-AUC / PR-AUC / FPR on the test split, plus an **unseen-family hold-out**
(families `matsnu`, `pykspa`, `symmi` never appear in training), CPU latency and model size.

## Pipeline

```
ingest ─► validate ─► preprocess ─┬─► train_baseline ───┐
                                  ├─► train_charlm ─────┼─► evaluate ─► register (MLflow Registry → Staging)
                                  └─► train_canine_lora ┘
```

| Stage | What it does | Main outputs (DVC-tracked) |
|---|---|---|
| `ingest` | Downloads the Tranco top-1M list (benign) and the 25-family DGA dataset extracted from the 360 Netlab feed (pinned commit). Records SHA-256 checksums. | `data/raw/` |
| `validate` | Schema, dtypes, nulls, label range, domain length range, allowed characters, class volume, #families, duplicates, label conflicts. **Fails the pipeline** on any violation. | `reports/validation_raw.json` |
| `preprocess` | Lower-cases, strips the public suffix offline (`google.co.uk → google`, so models can't cheat on TLDs), de-duplicates, samples, stratified train/val/test split, separate unseen-family hold-out; then leakage checks (no overlap between splits, hold-out families absent from train). | `data/processed/`, `reports/validation_splits.json` |
| `train_baseline` | char 1–4-gram TF-IDF + LogReg | `models/baseline/`, `metrics/baseline.json` |
| `train_charlm` | GPT-style char LM, benign-only, Platt-calibrated to a probability; logged as an MLflow **pyfunc** that accepts raw domains | `models/charlm/`, `metrics/charlm.json` |
| `train_canine_lora` | LoRA fine-tune of `google/canine-s`; logged as pyfunc | `models/canine_lora/`, `metrics/canine_lora.json` |
| `evaluate` | Comparison table, hybrid stack, ROC curves (test + unseen families) | `reports/model_comparison.md`, `reports/roc_*.png`, `metrics/evaluation.json` |
| `register` | Picks the main model with the best **validation** ROC-AUC, registers it as `dgahunter-detector`, transitions it to **Staging** | `reports/registry.json` |

Every stage logs parameters, metrics and artifacts to **MLflow on DagsHub**; all hyper-parameters
live in `params.yaml`, so changing one re-runs only the affected stages.

## Folder structure

```
.
├── dvc.yaml / dvc.lock        # pipeline definition + exact versions of every input/output
├── params.yaml                # all data, model and tracking parameters
├── requirements.txt
├── .env.example               # MLflow/DagsHub credentials template (copy to .env)
├── src/
│   ├── utils.py               # params, seeding, MLflow setup, paths
│   ├── data/
│   │   ├── ingest.py          # stage: ingest
│   │   ├── validate.py        # stage: validate (+ split leakage checks)
│   │   └── preprocess.py      # stage: preprocess
│   ├── models/
│   │   ├── baseline.py        # stage: train_baseline
│   │   ├── charlm.py          # stage: train_charlm (model + training + per-char surprise)
│   │   ├── canine_lora.py     # stage: train_canine_lora
│   │   ├── metrics.py         # shared evaluation protocol
│   │   └── pyfunc.py          # MLflow pyfunc wrappers (raw domain in → probability + risk out)
│   ├── evaluate.py            # stage: evaluate (+ hybrid)
│   └── register.py            # stage: register (Model Registry → Staging)
├── tests/                     # pytest unit tests (validation, preprocessing, models)
├── metrics/                   # small JSON metrics (git-tracked, `dvc metrics show`)
├── reports/                   # validation reports, comparison table, plots
├── data/  models/  predictions/   # DVC-managed, git-ignored
```

## Setup

Python **3.11 or 3.12** (3.11 recommended; macOS 12+ on Apple Silicon). Python 3.13 is not supported by the pinned PyTorch build.

```bash
git clone https://github.com/<github-user>/<repo-name>.git
cd <repo-name>
python3.11 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No Python 3.11 installed (or your default `python` is 3.13)? Use conda instead:

```bash
conda create -n dgahunter python=3.11 -y
conda activate dgahunter
pip install -r requirements.txt
```

### Credentials (DagsHub = DVC remote + MLflow server)

1. On DagsHub: **Create → New repository → Connect a repository → GitHub**, pick this repo.
2. DagsHub → *User settings → Tokens*: copy your token.
3. Configure MLflow:
   ```bash
   cp .env.example .env      # then fill in user, repo name and token
   ```
4. Configure the DVC remote (credentials go to `.dvc/config.local`, which is never committed):
   ```bash
   dvc remote add -d origin https://dagshub.com/<dagshub-user>/<repo-name>.dvc   # already in .dvc/config
   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user <dagshub-user>
   dvc remote modify origin --local password <dagshub-token>
   ```

## How to run

```bash
dvc pull            # (optional) fetch the exact data/models of the last committed run
dvc repro           # run / resume the full pipeline (only changed stages re-run)
dvc repro --force   # re-run everything from scratch (re-downloads and retrains)
dvc push            # upload data + models to the DagsHub DVC remote
git add dvc.lock metrics reports && git commit -m "pipeline run" && git push
```

Useful commands:

```bash
dvc dag                     # pipeline diagram
dvc metrics show            # metrics of all models
dvc params diff             # what changed vs. last commit
dvc exp run -S charlm.epochs=6 -S lora.lora_r=16    # experiment with new params
pytest -q                   # unit tests
```

Runtime on a laptop CPU: ingest+preprocess ≈ 1 min, baseline ≈ 1–2 min, char-LM ≈ 5–10 min,
CANINE-LoRA ≈ 15–30 min (much faster on Apple-Silicon `mps` or a CUDA GPU; picked automatically).
On a very slow machine set `lora.enabled: false` (or lower `lora.train_samples`) in `params.yaml`;
the stage then succeeds without training and the hybrid falls back to char-LM + baseline.

### Using the registered model

```python
import mlflow, pandas as pd
model = mlflow.pyfunc.load_model("models:/dgahunter-detector/Staging")
model.predict(pd.DataFrame({"domain": ["google.com", "xjw9qk2h1vbz3.net"]}))
#   domain  text  anomaly_score  dga_probability  is_dga  risk
```

## Data sources

* **Benign:** [Tranco](https://tranco-list.eu/) top-1M list (top 200k used). If Tranco is
  unreachable the ingest stage falls back to the Alexa benign rows of the dataset below and records
  this in `data/raw/ingest_meta.json` (`benign_source`).
* **DGA:** 337,500 domains from 25 malware families (13,500 each) extracted from the 360 Netlab
  DGA feed — Cucchiarelli et al., *Algorithmically generated malicious domain names detection based
  on n-grams features*, Expert Systems with Applications 170 (2021);
  [chrmor/DGA_domains_dataset](https://github.com/chrmor/DGA_domains_dataset) (pinned commit).

Raw data and model weights are **never committed to git** — they live in the DVC remote.

## Roadmap (later phases)

Docker images (GPU training / slim CPU serving with ONNX Runtime), FastAPI `/score` endpoint,
AWS (S3, ECR, EC2/ECS), GitHub Actions CI/CD, Evidently drift monitoring and scheduled retraining.
