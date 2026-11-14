"""Stage 3 - preprocess: clean, strip public suffix, sample, split (incl. unseen-family hold-out).

The models only ever see raw characters of the registrable label(s) with the public
suffix removed (e.g. 'cvyh1po636.ddns.net' -> 'cvyh1po636'), so they cannot shortcut
on TLDs that differ between benign and DGA sources.

Outputs: data/processed/{train,val,test,holdout}.csv  (text,label,family,domain)
         data/processed/stats.json
         reports/validation_splits.json
"""
from __future__ import annotations

import sys

import pandas as pd
import tldextract
from sklearn.model_selection import train_test_split

from src.data.validate import validate_splits
from src.utils import DATA_PROCESSED, DATA_RAW, REPORTS, get_logger, load_params, set_seed, write_json

log = get_logger("preprocess")

# Offline: uses the public-suffix snapshot bundled with tldextract (no network, reproducible).
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)


def domain_to_text(domain: str) -> str:
    e = _EXTRACT(domain)
    parts = [x for x in (e.subdomain, e.domain) if x]
    text = ".".join(parts)
    if text.startswith("www."):
        text = text[4:]
    return text


def clean(df: pd.DataFrame, allowed: set[str], min_len: int, max_len: int) -> pd.DataFrame:
    df = df.dropna(subset=["domain", "label"]).copy()
    df["domain"] = df["domain"].astype(str).str.strip().str.lower().str.rstrip(".")
    df = df[df["domain"].map(lambda s: set(s) <= allowed)]
    df["text"] = df["domain"].map(domain_to_text)
    df = df[df["text"].str.len().between(min_len, max_len)]
    # drop texts that appear under both labels, then de-duplicate
    conflict = df.groupby("text")["label"].nunique()
    df = df[~df["text"].isin(conflict[conflict > 1].index)]
    return df.drop_duplicates(subset="text").reset_index(drop=True)


def main() -> None:
    params = load_params()
    p, seed = params["preprocess"], params["seed"]
    set_seed(seed)
    raw = pd.read_csv(DATA_RAW / "domains_raw.csv")
    df = clean(raw, set(params["validate"]["allowed_chars"]), p["min_label_len"], p["max_label_len"])
    log.info("After cleaning: %d rows (from %d)", len(df), len(raw))

    benign = df[df["label"] == 0]
    benign = benign.sample(n=min(p["benign_sample"], len(benign)), random_state=seed)
    dga = df[df["label"] == 1]
    hold_mask = dga["family"].isin(p["holdout_families"])
    seen = (
        dga[~hold_mask]
        .groupby("family", group_keys=False)
        .apply(lambda g: g.sample(n=min(p["dga_per_family"], len(g)), random_state=seed))
    )
    holdout = (
        dga[hold_mask]
        .groupby("family", group_keys=False)
        .apply(lambda g: g.sample(n=min(p["holdout_per_family"], len(g)), random_state=seed))
    )

    pool = pd.concat([benign, seen], ignore_index=True)
    train, tmp = train_test_split(
        pool, test_size=p["val_size"] + p["test_size"], stratify=pool["family"], random_state=seed
    )
    rel_test = p["test_size"] / (p["val_size"] + p["test_size"])
    val, test = train_test_split(tmp, test_size=rel_test, stratify=tmp["family"], random_state=seed)

    cols = ["text", "label", "family", "domain"]
    splits = {
        "train": train[cols].sample(frac=1, random_state=seed).reset_index(drop=True),
        "val": val[cols].reset_index(drop=True),
        "test": test[cols].reset_index(drop=True),
        "holdout": holdout[cols].reset_index(drop=True),
    }
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, s in splits.items():
        s.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)
        log.info("%-8s %6d rows | DGA share %.3f", name, len(s), s["label"].mean())

    checks = validate_splits(splits, p["holdout_families"])
    passed = all(c["passed"] for c in checks)
    write_json({"passed": passed, "checks": checks}, REPORTS / "validation_splits.json")
    write_json(
        {
            name: {
                "rows": int(len(s)),
                "dga_fraction": round(float(s["label"].mean()), 4),
                "families": int(s["family"].nunique()),
                "mean_text_len": round(float(s["text"].str.len().mean()), 2),
            }
            for name, s in splits.items()
        },
        DATA_PROCESSED / "stats.json",
    )
    if not passed:
        log.error("Split validation FAILED - see reports/validation_splits.json")
        sys.exit(1)


if __name__ == "__main__":
    main()
