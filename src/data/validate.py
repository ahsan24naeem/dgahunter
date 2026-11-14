"""Stage 2 - validate: schema / null / range / balance checks before any training.

Fails the DVC pipeline (exit code 1) if a hard check fails.
Output: reports/validation_raw.json
"""
from __future__ import annotations

import sys

import pandas as pd

from src.utils import DATA_RAW, REPORTS, get_logger, load_params, write_json

log = get_logger("validate")

EXPECTED_SCHEMA = {"domain": "object", "label": "int64", "family": "object", "source": "object"}


def _check(results: list, name: str, passed: bool, detail) -> None:
    results.append({"check": name, "passed": bool(passed), "detail": detail})
    (log.info if passed else log.error)("[%s] %s: %s", "PASS" if passed else "FAIL", name, detail)


def validate_raw(df: pd.DataFrame, p: dict) -> list[dict]:
    r: list[dict] = []
    # --- schema -------------------------------------------------------------
    actual = {c: str(t) for c, t in df.dtypes.items()}
    _check(r, "schema_columns", list(df.columns) == list(EXPECTED_SCHEMA), list(df.columns))
    _check(r, "schema_dtypes", actual == EXPECTED_SCHEMA, actual)
    # --- nulls --------------------------------------------------------------
    null_frac = df.isna().mean().to_dict()
    _check(r, "null_fraction", max(null_frac.values()) <= p["max_null_fraction"], null_frac)
    # --- ranges / domains ---------------------------------------------------
    labels = sorted(df["label"].dropna().unique().tolist())
    _check(r, "label_range", set(labels) <= {0, 1}, labels)
    dom = df["domain"].astype(str).str.strip().str.lower()
    lens = dom.str.len()
    _check(
        r,
        "domain_length_range",
        bool(lens.min() >= 1 and lens.max() <= p["max_domain_len"]),
        {"min": int(lens.min()), "max": int(lens.max()), "mean": round(float(lens.mean()), 2)},
    )
    allowed = set(p["allowed_chars"])
    bad_chars = ~dom.map(lambda s: set(s) <= allowed)
    _check(r, "allowed_characters", bad_chars.mean() <= 0.01,
           {"invalid_fraction": round(float(bad_chars.mean()), 5), "examples": dom[bad_chars].head(5).tolist()})
    has_dot = dom.str.contains(r"\.", regex=True)
    _check(r, "has_tld", has_dot.mean() >= 0.99, {"fraction_with_dot": round(float(has_dot.mean()), 5)})
    # --- volume / balance ---------------------------------------------------
    counts = df["label"].value_counts().to_dict()
    _check(r, "min_rows_per_class",
           all(counts.get(k, 0) >= p["min_rows_per_class"] for k in (0, 1)), counts)
    n_fam = df.loc[df["label"] == 1, "family"].nunique()
    _check(r, "min_dga_families", n_fam >= p["min_families"], n_fam)
    # --- duplicates / label conflicts --------------------------------------
    dup_frac = float(dom.duplicated().mean())
    _check(r, "duplicate_fraction", dup_frac <= p["max_duplicate_fraction"], round(dup_frac, 5))
    conflicts = df.assign(d=dom).groupby("d")["label"].nunique()
    n_conf = int((conflicts > 1).sum())
    _check(r, "label_conflicts_le_0.1pct", n_conf <= 0.001 * len(df), n_conf)
    return r


def validate_splits(splits: dict[str, pd.DataFrame], holdout_families: list[str]) -> list[dict]:
    """Leakage / integrity checks on processed splits (called from preprocess)."""
    r: list[dict] = []
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = len(set(splits[a]["text"]) & set(splits[b]["text"]))
            _check(r, f"no_overlap_{a}_{b}", overlap == 0, overlap)
    train_fams = set(splits["train"]["family"])
    _check(r, "holdout_families_absent_from_train", not (train_fams & set(holdout_families)),
           sorted(train_fams & set(holdout_families)))
    for name, df in splits.items():
        _check(r, f"{name}_no_nulls", int(df.isna().sum().sum()) == 0, int(df.isna().sum().sum()))
        if name != "holdout":
            _check(r, f"{name}_both_classes", df["label"].nunique() == 2, df["label"].value_counts().to_dict())
    return r


def main() -> None:
    p = load_params()["validate"]
    df = pd.read_csv(DATA_RAW / "domains_raw.csv", dtype={"domain": "object", "family": "object", "source": "object"})
    results = validate_raw(df, p)
    passed = all(x["passed"] for x in results)
    write_json({"passed": passed, "n_rows": int(len(df)), "checks": results}, REPORTS / "validation_raw.json")
    if not passed:
        log.error("Validation FAILED - see reports/validation_raw.json")
        sys.exit(1)
    log.info("All %d validation checks passed", len(results))


if __name__ == "__main__":
    main()
