import pandas as pd

from src.data.preprocess import clean, domain_to_text
from src.data.validate import validate_raw
from src.utils import load_params

P = load_params()["validate"]


def _frame(n=20):
    rows = [{"domain": f"site{i}.com", "label": 0, "family": "benign", "source": "t"} for i in range(n)]
    rows += [{"domain": f"x{i}qz9k{i}.net", "label": 1, "family": f"fam{i % 12}", "source": "n"} for i in range(n)]
    return pd.DataFrame(rows)


def test_validation_passes_on_good_data():
    res = validate_raw(_frame(), {**P, "min_rows_per_class": 10})
    assert all(r["passed"] for r in res), [r for r in res if not r["passed"]]


def test_validation_catches_bad_labels_and_nulls():
    df = _frame()
    df.loc[0, "label"] = 7
    df.loc[1, "domain"] = None
    res = {r["check"]: r["passed"] for r in validate_raw(df, {**P, "min_rows_per_class": 10})}
    assert not res["label_range"] and not res["null_fraction"]


def test_suffix_stripping():
    assert domain_to_text("www.google.co.uk") == "google"
    assert domain_to_text("cvyh1po636.ddns.net") == "cvyh1po636"
    assert domain_to_text("mail.example.com") == "mail.example"


def test_clean_drops_conflicts_and_invalid():
    df = pd.DataFrame({"domain": ["a1b2.com", "a1b2.net", "bad$x.com", "ok.org"],
                       "label": [0, 1, 0, 0], "family": ["benign", "f", "benign", "benign"],
                       "source": ["t"] * 4})
    out = clean(df, set(P["allowed_chars"]), 2, 63)
    assert out["text"].tolist() == ["ok"]
