"""Stage 1 - ingest: download benign (Tranco) and DGA (Netlab-derived) domains.

Output: data/raw/domains_raw.csv  (domain,label,family,source)
        data/raw/ingest_meta.json (sources, checksums, row counts)
"""
from __future__ import annotations

import hashlib
import io
import time
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests

from src.utils import DATA_RAW, get_logger, load_params, write_json

log = get_logger("ingest")


def _download(url: str, timeout: int, retries: int = 3) -> bytes:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            log.info("GET %s (attempt %d)", url, attempt)
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("download failed: %s", e)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {url}: {last_err}")


def load_tranco(url: str, top_n: int, timeout: int) -> tuple[pd.DataFrame, str]:
    blob = _download(url, timeout)
    sha = hashlib.sha256(blob).hexdigest()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(name) as f:
            df = pd.read_csv(f, header=None, names=["rank", "domain"], nrows=top_n)
    df = df.assign(label=0, family="benign", source="tranco")[["domain", "label", "family", "source"]]
    return df, sha


def load_dga_dataset(url: str, timeout: int) -> tuple[pd.DataFrame, str]:
    blob = _download(url, timeout)
    sha = hashlib.sha256(blob).hexdigest()
    df = pd.read_csv(io.BytesIO(blob), header=None, names=["cls", "family", "domain"])
    return df, sha


def main() -> None:
    p = load_params()["ingest"]
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    dga_all, dga_sha = load_dga_dataset(p["dga_url"], p["timeout_s"])
    dga = dga_all[dga_all["cls"] == "dga"].assign(label=1, source="netlab360")
    dga = dga[["domain", "label", "family", "source"]]
    log.info("DGA rows: %d across %d families", len(dga), dga["family"].nunique())

    benign_source = "tranco"
    try:
        benign, benign_sha = load_tranco(p["tranco_url"], p["benign_top_n"], p["timeout_s"])
    except Exception as e:  # noqa: BLE001
        if not p.get("allow_benign_fallback", False):
            raise
        log.warning("Tranco unavailable (%s) -> falling back to Alexa rows of the DGA dataset", e)
        benign_source = "alexa_fallback"
        benign = dga_all[dga_all["cls"] == "legit"].head(p["benign_top_n"])
        benign = benign.assign(label=0, family="benign", source="alexa")[
            ["domain", "label", "family", "source"]
        ]
        benign_sha = dga_sha
    log.info("Benign rows: %d (source=%s)", len(benign), benign_source)

    raw = pd.concat([benign, dga], ignore_index=True)
    out = DATA_RAW / "domains_raw.csv"
    raw.to_csv(out, index=False)

    write_json(
        {
            "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
            "benign_source": benign_source,
            "benign_url": p["tranco_url"] if benign_source == "tranco" else p["dga_url"],
            "benign_sha256": benign_sha,
            "dga_url": p["dga_url"],
            "dga_sha256": dga_sha,
            "rows_total": int(len(raw)),
            "rows_benign": int(len(benign)),
            "rows_dga": int(len(dga)),
            "dga_families": sorted(dga["family"].unique().tolist()),
        },
        DATA_RAW / "ingest_meta.json",
    )
    log.info("Wrote %s (%d rows)", out, len(raw))


if __name__ == "__main__":
    main()
