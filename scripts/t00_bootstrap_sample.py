#!/usr/bin/env python3
"""
T00: fetch small development samples so the pipeline can be built without the
co-author's data.

Two sources, each optional:

  1. PhreshPhish (Hugging Face, phreshphish/phreshphish): streams the `train`
     split, takes N phishing and N benign, writes them in the Putra folder
     layout. No screenshots exist in this dataset.
  2. Putra (Zenodo record 8041387): downloads ONLY M site folders per class from
     the record's zip files using HTTP range requests (remotezip), so the
     multi-GB archives are never fetched in full. Includes screenshots.

Output layout (identical for both sources, mirrors the SVLM paper's data):

  <out>/<source>/<id>/index.html
  <out>/<source>/<id>/original.html           (Putra only: pre-JavaScript source)
  <out>/<source>/<id>/screenshot.jpg          (Putra only)
  <out>/<source>/meta.csv                      columns: _id,url,label,language,source,target,sha256
  <ids-dir>/<source>_dev_ids.txt               ids used here; T14 must EXCLUDE these

Verified layouts (2026-10-04, see reports/decisions.md D3, D4):
  PhreshPhish rows: sha256, url, label ("phish" | "benign"), target, date, lang, lang_score, html
  Zenodo zips:      phishing_NNNN-NNNN.zip / not-phishing_NNNN-NNNN.zip, inside
                    <id>/index.html, <id>/clean.html, <id>/original.html,
                    <id>/asset_details.json, <id>/assets/*, <id>/screenshots/{clean,index,original}_js_{on,off}.jpg
  Zenodo CSVs:      phishing.csv (5,151 rows), not-phishing.csv (5,244 rows); columns incl. _id,url,language,brands(phishing only)

Usage:
  python scripts/t00_bootstrap_sample.py --out data/derived/bootstrap --phresh 100 --putra 25

Everything fetched here is for building and debugging the pipeline only.
Nothing computed on these samples goes into a report table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from pathlib import Path

PHRESH_NAME = "phreshphish/phreshphish"
PHRESH_SPLIT = "train"            # never the benchmark `test` split
ZENODO_RECORD = "8041387"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
PUTRA_SCREENSHOT = "screenshots/index_js_on.jpg"   # pairs with index.html (post-JS DOM); D4
META_FIELDS = ["_id", "url", "label", "language", "source", "target", "sha256"]

csv.field_size_limit(sys.maxsize)   # Putra CSVs carry long quoted fields (whois_raw_text, features.*)


def log(msg: str) -> None:
    print(f"[t00] {msg}", flush=True)


def write_meta(rows: list[dict], out_dir: Path, ids_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in META_FIELDS})
    ids_path = ids_dir / f"{out_dir.name}_dev_ids.txt"
    with open(ids_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r["_id"] + "\n")
    log(f"wrote {len(rows)} rows to {out_dir / 'meta.csv'} and {ids_path}")


def is_phishing(value) -> bool | None:
    """Map a label value to a boolean. Returns None if unknown."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"phish", "phishing", "malicious", "1", "true", "bad"}:
            return True
        if v in {"benign", "legit", "legitimate", "0", "false", "good"}:
            return False
    return None


def class_from_name(name: str) -> int | None:
    """Zenodo file names: 'phishing_0001-0500.zip' -> 1, 'not-phishing_...' -> 0.
    (The original regex `no[_-]?phish` did not match 'not-phishing'; D3.)"""
    n = name.lower()
    if re.search(r"not[_-]?phish|no[_-]?phish|benign|legit", n):
        return 0
    if re.search(r"phish", n):
        return 1
    return None


# --------------------------------------------------------------------------- #
# 1. PhreshPhish via Hugging Face streaming
# --------------------------------------------------------------------------- #
def fetch_phreshphish(out: Path, ids_dir: Path, n_per_class: int) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        log("datasets not installed; skipping PhreshPhish (pip install datasets)")
        return

    log(f"streaming {PHRESH_NAME} split={PHRESH_SPLIT} ...")
    ds = load_dataset(PHRESH_NAME, split=PHRESH_SPLIT, streaming=True)

    out_dir = out / "phreshphish"
    rows: list[dict] = []
    counts = {True: 0, False: 0}
    seen = 0
    for ex in ds:
        seen += 1
        if seen == 1:
            keys = set(ex.keys())
            need = {"html", "url", "label", "lang", "sha256"}
            if not need <= keys:
                log(f"unexpected schema, keys = {sorted(keys)}; stopping PhreshPhish")
                return
        lab = is_phishing(ex["label"])
        if lab is None or counts[lab] >= n_per_class:
            if all(c >= n_per_class for c in counts.values()):
                break
            continue
        html = ex["html"] or ""
        url = ex["url"] or ""
        if not html.strip() or not url.strip():
            continue
        sha = ex.get("sha256") or hashlib.sha256(html.encode("utf-8", "ignore")).hexdigest()
        sid = "pp_" + sha[:16]
        site_dir = out_dir / sid
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "index.html").write_text(html, encoding="utf-8", errors="ignore")
        rows.append({"_id": sid, "url": url, "label": int(lab),
                     "language": ex.get("lang") or "", "source": "phreshphish",
                     "target": ex.get("target") or "", "sha256": sha})
        counts[lab] += 1
        if (counts[True] + counts[False]) % 25 == 0:
            log(f"  {counts[True]} phishing, {counts[False]} benign after {seen} rows")

    write_meta(rows, out_dir, ids_dir)
    log(f"PhreshPhish done: {counts[True]} phishing, {counts[False]} benign (streamed {seen} rows)")


# --------------------------------------------------------------------------- #
# 2. Putra via Zenodo range requests (partial zip extraction)
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path) -> None:
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def _load_putra_csv(path: Path, label: int) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            sid = row.get("_id")
            if not sid:
                continue
            meta[sid] = {"_id": sid, "url": row.get("url", ""), "label": label,
                         "language": row.get("language", ""), "source": "putra",
                         "target": row.get("brands", "") or "", "sha256": ""}
    return meta


def fetch_putra(out: Path, ids_dir: Path, n_per_class: int, csv_cache: Path, pace_s: float = 1.0) -> None:
    try:
        import requests
        from remotezip import RemoteZip
    except ImportError:
        log("requests/remotezip not installed; skipping Putra (pip install requests remotezip)")
        return

    log(f"listing Zenodo record {ZENODO_RECORD} ...")
    r = requests.get(ZENODO_API, timeout=60)
    if r.status_code != 200:
        log(f"Zenodo API returned {r.status_code}; skipping Putra")
        return
    files = {f["key"]: f for f in r.json().get("files", [])}
    zips = sorted(k for k in files if k.lower().endswith(".zip"))
    csvs = [k for k in files if k.lower().endswith(".csv") and class_from_name(k) is not None]
    if not zips:
        log("no zip files in record; skipping Putra")
        return

    # The two class CSVs are 61 MB and 107 MB: download once into the dataset
    # cache (never modified afterwards) and parse only _id,url,language,brands.
    meta_by_id: dict[str, dict] = {}
    for key in csvs:
        dest = csv_cache / key
        if not dest.exists():
            log(f"downloading {key} ({files[key]['size'] / 1e6:.1f} MB) -> {dest}")
            _download(files[key]["links"]["self"], dest)
        part = _load_putra_csv(dest, class_from_name(key))
        meta_by_id.update(part)
        log(f"  parsed {key}: {len(part)} ids (class {class_from_name(key)})")

    out_dir = out / "putra"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    taken = {0: 0, 1: 0}

    # Resume: sites already on disk (index.html present, known in the CSVs) count first.
    for d in sorted(out_dir.iterdir()):
        if d.is_dir() and (d / "index.html").exists() and d.name in meta_by_id:
            meta = meta_by_id[d.name]
            if taken[meta["label"]] < n_per_class:
                rows.append(meta)
                taken[meta["label"]] += 1
    if rows:
        log(f"resuming: {taken[1]} phishing, {taken[0]} benign already on disk")

    # Zenodo rate-limits bursts of range requests (429): retry with backoff and
    # honour retry-after; pace the per-site reads.
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    retry = Retry(total=10, connect=5, read=5, backoff_factor=2.0, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "HEAD"], respect_retry_after_header=True, raise_on_status=False)
    session.mount("https://", HTTPAdapter(max_retries=retry))

    for key in zips:
        zip_label = class_from_name(key)
        if zip_label is None or taken[zip_label] >= n_per_class:
            continue
        link = files[key]["links"]["self"]
        log(f"opening {key} remotely (central directory only) ...")
        with RemoteZip(link, session=session) as z:
            names = z.namelist()
            by_site: dict[str, set[str]] = {}
            for n in names:
                parts = n.split("/")
                if len(parts) >= 2 and parts[-1]:
                    by_site.setdefault(parts[0], set()).add("/".join(parts[1:]))
            log(f"  {len(by_site)} site folders in archive")
            for sid, members in by_site.items():
                if taken[zip_label] >= n_per_class:
                    break
                if "index.html" not in members:
                    continue
                if (out_dir / sid / "index.html").exists():
                    continue                     # already counted during resume
                meta = meta_by_id.get(sid)
                if meta is None:
                    log(f"  {sid}: not in CSV, skipped")
                    continue
                if meta["label"] != zip_label:
                    log(f"  {sid}: CSV class {meta['label']} != archive class {zip_label}, skipped")
                    continue
                site_dir = out_dir / sid
                site_dir.mkdir(parents=True, exist_ok=True)
                (site_dir / "index.html").write_bytes(z.read(f"{sid}/index.html"))
                if "original.html" in members:
                    (site_dir / "original.html").write_bytes(z.read(f"{sid}/original.html"))
                if PUTRA_SCREENSHOT in members:
                    (site_dir / "screenshot.jpg").write_bytes(z.read(f"{sid}/{PUTRA_SCREENSHOT}"))
                else:
                    log(f"  {sid}: no {PUTRA_SCREENSHOT}")
                rows.append(meta)
                taken[zip_label] += 1
                time.sleep(pace_s)
            log(f"  taken so far: {taken[1]} phishing, {taken[0]} benign")
        if all(v >= n_per_class for v in taken.values()):
            break

    if rows:
        write_meta(rows, out_dir, ids_dir)
    log(f"Putra done: {taken[1]} phishing, {taken[0]} benign")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/derived/bootstrap")
    ap.add_argument("--ids-dir", default="data/derived", help="where <source>_dev_ids.txt is written")
    ap.add_argument("--csv-cache", default="data/datasets/putra_zenodo", help="local copy of the Zenodo CSVs")
    ap.add_argument("--phresh", type=int, default=100, help="rows per class from PhreshPhish (0 = skip)")
    ap.add_argument("--putra", type=int, default=25, help="sites per class from Putra/Zenodo (0 = skip)")
    ap.add_argument("--pace", type=float, default=1.0, help="seconds between Putra sites (Zenodo rate limit)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.phresh > 0:
        fetch_phreshphish(out, Path(args.ids_dir), args.phresh)
    if args.putra > 0:
        fetch_putra(out, Path(args.ids_dir), args.putra, Path(args.csv_cache), args.pace)
    log("finished. data/ is git-ignored; these samples are never committed.")


if __name__ == "__main__":
    main()
