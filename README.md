# jev-phishing — efficient phishing website detection with a decision model

Follow-up to the SLM paper (JCP 2026) and the SVLM paper. The research plan and
ground rules are in `WORKORDER.md`; read it first. Every decision that deviated
from it is logged in `reports/decisions.md`. Task reports are `reports/T<nn>.md`.

## Status (2026-10-04, end of session 1)

| | state |
|---|---|
| Phase 1 code: bootstrap fetch, summariser port, state builders (S0–S3), OCR, Jev client, question bank v1, loaders, tests | **done**, 19 unit tests pass |
| T01 environment and data, T02 states, T03 client smoke test, T05 light determinism | **done** on bootstrap/fixture data (`reports/T01–T05.md`) |
| Scripts for T04, T06, T06b, T07, T08(a–c), T10, T11, T11b, T12, T13, T14 (sampler + transfer), T15 | **written and exercised end to end on the bootstrap samples** with a pseudo split; waiting for the real data (`reports/RUNBOOK.md`) |
| T08d ModernBERT fine-tune | written, **untested** (needs GPU + `torch transformers`) |
| T09 open-weight twin (`t09_open_weight_twin.py`, Ollama / vLLM, GPU), T19 stability, T20 VoI + conformal, T21 prior shift, T_STATS, A5 (v2.3) | written, exercised on the bootstrap samples with cached responses; T09 only with the fake backend |
| T15 SVLM comparison arm, T16, T17 crawler, T18 bundle | **not written** |
| Co-author artefacts (split, stored summaries, SVLM outputs) | **not on this machine** — see "Data to load" |

Jev spend so far: < 0.5 USD (ledger: `data/raw_responses/spend_ledger.jsonl`). Nothing run so far needs a GPU.
**Start with `reports/RUNBOOK.md`**: it lists the exact commands, in order, from "data arrived" to all tables.

## Setup

```bash
git clone <repo> && cd jev-phishing
cp .env.example .env            # put TYPESAFE_API_KEY=... in .env (never commit it)
make env                        # Python 3.11 venv (.venv), pins from pyproject.toml
brew install tesseract          # macOS; apt install tesseract-ocr on Linux (eng is enough for now)
make test                       # 28 tests, no network, no API calls
```

Everything reads `config.yaml` (laptop) or `config.server.yaml` (server);
pass `--config config.server.yaml` to any script. Key settings: `model:
jev-1.13.0` (the API rejects the alias-less `jev-1.13`, decisions D1),
`MAX_USD` (hard spend cap, 1.0 on the laptop, 5.0 on the server), `data_root`,
`SEED: 2107`.

Check the key without spending anything:
`curl https://api.typesafe.ai/v1/models -H "Authorization: Bearer $TYPESAFE_API_KEY"`.

## Data to load

`data/` is git-ignored. Place files as follows (paths relative to `data_root`,
which is `data/` on the laptop):

```
data/
  paper2/                       # co-author artefacts (Georg / Kevin) — NOT YET RECEIVED
    train.csv  test.csv         #   columns _id,label,filename ; 6,160 / 2,631 rows
    test_json/<id>.json         #   stored SVLM-paper summaries (byte-identity check, T02)
    res_<model>.json            #   per-site outputs of the six SVLMs and both frontier models (T06, T11)
    phreshphish_2500_ids.txt    #   the paper's 2,500-site PhreshPhish sample (T14)
  datasets/
    putra/                      # co-author's prepared Putra copy (8,791 sites) — NOT YET RECEIVED
      <id>/index.html           #   post-JavaScript DOM
      <id>/screenshot.jpg       #   one screenshot per site (see note below)
      meta.csv                  #   _id,url,label,language  (1 = phishing)
    putra_zenodo/               # present: phishing.csv, not-phishing.csv, brands.csv from Zenodo 8041387
    phreshphish/                # Hugging Face cache for phreshphish/phreshphish (created on first use)
  derived/                      # everything the scripts produce (states, bootstrap samples, dev ids)
  raw_responses/                # every Jev request/response, cache, spend ledger — COPY THIS TO THE SERVER
  fixture/                      # 20 sites; ids.txt is committed, files are rebuilt with `make fixture`
```

Notes for whoever brings the Putra copy:

- The loaders expect the folder layout above (`src/data/folder.py`). If the
  prepared copy names the screenshot differently or keeps the CSVs separate,
  write a small `scripts/import_putra.py` that produces `meta.csv` and
  `screenshot.jpg` per site (symlinks are fine) — this script does not exist yet.
- The Zenodo record has six screenshots per site
  (`screenshots/{clean,index,original}_js_{on,off}.jpg`). The bootstrap sample
  uses `index_js_on.jpg`. **Please tell us which file the paper used** (decisions D4).
- Never modify files under `data/datasets/`. The test split (2,631 ids) is
  read-only until Phase 2; every read of `test.csv` is logged to
  `results/test_set_access.log`.
- Compare the 50 Zenodo bootstrap ids in `data/derived/putra_dev_ids.txt` and
  the 20 fixture ids in `data/fixture/ids.txt` with `test.csv`; if any overlap,
  rebuild the fixture from training-split sites (D7).

## Running Phase 1 (what exists today)

```bash
# 0. development samples (only needed on a machine without the Putra copy)
make bootstrap                  # 200 PhreshPhish train rows + 50 Putra sites from Zenodo by range requests

# 1. fixture + inventory
make fixture                    # data/fixture/ from the bootstrap Putra sample (or --source <putra copy>)
make t01                        # results/env.txt, results/t01_*.csv

# 2. states for a dataset: S0, S1, S1-b (5 budgets), S2 (if screenshots), S3
.venv/bin/python scripts/t02_states.py --dataset fixture
.venv/bin/python scripts/t02_states.py --dataset bootstrap-phreshphish --variants S0,S1,S1b,S3

# 3. Jev smoke test (reads the states from step 2)
.venv/bin/python scripts/t03_jev_smoke.py --dataset fixture --variant S1 --questions direct   # or bank | structural

# 5. determinism + structure, light version (50 bootstrap sites)
.venv/bin/python scripts/t05_determinism.py --light
```

Datasets known to the scripts: `fixture`, `bootstrap-phreshphish`,
`bootstrap-putra` (`scripts/_bootstrap.py`). Adding the full Putra copy and
the train/test split as datasets is the first job of session 2.

Common flags on every script: `--config`, `--limit N`, `--offset N`,
`--resume`, `--no-cache`. Re-running never re-queries Jev unless `--no-cache`
is given: responses are cached by SHA-256 of (model, bank version, state,
questions) under `data/raw_responses/cache/`.

## Where things land

| what | where |
|---|---|
| states | `data/derived/states/<dataset>/<variant>/<id>.json` |
| raw Jev requests and responses | `data/raw_responses/<experiment>.jsonl` (verbatim, one line per call) |
| spend | `data/raw_responses/spend_ledger.jsonl` (cumulative; `MAX_USD` reads it) |
| per-site stage timing | `results/timing/<experiment>.csv` |
| result tables | `results/<experiment>*.csv` — the only legitimate source of any number in a report |
| reports | `reports/T<nn>.md`, `reports/decisions.md` |

## Server handover (Phase 2 onwards)

`git clone`, `make env`, copy `data/raw_responses/` and `data/derived/` from the
laptop, put the data under the `data_root` of `config.server.yaml`, then run
with `--config config.server.yaml`. Cached responses stay valid: nothing is
re-queried.

## Code map

```
src/summarise/html_truncation.py   SVLM paper's reduce_html, copied unchanged
src/summarise/jcp_truncation.py    JCP repo's tag-based truncation (state S3), copied unchanged
src/summarise/state.py             S0/S1/S1-b/S2/S3 builders, registrable_domain, favicon_host
src/summarise/ocr.py               Tesseract wrapper with timing
src/jev/client.py                  JevClient: pinned model, persistence, cache, retries, cost, MAX_USD
src/questions/bank_v1.yaml         question bank (not frozen yet; freeze before any test-set run)
src/data/                          folder loader, Putra/Zenodo CSVs, PhreshPhish streaming, split port
src/eval/timing.py                 stage timing log
scripts/t00..t05, make_fixture.py  one CLI per task
tests/                             pytest, offline
reference/                         SVLM and JCP paper code for reading, not running
```

## Code map, Phase 2 onwards

```
src/data/registry.py        dataset registry; real split from data/paper2/, else a labelled pseudo split (dev only)
src/jev/run.py              shared runner: one question map x all sites -> decision table CSV
src/eval/metrics.py         F1/P/R/Acc, AUROC, AUPRC, ECE, Brier, PPV@prevalence (bootstrap CI), FP/1000 at 95 % recall
src/eval/plots.py           frontier, reliability, curves, correlation heatmap (PDF + PNG)
src/combine/combiners.py    vote / LR / CatBoost, Platt, isotonic, LOQO, forward selection, learning curve, signals features
src/gate/rules.py           deterministic benign gates
src/cascade/band.py         uncertainty-band cascades (model cascade T11, evidence cascade T11b), band sweep
src/baselines/classical.py  URL features + GBM, TF-IDF + LR, 13-rule set
src/adversarial/inject.py   A1–A4 variants, untrusted_text mitigation
scripts/t04 … t15           one CLI per task; all accept --config --dataset --split --limit --resume --no-cache
```

## Still to build

T09 (open-weight boolean baseline via Ollama/vLLM, GPU), the T15 SVLM comparison
arm, T16 (PhiShark2026, access pending), T17 (Playwright crawler, Andreas's decision),
T18 (Zenodo bundle + `REPRODUCE.md`), `scripts/import_putra.py` (once the co-author's
layout is known), a byte-identity helper for T02, train out-of-fold scores in T07 for
the T11 band sweep (two lines), and `data/paper2/reference_rows.csv` built from the
co-author's `res_<model>.json`.
