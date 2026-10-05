# Handover: checking the pipeline, linking the real data, running Phase 1 to 4

For the colleague who has the SVLM-paper artefacts and a machine with the data (laptop or the GPU server). Repository state: commit `07f75d2` (work order v2.3, all scripts exercised on the bootstrap samples, zero Jev spend since 0.19 USD). Nothing has run on the real Putra split yet; the first real signal is the T04 pilot.

## 1. What you receive

| Item | Where | Why |
|---|---|---|
| Repository at `07f75d2` | `git clone`; read `WORKORDER.md` (v2.3), `reports/RUNBOOK.md`, `reports/decisions.md`, `reports/T21.md` (blockers at the top) | design, commands in order, every deviation taken so far |
| `data/raw_responses/` (49 MB) | copy from Andreas's laptop | every Jev request and response so far, the content-addressed cache and the spend ledger; without it the ledger restarts at zero |
| `data/derived/` (83 MB) | copy from the laptop | bootstrap samples, the 200 PhreshPhish dev ids that every transfer sample must exclude, states of the development samples |
| API key | `.env` from `.env.example`, `TYPESAFE_API_KEY=...` (or `OPENROUTER_API_KEY`) | never commit it |
| This file | | checklist below |

## 2. Set up and check the pipeline without the real data (15 minutes, no spend)

```bash
make env                                   # Python 3.11 venv, pins from pyproject.toml
brew install tesseract  |  apt install tesseract-ocr
cp .env.example .env && $EDITOR .env
make test                                  # 28 tests, offline, must pass
curl https://api.typesafe.ai/v1/models -H "Authorization: Bearer $TYPESAFE_API_KEY"   # key works, no spend
.venv/bin/python scripts/t02_states.py --dataset fixture
.venv/bin/python scripts/t03_jev_smoke.py --dataset fixture --variant S1 --questions direct   # 20 cache hits, 0 USD if raw_responses was copied
.venv/bin/python scripts/t07_bank.py --dataset bootstrap-phreshphish --variant S1 --skip-slow  # cache hits; writes results/t07_bootstrap-*
```

Every script prints a banner with the cumulative spend and the cap; a run on bootstrap data says `[BOOTSTRAP — never in a report table]`. If the smoke test reports live calls although `raw_responses/` was copied, stop: the cache did not arrive.

## 3. Link the real data

Paths are relative to `data_root` (`data/` in `config.yaml`; `/data/jev-phishing/data` in `config.server.yaml`, adjust it). Never modify files under `data/datasets/`.

```
data/paper2/train.csv, test.csv        # _id,label,filename ; 6,160 / 2,631 rows (the SVLM paper's split)
data/paper2/test_json/<id>.json        # stored summaries (byte-identity check, T02)
data/paper2/res_<model>.json           # per-site outputs of the six SVLMs and both frontier models
data/paper2/phreshphish_2500_ids.txt   # the paper's PhreshPhish sample ids
data/paper2/reference_rows.csv         # to build: model,mode,dataset,f1,auroc,seconds_per_site,usd_per_pass,source_file
data/datasets/putra/<id>/index.html    # post-JavaScript DOM
data/datasets/putra/<id>/screenshot.jpg
data/datasets/putra/meta.csv           # _id,url,label,language (1 = phishing)
```

If the prepared copy has another layout (CSV per class, screenshot named differently), write `scripts/import_putra.py` that produces `meta.csv` and one `screenshot.jpg` per site (symlinks are fine). Two questions only you can answer: which of the six Zenodo screenshots the SVLM paper used (decisions D4), and whether the 50 ids in `data/derived/putra_dev_ids.txt` and the 20 in `data/fixture/ids.txt` fall into `test.csv` (if any does, rebuild the fixture from training sites: delete `data/fixture/ids.txt`, `make fixture`, record it in `reports/decisions.md`, D7).

Then:

```bash
.venv/bin/python scripts/t01_env_and_data.py        # results/t01_split_status.csv must show 6,160 / 2,631 with match = True
.venv/bin/python scripts/t02_states.py --dataset putra --variants S0,S1,S1b,S2,S3   # ~1 h CPU; S3 is the slow stage
```

Byte-identity check of the summariser (T02): compare `data/derived/states/putra/S1/<id>.json` minus `registrable_domain` and `favicon_host` with 50 of your `test_json/<id>.json`; ten lines of `json.load` and `==`. Record the outcome in `reports/T02.md`.

Freeze the bank before the first test-split read: set `frozen: true` in `src/questions/bank_v1.yaml` and commit.

## 4. Run, in this order (Andreas decides at the two marked points)

```bash
# Phase 1 on the training split (< 0.5 USD)
.venv/bin/python scripts/t04_pilot.py --dataset putra --split train --n 300     # -> reports/T04.md   [GO / NO-GO: Andreas]
.venv/bin/python scripts/t05_determinism.py --dataset putra --variant S1 --n 200
# Phase 2 (first reads of test.csv are logged to results/test_set_access.log)
.venv/bin/python scripts/t06_direct.py --dataset putra --split test --variants S0,S1,S2,S3
.venv/bin/python scripts/t06b_budget.py --dataset putra --split train
.venv/bin/python scripts/t06b_budget.py --dataset putra --split test
.venv/bin/python scripts/t07_bank.py --dataset putra --variant S1
.venv/bin/python scripts/t07_bank.py --dataset putra --variant S2
```

From here `reports/RUNBOOK.md` has every remaining command in the v2.3 order (T19 next, then T08, T09 on the GPU, T10 to T14, T20, T21, T15 with A5, T_STATS, parity). The PhreshPhish samples are drawn with `scripts/t14_sample.py --role stability | adapt | eval | native` (about 6 GB each for the two test-split samples; server). Pass `--config config.server.yaml` on the server.

Spend: the whole plan is about 10 USD; the server config caps at `MAX_USD: 10.0`, the laptop at 1.0. The cap stops the next live call; raising it is Andreas's decision. **[SPEND: Andreas]** before T06 (about 0.6 USD, S3 dominates) and before A5.

## 5. What to send back after each phase

- `results/*.csv`, `results/timing/*.csv`, `figures/*` and the `reports/T<nn>.md` you wrote (they are committed; `data/` is not).
- `data/raw_responses/` (the cache and the ledger; the paper's reproducibility package regenerates every table from it).
- `results/test_set_access.log`.
- Any new entry in `reports/decisions.md` (ambiguities you resolved) and the answers to D4 and D7.

## 6. Things that will bite

- The API accepts only `jev-1.13.0`; `jev-1.13` is rejected (D1). Every response's model field is checked; a mismatch aborts the run.
- Re-runs are free: identical (model, bank, state, questions) are served from the cache. `--no-cache` forces live calls and spends.
- The Putra test split is read only by the Phase 2 scripts; do not open `test.csv` by hand before T04 is a go.
- S3 at 30k tokens is rejected by the API; the budget is 22k (D17). Rejections count as HTTP failures in the T06 table.
- Numbers from `bootstrap-*` and `fixture` files never go into a report; the scripts print the warning.
