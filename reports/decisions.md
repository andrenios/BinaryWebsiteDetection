# Decisions log

Ambiguities resolved by Claude Code under ground rule 11 (keep comparability
with the SVLM paper, write the choice down, continue). Andreas can overturn any
of them; the affected code is named so the change is local.

## 2026-10-04 (session 1, bootstrap phase)

**D1. Model id is `jev-1.13.0`, not `jev-1.13`.** The work order pins
`jev-1.13`. The API rejects that string (`400 {"error_type":"api_usage_error",
"message":"Unknown model: jev-1.13"}`, request `req_01a1068eb07274af882d7c75b9ef9ba2`)
and accepts the versioned id `jev-1.13.0`, which it also echoes in
`response.model`. The docs' jaggedness page calls the release "jev-1.13" and the
models page lists it as `jev-1.13.0`; they are the same release. `config.yaml`
sets `model: jev-1.13.0` and `expected_response_model: jev-1.13.0`; the client
aborts on any other returned model (ground rule 3). Aliases `jev-latest` /
`jev-preview` are never used. Probe files: `data/raw_responses/probe/`.

**D2. Cost is computed client-side.** The response carries `usage.input_tokens`
and `usage.output_tokens` but no `usage.cost`. Cost per request is
`input_tokens x 0.042 / 1e6 USD` (published price; output tokens are free) and
is stored with every record in the JSONL logs and in
`data/raw_responses/spend_ledger.jsonl`, which the `MAX_USD` stop reads across
runs. `config.yaml: price_usd_per_mtok_input`. If TypeSafe later reports cost
in the response, the client should prefer it.

**D3. `t00_bootstrap_sample.py` fixes.** (a) The class regex `no[_-]?phish`
did not match the Zenodo file names `not-phishing_*.zip`, so both classes would
have been labelled phishing; now `not[_-]?phish|no[_-]?phish|benign|legit` -> 0.
(b) Site folders in the Zenodo zips are `<id>/index.html`, `<id>/clean.html`,
`<id>/original.html`, `<id>/asset_details.json`, `<id>/assets/*`,
`<id>/screenshots/*.jpg`; the original grouping by the second-to-last path
component picked up `assets`/`screenshots` instead of the id. (c) The per-class
quota was applied per zip (11 phishing zips x 25); now per class across zips.
(d) The class CSVs are 61 MB and 107 MB, not "small": they are downloaded once
into `data/datasets/putra_zenodo/` and parsed for `_id,url,language,brands`.
(e) PhreshPhish ids are `pp_<sha256[:16]>` using the dataset's own `sha256`
column, and `meta.csv` keeps `sha256` and `target`; the dev-ids file is written
to `data/derived/phreshphish_dev_ids.txt` as the work order says (the script
wrote it next to the sample folder). (f) Only the `train` split is streamed;
the `test` split is PhreshPhish's benchmark split and stays untouched for T14.

**D4. Which Putra screenshot.** Each Zenodo site folder has six screenshots:
`{clean,index,original}_js_{on,off}.jpg`. The work order describes "one .jpg"
per site, so the co-author's copy already chose one. We take
`screenshots/index_js_on.jpg` (the rendered `index.html` with JavaScript on)
and store it as `screenshot.jpg`. **To confirm with Georg/Kevin** which file
their copy uses; if different, re-run `t00` with `PUTRA_SCREENSHOT` changed and
rebuild the fixture. `original.html` (pre-JavaScript source) is also kept
because it enables a raw-versus-rendered probe on Putra later at no extra cost.

**D5. `registrable_domain` uses the PSL including private suffixes.**
`tldextract` with `include_psl_private_domains=True` and the bundled offline
snapshot (no network). Thus `x.blob.core.windows.net`, `user.github.io`,
`site.blogspot.com` are reported as the registrable domain rather than
`windows.net` etc., which is what `q_ip_or_freehost` and `q_form_offsite` need.
IP addresses and suffix-less hosts return the host itself. `src/summarise/state.py`.

**D6. Cache key.** SHA-256 over canonical JSON of `(model, bank_version, state,
questions)` where `questions` is the exact question map sent. The work order
says `(model, state, bank version)`; including the question map makes a
`q_direct`-only request and a full-bank request on the same state distinct
cache entries, which they must be. Cache files are content-addressed under
`data/raw_responses/cache/<k[:2]>/<k>.json` and remain valid on the server.

**D7. Fixture is provisional.** `data/fixture/` is built from the Zenodo
bootstrap sample (25 + 25 sites taken in archive order from
`phishing_0001-0500.zip` and `not-phishing_0001-0500.zip`), not from the
co-author's 50-site sample, and the 20 ids in `data/fixture/ids.txt` were drawn
with `SEED=2107`. Some of these ids may fall into the co-author's test split;
when `train.csv`/`test.csv` arrive, T01 must report the overlap and the fixture
is rebuilt from training-split sites if any overlap exists. All Putra bootstrap
ids are listed in `data/derived/putra_dev_ids.txt`.

**D8. HTTP validation errors (400/422) are recorded, not raised.** They are
written to the experiment JSONL with `ok: false` and counted as HTTP failures
(T06 reports them); 401 stops the run; 429/529/5xx are retried with exponential
backoff (6 attempts, `retry-after` honoured).

**D9. `MAX_USD` is 1.0 for this session** (Andreas, 2026-10-04: "keep total
spend under 1 USD in this session"). The work-order default of 5.0 is kept in
`config.server.yaml`. Raising it is Andreas's decision.

**D10. S3 truncation.** The JCP repo's `truncate_html_to_tokens_merged` is
copied verbatim (`src/summarise/jcp_truncation.py`, from
`sbaresearch/benchmarking-SLMs/truncate_html_functions_github_version.py`).
Its trimming loop re-tokenises the whole document per removed tag, so it can be
slow on very large pages; `t02_states.py` applies a per-site wall-clock timeout
(`--s3-timeout`, default 180 s) and records timeouts in the timing CSV instead
of silently skipping.

**D12. Pilot site selection (T04).** The paper's stratification (language x 3
HTML-length buckets) needs the full training split; `t04_pilot.py` selects a
label-balanced sample after a seeded shuffle within the split. Replace with the
paper's stratification in `data/registry.py::sites_for` once the full data is
here, if Andreas wants exact comparability of the pilot sample.

**D13. Pseudo split for development.** Datasets without `paper2/train.csv`
and `test.csv` get a deterministic 70/30 label-stratified pseudo split
(`data/registry.py`, SEED). Every table produced on it carries
`split_source = pseudo` and is development-only; the scripts print a banner.
On the real data the split comes from the co-author's files and
`split_source = paper2`.

**D14. Evidence-acquisition seconds (T11b) are placeholders.** `config.yaml:
evidence_seconds` (S0 0 s, S5 1 s, S1 4 s, S2 4.5 s) stand in until the fresh
crawl measures fetch and render times; `evidence_seconds_source` is written
into every T11b table and must read "fresh-crawl medians (T17)" before any
T11b number is reported.

**D15. T08d uses ModernBERT-base only** (Andreas, 2026-10-04, option 3):
8k context fits the 1,000-token state; DeBERTa-v3-base dropped (512 tokens,
doubles GPU time). Three seeds. `scripts/t08d_encoder.py`, untested (no GPU).

**D16. Experiment names are shared across tasks on purpose.** T10 and T11b
request `q_direct` under the T06 experiment names, and T11b's LR arm under the
T07 names, so the content-addressed cache serves them and no site is queried
twice for the same (model, state, questions). The JSONL log of the first task
that ran holds the verbatim responses; later tasks log cache hits.

**D17. S3 budget lowered from 30k to 22k tiktoken tokens.** With the
work-order budget, 3 of 60 bootstrap S3 requests were rejected
(`400 {"error_type": "max_tokens_exceeded"}`; `results/t06_bootstrap-phreshphish_test_S3_direct.csv`).
Jev's tokenizer counts about 1.3x tiktoken-gpt-4 on JSON state, so 30k
tiktoken tokens exceed the API's 32k state+question limit. 22k keeps the
largest pages under the limit while preserving the intent (as much raw HTML
as the model accepts). Rejections are still recorded as HTTP failures if they
occur. `config.yaml: s3_max_tokens`.

**D11. Structural-invariance request shape (T05).** The three repeats of
`q_direct` are sent alone (same request shape as T03/T06). The choice and score
variants are sent in one extra request together with `q_direct`, which also
measures whether companion questions move the `q_direct` probability.

## 2026-10-04 (session 3, work order v2.3 implementation; no API calls)

**D18. PhreshPhish split roles and disjointness (T14, T19, T21).** Evaluation
samples come from the benchmark **test** split: `phreshphish-20k` (balanced,
language-stratified, `t14_sample.py --role eval`) and `phreshphish-native`
(seeded thinning of the test split at its native base rate, `--role native`).
Labelled samples come from the **train** split minus the dev ids
(`data/derived/phreshphish_dev_ids.txt` and the bootstrap `meta.csv`):
`phreshphish-stab` (5,000, `--role stability`) and `phreshphish-adapt` (5,700,
`--role adapt --exclude <stability meta.csv>`). The R1 / R2 / R3 adaptation sets
are disjoint slices (200, 500, 5,000) of a seeded permutation of the
adaptation sample, and the whole evaluation sample is scored
(`t14_transfer.py --adapt-dataset phreshphish-adapt`); "disjoint from each
other" in the addendum was read as covering the three adaptation sets too.
`src/data/registry.py`, `src/data/phreshphish.py::DevIds`.

**D19. `tokens_per_site` of the semantic question set is an estimate.** The
semantic-only request was never sent; its billed tokens are estimated as the
measured full-bank tokens scaled by the tiktoken share of the state plus the
kept questions over the state plus all questions
(`scripts/t07_bank.py::estimated_tokens_per_site`); the column
`tokens_source` says so. The deterministic row costs no model tokens. A
measured value needs one extra request per site (Andreas's call).

**D20. VoI conventions (T20).** Myopic one-step rule as specified; transition
model with 20 equal-mass bins (duplicate cut points from two-decimal
probabilities are dropped; the expectation uses the per-cell mean of the
next-level score); c_FP = 1, c_FN in {1, 10, 100}; acquisition costs are the
T11b unit-cost grid in seconds divided by the exchange rate
`voi_seconds_per_fp_unit` = 60 s per unit of c_FP (reported in the table and
to be quoted in the paper); stage scores are Platt-calibrated on train by
default (T13 -> T20); F1 is computed at the train-fitted threshold of the
stopping stage so that the rows are comparable with T11b's swept band.

**D21. Conformal rates (T20).** Thresholds are calibrated per level on the
calibration half of a 50/50 stratified split of train (the other half fits the
combiner and the calibration map). `realised_miss_rate` is the share of all
phishing pages declared benign at or before the level (the marginal quantity
the per-level quantile bounds by alpha), `realised_fp_rate` the share of benign
pages declared phishing at or before the level; both cumulative over levels.
A finite-sample quantile that exceeds the calibration set gives an infinite
threshold, i.e. "never stop at this level" (happens below about 1/alpha
phishing calibration pages; irrelevant on Putra).

**D22. Label-free threshold after prior correction (T21).** "Re-derived for the
same target FPR as on Putra" is implemented from the calibrated shifted
posteriors without labels: expected FPR(t) = sum_{p_i >= t}(1 - p_i) / sum_i
(1 - p_i), threshold = the smallest t whose expected FPR does not exceed the
FPR of the Putra threshold on Putra train benign pages (`prior_known`,
`prior_em`). The variant that keeps the Putra posterior threshold after the
shift is reported as `prior_known_bayes` / `prior_em_bayes`.

**D23. Output columns.** Every new table carries the columns of
`paper/tables.md` first, verbatim and in order; provenance columns follow
(`target`, `score`, `calibration`, `grid_point`, `status`, `split_source`, ...).
Where tables.md names one file but several arms or datasets produce rows,
rows are tagged instead of multiplying files: `t11b_<dataset>_sensitivity.csv`
(rows per `score` and `stages`), `t12_<dataset>_throughput.csv` (`source` =
burst or sustained), `t21_prior_shift.csv` (`target`). The decision model's
files keep the tables.md names; the twin instantiation appends
`_twin-<model>` (`scripts/_bootstrap.py::out_name`).

**D24. A5 spend cap.** The addendum budgets about 1 USD for A5, but 200 pages
x 1,000 queries x 1,300 to 1,900 billed tokens is 11 to 16 USD per target in
the worst case (no early flips). `a5_max_usd_per_target` (config, 1.0 USD)
stops the arm per target; the summary reports the pages attempted before the
stop (`n_pages_attempted`, `arm_usd`). Budgets 50 / 200 / 1,000 are prefixes
of one greedy random search per page (same seed), so the three rows describe
the same attack trajectory. The twin target costs 15 generations per query.

**D25. Twin fake backend.** `t09_open_weight_twin.py --backend fake` is a
deterministic stand-in for exercising the code path without a GPU; it is
refused outside bootstrap / fixture datasets and writes under
`results/dryrun/` (git-ignored). Twin cost is generation seconds x
`gpu_usd_per_hour`; whether Ollama returns first-token logprobs for the
installed version is unverified (fallback: P(yes) from the answer text,
flagged in `notes`).

**D26. Stable-set freeze (T19).** The frozen file
`src/questions/stable_set_v1.yaml` is written only by `--dataset putra
--variant S1` with the decision model, once; later runs load it and warn on a
mismatch. Other datasets and the twin write development files under
`results/`. The stable-set and all-indicator combiners exclude `q_direct` so
that the comparison isolates the indicators; the T07 combiner with `q_direct`
is kept as a fourth row.

**D27. `MAX_USD` in `config.server.yaml` is 10.0** (already above the
addendum's 9 USD); left unchanged, Andreas decides.

**D28. Sustained throughput (T12 `--sustained`)** is computed from the
persisted live records of the T06 S1 test run (completion timestamps and
latencies; wall = last completion - first start). `jev/run.py` now logs the
worker count per request; for older logs the concurrency column falls back to
the config value.

**D29. `data/raw_responses/` and `data/derived/` are tracked in git while the
repository is private** (Andreas, 2026-10-05, for the handover to the
co-author). Ground rule 9 and `CLAUDE.md` said `data/` is never committed; the
exception covers the Jev cache, the spend ledger and the development samples
(bootstrap HTML of 250 PhreshPhish and Putra pages, states). The datasets
themselves (`data/datasets/`) stay out. Before the repository is made public
or the Zenodo bundle is built, the licence question for the page content must
be settled and the folders removed from git history if it is not. The ledger
is append-only: never rebase or rewrite history on `main`.
