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

**D11. Structural-invariance request shape (T05).** The three repeats of
`q_direct` are sent alone (same request shape as T03/T06). The choice and score
variants are sent in one extra request together with `q_direct`, which also
measures whether companion questions move the `q_direct` probability.
