# Work order for Claude Code: efficient phishing website detection with a decision model (Jev)

Version 2.1 (2026-10-04). Supersedes v1. Changes: efficiency is now the organising axis; state schema uses the real field names of the SVLM paper's summariser (`html_truncation_github.py`); new tasks for evidence-budget sweeps, question-count sweeps, a rule gate, raw-versus-rendered HTML, and pipeline-stage timing; Section 1 now lists data sources and the laptop-to-server workflow.

**How to start a Claude Code session with this file:** place it in the repo root as `WORKORDER.md`, add a one-line `CLAUDE.md` saying "Read WORKORDER.md first; follow its ground rules; work on the lowest unfinished task; write the task report before stopping." Then begin with: "Start with T01. Data is at `<path>`. The co-author artefacts are / are not yet available."

Project: follow-up to the SLM paper (JCP 2026, repo `sbaresearch/benchmarking-SLMs`) and the SVLM paper (code in `svlm_zenodo.zip`, Zenodo 10.5281/zenodo.22812951).
Owner: Andreas Ekelhart. Claude Code implements and runs experiments; Andreas makes every spend, data-access and publication decision.

---

## 0. Ground rules (read before every session)

1. No number in any report may come from anywhere except a file under `results/` produced by a script in this repo from raw responses under `data/raw_responses/`. If a value cannot be traced to a raw response, it does not exist.
2. Every Jev request and response is persisted verbatim (JSON lines per experiment, including `model`, `usage.cost`, client-side latency, request id, state hash). Re-runs reuse cached responses unless `--no-cache` is passed.
3. Pin the model: `jev-1.13` (never `jev-latest`). Log the returned `model` field with every decision. If it differs from the pinned one, abort and report.
4. Everything is resumable and idempotent: `--limit`, `--offset`, `--resume`, checkpoints every 200 sites, no duplicate API calls after a restart.
5. Spend cap: hard `MAX_USD` in `config.yaml` (start at 5 USD). Scripts stop when cumulative `usage.cost` exceeds it. Any increase is Andreas's decision.
6. Test-set discipline: the Putra test split (2,631 site ids) is read-only for development. Question design, combiner fitting, threshold and calibration fitting, budget and question-count selection use the training split or its development subset only. Log every test-set access.
7. Fixture first: every new script runs on `data/fixture/` (20 sites, 10 phishing, 10 benign, fixed ids) before any full run.
8. Fixed seeds (`SEED = 2107`). Record library versions in `results/env.txt`.
9. Never modify files under `data/datasets/`. Derived data goes under `data/derived/`.
10. Report format per task: `reports/T<nn>.md` with what ran, how many sites, cost, wall time, result tables (markdown), paths to CSVs and figures, open issues. One paragraph of interpretation at most.
11. If something is ambiguous, choose the option that keeps comparability with the SVLM paper, write the choice into `reports/decisions.md`, and continue. Stop to ask only for spend or test-set access.
12. Time everything. Every pipeline stage (fetch, render, summarise, OCR, decide, combine) records wall time per site into `results/timing/`. Efficiency claims are built from these files, not from estimates.

---

## 1. Inputs, data sources, and the laptop-to-server workflow

### 1.1 Secrets
`JEV_API_KEY` (TypeSafe) or `OPENROUTER_API_KEY` (model id `typesafe/jev-1.13`) in `.env`, never committed. `.env.example` lists every variable.

### 1.2 Data sources

| Dataset | Source | Contents | Use |
|---|---|---|---|
| Putra (2023) | Zenodo record 8041387 (also on Kaggle); the co-author's prepared copy is preferred | per site a folder with `index.html` (post-JavaScript DOM) and one `.jpg` screenshot; CSVs with `_id`, `url`, `language`, label | primary set; identical to the SVLM paper |
| SVLM paper artefacts | co-author (Georg / Kevin); not in `svlm_zenodo.zip`, which holds code only | cleaned 8,791-site list; `train.csv` / `test.csv` (`_id,label,filename`, 6,160 / 2,631); `test_json/<id>.json` summaries; `res_<model>.json` per-site outputs for six SVLMs and both frontier models (direct original, direct tuned, feature extraction); 2,500 PhreshPhish sample ids | comparability; all baseline rows; cascade (T11) without GPU |
| PhreshPhish | Hugging Face `phreshphish/phreshphish` (Dalton et al. 2025, updated Feb 2026), >660k rows, URL + HTML, 52 languages, no screenshots; ships benchmark splits with adjusted base rates | transfer set (T14); use its realistic-base-rate benchmark split in addition to the simulated prevalence of T13 |
| PhiShark2026 | controlled access, https://phishark2026dataset.com, institutional application | 67,502 scans, rendered HTML, screenshots, redirects, TLS, DNS, WHOIS; collected Apr to Aug 2026 | second transfer set (T16); leakage argument |
| Fresh crawl | own crawler (T17) | raw GET body and rendered DOM per page | render-or-not (RQ2); post-model-release evaluation |

Do not re-download Putra if the co-author's prepared copy is available: the prepared copy carries the exact cleaning (black/white screenshot removal, 46 sites without files) and the exact split, and regenerating them from Zenodo risks a different 8,791 and a different 2,631. Download only PhreshPhish fresh (`datasets.load_dataset("phreshphish/phreshphish")`, cache under `data/datasets/phreshphish/`). Until the co-author's artefacts arrive, regenerate summaries with the ported summariser and recreate the split with the `select_sample_gtihub.py` logic (seed 42, stratified on language x 3 length buckets, 30% test), and mark every result produced on that provisional split as provisional in the report.

### 1.3 Laptop first, server second
Phase 1 (T01 to T05) is designed to run on a laptop: Jev is an HTTP call, the summariser and Tesseract run on CPU, and the pilot touches 300 sites. The repository must therefore be portable by construction:
- all paths come from `config.yaml` (`data_root`, `results_root`), never hard-coded; a second profile `config.server.yaml` points at the server's data root;
- `make fixture` builds `data/fixture/` from any Putra copy (20 fixed ids listed in `data/fixture/ids.txt`) so the laptop can run every script end to end on 20 sites;
- every script accepts `--limit N` and `--config <file>`;
- the cache under `data/raw_responses/` is content-addressed, so responses collected on the laptop remain valid on the server (copy the folder, nothing is re-queried);
- environment pinned in `pyproject.toml` plus a `uv.lock` or `requirements.txt`; `make env` recreates it on the server.

Handover to the server: `git clone`, `make env`, copy `data/raw_responses/` and `data/derived/` from the laptop, set `--config config.server.yaml`, then run Phase 2 onwards. T08d and T09 require the server's GPU; everything else runs on CPU.

### 1.4 Decisions that remain with Andreas
PhiShark2026 application, fresh crawl (Phase 5), any frontier-model spend, spend above `MAX_USD`.

Implement everything that does not depend on a missing input and list blockers at the top of the report.

---

## 2. The efficiency framing (what the paper is about)

The paper measures how much detection a phishing pipeline buys per unit of cost, latency and evidence, and where each unit of spend stops paying. Five efficiency axes, each with one headline plot:

| Axis | Lever | Headline figure |
|---|---|---|
| Model | decision model vs SVLM vs frontier LLM vs classical classifier | detection (F1, AUROC, PPV@1%) vs USD per 1,000 sites and vs latency per site |
| Evidence | summary token budget 100 to 2,000; URL only; raw vs rendered HTML; with/without OCR | detection vs tokens per site (knee of the curve) |
| Questions | 1 direct question vs bank of k indicator questions, k = 1..17 | detection vs k (marginal cost of a question is near zero in one request) |
| Pipeline | rule gate before any model call; Jev to SVLM escalation band | detection vs fraction of sites that reach each stage; cost-detection frontier |
| Operating point | prevalence 0.1 / 1 / 5%, threshold on the noul probability | false positives per 1,000 benign sites and PPV at fixed recall |

Soundness checks (calibration, cross-dataset transfer, adversarial steering) remain, in reduced scope, because the efficiency claims are worthless if the cheap stage is miscalibrated or trivially steerable.

Research questions in reader order:
- RQ1: What detection performance does a fast, non-generative decision model achieve from URL and HTML evidence, and at what latency and cost, compared with generative SVLMs, frontier LLMs and classical classifiers on the same websites?
- RQ2: How does detection performance depend on the amount and kind of evidence supplied to the decision model (token budget, rendered versus raw HTML, OCR text), and where does additional evidence stop improving detection?
- RQ3: How does decomposing the phishing judgment into indicator-level questions change detection performance, how many questions are needed, and which indicators carry the signal?
- RQ4: To what extent do staged pipelines (deterministic gate, decision model, selective escalation to an SVLM) shift the cost-detection frontier, and under which phishing prevalence do they pay off?
- RQ5: How well calibrated and how transferable are the decision model's probabilities across independently collected datasets, and how much do adversarial text and non-English content degrade them?

---

## 3. Repository layout

```
jev-phishing/
  config.yaml              # model id, MAX_USD, seeds, paths, bank version, budgets
  .env.example
  src/
    data/                  # loaders: putra, phreshphish, phishark, fresh; split handling
    summarise/             # ported html_truncation_github.reduce_html, OCR, state builders
    jev/                   # client: cache, retries, cost, version check, latency
    questions/             # bank_v1.yaml ...
    combine/               # rule vote, logistic regression, CatBoost; Platt, isotonic
    baselines/             # url-feature GBM, tfidf+LR, rule set, encoder, open-weight boolean
    gate/                  # deterministic pre-filter rules
    cascade/               # escalation band logic over stored SVLM outputs
    adversarial/           # injection generators, mitigation variants
    eval/                  # metrics, prevalence simulation, reliability, cost/latency/frontier plots
  scripts/                 # one CLI per task T01..Tnn
  data/ datasets/ paper2/ derived/ raw_responses/ fixture/
  results/ results/timing/ figures/ reports/ tests/
```

Python 3.11, `pyproject.toml`. Dependencies: httpx, pandas, scikit-learn, catboost, beautifulsoup4, lxml, tiktoken, pytesseract (PaddleOCR only if Tesseract fails on the fixture), matplotlib, pyyaml, pytest, playwright (Phase 5 only).

---

## 4. State schema (exactly what the SVLM summariser produces)

`reduce_html(html, page_url, max_tokens)` from `html_truncation_github.py` returns, in priority order `page_url, signals, forms, hosts, meta, scripts, visible_text`:

```json
{
  "page_url": "https://...",
  "signals": {"num_forms": 1, "num_password_fields": 1, "num_hidden_fields": 2,
              "num_sensitive_fields": 3, "form_posts_offsite": true, "num_iframes": 0},
  "forms": [{"action": "...", "action_host": "evil.example", "method": "post", "has_password": true,
             "inputs": [{"name": "email", "type": "text", "sensitive": true, "hidden": false}]}],
  "hosts": {"page_host": "login-paypa1.example", "external_hosts": {"cdn.example": 3},
            "num_external_hosts": 4, "num_phishy_keyword_links": 2},
  "meta": {"title": "...", "meta": {"description": "...", "og:site_name": "..."}, "favicon": "https://..."},
  "scripts": [{"external": true, "src": "..."}, {"external": false, "length": 4100, "entropy": 5.4, "tokens": ["atob", "eval"]}],
  "visible_text": "...",
  "_truncated": false
}
```

Do not rename fields. Add two derived, deterministic fields in the state builder (not in the summariser) because Jev handles them badly on its own: `registrable_domain` (public-suffix aware, from `page_url`) and `favicon_host`. Add `screenshot_text` (OCR) only in the OCR variant. Strip `_truncated` before sending.

State variants:
- `S0` `page_url` + `registrable_domain` only
- `S1` full summary at 1,000 tokens (comparable to the SVLM paper's text mode)
- `S1-b` budget sweep: 100, 250, 500, 1,000, 2,000 tokens (RQ2)
- `S2` S1 + `screenshot_text` (Putra only)
- `S3` `page_url` + raw rendered HTML truncated to 30k tokens with the JCP repo's tag-based truncation (context-rot probe)
- `S5` (fresh crawl only) summary built from the plain HTTP GET body without JavaScript execution, versus the same page rendered (RQ2, render-or-not)

`visible_text` and `screenshot_text` are untrusted fields; question instructions reference fields with backticks, as the Jev documentation recommends.

---

## 5. Question bank v1 (`src/questions/bank_v1.yaml`)

All `noul`. Freeze before any test-set run; later changes are `bank_v2`. Field references match Section 4.

- `q_direct`: "Judging only from the fields in the state, does this web page attempt to obtain login credentials, payment data or other personal data while presenting itself as an organisation other than the one that controls `registrable_domain`?"
- `q_collects_credentials`: "Does any entry in `forms` collect a password or other sensitive data (`has_password` true, or an input with `sensitive` true)?"
- `q_form_offsite`: "Does any form in `forms` have an `action_host` that is not `registrable_domain` or one of its subdomains?"
- `q_brand_domain_mismatch`: "Does `meta.title`, `meta.meta` or `visible_text` name a well-known brand or organisation whose official domain is not `registrable_domain`?"
- `q_brand_in_url_not_domain`: "Does `page_url` contain a well-known brand name in its subdomain or path while `registrable_domain` is not that brand's domain?"
- `q_ip_or_freehost`: "Is `registrable_domain` an IP address, a free hosting or site-builder domain, a dynamic-DNS domain, or a URL shortener?"
- `q_urgency`: "Does `visible_text` pressure the reader with urgency, threats, account suspension or a time limit?"
- `q_reward_bait`: "Does `visible_text` promise a prize, refund, gift or exclusive offer in exchange for entering data?"
- `q_obfuscated_scripts`: "Does any entry in `scripts` indicate obfuscation (`entropy` above 5, or `tokens` containing eval, atob, unescape or fromcharcode)?"
- `q_external_favicon`: "Is `favicon_host` a different registrable domain than `registrable_domain`?"
- `q_shell_page`: "Is the page essentially a single login or data-entry form with very little other content in `visible_text`?"
- `q_missing_legitimacy`: "Does `visible_text` lack the elements a legitimate organisation's page normally has, such as contact information, a privacy policy, terms, or a copyright notice?"
- `q_language_brand_mismatch`: "Is the language of `visible_text` implausible for the brand or organisation named on the page?"
- `q_suspicious_url_shape`: "Does `page_url` have an unusual shape for a legitimate site: many hyphens, many subdomains, a very long path, encoded characters, or a misspelled brand?"
- `q_cloned_assets`: "Are most hosts in `hosts.external_hosts` unrelated to `registrable_domain`, suggesting a cloned page loading its assets from the original site or a CDN?"
- `q_ocr_brand_mismatch` (S2 only): "Does `screenshot_text` show a brand or organisation name whose official domain is not `registrable_domain`?"
- `q_ocr_login_prompt` (S2 only): "Does `screenshot_text` show a sign-in or payment prompt?"

Structural-invariance check (Design C): `q_direct_score` (type `score`, levels none / weak / moderate / strong / certain) and `q_direct_choice` (type `choice`, options phishing, benign).

Note for question wording: the summariser already computes `signals.form_posts_offsite`, `num_password_fields`, `num_sensitive_fields`. Keep both the deterministic signal and the Jev question so T07 can report whether Jev adds anything over the deterministic field for those indicators.

---

## 6. Tasks

### Phase 1: foundations (fixture, then training split only)

**T01 Environment and data.** Layout, `config.yaml`, loaders (Putra CSVs + folders; PhreshPhish), fixture, unit tests. Verify counts against `data/paper2/` when present; otherwise recreate the split (Section 1) and record it as provisional.

**T02 Summariser port, OCR, state builders, stage timing.** Port `reduce_html` unchanged (keep `make_token_counter("gpt-4")`). Verify byte-identity with stored summaries on 50 sites when available. Build S0, S1, S1-b (all five budgets), S2, S3. Tesseract on Putra screenshots; report OCR coverage and seconds per screenshot. Record summarise and OCR wall time per site into `results/timing/`.

**T03 Jev client.** `JevClient`: pinned model, persistence, SHA-256 cache over (model, state, bank version), retries, latency p50/p95, `usage.cost` accumulation, `MAX_USD` stop, concurrency parameter. Smoke test on the fixture with `q_direct`. Confirm every response carries `model == jev-1.13`.

**T04 Go/no-go pilot (training split, 300 balanced sites, stratified as in paper 2).** bank_v1 on S1. Per question: AUROC vs label, noul distribution, inter-question correlation matrix. Cost and latency. Andreas decides continuation on `reports/T04.md`.

**T05 Determinism and structure (training split, 200 sites).** Three repeated queries: share of identical probabilities, mean absolute difference. `q_direct` vs `q_direct_choice` vs `q_direct_score`: agreement and rank correlation.

### Phase 2: model and evidence efficiency on Putra (test split, cached)

**T06 RQ1 direct question.** `q_direct` on S0, S1, S2, S3 over the 2,631 test sites. F1/P/R/Acc at 0.5, AUROC, AUPRC; latency p50/p95; USD per 1,000 sites; HTTP failures. Table places paper-2 text-mode rows beside Jev (muse-glimmer 0.962, gemma4:31b tuned 0.977, claude-opus-4.8 0.967 / 0.975, gpt-5.6-sol 0.967 / 0.968), sourced from `data/paper2/`.

**T06b RQ2 evidence budget.** `q_direct` and bank_v1 on S1-b for all five budgets. Plot AUROC and F1 against mean tokens per site and against USD per 1,000 sites; report the smallest budget within 1 F1 point of the 2,000-token result. Budget selection for later tasks uses the training split curve, not this one.

**T07 RQ3 indicator bank.** Full bank_v1 on S1 and S2 (one request per site). Combiners fitted on the training split with 5-fold CV, applied once to the test split: fixed weighted vote, logistic regression (L2, standardised), CatBoost (defaults, as in paper 2). Report per combiner and variant; LR odds ratios; CatBoost importances; leave-one-question-out ablation (CV on train); greedy forward selection giving the detection-vs-k curve for k = 1..17 (CV on train, final curve evaluated once on test); learning curve of the LR combiner at 50, 100, 200, 500, 1,000, 6,160 training sites. Separately: LR on the deterministic `signals` fields alone, and on `signals` plus Jev nouls, to show what the decision model adds over what the summariser already computes.

**T08 Classical and encoder baselines (train 6,160, test 2,631).** (a) URL-feature gradient boosting; (b) TF-IDF over the serialised S1 + logistic regression; (c) hand-written rule set over summary fields mirroring the bank; (d) ModernBERT-base or DeBERTa-v3-base fine-tuned on serialised S1, 3 seeds. Record training time, inference time per site on CPU and GPU, and USD per 1,000 sites at the paper-2 L40S rate ($0.99/h). GPU needed for (d); otherwise implement and leave a blocker.

**T09 Open-weight boolean baseline (GPU; Ollama or vLLM).** bank_v1 answered by gemma4:12b and qwen3.5:9b with constrained decoding to {yes, no}, P(yes) from logprobs. Same combiners as T07. Runtime per site and USD per 1,000 sites. This separates paradigm-level findings from Jev-specific ones.

### Phase 3: pipeline efficiency

**T10 RQ4 rule gate.** Deterministic rules over `signals` and `hosts` that declare a site benign without any model call (candidate: `num_forms == 0 and num_password_fields == 0 and num_sensitive_fields == 0 and num_phishy_keyword_links == 0`). On the training split: fraction of sites gated, phishing missed by the gate. On the test split, once: the gate followed by Jev; report detection, fraction of model calls saved, and the shift of the class ratio seen by Jev (paper 2 argued a pre-filter moves the detector toward a more favourable prevalence; measure it).

**T11 RQ4 escalation cascade.** Jev combined probability; sites inside an uncertainty band [t_low, t_high] are escalated to the stored SVLM decision (gemma4:31b tuned and muse-glimmer original from `data/paper2/`; no new SVLM runs) and, as an upper bound, to the stored gpt-5.6-sol decision. Sweep the band on the training split; evaluate the chosen bands once on test. Plot F1 and PPV@1% against escalation rate, against USD per 1,000 sites, and against mean latency per site (Jev latency plus escalation fraction times paper-2 per-site runtime). Produce the cost-detection frontier with every configuration from T06 to T11 as a point.

**T12 RQ1/RQ4 cost, latency, throughput, and stage breakdown.** Per configuration: tokens per site, USD per 1,000 sites from `usage.cost`, latency p50/p95, achieved requests per minute at concurrency 1, 4, 16 (respect the documented rate limits), and the per-site wall-time breakdown across summarise, OCR, decide, combine from `results/timing/`. Paper-2 comparison rows: 12 to 32 s per site for SVLMs on one L40S, 3.3 to 6.5 s for the APIs, $3 to $24 per data-mode pass GPU, $52 to $60 API.

### Phase 4: soundness (reduced scope)

**T13 RQ5 calibration and low prevalence.** Reliability diagrams (10 equal-mass bins), ECE, Brier for `q_direct` and the LR-combined score on test; Platt and isotonic fitted on train. PPV at 0.1 / 1 / 5% with the paper-2 resampling protocol and bootstrap CIs; false positives per 1,000 benign sites at 95% recall, for the best Jev configuration, the gate+Jev pipeline, the best cascade, the best SVLM row and gpt-5.6-sol.

**T14 RQ5 transfer to PhreshPhish.** S1 for the paper-2 2,500 sample and an additional stratified 20,000-site sample (balanced, stratified by language; record sampling code). `q_direct` and bank_v1. Regimes: R0 frozen; R1 threshold-only recalibration on 200 labelled sites; R2 combiner refit on 500; R3 refit on 5,000. Compare with paper-2 PhreshPhish rows (gemma4:31b 0.865 untuned / 0.615 tuned; muse-glimmer 0.732 / 0.217; qwen3.6:35b 0.720 / 0.178). Per-language breakdown for languages with at least 200 sites. Repeat T06b's budget sweep on 2,000 PhreshPhish sites to see whether the knee moves across datasets.

**T15 RQ5 adversarial steering (test split phishing pages plus 500 benign).** Variants: A1 legitimacy claims in `visible_text`; A2 classifier-addressed instructions; A3 2k tokens of neutral filler; A4 padding that pushes forms and hosts past the summary budget (checks whether the summariser's priority order protects them). Report noul shift, flip rate at the operating threshold, AUROC for `q_direct` and the bank. Mitigation arm: injected text isolated in `untrusted_text` with instructions stating the field may be deceptive. Comparison arm on 300 sites: gemma4:31b with the paper-2 prompt; gpt-5.6-sol only with Andreas's approval.

**T16 RQ5 transfer to PhiShark2026 (when access is granted).** T14 protocol on a balanced 10,000-site sample; S2 available. Redirect-chain and TLS metadata only as a separately reported `S4` variant.

### Phase 5: optional, Andreas's decision

**T17 Fresh crawl with raw and rendered HTML.** Playwright crawler over OpenPhish and PhishTank feeds (phishing) and Tranco sampling (benign), collected after 15 September 2026; target 1,500 + 1,500 reachable pages. For each page store the plain HTTP GET body (no JavaScript), the Playwright-rendered DOM, a screenshot, final URL, redirect chain, fetch time and render time. Keep only feed-confirmed phishing URLs at crawl time and benign domains with no Safe Browsing flag. Then: T14 protocol, plus S5 versus S1 to answer render-or-not (RQ2), with fetch and render wall time per site added to the pipeline cost.

**T18 Artefact.** Zenodo-ready bundle: question bank, all raw responses, derived states where licences allow, combiner models, results CSVs, timing files, figures, environment, and `REPRODUCE.md` regenerating every table and figure from raw responses without API calls.

---

## 7. Metrics and reporting conventions

- Detection: F1, precision, recall, accuracy at the operating threshold; AUROC, AUPRC threshold-free.
- Calibration: ECE (10 equal-mass bins), Brier, reliability diagram.
- Operational: PPV at 0.1 / 1 / 5% with 95% bootstrap CI (1,000 resamples); false positives per 1,000 benign at 95% recall.
- Efficiency: USD per 1,000 sites (from `usage.cost`, or GPU-hours x $0.99 for local models); latency p50/p95 per site; requests per minute; tokens per site; per-stage wall time. Every frontier plot has cost on the x-axis in USD per 1,000 sites with a secondary latency axis, detection on the y-axis, one point per configuration, with the paper-2 configurations as reference points.
- Tables as CSV in `results/` and markdown in the report; figures as PDF and PNG in `figures/`, matplotlib defaults.
- Every comparison with paper 2 names the paper-2 file the numbers come from.

## 8. Stop points that require Andreas

- Spend above `MAX_USD` (all Jev runs together, roughly 60,000 sites at 1,000 to 2,000 tokens, should stay under 10 USD; the stop exists for safety).
- Any frontier-model calls (T15 comparison arm).
- PhiShark2026 application and fresh crawl (T16, T17).
- Changes to the question bank after the freeze (bank_v2).
- Any read of the Putra test split before Phase 2.

## 9. Order of work

T01 to T05 first week (go/no-go after T04). T06, T06b, T07 week 2. T08, T09 week 3 (GPU session). T10 to T12 week 4. T13 to T15 weeks 5 to 6. T16, T17 when data exists. T18 last.
