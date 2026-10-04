# Status summary for drafting the paper

Written 2026-10-04 at the end of session 1, for a Claude session that will
prepare the paper draft. Read `WORKORDER.md` (v2.2) for the research design
and `reports/decisions.md` for every deviation. **Rule for the draft: every
number must come from a file under `results/` listed below. Numbers from the
bootstrap samples are development-only and may appear in the draft only as
"pipeline characteristics" (token sizes, latency, cost, determinism), never as
detection results.**

## 1. The paper in one paragraph

Follow-up to the SLM (JCP 2026) and SVLM papers. A non-generative "decision
model" (TypeSafe Jev, `jev-1.13.0`) answers typed yes/no questions about a
structured summary of a web page and returns calibrated probabilities instead
of text. The paper measures how much phishing detection a pipeline buys per
unit of cost, latency and evidence, along five axes (model, evidence,
questions, pipeline, operating point), on the Putra dataset used by the SVLM
paper (same 2,631-site test split) with transfer to PhreshPhish and, when
granted, PhiShark2026. RQ1–RQ5 are in WORKORDER.md Section 2.

## 2. What is established (can be written now)

### 2.1 Method facts, measured

| fact | value | source |
|---|---|---|
| Model | `jev-1.13.0` (API rejects the alias-less `jev-1.13`); response echoes the versioned id | `data/raw_responses/probe/`, decisions D1 |
| Price | 0.042 USD per million input tokens; output free; no cost field in the response, computed client-side | docs.typesafe.ai/models, D2 |
| Documented limits | 80 requests/s, 100k tokens/s, 64k tokens per request, 32k for state + longest question | docs |
| Question types | noul (P(yes)), choice (distribution + confidence), score (ordered levels) | docs, `src/questions/bank_v1.yaml` |
| State schema | SVLM paper's `reduce_html` output, unchanged, + `registrable_domain`, `favicon_host`; variants S0 (URL), S1 (1,000 tok), S1-b (100–2,000), S2 (+OCR), S3 (raw HTML, 22k tok) | `src/summarise/`, WORKORDER §4, D5, D17 |
| S1 size | ≈ 820–870 tiktoken tokens of state; 1,290–1,410 input tokens as billed with `q_direct`; 60–69 % of pages truncated at 1,000 | `results/t02_*_summary.csv`, `results/t03_*_summary.csv` |
| Bank cost | 15 questions in one request: +≈ 500 tokens (1,830 vs 1,333), no latency change | `results/t03_fixture_S1_{direct,bank}_summary.csv` |
| Latency | p50 ≈ 250–280 ms, p95 ≈ 330–415 ms per request (client-side, Vienna) | `results/t03_*_summary.csv`, `results/t05_*_summary.csv` |
| Throughput | 223 / 815 / 1,780 requests per minute at concurrency 1 / 4 / 16 (24 live requests each), no 429s | `results/t12_bootstrap-phreshphish_throughput.csv` |
| Cost per 1,000 sites | S0 0.015, S1 `q_direct` 0.056–0.059, S1 bank 0.077–0.080, S2 0.061–0.063, S3 0.27–0.38 USD | `results/t06_*_table.csv`, `results/t03_*_summary.csv` |
| Summarise time | S1 ≈ 15–30 ms/site; S3 (JCP tag truncation) 0.56 s mean, 23 s max | `results/t02_*_summary.csv`, `results/timing/t02_*.csv` |
| OCR | Tesseract 0.17–0.25 s per screenshot; 75–82 % of Putra screenshots yield ≥ 20 chars | `results/t02_*_ocr.csv` |
| Determinism (50 sites x 3 repeats) | 38 % bit-identical, mean abs. diff 0.010, max 0.08, 0 % decisions flip at 0.5 | `results/t05_light_bootstrap-phreshphish_S1_summary.csv` |
| Companion-question invariance | `q_direct` alone vs with two companions: mean abs. diff 0.007, Spearman 0.995 | same |
| Structure (direct vs choice vs score) | agreement 96 %, Spearman 0.94–0.98 | same |
| Option order (choice vs reversed, 60 sites) | flip rate 0.00, mean abs. shift of P(phishing) 0.012 | `results/t06_bootstrap-phreshphish_test_option_order.csv` |
| API failure mode | S3 at 30k tiktoken tokens: `max_tokens_exceeded` on 3/60 sites → budget lowered to 22k | D17 |

Paper-2 reference values to compare against (from WORKORDER.md; must be
re-derived from `data/paper2/res_<model>.json` into
`data/paper2/reference_rows.csv` before they go in a table): text-mode F1 on
Putra test — muse-glimmer 0.962, gemma4:31b tuned 0.977, claude-opus-4.8
0.967/0.975, gpt-5.6-sol 0.967/0.968; PhreshPhish — gemma4:31b 0.865 untuned /
0.615 tuned, muse-glimmer 0.732/0.217, qwen3.6:35b 0.720/0.178; runtime 12–32 s
per site (SVLM on one L40S), 3.3–6.5 s (APIs); cost $3–24 per data-mode pass
(GPU), $52–60 (API). Jev: 0.3 s and ≈ 0.15 USD per pass of 2,631 sites.

### 2.2 Methods that are implemented and exercised (describe in the methods section)

- Question bank v1 (15 nouls for S1, +2 for S2, + score/choice/reversed-choice variants), §5.
- Combiners: fixed vote, L2 logistic regression (standardised), CatBoost; Platt and isotonic calibration; leave-one-question-out; greedy forward selection (detection vs k); learning curve; LR on deterministic `signals` alone / with nouls.
- Deterministic benign gate (T10), model cascade with an uncertainty band (T11), **evidence-acquisition cascade** S0→S1→S2 (T11b, new in v2.2: cost axis = seconds to *acquire* evidence, not to judge it).
- Baselines: URL-feature GBM, TF-IDF+LR over the serialised state, 13-rule set, ModernBERT-base fine-tune (script written, GPU pending).
- Soundness: calibration (ECE 10 equal-mass bins, Brier, reliability), PPV at 0.1/1/5 % prevalence with the paper-2 resampling protocol and bootstrap CIs, FP per 1,000 benign at 95 % recall; transfer regimes R0–R3 with per-language breakdown; adversarial variants A1–A4 and an `untrusted_text` mitigation.
- Infrastructure facts worth one sentence: verbatim persistence of every request, content-addressed cache, spend ledger with hard cap, pinned model with abort on mismatch, test-set access log.

## 3. What is NOT established (do not write as results)

Nothing has run on the Putra training or test split. The co-author's
artefacts (`train.csv`/`test.csv`, `test_json/`, `res_<model>.json`, 2,500
PhreshPhish ids) and the prepared Putra copy are not on this machine. All
detection numbers so far are from 20–200 bootstrap sites with a pseudo split;
they are hints for the writing plan, not findings:

- Putra fixture (20 sites): `q_direct` AUROC 0.975 (95 % CI 0.90–1.00), F1 0.84 at 0.5, 0.95 at a fitted threshold.
- PhreshPhish bootstrap (50–60 sites): `q_direct` AUROC 0.86–0.87; LR over the bank 0.99 (pseudo test, 60 sites); CatBoost similar; vote 0.91.
- Probabilities for phishing pages sit low (0.2–0.4) while benign ≈ 0.1: ranking good, 0.5 threshold bad → thresholds must be fitted on train (T13).
- Budget curve hint: AUROC rises 0.79 → 0.88 from 100 to 2,000 tokens for `q_direct`; the bank+LR is already strong at 100 tokens (URL-shaped questions carry it).
- Gate hint (PhreshPhish): the "no forms, no phishy links" gate removes 1/3 of model calls but blocks half of the phishing pages — PhreshPhish phishing often has no form in the summarised DOM. Expect a very different picture on Putra.
- Evidence cascade hint: S0→S1 with the LR combiner reaches S1-alone F1 at 47 % of S1's evidence cost.
- Adversarial hint (60 sites, `q_direct`): A1 legitimacy claims raise probabilities on both classes (+0.05/+0.08); A2 injected instructions barely move it (+0.01); A3 filler and A4 padding lower phishing probabilities by 0.03–0.06; flip rates ≤ 7 %. Mitigation arm did not hurt.
- Transfer hint: threshold-only recalibration on 20 sites moved F1 from 0.49 to 0.75 — most of the transfer loss may be threshold shift, not ranking.

## 4. Planned tables and figures (what each RQ will cite)

| RQ | table / figure | produced by | file pattern |
|---|---|---|---|
| RQ1 | Jev S0/S1/S2/S3 vs paper-2 rows: F1, AUROC, AUPRC, latency, USD/1,000, failures | `t06_direct.py` | `results/t06_putra_test_table.csv` |
| RQ1/RQ4 | cost-detection frontier (USD/1,000 on x, latency secondary, every configuration a point) | `t12_efficiency.py` | `figures/t12_putra_frontier.*`, `results/t12_putra_frontier_points.csv` |
| RQ2 | detection vs tokens and vs USD (budget sweep), knee; raw vs rendered (S3, later S5) | `t06b_budget.py` | `results/t06b_putra_test_curve.csv`, `figures/t06b_*` |
| RQ3 | combiners table, LR odds ratios, CatBoost importances, LOQO, detection vs k, learning curve, signals vs nouls | `t07_bank.py` | `results/t07_putra_S1_*.csv`, `figures/t07_*_forward_selection.*` |
| RQ4 | gate: fraction gated / phishing missed / prevalence shift; cascade and evidence cascade: detection vs escalation rate, USD, seconds | `t10_gate.py`, `t11_cascade.py`, `t11b_evidence_cascade.py` | `results/t10_putra_*.csv`, `results/t11_putra_*_test.csv`, `results/t11b_putra_*_test.csv` |
| RQ5 | reliability diagrams, ECE/Brier, PPV at prevalence with CIs, FP/1,000 at 95 % recall; transfer regimes R0–R3 and per-language table; adversarial shift/flip/AUROC per arm | `t13_calibration.py`, `t14_transfer.py`, `t15_adversarial.py` | `results/t13_putra_S1_*.csv`, `figures/t13_*`, `results/t14_*_regimes.csv`, `results/t15_putra_*_summary.csv` |
| baselines | classical + encoder rows for the RQ1 table and the frontier | `t08_baselines.py`, `t08d_encoder.py` | `results/t08_putra_baselines.csv`, `results/t08d_putra_encoder.csv` |
| soundness | determinism, structure, option order (train 200 + test) | `t05_determinism.py`, `t06_direct.py` | `results/t05_putra_S1_summary.csv`, `results/t06_putra_test_option_order.csv` |

## 5. Narrative hypotheses to test, not assume

1. Jev reaches within a few F1 points of the SVLMs on Putra at ~1/100 of the cost and ~1/50 of the latency (RQ1).
2. Detection saturates around 500–1,000 summary tokens; raw HTML (S3) does not help and costs 5x (RQ2).
3. A handful of indicator questions carry the signal; a learned combiner beats `q_direct`; Jev adds information beyond the summariser's deterministic fields (RQ3).
4. The evidence-acquisition cascade, not the model cascade, is where deployment cost is saved; the gate's value is dataset-dependent (RQ4).
5. Jev transfers better than tuned SVLM prompts because there is nothing tuned to Putra; most transfer loss is threshold shift (RQ5).
6. Repeated sampling buys nothing (near-deterministic); phrasing and companion questions do not move answers; adversarial text moves probabilities modestly and mostly toward "benign" (soundness).

## 6. Blockers and open decisions (also in reports/T05.md)

Co-author artefacts and Putra copy missing; which Zenodo screenshot the paper used (D4); fixture may overlap the test split (D7); `MAX_USD` must rise to ≈ 7 USD for the full plan (RUNBOOK); T09, T16, T17, T18 not written; T08d needs a GPU; evidence-acquisition seconds are placeholders until T17 (D14).

## 7. Files to read first

`WORKORDER.md`, `reports/decisions.md`, `reports/RUNBOOK.md`, `reports/T01.md`–`T05.md`, `README.md`, `results/` (CSV only), `src/questions/bank_v1.yaml`.
