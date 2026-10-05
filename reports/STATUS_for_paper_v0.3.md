# Status for paper draft v0.3: hypotheses H1 to H9 (skeleton, 2026-10-04)

Skeleton written before any run on the real data (WORKORDER v2.3, Section 10). One block per hypothesis, in the order of `paper/draft.md` Section 6: the deciding file(s) and columns, the quantity and the decision rule as the draft states it, the status, and the sentence to paste into the draft once the value is read from the file. Rule for filling: every number comes from the named `results/` file produced on the Putra test split (`split_source = paper2`) or the real PhreshPhish samples; bootstrap and fixture files (`split_source = pseudo`) never fill a line here. Paired intervals come from `results/t_stats_putra_pairwise.csv`. The parity column of each block is read from `results/t09_putra_parity.csv` after T19 to T21 have run for both instantiations.

Legend for **Status**: `not run` (script exists, exercised on bootstrap data only), `blocked: <input>`, `run`, `held`, `failed`, `inconclusive`.

| Hypothesis | Section | Deciding file(s) | Status |
|---|---|---|---|
| H1 | 6.1 | `t06_putra_test_table.csv`, `data/paper2/reference_rows.csv`, `t_stats_putra_pairwise.csv`, `t08_putra_baselines.csv`, `t08d_putra_encoder.csv`, `t09_putra_S1_{combiners,summary,per_question}.csv`, `t12_putra_configs.csv` | not run |
| H3 | 6.2 | `t07_putra_S1_combiners.csv`, `t07_putra_S1_{lr_odds_ratios,catboost_importances,loqo,forward_selection,learning_curve,signals}.csv`, `t19_semantic_set.csv`, `t07_putra_S2_combiners.csv` | not run |
| H7 | 6.3 | `t19_stability_indicators.csv`, `t19_transfer_by_combiner.csv`, `t_stats_putra_pairwise.csv` | blocked: `phreshphish-stab` sample |
| H2 | 6.4 | `t06b_putra_test_curve.csv`, `t06_putra_test_table.csv` (S0, S2, S3 rows), `t12_putra_configs.csv` | not run |
| H4 | 6.5 | `t10_putra_{train_gates,test}.csv`, `t11_putra_lr_to_{gemma,muse,gpt}_test.csv`, `t11b_putra_{direct,lr}_S0-S1-S2_test.csv`, `t11b_putra_sensitivity.csv`, `t12_putra_frontier_points.csv` | not run; T11 blocked: paper-2 per-site decisions |
| H8 | 6.6 | `t20_putra_voi_test.csv`, `t20_putra_conformal_test.csv`, `t20_phreshphish_conformal_test.csv` | not run |
| H5 | 6.7 | `t13_putra_S1_{calibration,ppv,fp95}.csv`, `t14_phreshphish-20k_S1_{regimes,per_language}.csv`, `t06b_phreshphish-20k_all_curve.csv` | blocked: `phreshphish-20k`, `phreshphish-adapt` samples |
| H9 | 6.8 | `t21_prior_shift.csv` | blocked: `phreshphish-native`, `phreshphish-20k`, `phreshphish-adapt` samples |
| H6 | 6.9 | `t05_putra_S1_summary.csv`, `t06_putra_test_option_order.csv`, `t15_putra_{direct,bank}_summary.csv`, `t15_putra_A5_summary.csv`, `t09_putra_parity.csv` | not run; A5 needs the spend decision (D24) |

---

## H1 (Section 6.1): indicator-level detection closes to within three F1 points of the best SVLM; the twin reproduces the ranking (RQ1)

- **Quantity:** F1 difference `q_direct` S1 (and lr bank S1) minus the best paper-2 SVLM text-mode row, with its paired 95 % interval and McNemar p. Files: `t06_putra_test_table.csv` (`f1`, `precision`, `recall`, `auroc`, `auprc`, `http_failures` per variant), `t07_putra_S1_combiners.csv` (`test_f1_at_fit`, lr), `data/paper2/reference_rows.csv` (best SVLM `f1`), `t_stats_putra_pairwise.csv` (pair "decision model (bank lr) vs best SVLM (paper 2)", `estimate`, `ci_low`, `ci_high`, `mcnemar_p`). Context rows: `t08_putra_baselines.csv`, `t08d_putra_encoder.csv`, `t12_putra_configs.csv` (`usd_per_1000`, `latency_p50_ms`).
- **Rule:** holds if the paired interval of the F1 difference excludes a gap larger than three points.
- **H1b (twin):** `t09_putra_S1_combiners.csv` (`test_f1_at_fit`, `test_auroc` per model), `t09_putra_S1_summary.csv` (`seconds_per_site`, `usd_per_1000`, `spearman_per_question_auroc_vs_decision_model`), `t09_putra_S1_per_question.csv`.
- **Status:** not run (needs the Putra data, `reference_rows.csv` and the per-site decisions of the best SVLM for the paired pair; GPU session for H1b).
- **Held:** `[ ]` yes `[ ]` no `[ ]` inconclusive
- **Sentence for the draft:** "On the 2,631-site test split the indicator bank with the logistic combiner reaches an F1 of [t07: test_f1_at_fit, lr] against [paper2: f1] for the best SVLM, a difference of [t_stats: estimate] points (paired 95 % interval [ci_low, ci_high], McNemar p = [mcnemar_p]), which [is / is not] within the three points H1 allows; the open-weight twin reaches [t09: test_f1_at_fit per model] with a per-question AUROC correlation of [t09 summary: spearman] to the decision model."

## H3 (Section 6.2): a handful of questions carry the signal, the combiner beats the direct question, the model adds information beyond the deterministic fields (RQ2)

- **Quantity:** lr `test_f1_at_fit` minus `q_direct alone` `test_f1_at_fit` (`t07_putra_S1_combiners.csv`) with the paired interval (pair "bank lr vs q_direct (S1)"); number of questions at which `cv_auroc` is within 0.005 of the full bank (`t07_putra_S1_forward_selection.csv`, columns `k`, `added`, `cv_auroc`, `test_auroc`); `t07_putra_S1_signals.csv` rows "signals only", "nouls only", "signals + nouls" (`test_auroc`); `t19_semantic_set.csv` rows deterministic / semantic / both (`test_auroc`, `test_f1_at_fit`, `tokens_per_site`; tokens of the semantic row are an estimate, D19); `t07_putra_S1_forward_selection_semantic.csv`.
- **Rule:** holds if the combiner's interval excludes zero, if the forward-selection curve is flat before k = 15, and if the nouls-only or semantic rows exceed the signals-only / deterministic row.
- **Status:** not run.
- **Held:** `[ ]` yes `[ ]` no `[ ]` partly (state which clause)
- **Sentence for the draft:** "The logistic combiner reaches [test_f1_at_fit] against [q_direct alone] for the direct question ([estimate], 95 % interval [ci_low, ci_high]); [k] questions reach the full bank's cross-validated AUROC within 0.005, the first three being [added rows 1-3]; logistic regression on the deterministic fields alone reaches a test AUROC of [signals only], on the semantic indicators alone [semantic], and on both [both]."

## H7 (Section 6.3): indicators that discriminate consistently across corpora transfer, in-distribution favourites do not (RQ2)

- **Quantity:** counts of `class` = stable / artefact / uninformative and the top-three `rank_putra` versus `rank_phresh` (`t19_stability_indicators.csv`); `transfer_loss` per `combiner` (`t19_transfer_by_combiner.csv`: "lr all indicators", "lr stable set", "q_direct"; `putra_test_f1`, `r0_f1_phreshphish`, `ci_low`, `ci_high`); paired interval of the R0 F1 difference stable-set minus all-indicator on the transfer sites (`t_stats_putra_pairwise.csv`, pair "stable-set vs all-indicator lr under transfer").
- **Rule:** holds if the stable-set loss is at most half the all-indicator loss and the in-distribution cost (`putra_test_f1` all minus stable) is under two points; fails if the stable set transfers no better.
- **Status:** blocked: `phreshphish-stab` (5,000 train-split sites, `t14_sample.py --role stability`, about 0.5 USD) and `phreshphish-20k` for the transfer column. The stable set is frozen in `src/questions/stable_set_v1.yaml` on the first real run; record the commit hash here: `[ ]`.
- **Held:** `[ ]` yes `[ ]` no (then contribution (i) is written as an inventory, draft Section 6.3 interpretation)
- **Sentence for the draft:** "Of the 15 indicators, [n stable] are stable, [n artefact] are artefacts and [n uninformative] are uninformative; frozen on PhreshPhish the all-indicator combiner loses [transfer_loss all] F1 points, the stable-set combiner [transfer_loss stable] (paired difference [estimate], 95 % interval [ci_low, ci_high]) and the direct question [transfer_loss q_direct], at an in-distribution cost of [putra_test_f1 all minus stable] points."

## H2 (Section 6.4): detection saturates between 500 and 1,000 summary tokens; raw HTML neither helps nor pays (RQ3)

- **Quantity:** `smallest_budget_within_1pt_f1`, `auroc_direct`, `auroc_bank_lr_cv`, `tokens_mean_direct` per `budget` (`t06b_putra_test_curve.csv`; selection curve `t06b_putra_train_curve.csv`); S0, S2, S3 rows of `t06_putra_test_table.csv` (`f1`, `auroc`, `http_failures`); `t12_putra_configs.csv` (`usd_per_1000` for S1, S2, S3); paired pairs "q_direct S2 vs S1", "q_direct S3 vs S1", "q_direct budget b vs 2000".
- **Rule:** holds if the smallest budget within one F1 point of 2,000 tokens lies in 500 to 1,000 and the S3 minus S1 interval does not exclude zero in S3's favour.
- **Status:** not run.
- **Held:** `[ ]` yes `[ ]` no
- **Sentence for the draft:** "The smallest budget within one F1 point of the 2,000-token result is [smallest_budget_within_1pt_f1] tokens; raw HTML (S3) gives an F1 of [f1 S3] against [f1 S1] for the summary (paired difference [estimate], 95 % interval [ci_low, ci_high]) at [usd_per_1000 S3 / S1] times the cost, with [http_failures S3] requests rejected for size; the URL alone (S0) reaches an AUROC of [auroc S0]."

## H4 (Section 6.5): deployment cost is saved by staging evidence acquisition rather than by escalating to a larger model; the gate's value is dataset-dependent (RQ3)

- **Quantity:** `fraction_gated`, `phishing_missed_rate`, `prevalence_after_gate` per `gate` (`t10_putra_train_gates.csv`, `t10_phreshphish-20k_train_gates.csv`); `f1`, `model_calls_saved`, `phishing_blocked_by_gate` (`t10_putra_test.csv`); cascade `f1`, `1 - stop_jev`, `usd_per_1000`, `mean_seconds_per_site` (`t11_putra_lr_to_{gemma,muse,gpt}_test.csv`); evidence cascade `f1`, `auroc`, `stop_S0`, `stop_S1`, `stop_S2` for the chosen band (`t11b_putra_{direct,lr}_S0-S1-S2_test.csv`); `ordering_changed` count (`t11b_putra_sensitivity.csv`); pair "evidence cascade (lr arm) vs S1 alone (lr)".
- **Rule:** holds if the evidence cascade reaches S1-level detection (paired interval of the F1 difference covers zero) with a materially smaller fetched fraction (`1 - stop_S0`), and if the model cascade's gain over the decision model alone is small for the escalation rate it needs.
- **Status:** not run; T11 additionally blocked by the paper-2 per-site decisions (`site_id,decision` CSVs from `res_<model>.json`). Acquisition seconds stay placeholders until T17 (D14); the primary unit is pages fetched.
- **Held:** `[ ]` yes `[ ]` no `[ ]` partly
- **Sentence for the draft:** "The gate removes [fraction_gated] of training sites while blocking [phishing_missed_rate] of phishing pages on Putra and [fraction_gated / phishing_missed_rate] on PhreshPhish; the evidence cascade with the logistic combiner reaches an F1 of [f1] (difference to S1 alone [estimate], 95 % interval [ci_low, ci_high]) while never fetching [stop_S0] of pages, and the frontier ordering of its configurations changed at [n] of 27 unit-cost grid points; the gemma4:31b cascade gains [f1 cascade minus Jev only] points for [1 - stop_jev] of sites escalated."

## H8 (Section 6.6): the cost-derived stopping rule fetches fewer pages than the best swept band at equal detection; the conformal rule keeps its guarantee on the test corpus (RQ3)

- **Quantity:** `fetched_fraction`, `f1`, `stop_S0` for `cost_ratio` 1:1 / 10:1 / 100:1 at the central grid point (`central = True`) and the range over the grid, against the rows "swept band ... (T11b)" and "always fetch S1" (`t20_putra_voi_test.csv`; exchange rate column `exchange_seconds_per_fp_unit`); `fraction_stopped`, `realised_miss_rate` per `alpha` and `level` (`t20_putra_conformal_test.csv`, `t20_phreshphish_conformal_test.csv`); pairs "VoI 10:1 vs swept band (lr arm)" and "VoI 10:1 vs S1 alone (lr)".
- **Rule:** holds if, at each cost ratio, the rule's fetched fraction is below the swept band's at an F1 within the paired interval of the band's; the conformal clause holds if `realised_miss_rate` at level S0 is at most alpha on Putra test (a higher value on PhreshPhish is reported as the covariate-shift price, not as a failure of H8).
- **Status:** not run (cache hits only once T06, T07 and T11b exist; the transfer file needs `phreshphish-20k`).
- **Held:** `[ ]` yes `[ ]` no `[ ]` VoI clause only `[ ]` conformal clause only
- **Sentence for the draft:** "At c_FN:c_FP = 10:1 and the central acquisition costs (one unit of c_FP = [exchange_seconds_per_fp_unit] s of acquisition) the rule fetches [fetched_fraction] of pages at an F1 of [f1], against [fetched_fraction band] and [f1 band] for the best swept band (paired difference [estimate], 95 % interval [ci_low, ci_high]); at 1:1 and 100:1 it fetches [fetched_fraction 1:1] and [fetched_fraction 100:1]. The conformal rule at alpha = 0.01 stops [fraction_stopped S0] of pages at the URL with a realised miss rate of [realised_miss_rate]; on PhreshPhish the realised rate is [realised_miss_rate transfer]."

## H5 (Section 6.7): the probabilities are usable as probabilities; most of the remaining transfer loss is a threshold shift (RQ4)

- **Quantity:** `ece_10bins`, `brier`, `mean_p_phish`, `mean_p_benign` for `q_direct` and `lr_bank` raw / platt / isotonic (`t13_putra_S1_calibration.csv`); `ppv`, `ppv_ci_low`, `ppv_ci_high` per `prevalence` (`t13_putra_S1_ppv.csv`); `fp_per_1000_benign` (`t13_putra_S1_fp95.csv`); `f1`, `precision`, `recall`, `auroc` per `regime` R0 to R3 and `score` (`t14_phreshphish-20k_S1_regimes.csv`, `t14_phreshphish-2500_S1_regimes.csv`); recovered share (f1_R1 - f1_R0) / (f1_R3 - f1_R0); `auroc_q_direct` per `language` (`t14_phreshphish-20k_S1_per_language.csv`); `smallest_budget_within_1pt_f1` on PhreshPhish (`t06b_phreshphish-20k_all_curve.csv`).
- **Rule:** calibration clause holds if the raw lr score's ECE is below the Platt-calibrated `q_direct`'s by less than the Brier difference suggests (report the numbers; no fixed cut); threshold clause holds if R1 recovers more than half of the R0-to-R3 gap.
- **Status:** T13 not run; T14 blocked: `phreshphish-20k` (benchmark test split) and `phreshphish-adapt` (train split) samples.
- **Held:** `[ ]` yes `[ ]` no `[ ]` calibration clause only `[ ]` threshold clause only
- **Sentence for the draft:** "The raw `q_direct` probability has an ECE of [ece raw] and a Brier score of [brier raw] ([ece platt] and [brier platt] after Platt scaling); the logistic score has [ece lr raw]. Frozen on PhreshPhish the combiner reaches an F1 of [f1 R0] (precision [precision R0], recall [recall R0]); threshold-only recalibration on 200 sites moves it to [f1 R1] and refitting on 5,000 to [f1 R3], so R1 recovers [share] of the gap."

## H9 (Section 6.8): prior correction with an estimated base rate recovers most of the gain of labelled threshold recalibration (RQ4)

- **Quantity:** rows `R0`, `prior_known`, `prior_em`, `R1` for `target = phreshphish-native` and `phreshphish-20k` (`t21_prior_shift.csv`: `prevalence_true`, `prevalence_estimated`, `threshold`, `f1`, `precision`, `recall`, `ece_before`, `ece_after`); recovered share (f1_prior_em - f1_R0) / (f1_R1 - f1_R0); pair "prior_em vs R1 threshold" in `t_stats_putra_pairwise.csv`. The `_bayes` rows (Putra posterior threshold kept after the shift, D22) are context.
- **Rule:** holds if the recovered share on the native-rate target is above two thirds; undefined if R1 does not improve on R0 (report as such).
- **Status:** blocked: `phreshphish-native`, `phreshphish-20k`, `phreshphish-adapt` samples (no API spend beyond the T14 bank requests; T21 itself makes no calls).
- **Held:** `[ ]` yes `[ ]` no `[ ]` undefined
- **Sentence for the draft:** "At the benchmark split's native base rate of [prevalence_true], the frozen threshold gives an F1 of [f1 R0] (precision [precision R0]); prior correction with the known base rate [f1 prior_known]; with the EM-estimated base rate of [prevalence_estimated] it gives [f1 prior_em]; recalibration on 200 labelled sites [f1 R1]. The share of the labelled gain recovered without labels is [share]. Per-indicator isotonic maps fitted on Putra change the combined score's ECE on PhreshPhish from [ece_before] to [ece_after]."

## H6 (Section 6.9): repeated sampling buys nothing, phrasing and companions do not move answers, adversarial text moves probabilities modestly, the twin reproduces the main findings (RQ5)

- **Quantity:** `share_identical`, `mean_abs_diff`, `max_abs_diff`, `share_flip_at_0.5`, `agree_direct_vs_choice`, `spearman_*`, `mean_abs_diff_alone_vs_with_companions` (`t05_putra_S1_summary.csv`); `flip_rate`, `mean_abs_shift_p_phishing`, `auroc_first_order`, `auroc_reversed` (`t06_putra_test_option_order.csv`); `shift_mean_phish`, `shift_mean_benign`, `flip_rate_phish_to_benign`, `auroc_clean`, `auroc_arm` per `arm` (`t15_putra_direct_summary.csv`, `t15_putra_bank_summary.csv`); `flip_rate`, `mean_prob_reduction`, `mean_queries_to_flip`, `usd_per_flipped_page`, `mean_indicators_dropped` per `score` and `budget` (`t15_putra_A5_summary.csv`; note `n_pages_attempted` and `arm_usd` if the per-target cap stopped the arm, D24); `same_direction` per hypothesis (`t09_putra_parity.csv`).
- **Rule:** holds if no decision flips across repeats at the fitted threshold, the structural and companion shifts stay at the second decimal, the fixed arms move the probability by less than the flip rate needs, and the A5 flip rate at 1,000 queries is reported with its cost; the twin clause is read from the parity table (count of `same_direction = yes` over the hypotheses that have a twin value).
- **Status:** T05, T06 and the fixed arms not run; A5 needs the spend decision (D24); the parity table needs T09 plus the twin runs of T19 to T21.
- **Held:** `[ ]` yes `[ ]` no `[ ]` partly (state which clause)
- **Sentence for the draft:** "Three repeats return identical probabilities on [share_identical] of sites with a maximum difference of [max_abs_diff] and no flipped decision; companions move `q_direct` by [mean_abs_diff_alone_vs_with_companions] and option order flips [flip_rate] of choices. Legitimacy claims shift the phishing-page probability by [shift_mean_phish A1] and flip [flip_rate_phish_to_benign A1]; the adaptive attacker flips [flip_rate A5, 1000, q_direct] of 200 pages for `q_direct` and [flip_rate A5, 1000, lr_bank] for the bank at [usd_per_flipped_page] USD per flipped page, moving [mean_indicators_dropped] indicators on average. The twin agrees in direction on [n yes] of [n available] hypotheses."

---

## Prerequisites before any block can be filled (unchanged from `reports/T21.md`)

1. Co-author data and artefacts into `data/paper2/` and `data/datasets/putra/`; T01 split check; fixture overlap check (D7).
2. PhreshPhish samples: `phreshphish-stab`, `phreshphish-adapt` (train split), `phreshphish-20k`, `phreshphish-native` (benchmark test split); `reports/RUNBOOK.md` Sections T19 and 5.
3. `MAX_USD` and the A5 per-target cap (Andreas; D24, D27).
4. GPU session for T08d and T09; the twin runs of T19 to T21 before `t09_open_weight_twin.py --parity`.
5. Paper-2 per-site decisions as `site_id,decision` CSVs (T11, T_STATS pair "decision model vs best SVLM", parity reference).
