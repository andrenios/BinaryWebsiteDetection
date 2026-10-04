# Runbook: from "data arrived" to all Phase 2–4 tables (work order v2.3)

Every command below has been exercised end to end on the bootstrap samples
(pseudo split; the v2.3 scripts with cached responses only, T09 with the fake
backend). On the real data only the dataset name and split source change. Run
from the repo root with the venv active (`make env`), and pass
`--config config.server.yaml` on the server.

**Order of work (WORKORDER v2.3, Section 9):** T01 checks, bank freeze, T02,
T04 (go/no-go), T05, T06, T07 (with the new outputs), **T19**, T08, T09, T10,
T11, T11b, T12, T13, T14, **T20**, **T21**, T15 (with A5), T_STATS, parity
table (T09 `--parity`), reports. T19, T20 and T21 run twice: once on the
decision model's tables and once with `--instantiation twin-<model>` on the
T09 tables. The sections below follow that order.

## 0. Prerequisites and checks

```bash
make test                                              # 28 tests, offline
curl https://api.typesafe.ai/v1/models -H "Authorization: Bearer $TYPESAFE_API_KEY"   # key works, no spend
```

Place the data as described in `README.md` ("Data to load"). Then:

```bash
python scripts/t01_env_and_data.py        # inventory; results/t01_split_status.csv must show 6,160 / 2,631 match=True
```

If the Putra copy's layout differs from `<id>/index.html`, `<id>/screenshot.jpg`, `meta.csv`,
write `scripts/import_putra.py` to produce that layout (symlinks are fine) before continuing.
Check `data/derived/putra_dev_ids.txt` and `data/fixture/ids.txt` against `data/paper2/test.csv`;
if any id is in the test split, rebuild the fixture from training sites (`make fixture --source ...`
after deleting `data/fixture/ids.txt`) and record it in `reports/decisions.md`.

Freeze the bank before the first test-split run: set `frozen: true` in `src/questions/bank_v1.yaml`
and commit. Any later change is `bank_v2.yaml` (Andreas's decision).

## 1. States (CPU, ~1 h for 8,791 sites; S3 is the slow stage)

```bash
python scripts/t02_states.py --dataset putra --variants S0,S1,S1b,S2,S3
```

Byte-identity check against the co-author's `data/paper2/test_json/<id>.json` on 50 sites:
compare `data/derived/states/putra/S1/<id>.json` minus the two derived fields (`registrable_domain`,
`favicon_host`) with the stored summaries (`_truncated` is stripped on our side). A helper for this
is not written yet; a 10-line script with `json.load` and `==` suffices. Record the result in T02.

## 2. Phase 1 on the training split

```bash
python scripts/t04_pilot.py --dataset putra --split train --n 300        # go/no-go -> reports/T04.md, Andreas decides
python scripts/t05_determinism.py --dataset putra --variant S1 --n 200   # three repeats + structure (+ option order, see note)
```

Note: the option-order check (`q_direct_choice_rev`) is implemented in T06 (`--variants S1`);
T05's script does the repeat and score/choice parts. Spend so far should be < 0.5 USD.

## 3. Phase 2: test split (first reads of test.csv are logged to results/test_set_access.log)

```bash
python scripts/t06_direct.py --dataset putra --split test --variants S0,S1,S2,S3      # + option order on S1; paper-2 rows from data/paper2/reference_rows.csv
python scripts/t06b_budget.py --dataset putra --split train                          # budget selection curve
python scripts/t06b_budget.py --dataset putra --split test                           # reported curve
python scripts/t07_bank.py --dataset putra --variant S1                              # combiners, odds ratios, LOQO, forward selection (+ semantic-only), learning curve,
                                                                                     # signals, train out-of-fold scores, results/t19_semantic_set.csv
python scripts/t07_bank.py --dataset putra --variant S2
```

### T19 (run as early as the data allows; decides the framing of contribution (i))

```bash
# PhreshPhish samples (train split minus dev ids; disjoint from each other), then states; ~0.5 USD for the bank on the stability sample
python scripts/t14_sample.py --role stability --n 5000 --out data/derived/phreshphish/stability5k
python scripts/t14_sample.py --role adapt     --n 5700 --out data/derived/phreshphish/adapt5700 --exclude data/derived/phreshphish/stability5k/meta.csv
python scripts/t02_states.py --dataset phreshphish-stab  --variants S1
python scripts/t02_states.py --dataset phreshphish-adapt --variants S1
python scripts/t19_stability.py --dataset putra --variant S1 --stability-dataset phreshphish-stab \
    --transfer-dataset phreshphish-20k            # transfer column fills once T14's 20k sample exists; freezes src/questions/stable_set_v1.yaml -> commit it
```

### T08, T09

```bash
python scripts/t08_baselines.py --dataset putra                                      # (a)(b)(c), CPU
python scripts/t08d_encoder.py --dataset putra                                       # (d) GPU; pip install torch transformers accelerate first
# T09 open-weight twin (GPU; Ollama >= 0.12 with logprobs, or vLLM: --backend vllm --base-url http://localhost:8000)
python scripts/t09_open_weight_twin.py --dataset putra --variant S1 --models gemma4:12b,qwen3.5:9b --backend ollama
python scripts/t09_open_weight_twin.py --dataset putra --variant S0 --models gemma4:12b,qwen3.5:9b --backend ollama   # q_direct on S0 for the twin's T20 arm
python scripts/t09_open_weight_twin.py --dataset putra --variant S2 --models gemma4:12b,qwen3.5:9b --backend ollama
```

`data/paper2/reference_rows.csv` must be created from the co-author's `res_<model>.json` files
(columns `model,mode,dataset,f1,auroc,seconds_per_site,usd_per_pass,source_file`); the numbers in
WORKORDER.md T06/T12 are the targets to reproduce from those files.

## 4. Phase 3: pipeline efficiency

```bash
python scripts/t10_gate.py --dataset putra --variant S1
python scripts/t11_cascade.py --dataset putra --name lr_to_gemma --first results/t07_putra_S1_test_scores.csv --first-col p_lr \
    --first-train results/t07_putra_S1_train_scores.csv \
    --second data/paper2/res_gemma4_31b_tuned_decisions.csv --second-col decision --second-is-decision \
    --second-seconds 18 --second-usd 0.005 --second-train data/paper2/res_gemma4_31b_tuned_decisions_train.csv
python scripts/t11b_evidence_cascade.py --dataset putra --stages S0,S1,S2            # q_direct stages (+ unit-cost sensitivity, per-site scores)
python scripts/t11b_evidence_cascade.py --dataset putra --stages S0,S1,S2 --score lr # LR-combined stages (same sensitivity file, rows tagged lr)
python scripts/t12_efficiency.py --dataset putra --throughput --sustained            # configs, frontier, requests/min at 1/4/16, sustained row from the T06 S1 run,
                                                                                     # results/t12_putra_cost_sensitivity.csv
```

For T11 the stored SVLM/frontier decisions need converting to `site_id,decision` CSVs (train and
test) from `res_<model>.json`; per-site seconds and USD come from paper 2 (12–32 s, $3–24 per pass).
T07 writes `results/t07_putra_S1_train_scores.csv` (out-of-fold combiner scores) for the T11 train sweep and the T20 transition model.

## 5. Phase 4: soundness

```bash
python scripts/t13_calibration.py --dataset putra --variant S1 \
    --extra gate_jev=results/t10_putra_test.csv:p  cascade=results/t11_putra_lr_to_gemma_test.csv:p
python scripts/t14_sample.py --ids data/paper2/phreshphish_2500_ids.txt --out data/derived/phreshphish/paper2_2500
python scripts/t14_sample.py --role eval   --n 20000 --out data/derived/phreshphish/sample20k    # benchmark TEST split, balanced; ~6 GB of HTML; server
python scripts/t14_sample.py --role native --n 20000 --out data/derived/phreshphish/native_test  # benchmark TEST split at its native base rate
python scripts/t02_states.py --dataset phreshphish-20k    --variants S0,S1,S1b
python scripts/t02_states.py --dataset phreshphish-native --variants S0,S1
python scripts/t14_transfer.py --dataset phreshphish-20k    --putra-train results/t07_putra_S1_train_table.csv --adapt-dataset phreshphish-adapt
python scripts/t14_transfer.py --dataset phreshphish-native --putra-train results/t07_putra_S1_train_table.csv --adapt-dataset phreshphish-adapt
python scripts/t14_transfer.py --dataset phreshphish-2500   --putra-train results/t07_putra_S1_train_table.csv --adapt-dataset phreshphish-adapt
python scripts/t06b_budget.py --dataset phreshphish-20k --split all --limit 2000          # does the knee move?
python scripts/t10_gate.py --dataset phreshphish-20k --variant S1                          # gate candidates on PhreshPhish (Table 5a)
```

### T20, T21 (no new API calls)

```bash
python scripts/t20_acquisition.py --dataset putra --stages S0,S1,S2 --score lr     --transfer-dataset phreshphish-20k --transfer-stages S0,S1
python scripts/t20_acquisition.py --dataset putra --stages S0,S1,S2 --score direct --transfer-dataset phreshphish-20k --transfer-stages S0,S1
python scripts/t21_prior_shift.py --dataset putra --variant S1 --targets phreshphish-native,phreshphish-20k --adapt-dataset phreshphish-adapt
```

### T15 with A5, T_STATS, twin instantiation, parity

```bash
python scripts/t15_adversarial.py --dataset putra --split test --benign 500 --questions direct
python scripts/t15_adversarial.py --dataset putra --split test --benign 500 --questions bank
python scripts/t15_adversarial.py --dataset putra --split test --arms A5 --a5-targets direct,lr,twin:gemma4:12b   # per-target cap a5_max_usd_per_target (1 USD)
python scripts/t_stats.py --dataset putra --paper2-best data/paper2/best_svlm_decisions.csv:decision@0.5          # results/t_stats_putra_pairwise.csv
# second instantiation (reads the T09 tables; outputs get the suffix _twin-<model>)
for M in gemma4:12b qwen3.5:9b; do
  python scripts/t19_stability.py   --dataset putra --variant S1 --stability-dataset phreshphish-stab --transfer-dataset phreshphish-20k --instantiation twin-$M
  python scripts/t20_acquisition.py --dataset putra --stages S0,S1,S2 --score lr --transfer-dataset phreshphish-20k --transfer-stages S0,S1 --instantiation twin-$M
  python scripts/t21_prior_shift.py --dataset putra --variant S1 --targets phreshphish-native,phreshphish-20k --adapt-dataset phreshphish-adapt --instantiation twin-$M
done
python scripts/t09_open_weight_twin.py --dataset putra --parity                                                   # results/t09_putra_parity.csv (Table 18)
```

The twin runs of T19 to T21 need the twin's bank tables on the PhreshPhish samples too
(`t09_open_weight_twin.py --dataset phreshphish-stab|phreshphish-20k|phreshphish-native|phreshphish-adapt --variant S1`).
T15's gemma4:31b comparison arm (300 sites, paper-2 prompt) and any gpt-5.6-sol calls need a GPU
session and Andreas's approval respectively; neither is scripted.

## 6. Reports

One `reports/T<nn>.md` per task: what ran, sites, cost, wall time, tables (from `results/*.csv`),
figure paths, open issues, one paragraph of interpretation. Numbers only from `results/`.
Spend to date is in `data/raw_responses/spend_ledger.jsonl`. After T19, T20 and T21:
`reports/STATUS_for_paper_v0.3.md` (per hypothesis H1 to H9: deciding file, held or not, sentence for `paper/draft.md`).

## Expected spend (at 0.042 USD per Mtok, measured token sizes)

| run | requests | approx. USD |
|---|---|---|
| T04 pilot (300 sites, bank) | 300 | 0.02 |
| T06 S0/S1/S2/S3 + option order on 2,631 | ~13,000 | 0.6 (S3 dominates: ~9k tokens/site) |
| T06b 5 budgets x direct+bank on 2,631 (+ train 6,160) | ~90,000 | 1.5 |
| T07 bank on S1 and S2, train + test (8,791 x 2) | ~17,600 | 1.4 |
| T10/T11/T11b/T12/T13 | cache hits mostly | < 0.1 |
| T14 20,000 sites bank + budget sweep on 2,000 | ~40,000 | 2.0 |
| T15 ~1,300 sites x 7 arms x 2 question sets | ~18,000 | 1.0 |
| T19 bank on the 5,000-site stability sample (v2.3) | 5,000 | 0.5 |
| T15 A5 (v2.3), capped per target by `a5_max_usd_per_target` | up to 200,000 per target | 1.0 per target (cap); worst case without the cap 11 to 16 |
| T14 native-rate sample and adaptation sets (v2.3) | ~26,000 | 1.3 |
| T20, T21, T_STATS, parity | cache hits | 0 |

Total ≈ 10 USD with the caps. `MAX_USD` in `config.server.yaml` is 10.0 (the addendum asked for
9); any change is Andreas's call (WORKORDER Section 8). The laptop config is capped at 1.0.
