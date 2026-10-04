#!/usr/bin/env python3
"""T20: value-of-information and conformal evidence acquisition (RQ3, method M2).

  python scripts/t20_acquisition.py --dataset putra --stages S0,S1,S2 --score lr \
      --transfer-dataset phreshphish-20k --transfer-stages S0,S1 [--instantiation twin-<model>]

No new API calls on Putra: stage probabilities are the cached q_direct (S0) and
bank responses (S1, S2) of T06 / T07; the lr combiner per stage is refitted on
the training split (out-of-fold scores on train, one fit applied to test) and,
by default, Platt-calibrated on train (--calibration; T13 feeds T20).

VoI (src/cascade/voi.py): transition model p_k -> p_{k+1} in 20 equal-mass bins
fitted on train (results/t20_<dataset>_voi_transition.csv); the myopic rule
acquires iff a_{k+1} + E[decide(p_{k+1}) | p_k] < decide(p_k) with
decide(p) = min(c_FN p, c_FP (1-p)), c_FP = 1, c_FN in --cost-ratios, and a_k the
acquisition seconds of the T11b unit-cost grid converted at the stated exchange
rate (config voi_seconds_per_fp_unit seconds per unit of c_FP). F1 is reported
at the train-fitted threshold of the stopping stage's score, as in T11b, so the
rows are comparable with the swept band and with always-fetch (reference rows
in the same table). Conformal (src/cascade/conformal.py): train split 50/50,
thresholds per level on the calibration half for alpha in --alphas and beta,
evaluated on test and frozen on the transfer sample.

Outputs (paper/tables.md Tables 15, 16): results/t20_<dataset>_voi_test.csv,
results/t20_<dataset>_voi_transition.csv, results/t20_<dataset>_conformal_test.csv,
results/t20_<transfer>_conformal_test.csv (t20_phreshphish_conformal_test.csv for
the PhreshPhish benchmark sample), per-site results/t20_<dataset>_voi_test_scores.csv,
figures/t20_fetched_vs_costratio.*, figures/t20_conformal_miss_vs_alpha.*.
"""
from __future__ import annotations

import argparse
import sys
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from _bootstrap import load_config, add_common_args, add_split_args, banner, REPO, decision_table, out_name
sys.path.insert(0, str(REPO / "src"))
from jev.client import client_from_config  # noqa: E402
from jev.run import save_table  # noqa: E402
from questions.bank import load_bank, direct_question, bank_questions  # noqa: E402
from eval.metrics import detection, best_f1_threshold  # noqa: E402
from eval.plots import curve  # noqa: E402
from combine.combiners import make_lr, cv_scores, fit_apply, question_columns, Platt, Isotonic  # noqa: E402
from cascade.voi import fit_transition, run_voi, decide_cost  # noqa: E402
from cascade.conformal import conformal_thresholds, run_conformal, conformal_summary  # noqa: E402

UNIT_COST_GRID = {"get_s": [0.5, 1.5, 3.0], "render_s": [2.0, 5.0, 10.0], "ocr_s": [0.3, 1.0, 2.0]}   # as T11b
CENTRE = (1.5, 5.0, 1.0)


def stage_seconds(get_s, render_s, ocr_s) -> dict[str, float]:
    return {"S0": 0.0, "S5": get_s, "S1": render_s, "S2": render_s + ocr_s, "S3": render_s}


def questions_for(bank, stage: str, score: str) -> dict:
    return direct_question(bank) if (stage == "S0" or score == "direct") else bank_questions(bank, stage)


def stage_tables(cfg, client, dataset, stages, score, inst, limit, workers, bank, splits=("train", "test")):
    """Decision tables per split and stage (cache hits under the T06 / T07 experiment names)."""
    out, src = {}, {}
    for split in splits:
        out[split] = {}
        for v in stages:
            exp = f"t06_{dataset}_{split}_{v}_direct" if (v == "S0" or score == "direct") else f"t07_{dataset}_{v}_{split}"
            try:
                df, src[split] = decision_table(cfg, client, dataset, v, split if split in ("train", "test") else None,
                                                questions_for(bank, v, score), exp, inst, limit, workers, f"t20 {split} {v}")
            except SystemExit as e:
                print(f"[t20] {dataset} {split} {v}: {e}"); continue
            if len(df):
                out[split][v] = df.set_index("site_id")
    return out, src


class StageScorer:
    """Per-stage score: q_direct, or lr over the bank refitted on the training
    split (oof on train), followed by an optional calibration map fitted on
    train. `fit_rows` restricts the fit to a subset of train (conformal half)."""

    def __init__(self, score: str, calibration: str, seed: int):
        self.score, self.calibration, self.seed = score, calibration, seed
        self.models, self.cals, self.cols, self.thresholds = {}, {}, {}, {}

    def fit(self, stage: str, tr: pd.DataFrame, fit_idx=None) -> pd.Series:
        """Returns the (out-of-fold, calibrated) train scores for the rows used to fit."""
        d = tr if fit_idx is None else tr.loc[fit_idx]
        y = d.label.values.astype(int)
        if self.score == "direct" or stage == "S0":
            cols = ["q_direct"]; raw = d.q_direct.values.astype(float); self.models[stage] = None
        else:
            cols = ["q_direct"] + question_columns(d)
            raw = cv_scores(make_lr, d[cols].astype(float), y, seed=self.seed)
            self.models[stage] = make_lr().fit(d[cols].astype(float), y)
        self.cols[stage] = cols
        cal = {"none": None, "platt": Platt, "isotonic": Isotonic}[self.calibration]
        self.cals[stage] = cal().fit(raw, y) if cal else None
        p = self.cals[stage].predict(raw) if self.cals[stage] is not None else raw
        self.thresholds[stage] = best_f1_threshold(y, p)[0]
        return pd.Series(p, index=d.index)

    def apply(self, stage: str, df: pd.DataFrame) -> pd.Series:
        cols = [c for c in self.cols[stage] if c in df.columns]
        if len(cols) < len(self.cols[stage]):
            raise SystemExit(f"[t20] stage {stage}: columns {set(self.cols[stage]) - set(cols)} missing in the target table")
        raw = df.q_direct.values.astype(float) if self.models[stage] is None else self.models[stage].predict_proba(df[cols].astype(float))[:, 1]
        p = self.cals[stage].predict(raw) if self.cals[stage] is not None else raw
        return pd.Series(p, index=df.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.add_argument("--stages", default="S0,S1,S2")
    ap.add_argument("--score", default="lr", choices=["lr", "direct"])
    ap.add_argument("--calibration", default="platt", choices=["none", "platt", "isotonic"])
    ap.add_argument("--cost-ratios", default=None, help="c_FN:c_FP ratios, default config voi_cost_ratios")
    ap.add_argument("--alphas", default=None); ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--exchange-seconds-per-fp", type=float, default=None, help="seconds of acquisition worth one unit of c_FP")
    ap.add_argument("--transfer-dataset", default=None, help="frozen conformal evaluation (phreshphish-20k)")
    ap.add_argument("--transfer-stages", default="S0,S1")
    ap.add_argument("--transfer-split", default="all", choices=["train", "test", "all"])
    ap.add_argument("--instantiation", default="jev")
    args = ap.parse_args()
    cfg = load_config(args.config)
    inst, seed = args.instantiation, cfg["SEED"]
    stages = args.stages.split(",")
    ratios = [float(r) for r in (args.cost_ratios.split(",") if args.cost_ratios else cfg.get("voi_cost_ratios", [1, 10, 100]))]
    alphas = [float(a) for a in (args.alphas.split(",") if args.alphas else cfg.get("conformal_alphas", [0.01, 0.05]))]
    beta = args.beta if args.beta is not None else float(cfg.get("conformal_beta", 0.05))
    xrate = args.exchange_seconds_per_fp or float(cfg.get("voi_seconds_per_fp_unit", 60))
    base = out_name(f"t20_{args.dataset}", inst) if args.score == "lr" else out_name(f"t20_{args.dataset}_direct", inst)
    client = client_from_config(cfg, f"t20_{args.dataset}_{args.score}", use_cache=not args.no_cache) if inst == "jev" else None
    bank = load_bank(cfg.bank_path)
    thr05 = cfg.get("operating_threshold", 0.5)

    tabs, src = stage_tables(cfg, client, args.dataset, stages, args.score, inst, args.limit, args.workers, bank)
    stages = [v for v in stages if v in tabs["train"] and v in tabs["test"]]
    if len(stages) < 2:
        raise SystemExit(f"[t20] fewer than two stages with train and test tables ({stages}); nothing to do")
    args.split = "train/test"; banner("t20", args, list(tabs["train"][stages[0]].index), src.get("test", ""), client)
    y_tr = pd.concat([t.label for t in tabs["train"].values()]).groupby(level=0).first().astype(int)
    y_te = pd.concat([t.label for t in tabs["test"].values()]).groupby(level=0).first().astype(int)

    # ---- stage scores (full train fit for VoI)
    sc = StageScorer(args.score, args.calibration, seed)
    p_tr = {v: sc.fit(v, tabs["train"][v]) for v in stages}
    p_te = {v: sc.apply(v, tabs["test"][v]) for v in stages}
    usd_te = {v: float(tabs["test"][v].cost_usd.mean()) if "cost_usd" in tabs["test"][v] else 0.0 for v in stages}
    print("[t20] train-fitted thresholds per stage: " + ", ".join(f"{v}={sc.thresholds[v]:.2f}" for v in stages))

    # ---- transition model on train (Table 15 companion)
    trans = {}
    trows = []
    for a, b in zip(stages[:-1], stages[1:]):
        trans[(a, b)] = fit_transition(p_tr[a], p_tr[b], a, b)
        trows.append(trans[(a, b)].table())
    save_table(cfg, pd.concat(trows, ignore_index=True).assign(score=args.score, calibration=args.calibration, split_source=src.get("train", "")),
               f"{base}_voi_transition")

    # ---- VoI policy over cost ratios x acquisition-cost grid, evaluated once on test
    rows, per_site = [], pd.DataFrame({"site_id": y_te.index, "label": y_te.values})
    grid = list(product(UNIT_COST_GRID["get_s"], UNIT_COST_GRID["render_s"], UNIT_COST_GRID["ocr_s"]))
    idx = y_te.index
    probs_te = {v: p_te[v].reindex(idx) for v in stages}

    def f1_of(res):
        # F1 at the train-fitted threshold of the stage each site stopped at (comparable with T11b)
        pred = pd.Series(0, index=res.index)
        for v in stages:
            m = res.stop_level == v
            pred[m] = (res.p[m] >= sc.thresholds[v]).astype(int)
        d = detection(y_te.reindex(res.index), res.p, 0.5)      # auroc / counts from the mixed score
        from sklearn.metrics import f1_score
        d["f1"] = f1_score(y_te.reindex(res.index), pred, zero_division=0)
        return d

    for ratio in ratios:
        for gi, (g, r, o) in enumerate(grid):
            secs = stage_seconds(g, r, o)
            acq = {v: secs.get(v, 0.0) / xrate for v in stages}
            res = run_voi(stages, probs_te, trans, acq, c_fn=ratio, c_fp=1.0, threshold=thr05)
            d = f1_of(res)
            stops = res.stop_level.value_counts(normalize=True)
            row = {"cost_ratio": f"{int(ratio)}:1", "a_S1": acq.get("S1", np.nan), "a_S2": acq.get("S2", np.nan), "f1": d["f1"], "auroc": d["auroc"],
                   "stop_S0": float(stops.get("S0", 0.0)), "stop_S1": float(stops.get("S1", 0.0)), "stop_S2": float(stops.get("S2", 0.0)),
                   "fetched_fraction": float(1 - stops.get(stages[0], 0.0)), "expected_cost": float(res.expected_cost.mean()),
                   "split_source": src.get("test", ""), "grid_point": gi, "get_s": g, "render_s": r, "ocr_s": o,
                   "central": (g, r, o) == CENTRE, "usd_per_1000": float(1000 * sum(usd_te[v] * (res.stop_level.map(lambda s, v=v: stages.index(s) >= stages.index(v) if s in stages else False)).mean() for v in stages)),
                   "score": args.score, "calibration": args.calibration, "exchange_seconds_per_fp_unit": xrate}
            rows.append(row)
            if (g, r, o) == CENTRE:
                per_site[f"p_voi_{int(ratio)}"] = res.p.reindex(idx).values
                per_site[f"stop_voi_{int(ratio)}"] = res.stop_level.reindex(idx).values
                per_site[f"pred_voi_{int(ratio)}"] = [int(p >= sc.thresholds[s]) if s in sc.thresholds else 0 for p, s in zip(res.p.reindex(idx), res.stop_level.reindex(idx))]
    # reference rows: always fetch (single stage) and the swept band of T11b
    for v in stages:
        d = detection(y_te, probs_te[v], sc.thresholds[v])
        rows.append({"cost_ratio": f"always fetch {v}", "a_S1": np.nan, "a_S2": np.nan, "f1": d["f1"], "auroc": d["auroc"],
                     "stop_S0": float(v == "S0"), "stop_S1": float(v == "S1"), "stop_S2": float(v == "S2"),
                     "fetched_fraction": float(v != stages[0]), "expected_cost": np.nan, "split_source": src.get("test", ""),
                     "usd_per_1000": 1000 * usd_te[v], "score": args.score, "calibration": args.calibration})
        per_site[f"p_{v}"] = probs_te[v].values; per_site[f"thr_{v}"] = sc.thresholds[v]
    t11b = cfg.results_root / f"t11b_{args.dataset}_{args.score}_{'-'.join(stages)}_test.csv"
    if t11b.exists():
        b = pd.read_csv(t11b); b = b[b.config.str.startswith("cascade")]
        for r in b.itertuples():
            rows.append({"cost_ratio": f"swept band [{r.t_low},{r.t_high}] (T11b)", "a_S1": np.nan, "a_S2": np.nan, "f1": r.f1, "auroc": r.auroc,
                         "stop_S0": getattr(r, "stop_S0", np.nan), "stop_S1": getattr(r, "stop_S1", np.nan), "stop_S2": getattr(r, "stop_S2", np.nan),
                         "fetched_fraction": 1 - (getattr(r, "stop_S0", 0.0) if not pd.isna(getattr(r, "stop_S0", np.nan)) else 0.0),
                         "expected_cost": np.nan, "split_source": r.split_source, "usd_per_1000": r.usd_per_1000, "score": args.score})
    else:
        print(f"[t20] {t11b} not found: no swept-band reference row (run T11b with --score {args.score} first)")
    voi = pd.DataFrame(rows)
    lead = ["cost_ratio", "a_S1", "a_S2", "f1", "auroc", "stop_S0", "stop_S1", "stop_S2", "fetched_fraction", "expected_cost", "split_source"]
    voi = voi[lead + [c for c in voi.columns if c not in lead]]
    save_table(cfg, voi, f"{base}_voi_test")
    save_table(cfg, per_site, f"{base}_voi_test_scores")
    show = voi[(voi.get("central", False) == True) | voi.cost_ratio.str.startswith(("always", "swept"))]  # noqa: E712
    print(show[lead[:-1]].round(3).to_string(index=False))
    cen = voi[voi.get("central", False) == True]  # noqa: E712
    if len(cen):
        cen = cen.assign(ratio=cen.cost_ratio.str.split(":").str[0].astype(float)).sort_values("ratio")
        curve(cen, "ratio", ["fetched_fraction", "f1"], cfg.figures_root / (out_name("t20_fetched_vs_costratio", inst) if args.dataset == "putra" and args.score == "lr" else f"{base}_fetched_vs_costratio"),
              xlabel="c_FN : c_FP", ylabel="fraction fetched / F1", logx=True)

    # ---- conformal stopping: train 50/50 (fit half / calibration half), thresholds per non-final level
    tr_ids = y_tr.index
    fit_ids, cal_ids = train_test_split(np.array(tr_ids), test_size=0.5, random_state=seed, stratify=y_tr.values)
    sc2 = StageScorer(args.score, args.calibration, seed)
    p_cal = {}
    for v in stages:
        t = tabs["train"][v]
        sc2.fit(v, t, [i for i in fit_ids if i in t.index])
        p_cal[v] = sc2.apply(v, t.loc[[i for i in cal_ids if i in t.index]])
    conf_rows, transfer_rows = [], []
    probs_te2 = {v: sc2.apply(v, tabs["test"][v]).reindex(idx) for v in stages}
    ttabs = {}
    if args.transfer_dataset:
        tstages = [v for v in args.transfer_stages.split(",") if v in stages]
        tt, tsrc = stage_tables(cfg, client, args.transfer_dataset, tstages, args.score, inst, args.limit, args.workers, bank,
                                splits=(args.transfer_split,))
        ttabs = tt.get(args.transfer_split, {})
        tstages = [v for v in tstages if v in ttabs]
        if len(tstages) >= 2:
            y_tt = pd.concat([t.label for t in ttabs.values()]).groupby(level=0).first().astype(int)
            probs_tt = {v: sc2.apply(v, ttabs[v]).reindex(y_tt.index) for v in tstages}
        else:
            print(f"[t20] transfer {args.transfer_dataset}: fewer than two stages available ({tstages}); conformal transfer skipped"); tstages = []
    for alpha in alphas:
        thr = {}
        for v in stages[:-1]:
            yc = y_tr.reindex(p_cal[v].index)
            thr[v] = conformal_thresholds(p_cal[v].values, yc.values, alpha, beta)
        res = run_conformal(stages, probs_te2, thr, sc2.thresholds[stages[-1]])
        conf_rows += conformal_summary(res, y_te, stages, thr, alpha, beta)
        if args.transfer_dataset and tstages:
            thr_t = {v: thr[v] for v in tstages[:-1]}
            res_t = run_conformal(tstages, probs_tt, thr_t, sc2.thresholds[tstages[-1]])
            transfer_rows += conformal_summary(res_t, y_tt, tstages, thr_t, alpha, beta)
    conf = pd.DataFrame(conf_rows).assign(score=args.score, calibration=args.calibration, split_source=src.get("test", ""))
    save_table(cfg, conf, f"{base}_conformal_test")
    print(conf.round(4).to_string(index=False))
    if transfer_rows:
        tconf = pd.DataFrame(transfer_rows).assign(score=args.score, calibration=args.calibration, split_source=f"{args.transfer_dataset}:{args.transfer_split}",
                                                   thresholds_from=f"{args.dataset} train (frozen)")
        tname = "t20_phreshphish_conformal_test" if args.transfer_dataset.startswith("phreshphish") and args.dataset == "putra" else f"t20_{args.transfer_dataset}_conformal_test"
        save_table(cfg, tconf, out_name(tname, inst) if args.score == "lr" else out_name(tname + "_direct", inst))
        print(tconf.round(4).to_string(index=False))
    first = conf[conf.level == stages[0]].sort_values("alpha")
    if len(first):
        curve(first, "alpha", ["realised_miss_rate", "fraction_stopped"],
              cfg.figures_root / (out_name("t20_conformal_miss_vs_alpha", inst) if args.dataset == "putra" and args.score == "lr" else f"{base}_conformal_miss_vs_alpha"),
              xlabel="alpha (miss tolerance)", ylabel=f"rate at level {stages[0]}")
    if client:
        client.close()


if __name__ == "__main__":
    main()
