#!/usr/bin/env python3
"""T_STATS: paired statistics for every interpreted difference (paper/draft.md
Section 4.12).

  python scripts/t_stats.py --dataset putra [--pair "name=path:col[@thr],path:col[@thr]" ...]
                                            [--paper2-best data/paper2/best_svlm_decisions.csv:decision]

For each pair of per-site score tables on the same sites: paired bootstrap
(1,000 resamples over sites) 95 % intervals for the F1 difference (at the two
operating thresholds) and the AUROC difference, and McNemar at the operating
thresholds (src/eval/metrics.py). The default pair list covers: bank lr vs
q_direct; S2 and S3 vs S1 (q_direct); each budget vs 2,000 tokens; cascade
(lr arm) vs S1 alone; VoI (10:1, central costs) vs swept band and vs S1 alone;
stable-set vs all-indicator combiner under transfer (T19 per-site scores);
prior-corrected (EM) vs R1 threshold (T21 per-site scores); decision model vs
open-weight twin (T09 scores); decision model vs best SVLM (paper-2 per-site
decisions, --paper2-best). Pairs whose files do not exist are reported with
status "missing" so the table shows what is still to be run.

Output: results/t_stats_<dataset>_pairwise.csv (Table 1e columns pair, metric,
estimate, ci_low, ci_high, mcnemar_p, n_sites, n_test_evaluations; then status
and the two sides) and results/t_stats_<dataset>_test_evaluations.csv (the
configurations evaluated on the test split that the count refers to).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import load_config, add_common_args, REPO
sys.path.insert(0, str(REPO / "src"))
from eval.metrics import paired_bootstrap_diff, mcnemar, best_f1_threshold  # noqa: E402


def parse_side(spec: str) -> tuple[str, str, float | None]:
    path, _, rest = spec.partition(":")
    col, _, thr = rest.partition("@")
    return path, col, (float(thr) if thr else None)


def load_side(R: Path, spec: str) -> tuple[pd.Series, pd.Series, float] | None:
    path, col, thr = parse_side(spec)
    p = Path(path) if Path(path).is_absolute() else (REPO / path if (REPO / path).exists() else R / path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if col not in df or "site_id" not in df:
        return None
    if "ok" in df:
        df = df[df.ok]
    df = df.set_index("site_id")
    if thr is None:
        thr = float(df["threshold"].iloc[0]) if "threshold" in df and df["threshold"].notna().any() else 0.5
    return df[col].astype(float), df["label"].astype(int) if "label" in df else None, thr


def train_threshold(R: Path, train_csv: str, col: str, default: float = 0.5) -> float:
    p = R / train_csv
    if not p.exists():
        return default
    d = pd.read_csv(p)
    if "ok" in d:
        d = d[d.ok]
    return best_f1_threshold(d.label, d[col])[0] if col in d and len(d) else default


def combiner_threshold(R: Path, csv: str, combiner: str, default: float = 0.5) -> float:
    p = R / csv
    if not p.exists():
        return default
    d = pd.read_csv(p)
    r = d[d.combiner == combiner]
    return float(r.threshold_fit_on_train.iloc[0]) if len(r) else default


def default_pairs(R: Path, D: str, paper2_best: str | None) -> list[tuple[str, str, str]]:
    t_lr = combiner_threshold(R, f"t07_{D}_S1_combiners.csv", "lr")
    t_qd = combiner_threshold(R, f"t07_{D}_S1_combiners.csv", "q_direct alone")
    pairs = [("bank lr vs q_direct (S1)", f"t07_{D}_S1_test_scores.csv:p_lr@{t_lr}", f"t07_{D}_S1_test_scores.csv:q_direct@{t_qd}")]
    t_s1 = train_threshold(R, f"t06_{D}_train_S1_direct.csv", "q_direct")
    for v in ("S2", "S3", "S0"):
        t_v = train_threshold(R, f"t06_{D}_train_{v}_direct.csv", "q_direct")
        pairs.append((f"q_direct {v} vs S1", f"t06_{D}_test_{v}_direct.csv:q_direct@{t_v}", f"t06_{D}_test_S1_direct.csv:q_direct@{t_s1}"))
    t_2k = train_threshold(R, f"t06b_{D}_train_S1b-2000_direct.csv", "q_direct")
    for b in (100, 250, 500, 1000):
        t_b = train_threshold(R, f"t06b_{D}_train_S1b-{b}_direct.csv", "q_direct")
        pairs.append((f"q_direct budget {b} vs 2000", f"t06b_{D}_test_S1b-{b}_direct.csv:q_direct@{t_b}", f"t06b_{D}_test_S1b-2000_direct.csv:q_direct@{t_2k}"))
    pairs.append(("evidence cascade (lr arm) vs S1 alone (lr)", f"t11b_{D}_lr_S0-S1-S2_test_scores.csv:p", f"t07_{D}_S1_test_scores.csv:p_lr@{t_lr}"))
    pairs.append(("VoI 10:1 vs swept band (lr arm)", f"t20_{D}_voi_test_scores.csv:p_voi_10", f"t11b_{D}_lr_S0-S1-S2_test_scores.csv:p"))
    pairs.append(("VoI 10:1 vs S1 alone (lr)", f"t20_{D}_voi_test_scores.csv:p_voi_10", f"t07_{D}_S1_test_scores.csv:p_lr@{t_lr}"))
    # transfer: stable-set vs all-indicator combiner on the transfer sites (thresholds stored in the scores file)
    for p in sorted(glob.glob(str(R / f"t19_{D}_S1_transfer_scores*.csv"))):
        d = pd.read_csv(p)
        if "p_lr_stable_set" in d and "p_lr_all_indicators" in d:
            pairs.append((f"stable-set vs all-indicator lr under transfer ({Path(p).stem})",
                          f"{Path(p).name}:p_lr_stable_set@{d.thr_lr_stable_set.iloc[0]}", f"{Path(p).name}:p_lr_all_indicators@{d.thr_lr_all_indicators.iloc[0]}"))
    for p in sorted(glob.glob(str(R / "t21_*_scores.csv"))):
        d = pd.read_csv(p)
        if "thr_prior_em" in d and "thr_R1" in d:
            pairs.append((f"prior_em vs R1 threshold ({Path(p).stem})", f"{Path(p).name}:p_shift_em@{d.thr_prior_em.iloc[0]}", f"{Path(p).name}:p_cal@{d.thr_R1.iloc[0]}"))
    for p in sorted(glob.glob(str(R / f"t09_{D}_S1_*_test_scores.csv"))):
        m = Path(p).stem.replace(f"t09_{D}_S1_", "").replace("_test_scores", "")
        t_tw = combiner_threshold(R, f"t09_{D}_S1_combiners.csv", "lr")
        pairs.append((f"decision model lr vs twin {m} lr", f"t07_{D}_S1_test_scores.csv:p_lr@{t_lr}", f"{Path(p).name}:p_lr@{t_tw}"))
    pairs.append(("decision model (bank lr) vs best SVLM (paper 2)", f"t07_{D}_S1_test_scores.csv:p_lr@{t_lr}",
                  paper2_best or "data/paper2/best_svlm_decisions.csv:decision@0.5"))
    return pairs


def count_test_evaluations(R: Path, D: str) -> pd.DataFrame:
    """Configurations evaluated on the test split: one row per configuration in
    the known test-split result tables."""
    spec = [(f"t06_{D}_test_table.csv", "config"), (f"t07_{D}_*_combiners.csv", "combiner"), (f"t06b_{D}_test_curve.csv", "variant"),
            (f"t10_{D}_test.csv", "config"), (f"t11_{D}_*_test.csv", "config"), (f"t11b_{D}_*_test.csv", "config"),
            (f"t20_{D}_voi_test.csv", "cost_ratio"), (f"t20_{D}_conformal_test.csv", "alpha"), (f"t13_{D}_S1_calibration.csv", "score"),
            (f"t15_{D}_*_summary.csv", "arm"), ("t19_transfer_by_combiner.csv", "combiner"), ("t21_prior_shift.csv", "method"),
            (f"t08_{D}_baselines.csv", "baseline"), (f"t09_{D}_S1_combiners.csv", "combiner")]
    rows = []
    for pat, col in spec:
        for p in sorted(glob.glob(str(R / pat))):
            d = pd.read_csv(p)
            if col not in d:
                continue
            if "config" in d and pat.startswith("t06_"):
                d = d[~d.config.astype(str).str.startswith("paper2")]
            n = int(d[col].astype(str).nunique()) if col != "alpha" else int(d[["alpha", "level"]].drop_duplicates().shape[0])
            rows.append({"file": Path(p).name, "configurations": n})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--dataset", default="bootstrap-phreshphish")
    ap.add_argument("--pair", nargs="*", default=[], help='"name=sideA,sideB" with side = path:col[@thr]; replaces the defaults when --only-given')
    ap.add_argument("--only-given", action="store_true")
    ap.add_argument("--paper2-best", default=None, help="per-site decisions of the best paper-2 SVLM, path:col[@thr]")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    R = cfg.results_root
    pairs = [] if args.only_given else default_pairs(R, args.dataset, args.paper2_best)
    for spec in args.pair:
        name, _, rest = spec.partition("="); a, _, b = rest.partition(",")
        pairs.append((name, a, b))
    evals = count_test_evaluations(R, args.dataset)
    n_eval = int(evals.configurations.sum()) if len(evals) else 0
    evals.to_csv(R / f"t_stats_{args.dataset}_test_evaluations.csv", index=False)

    rows = []
    for name, sa, sb in pairs:
        A, B = load_side(R, sa), load_side(R, sb)
        if A is None or B is None:
            miss = [s for s, x in ((sa, A), (sb, B)) if x is None]
            for metric in ("f1", "auroc"):
                rows.append({"pair": name, "metric": metric, "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "mcnemar_p": np.nan,
                             "n_sites": 0, "n_test_evaluations": n_eval, "status": "missing: " + "; ".join(miss), "side_a": sa, "side_b": sb,
                             "threshold_a": np.nan, "threshold_b": np.nan})
            continue
        pa, ya, ta = A; pb, yb, tb = B
        common = pa.index.intersection(pb.index)
        y = (ya if ya is not None else yb).reindex(common)
        pa, pb = pa.reindex(common), pb.reindex(common)
        m = pa.notna() & pb.notna() & y.notna()
        if m.sum() < 10 or y[m].nunique() < 2:
            for metric in ("f1", "auroc"):
                rows.append({"pair": name, "metric": metric, "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "mcnemar_p": np.nan,
                             "n_sites": int(m.sum()), "n_test_evaluations": n_eval, "status": "too few paired sites", "side_a": sa, "side_b": sb,
                             "threshold_a": ta, "threshold_b": tb})
            continue
        mc = mcnemar(y[m], pa[m] >= ta, pb[m] >= tb)
        for metric in ("f1", "auroc"):
            r = paired_bootstrap_diff(y[m], pa[m], pb[m], metric, ta, tb, n=args.n, seed=cfg["SEED"])
            rows.append({"pair": name, "metric": metric, "estimate": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                         "mcnemar_p": mc["mcnemar_p"] if metric == "f1" else np.nan, "n_sites": r["n_sites"], "n_test_evaluations": n_eval,
                         "status": "ok", "side_a": sa, "side_b": sb, "threshold_a": ta, "threshold_b": tb,
                         "mcnemar_b": mc["mcnemar_b"], "mcnemar_c": mc["mcnemar_c"], "mcnemar_method": mc["mcnemar_method"]})
    out = pd.DataFrame(rows)
    out.to_csv(R / f"t_stats_{args.dataset}_pairwise.csv", index=False)
    show = out[["pair", "metric", "estimate", "ci_low", "ci_high", "mcnemar_p", "n_sites", "status"]]
    print(show.round(4).to_string(index=False))
    print(f"[t_stats] configurations evaluated on the test split: {n_eval} (results/t_stats_{args.dataset}_test_evaluations.csv)")


if __name__ == "__main__":
    main()
