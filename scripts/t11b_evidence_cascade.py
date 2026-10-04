#!/usr/bin/env python3
"""T11b: evidence-acquisition cascade S0 -> S1 -> S2 (-> S3 optional).

  python scripts/t11b_evidence_cascade.py --dataset putra --stages S0,S1,S2 [--score lr]

Stage probabilities are q_direct (default) or, with --score lr, the LR
combiner fitted on the training split's bank run per stage. Per-stage
evidence-acquisition seconds come from config `evidence_seconds`
(PLACEHOLDERS until T17 measures them; the table says so). Bands are swept on
train and evaluated once on test. Outputs results/t11b_<dataset>_{sweep_train,test}.csv,
per-site test scores (_test_scores.csv, for T_STATS), appends frontier points and
(v2.3) writes the unit-cost sensitivity sweep results/t11b_<dataset>_sensitivity.csv:
27 grid points over GET / render / screenshot+OCR seconds; the frontier ordering
(configurations sorted by mean acquisition seconds per site) per grid point and a
flag whether it differs from the central grid point. The primary reported quantity
remains the fraction of pages stopped per stage.
"""
from __future__ import annotations

from itertools import product

import argparse
import sys

import numpy as np
import pandas as pd

from _bootstrap import load_config, add_common_args, add_split_args, sites_for, banner, REPO
sys.path.insert(0, str(REPO / "src"))
from jev.client import client_from_config  # noqa: E402
from jev.run import run_questions  # noqa: E402
from questions.bank import load_bank, direct_question, bank_questions  # noqa: E402
from cascade.band import Stage, run_cascade, evaluate_cascade, sweep_bands, pick_band  # noqa: E402
from combine.combiners import make_lr, fit_apply, cv_scores, question_columns  # noqa: E402

UNIT_COST_GRID = {"get_s": [0.5, 1.5, 3.0], "render_s": [2.0, 5.0, 10.0], "ocr_s": [0.3, 1.0, 2.0]}   # WORKORDER v2.3, T11b


def stage_seconds(get_s: float, render_s: float, ocr_s: float) -> dict[str, float]:
    """Evidence levels in seconds: S0 nothing, S5 a plain GET, S1 a rendered DOM
    (the browser performs the GET), S2 the rendered DOM plus screenshot and OCR,
    S3 the rendered DOM (raw HTML)."""
    return {"S0": 0.0, "S5": get_s, "S1": render_s, "S2": render_s + ocr_s, "S3": render_s}


def sensitivity_table(configs: dict[str, list[Stage]], bands: dict[str, tuple[float, float]], y: pd.Series,
                      lat: dict[str, float], threshold: float) -> pd.DataFrame:
    """Re-cost every configuration at every grid point and record the ordering
    by mean seconds per site (acquisition + decide)."""
    rows, central = [], None
    grid = list(product(UNIT_COST_GRID["get_s"], UNIT_COST_GRID["render_s"], UNIT_COST_GRID["ocr_s"]))
    centre = (1.5, 5.0, 1.0)
    for gi, (g, r, o) in enumerate(grid):
        secs = stage_seconds(g, r, o)
        pts = []
        for name, stages in configs.items():
            st = [Stage(s.name, s.prob, s.usd_per_site, lat.get(s.name, 0.0) + secs.get(s.name, 0.0), s.is_decision) for s in stages]
            lo, hi = bands.get(name, (0, 0))
            ev = evaluate_cascade(run_cascade(st, lo, hi), y, threshold)
            pts.append((name, ev["mean_seconds_per_site"], ev["f1"]))
        pts.sort(key=lambda t: (t[1], -t[2]))
        ordering = " < ".join(f"{n}" for n, _, _ in pts)
        if (g, r, o) == centre:
            central = ordering
        rows.append({"grid_point": gi, "get_s": g, "render_s": r, "ocr_s": o, "ordering": ordering,
                     "seconds": "|".join(f"{n}={sec:.2f}" for n, sec, _ in pts)})
    df = pd.DataFrame(rows)
    df["ordering_changed"] = df.ordering != central
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.add_argument("--stages", default="S0,S1,S2")
    ap.add_argument("--score", default="direct", choices=["direct", "lr"])
    ap.add_argument("--objective", default="f1")
    ap.add_argument("--max-escalation", type=float, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    thr = cfg.get("operating_threshold", 0.5)
    stages = args.stages.split(",")
    tag = f"t11b_{args.dataset}_{args.score}_{'-'.join(stages)}"
    client = client_from_config(cfg, tag, use_cache=not args.no_cache)
    bank = load_bank(cfg.bank_path)
    ev = cfg.get("evidence_seconds", {})

    probs = {"train": {}, "test": {}}; usd = {}; lat = {}
    labels = {}
    for split in ("train", "test"):
        sites, src = sites_for(cfg, args.dataset, split, limit=args.limit, reason="t11b")
        args.split = split; banner("t11b", args, sites, src, client)
        labels[split] = pd.Series({s.id: s.label for s in sites})
        for v in stages:
            if args.score == "direct":
                # reuse T06 experiment names so cached responses are shared
                df = run_questions(cfg, client, sites, args.dataset, v, direct_question(bank), f"t06_{args.dataset}_{split}_{v}_direct", split, src, workers=args.workers)
                if not len(df):
                    continue
                df = df[df.ok].set_index("site_id")
                probs[split][v] = df.q_direct.astype(float)
            else:
                exp = f"t07_{args.dataset}_{v}_{split}" if v != "S0" else f"t06_{args.dataset}_{split}_S0_direct"
                qs = bank_questions(bank, v) if v != "S0" else direct_question(bank)
                df = run_questions(cfg, client, sites, args.dataset, v, qs, exp, split, src, workers=args.workers)
                if not len(df):
                    continue
                df = df[df.ok].set_index("site_id")
                probs[split][v] = df   # fit later
            usd[v] = float(df.cost_usd.mean()); lat[v] = float(df.latency_ms.mean() / 1000)
    if args.score == "lr":
        for v in list(probs["train"]):
            tr, te = probs["train"][v], probs["test"].get(v)
            if te is None:
                continue
            cols = ["q_direct"] + question_columns(tr) if v != "S0" else ["q_direct"]
            cols = [c for c in cols if c in te.columns]
            y = tr.label.values.astype(int)
            probs["train"][v] = pd.Series(cv_scores(make_lr, tr[cols].astype(float), y, seed=cfg["SEED"]), index=tr.index)
            probs["test"][v] = pd.Series(fit_apply(make_lr, tr[cols].astype(float), y, te[cols].astype(float))[1], index=te.index)

    def build(split):
        idx = labels[split].index
        return [Stage(v, probs[split][v].reindex(idx), usd.get(v, 0.0), lat.get(v, 0.0) + float(ev.get(v, 0.0)))
                for v in stages if v in probs[split]]

    st_tr, st_te = build("train"), build("test")
    if len(st_te) < 2:
        raise SystemExit("[t11b] fewer than two stages have states/probabilities; nothing to cascade")
    sweep = sweep_bands(st_tr, labels["train"], threshold=thr)
    sweep["evidence_seconds_source"] = cfg.get("evidence_seconds_source", "placeholder")
    sweep.to_csv(cfg.results_root / f"{tag}_sweep_train.csv", index=False)
    band = pick_band(sweep, args.objective, args.max_escalation, stages[0])
    print(f"[t11b] band chosen on train: [{band.t_low}, {band.t_high}] {args.objective}={band[args.objective]:.3f}")

    rows = []
    configs, bands = {}, {}
    for s in st_te:   # single-stage references
        name = f"{s.name} alone ({args.score})"
        rows.append({"config": name, **evaluate_cascade(run_cascade([s], 0, 0), labels["test"], thr)})
        configs[name] = [s]; bands[name] = (0, 0)
    res = run_cascade(st_te, band.t_low, band.t_high)
    cname = f"cascade {'->'.join(s.name for s in st_te)} band [{band.t_low},{band.t_high}] ({args.score})"
    rows.append({"config": cname, **evaluate_cascade(res, labels["test"], thr), "t_low": band.t_low, "t_high": band.t_high})
    configs[cname] = st_te; bands[cname] = (band.t_low, band.t_high)
    # per-site scores of the chosen cascade (paired statistics, T_STATS)
    res.rename_axis("site_id").reset_index().assign(label=labels["test"].reindex(res.index).values, threshold=thr) \
        .to_csv(cfg.results_root / f"{tag}_test_scores.csv", index=False)
    # additional cascade points at bounded escalation rates, for the frontier ordering (sensitivity sweep)
    for cap in (0.25, 0.5, 0.75):
        b2 = pick_band(sweep, args.objective, cap, stages[0])
        n2 = f"cascade {'->'.join(s.name for s in st_te)} band [{b2.t_low},{b2.t_high}] esc<={cap} ({args.score})"
        if (b2.t_low, b2.t_high) not in bands.values():
            configs[n2] = st_te; bands[n2] = (b2.t_low, b2.t_high)
    out = pd.DataFrame(rows)
    out["evidence_seconds_source"] = cfg.get("evidence_seconds_source", "placeholder")
    out["split_source"] = sites_for(cfg, args.dataset, "test", limit=1)[1]
    out.to_csv(cfg.results_root / f"{tag}_test.csv", index=False)
    cols = [c for c in ["config", "f1", "auroc", "ppv_at_1pct", "usd_per_1000", "mean_seconds_per_site"] + [f"stop_{s.name}" for s in st_te] if c in out]
    print(out[cols].round(3).to_string(index=False))
    fp = cfg.results_root / "frontier_points.csv"
    out[cols].rename(columns={"mean_seconds_per_site": "latency_s_per_site"}).assign(task="t11b", dataset=args.dataset) \
        .to_csv(fp, mode="a", header=not fp.exists(), index=False)
    # v2.3 unit-cost sensitivity sweep (27 grid points)
    sens = sensitivity_table(configs, bands, labels["test"], lat, thr)
    sens["score"] = args.score; sens["stages"] = "-".join(stages); sens["split_source"] = out.split_source.iloc[0]
    sp = cfg.results_root / f"t11b_{args.dataset}_sensitivity.csv"     # one file per dataset, rows tagged by score and stages
    if sp.exists():
        prev = pd.read_csv(sp)
        prev = prev[~((prev.score == args.score) & (prev.stages == "-".join(stages)))]
        sens = pd.concat([prev, sens], ignore_index=True)
    sens.to_csv(sp, index=False)
    print(f"[t11b] sensitivity: ordering changed at {int(sens.ordering_changed.sum())} of {len(sens)} grid points "
          f"(central ordering: {sens[~sens.ordering_changed].ordering.iloc[0] if (~sens.ordering_changed).any() else 'n/a'})")
    client.close()


if __name__ == "__main__":
    main()
