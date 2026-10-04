#!/usr/bin/env python3
"""T19: indicator stability and transfer-aware selection (RQ2, method M1).

  python scripts/t19_stability.py --dataset putra --variant S1 \
      --stability-dataset phreshphish-stab --transfer-dataset phreshphish-20k \
      [--extra-transfer phishark=phishark fresh=fresh] [--instantiation twin-gemma4:12b]

Inputs: bank_v1 probabilities on the source training split (cached T07
responses), on the PhreshPhish stability sample (one bank request per site; the
only new API calls of this task) and on the transfer evaluation sample (cached
T14 responses). Per indicator: AUROC and sign on each corpus and the class
stable / artefact / uninformative (src/combine/stability.py). The stable set is
frozen in src/questions/stable_set_v1.yaml the first time the task runs on the
real data (dataset putra); later runs read the frozen file. Combiners fitted on
the source training split only: lr on all indicators, lr on the stable set,
q_direct alone (thresholds on train), evaluated once on the source test split
and frozen (R0) on the transfer sample(s).

Outputs (paper/tables.md Tables 13, 14):
  results/t19_stability_indicators.csv   (prefixed with the dataset when not putra/S1)
  results/t19_transfer_by_combiner.csv   transfer loss per combiner with two-sample bootstrap CIs
  results/t19_<tag>_transfer_scores.csv  per-site scores on the transfer sample (paired statistics, T_STATS)
  figures/t19_stability_scatter.{pdf,png}
Twin instantiation (--instantiation twin-<model>) reads the T09 tables instead
of the Jev cache and suffixes every output with _twin-<model>.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from _bootstrap import load_config, add_common_args, add_split_args, banner, REPO, decision_table, out_name
sys.path.insert(0, str(REPO / "src"))
from jev.client import client_from_config  # noqa: E402
from jev.run import save_table  # noqa: E402
from questions.bank import load_bank, bank_questions  # noqa: E402
from eval.metrics import detection, best_f1_threshold, two_sample_bootstrap_diff  # noqa: E402
from combine.combiners import make_lr, cv_scores, fit_apply, question_columns  # noqa: E402
from combine.stability import stability_table, stable_set, STABLE_FLOOR, ARTEFACT_HIGH, ARTEFACT_LOW  # noqa: E402
from eval.plots import stability_scatter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.add_argument("--variant", default="S1")
    ap.add_argument("--stability-dataset", default="phreshphish-stab")
    ap.add_argument("--stability-split", default="all", choices=["train", "test", "all"])
    ap.add_argument("--stability-variant", default="S1")
    ap.add_argument("--transfer-dataset", default="phreshphish-20k")
    ap.add_argument("--transfer-split", default="all", choices=["train", "test", "all"])
    ap.add_argument("--transfer-variant", default="S1")
    ap.add_argument("--extra-transfer", nargs="*", default=[], help="name=dataset[:split] additional frozen transfer corpora (phishark, fresh)")
    ap.add_argument("--instantiation", default="jev", help="jev | twin-<model>")
    ap.add_argument("--freeze-path", default=None, help="default src/questions/stable_set_v1.yaml for --dataset putra, else a dev file under results/")
    args = ap.parse_args()
    cfg = load_config(args.config)
    inst = args.instantiation
    real = args.dataset == "putra" and args.variant == "S1" and inst == "jev"
    tag = out_name(f"t19_{args.dataset}_{args.variant}", inst)
    client = client_from_config(cfg, tag, use_cache=not args.no_cache) if inst == "jev" else None
    bank = load_bank(cfg.bank_path)
    qs = bank_questions(bank, args.variant)
    seed = cfg["SEED"]

    # ---- decision tables (cache hits for the source and transfer sets; the stability sample is the only new spend)
    tr, src_tr = decision_table(cfg, client, args.dataset, args.variant, "train", qs, f"t07_{args.dataset}_{args.variant}_train", inst, args.limit, args.workers, "t19 source train")
    te, src_te = decision_table(cfg, client, args.dataset, args.variant, "test", qs, f"t07_{args.dataset}_{args.variant}_test", inst, args.limit, args.workers, "t19 source test")
    args.split = "train"; banner("t19", args, tr, src_tr, client)
    st, src_st = decision_table(cfg, client, args.stability_dataset, args.stability_variant, args.stability_split,
                                bank_questions(bank, args.stability_variant), f"{tag}_stability", inst, args.limit, args.workers, "t19 stability sample")
    print(f"[t19] stability sample {args.stability_dataset}: {len(st)} sites")
    def corpus_key(ds: str) -> str:
        return next((k for k in ("phreshphish", "phishark", "fresh") if k in ds), ds)
    transfers = {corpus_key(args.transfer_dataset): (args.transfer_dataset, args.transfer_split, args.transfer_variant)}
    for e in args.extra_transfer:
        name, rest = e.split("="); ds, _, sp = rest.partition(":")
        transfers[name] = (ds, sp or "all", "S1")
    ttabs = {}
    for name, (ds, sp, var) in transfers.items():
        try:
            ttabs[name], _ = decision_table(cfg, client, ds, var, sp, bank_questions(bank, var), f"t14_{ds}_{var}", inst, args.limit, args.workers, f"t19 transfer {name}")
            print(f"[t19] transfer {name} ({ds}): {len(ttabs[name])} sites")
        except SystemExit as e:
            print(f"[t19] transfer {name}: {e}; column left empty")
    if not len(tr) or not len(st):
        raise SystemExit("[t19] no source train or stability table; nothing to do")

    # ---- per-indicator inventory (Table 13)
    ind = [c for c in question_columns(tr) if c in st.columns]          # indicators without q_direct
    src_name = f"{args.dataset.replace('-', '_')}_train" if not real else "putra_train"
    tgt_name = "phresh_stab" if args.stability_dataset.startswith("phreshphish") else f"{args.stability_dataset.replace('-', '_')}_stab"
    inv = stability_table(tr, st, ind, src_name, tgt_name)
    inv["split_source"] = src_tr
    inv_name = "t19_stability_indicators" if real else out_name(f"t19_{args.dataset}_{args.variant}_stability_indicators", inst)
    save_table(cfg, inv, inv_name)
    print(inv.round(3).to_string(index=False))
    stability_scatter(inv, f"auroc_{src_name}", f"auroc_{tgt_name}", cfg.figures_root / ("t19_stability_scatter" if real else f"{tag}_stability_scatter"),
                      xlabel=f"AUROC on {args.dataset} train", ylabel=f"AUROC on {args.stability_dataset}")

    # ---- freeze or load the stable set
    fp = REPO / (args.freeze_path or ("src/questions/stable_set_v1.yaml" if real else f"results/{tag}_stable_set_dev.yaml"))
    computed = stable_set(inv)
    if fp.exists():
        frozen = yaml.safe_load(open(fp, encoding="utf-8"))
        stable = list(frozen.get("questions", []))
        if set(stable) != set(computed):
            print(f"[t19] WARNING frozen stable set {fp} differs from the recomputed one: frozen {stable} vs computed {computed}; the frozen set is used")
        else:
            print(f"[t19] frozen stable set loaded from {fp}: {stable}")
    else:
        stable = computed
        fp.parent.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump({"version": "stable_set_v1" if real else f"stable_set_dev_{tag}", "frozen": real,
                        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source_corpus": f"{args.dataset} {args.variant} train",
                        "stability_corpus": f"{args.stability_dataset} {args.stability_variant} {args.stability_split}",
                        "rule": {"stable": f"same sign and oriented AUROC > {STABLE_FLOOR} on both corpora",
                                 "artefact": f"oriented AUROC > {ARTEFACT_HIGH} on one corpus and < {ARTEFACT_LOW} or reversed on the other"},
                        "n_source": int(len(tr)), "n_stability": int(len(st)), "questions": stable,
                        "classes": dict(zip(inv.question, inv["class"])),
                        "note": "development file (bootstrap / pseudo split); never used for a report" if not real else
                                "frozen before any test-split evaluation (WORKORDER v2.3, T19)"},
                       open(fp, "w", encoding="utf-8"), sort_keys=False)
        print(f"[t19] stable set {'FROZEN' if real else 'written (dev)'} to {fp}: {stable}")

    # ---- combiners on the source train split, evaluated once on source test and frozen on the transfer corpora (Table 14)
    ytr, yte = tr.label.values.astype(int), te.label.values.astype(int)
    thr05 = cfg.get("operating_threshold", 0.5)
    combos = {"lr all indicators": ind, "lr stable set": [c for c in stable if c in ind], "q_direct": ["q_direct"],
              "lr all indicators + q_direct": ["q_direct"] + ind}
    rows, per_site = [], {}
    for name, cols in combos.items():
        if not cols:
            print(f"[t19] {name}: empty feature set (no stable indicators); row left empty")
            rows.append({"combiner": name, "n_indicators": 0}); continue
        if name == "q_direct":
            oof = tr.q_direct.values.astype(float); p_te = te.q_direct.values.astype(float); model = None
        else:
            oof = cv_scores(make_lr, tr[cols].astype(float), ytr, seed=seed)
            model, p_te = fit_apply(make_lr, tr[cols].astype(float), ytr, te[cols].astype(float))
        t_fit = best_f1_threshold(ytr, oof)[0]
        d_te = detection(yte, p_te, t_fit)
        row = {"combiner": name, "n_indicators": len([c for c in cols if c != "q_direct"]), "putra_test_f1": d_te["f1"]}
        for tname in ("phreshphish", "phishark", "fresh"):
            row[f"r0_f1_{tname}"] = np.nan
        loss = {}
        for tname, tt in ttabs.items():
            tcols = [c for c in cols if c in tt.columns]
            if len(tcols) < len(cols):
                print(f"[t19] {name} on {tname}: missing columns {set(cols) - set(tcols)}; skipped"); continue
            p_t = tt.q_direct.values.astype(float) if model is None else model.predict_proba(tt[tcols].astype(float))[:, 1]
            yt = tt.label.values.astype(int)
            d_t = detection(yt, p_t, t_fit)
            row[f"r0_f1_{tname}"] = d_t["f1"]; row[f"r0_auroc_{tname}"] = d_t["auroc"]
            ci = two_sample_bootstrap_diff(yte, p_te, yt, p_t, "f1", t_fit, t_fit, seed=seed)
            loss[tname] = (ci["estimate"], ci["ci_low"], ci["ci_high"])
            per_site.setdefault(tname, pd.DataFrame({"site_id": tt.site_id, "label": yt}))[f"p_{name.replace(' ', '_')}"] = p_t
            per_site[tname][f"thr_{name.replace(' ', '_')}"] = t_fit
        main_t = next((t for t in ("phreshphish", "phishark", "fresh") if t in loss), None)
        row["transfer_loss"], row["ci_low"], row["ci_high"] = loss.get(main_t, (np.nan, np.nan, np.nan))
        row.update({"threshold_fit_on_train": t_fit, "cv_auroc_train": roc_auc_score(ytr, oof) if len(set(ytr)) == 2 else np.nan,
                    "putra_test_auroc": d_te["auroc"], "putra_test_f1_at_0.5": detection(yte, p_te, thr05)["f1"],
                    "questions": "|".join(cols)})
        rows.append(row)
    cols_out = ["combiner", "n_indicators", "putra_test_f1", "r0_f1_phreshphish", "r0_f1_phishark", "r0_f1_fresh", "transfer_loss", "ci_low", "ci_high"]
    out = pd.DataFrame(rows)
    out = out[[c for c in cols_out if c in out] + [c for c in out.columns if c not in cols_out]]
    out["transfer_loss_ci"] = "two-sample bootstrap (source test vs transfer sites), 1,000 resamples"
    out["split_source"] = src_te
    save_table(cfg, out, "t19_transfer_by_combiner" if real else out_name(f"t19_{args.dataset}_{args.variant}_transfer_by_combiner", inst))
    print(out[[c for c in cols_out if c in out]].round(3).to_string(index=False))
    for tname, ps in per_site.items():
        save_table(cfg, ps, f"{tag}_transfer_scores" if len(per_site) == 1 else f"{tag}_transfer_scores_{tname}")
    if client:
        client.close()


if __name__ == "__main__":
    main()
