#!/usr/bin/env python3
"""T07: indicator bank with combiners. Fits on the training split (5-fold CV),
applies once to the test split.

  python scripts/t07_bank.py --dataset putra --variant S1

Outputs (prefix results/t07_<dataset>_<variant>_):
  train_table.csv, test_table.csv       decision tables (bank_v1 incl. q_direct)
  combiners.csv                         per combiner: CV AUROC on train, test F1/AUROC/AUPRC at threshold fitted on train
  lr_odds_ratios.csv, catboost_importances.csv
  loqo.csv                              leave-one-question-out (CV on train)
  forward_selection.csv                 detection vs k (CV on train, final curve on test)
  learning_curve.csv                    LR test AUROC vs training-set size
  signals.csv                           LR on deterministic signals alone, signals + nouls
  train_scores.csv                      out-of-fold combiner scores on train (v2.3; used by T11, T20)
  forward_selection_semantic.csv        forward selection restricted to the semantic indicators (v2.3)
  results/t19_<dataset>_semantic_set.csv  deterministic / semantic / both partition (v2.3, shared with T19;
                                          written as results/t19_semantic_set.csv for --dataset putra --variant S1)
  figures/t07_<dataset>_<variant>_forward_selection
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _bootstrap import load_config, add_common_args, add_split_args, sites_for, banner, REPO
sys.path.insert(0, str(REPO / "src"))
from jev.client import client_from_config  # noqa: E402
from jev.run import run_questions, save_table  # noqa: E402
from questions.bank import load_bank, bank_questions  # noqa: E402
from eval.metrics import detection, best_f1_threshold  # noqa: E402
from eval.plots import curve  # noqa: E402
from eval.timing import TimingLog  # noqa: E402
from combine.combiners import (COMBINERS, make_lr, make_catboost, cv_auroc, cv_scores, fit_apply, lr_odds_ratios,  # noqa: E402
                               catboost_importances, leave_one_out, forward_selection, learning_curve,
                               question_columns, signals_from_states, SIGNAL_COLS)
from combine.stability import SEMANTIC_QUESTIONS  # noqa: E402
from summarise.html_truncation import make_token_counter  # noqa: E402


def estimated_tokens_per_site(df: pd.DataFrame, states_tokens: float, qs: dict, keep: list[str], token_model: str) -> float:
    """Billed tokens of the full-bank request scaled by the tiktoken share of the
    questions kept: measured(bank) x (state + kept questions) / (state + all
    questions). An estimate, flagged in reports/decisions.md (D19); the measured
    value is the full-bank row."""
    cnt = make_token_counter(token_model)
    q_tok = {q: cnt(v["instructions"]) for q, v in qs.items()}
    full = states_tokens + sum(q_tok.values())
    kept = states_tokens + sum(q_tok[q] for q in keep if q in q_tok)
    return float(df.input_tokens.mean() * kept / full) if full else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.add_argument("--variant", default="S1")
    ap.add_argument("--skip-slow", action="store_true", help="skip LOQO / forward selection / learning curve")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tag = f"t07_{args.dataset}_{args.variant}"
    client = client_from_config(cfg, tag, use_cache=not args.no_cache)
    bank = load_bank(cfg.bank_path)
    qs = bank_questions(bank, args.variant)
    timing = TimingLog(cfg.timing_root / f"{tag}.csv")
    seed = cfg["SEED"]

    tables = {}
    for split in ("train", "test"):
        sites, source = sites_for(cfg, args.dataset, split, limit=args.limit, reason="t07")
        args.split = split
        banner("t07", args, sites, source, client)
        df = run_questions(cfg, client, sites, args.dataset, args.variant, qs, f"{tag}_{split}", split, source,
                           workers=args.workers, timing=timing)
        save_table(cfg, df, f"{tag}_{split}_table")
        tables[split] = df[df.ok].reset_index(drop=True)
    tr, te = tables["train"], tables["test"]
    qcols = [c for c in question_columns(tr) if c in te.columns]           # indicators without q_direct
    allq = ["q_direct"] + qcols
    Xtr, ytr = tr[allq].astype(float), tr.label.values.astype(int)
    Xte, yte = te[allq].astype(float), te.label.values.astype(int)
    thr = cfg.get("operating_threshold", 0.5)

    rows, models, oofs = [], {}, {}
    for name, factory in COMBINERS.items():
        fac = (lambda f=factory: f(seed=seed)) if name == "catboost" else factory
        oof = cv_scores(fac, Xtr, ytr, seed=seed)
        oofs[name] = oof
        t_fit, _ = best_f1_threshold(ytr, oof)
        model, p = fit_apply(fac, Xtr, ytr, Xte)
        models[name] = model
        det_fit = detection(yte, p, t_fit)
        det_05 = detection(yte, p, thr)
        rows.append({"combiner": name, "features": "q_direct + 14 indicators", "cv_auroc_train": roc_auc_score(ytr, oof),
                     "threshold_fit_on_train": t_fit, "test_f1_at_fit": det_fit["f1"], "test_precision_at_fit": det_fit["precision"],
                     "test_recall_at_fit": det_fit["recall"], "test_f1_at_0.5": det_05["f1"], "test_auroc": det_05["auroc"],
                     "test_auprc": det_05["auprc"], "n_train": len(ytr), "n_test": len(yte)})
        te[f"p_{name}"] = p
    # q_direct alone as the reference row
    t_fit, _ = best_f1_threshold(ytr, Xtr.q_direct)
    d = detection(yte, Xte.q_direct, t_fit)
    rows.append({"combiner": "q_direct alone", "features": "q_direct", "cv_auroc_train": roc_auc_score(ytr, Xtr.q_direct),
                 "threshold_fit_on_train": t_fit, "test_f1_at_fit": d["f1"], "test_precision_at_fit": d["precision"],
                 "test_recall_at_fit": d["recall"], "test_f1_at_0.5": detection(yte, Xte.q_direct, thr)["f1"],
                 "test_auroc": d["auroc"], "test_auprc": d["auprc"], "n_train": len(ytr), "n_test": len(yte)})
    comb = pd.DataFrame(rows)
    comb["split_source"] = tables["test"].split_source.iloc[0] if len(te) else ""
    save_table(cfg, comb, f"{tag}_combiners")
    save_table(cfg, te[["site_id", "label", "split", "split_source"] + allq + [f"p_{n}" for n in COMBINERS]], f"{tag}_test_scores")
    trs = tr[["site_id", "label", "split", "split_source"] + allq].copy()
    for n, oof in oofs.items():
        trs[f"p_{n}"] = oof
    save_table(cfg, trs, f"{tag}_train_scores")          # out-of-fold scores on train (T11 band sweep, T20 transition model)
    print(comb.round(3).to_string(index=False))

    save_table(cfg, lr_odds_ratios(models["lr"], allq), f"{tag}_lr_odds_ratios")
    save_table(cfg, catboost_importances(models["catboost"], allq), f"{tag}_catboost_importances")

    # deterministic signals alone and with nouls
    sig_tr = signals_from_states(cfg, args.dataset, args.variant, list(tr.site_id))
    sig_te = signals_from_states(cfg, args.dataset, args.variant, list(te.site_id))
    if len(sig_tr) and len(sig_te):
        str_ = tr[["site_id"]].merge(sig_tr, on="site_id", how="left").fillna(0)
        ste_ = te[["site_id"]].merge(sig_te, on="site_id", how="left").fillna(0)
        srows = []
        for name, Xa, Xb in (("signals only", str_[SIGNAL_COLS], ste_[SIGNAL_COLS]),
                             ("signals + nouls", pd.concat([str_[SIGNAL_COLS], Xtr.reset_index(drop=True)], axis=1),
                              pd.concat([ste_[SIGNAL_COLS], Xte.reset_index(drop=True)], axis=1)),
                             ("nouls only", Xtr, Xte)):
            oof = cv_scores(make_lr, Xa.astype(float), ytr, seed=seed)
            _, p = fit_apply(make_lr, Xa.astype(float), ytr, Xb.astype(float))
            srows.append({"features": name, "cv_auroc_train": roc_auc_score(ytr, oof), "test_auroc": roc_auc_score(yte, p),
                          "test_f1_at_fit": detection(yte, p, best_f1_threshold(ytr, oof)[0])["f1"]})
        save_table(cfg, pd.DataFrame(srows), f"{tag}_signals")
        print(pd.DataFrame(srows).round(3).to_string(index=False))

        # v2.3: semantic-versus-deterministic partition (Table 14b). Deterministic =
        # the summariser's signal fields (computed in code, no model call); semantic =
        # the indicators no regular expression over the summary computes; both = union.
        sem = [q for q in SEMANTIC_QUESTIONS if q in qcols]
        import json as _json
        from summarise.state import state_tokens
        toks = []
        for sid in list(tr.site_id)[:200]:
            sp = cfg.derived_root / "states" / args.dataset / args.variant / f"{sid}.json"
            if sp.exists():
                toks.append(state_tokens(_json.load(open(sp, encoding="utf-8")), cfg["token_counter_model"]))
        st_tok = float(np.mean(toks)) if toks else float("nan")
        parts = {"deterministic": (str_[SIGNAL_COLS], ste_[SIGNAL_COLS], 0.0),
                 "semantic": (Xtr[sem], Xte[sem], estimated_tokens_per_site(tr, st_tok, qs, sem, cfg["token_counter_model"])),
                 "both": (pd.concat([str_[SIGNAL_COLS], Xtr[sem].reset_index(drop=True)], axis=1),
                          pd.concat([ste_[SIGNAL_COLS], Xte[sem].reset_index(drop=True)], axis=1),
                          estimated_tokens_per_site(tr, st_tok, qs, sem, cfg["token_counter_model"]))}
        prow = []
        for name, (Xa, Xb, tok) in parts.items():
            oof = cv_scores(make_lr, Xa.astype(float), ytr, seed=seed)
            _, p = fit_apply(make_lr, Xa.astype(float), ytr, Xb.astype(float))
            prow.append({"features": name, "cv_auroc_train": roc_auc_score(ytr, oof), "test_auroc": roc_auc_score(yte, p),
                         "test_f1_at_fit": detection(yte, p, best_f1_threshold(ytr, oof)[0])["f1"], "tokens_per_site": tok})
        sem_tab = pd.DataFrame(prow)
        sem_tab["tokens_source"] = ["none (computed in code)", "estimated from the full-bank request (D19)", "estimated from the full-bank request (D19)"]
        sem_tab["split_source"] = comb.split_source.iloc[0]
        name19 = "t19_semantic_set" if (args.dataset == "putra" and args.variant == "S1") else f"t19_{args.dataset}_{args.variant}_semantic_set"
        save_table(cfg, sem_tab, name19)
        print(sem_tab.round(3).to_string(index=False))

    if not args.skip_slow:
        loqo = leave_one_out(make_lr, Xtr, ytr, seed=seed)
        save_table(cfg, loqo, f"{tag}_loqo")
        fs = forward_selection(make_lr, Xtr, ytr, seed=seed)
        # final curve: for each k, fit on train with the chosen set, evaluate once on test
        fs["test_auroc"] = [roc_auc_score(yte, fit_apply(make_lr, Xtr[r.questions.split("|")], ytr, Xte[r.questions.split("|")])[1])
                            for r in fs.itertuples()]
        save_table(cfg, fs, f"{tag}_forward_selection")
        curve(fs, "k", ["cv_auroc", "test_auroc"], cfg.figures_root / f"{tag}_forward_selection",
              xlabel="number of questions k", ylabel="AUROC")
        # v2.3: forward selection restricted to the semantic indicators
        sem = [q for q in SEMANTIC_QUESTIONS if q in qcols]
        if sem:
            fss = forward_selection(make_lr, Xtr[sem], ytr, seed=seed)
            fss["test_auroc"] = [roc_auc_score(yte, fit_apply(make_lr, Xtr[r.questions.split("|")], ytr, Xte[r.questions.split("|")])[1])
                                 for r in fss.itertuples()]
            save_table(cfg, fss, f"{tag}_forward_selection_semantic")
        sizes = [s for s in cfg.get("learning_curve_sizes", [50, 100, 200]) if s <= len(ytr)] or [len(ytr)]
        lc = learning_curve(make_lr, Xtr, ytr, Xte, yte, sizes, seed=seed)
        save_table(cfg, lc, f"{tag}_learning_curve")
        print(fs[["k", "added", "cv_auroc", "test_auroc"]].round(3).to_string(index=False))
    timing.close(); client.close()


if __name__ == "__main__":
    main()
