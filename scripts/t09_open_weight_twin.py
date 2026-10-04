#!/usr/bin/env python3
"""T09: open-weight twin (mandatory since v2.3; GPU session).

  python scripts/t09_open_weight_twin.py --dataset putra --variant S1 --models gemma4:12b,qwen3.5:9b --backend ollama
  python scripts/t09_open_weight_twin.py --dataset putra --variant S0 --models ...          # q_direct on the URL state (T20 twin arm)
  python scripts/t09_open_weight_twin.py --dataset putra --parity                             # after T19-T21 ran for both instantiations

For every bank question and every state the model is forced to a single token
from {yes, no}; P(yes) is read from the logprobs renormalised over the two
tokens (src/baselines/twin.py; vLLM guided decoding or Ollama logprobs; every
call persisted and cached under data/raw_responses/twin/). Same states, same
combiners (vote, lr, CatBoost), same thresholds-on-train protocol and the same
test split as T07. Per model: seconds per site (sum of generation seconds over
the questions) and USD per 1,000 sites at config gpu_usd_per_hour.

Outputs: results/t09_<dataset>_<variant>_<model>_{train,test}_table.csv (decision
tables with the T07 layout; inputs of T19-T21 with --instantiation twin-<model>),
results/t09_<dataset>_<variant>_<model>_{train,test}_scores.csv,
results/t09_<dataset>_<variant>_combiners.csv (Table 2 columns plus model),
results/t09_<dataset>_<variant>_summary.csv, results/t09_<dataset>_<variant>_per_question.csv
(per-question AUROC, twin vs decision model, Spearman in the summary).
--backend fake is a deterministic stand-in for exercising the code path without
a GPU; it is accepted on bootstrap / fixture datasets only and writes under
results/dryrun/. --parity writes results/t09_<dataset>_parity.csv (Table 18).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from _bootstrap import load_config, add_common_args, add_split_args, sites_for, banner, REPO, is_bootstrap, inst_suffix
sys.path.insert(0, str(REPO / "src"))
from jev.run import load_state  # noqa: E402
from questions.bank import load_bank, bank_questions, direct_question  # noqa: E402
from eval.metrics import detection, best_f1_threshold  # noqa: E402
from eval.timing import TimingLog  # noqa: E402
from combine.combiners import COMBINERS, cv_scores, fit_apply, question_columns  # noqa: E402
from baselines.twin import TwinClient  # noqa: E402


def mtag(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def answer_split(cfg, twin, sites, dataset, variant, qs, split, source, timing, model, limit_note=""):
    rows = []
    for i, s in enumerate(sites):
        st = load_state(cfg, dataset, variant, s.id)
        if st is None:
            continue
        row = {"site_id": s.id, "label": s.label, "dataset": dataset, "variant": variant, "split": split, "split_source": source,
               "ok": True, "model": model, "backend": twin.backend, "seconds": 0.0, "cached": True, "notes": ""}
        for qid, q in qs.items():
            r = twin.ask(st, q["instructions"])
            row[qid] = r.p_yes
            row["seconds"] += r.seconds; row["cached"] = row["cached"] and r.cached
            if not r.ok:
                row["ok"] = False
            if r.note:
                row["notes"] = (row["notes"] + "; " if row["notes"] else "") + f"{qid}: {r.note}"
        timing.record(f"t09_{dataset}_{variant}_{mtag(model)}", s.id, variant, "decide", row["seconds"], "cached" if row["cached"] else "")
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"[t09] {model} {split}: {i + 1}/{len(sites)}", flush=True)
    return pd.DataFrame(rows)


def parity(cfg, dataset: str, models: list[str]) -> None:
    """Table 18: per hypothesis H1..H9 the metric, the decision-model value, the
    twin value and whether both lie on the same side of the reference."""
    R = cfg.results_root

    def val(path, where, col):
        p = R / path
        if not p.exists():
            return np.nan, f"missing {path}"
        d = pd.read_csv(p)
        for k, v in where.items():
            d = d[d[k].astype(str) == str(v)]
        return (float(d[col].iloc[0]) if len(d) and col in d else np.nan), ""

    ref_f1 = np.nan
    refp = REPO / cfg.get("paper2_reference_csv", "data/paper2/reference_rows.csv")
    if refp.exists():
        r = pd.read_csv(refp); r = r[(r.dataset.astype(str) == "putra") & r.model.astype(str).str.contains("gemma|muse|qwen|gpt-oss|llama|mistral", case=False)]
        ref_f1 = float(r.f1.max()) if len(r) else np.nan
    rows = []
    for m in models:
        sfx = inst_suffix(f"twin-{m}")
        spec = [
            ("H1", "lr test F1 at the train-fitted threshold (S1); reference best SVLM F1 - 0.03",
             (f"t07_{dataset}_S1_combiners.csv", {"combiner": "lr"}, "test_f1_at_fit"),
             (f"t09_{dataset}_S1_combiners.csv", {"combiner": "lr", "model": m}, "test_f1_at_fit"), ref_f1 - 0.03 if not np.isnan(ref_f1) else np.nan),
            ("H2", "smallest budget within 1 F1 point of 2,000 tokens (not run for the twin)",
             (f"t06b_{dataset}_test_curve.csv", {}, "smallest_budget_within_1pt_f1"), (f"t06b_{dataset}_test_curve{sfx}.csv", {}, "smallest_budget_within_1pt_f1"), 1000),
            ("H3", "lr F1 minus q_direct F1 (S1); reference 0",
             (f"t07_{dataset}_S1_combiners.csv", {"combiner": "lr"}, "test_f1_at_fit"), (f"t09_{dataset}_S1_combiners.csv", {"combiner": "lr", "model": m}, "test_f1_at_fit"), None),
            ("H4", "fraction of pages never fetched by the swept-band evidence cascade (lr arm)",
             (f"t11b_{dataset}_lr_S0-S1-S2_test.csv", {}, "stop_S0"), (f"t11b_{dataset}_lr_S0-S1-S2_test{sfx}.csv", {}, "stop_S0"), 0.5),
            ("H5", "ECE (10 equal-mass bins) of the raw lr score; reference 0.10",
             (f"t13_{dataset}_S1_calibration.csv", {"score": "lr_bank", "calibration": "raw"}, "ece_10bins"),
             (f"t13_{dataset}_S1_calibration{sfx}.csv", {"score": "lr_bank", "calibration": "raw"}, "ece_10bins"), 0.10),
            ("H6", "A5 flip rate at 1,000 queries against the lr bank score; reference 0.5",
             (f"t15_{dataset}_A5_summary.csv", {"score": "lr_bank", "budget": 1000}, "flip_rate"),
             (f"t15_{dataset}_A5_summary.csv", {"score": f"twin:{m}", "budget": 1000}, "flip_rate"), 0.5),
            ("H7", "transfer loss of the stable-set lr minus the all-indicator lr; reference 0",
             ("t19_transfer_by_combiner.csv", {"combiner": "lr stable set"}, "transfer_loss"),
             (f"t19_transfer_by_combiner{sfx}.csv", {"combiner": "lr stable set"}, "transfer_loss"), "H7ref"),
            ("H8", "fetched fraction under VoI (10:1, central costs) minus the swept band's; reference 0",
             (f"t20_{dataset}_voi_test.csv", {"cost_ratio": "10:1", "central": True}, "fetched_fraction"),
             (f"t20_{dataset}{sfx}_voi_test.csv", {"cost_ratio": "10:1", "central": True}, "fetched_fraction"), "H8ref"),
            ("H9", "share of the R1 gain recovered by prior_em on the native-rate target; reference 2/3",
             ("t21_prior_shift.csv", {"method": "prior_em", "target": "phreshphish-native"}, "f1"),
             (f"t21_prior_shift{sfx}.csv", {"method": "prior_em", "target": "phreshphish-native"}, "f1"), 2 / 3),
        ]
        for h, metric, dm, tw, ref in spec:
            vd, nd = val(*dm); vt, nt = val(*tw)
            note = "; ".join(x for x in (nd, nt) if x)
            if h == "H3":
                qd_d, _ = val(f"t07_{dataset}_S1_combiners.csv", {"combiner": "q_direct alone"}, "test_f1_at_fit")
                qd_t, _ = val(f"t09_{dataset}_S1_combiners.csv", {"combiner": "q_direct alone", "model": m}, "test_f1_at_fit")
                vd, vt, ref = vd - qd_d, vt - qd_t, 0.0
            elif ref == "H7ref":
                ad, _ = val("t19_transfer_by_combiner.csv", {"combiner": "lr all indicators"}, "transfer_loss")
                at, _ = val(f"t19_transfer_by_combiner{sfx}.csv", {"combiner": "lr all indicators"}, "transfer_loss")
                vd, vt, ref = vd - ad, vt - at, 0.0
            elif ref == "H8ref":
                bd, _ = val(f"t11b_{dataset}_lr_S0-S1-S2_test.csv", {}, "stop_S0")
                bt, _ = val(f"t11b_{dataset}_lr_S0-S1-S2_test{sfx}.csv", {}, "stop_S0")
                vd, vt, ref = vd - (1 - bd), vt - (1 - bt), 0.0
            elif h == "H9":
                def share(sfx_):
                    r0, _ = val(f"t21_prior_shift{sfx_}.csv", {"method": "R0", "target": "phreshphish-native"}, "f1")
                    r1, _ = val(f"t21_prior_shift{sfx_}.csv", {"method": "R1", "target": "phreshphish-native"}, "f1")
                    pe, _ = val(f"t21_prior_shift{sfx_}.csv", {"method": "prior_em", "target": "phreshphish-native"}, "f1")
                    return (pe - r0) / (r1 - r0) if r1 != r0 else np.nan
                vd, vt = share(""), share(sfx)
            same = "not available" if (np.isnan(vd) or np.isnan(vt) or ref is None or (isinstance(ref, float) and np.isnan(ref))) \
                else ("yes" if (vd >= ref) == (vt >= ref) else "no")
            rows.append({"hypothesis": h, "metric": metric, "decision_model_value": vd, "twin_value": vt, "same_direction": same,
                         "twin_model": m, "reference": ref if not isinstance(ref, str) else np.nan, "note": note})
    out = pd.DataFrame(rows)
    out.to_csv(R / f"t09_{dataset}_parity.csv", index=False)
    print(out[["hypothesis", "twin_model", "decision_model_value", "twin_value", "same_direction", "note"]].round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.add_argument("--variant", default="S1")
    ap.add_argument("--models", default=None, help="comma list; default config twin_models")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "vllm", "fake"])
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--parity", action="store_true", help="write the per-hypothesis parity table and exit")
    args = ap.parse_args()
    cfg = load_config(args.config)
    models = [m.strip() for m in (args.models.split(",") if args.models else cfg.get("twin_models", ["gemma4:12b", "qwen3.5:9b"]))]
    if args.parity:
        parity(cfg, args.dataset, models); return
    if args.backend == "fake" and not is_bootstrap(args.dataset):
        raise SystemExit("[t09] --backend fake is allowed on bootstrap / fixture datasets only")
    R = cfg.results_root / "dryrun" if args.backend == "fake" else cfg.results_root
    R.mkdir(parents=True, exist_ok=True)
    bank = load_bank(cfg.bank_path)
    qs = direct_question(bank) if args.variant == "S0" else bank_questions(bank, args.variant)
    seed, thr = cfg["SEED"], cfg.get("operating_threshold", 0.5)
    rate = float(cfg.get("gpu_usd_per_hour", 0.99)) / 3600
    timing = TimingLog(cfg.timing_root / f"t09_{args.dataset}_{args.variant}{'_fake' if args.backend == 'fake' else ''}.csv")
    dm_test = cfg.results_root / f"t07_{args.dataset}_{args.variant}_test_table.csv"
    comb_rows, summ_rows, pq_rows = [], [], []
    for model in models:
        twin = TwinClient(model, args.backend, cfg.raw_responses_root, args.base_url or cfg.get("twin_base_url", "http://localhost:11434"))
        tabs = {}
        for split in ("train", "test"):
            sites, source = sites_for(cfg, args.dataset, split, limit=args.limit, reason="t09")
            args.split = split; banner(f"t09 {model}", args, sites, source)
            t0 = time.time()
            df = answer_split(cfg, twin, sites, args.dataset, args.variant, qs, split, source, timing, model)
            df.to_csv(R / f"t09_{args.dataset}_{args.variant}_{mtag(model)}_{split}_table.csv", index=False)
            tabs[split] = df[df.ok].reset_index(drop=True)
            print(f"[t09] {model} {split}: {len(df)} sites, {time.time() - t0:.1f} s wall, {twin.gpu_seconds:.1f} s generation so far")
        tr, te = tabs["train"], tabs["test"]
        if not len(tr) or not len(te):
            print(f"[t09] {model}: empty tables; skipped"); continue
        allq = ["q_direct"] + [c for c in question_columns(tr) if c in te.columns]
        Xtr, ytr = tr[allq].astype(float).fillna(0.5), tr.label.values.astype(int)
        Xte, yte = te[allq].astype(float).fillna(0.5), te.label.values.astype(int)
        scores_tr, scores_te = tr[["site_id", "label", "split", "split_source"] + allq].copy(), te[["site_id", "label", "split", "split_source"] + allq].copy()
        if len(allq) > 1:
            for name, factory in COMBINERS.items():
                fac = (lambda f=factory: f(seed=seed)) if name == "catboost" else factory
                oof = cv_scores(fac, Xtr, ytr, seed=seed)
                t_fit = best_f1_threshold(ytr, oof)[0]
                _, p = fit_apply(fac, Xtr, ytr, Xte)
                d_fit, d_05 = detection(yte, p, t_fit), detection(yte, p, thr)
                comb_rows.append({"combiner": name, "features": f"q_direct + {len(allq) - 1} indicators (twin P(yes))", "cv_auroc_train": roc_auc_score(ytr, oof),
                                  "threshold_fit_on_train": t_fit, "test_f1_at_fit": d_fit["f1"], "test_precision_at_fit": d_fit["precision"],
                                  "test_recall_at_fit": d_fit["recall"], "test_f1_at_0.5": d_05["f1"], "test_auroc": d_05["auroc"], "test_auprc": d_05["auprc"],
                                  "n_train": len(ytr), "n_test": len(yte), "split_source": te.split_source.iloc[0], "model": model})
                scores_te[f"p_{name}"] = p; scores_tr[f"p_{name}"] = oof
        t_fit = best_f1_threshold(ytr, Xtr.q_direct)[0]
        d = detection(yte, Xte.q_direct, t_fit)
        comb_rows.append({"combiner": "q_direct alone", "features": "q_direct (twin P(yes))", "cv_auroc_train": roc_auc_score(ytr, Xtr.q_direct) if len(set(ytr)) == 2 else np.nan,
                          "threshold_fit_on_train": t_fit, "test_f1_at_fit": d["f1"], "test_precision_at_fit": d["precision"], "test_recall_at_fit": d["recall"],
                          "test_f1_at_0.5": detection(yte, Xte.q_direct, thr)["f1"], "test_auroc": d["auroc"], "test_auprc": d["auprc"],
                          "n_train": len(ytr), "n_test": len(yte), "split_source": te.split_source.iloc[0], "model": model})
        scores_te.to_csv(R / f"t09_{args.dataset}_{args.variant}_{mtag(model)}_test_scores.csv", index=False)
        scores_tr.to_csv(R / f"t09_{args.dataset}_{args.variant}_{mtag(model)}_train_scores.csv", index=False)
        # per-question AUROC, twin vs decision model
        dm = pd.read_csv(dm_test) if dm_test.exists() else None
        if dm is not None:
            dm = dm[dm.ok].set_index("site_id").reindex(te.site_id)
        for q in allq:
            a_t = roc_auc_score(yte, Xte[q]) if len(set(yte)) == 2 else np.nan
            a_d = roc_auc_score(yte, dm[q].astype(float)) if dm is not None and q in dm and dm[q].notna().all() else np.nan
            pq_rows.append({"question": q, "model": model, "auroc_twin": a_t, "auroc_decision_model": a_d, "n_test": len(yte)})
        pq = pd.DataFrame([r for r in pq_rows if r["model"] == model])
        rho = spearmanr(pq.auroc_twin, pq.auroc_decision_model).correlation if pq.auroc_decision_model.notna().all() and len(pq) > 2 else np.nan
        secs = pd.concat([tr.seconds, te.seconds])
        summ_rows.append({"model": model, "backend": args.backend, "n_sites": len(tr) + len(te), "questions_per_site": len(qs),
                          "seconds_per_site": float(secs.mean()), "seconds_per_site_p95": float(secs.quantile(0.95)),
                          "usd_per_1000": float(1000 * secs.mean() * rate), "gpu_usd_per_hour": rate * 3600,
                          "share_cached": float(pd.concat([tr.cached, te.cached]).mean()),
                          "share_rows_with_notes": float((pd.concat([tr.notes, te.notes]).astype(str) != "").mean()),
                          "test_f1_at_fit": next((r["test_f1_at_fit"] for r in comb_rows if r["model"] == model and r["combiner"] == "lr"), np.nan),
                          "spearman_per_question_auroc_vs_decision_model": rho, "split_source": te.split_source.iloc[0]})
        twin.close()
    for name, rows in (("combiners", comb_rows), ("summary", summ_rows), ("per_question", pq_rows)):
        d = pd.DataFrame(rows)
        d.to_csv(R / f"t09_{args.dataset}_{args.variant}_{name}.csv", index=False)
        if len(d):
            print(d.round(3).to_string(index=False))
    timing.close()


if __name__ == "__main__":
    main()
