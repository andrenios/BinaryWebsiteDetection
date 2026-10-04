#!/usr/bin/env python3
"""T15: adversarial steering on the test split's phishing pages + N benign.

  python scripts/t15_adversarial.py --dataset putra --benign 500 --questions direct,bank

Variants A1–A4 (src/adversarial/inject.py) plus the mitigation arm (text in
`untrusted_text` with a warning in the instructions). For each arm: noul shift
vs the clean state, flip rate at the operating threshold, AUROC.
Outputs results/t15_<dataset>_<questions>_{table,summary}.csv.

v2.3 arm A5 (adaptive black-box attack, src/adversarial/adaptive.py):

  python scripts/t15_adversarial.py --dataset putra --arms A5 --a5-targets direct,lr [--a5-targets twin:gemma4:12b]

For --a5-pages (200) test-split phishing pages the attacker edits `visible_text`
with sentences from benign TRAIN-split pages of the same language, keeps an edit
when the returned probability falls, and stops at the budget (max of
--a5-budgets) or at the first flip below the train-fitted operating threshold.
Targets: q_direct, the lr bank score (T07 combiner refitted on the train table),
and the twin (T09 tables + TwinClient). Each target has its own spend cap
(--a5-max-usd, config a5_max_usd_per_target); the Jev MAX_USD cap applies too.
Outputs results/t15_<dataset>_A5_summary.csv (Table 12b) and _A5_pages.csv.
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
from jev.run import run_questions, load_state, save_table  # noqa: E402
from questions.bank import load_bank, direct_question, bank_questions  # noqa: E402
from summarise.state import build_s1  # noqa: E402
from adversarial.inject import VARIANTS, a4_padding_html, mitigation_state, mitigation_questions  # noqa: E402
from combine.combiners import question_columns, make_lr, cv_scores, fit_apply  # noqa: E402
from eval.metrics import best_f1_threshold  # noqa: E402
from eval.timing import TimingLog  # noqa: E402
from jev.client import SpendCapReached  # noqa: E402
from adversarial.adaptive import benign_sentence_pool, attack_page, summarise  # noqa: E402


class ArmBudgetReached(Exception):
    pass


def run_a5(cfg, args, sites_all, src, bank):
    """Arm A5 for every target in --a5-targets. Returns (summary rows, page rows)."""
    budgets = sorted(int(b) for b in args.a5_budgets.split(","))
    pos = [s for s in sites_all if s.label == 1][:args.a5_pages]
    tr_sites, _ = sites_for(cfg, args.dataset, "train", reason="t15 A5 benign sentence pool / thresholds")
    benign_states = {s.id: load_state(cfg, args.dataset, args.variant, s.id) for s in tr_sites if s.label == 0}
    benign_states = {k: v for k, v in benign_states.items() if v}
    pools = benign_sentence_pool(benign_states, {s.id: s.language for s in tr_sites})
    qd, qb = direct_question(bank), bank_questions(bank, args.variant)
    qcols = [q for q in qb if q != "q_direct"]
    summ, pages = [], []
    for target in args.a5_targets.split(","):
        client = client_from_config(cfg, f"t15_{args.dataset}_A5_{target.replace(':', '-')}", use_cache=not args.no_cache)
        spent = {"usd": 0.0}

        def charge(res_cost):
            spent["usd"] += float(res_cost or 0.0)
            if spent["usd"] > args.a5_max_usd:
                raise ArmBudgetReached(f"A5 {target}: arm spend {spent['usd']:.4f} USD > {args.a5_max_usd}")

        if target == "direct":
            trd = run_questions(cfg, client, tr_sites, args.dataset, args.variant, qd, f"t06_{args.dataset}_train_{args.variant}_direct", "train", src)
            trd = trd[trd.ok]
            thr = best_f1_threshold(trd.label, trd.q_direct)[0] if len(trd) else cfg.get("operating_threshold", 0.5)

            def oracle(state):
                r = client.ask(state, qd, {"arm": "A5", "target": target})
                if not r.ok:
                    raise ArmBudgetReached(f"HTTP failure {r.http_status}: {r.error}")
                if not r.cached:
                    charge(r.cost_usd)
                return float(r.noul("q_direct")), {"cost_usd": 0.0 if r.cached else r.cost_usd, "nouls": {"q_direct": r.noul("q_direct")}}
            ind_keys = None
        elif target == "lr":
            trb = run_questions(cfg, client, tr_sites, args.dataset, args.variant, qb, f"t07_{args.dataset}_{args.variant}_train", "train", src)
            trb = trb[trb.ok]
            cols = ["q_direct"] + [c for c in question_columns(trb) if c in qcols]
            y = trb.label.values.astype(int)
            thr = best_f1_threshold(y, cv_scores(make_lr, trb[cols].astype(float), y, seed=cfg["SEED"]))[0]
            model = make_lr().fit(trb[cols].astype(float), y)

            def oracle(state):
                r = client.ask(state, qb, {"arm": "A5", "target": target})
                if not r.ok:
                    raise ArmBudgetReached(f"HTTP failure {r.http_status}: {r.error}")
                if not r.cached:
                    charge(r.cost_usd)
                nouls = {q: r.noul(q) for q in cols}
                X = pd.DataFrame([[nouls[c] for c in cols]], columns=cols).astype(float)
                return float(model.predict_proba(X)[0, 1]), {"cost_usd": 0.0 if r.cached else r.cost_usd, "nouls": nouls}
            ind_keys = cols
        elif target.startswith("twin:"):
            from baselines.twin import TwinClient
            tm = target.split(":", 1)[1]
            tag = tm.replace(":", "_").replace("/", "_")
            tp = cfg.results_root / f"t09_{args.dataset}_{args.variant}_{tag}_train_table.csv"
            if not tp.exists():
                print(f"[t15] A5 twin target {tm}: {tp} missing (run T09 first); skipped"); continue
            trb = pd.read_csv(tp)
            cols = ["q_direct"] + [c for c in question_columns(trb) if c in qcols]
            y = trb.label.values.astype(int)
            thr = best_f1_threshold(y, cv_scores(make_lr, trb[cols].astype(float), y, seed=cfg["SEED"]))[0]
            model = make_lr().fit(trb[cols].astype(float), y)
            twin = TwinClient(tm, args.a5_twin_backend, cfg.raw_responses_root, cfg.get("twin_base_url", "http://localhost:11434"))
            rate = float(cfg.get("gpu_usd_per_hour", 0.99)) / 3600

            def oracle(state):
                nouls, cost = {}, 0.0
                for q in cols:
                    r = twin.ask(state, qb[q]["instructions"])
                    nouls[q] = r.p_yes if r.p_yes is not None else 0.5
                    cost += 0.0 if r.cached else r.seconds * rate
                charge(cost)
                X = pd.DataFrame([[nouls[c] for c in cols]], columns=cols).astype(float)
                return float(model.predict_proba(X)[0, 1]), {"cost_usd": cost, "nouls": nouls}
            ind_keys = cols
        else:
            print(f"[t15] unknown A5 target {target}"); continue

        results = []
        for s in pos:
            st = load_state(cfg, args.dataset, args.variant, s.id)
            if st is None:
                continue
            pool = pools.get((s.language or "unk").lower()) or pools.get("", [])
            r = attack_page(s.id, st, oracle, pool, max(budgets), thr, seed=cfg["SEED"],
                            stop_cap_exc=(SpendCapReached, ArmBudgetReached))
            results.append(r)
            pages.append({"site_id": s.id, "target": target, "language": s.language, "p0": r.p0, "queries": r.queries,
                          "first_flip_query": r.first_flip_query, "cost_usd": r.cost_usd, "stopped_reason": r.stopped_reason,
                          **{f"p_at_{b}": r.p_at(b) for b in budgets}, "threshold": thr})
            if r.stopped_reason.startswith("spend cap"):
                print(f"[t15] A5 {target}: stopped ({r.stopped_reason})"); break
        tab = summarise(results, budgets, thr, {"direct": "q_direct", "lr": "lr_bank"}.get(target, target), indicator_keys=ind_keys)
        tab["threshold"] = thr; tab["n_pages_attempted"] = len(pos); tab["arm_usd"] = spent["usd"]; tab["split_source"] = src
        summ.append(tab)
        print(f"[t15] A5 {target}: {len(results)} pages, arm spend {spent['usd']:.4f} USD\n" + tab.round(3).to_string(index=False))
        client.close()
    return summ, pages


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.add_argument("--variant", default="S1")
    ap.add_argument("--benign", type=int, default=500)
    ap.add_argument("--questions", default="direct", help="direct | bank")
    ap.add_argument("--arms", default="clean,A1,A2,A3,A4,A1m,A2m")
    ap.add_argument("--threshold", type=float, default=None, help="operating threshold (default: config)")
    ap.add_argument("--a5-pages", type=int, default=200)
    ap.add_argument("--a5-budgets", default="50,200,1000")
    ap.add_argument("--a5-targets", default="direct,lr", help="comma list of direct | lr | twin:<model>")
    ap.add_argument("--a5-max-usd", type=float, default=None, help="per-target spend cap for A5 (config a5_max_usd_per_target)")
    ap.add_argument("--a5-twin-backend", default="ollama")
    args = ap.parse_args()
    cfg = load_config(args.config)
    thr = args.threshold if args.threshold is not None else cfg.get("operating_threshold", 0.5)
    all_sites, src = sites_for(cfg, args.dataset, args.split, reason="t15")
    if args.a5_max_usd is None:
        args.a5_max_usd = float(cfg.get("a5_max_usd_per_target", 1.0))
    arms = [a for a in args.arms.split(",") if a]
    if "A5" in arms:
        bank = load_bank(cfg.bank_path)
        summ, pages = run_a5(cfg, args, all_sites, src, bank)
        if summ:
            save_table(cfg, pd.concat(summ, ignore_index=True), f"t15_{args.dataset}_A5_summary")
            save_table(cfg, pd.DataFrame(pages), f"t15_{args.dataset}_A5_pages")
        arms = [a for a in arms if a != "A5"]
        if not arms:
            return
        args.arms = ",".join(arms)
    pos = [s for s in all_sites if s.label == 1]
    neg = [s for s in all_sites if s.label == 0][:args.benign]
    sites = sorted(pos + neg, key=lambda s: s.id)
    if args.limit:
        sites = sites[:args.limit]
    tag = f"t15_{args.dataset}_{args.questions}"
    client = client_from_config(cfg, tag, use_cache=not args.no_cache)
    banner("t15", args, sites, src, client)
    bank = load_bank(cfg.bank_path)
    qs = direct_question(bank) if args.questions == "direct" else bank_questions(bank, args.variant)
    timing = TimingLog(cfg.timing_root / f"{tag}.csv")

    clean = {s.id: load_state(cfg, args.dataset, args.variant, s.id) for s in sites}
    clean = {k: v for k, v in clean.items() if v is not None}
    tables = {}
    for arm in args.arms.split(","):
        mitig = arm.endswith("m")
        base = arm[:-1] if mitig else arm
        if base == "clean":
            states = clean
        elif base in VARIANTS:
            states = {k: VARIANTS[base](v) for k, v in clean.items()}
        elif base == "A4":
            states = {}
            for s in sites:
                if s.id in clean and s.html_path.exists():
                    st, _ = build_s1(a4_padding_html(s.html()), s.url, cfg["s1_budget"], cfg["token_counter_model"])
                    states[s.id] = st
        else:
            print(f"[t15] unknown arm {arm}"); continue
        q = qs
        if mitig:
            states = {k: mitigation_state(v) for k, v in states.items()}
            q = mitigation_questions(qs)
        df = run_questions(cfg, client, [s for s in sites if s.id in states], args.dataset, args.variant, q, f"{tag}_{arm}",
                           args.split, src, states=states, workers=args.workers, timing=timing, extra_meta={"arm": arm})
        df = df[df.ok]
        tables[arm] = df
        save_table(cfg, df, f"{tag}_{arm}")
        print(f"[t15] {arm}: {len(df)} ok, mean q_direct phish {df.q_direct[df.label == 1].mean():.3f} benign {df.q_direct[df.label == 0].mean():.3f}")

    c = tables["clean"].set_index("site_id")
    cols = ["q_direct"] + (question_columns(c) if args.questions == "bank" else [])
    rows = []
    for arm, df in tables.items():
        d = df.set_index("site_id").reindex(c.index)
        for col in cols:
            if col not in d:
                continue
            a, b = c[col].astype(float), d[col].astype(float)
            m = a.notna() & b.notna()
            y = c.label[m].astype(int)
            rows.append({"arm": arm, "question": col, "n": int(m.sum()),
                         "shift_mean_phish": float((b - a)[m & (c.label == 1)].mean()),
                         "shift_mean_benign": float((b - a)[m & (c.label == 0)].mean()),
                         "flip_rate_phish_to_benign": float(((a >= thr) & (b < thr))[m & (c.label == 1)].mean()),
                         "flip_rate_benign_to_phish": float(((a < thr) & (b >= thr))[m & (c.label == 0)].mean()),
                         "auroc_clean": roc_auc_score(y, a[m]) if y.nunique() == 2 else np.nan,
                         "auroc_arm": roc_auc_score(y, b[m]) if y.nunique() == 2 else np.nan,
                         "tokens_mean": float(d.input_tokens[m].mean()), "threshold": thr, "split_source": src})
    out = pd.DataFrame(rows)
    save_table(cfg, out, f"{tag}_summary")
    print(out[out.question == "q_direct"].round(3).to_string(index=False))
    timing.close(); client.close()


if __name__ == "__main__":
    main()
