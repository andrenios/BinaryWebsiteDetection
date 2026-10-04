#!/usr/bin/env python3
"""T08 (d): ModernBERT-base fine-tuned on the serialised S1 state, 3 seeds. GPU.

  pip install torch transformers>=4.48 accelerate
  python scripts/t08d_encoder.py --dataset putra --seeds 2107,2108,2109 --epochs 3

UNTESTED in session 1 (no GPU, transformers not installed). Records training
wall time, inference ms per site on the available device, and GPU USD at
config gpu_usd_per_hour. Outputs results/t08d_<dataset>_encoder.csv and
results/t08d_<dataset>_test_scores.csv.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _bootstrap import load_config, add_common_args, add_split_args, sites_for, banner, REPO
sys.path.insert(0, str(REPO / "src"))
from jev.run import load_state  # noqa: E402
from eval.metrics import detection, best_f1_threshold  # noqa: E402
from baselines.classical import serialise_state  # noqa: E402


def texts(cfg, dataset, split, variant, limit):
    sites, source = sites_for(cfg, dataset, split, limit=limit, reason="t08d")
    X, y, ids = [], [], []
    for s in sites:
        st = load_state(cfg, dataset, variant, s.id)
        if st is None:
            continue
        X.append(serialise_state(st)); y.append(s.label); ids.append(s.id)
    return X, np.array(y), ids, source


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap); add_split_args(ap)
    ap.add_argument("--variant", default="S1")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--seeds", default="2107,2108,2109")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()
    cfg = load_config(args.config)
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    Xtr, ytr, _, _ = texts(cfg, args.dataset, "train", args.variant, args.limit)
    Xte, yte, ids_te, source = texts(cfg, args.dataset, "test", args.variant, args.limit)
    args.split = "train/test"; banner("t08d", args, list(range(len(ytr) + len(yte))), source)
    tok = AutoTokenizer.from_pretrained(args.model)

    def enc(X):
        e = tok(X, truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    itr, mtr = enc(Xtr); ite, mte = enc(Xte)
    rows, scores = [], pd.DataFrame({"site_id": ids_te, "label": yte})
    for seed in [int(s) for s in args.seeds.split(",")]:
        torch.manual_seed(seed); np.random.seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        dl = DataLoader(TensorDataset(itr, mtr, torch.tensor(ytr)), batch_size=args.batch, shuffle=True)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * len(dl) * args.epochs), len(dl) * args.epochs)
        t0 = time.perf_counter(); model.train()
        for _ in range(args.epochs):
            for a, m, y in dl:
                out = model(input_ids=a.to(dev), attention_mask=m.to(dev), labels=y.to(dev))
                out.loss.backward(); opt.step(); sched.step(); opt.zero_grad()
        train_s = time.perf_counter() - t0
        model.eval(); probs = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(0, len(ite), args.batch):
                lg = model(input_ids=ite[i:i + args.batch].to(dev), attention_mask=mte[i:i + args.batch].to(dev)).logits
                probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
        infer_s = time.perf_counter() - t0
        p = np.concatenate(probs)
        # threshold: fitted on train predictions
        with torch.no_grad():
            ptr = np.concatenate([torch.softmax(model(input_ids=itr[i:i + args.batch].to(dev), attention_mask=mtr[i:i + args.batch].to(dev)).logits, -1)[:, 1].cpu().numpy()
                                  for i in range(0, len(itr), args.batch)])
        t_fit, _ = best_f1_threshold(ytr, ptr)
        d = detection(yte, p, t_fit)
        gpu_h = train_s / 3600
        rows.append({"model": args.model, "seed": seed, "device": dev, "epochs": args.epochs, "max_length": args.max_length,
                     "n_train": len(ytr), "n_test": len(yte), "test_f1_at_fit": d["f1"], "test_auroc": d["auroc"],
                     "test_auprc": d["auprc"], "threshold_fit_on_train": t_fit, "train_seconds": train_s,
                     "infer_ms_per_site": 1000 * infer_s / len(yte), "train_usd_at_gpu_rate": gpu_h * cfg.get("gpu_usd_per_hour", 0.99),
                     "infer_usd_per_1000_sites": (infer_s / len(yte) * 1000 / 3600) * cfg.get("gpu_usd_per_hour", 0.99),
                     "split_source": source})
        scores[f"p_seed{seed}"] = p
        print(rows[-1])
    res = pd.DataFrame(rows)
    res.to_csv(cfg.results_root / f"t08d_{args.dataset}_encoder.csv", index=False)
    scores.to_csv(cfg.results_root / f"t08d_{args.dataset}_test_scores.csv", index=False)
    print(res[["seed", "test_f1_at_fit", "test_auroc", "train_seconds", "infer_ms_per_site"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
