"""Provisional recreation of the SVLM paper's split (select_sample_gtihub.py):
per class, stratified on language x 3 quantile buckets of HTML token length,
30% test, seed 42. Used only until the co-author's train.csv/test.csv arrive;
every result on it is marked provisional (WORKORDER.md 1.2).

The paper's cleaning (removal of all-black / all-white screenshots and of 46
sites without files) must be applied before calling `stratified_split`; see
`screenshot_is_blank`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def screenshot_is_blank(path: str | Path) -> bool:
    from PIL import Image
    with Image.open(path) as im:
        arr = np.array(im.convert("L"))
    return bool(np.all(arr == 0) or np.all(arr == 255))


def html_token_length(html: str, encoding=None) -> int:
    import tiktoken
    enc = encoding or tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(str(html)))


def stratified_split(df: pd.DataFrame, test_frac: float = 0.30, n_length_buckets: int = 3,
                     seed: int = 42, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verbatim logic of the paper's function. df columns: id, language, html_length."""
    df = df.copy()
    df["_len_bucket"] = pd.qcut(df["html_length"], q=n_length_buckets,
                                labels=False, duplicates="drop")
    test_parts = []
    for _, group in df.groupby(["language", "_len_bucket"]):
        test_parts.append(group.sample(frac=test_frac, random_state=seed))
    test_df = pd.concat(test_parts)
    train_df = df.drop(test_df.index)
    train_df = train_df.drop(columns="_len_bucket")
    test_df = test_df.drop(columns="_len_bucket")
    if verbose:
        print(f"train: {len(train_df):,}  test: {len(test_df):,}  "
              f"(test = {len(test_df) / len(df):.1%})")
    return train_df, test_df


def provisional_split(meta: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """meta columns: _id,label,language,html_length. Returns meta + 'split'
    ('train'/'test'), computed per class as in the paper."""
    parts = []
    for label, g in meta.groupby("label"):
        d = g.rename(columns={"_id": "id"})[["id", "language", "html_length"]]
        tr, te = stratified_split(d, seed=seed)
        parts.append(pd.DataFrame({"_id": tr["id"], "split": "train"}))
        parts.append(pd.DataFrame({"_id": te["id"], "split": "test"}))
    s = pd.concat(parts, ignore_index=True)
    return meta.merge(s, on="_id", how="left")
