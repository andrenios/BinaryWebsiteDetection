"""Figures (matplotlib defaults; PDF + PNG under figures/)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _save(fig, out_base: Path) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=150)
        paths.append(p)
    plt.close(fig)
    return paths


def frontier(points: pd.DataFrame, out_base: Path, y: str = "f1", x: str = "usd_per_1000",
             lat: str = "latency_s_per_site", label: str = "config", title: str = "") -> list[Path]:
    """Cost-detection frontier: USD per 1,000 sites on x (log), detection on y,
    one point per configuration, secondary x axis with latency per site."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(points[x], points[y])
    for _, r in points.iterrows():
        ax.annotate(str(r[label]), (r[x], r[y]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("USD per 1,000 sites")
    ax.set_ylabel(y.upper() if len(y) <= 5 else y)
    if lat in points and points[lat].notna().any():
        ax2 = ax.twiny()
        ax2.set_xscale("log")
        ax2.scatter(points[lat], points[y], alpha=0)
        ax2.set_xlabel("latency per site (s)")
    if title:
        ax.set_title(title)
    return _save(fig, out_base)


def reliability(table: pd.DataFrame, out_base: Path, title: str = "") -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.plot(table.p_mean, table.observed, marker="o")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed phishing rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)
    return _save(fig, out_base)


def curve(df: pd.DataFrame, x: str, ys: list[str], out_base: Path, xlabel: str = "", ylabel: str = "",
          title: str = "", logx: bool = False) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6, 4))
    for y in ys:
        ax.plot(df[x], df[y], marker="o", label=y)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel or x); ax.set_ylabel(ylabel)
    ax.legend()
    if title:
        ax.set_title(title)
    return _save(fig, out_base)


def heatmap(corr: pd.DataFrame, out_base: Path, title: str = "") -> list[Path]:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    if title:
        ax.set_title(title)
    return _save(fig, out_base)
