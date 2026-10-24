"""Sanity-check the IEEE-CIS feature file and compare it with the synthetic source.

    python scripts/check_ieee.py [--data data/ieee_features.npz] [--out results_ieee]

Prints rows per pool, fraud share and per-feature means (genuine vs fraud), then saves
<out>/figures/fig_data_compare.png: flag rates (grouped bars) and log_amount_ratio histograms,
synthetic vs real, for genuine and fraud transactions.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from frictionrl.config import SimConfig  # noqa: E402
from frictionrl.simulator import SyntheticSource  # noqa: E402

COLS = ["log_amount_ratio", "new_device", "new_payee", "odd_hour", "burst"]
FLAGS = COLS[1:]
# same palette as make_figures.py
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK2, GRID = "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def synthetic_pool(kind: str, n: int, seed: int = 0) -> np.ndarray:
    src, rng = SyntheticSource(SimConfig()), np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        t = getattr(src, kind)(rng, src.sample_user(rng))
        rows.append([t.log_amount_ratio, t.new_device, t.new_payee, t.odd_hour, t.burst])
    return np.array(rows, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ieee_features.npz")
    ap.add_argument("--out", default="results_ieee")
    a = ap.parse_args()
    if not os.path.exists(a.data):
        sys.exit(f"{a.data} not found. Build it first, see data/README.md")
    d = np.load(a.data)

    print(f"pools in {a.data}:")
    for k in [k for k in d.files if d[k].ndim]:
        print(f"  {k:14s} {len(d[k]):>9,} rows")
    real = {kind: np.concatenate([d[f"{kind}_train"], d[f"{kind}_test"]]) for kind in ("genuine", "fraud")}
    n_g, n_f = len(real["genuine"]), len(real["fraud"])
    print(f"fraud share: {n_f / (n_g + n_f):.2%}  ({n_f:,} fraud / {n_g + n_f:,} transactions)")
    if "user_mu_train" in d.files:
        mu = np.concatenate([d["user_mu_train"], d["user_mu_test"]])
        rate = float(d["inr_per_usd"]) if "inr_per_usd" in d.files else 1.0
        print(f"customers' typical amount (Rs, at Rs {rate:g}/USD): median {np.exp(np.median(mu)):,.0f}, "
              f"middle 50% {np.exp(np.percentile(mu, 25)):,.0f} - {np.exp(np.percentile(mu, 75)):,.0f}")

    print(f"\n{'feature':18s} {'genuine':>9s} {'fraud':>9s}")
    for i, c in enumerate(COLS):
        print(f"{c:18s} {real['genuine'][:, i].mean():9.3f} {real['fraud'][:, i].mean():9.3f}")

    syn = {kind: synthetic_pool(kind, 20_000, seed=i) for i, kind in enumerate(("genuine", "fraud"))}
    series = [("Synthetic genuine", syn["genuine"]), ("Real genuine", real["genuine"]),
              ("Synthetic fraud", syn["fraud"]), ("Real fraud", real["fraud"])]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    x, w = np.arange(len(FLAGS)), 0.2
    for j, (lab, pool) in enumerate(series):
        ax.bar(x + (j - 1.5) * w, pool[:, 1:].mean(0) * 100, w, color=C[j], label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", " ") for f in FLAGS])
    ax.set_ylabel("% of transactions with flag")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.5)
    ax.set_title("Risk flags", fontsize=9, loc="left")

    ax = axes[1]
    bins = np.linspace(-3, 4, 50)
    for j, (lab, pool) in enumerate(series):
        ax.hist(pool[:, 0], bins=bins, density=True, histtype="step", lw=1.5, color=C[j], label=lab)
    ax.set_xlabel("log(amount / user's usual amount)")
    ax.set_ylabel("density")
    ax.set_title("Amount vs. usual", fontsize=9, loc="left")

    os.makedirs(f"{a.out}/figures", exist_ok=True)
    path = f"{a.out}/figures/fig_data_compare.png"
    fig.savefig(path)
    plt.close(fig)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
