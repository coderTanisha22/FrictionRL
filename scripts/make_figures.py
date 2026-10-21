"""Build every figure + results table from results/ (run scripts/run_all.py first).

    python scripts/make_figures.py [--results results]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from frictionrl.env import N_DISCRETE_STATES, describe_state  # noqa: E402

# palette (validated categorical order) + neutrals
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY, INK, INK2, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2, "figure.dpi": 200, "savefig.bbox": "tight", "text.color": INK,
})
RS = 1000.0  # reward_scale -> rupees


def smooth(y, k):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    R = ap.parse_args().results
    F = os.path.join(R, "figures")
    os.makedirs(F, exist_ok=True)
    runs = pd.read_csv(f"{R}/all_runs.csv")
    logs = json.load(open(f"{R}/logs.json"))
    sweep = pd.read_csv(f"{R}/threshold_sweep.csv")
    agg = runs.drop(columns="seed").groupby(["agent", "family"], sort=False).agg(["mean", "std"])
    base_rule = agg.loc[("Rule-KB", "baseline"), ("net_rs_per_session", "mean")]
    base_thr = agg.loc[("Amount-Threshold", "baseline"), ("net_rs_per_session", "mean")]

    # ------------------------------------------------ Fig 1: learning curves (rupees per session)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    ax = axes[0]
    for i, algo in enumerate(["dqn", "ppo"]):
        cs = [logs["curves"][k] for k in logs["curves"] if k.startswith(algo)]
        grid = np.linspace(0, min(c["steps"][-1] for c in cs), 200)
        ys = np.array([np.interp(grid, smooth(c["steps"], 500), smooth(c["returns"], 500) * RS) for c in cs])
        m, s = ys.mean(0), ys.std(0)
        ax.plot(grid / 1e3, m, color=C[i], label=algo.upper())
        ax.fill_between(grid / 1e3, m - s, m + s, color=C[i], alpha=0.15, lw=0)
        if "val_steps" in cs[0]:
            vs = np.array(cs[0]["val_steps"]) / 1e3
            vm = np.mean([c["val_money"] for c in cs], axis=0)
            ax.plot(vs, vm, "o", ms=3.5, color=C[i], markeredgecolor="white", markeredgewidth=0.8,
                    label=f"{algo.upper()} greedy, validation")
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("return per session (Rs)")
    ax.set_title("Deep RL", fontsize=9, loc="left")
    ax = axes[1]
    for i, (key, lab) in enumerate([("egreedy", "Tabular Q (eps-greedy)"), ("explfn", "Tabular Q (expl. fn)")]):
        ls = [logs["qlogs"][k] for k in logs["qlogs"] if k.startswith(key)]
        ep = np.array([r["episode"] for r in ls[0]])
        y = np.array([[r["avg_return"] for r in l] for l in ls]) * RS
        ax.plot(ep / 1e3, y.mean(0), color=C[2 + i], label=lab)
    ll = list(logs["lin_logs"].values())
    ep = np.array([r["episode"] for r in ll[0]])
    y = np.array([[r["avg_return"] for r in l] for l in ll]) * RS
    ax.plot(ep / 1e3, y.mean(0), color=C[6], label="Linear approx. Q")
    ax.set_xlabel("training episodes (thousands)")
    ax.set_title("Tabular / linear (training returns include exploration)", fontsize=9, loc="left")
    for a in axes:
        a.axhline(base_rule, color=GRAY, ls="--", lw=1.2)
        a.text(a.get_xlim()[1], base_rule, " Rule-KB", color=INK2, va="center", fontsize=7)
        a.legend(fontsize=7, loc="lower right")
    axes[0].set_ylim(-1500, None)
    fig.savefig(f"{F}/fig1_learning_curves.png")
    plt.close(fig)

    # ------------------------------------------------ Fig 2: net value per session, all agents
    d = agg["net_rs_per_session"].reset_index().sort_values("mean")
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    cols = [GRAY if f == "baseline" else C[0] for f in d.family]
    ax.barh(d.agent, d["mean"], xerr=d["std"].fillna(0), color=cols, height=0.6,
            error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.axvline(0, color=INK2, lw=0.8)
    for y, (v, sd) in enumerate(zip(d["mean"], d["std"].fillna(0))):
        off = sd + 50
        ax.text(v + (off if v >= 0 else -off), y, f"{v:,.0f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7, color=INK2)
    ax.set_xlabel("net value per session (Rs, higher is better)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(d["mean"].min() - 900, d["mean"].max() + 400)
    fig.savefig(f"{F}/fig2_net_value.png")
    plt.close(fig)

    # ------------------------------------------------ Fig 3: Pareto - fraud loss vs churn
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(sweep.fraud_loss_rs_per_session, sweep.churn_pct, "-o", color=GRAY, ms=3, lw=1.2,
            label="Amount threshold (swept)")
    show = ["Rule-KB", "Model-based VI", "Q-learning (expl. fn)", "Linear approx. Q", "DQN", "PPO"]
    marks = ["s", "o", "o", "o", "o", "o"]
    for i, name in enumerate(show):
        sub = agg.xs(name, level="agent")
        x, yv = sub[("fraud_loss_rs_per_session", "mean")].iloc[0], sub[("churn_pct", "mean")].iloc[0]
        col = GRAY if name == "Rule-KB" else C[i - 1]
        ax.scatter(x, yv, s=40, marker=marks[i], color=col, edgecolor="white", linewidth=1.5, zorder=3,
                   label=name)
    ax.set_xlabel("fraud loss per session (Rs)")
    ax.set_ylabel("genuine users churned (%)")
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 35)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_title("Fraud loss vs. churn (better = bottom-left)", fontsize=8, loc="left")
    fig.savefig(f"{F}/fig3_pareto.png")
    plt.close(fig)

    # ------------------------------------------------ Fig 4: Q-learning vs value iteration
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.3), sharex=True)
    for i, (key, lab) in enumerate([("markov_check", "on learned MDP (exactly Markov)"),
                                    ("egreedy_0", "on real simulator (hidden fraudster)")]):
        l = logs["qlogs"][key]
        ep = np.array([r["episode"] for r in l]) / 1e3
        axes[0].plot(ep, np.array([r["mean_abs_q_err"] for r in l]) * RS, color=C[i], label=lab)
        axes[1].plot(ep, np.array([r["policy_agreement"] for r in l]) * 100, color=C[i], label=lab)
    axes[0].set_ylabel("mean |Q - Q*| (Rs)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("same action as VI (%)")
    axes[1].set_ylim(80, 100)
    axes[1].set_xlabel("Q-learning episodes (thousands)")
    axes[0].legend(fontsize=6.5)
    fig.savefig(f"{F}/fig4_q_vs_vi.png")
    plt.close(fig)

    # ------------------------------------------------ Fig 5: learned policy maps (VI vs Q-learning)
    pl = np.load(f"{R}/planning.npz")
    Qe = np.load(f"{R}/Q_explfn.npy")
    Nvisit = pl["N"][:-1].sum(1)
    acts = ["A", "O", "B"]
    acol = [C[0], C[3], C[7]]
    rows_spec = [(n, f, p) for n in range(3) for f in range(2) for p in range(2) if not (f and p)]
    cols_spec = [(a, fl) for a in range(3) for fl in range(3)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, pol, title in [(axes[0], pl["pi"][:-1], "Model-based value iteration"),
                           (axes[1], Qe.argmax(1), "Tabular Q-learning (expl. fn)")]:
        for r_i, (n, f, p) in enumerate(rows_spec):
            for c_i, (a, fl) in enumerate(cols_spec):
                # session not risky, not blocked
                idx = ((((((a * 3 + fl) * 3 + n) * 2 + f) * 2 + p) * 2 + 0) * 2 + 0)
                if Nvisit[idx] < 30:
                    ax.add_patch(plt.Rectangle((c_i, r_i), 1, 1, color="#f3f2ee"))
                    continue
                act = int(pol[idx])
                ax.add_patch(plt.Rectangle((c_i + 0.04, r_i + 0.04), 0.92, 0.92, color=acol[act]))
                ax.text(c_i + 0.5, r_i + 0.5, acts[act], ha="center", va="center", color="white", fontsize=7,
                        fontweight="bold")
        ax.set_xlim(0, 9); ax.set_ylim(0, len(rows_spec)); ax.invert_yaxis()
        ax.set_xticks(np.arange(9) + 0.5)
        ax.set_xticklabels([f"{['norm', 'high', 'v.high'][a]}\n{fl}f" for a, fl in cols_spec], fontsize=6)
        ax.set_yticks(np.arange(len(rows_spec)) + 0.5)
        ax.set_yticklabels([f"OTPs={n}{' failed' if f else ''}{' passed' if p else ''}" for n, f, p in rows_spec],
                           fontsize=6.5)
        ax.grid(False)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("amount bucket / # risk flags")
    fig.text(0.5, -0.1, "A = approve   O = OTP step-up   B = block   (grey = state rarely visited)",
             ha="center", fontsize=7, color=INK2)
    fig.savefig(f"{F}/fig5_policy_maps.png")
    plt.close(fig)

    # ------------------------------------------------ Fig 6: behaviour around the takeover
    best = logs["best_deep"]
    tr = pd.read_csv(f"{R}/behaviour_trace_{best}.csv")
    tk = tr[tr.takeover == 1].dropna(subset=["rel"])
    tk = tk[(tk.rel >= -4) & (tk.rel <= 6)]
    share = tk.groupby("rel").action.value_counts(normalize=True).unstack(fill_value=0) * 100
    gen = tr[tr.takeover == 0].action.value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for a, lab, col in [(1, "OTP", C[3]), (2, "Block", C[7])]:
        if a in share:
            ax.plot(share.index, share[a], "-o", ms=3, color=col, label=f"{lab} (taken-over sessions)")
        ax.axhline(gen.get(a, 0), color=col, ls=":", lw=1.2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("transactions relative to takeover (0 = first fraudulent)")
    ax.set_ylabel("% of decisions")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_title(f"{best.upper()} policy behaviour (dotted = genuine sessions)", fontsize=8, loc="left")
    fig.savefig(f"{F}/fig6_behaviour.png")
    plt.close(fig)
    print("figures ->", F)


if __name__ == "__main__":
    main()
