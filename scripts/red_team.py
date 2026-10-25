"""Red team: train a PPO fraudster against each FROZEN defender and compare it with the scripted fraudster
and a random attacker on the same test sessions (all with a guaranteed takeover).

Question: how much more fraud loss does a learned fraudster cause than the scripted one?
Exploitability = fraud loss Rs/session (learned) - fraud loss Rs/session (scripted).

    python scripts/red_team.py --quick     # smoke test (20k steps, 500 eval sessions)
    python scripts/red_team.py             # full run (150k steps, 3,000 eval sessions)

No defender is retrained. Nothing in results/ is written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frictionrl.attacker import LEAVE, N_ATTACK_ACTIONS, AttackConfig  # noqa: E402
from frictionrl.config import SimConfig                                 # noqa: E402
from frictionrl.evaluate import run_policy, summarize                   # noqa: E402

DEFENDER_LABELS = {"rule": "Rule-KB", "threshold": "Amount-Threshold", "dqn": "DQN", "ppo": "PPO"}
ATTACKERS = ["scripted", "random", "learned"]
FRAUDSTER_PPO = dict(learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10, gamma=0.99,
                     gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]))


def make_defender(name: str, model_seed: int):
    """Frozen defender policies (obs -> action). DQN/PPO are loaded from results/models/ and never trained."""
    from frictionrl.baselines import AmountThreshold, rule_kb
    if name == "rule":
        return rule_kb
    if name == "threshold":
        return AmountThreshold(1.0)
    import torch
    from stable_baselines3 import DQN, PPO
    from frictionrl.train_sb3 import SB3Policy
    torch.set_num_threads(1)
    cls = DQN if name == "dqn" else PPO
    return SB3Policy(cls.load(f"results/models/{name}_seed{model_seed}_ltv3000"))


def evaluate_attack(defender, attacker, n_eps: int, p_takeover: float) -> dict:
    """Play n_eps test sessions (same seeds for every attacker). attacker=None -> the scripted fraudster."""
    cfg = SimConfig(p_takeover=p_takeover)
    df = run_policy(defender, n_eps, cfg=cfg, attacker=attacker)
    if attacker is None:   # the scripted fraudster keeps everything it steals
        df["fraud_received"] = df["fraud_loss"] - cfg.chargeback_fee * df["fraud_completed"]
        actions = [[] for _ in range(len(df))]
    else:
        actions = df.pop("attack_actions").tolist()
    res = summarize(df)
    res["fraudster_received_rs_per_session"] = df["fraud_received"].mean()
    res.update(attack_mix(actions))
    return res


def attack_mix(actions: list[list[int]]) -> dict:
    """Attack style: share of chosen amounts, % stealth, attack length, % of attacks ended by cashing out."""
    flat = [a for acts in actions for a in acts if a != LEAVE]
    if not flat:
        return dict(pct_small=np.nan, pct_medium=np.nan, pct_large=np.nan, pct_stealth=np.nan,
                    mean_attack_len=np.nan, pct_cash_out=np.nan)
    flat = np.array(flat)
    attacked = [acts for acts in actions if acts]
    return dict(pct_small=100 * np.mean(flat // 2 == 0), pct_medium=100 * np.mean(flat // 2 == 1),
                pct_large=100 * np.mean(flat // 2 == 2), pct_stealth=100 * np.mean(flat % 2 == 1),
                mean_attack_len=float(np.mean([sum(a != LEAVE for a in acts) for acts in attacked])),
                pct_cash_out=100 * float(np.mean([acts[-1] == LEAVE for acts in attacked])))


def baseline_job(args):
    """Scripted and random attackers against one defender (do not depend on the attacker seed)."""
    name, model_seed, eval_eps = args
    defender = make_defender(name, model_seed)
    rng = np.random.default_rng(12345)
    random_attacker = lambda fobs: int(rng.integers(N_ATTACK_ACTIONS))   # noqa: E731
    out = {}
    for att, fn in [("scripted", None), ("random", random_attacker)]:
        out[att] = evaluate_attack(defender, fn, eval_eps, p_takeover=1.0)
        out[att]["net_rs_per_session_p015"] = evaluate_attack(defender, fn, eval_eps, 0.15)["net_rs_per_session"]
    return name, out


def learned_job(args):
    """Train one fraudster (PPO) against a frozen defender, save it, evaluate it."""
    name, model_seed, att_seed, steps, eval_eps, out_dir = args
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from frictionrl.attacker import FraudsterEnv
    torch.set_num_threads(1)
    defender = make_defender(name, model_seed)
    env = Monitor(FraudsterEnv(defender))
    env.reset(seed=att_seed)
    model = PPO("MlpPolicy", env, seed=att_seed, verbose=0, **FRAUDSTER_PPO)
    t0 = time.time()
    model.learn(total_timesteps=steps)
    os.makedirs(f"{out_dir}/models", exist_ok=True)
    model.save(f"{out_dir}/models/fraudster_vs_{name}_seed{att_seed}")
    curve = dict(steps=np.cumsum(env.get_episode_lengths()).tolist(),
                 returns_rs=(np.array(env.get_episode_rewards()) * 1000).tolist())
    attacker = lambda fobs: int(model.predict(fobs, deterministic=True)[0])   # noqa: E731
    res = evaluate_attack(defender, attacker, eval_eps, p_takeover=1.0)
    res["net_rs_per_session_p015"] = evaluate_attack(defender, attacker, eval_eps, 0.15)["net_rs_per_session"]
    return name, att_seed, res, curve, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defenders", default="rule,threshold,dqn,ppo")
    ap.add_argument("--model-seed", type=int, default=0, help="which saved DQN/PPO defender to load")
    ap.add_argument("--attacker-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=150_000, help="fraudster training steps per run")
    ap.add_argument("--eval-eps", type=int, default=3_000)
    ap.add_argument("--out", default="results_adversary")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="smoke test: 20k steps, 500 eval sessions")
    a = ap.parse_args()
    if a.quick:
        a.steps, a.eval_eps = 20_000, 500
    defenders = [d.strip() for d in a.defenders.split(",") if d.strip()]
    unknown = set(defenders) - set(DEFENDER_LABELS)
    if unknown:
        sys.exit(f"unknown defender(s) {sorted(unknown)}; choose from {sorted(DEFENDER_LABELS)}")
    if os.path.abspath(a.out) in {os.path.abspath("results"), os.path.abspath("results_ieee")}:
        sys.exit("--out must not be results/ or results_ieee/")
    os.makedirs(f"{a.out}/figures", exist_ok=True)
    t0 = time.time()
    print(f"red team: defenders={defenders} attacker seeds={a.attacker_seeds} steps={a.steps:,} "
          f"eval sessions={a.eval_eps:,} -> {a.out}/", flush=True)

    rows, curves, warnings = [], {}, []
    with ProcessPoolExecutor(a.workers) as ex:
        base = list(ex.map(baseline_job, [(d, a.model_seed, a.eval_eps) for d in defenders]))
        learned = list(ex.map(learned_job, [(d, a.model_seed, s, a.steps, a.eval_eps, a.out)
                                            for d in defenders for s in a.attacker_seeds]))
    for name, out in base:
        for att, res in out.items():
            rows.append(dict(defender=DEFENDER_LABELS[name], attacker=att, attacker_seed=np.nan, **res))
    for name, seed, res, curve, secs in learned:
        rows.append(dict(defender=DEFENDER_LABELS[name], attacker="learned", attacker_seed=seed, **res))
        curves[f"{name}_{seed}"] = curve
        print(f"[{time.time() - t0:6.0f}s] trained fraudster vs {DEFENDER_LABELS[name]:16s} seed {seed} "
              f"({secs:.0f}s): fraud loss Rs {res['fraud_loss_rs_per_session']:8.0f}, "
              f"receives Rs {res['fraudster_received_rs_per_session']:8.0f}", flush=True)

    df = pd.DataFrame(rows)
    mix_cols = ["pct_small", "pct_medium", "pct_large", "pct_stealth", "mean_attack_len", "pct_cash_out"]
    df[["defender", "attacker", "attacker_seed"] + mix_cols].to_csv(f"{a.out}/attack_mix.csv", index=False)
    df.drop(columns=mix_cols).to_csv(f"{a.out}/exploit.csv", index=False)

    # per defender: mean/std over attacker seeds, exploitability, and a sanity check
    summary = []
    for name in defenders:
        lab = DEFENDER_LABELS[name]
        d = df[df.defender == lab]
        sc, rnd, le = (d[d.attacker == x] for x in ATTACKERS)
        row = dict(defender=lab)
        for col in ["fraud_loss_rs_per_session", "fraudster_received_rs_per_session", "fraud_txn_completed_pct",
                    "churn_pct", "net_rs_per_session_p015"] + mix_cols:
            row[f"scripted_{col}"] = sc[col].iloc[0]
            row[f"random_{col}"] = rnd[col].iloc[0]
            row[f"learned_{col}"] = le[col].mean()
            row[f"learned_{col}_std"] = le[col].std(ddof=0)
        row["exploitability"] = row["learned_fraud_loss_rs_per_session"] - row["scripted_fraud_loss_rs_per_session"]
        row["exploitability_std"] = row["learned_fraud_loss_rs_per_session_std"]
        row["learning_gain_vs_random"] = (row["learned_fraud_loss_rs_per_session"]
                                          - row["random_fraud_loss_rs_per_session"])
        summary.append(row)
        for _, r in le.iterrows():
            if r["fraudster_received_rs_per_session"] < rnd["fraudster_received_rs_per_session"].iloc[0]:
                msg = (f"WARNING: learned fraudster vs {lab} (seed {int(r['attacker_seed'])}) receives less than the "
                       f"random attacker (Rs {r['fraudster_received_rs_per_session']:.0f} < "
                       f"Rs {rnd['fraudster_received_rs_per_session'].iloc[0]:.0f}): training failed for this defender.")
                warnings.append(msg)
                print(msg, flush=True)
    summ = pd.DataFrame(summary)

    json.dump(dict(args=vars(a), attack_cfg=vars(AttackConfig()), fraudster_ppo=FRAUDSTER_PPO, curves=curves,
                   summary=summary, warnings=warnings, seconds=time.time() - t0),
              open(f"{a.out}/logs.json", "w"), default=float)
    write_table(summ, a, warnings)
    make_figures(df, summ, curves, defenders, a.out)
    print(open(f"{a.out}/results_table.md").read())
    print(f"done in {time.time() - t0:.0f}s -> {a.out}/")


def pm(m, s):
    return f"{m:,.0f}" + (f" ± {s:,.0f}" if np.isfinite(s) and s > 0 else "")


def write_table(summ: pd.DataFrame, a, warnings):
    n = len(a.attacker_seeds)
    lines = [
        f"Fraud loss per session on {a.eval_eps:,} test sessions with a guaranteed takeover "
        f"(learned fraudster: mean ± std over {n} attacker seed{'s' * (n > 1)}, {a.steps:,} training steps each).",
        "Exploitability = learned − scripted fraud loss. Learning gain = learned − random.", "",
        "| Defender | Scripted Rs | Random Rs | Learned Rs | Exploitability Rs | Learning gain Rs "
        "| Fraudster receives Rs (learned) | Net Rs/session at p=0.15: scripted → learned |",
        "|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        lines.append(
            f"| {r.defender} | {r.scripted_fraud_loss_rs_per_session:,.0f} | {r.random_fraud_loss_rs_per_session:,.0f} "
            f"| {pm(r.learned_fraud_loss_rs_per_session, r.learned_fraud_loss_rs_per_session_std)} "
            f"| {pm(r.exploitability, r.exploitability_std)} | {r.learning_gain_vs_random:+,.0f} "
            f"| {pm(r.learned_fraudster_received_rs_per_session, r.learned_fraudster_received_rs_per_session_std)} "
            f"| {r.scripted_net_rs_per_session_p015:,.0f} → {r.learned_net_rs_per_session_p015:,.0f} |")
    lines += ["", "Learned attack style (mean over attacker seeds):", "",
              "| Defender | Small % | Medium % | Large % | Stealth % | Fraud txns per attack | Attacks ended by cash-out % |",
              "|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        lines.append(f"| {r.defender} | {r.learned_pct_small:.0f} | {r.learned_pct_medium:.0f} | {r.learned_pct_large:.0f} "
                     f"| {r.learned_pct_stealth:.0f} | {r.learned_mean_attack_len:.1f} | {r.learned_pct_cash_out:.0f} |")
    if warnings:
        lines += ["", "**Warnings:**", ""] + [f"* {w}" for w in warnings]
    open(f"{a.out}/results_table.md", "w").write("\n".join(lines) + "\n")


def make_figures(df, summ, curves, defenders, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # one colour per attacker type in every figure; defenders get different colours
    INK2, GRID = "#52514e", "#e6e5e0"
    ATT_COL = {"scripted": "#8a8985", "random": "#eda100", "learned": "#e34948"}
    DEF_COL = {"Rule-KB": "#6250d6", "Amount-Threshold": "#e87ba4", "DQN": "#2a78d6", "PPO": "#1baf7a"}
    SHORT = {"Rule-KB": "Rule-KB", "Amount-Threshold": "Thresh.", "DQN": "DQN", "PPO": "PPO"}
    plt.rcParams.update({
        "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.8, "figure.dpi": 200,
        "savefig.bbox": "tight"})
    labs = [DEFENDER_LABELS[d] for d in defenders]

    # fraud loss per defender: scripted / random / learned
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x, w = np.arange(len(labs)), 0.26
    for i, att in enumerate(ATTACKERS):
        m = [summ.set_index("defender").loc[l, f"{att}_fraud_loss_rs_per_session"] for l in labs]
        err = ([summ.set_index("defender").loc[l, "learned_fraud_loss_rs_per_session_std"] for l in labs]
               if att == "learned" else None)
        ax.bar(x + (i - 1) * w, m, w, yerr=err, color=ATT_COL[att], label=f"{att} fraudster",
               error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("Amount-", "Amount-\n") for l in labs])
    ax.set_ylabel("fraud loss per session (Rs)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_title("Takeover sessions: fraud loss by attacker", fontsize=8, loc="left")
    fig.savefig(f"{out}/figures/fig_adv_exploit.png")
    plt.close(fig)

    # learned attack style: amount shares (stacked) and % stealth, side by side
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.3), gridspec_kw=dict(width_ratios=[1.3, 1]))
    s = summ.set_index("defender").loc[labs]
    bottom = np.zeros(len(labs))
    for col, lab, colr in [("learned_pct_small", "small (0.5×)", "#9cc3ef"), ("learned_pct_medium", "medium (1×)", "#2a78d6"),
                           ("learned_pct_large", "large (2.5×)", "#173f73")]:
        v = s[col].fillna(0).to_numpy()
        axes[0].bar(labs, v, bottom=bottom, color=colr, label=lab, width=0.6)
        bottom += v
    axes[0].set_ylabel("% of fraud transactions")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Amount chosen", fontsize=8, loc="left")
    axes[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    axes[1].bar(labs, s["learned_pct_stealth"].fillna(0), color=ATT_COL["learned"], width=0.6)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Stealth used (%)", fontsize=8, loc="left")
    for axx in axes:
        axx.grid(axis="x", visible=False)
        axx.set_xticks(range(len(labs)))
        axx.set_xticklabels([SHORT[l] for l in labs], fontsize=6.5)   # short names: no overlapping labels
    fig.savefig(f"{out}/figures/fig_adv_attack_mix.png")
    plt.close(fig)

    # fraudster training return vs steps, one line per defender (mean over attacker seeds)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for d in defenders:
        cs = [c for k, c in curves.items() if k.rsplit("_", 1)[0] == d and len(c["steps"]) > 1]
        if not cs:
            continue
        grid = np.linspace(0, min(c["steps"][-1] for c in cs), 100)
        k = max(1, min(len(c["returns_rs"]) for c in cs) // 20)
        ys = [np.interp(grid, np.convolve(c["steps"], np.ones(k) / k, "valid"),
                        np.convolve(c["returns_rs"], np.ones(k) / k, "valid")) for c in cs]
        ax.plot(grid / 1e3, np.mean(ys, 0), color=DEF_COL[DEFENDER_LABELS[d]], label=DEFENDER_LABELS[d])
    ax.set_xlabel("fraudster training steps (thousands)")
    ax.set_ylabel("received per attack (Rs)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4)
    ax.set_title("Learned fraudster: training return", fontsize=8, loc="left")
    fig.savefig(f"{out}/figures/fig_adv_learning.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
