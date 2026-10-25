"""Run every experiment in the report and write the results folder (CSVs, JSON, models).

    python scripts/run_all.py            # full run (~35 min on a 2-core laptop)
    python scripts/run_all.py --quick    # smoke test (~2 min)
    python scripts/run_all.py --source ieee --out results_ieee   # replay real IEEE-CIS transactions
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

from frictionrl.baselines import BASELINES, AmountThreshold          # noqa: E402
from frictionrl.config import SimConfig                              # noqa: E402
from frictionrl.env import FrictionEnv                               # noqa: E402
from frictionrl.evaluate import run_policy, summarize                # noqa: E402
from frictionrl.planning import (LearnedMDPEnv, TabularPolicy, iterative_model_based,  # noqa: E402
                                 policy_iteration, value_iteration)
from frictionrl.qlearning import tabular_q_learning, train_linear_q  # noqa: E402


def sb3_job(args):
    """Train and evaluate one DQN/PPO run. All inputs come in the tuple so it works in a worker process."""
    algo, steps, seed, ltv, eval_eps, out, source = args
    from frictionrl.evaluate import evaluate
    from frictionrl.train_sb3 import SB3Policy, train
    cfg = SimConfig(ltv=ltv)
    model, curve = train(algo, steps, seed, cfg=cfg, source=source)
    os.makedirs(f"{out}/models", exist_ok=True)
    model.save(f"{out}/models/{algo}_seed{seed}_ltv{int(ltv)}")
    res = evaluate(SB3Policy(model), n_episodes=eval_eps, cfg=SimConfig(ltv=ltv), seed=seed, source=source)
    return algo, seed, ltv, res, {k: v.tolist() for k, v in curve.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results", help="output folder (default: results)")
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "ieee"],
                    help="where transaction content comes from")
    args = ap.parse_args()
    q, OUT, SRC = args.quick, args.out, args.source
    EVAL_EPS = 1_000 if q else 5_000
    SEEDS = [0] if q else [0, 1, 2]
    TAB_EPS = 30_000 if q else 200_000
    LIN_EPS = 10_000 if q else 60_000
    SB3_STEPS = 30_000 if q else 500_000
    MB_EPS = 15_000 if q else 80_000
    os.makedirs(OUT, exist_ok=True)
    rows, t0 = [], time.time()

    def add(name, family, res, seed=0):
        rows.append(dict(agent=name, family=family, seed=seed, **{k: float(v) for k, v in res.items()}))
        print(f"[{time.time() - t0:6.0f}s] {name:22s} seed={seed} net=Rs{res['net_rs_per_session']:8.1f}", flush=True)

    # 1. baselines, and the amount threshold swept over 13 values
    for name, pol in BASELINES.items():
        add(name, "baseline", summarize(run_policy(pol, EVAL_EPS, source=SRC)))
    sweep = []
    for tau in np.linspace(-0.5, 2.5, 13):
        r = summarize(run_policy(AmountThreshold(tau), EVAL_EPS, source=SRC))
        sweep.append(dict(tau=float(tau), **{k: float(v) for k, v in r.items()}))
    pd.DataFrame(sweep).to_csv(f"{OUT}/threshold_sweep.csv", index=False)

    # 2. model-based learning + value iteration + policy iteration
    denv = FrictionEnv(mode="discrete", source=SRC)
    T, R, N, start, V, Q_star, pi_vi, mb_log = iterative_model_based(denv, rounds=3, episodes_per_round=MB_EPS)
    _, _, _, vi_deltas = value_iteration(T, R)
    _, _, pi_pi, pi_hist = policy_iteration(T, R)
    planning = dict(model_rounds=mb_log, vi_iterations=len(vi_deltas), vi_deltas=vi_deltas[:2000],
                    pi_iterations=len(pi_hist), pi_changes=pi_hist,
                    vi_pi_same_policy=float((pi_vi == pi_pi).mean()))
    np.savez(f"{OUT}/planning.npz", V=V, Q=Q_star, pi=pi_vi, N=N)
    add("Model-based VI", "planning", summarize(run_policy(TabularPolicy(pi_vi), EVAL_EPS, source=SRC)))

    # 3. tabular Q-learning. First a check: on the learned (exactly Markov) MDP it must reach VI's Q*
    _, _, check_log = tabular_q_learning(LearnedMDPEnv(T, R, start), TAB_EPS, explore="egreedy", seed=0,
                                         q_star=Q_star, eval_every=max(TAB_EPS // 40, 1), alpha_min=0.0)
    qlogs = {"markov_check": check_log}
    # then model-free on the real simulator (never sees T or R)
    for explore, label in [("egreedy", "Q-learning (eps-greedy)"), ("explfn", "Q-learning (expl. fn)")]:
        for seed in SEEDS:
            Q, _, log = tabular_q_learning(FrictionEnv(mode="discrete", source=SRC), TAB_EPS, explore=explore,
                                           seed=seed, q_star=Q_star, eval_every=max(TAB_EPS // 40, 1))
            qlogs[f"{explore}_{seed}"] = log
            if seed == 0:
                np.save(f"{OUT}/Q_{explore}.npy", Q)
            add(label, "tabular", summarize(run_policy(TabularPolicy(Q.argmax(1)), EVAL_EPS, seed=seed,
                                                       source=SRC)), seed)

    # 4. linear approximate Q-learning
    lin_logs = {}
    for seed in SEEDS:
        agent, log = train_linear_q(FrictionEnv(mode="continuous", source=SRC), LIN_EPS, seed=seed,
                                    eval_every=max(LIN_EPS // 30, 1))
        lin_logs[seed] = log
        if seed == 0:
            json.dump(agent.weight_table(), open(f"{OUT}/linear_q_weights.json", "w"), indent=1, default=float)
        add("Linear approx. Q", "approx", summarize(run_policy(agent, EVAL_EPS, seed=seed, source=SRC)), seed)

    # 5. DQN and PPO in parallel, plus PPO retrained with other customer lifetime values
    jobs = [(a, SB3_STEPS, s, 3000.0, EVAL_EPS, OUT, SRC) for a in ("dqn", "ppo") for s in SEEDS]
    if not q:
        jobs += [("ppo", SB3_STEPS, 0, ltv, EVAL_EPS, OUT, SRC) for ltv in (1000.0, 8000.0)]
    curves, sens = {}, []
    with ProcessPoolExecutor(args.workers) as ex:
        for algo, seed, ltv, res, curve in ex.map(sb3_job, jobs):
            if ltv == 3000.0:
                add(algo.upper(), "deep", res, seed)
                curves[f"{algo}_{seed}"] = curve
            sens.append(dict(algo=algo, seed=seed, ltv=ltv, **{k: float(v) for k, v in res.items()}))
    pd.DataFrame(sens).to_csv(f"{OUT}/sensitivity_ltv.csv", index=False)

    # 6. every decision of the best deep agent, for the behaviour figure
    from stable_baselines3 import DQN, PPO
    from frictionrl.train_sb3 import SB3Policy
    df = pd.DataFrame(rows)
    deep = df[df.family == "deep"].groupby("agent").net_rs_per_session.mean()
    best = deep.idxmax().lower()
    model = (PPO if best == "ppo" else DQN).load(f"{OUT}/models/{best}_seed0_ltv3000")
    _, trace = run_policy(SB3Policy(model), EVAL_EPS, log_actions=True, source=SRC)
    trace.to_csv(f"{OUT}/behaviour_trace_{best}.csv", index=False)

    df.to_csv(f"{OUT}/all_runs.csv", index=False)
    summ = df.groupby(["agent", "family"], sort=False).agg(["mean", "std"]).drop(columns="seed")
    summ.to_csv(f"{OUT}/summary.csv")
    json.dump(dict(planning=planning, qlogs=qlogs, lin_logs=lin_logs, curves=curves, best_deep=best),
              open(f"{OUT}/logs.json", "w"), default=float)
    print(f"done in {time.time() - t0:.0f}s -> now run: python scripts/make_figures.py --results {OUT}")


if __name__ == "__main__":
    main()
