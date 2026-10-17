"""Evaluate any policy (callable obs -> action) on held-out sessions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SimConfig
from .env import FrictionEnv

TEST_SEED_OFFSET = 900_000_000   # test sessions never overlap training seeds
VAL_SEED_OFFSET = 500_000_000    # validation sessions (model selection) overlap neither


def run_policy(policy, n_episodes=5_000, cfg: SimConfig | None = None, source=None,
               seed=0, log_actions=False, seed_offset=TEST_SEED_OFFSET):
    env = FrictionEnv(cfg, mode="continuous", source=source, split="test")
    rows, trace = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + seed * 100_000 + ep)
        done = False
        while not done:
            a = int(policy(obs))
            if log_actions:
                rel = None if env.takeover_step is None else env.t - env.takeover_step
                trace.append(dict(ep=ep, t=env.t, action=a, fraud=int(env.fraud_active),
                                  takeover=int(env.takeover_step is not None), rel=rel))
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
        rows.append(dict(env.stats))
    df = pd.DataFrame(rows)
    return (df, pd.DataFrame(trace)) if log_actions else df


def summarize(df: pd.DataFrame) -> dict:
    g = df["genuine_txns"].sum()
    f = df["fraud_txns"].sum()
    return {
        "net_rs_per_session": df["money"].mean(),
        "fraud_loss_rs_per_session": df["fraud_loss"].mean(),
        "fraud_txn_completed_pct": 100 * df["fraud_completed"].sum() / max(f, 1),
        "churn_pct": 100 * df["churned"].mean(),
        "false_block_pct": 100 * df["false_blocks"].sum() / max(g, 1),
        "otp_per_session": df["otp"].mean(),
        "genuine_value_rs_per_session": df["genuine_value"].mean(),
    }


def evaluate(policy, **kw) -> dict:
    return summarize(run_policy(policy, **kw))
