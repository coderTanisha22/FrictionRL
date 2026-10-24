"""Transaction sources: where the *content* of each transaction comes from.

A source only answers: "give me the next transaction for a genuine user / a fraudster".
How users and fraudsters *react* to the agent's actions lives in env.py,
because no dataset records those reactions (counterfactuals).

Two sources:
  * SyntheticSource  - parametric generator (default, no download needed)
  * IEEECISSource    - replays real transaction feature rows from Kaggle's IEEE-CIS
                       Fraud Detection data (see data/README.md)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .config import SimConfig


@dataclass
class Txn:
    log_amount_ratio: float   # log(amount / user's typical amount)
    amount: float             # rupees
    new_device: int
    new_payee: int
    odd_hour: int
    burst: int


class SyntheticSource:
    """Parametric generator. Genuine and fraud distributions overlap on purpose,
    so a single transaction is ambiguous and the agent must use session context."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def sample_user(self, rng: np.random.Generator) -> float:
        return rng.uniform(self.cfg.user_mu_low, self.cfg.user_mu_high)

    def genuine(self, rng, user_mu: float) -> Txn:
        c = self.cfg
        lr = rng.normal(0.0, c.genuine_sigma)
        return Txn(lr, float(np.exp(user_mu + lr)),
                   int(rng.random() < c.genuine_p_new_device),
                   int(rng.random() < c.genuine_p_new_payee),
                   int(rng.random() < c.genuine_p_odd_hour),
                   int(rng.random() < c.genuine_p_burst))

    def fraud(self, rng, user_mu: float) -> Txn:
        c = self.cfg
        lr = rng.normal(c.fraud_mu_shift, c.fraud_sigma)
        return Txn(lr, float(np.exp(user_mu + lr)),
                   int(rng.random() < c.fraud_p_new_device),
                   int(rng.random() < c.fraud_p_new_payee),
                   int(rng.random() < c.fraud_p_odd_hour),
                   int(rng.random() < c.fraud_p_burst))


class IEEECISSource(SyntheticSource):
    """Replays real per-transaction features from IEEE-CIS (Kaggle).

    Build the feature file once with:
        python -m frictionrl.data_ieee --raw data/ieee-fraud-detection --out data/ieee_features.npz
    The file holds two pools of rows [log_amount_ratio, new_device, new_payee, odd_hour, burst]:
    one from genuine transactions and one from fraud transactions. Each sampled transaction
    is a real row; session structure and reactions remain simulated.
    If the file also holds user_mu_<split> (per-customer mean log amount), each session's
    typical amount is drawn from real customers; older files fall back to the synthetic range.
    """

    def __init__(self, cfg: SimConfig, path: str = "data/ieee_features.npz", split: str = "train"):
        super().__init__(cfg)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. See data/README.md to build it.")
        d = np.load(path)
        self.gen = d[f"genuine_{split}"]
        self.fr = d[f"fraud_{split}"]
        key = f"user_mu_{split}"
        self.user_mu = d[key] if key in d.files and len(d[key]) else None

    def sample_user(self, rng: np.random.Generator) -> float:
        if self.user_mu is None:
            return super().sample_user(rng)
        return float(self.user_mu[rng.integers(len(self.user_mu))])

    def _row(self, rng, pool, user_mu) -> Txn:
        lr, nd, npy, oh, b = pool[rng.integers(len(pool))]
        return Txn(float(lr), float(np.exp(user_mu + lr)), int(nd), int(npy), int(oh), int(b))

    def genuine(self, rng, user_mu):
        return self._row(rng, self.gen, user_mu)

    def fraud(self, rng, user_mu):
        return self._row(rng, self.fr, user_mu)


def make_source(name: str, cfg: SimConfig, split: str = "train"):
    if name == "synthetic":
        return SyntheticSource(cfg)
    if name == "ieee":
        return IEEECISSource(cfg, split=split)
    raise ValueError(name)
