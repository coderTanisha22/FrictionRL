"""Transaction sources: what each payment looks like (synthetic, or real IEEE-CIS rows).

How users and fraudsters REACT to the agent lives in env.py, because no dataset records that.
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
    """Genuine and fraud payments overlap on purpose, so one payment alone is not enough to decide."""

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
    """Replays real feature rows from Kaggle's IEEE-CIS data (built by data_ieee.py).

    Session structure and reactions stay simulated; each payment is a real row, and each session's
    typical amount comes from a real customer (user_mu_<split>), if the file has it.
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
        if self.user_mu is None:   # older feature files without per-customer amounts
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
