"""FrictionEnv: one user payment session as an MDP (Gymnasium API).

Episode  = one session of up to H transactions.
Hidden   = whether the account has been taken over by a fraudster (and when).
Actions  = 0 APPROVE, 1 OTP (step-up), 2 BLOCK.
Reward   = rupees / reward_scale:
             + fee * amount            completed genuine txn
             - (amount + chargeback)   completed fraud txn
             - otp_cost                every OTP
             - false_block_cost        blocking a genuine txn
             - LTV                     genuine user churns (absorbing terminal state)
             - freeze_genuine_cost     freezing a genuine user's account
Terminal = churn, account frozen (2nd block), or horizon reached (truncation).

mode="continuous" -> Box observation (11 features), for linear-Q / DQN / PPO.
mode="discrete"   -> Discrete(432) observation, for value iteration / tabular Q-learning.
"""
from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import APPROVE, BLOCK, OBS_DIM, OTP, SimConfig
from .simulator import make_source

N_DISCRETE_STATES = 3 * 3 * 3 * 2 * 2 * 2 * 2  # = 432


def discretize(obs: np.ndarray) -> int:
    """Map the continuous observation to one of 432 discrete states:
    amount bucket(3) x risk flags(3) x #OTPs(3) x otp_failed(2) x otp_passed(2)
    x session_risky(2) x blocked(2)."""
    lr = obs[0]
    amt = 0 if lr < 0.5 else (1 if lr < 1.5 else 2)
    flags = int(round(obs[1] + obs[2] + obs[3]))
    flags = min(flags, 2)
    n_otp = min(int(round(obs[7] * 3)), 2)
    failed = int(obs[8] > 0.5)
    passed = int(obs[9] > 0.5)
    blocked = int(obs[10] > 0.25)
    risky = int(obs[5] > 0.3)
    idx = amt
    idx = idx * 3 + flags
    idx = idx * 3 + n_otp
    idx = idx * 2 + failed
    idx = idx * 2 + passed
    idx = idx * 2 + risky
    idx = idx * 2 + blocked
    return idx


def describe_state(idx: int) -> dict:
    blocked = idx % 2; idx //= 2
    risky = idx % 2; idx //= 2
    passed = idx % 2; idx //= 2
    failed = idx % 2; idx //= 2
    n_otp = idx % 3; idx //= 3
    flags = idx % 3; idx //= 3
    amt = idx
    return dict(amount=["normal", "high", "very_high"][amt], flags=flags, n_otp=n_otp,
                otp_failed=failed, otp_passed=passed, session_risky=risky, blocked=blocked)


class FrictionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: SimConfig | None = None, mode: str = "continuous",
                 source: str | None = None, split: str = "train"):
        super().__init__()
        # transaction source: "synthetic" (default) or "ieee"; can be set with env var FRICTIONRL_SOURCE
        source = source or os.environ.get("FRICTIONRL_SOURCE", "synthetic")
        self.cfg = cfg or SimConfig()
        self.mode = mode
        self.src = make_source(source, self.cfg, split)
        self.action_space = spaces.Discrete(3)
        if mode == "continuous":
            self.observation_space = spaces.Box(-3.0, 4.0, shape=(OBS_DIM,), dtype=np.float32)
        else:
            self.observation_space = spaces.Discrete(N_DISCRETE_STATES)

    # ------------------------------------------------------------------ helpers
    def _new_txn(self):
        if self.fraud_active:
            if self.retry_amount_shrink is not None:
                txn = self.src.fraud(self.np_random, self.user_mu)
                txn.log_amount_ratio = float(self.last_fraud_lr + np.log(self.retry_amount_shrink))
                txn.amount = float(np.exp(self.user_mu + txn.log_amount_ratio))
                self.retry_amount_shrink = None
            else:
                txn = self.src.fraud(self.np_random, self.user_mu)
            self.last_fraud_lr = txn.log_amount_ratio
        else:
            txn = self.src.genuine(self.np_random, self.user_mu)
        self.txn = txn
        self.n_seen += 1
        self.n_risky += int(txn.new_device or txn.new_payee)

    def _obs_vec(self) -> np.ndarray:
        t = self.txn
        v = np.array([
            np.clip(t.log_amount_ratio, -3, 4),
            t.new_device, t.new_payee, t.odd_hour, t.burst,
            self.n_risky / max(self.n_seen, 1),
            self.t / self.cfg.horizon,
            min(self.n_otp, 3) / 3.0,
            float(self.otp_failed),
            float(self.otp_passed),
            min(self.n_blocks, 2) / 2.0,
        ], dtype=np.float32)
        return v

    def _obs(self):
        v = self._obs_vec()
        return v if self.mode == "continuous" else discretize(v)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.cfg
        rng = self.np_random
        self.user_mu = self.src.sample_user(rng)
        self.takeover_step = (int(rng.integers(c.takeover_min_step, c.takeover_max_step + 1))
                              if rng.random() < c.p_takeover else None)
        self.fraud_active = False
        self.fraud_done = False      # fraudster gave up / was locked out -> account secured
        self.retry_amount_shrink = None
        self.last_fraud_lr = 0.0
        self.t = 0
        self.n_otp = 0
        self.otp_failed = 0
        self.otp_passed = 0
        self.n_blocks = 0
        self.n_seen = 0
        self.n_risky = 0
        self.stats = dict(genuine_value=0.0, fraud_loss=0.0, otp=0, false_blocks=0, churned=0,
                          frozen=0, fraud_txns=0, fraud_stopped=0, fraud_completed=0, genuine_txns=0,
                          takeover=int(self.takeover_step is not None), money=0.0)
        self._maybe_takeover()
        self._new_txn()
        return self._obs(), {}

    def _maybe_takeover(self):
        if (self.takeover_step is not None and not self.fraud_active and not self.fraud_done
                and self.t >= self.takeover_step):
            self.fraud_active = True

    def _fraud_gives_up(self):
        self.fraud_active = False
        self.fraud_done = True
        self.retry_amount_shrink = None

    def step(self, action):
        c, rng, txn = self.cfg, self.np_random, self.txn
        is_fraud = self.fraud_active
        money = 0.0
        completed = False
        terminated = False

        if is_fraud:
            self.stats["fraud_txns"] += 1
        else:
            self.stats["genuine_txns"] += 1

        if action == APPROVE:
            completed = True

        elif action == OTP:
            money -= c.otp_cost
            prev = self.n_otp
            self.n_otp += 1
            self.stats["otp"] += 1
            if not is_fraud:
                if rng.random() < c.churn_otp_base + c.churn_otp_slope * prev:
                    money -= c.ltv
                    self.stats["churned"] = 1
                    terminated = True
                elif rng.random() < c.genuine_otp_pass:
                    completed = True
                    self.otp_passed = 1
            else:
                if rng.random() < c.fraud_otp_pass:
                    completed = True           # fraudster fooled the OTP
                    self.otp_passed = 1
                else:
                    self.otp_failed = 1
                    self.otp_passed = 0
                    self.stats["fraud_stopped"] += 1
                    if rng.random() < c.fraud_quit_after_otp_fail:
                        self._fraud_gives_up()

        elif action == BLOCK:
            prev = self.n_blocks
            self.n_blocks += 1
            if not is_fraud:
                money -= c.false_block_cost
                self.stats["false_blocks"] += 1
                if rng.random() < c.churn_block_base + c.churn_block_slope * prev:
                    money -= c.ltv
                    self.stats["churned"] = 1
                    terminated = True
            else:
                self.stats["fraud_stopped"] += 1
                if rng.random() < c.fraud_retry_after_block:
                    self.retry_amount_shrink = c.fraud_retry_shrink   # adaptive: probe with a smaller amount
                else:
                    self._fraud_gives_up()
            if not terminated and self.n_blocks >= c.max_blocks_before_freeze:
                terminated = True
                self.stats["frozen"] = 1
                if not is_fraud:
                    money -= c.freeze_genuine_cost

        if completed:
            if is_fraud:
                loss = txn.amount + c.chargeback_fee
                money -= loss
                self.stats["fraud_loss"] += loss
                self.stats["fraud_completed"] += 1
            else:
                v = c.fee_rate * txn.amount
                money += v
                self.stats["genuine_value"] += v

        if is_fraud and self.fraud_active and not terminated and rng.random() > c.fraud_continue:
            self._fraud_gives_up()   # fraudster cashes out and leaves; owner regains the account

        self.stats["money"] += money
        self.t += 1
        truncated = (not terminated) and self.t >= c.horizon
        if not (terminated or truncated):
            self._maybe_takeover()
            self._new_txn()
        info = {"stats": dict(self.stats), "was_fraud": is_fraud}
        return self._obs(), money / c.reward_scale, terminated, truncated, info
