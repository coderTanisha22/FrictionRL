"""Red team: a learned fraudster that chooses its own attack, trained against a frozen defender.

Per fraudulent transaction the fraudster picks one of 7 actions:
    0..5 -> amount level = amount_levels[a // 2] x the user's usual amount, stealth = a % 2
    6    -> cash out and leave (the account goes back to the owner)
Stealth (spoofed device + a payee the owner has paid before) raises fewer risk flags, but the fraudster
keeps only part of the money and the owner may notice. The bank's loss is unchanged (amount + chargeback).
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import SimConfig
from .simulator import Txn

N_ATTACK_ACTIONS = 7
LEAVE = 6
EXTERNAL = "external"   # FrictionEnv(attacker=EXTERNAL): decisions arrive later through env.apply_attack()
FRAUD_OBS_DIM = 8       # see FrictionEnv.fraud_obs


@dataclass
class AttackConfig:
    amount_levels: tuple = (0.5, 1.0, 2.5)   # x the user's usual amount
    amount_sigma: float = 0.2                # log-normal jitter around the chosen level
    stealth_received_frac: float = 0.6       # stealth costs laundering: the fraudster keeps 60%
    stealth_notice_prob: float = 0.10        # per stealth txn, the owner notices -> attack ends
    stealth_p_new_device: float = 0.10
    stealth_p_new_payee: float = 0.15
    stealth_p_odd_hour: float = 0.20
    stealth_p_burst: float = 0.20


def decode_action(a: int) -> tuple[int, int]:
    """Attack action -> (amount level index, stealth flag). Not defined for LEAVE."""
    a = int(a)
    if not 0 <= a < LEAVE:
        raise ValueError(f"action {a} has no amount/stealth (LEAVE is {LEAVE})")
    return a // 2, a % 2


def sample_attack_txn(rng: np.random.Generator, cfg: SimConfig, acfg: AttackConfig,
                      user_mu: float, action: int) -> Txn:
    """Draw the fraudulent transaction the attacker chose with `action` (0..5)."""
    level, stealth = decode_action(action)
    lr = float(np.log(acfg.amount_levels[level]) + rng.normal(0.0, acfg.amount_sigma))
    if stealth:
        p = (acfg.stealth_p_new_device, acfg.stealth_p_new_payee, acfg.stealth_p_odd_hour, acfg.stealth_p_burst)
    else:   # the scripted fraudster's flag probabilities
        p = (cfg.fraud_p_new_device, cfg.fraud_p_new_payee, cfg.fraud_p_odd_hour, cfg.fraud_p_burst)
    nd, npy, oh, b = (int(rng.random() < q) for q in p)
    return Txn(lr, float(np.exp(user_mu + lr)), nd, npy, oh, b)


class FraudsterEnv(gym.Env):
    """The fraudster's side of a session against a FIXED defender (callable defender_obs -> action).

    One episode = one session with a guaranteed takeover. The defender handles the owner's payments on its own;
    the fraudster acts once per fraudulent payment and its reward is the money it receives / 1000.
    The episode ends when the attack ends (cash out, OTP lockout, owner notices, frozen) or the session ends.
    """
    metadata = {"render_modes": []}

    def __init__(self, defender):
        super().__init__()
        from .env import FrictionEnv   # local import: env.py imports this module
        self.defender = defender
        self.env = FrictionEnv(SimConfig(p_takeover=1.0), mode="continuous", attacker=EXTERNAL)
        self.action_space = spaces.Discrete(N_ATTACK_ACTIONS)
        self.observation_space = spaces.Box(0.0, 10.0, shape=(FRAUD_OBS_DIM,), dtype=np.float32)

    def _defender_step(self):
        """One defender decision on the current payment. Returns (money the fraudster received, session over)."""
        _, _, term, trunc, info = self.env.step(int(self.defender(self.env._obs())))
        return info["fraud_received"], term or trunc

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        while True:
            # if the owner leaves before the takeover there is no fraud decision to make: draw another session
            self.env.reset(seed=int(self.np_random.integers(2**31)))
            over = False
            while not self.env.awaiting_attack and not over:
                _, over = self._defender_step()
            if self.env.awaiting_attack:
                return self.env.fraud_obs(), {}

    def step(self, action):
        self.env.apply_attack(int(action))
        received, over = 0.0, False
        if self.env.fraud_active:          # LEAVE ends the attack at once; otherwise the defender reacts
            while True:
                r, over = self._defender_step()
                received += r
                if over or self.env.awaiting_attack or not self.env.fraud_active:
                    break
        attack_over = not self.env.fraud_active or self.env.stats["frozen"] == 1
        terminated = attack_over or (over and not self.env.awaiting_attack and self.env.t < self.env.cfg.horizon)
        truncated = over and not terminated
        info = {"fraud_received": received, "stats": dict(self.env.stats)}
        return self.env.fraud_obs(), received / 1000.0, bool(terminated), bool(truncated), info
