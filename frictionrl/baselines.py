"""Non-learning baselines. Each maps a continuous observation to an action.

Observation indices (config.OBS_NAMES): 0 amount ratio, 1 new device, 2 new payee, 3 odd hour, 4 burst,
5 session risk rate, 6 progress, 7 #OTPs/3, 8 OTP failed, 9 OTP passed, 10 #blocks/2.
"""
import numpy as np

from .config import APPROVE, BLOCK, OTP


def always_approve(obs):
    return APPROVE


def always_otp(obs):
    return OTP


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return int(self.rng.integers(3))


class AmountThreshold:
    """OTP above an amount threshold, block after a failed OTP. tau is a log ratio: tau=1.0 -> 2.7x usual."""

    def __init__(self, tau: float = 1.0):
        self.tau = tau

    def __call__(self, obs):
        if obs[8] > 0.5:
            return BLOCK
        return OTP if obs[0] > self.tau else APPROVE


def rule_kb(obs):
    """Hand-written propositional rule base (Lec 16): IF conditions THEN action.
      R1: OtpFailed                              => BLOCK
      R2: OtpPassed and not VeryHighAmount       => APPROVE   (user already verified)
      R3: HighAmount and (NewDevice or NewPayee) => OTP
      R4: NewDevice and NewPayee and Burst       => OTP
      R5: otherwise                              => APPROVE
    """
    lr, nd, npy, burst = obs[0], obs[1] > 0.5, obs[2] > 0.5, obs[4] > 0.5
    if obs[8] > 0.5:
        return BLOCK
    if obs[9] > 0.5 and lr < 1.5:
        return APPROVE
    if lr > 0.8 and (nd or npy):
        return OTP
    if nd and npy and burst:
        return OTP
    return APPROVE


BASELINES = {
    "Always-Approve": always_approve,
    "Always-OTP": always_otp,
    "Random": RandomPolicy(0),
    "Amount-Threshold": AmountThreshold(1.0),
    "Rule-KB": rule_kb,
}
