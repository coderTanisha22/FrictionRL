"""Non-learning baselines. Each policy maps a continuous observation vector -> action.

Observation index reminder (see config.OBS_NAMES):
 0 log_amount_ratio, 1 new_device, 2 new_payee, 3 odd_hour, 4 burst,
 5 risk_rate_session, 6 progress, 7 n_otp_issued/3, 8 otp_failed_seen,
 9 otp_passed_seen, 10 n_blocks/2
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
    """What many systems do today: step up any transaction above an amount threshold,
    block after a failed OTP. tau is in log-ratio units (tau=1.0 -> 2.7x the user's usual amount)."""

    def __init__(self, tau: float = 1.0):
        self.tau = tau

    def __call__(self, obs):
        if obs[8] > 0.5:
            return BLOCK
        return OTP if obs[0] > self.tau else APPROVE


def rule_kb(obs):
    """Hand-written propositional rule base (Lec 16 style: IF conditions THEN action).
      R1: OtpFailed                                   => BLOCK
      R2: OtpPassed  and  not VeryHighAmount          => APPROVE   (user already verified)
      R3: HighAmount and (NewDevice or NewPayee)      => OTP
      R4: NewDevice and NewPayee and Burst            => OTP
      R5: otherwise                                   => APPROVE
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
