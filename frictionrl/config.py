"""All tunable constants for FrictionRL in one place.

Money is in rupees. Rewards given to the agent are divided by REWARD_SCALE
so that neural-network value targets stay around O(1).
"""
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    # ---- episode ----
    horizon: int = 20                 # max transactions per session (finite horizon, Lec 26)
    p_takeover: float = 0.15          # P(account gets taken over during this session)
    takeover_min_step: int = 2        # takeover step t0 ~ Uniform[min, max]
    takeover_max_step: int = 14

    # ---- genuine user transactions ----
    user_mu_low: float = 6.0          # user's typical log-amount ~ U(low, high)  (~Rs 400 .. Rs 3000)
    user_mu_high: float = 8.0
    genuine_sigma: float = 0.6        # spread of a genuine user's amounts
    genuine_p_new_device: float = 0.08
    genuine_p_new_payee: float = 0.20
    genuine_p_odd_hour: float = 0.10
    genuine_p_burst: float = 0.15     # quick back-to-back transaction

    # ---- fraudster transactions (deliberately overlapping with genuine) ----
    fraud_mu_shift: float = 0.9       # fraud amounts are ~e^0.9 = 2.5x the user's typical amount
    fraud_sigma: float = 0.8
    fraud_p_new_device: float = 0.65
    fraud_p_new_payee: float = 0.75
    fraud_p_odd_hour: float = 0.40
    fraud_p_burst: float = 0.60
    fraud_continue: float = 0.70      # after each fraud txn the fraudster keeps going w.p. this (else cashes out)

    # ---- reactions to the agent's actions (the part no dataset contains) ----
    genuine_otp_pass: float = 0.97    # genuine user completes OTP
    fraud_otp_pass: float = 0.15      # fraudster beats OTP (SIM swap / phished OTP)
    churn_otp_base: float = 0.01      # P(genuine user quits after an OTP) = base + slope * (#previous OTPs)
    churn_otp_slope: float = 0.04
    churn_block_base: float = 0.20    # P(genuine user quits after a false block) = base + slope * (#previous blocks)
    churn_block_slope: float = 0.20
    fraud_quit_after_otp_fail: float = 0.70
    fraud_retry_after_block: float = 0.50   # adaptive fraudster: retries with a smaller amount
    fraud_retry_shrink: float = 0.5         # ... amount multiplied by this
    max_blocks_before_freeze: int = 2       # 2nd block freezes the account (terminal)

    # ---- money (Rs) ----
    fee_rate: float = 0.02            # value of a completed genuine transaction = fee_rate * amount
    chargeback_fee: float = 200.0     # extra cost on top of the stolen amount
    otp_cost: float = 2.0             # SMS + small friction
    false_block_cost: float = 50.0    # support call, lost sale
    ltv: float = 3000.0               # customer lifetime value lost on churn
    freeze_genuine_cost: float = 1500.0  # freezing a genuine user's account

    reward_scale: float = 1000.0


ACTIONS = ["APPROVE", "OTP", "BLOCK"]
APPROVE, OTP, BLOCK = 0, 1, 2

OBS_NAMES = [
    "log_amount_ratio",   # log(amount / user's typical amount)
    "new_device",
    "new_payee",
    "odd_hour",
    "burst",              # back-to-back with previous transaction
    "risk_rate_session",  # fraction of session txns so far with new_device or new_payee
    "progress",           # t / H
    "n_otp_issued",       # / 3, clipped
    "otp_failed_seen",    # any OTP failed this session
    "otp_passed_seen",    # any OTP passed since last failure
    "n_blocks",           # / 2
]
OBS_DIM = len(OBS_NAMES)
