"""Quick checks: python -m pytest -q"""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from frictionrl.baselines import BASELINES
from frictionrl.config import APPROVE, BLOCK, OTP, SimConfig
from frictionrl.env import N_DISCRETE_STATES, FrictionEnv, describe_state, discretize
from frictionrl.planning import LearnedMDPEnv, learn_model, policy_iteration, value_iteration


def test_gym_api():
    check_env(FrictionEnv(mode="continuous"), skip_render_check=True)
    check_env(FrictionEnv(mode="discrete"), skip_render_check=True)


def test_discretize_in_range():
    env = FrictionEnv(mode="continuous")
    for ep in range(200):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            assert 0 <= discretize(obs) < N_DISCRETE_STATES
            obs, _, term, trunc, _ = env.step(np.random.randint(3))
            done = term or trunc
    assert set(describe_state(0)) >= {"amount", "flags", "n_otp"}


def test_always_approve_loses_all_fraud():
    env = FrictionEnv(SimConfig(p_takeover=1.0))
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(APPROVE)
        done = term or trunc
    s = info["stats"]
    assert s["fraud_completed"] == s["fraud_txns"] > 0 and s["otp"] == 0


def test_second_block_freezes_account():
    env = FrictionEnv(SimConfig(churn_block_base=0.0, churn_block_slope=0.0))
    env.reset(seed=0)
    _, _, term1, _, _ = env.step(BLOCK)
    _, _, term2, _, info = env.step(BLOCK)
    assert not term1 and term2 and info["stats"]["frozen"] == 1


def test_otp_fatigue_increases_churn():
    cfg = SimConfig(p_takeover=0.0)
    churned = 0
    for ep in range(300):
        env = FrictionEnv(cfg)
        env.reset(seed=ep)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(OTP)
            done = term or trunc
        churned += info["stats"]["churned"]
    assert churned > 250          # OTP on every transaction drives most genuine users away


def test_vi_equals_pi():
    T, R, N, start = learn_model(FrictionEnv(mode="discrete"), n_episodes=3000)
    _, _, pi_vi, _ = value_iteration(T, R)
    _, _, pi_pi, _ = policy_iteration(T, R)
    assert (pi_vi == pi_pi).mean() > 0.99
    env = LearnedMDPEnv(T, R, start)
    s, _ = env.reset(seed=0)
    assert 0 <= s < N_DISCRETE_STATES


def test_baselines_run():
    obs, _ = FrictionEnv().reset(seed=0)
    for p in BASELINES.values():
        assert p(obs) in (0, 1, 2)


def test_ieee_source_from_fake_csv(tmp_path, monkeypatch):
    """Build a tiny fake IEEE-CIS dataset, run the feature builder, and play one episode on it."""
    import pytest
    from frictionrl import data_ieee

    rng = np.random.default_rng(0)
    n = 2_000
    raw = tmp_path / "raw"
    raw.mkdir()
    tx = {
        "TransactionID": np.arange(n),
        "isFraud": (rng.random(n) < 0.2).astype(int),
        "TransactionDT": np.sort(rng.integers(86_400, 86_400 * 30, n)),
        "TransactionAmt": rng.lognormal(4.0, 1.0, n).round(2),
        "card1": rng.integers(1_000, 1_200, n),
        "addr1": np.where(rng.random(n) < 0.1, np.nan, rng.integers(100, 110, n)),
        "D1": np.where(rng.random(n) < 0.1, np.nan, rng.integers(0, 5, n)),
        "R_emaildomain": rng.choice(["gmail.com", "yahoo.com", "outlook.com", None], n),
    }
    ident = {"TransactionID": np.arange(0, n, 2),
             "DeviceInfo": rng.choice(["Windows", "iOS Device", "MacOS", None], n // 2)}
    try:
        import pandas as pd
        pd.DataFrame(tx).to_csv(raw / "train_transaction.csv", index=False)
        pd.DataFrame(ident).to_csv(raw / "train_identity.csv", index=False)
        pools = data_ieee.build(str(raw))
    except (ImportError, ValueError, OSError, pd.errors.ParserError) as e:
        pytest.skip(f"pandas could not read the fake IEEE CSVs: {e}")

    for split in ("train", "test"):
        assert len(pools[f"genuine_{split}"]) and len(pools[f"fraud_{split}"])
        assert pools[f"genuine_{split}"].shape[1] == 5
        assert len(pools[f"user_mu_{split}"]) > 0
    # dollar amounts converted to rupees: fake amounts ~ lognormal(4) dollars -> ~4 + log(85) in rupees
    assert abs(np.median(pools["user_mu_train"]) - (4.0 + np.log(float(pools["inr_per_usd"])))) < 0.5
    (tmp_path / "data").mkdir()
    np.savez_compressed(tmp_path / "data" / "ieee_features.npz", **pools)

    monkeypatch.chdir(tmp_path)          # IEEECISSource reads data/ieee_features.npz
    for split in ("train", "test"):
        env = FrictionEnv(SimConfig(p_takeover=1.0), source="ieee", split=split)
        obs, _ = env.reset(seed=0)
        done, steps = False, 0
        while not done:
            obs, r, term, trunc, info = env.step(OTP)
            done, steps = term or trunc, steps + 1
        assert steps >= 1 and np.isfinite(info["stats"]["money"])
        assert env.user_mu in set(pools[f"user_mu_{split}"].tolist())


# ---------------------------------------------------------------- exploratory red team (attacker.py)
# Episode stats recorded with the ORIGINAL env.py (before the attacker hook existed): Rule-KB defender,
# guaranteed takeover. The hook must not change a single number when attacker=None.
EXPECTED_NO_ATTACKER = {
    0: {'genuine_value': 169.38069862892695, 'fraud_loss': 0.0, 'otp': 1, 'false_blocks': 2, 'churned': 1,
        'frozen': 0, 'fraud_txns': 1, 'fraud_stopped': 1, 'fraud_completed': 0, 'genuine_txns': 8, 'takeover': 1,
        'money': -2932.619301371073},
    1: {'genuine_value': 38.22362900104803, 'fraud_loss': 0.0, 'otp': 1, 'false_blocks': 2, 'churned': 1,
        'frozen': 0, 'fraud_txns': 1, 'fraud_stopped': 1, 'fraud_completed': 0, 'genuine_txns': 4, 'takeover': 1,
        'money': -3063.776370998952},
    2: {'genuine_value': 260.0484446573248, 'fraud_loss': 2145.1937583039166, 'otp': 1, 'false_blocks': 0,
        'churned': 0, 'frozen': 0, 'fraud_txns': 2, 'fraud_stopped': 0, 'fraud_completed': 2, 'genuine_txns': 18,
        'takeover': 1, 'money': -1887.1453136465918},
}


def _play(env, defender, seed):
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(defender(obs))
        done = term or trunc
    return env.stats


def test_no_attacker_is_unchanged():
    from frictionrl.baselines import rule_kb
    env = FrictionEnv(SimConfig(p_takeover=1.0), attacker=None)
    for seed, expected in EXPECTED_NO_ATTACKER.items():
        assert _play(env, rule_kb, seed) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_fraudster_env_api():
    from frictionrl.attacker import FraudsterEnv
    from frictionrl.baselines import rule_kb
    check_env(FraudsterEnv(rule_kb), skip_render_check=True)


def test_attacker_that_leaves_steals_nothing():
    from frictionrl.attacker import LEAVE
    env = FrictionEnv(SimConfig(p_takeover=1.0), attacker=lambda fobs: LEAVE)
    for seed in range(20):
        stats = _play(env, lambda o: APPROVE, seed)
        assert stats["fraud_received"] == 0 and stats["fraud_loss"] == 0 and stats["fraud_txns"] == 0
        assert stats["attack_actions"] == [LEAVE]      # asked once, at the first fraudulent transaction


def test_stealth_attacks_look_less_risky():
    def new_device_rate(action):
        env = FrictionEnv(SimConfig(p_takeover=1.0), attacker=lambda fobs: action)
        seen = flagged = 0
        for seed in range(150):
            obs, _ = env.reset(seed=seed)
            done = False
            while not done:
                if env.fraud_active:
                    seen += 1
                    flagged += env.txn.new_device
                obs, _, term, trunc, _ = env.step(APPROVE)
                done = term or trunc
        return flagged / seen
    assert new_device_rate(5) < new_device_rate(4) - 0.3     # large + stealth vs large, no stealth (0.10 vs 0.65)
