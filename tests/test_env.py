"""Quick checks: python -m pytest -q"""
import numpy as np
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
