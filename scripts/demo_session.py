"""Viva demo: watch a trained agent handle one session with a takeover, transaction by transaction.

    python scripts/demo_session.py --agent dqn --seed 7
    python scripts/demo_session.py --agent rule      # compare with the rule baseline
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from frictionrl.baselines import rule_kb  # noqa: E402
from frictionrl.config import ACTIONS  # noqa: E402
from frictionrl.env import FrictionEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="dqn", choices=["dqn", "ppo", "rule"])
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.agent == "rule":
        policy = rule_kb
    else:
        from stable_baselines3 import DQN, PPO
        from frictionrl.train_sb3 import SB3Policy
        cls = DQN if a.agent == "dqn" else PPO
        policy = SB3Policy(cls.load(f"results/models/{a.agent}_seed0_ltv3000"))

    # start at --seed and move on until a session has a takeover
    env = FrictionEnv(mode="continuous", split="test")
    seed = a.seed
    obs, _ = env.reset(seed=seed)
    while env.takeover_step is None:
        seed += 1
        obs, _ = env.reset(seed=seed)
    print(f"session seed={seed}  takeover at step {env.takeover_step}  (hidden from the agent)\n")
    print(f"{'t':>2} {'who':8} {'amount':>9} {'x usual':>7} {'newDev':>6} {'newPayee':>8} {'oddHr':>5} "
          f"{'burst':>5}  {'action':8} {'reward Rs':>9}")
    done = False
    while not done:
        txn, who = env.txn, ("FRAUD" if env.fraud_active else "owner")
        act = policy(obs)
        t = env.t
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc
        print(f"{t:>2} {who:8} {txn.amount:>9,.0f} {math.exp(txn.log_amount_ratio):>7.1f} {txn.new_device:>6} "
              f"{txn.new_payee:>8} {txn.odd_hour:>5} {txn.burst:>5}  {ACTIONS[act]:8} {r * 1000:>9,.0f}")
    s = info["stats"]
    print(f"\nnet Rs {s['money']:,.0f} | fraud lost Rs {s['fraud_loss']:,.0f} | OTPs {s['otp']} | "
          f"churned {bool(s['churned'])} | frozen {bool(s['frozen'])}")


if __name__ == "__main__":
    main()
