"""Tabular Q-learning (Lec 28-29) and approximate Q-learning with linear features (Lec 29), from scratch."""
from __future__ import annotations

import numpy as np

from .config import ACTIONS, OBS_NAMES
from .env import N_DISCRETE_STATES, FrictionEnv

A = 3


def tabular_q_learning(env: FrictionEnv, n_episodes=200_000, gamma=0.99, explore="egreedy",
                       eps_start=1.0, eps_end=0.05, eps_decay_frac=0.7, k_bonus=0.05,
                       alpha_power=0.6, alpha_min=0.02, seed=0, q_star=None, eval_every=5_000):
    """Off-policy Q-learning on discrete states. explore = "egreedy" or "explfn" (exploration function).
    Returns Q, visit counts N and a log (avg return, and |Q - Q*| / agreement with Q* if q_star is given)."""
    assert env.mode == "discrete"
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_DISCRETE_STATES, A))
    N = np.zeros((N_DISCRETE_STATES, A))
    log, window = [], []
    decay_eps = int(n_episodes * eps_decay_frac)

    def f(u, n):
        # exploration function f(u, n) = u + k / (n + 1): optimism for rarely tried actions (Lec 29)
        return u + k_bonus / (n + 1.0)

    for ep in range(n_episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / decay_eps)   # linear epsilon decay
        s, _ = env.reset(seed=seed * 10_000_000 + ep)
        done, G = False, 0.0
        while not done:
            if explore == "egreedy":
                a = int(rng.integers(A)) if rng.random() < eps else int(Q[s].argmax())
            else:
                a = int(f(Q[s], N[s]).argmax())
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            # sample = r + gamma * max_a' Q(s', a')   (just r at the end of the episode)
            if done:
                target = r
            elif explore == "egreedy":
                target = r + gamma * Q[s2].max()
            else:
                target = r + gamma * f(Q[s2], N[s2]).max()
            N[s, a] += 1
            alpha = max(alpha_min, 1.0 / N[s, a] ** alpha_power)   # decaying learning rate, not too fast (Lec 28)
            # Q-learning update: Q(s,a) <- (1 - alpha) Q(s,a) + alpha * sample
            Q[s, a] = (1 - alpha) * Q[s, a] + alpha * target
            G += r
            s = s2
        window.append(G)
        if (ep + 1) % eval_every == 0:
            row = dict(episode=ep + 1, avg_return=float(np.mean(window)), eps=eps)
            if q_star is not None:
                # compare with Q* only on well-visited entries, weighted by how often they are visited
                qs = q_star[:-1]
                sa = N >= 30
                err = np.abs(Q - qs)
                row["max_abs_q_err"] = float(err[sa].max()) if sa.any() else np.nan
                row["mean_abs_q_err"] = float((err[sa] * N[sa]).sum() / N[sa].sum()) if sa.any() else np.nan
                sv = N.sum(axis=1)
                st = sv >= 30
                same = (Q.argmax(1) == qs.argmax(1))
                row["policy_agreement"] = float((same[st] * sv[st]).sum() / sv[st].sum())
            log.append(row)
            window = []
    return Q, N, log


FEATURE_NAMES = ["bias"] + OBS_NAMES + ["amount_x_newcontext", "amount_sq"]


def features(obs: np.ndarray) -> np.ndarray:
    """phi(s): bias, the 11 observations, amount x (new device + new payee), amount^2 / 4."""
    lr = obs[0]
    return np.concatenate([[1.0], obs, [lr * (obs[1] + obs[2]), lr * lr / 4.0]]).astype(np.float64)


class LinearQ:
    """Q(s,a) = w_a . phi(s): one weight vector per action."""

    def __init__(self):
        self.W = np.zeros((A, len(FEATURE_NAMES)))

    def q(self, obs):
        return self.W @ features(obs)

    def __call__(self, obs):
        return int(self.q(obs).argmax())

    def weight_table(self):
        return {ACTIONS[a]: dict(zip(FEATURE_NAMES, self.W[a].round(3))) for a in range(A)}


def train_linear_q(env: FrictionEnv, n_episodes=60_000, gamma=0.99, alpha=0.01, eps_start=1.0,
                   eps_end=0.05, eps_decay_frac=0.7, seed=0, eval_every=2_000):
    """Approximate Q-learning with epsilon-greedy exploration on the continuous env."""
    assert env.mode == "continuous"
    rng = np.random.default_rng(seed)
    agent = LinearQ()
    log, window = [], []
    decay_eps = int(n_episodes * eps_decay_frac)
    for ep in range(n_episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / decay_eps)
        a_lr = alpha * (1.0 - 0.9 * ep / n_episodes)   # learning rate decays to 10% of alpha
        s, _ = env.reset(seed=seed * 10_000_000 + ep)
        done, G = False, 0.0
        while not done:
            a = int(rng.integers(A)) if rng.random() < eps else agent(s)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            phi = features(s)
            # difference = [r + gamma max_a' Q(s', a')] - Q(s, a), clipped for stability
            target = r if done else r + gamma * agent.q(s2).max()
            diff = np.clip(target - agent.W[a] @ phi, -2.0, 2.0)
            # w_i <- w_i + alpha * difference * f_i(s, a)
            agent.W[a] += a_lr * diff * phi
            G += r
            s = s2
        window.append(G)
        if (ep + 1) % eval_every == 0:
            log.append(dict(episode=ep + 1, avg_return=float(np.mean(window)), eps=eps))
            window = []
    return agent, log
