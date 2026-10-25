"""Model-based learning (Lec 28) + value iteration and policy iteration (Lec 27).

1. Run an exploratory policy in the discrete env, COUNT outcomes s' for each (s, a) -> T_hat(s,a,s'),
   and average the rewards -> R_hat(s,a,s').
2. Solve that learned MDP with value iteration and policy iteration.
Every episode end goes to one extra absorbing state with value 0 (Lec 26).
"""
from __future__ import annotations

import numpy as np

from .env import N_DISCRETE_STATES, FrictionEnv, discretize

S = N_DISCRETE_STATES          # 432 real states
TERM = S                       # index of the absorbing state
NS = S + 1
A = 3


def learn_model(env: FrictionEnv, n_episodes: int = 100_000, behaviour=(0.7, 0.25, 0.05), seed: int = 0,
                policy=None, mix: float = 0.3):
    """Count-based T_hat, R_hat. Acts with `policy` w.p. (1 - mix), else samples from `behaviour`.
    Returns T[s,a,s'], R[s,a,s'], N[s,a] and the start-state distribution."""
    assert env.mode == "discrete"
    rng = np.random.default_rng(seed)
    counts = np.zeros((NS, A, NS))
    rsum = np.zeros((NS, A, NS))
    start = np.zeros(S)
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed * 10_000_000 + ep)
        start[s] += 1
        done = False
        while not done:
            if policy is not None and rng.random() > mix:
                a = int(policy[s])
            else:
                a = int(rng.choice(3, p=behaviour))
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            nxt = TERM if done else s2
            counts[s, a, nxt] += 1
            rsum[s, a, nxt] += r
            s = s2
    N = counts.sum(axis=2)
    T = np.zeros_like(counts)
    R = np.zeros_like(counts)
    seen = N > 0
    T[seen] = counts[seen] / N[seen][:, None]          # T_hat(s,a,s') = count(s,a,s') / count(s,a)
    nz = counts > 0
    R[nz] = rsum[nz] / counts[nz]                       # R_hat = average observed reward
    for s in range(NS):                                 # never-tried (s,a) and the absorbing state -> TERM, reward 0
        for a in range(A):
            if N[s, a] == 0:
                T[s, a, TERM] = 1.0
    return T, R, N, start / start.sum()


def q_from_v(T, R, V, gamma):
    # Q(s,a) = sum_s' T(s,a,s') [R(s,a,s') + gamma V(s')]
    return np.einsum("ijk,ijk->ij", T, R + gamma * V[None, None, :])


def value_iteration(T, R, gamma=0.99, tol=1e-6, max_iter=10_000):
    """Repeat the Bellman optimality update until max |V_new - V| < tol."""
    V = np.zeros(NS)
    deltas = []
    for k in range(max_iter):
        # V_{k+1}(s) = max_a sum_s' T(s,a,s') [R(s,a,s') + gamma V_k(s')]
        Q = q_from_v(T, R, V, gamma)
        V_new = Q.max(axis=1)
        V_new[TERM] = 0.0
        delta = np.abs(V_new - V).max()
        deltas.append(delta)
        V = V_new
        if delta < tol:
            break
    Q = q_from_v(T, R, V, gamma)
    pi = Q.argmax(axis=1)            # policy extraction: pi(s) = argmax_a Q(s,a)
    return V, Q, pi, deltas


def policy_evaluation(T, R, pi, gamma=0.99, tol=1e-8, max_iter=10_000):
    """Value of a fixed policy (no max, so each sweep is cheaper than in value iteration)."""
    V = np.zeros(NS)
    idx = np.arange(NS)
    Tp, Rp = T[idx, pi], R[idx, pi]
    for _ in range(max_iter):
        # V^pi_{k+1}(s) = sum_s' T(s,pi(s),s') [R(s,pi(s),s') + gamma V^pi_k(s')]
        V_new = (Tp * (Rp + gamma * V[None, :])).sum(axis=1)
        V_new[TERM] = 0.0
        if np.abs(V_new - V).max() < tol:
            return V_new
        V = V_new
    return V


def policy_iteration(T, R, gamma=0.99, start_action=0):
    """Evaluate the policy, improve it greedily, repeat until it stops changing (Lec 27)."""
    pi = np.full(NS, start_action)
    history = []
    while True:
        V = policy_evaluation(T, R, pi, gamma)
        Q = q_from_v(T, R, V, gamma)
        new_pi = Q.argmax(axis=1)
        changed = int((new_pi != pi).sum())
        history.append(changed)
        if changed == 0:
            return V, Q, pi, history
        pi = new_pi


class TabularPolicy:
    """Wrap a state -> action table so it can act on continuous observations."""

    def __init__(self, table):
        self.table = np.asarray(table)

    def __call__(self, obs):
        return int(self.table[discretize(obs)])


def iterative_model_based(env: FrictionEnv, rounds: int = 3, episodes_per_round: int = 60_000,
                          gamma: float = 0.99, seed: int = 0):
    """Learn T,R -> plan with VI -> collect new data with that policy -> repeat.
    Needed because the hidden fraudster makes the learned model depend on which policy collected the data."""
    policy, log = None, []
    for k in range(rounds):
        T, R, N, start = learn_model(env, episodes_per_round, seed=seed * 100 + k, policy=policy)
        V, Q, pi, deltas = value_iteration(T, R, gamma)
        log.append(dict(round=k, vi_iterations=len(deltas), states_visited=int((N[:-1].sum(1) > 0).sum())))
        policy = pi
    return T, R, N, start, V, Q, pi, log


class LearnedMDPEnv:
    """Samples from the learned T_hat, R_hat, which is exactly Markov. Q-learning here must reach value
    iteration's Q*: this checks our Q-learning code separately from whether the real problem is Markov."""
    mode = "discrete"

    def __init__(self, T, R, start):
        self.T, self.R, self.start = T, R, start
        self.rng = np.random.default_rng(0)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.s = int(self.rng.choice(S, p=self.start))
        return self.s, {}

    def step(self, a):
        s2 = int(self.rng.choice(NS, p=self.T[self.s, a]))
        r = float(self.R[self.s, a, s2])
        self.s = s2
        done = s2 == TERM
        return (0 if done else s2), r, done, False, {}
