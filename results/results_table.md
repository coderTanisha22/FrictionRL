| Agent | Net Rs/session | Fraud loss Rs/session | Fraud txns completed % | Churn % | False-block % | OTPs/session |
|---|---|---|---|---|---|---|
| Always-Approve | -1,517.3 | 2,115.9 | 100.0 | 0.0 | 0.0 | 0.0 |
| Always-OTP | -2,885.8 | 40.2 | 11.9 | 100.0 | 0.0 | 6.7 |
| Random | -2,411.2 | 85.5 | 36.8 | 57.2 | 33.3 | 1.7 |
| Amount-Threshold | -170.7 | 457.4 | 56.9 | 6.7 | 0.8 | 1.0 |
| Rule-KB | -155.1 | 475.3 | 52.0 | 5.6 | 0.9 | 0.5 |
| Model-based VI | 177.7 | 372.8 | 46.6 | 1.4 | 0.2 | 0.4 |
| Q-learning (eps-greedy) | 199.6 ± 48.8 | 219.9 ± 4.3 | 36.0 ± 2.7 | 5.0 ± 1.5 | 0.3 ± 0.0 | 1.4 ± 0.4 |
| Q-learning (expl. fn) | 158.9 ± 64.7 | 200.1 ± 20.3 | 34.5 ± 4.2 | 6.3 ± 1.5 | 0.7 ± 0.1 | 1.4 ± 0.2 |
| Linear approx. Q | 9.8 ± 97.3 | 504.1 ± 144.4 | 55.4 ± 5.9 | 2.5 ± 1.5 | 0.0 ± 0.0 | 1.2 ± 0.5 |
| DQN | 315.0 ± 20.9 | 237.8 ± 19.4 | 37.3 ± 3.8 | 1.4 ± 0.1 | 0.2 ± 0.0 | 0.4 ± 0.0 |
| PPO | 299.3 ± 29.9 | 273.1 ± 43.8 | 38.6 ± 3.2 | 0.9 ± 0.6 | 0.0 ± 0.0 | 0.5 ± 0.2 |

Value iteration: 226 sweeps to converge (tol 1e-6). Policy iteration: 5 improvement steps (policy changes per step: [182, 80, 33, 9, 0]). VI and PI policies identical on 100% of states.

PPO sensitivity to customer lifetime value (LTV):

| LTV (Rs) | OTPs/session | Churn % | Fraud loss Rs/session |
|---|---|---|---|
| 1,000 | 0.98 | 2.2 | 203 |
| 3,000 | 0.49 | 0.8 | 309 |
| 8,000 | 0.33 | 0.3 | 344 |
