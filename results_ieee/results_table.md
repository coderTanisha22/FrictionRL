| Agent | Net Rs/session | Fraud loss Rs/session | Fraud txns completed % | Churn % | False-block % | OTPs/session |
|---|---|---|---|---|---|---|
| Always-Approve | -1,182.0 | 7,102.3 | 100.0 | 0.0 | 0.0 | 0.0 |
| Always-OTP | -1,504.5 | 220.5 | 13.4 | 100.0 | 0.0 | 6.7 |
| Random | -1,725.5 | 256.8 | 38.4 | 56.3 | 33.3 | 1.7 |
| Amount-Threshold | 161.1 | 5,462.3 | 93.3 | 3.1 | 0.2 | 0.9 |
| Rule-KB | -197.5 | 6,048.9 | 96.4 | 0.7 | 0.1 | 0.2 |
| Model-based VI | -1,054.0 | 6,963.4 | 99.8 | 0.3 | 0.0 | 0.0 |
| Q-learning (eps-greedy) | -1,232.6 ± 853.2 | 958.4 ± 610.3 | 40.6 ± 12.0 | 80.8 ± 10.4 | 6.4 ± 3.4 | 5.2 ± 0.7 |
| Q-learning (expl. fn) | -2,002.7 ± 138.3 | 38.6 ± 51.7 | 5.7 ± 3.0 | 54.6 ± 1.0 | 58.1 ± 15.4 | 1.3 ± 0.9 |
| Linear approx. Q | -1,471.6 ± 97.5 | 6,964.6 ± 348.1 | 96.9 ± 1.7 | 5.5 ± 3.5 | 0.0 ± 0.0 | 1.8 ± 0.6 |
| DQN | 87.5 ± 1,007.8 | 4,918.2 ± 1,590.3 | 84.3 ± 13.5 | 12.5 ± 10.3 | 2.0 ± 2.5 | 1.4 ± 1.4 |
| PPO | 931.0 ± 540.3 | 4,394.6 ± 305.9 | 83.4 ± 5.1 | 8.5 ± 3.6 | 0.0 ± 0.0 | 2.1 ± 0.5 |

Value iteration: 1217 sweeps to converge (tol 1e-6). Policy iteration: 6 improvement steps (policy changes per step: [147, 94, 29, 9, 1, 0]). VI and PI policies identical on 100% of states.

PPO sensitivity to customer lifetime value (LTV):

| LTV (Rs) | OTPs/session | Churn % | Fraud loss Rs/session |
|---|---|---|---|
| 1,000 | 2.83 | 16.5 | 4,058 |
| 3,000 | 2.42 | 10.1 | 4,461 |
| 8,000 | 0.83 | 4.3 | 5,139 |
