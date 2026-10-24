# FrictionRL — learning when to ask for an OTP

**CSL3090 Artificial Intelligence (IIT Jodhpur) · Assignment 2: Real-Life Use Cases of RL**
Team: Tanisha Ray (B24CM1061), Cherry Aggarwal (B24CM1016)

Every UPI or card payment gets one of three decisions: **approve**, **step-up (OTP)**, or **block**.
Static rules either let fraudsters drain accounts or bury genuine users in OTPs until they leave the app.
FrictionRL treats each user session as a **Markov Decision Process** and trains RL agents to trade fraud
loss against friction and churn — in rupees.

Why this needs RL and not a classifier:
* **Friction compounds:** every OTP raises the chance the next one makes a genuine user quit.
* **Acting early pays later:** stopping a fraudster now prevents all of their next transactions.
* **Actions reveal information:** a failed OTP is strong evidence of account takeover.
* **The attacker adapts:** a blocked fraudster retries with a smaller amount.

## The MDP

| | Definition |
|---|---|
| **Episode** | one user session, up to H = 20 transactions. With p = 0.15 the account is taken over at a random step (hidden from the agent); the fraudster cashes out or gives up later |
| **State S** | 11-dim vector: amount vs. user's usual (log-ratio), new device, new payee, odd hour, burst, session risk rate, progress t/H, #OTPs issued, OTP-failed seen, OTP-passed seen, #blocks. Discrete version (432 states) for tabular methods |
| **Actions A** | Discrete(3): APPROVE, OTP step-up, BLOCK |
| **Reward R** (Rs) | +2% of a completed genuine payment; −(amount + Rs 200 chargeback) for completed fraud; −Rs 2 per OTP; −Rs 50 per false block; −Rs 3000 (lifetime value) if the genuine user churns; −Rs 1500 if a genuine account is frozen |
| **Dynamics T** | genuine OTP pass 97%; churn after OTP = 1% + 4%·(#previous OTPs); churn after false block = 20% + 20%·(#previous blocks); fraudster beats OTP 15%; after a failed OTP the fraudster quits 70%; after a block it retries at half the amount 50%; 2nd block freezes the account (terminal) |
| **γ** | 0.99 |

All constants live in [`frictionrl/config.py`](frictionrl/config.py).

## Algorithms (one per lecture)

| Method | Lecture | File |
|---|---|---|
| Baselines: always-approve / always-OTP / random / amount threshold / propositional rule base | Lec 16 | `baselines.py` |
| Model-based learning (count T̂, R̂) + **value iteration** + **policy iteration** | Lec 27–28 | `planning.py` |
| **Tabular Q-learning**: ε-greedy and exploration function f(u,n)=u+k/n | Lec 28–29 | `qlearning.py` |
| **Approximate Q-learning** with linear features (interpretable weights) | Lec 29 | `qlearning.py` |
| **DQN** (experience replay + target network) and **PPO** (actor–critic, GAE, clipped objective) | Lec 29 | `train_sb3.py` |

Tabular methods, VI/PI and linear Q are written from scratch with NumPy; DQN and PPO use Stable-Baselines3.

## Setup

```bash
pip install -r requirements.txt
```
