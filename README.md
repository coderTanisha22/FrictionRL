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

## Real data (IEEE-CIS)

The default transaction source is synthetic, so the project runs straight after `git clone` with nothing to
download. It can also replay real transactions from Kaggle's IEEE-CIS Fraud Detection data.

**The dataset is not in this repository.** Kaggle's competition rules restrict redistributing it, and the
raw CSVs are about 1.3 GB. Each person downloads it with their own Kaggle account (free).

### Reproduce on real data

1. One-time Kaggle setup:
   * `pip install kaggle`
   * kaggle.com → your profile → **Settings → API → Create New Token**, then
     `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
   * Accept the competition rules: <https://www.kaggle.com/c/ieee-fraud-detection/rules>
2. Download the data, build the features and check them (one command, safe to re-run):
   ```bash
   bash scripts/get_ieee_data.sh            # optional: --inr-per-usd <rate>
   ```
3. Run every experiment on real transactions and draw the figures:
   ```bash
   python scripts/run_all.py --source ieee --out results_ieee   # about 35-45 min on 2 cores
   python scripts/make_figures.py --results results_ieee
   ```

Prefer to do it by hand? `get_ieee_data.sh` runs exactly these steps:

```bash
kaggle competitions download -c ieee-fraud-detection -p data/ieee-fraud-detection
unzip data/ieee-fraud-detection/ieee-fraud-detection.zip -d data/ieee-fraud-detection
python -m frictionrl.data_ieee --raw data/ieee-fraud-detection --out data/ieee_features.npz
python scripts/check_ieee.py
```

Dollar amounts are converted to rupees (default Rs 85 per USD, `--inr-per-usd` to change), so the costs in
`config.py` apply unchanged. Real-data results are written to `results_ieee/`; the synthetic results in
`results/` are not touched. The trained real-data agents, result tables and figures are committed in
`results_ieee/`, so you can inspect them without downloading anything.
See [`data/README.md`](data/README.md) for details. User and fraudster *reactions* (churn, OTP pass rates,
retries) are always simulated: no public dataset records them.

### Results on real transactions (5,000 held-out sessions, unseen customers)

<!-- TODO: fill in from results_ieee/results_table.md after running the commands above -->

| Agent | Net Rs / session ↑ | Fraud loss Rs / session ↓ | Churn % ↓ | OTPs / session |
|---|---|---|---|---|
| Rule base (propositional) | TODO | TODO | TODO | TODO |
| Model-based value iteration | TODO | TODO | TODO | TODO |
| Tabular Q-learning (ε-greedy) | TODO | TODO | TODO | TODO |
| **DQN** | TODO | TODO | TODO | TODO |
| **PPO** | TODO | TODO | TODO | TODO |
