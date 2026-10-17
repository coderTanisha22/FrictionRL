# FrictionRL

Millions of UPI users face fraud, while rigid rules block genuine payments. FrictionRL models user sessions as an MDP:

- **State:** transaction and session-risk signals
- **Actions:** approve, OTP step-up, or block
- **Reward:** trades fraud loss against friction and churn

PPO/DQN agents train against adaptive fraudsters and are benchmarked against threshold rules.

## Structure

```
FrictionRL/
├── frictionrl/ # environment, agents, baselines
├── results/    # logs, plots, saved models
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```
