# Data

FrictionRL runs out of the box on its **synthetic** transaction source; nothing needs downloading.

To use **real transactions** from Kaggle's IEEE-CIS Fraud Detection competition (590k card transactions, ~3.5% fraud):

```bash
pip install kaggle                       # needs ~/.kaggle/kaggle.json and accepting the competition rules
kaggle competitions download -c ieee-fraud-detection -p data/ieee-fraud-detection
unzip data/ieee-fraud-detection/ieee-fraud-detection.zip -d data/ieee-fraud-detection
python -m frictionrl.data_ieee --raw data/ieee-fraud-detection --out data/ieee_features.npz
```

The builder rebuilds customers with the Kaggle `uid = card1 + addr1 + (day - D1)` trick, computes the same
5 per-transaction features the simulator uses (amount vs. the customer's usual, new device, new payee,
odd hour, burst), and saves genuine/fraud pools split 80/20 **by customer** into train/test.

Then run any experiment on real transaction content:

```bash
FRICTIONRL_SOURCE=ieee python scripts/run_all.py
```

What stays simulated, because no public dataset records it: how genuine users react to OTPs and blocks
(churn), and how fraudsters react (passing OTP, retrying with a smaller amount, giving up).
Those constants live in `frictionrl/config.py`; `results/sensitivity_ltv.csv` shows how the policy
shifts when they change.
