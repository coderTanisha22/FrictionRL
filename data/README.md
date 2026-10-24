# Data

FrictionRL runs out of the box on its **synthetic** transaction source; nothing needs downloading.

To use **real transactions** from Kaggle's IEEE-CIS Fraud Detection competition (590k card transactions, ~3.5% fraud),
download it with your own Kaggle account. The data is not stored in this repository: the competition rules restrict
redistributing it, and the raw CSVs are about 1.3 GB. `data/ieee-fraud-detection/` and `data/*.npz` are git-ignored.

Quickest way, from the repository root (needs `pip install kaggle`, an API token in `~/.kaggle/kaggle.json`,
and the rules accepted at https://www.kaggle.com/c/ieee-fraud-detection/rules):

```bash
bash scripts/get_ieee_data.sh      # download, unzip, build data/ieee_features.npz, run check_ieee.py
```

Or step by step:

```bash
pip install kaggle                       # needs ~/.kaggle/kaggle.json and accepting the competition rules
kaggle competitions download -c ieee-fraud-detection -p data/ieee-fraud-detection
unzip data/ieee-fraud-detection/ieee-fraud-detection.zip -d data/ieee-fraud-detection
python -m frictionrl.data_ieee --raw data/ieee-fraud-detection --out data/ieee_features.npz
```

The builder rebuilds customers with the Kaggle `uid = card1 + addr1 + (day - D1)` trick, computes the same
5 per-transaction features the simulator uses (amount vs. the customer's usual, new device, new payee,
odd hour, burst), and saves genuine/fraud pools split 80/20 **by customer** into train/test.
It also stores each customer's typical (mean log) amount, so every simulated session uses a real customer's
spending level. IEEE-CIS amounts are in US dollars, so they are converted to rupees at `INR_PER_USD`
(default 85, in `frictionrl/data_ieee.py`; override with `--inr-per-usd`). Amounts and the rupee costs in
`config.py` are then in the same currency, and real-data results are directly comparable with `results/`.

Check the data, then run every experiment on real transaction content. Results go to `results_ieee/`,
so the synthetic results in `results/` are never overwritten:

```bash
python scripts/check_ieee.py                               # pool sizes, fraud share, feature means
                                                           # + results_ieee/figures/fig_data_compare.png
python scripts/run_all.py --source ieee --out results_ieee
python scripts/make_figures.py --results results_ieee
```

All six steps in order:

```bash
kaggle competitions download -c ieee-fraud-detection -p data/ieee-fraud-detection
unzip data/ieee-fraud-detection/ieee-fraud-detection.zip -d data/ieee-fraud-detection
python -m frictionrl.data_ieee --raw data/ieee-fraud-detection --out data/ieee_features.npz
python scripts/check_ieee.py
python scripts/run_all.py --source ieee --out results_ieee
python scripts/make_figures.py --results results_ieee
```

What stays simulated, because no public dataset records it: how genuine users react to OTPs and blocks
(churn), and how fraudsters react (passing OTP, retrying with a smaller amount, giving up).
Those constants live in `frictionrl/config.py`; `results/sensitivity_ltv.csv` shows how the policy
shifts when they change.
