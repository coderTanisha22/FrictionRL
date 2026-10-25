"""Build real feature pools from Kaggle's IEEE-CIS Fraud Detection data (see data/README.md).

    python -m frictionrl.data_ieee --raw data/ieee-fraud-detection --out data/ieee_features.npz

Customers are rebuilt with the Kaggle "uid" trick: uid = card1 + addr1 + (transaction day - D1).
Per transaction we compute the simulator's 5 features:
    log_amount_ratio : log(amount) - mean log(amount) of this customer's PREVIOUS transactions
    new_device       : DeviceInfo differs from the customer's first device (missing -> 0)
    new_payee        : recipient e-mail domain not seen before for this customer (missing -> 0)
    odd_hour         : transaction hour in [0, 5]
    burst            : < 10 minutes since the customer's previous transaction
Customers are split 80/20 into train/test by uid, so test sessions use unseen customers. Each split also
stores every customer's mean log amount (user_mu_*), converted from dollars to rupees with INR_PER_USD.
"""
from __future__ import annotations

import argparse
import os
import zlib

import numpy as np
import pandas as pd

INR_PER_USD = 85.0   # set to the current exchange rate (or pass --inr-per-usd)

USECOLS = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "card1", "addr1", "D1",
           "R_emaildomain"]


def build(raw_dir: str, inr_per_usd: float = INR_PER_USD) -> dict:
    tx = pd.read_csv(os.path.join(raw_dir, "train_transaction.csv"), usecols=USECOLS)
    idp = os.path.join(raw_dir, "train_identity.csv")
    if os.path.exists(idp):
        ident = pd.read_csv(idp, usecols=["TransactionID", "DeviceInfo"])
        tx = tx.merge(ident, on="TransactionID", how="left")
    else:
        tx["DeviceInfo"] = np.nan

    day = tx.TransactionDT // 86400
    tx["uid"] = (tx.card1.astype(str) + "_" + tx.addr1.fillna(-1).astype(int).astype(str) + "_"
                 + (day - tx.D1.fillna(0)).astype(int).astype(str))
    tx = tx.sort_values(["uid", "TransactionDT"]).reset_index(drop=True)
    g = tx.groupby("uid", sort=False)

    la = np.log(tx.TransactionAmt.clip(lower=1.0))
    prev_mean = la.groupby(tx.uid).transform(lambda s: s.expanding().mean().shift(1))
    tx["log_amount_ratio"] = (la - prev_mean).fillna(0.0).clip(-3, 4)

    first_dev = g.DeviceInfo.transform("first")
    tx["new_device"] = (tx.DeviceInfo.notna() & first_dev.notna() & (tx.DeviceInfo != first_dev)).astype(int)

    seen = tx.groupby(["uid", "R_emaildomain"]).cumcount()
    tx["new_payee"] = (tx.R_emaildomain.notna() & (seen == 0) & (g.cumcount() > 0)).astype(int)

    tx["odd_hour"] = (((tx.TransactionDT // 3600) % 24) < 6).astype(int)
    gap = g.TransactionDT.diff()
    tx["burst"] = (gap.notna() & (gap < 600)).astype(int)

    is_test = tx.uid.map(lambda u: zlib.crc32(u.encode()) % 5 == 0)
    cols = ["log_amount_ratio", "new_device", "new_payee", "odd_hour", "burst"]
    out = {}
    for split, m in [("train", ~is_test), ("test", is_test)]:
        part = tx[m]
        out[f"genuine_{split}"] = part.loc[part.isFraud == 0, cols].to_numpy(np.float32)
        out[f"fraud_{split}"] = part.loc[part.isFraud == 1, cols].to_numpy(np.float32)
        out[f"user_mu_{split}"] = (la[m].groupby(part.uid).mean() + np.log(inr_per_usd)).to_numpy(np.float32)
    sessions = g.size()
    print(f"{len(tx):,} transactions, {tx.uid.nunique():,} reconstructed customers, "
          f"{(sessions >= 5).sum():,} with >= 5 transactions, fraud rate {tx.isFraud.mean():.2%}")
    for k, v in out.items():
        print(f"  {k:14s} {len(v):>8,} rows   feature means {np.round(v.mean(0), 3)}")
    out["inr_per_usd"] = np.array(inr_per_usd, dtype=np.float32)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/ieee-fraud-detection")
    ap.add_argument("--out", default="data/ieee_features.npz")
    ap.add_argument("--inr-per-usd", type=float, default=INR_PER_USD,
                    help="exchange rate used to convert dollar amounts to rupees")
    a = ap.parse_args()
    np.savez_compressed(a.out, **build(a.raw, a.inr_per_usd))
    print("saved", a.out)
