#!/usr/bin/env python
import argparse
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def time_split(df: pd.DataFrame, test_days: int = 1, val_days: int = 1):
    # Split by time per symbol jointly
    df = df.sort_values("timestamp")
    max_ts = df["timestamp"].max()
    day_ms = 24 * 3600 * 1000
    test_start = max_ts - test_days * day_ms
    val_start = test_start - val_days * day_ms

    train = df[df["timestamp"] < val_start]
    val = df[(df["timestamp"] >= val_start) & (df["timestamp"] < test_start)]
    test = df[df["timestamp"] >= test_start]
    return train, val, test


def train_baseline(parquet_path: str, out_dir: str, horizon: int) -> None:
    df = pd.read_parquet(parquet_path)
    target_col = f"target_sign_dmid_{horizon}s"
    features = [
        "mid",
        "spread",
        "trade_count",
        "volume_total",
        "volume_buy",
        "volume_sell",
        "ofi",
        "vwap",
    ]
    df = df.dropna(subset=features + [target_col])

    # Encode symbol as category if multiple symbols
    if df["symbol"].nunique() > 1:
        df["symbol_cat"] = df["symbol"].astype("category").cat.codes
        features.append("symbol_cat")

    train, val, test = time_split(df, test_days=1, val_days=1)

    X_train, y_train = train[features], train[target_col]
    X_val, y_val = val[features], val[target_col]
    X_test, y_test = test[features], test[target_col]

    os.makedirs(out_dir, exist_ok=True)

    # Baseline 1: Logistic Regression
    lr = LogisticRegression(max_iter=200, n_jobs=1)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("LogReg report (test):\n", classification_report(y_test, y_pred, digits=3))
    joblib.dump(lr, os.path.join(out_dir, f"logreg_{horizon}s.joblib"))

    # Baseline 2: XGBoost
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        n_jobs=2,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = xgb.predict(X_test)
    print("XGB report (test):\n", classification_report(y_test, y_pred, digits=3))
    joblib.dump(xgb, os.path.join(out_dir, f"xgb_{horizon}s.joblib"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train baseline models on dataset")
    ap.add_argument("--data", default="data/ml/dataset.parquet", help="Parquet dataset path")
    ap.add_argument("--out", default="data/ml/models", help="Output models dir")
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 3, 5], help="Target horizon seconds")
    args = ap.parse_args()

    train_baseline(args.data, args.out, args.horizon)


if __name__ == "__main__":
    main()


