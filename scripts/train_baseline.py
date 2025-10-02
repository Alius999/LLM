#!/usr/bin/env python
import argparse
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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


def _hcol(h: float) -> str:
    return f"target_sign_dmid_{int(h) if float(h).is_integer() else h}s"


def train_baseline(parquet_path: str, out_dir: str, horizon: float, binary: bool = False) -> None:
    df = pd.read_parquet(parquet_path)
    target_col = _hcol(horizon)
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
    # include rolling features if present
    for extra in [
        "mid_ema_3s","mid_ema_5s","mid_ema_10s",
        "mid_ret_3s","mid_ret_5s","mid_ret_10s",
        "vola_3s","vola_5s","vola_10s",
        "ofi_sum_3s","ofi_sum_5s","ofi_sum_10s",
        "tc_sum_3s","tc_sum_5s","tc_sum_10s",
    ]:
        if extra in df.columns:
            features.append(extra)
    df = df.dropna(subset=features + [target_col])

    # Encode symbol as category if multiple symbols
    if df["symbol"].nunique() > 1:
        df["symbol_cat"] = df["symbol"].astype("category").cat.codes
        features.append("symbol_cat")

    train, val, test = time_split(df, test_days=1, val_days=1)

    X_train, y_train = train[features], train[target_col]
    X_val, y_val = val[features], val[target_col]
    X_test, y_test = test[features], test[target_col]

    if binary:
        # Exclude class 0, map {-1,1}->{0,1}
        mask_tr = y_train != 0
        mask_va = y_val != 0
        mask_te = y_test != 0
        X_train, y_train = X_train[mask_tr], (y_train[mask_tr] == 1).astype(int)
        X_val, y_val     = X_val[mask_va],     (y_val[mask_va] == 1).astype(int)
        X_test, y_test   = X_test[mask_te],    (y_test[mask_te] == 1).astype(int)
    else:
        # Map targets {-1,0,1} -> {0,1,2} for XGBoost multiclass
        label_map = {-1: 0, 0: 1, 1: 2}
        y_train_m = y_train.map(label_map).astype(int)
        y_val_m = y_val.map(label_map).astype(int)
        y_test_m = y_test.map(label_map).astype(int)

    os.makedirs(out_dir, exist_ok=True)

    # Baseline 1: Logistic Regression (scaled, balanced, more iters)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto", solver="lbfgs")),
    ])
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("LogReg report (test):\n", classification_report(y_test, y_pred, digits=3))
    joblib.dump(lr, os.path.join(out_dir, f"logreg_{horizon}s.joblib"))

    # Baseline 2: XGBoost
    if binary:
        # balance classes
        pos = (y_train == 1).sum(); neg = (y_train == 0).sum()
        spw = (neg / max(pos, 1)) if neg > 0 else 1.0
        xgb = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=spw, n_jobs=2,
        )
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = xgb.predict(X_test)
    else:
        xgb = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_jobs=2,
        )
        xgb.fit(X_train, y_train_m, eval_set=[(X_val, y_val_m)], verbose=False)
        y_pred_m = xgb.predict(X_test)
        inv_map = {0: -1, 1: 0, 2: 1}
        y_pred = pd.Series(y_pred_m).map(inv_map)
    print("XGB report (test):\n", classification_report(y_test, y_pred, digits=3))
    joblib.dump(xgb, os.path.join(out_dir, f"xgb_{horizon}s.joblib"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train baseline models on dataset")
    ap.add_argument("--data", default="data/ml/dataset.parquet", help="Parquet dataset path")
    ap.add_argument("--out", default="data/ml/models", help="Output models dir")
    ap.add_argument("--horizon", type=float, default=1.0, help="Target horizon seconds (e.g., 0.5, 1, 2, 3)")
    ap.add_argument("--binary", action="store_true", help="Train binary classifier (drops 0 class)")
    args = ap.parse_args()

    train_baseline(args.data, args.out, float(args.horizon), binary=bool(args.binary))


if __name__ == "__main__":
    main()


