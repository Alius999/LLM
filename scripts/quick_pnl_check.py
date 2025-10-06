import argparse
import numpy as np
import pandas as pd
from typing import Tuple

from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support

# Reuse time-based split from training utilities
from scripts.train_baseline import time_split


def compute_forward_return(series_mid: pd.Series, horizon_seconds: float) -> pd.Series:
    """
    Compute forward return for the specified horizon.

    Notes
    -----
    - For integer horizons measured in seconds, we use an exact shift by k rows (1 Hz data).
    - For fractional horizons (e.g., 0.5s) on 1 Hz data, we approximate by scaling the 1s return.
    """
    if float(horizon_seconds).is_integer():
        k = int(horizon_seconds)
        if k <= 0:
            raise ValueError("horizon_seconds must be positive")
        return series_mid.shift(-k) / series_mid - 1.0

    # Fractional second approximation on 1 Hz data
    ret_1s = series_mid.shift(-1) / series_mid - 1.0
    return ret_1s * float(horizon_seconds)


def run_case(
    df: pd.DataFrame,
    horizon_seconds: float,
    threshold: float,
    commission_bps_roundtrip: float,
    n_jobs: int = 4,
) -> Tuple[float, float, float, float, float, int]:
    """
    Train XGB on train/val, evaluate on test with a fixed classification threshold,
    compute long-only PnL after subtracting round-trip commission.

    Returns
    -------
    selected_share, precision, recall, f1, avg_pnl_per_trade_bps, num_trades
    """
    # Build column name for target
    ycol_suffix = int(horizon_seconds) if float(horizon_seconds).is_integer() else horizon_seconds
    ycol = f"target_sign_dmid_{ycol_suffix}s"
    if ycol not in df.columns:
        raise KeyError(f"Target column '{ycol}' not found in dataset.")

    data = df[df[ycol] != 0].copy()
    if data.empty:
        raise ValueError("Filtered dataset is empty after removing neutral targets (0).")

    # Prepare features/target
    y = (data[ycol] == 1).astype(int)
    drop_cols = [c for c in data.columns if c.startswith("target_sign_dmid_") or c in ["timestamp", "symbol"]]
    X = data.drop(columns=drop_cols, errors="ignore")

    # Time split: 1 day for validation, 1 day for test
    tr, va, te = time_split(data, val_days=1, test_days=1)

    # Class imbalance handling
    pos = (y.loc[tr.index] == 1).sum()
    neg = (y.loc[tr.index] == 0).sum()
    scale_pos_weight = (neg / max(pos, 1)) if neg > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=n_jobs,
        tree_method="hist",
    )

    clf.fit(X.loc[tr.index], y.loc[tr.index], eval_set=[(X.loc[va.index], y.loc[va.index])], verbose=False)

    # Predictions on test
    p_test = clf.predict_proba(X.loc[te.index])[:, 1]

    # Long-only signal based on threshold
    signal = (p_test >= threshold).astype(int)

    # Forward return on horizon
    ret_h = compute_forward_return(data.loc[te.index, "mid"], horizon_seconds)

    # Round-trip commission in decimal
    commission = commission_bps_roundtrip / 10000.0

    trade_mask = signal != 0
    pnl = signal * ret_h - (commission * trade_mask.astype(int))

    # Classification metrics (binary, positive class = 1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y.loc[te.index], (p_test >= threshold).astype(int), average="binary", zero_division=0
    )

    selected_share = float(trade_mask.mean())
    num_trades = int(trade_mask.sum())
    avg_pnl_per_trade_bps = float((pnl[trade_mask].mean() * 1e4) if num_trades > 0 else 0.0)

    return selected_share, precision, recall, f1, avg_pnl_per_trade_bps, num_trades


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick PnL check for preset horizons/thresholds")
    parser.add_argument(
        "--data",
        type=str,
        default="data/ml/ds_btc_0p5_1_2.parquet",
        help="Path to parquet dataset (built by scripts/build_dataset.py)",
    )
    parser.add_argument(
        "--comm_bps",
        type=float,
        default=2.0,
        help="Round-trip commission in basis points (e.g., 2.0 bps)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=4,
        help="Parallel threads for XGBoost",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.data).dropna().sort_values("timestamp")

    cases = [
        (0.5, 0.80),  # horizon=0.5s, threshold=0.80
        (1.0, 0.50),  # horizon=1.0s, threshold=0.50
        (2.0, 0.50),  # horizon=2.0s, threshold=0.50
    ]

    print(f"Dataset: {args.data}  rows={len(df)}  commission={args.comm_bps} bps\n")

    for horizon_seconds, threshold in cases:
        try:
            selected, prec, rec, f1, avg_pnl_bps, n_trades = run_case(
                df, horizon_seconds, threshold, args.comm_bps, n_jobs=args.n_jobs
            )
            print(
                "h={:.1f}s thr={:.2f} | selected={:.1%} trades={} | precision={:.3f} recall={:.3f} f1={:.3f} | avg_pnl_per_trade={:.2f} bps".format(
                    horizon_seconds, threshold, selected, n_trades, prec, rec, f1, avg_pnl_bps
                )
            )
        except Exception as e:
            print(f"h={horizon_seconds}s thr={threshold:.2f} | ERROR: {e}")


if __name__ == "__main__":
    main()


