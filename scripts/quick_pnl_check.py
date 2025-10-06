import argparse
import numpy as np
import pandas as pd
from typing import Tuple

from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support

# Reuse time-based split from training utilities (sibling module)
try:
    from train_baseline import time_split  # when running as `python scripts/quick_pnl_check.py`
except ModuleNotFoundError:
    # Fallback if executed differently and package path is expected
    from scripts.train_baseline import time_split  # type: ignore


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
    mode: str = "long",
    tune_threshold: bool = False,
    tune_min: float = 0.50,
    tune_max: float = 0.90,
    tune_steps: int = 21,
    max_spread_bps: float | None = None,
    min_abs_ofi: float | None = None,
    min_volume_total: float | None = None,
    min_trade_count: int | None = None,
) -> Tuple[float, float, float, float, float, int, float]:
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
    p_val = clf.predict_proba(X.loc[va.index])[:, 1]

    # Forward return on horizon
    ret_val = compute_forward_return(data.loc[va.index, "mid"], horizon_seconds)
    ret_test = compute_forward_return(data.loc[te.index, "mid"], horizon_seconds)

    # Build filter masks for validation and test (entry filters)
    def build_filter(mask_index: pd.Index) -> np.ndarray:
        filt = np.ones(len(mask_index), dtype=bool)
        if max_spread_bps is not None:
            spread_bps = (data.loc[mask_index, "spread"] / data.loc[mask_index, "mid"]) * 1e4
            filt &= spread_bps.values <= float(max_spread_bps)
        if min_abs_ofi is not None and "ofi" in data.columns:
            filt &= np.abs(data.loc[mask_index, "ofi"].values) >= float(min_abs_ofi)
        if min_volume_total is not None and "volume_total" in data.columns:
            filt &= data.loc[mask_index, "volume_total"].values >= float(min_volume_total)
        if min_trade_count is not None and "trade_count" in data.columns:
            filt &= data.loc[mask_index, "trade_count"].values >= int(min_trade_count)
        return filt

    filt_val = build_filter(va.index)
    filt_test = build_filter(te.index)

    # Threshold tuning on validation to maximize total PnL (with filters)
    commission = commission_bps_roundtrip / 10000.0
    best_thr = threshold
    if tune_threshold or threshold is None:
        thr_grid = np.linspace(tune_min, tune_max, tune_steps)
        best_pnl = -np.inf
        best_thr_local = None
        for thr in thr_grid:
            if mode == "ls":
                sig_val_raw = np.where(p_val >= thr, 1, np.where(p_val <= 1 - thr, -1, 0))
                sig_val = sig_val_raw.copy()
                sig_val[~filt_val] = 0
            else:
                sig_val = (p_val >= thr).astype(int)
                sig_val[~filt_val] = 0
            trade_mask_val = sig_val != 0
            pnl_val = sig_val * ret_val - (commission * trade_mask_val.astype(int))
            total_pnl_val = float(pnl_val.sum())
            if total_pnl_val > best_pnl:
                best_pnl = total_pnl_val
                best_thr_local = float(thr)
        if best_thr_local is not None:
            best_thr = best_thr_local
    # Fall back if still None
    if best_thr is None:
        best_thr = 0.5

    # Build signal on test with chosen threshold and mode
    if mode == "ls":
        signal = np.where(p_test >= best_thr, 1, np.where(p_test <= 1 - best_thr, -1, 0))
        signal[~filt_test] = 0
    else:
        signal = (p_test >= best_thr).astype(int)
        signal[~filt_test] = 0

    trade_mask = signal != 0
    pnl = signal * ret_test - (commission * trade_mask.astype(int))

    # Classification metrics (binary, positive class = 1)
    # For metrics in long-only: positive prediction only when threshold AND filters pass
    y_pred_bin = (p_test >= best_thr).astype(int)
    y_pred_bin[~filt_test] = 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        y.loc[te.index], y_pred_bin, average="binary", zero_division=0
    )

    selected_share = float(trade_mask.mean())
    num_trades = int(trade_mask.sum())
    avg_pnl_per_trade_bps = float((pnl[trade_mask].mean() * 1e4) if num_trades > 0 else 0.0)

    return selected_share, precision, recall, f1, avg_pnl_per_trade_bps, num_trades, best_thr


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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["long", "ls"],
        default="long",
        help="Signal mode: long-only ('long') or long-short symmetric ('ls')",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=None,
        help="Optional single horizon to run (overrides presets). If set, use --thr or --tune",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=None,
        help="Classification threshold. If omitted with --tune, it will be tuned on validation",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable threshold tuning on validation to maximize total PnL",
    )
    parser.add_argument(
        "--th_min",
        type=float,
        default=0.50,
        help="Min threshold for tuning grid",
    )
    parser.add_argument(
        "--th_max",
        type=float,
        default=0.90,
        help="Max threshold for tuning grid",
    )
    parser.add_argument(
        "--th_steps",
        type=int,
        default=21,
        help="Number of thresholds in tuning grid",
    )
    # Entry filters
    parser.add_argument(
        "--max_spread_bps",
        type=float,
        default=None,
        help="Max allowed spread in bps for trade entry (filter)",
    )
    parser.add_argument(
        "--min_abs_ofi",
        type=float,
        default=None,
        help="Min absolute OFI for trade entry (filter)",
    )
    parser.add_argument(
        "--min_volume_total",
        type=float,
        default=None,
        help="Min total volume for trade entry (filter)",
    )
    parser.add_argument(
        "--min_trade_count",
        type=int,
        default=None,
        help="Min trade count for trade entry (filter)",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.data).dropna().sort_values("timestamp")

    print(f"Dataset: {args.data}  rows={len(df)}  commission={args.comm_bps} bps  mode={args.mode}\n")

    def run_and_print(h: float, thr: float | None, tune: bool) -> None:
        try:
            selected, prec, rec, f1, avg_pnl_bps, n_trades, used_thr = run_case(
                df,
                h,
                thr,
                args.comm_bps,
                n_jobs=args.n_jobs,
                mode=args.mode,
                tune_threshold=tune,
                tune_min=args.th_min,
                tune_max=args.th_max,
                tune_steps=args.th_steps,
                max_spread_bps=args.max_spread_bps,
                min_abs_ofi=args.min_abs_ofi,
                min_volume_total=args.min_volume_total,
                min_trade_count=args.min_trade_count,
            )
            print(
                "h={:.1f}s thr={:.2f} | selected={:.1%} trades={} | precision={:.3f} recall={:.3f} f1={:.3f} | avg_pnl_per_trade={:.2f} bps".format(
                    h, used_thr, selected, n_trades, prec, rec, f1, avg_pnl_bps
                )
            )
        except Exception as e:
            thr_txt = "auto" if (thr is None and tune) else ("{:.2f}".format(thr) if thr is not None else "?")
            print(f"h={h}s thr={thr_txt} | ERROR: {e}")

    if args.horizon is not None:
        run_and_print(args.horizon, args.thr, args.tune)
    else:
        cases = [
            (0.5, 0.80),  # horizon=0.5s, threshold=0.80
            (1.0, 0.50),  # horizon=1.0s, threshold=0.50
            (2.0, 0.50),  # horizon=2.0s, threshold=0.50
        ]
        for horizon_seconds, threshold in cases:
            run_and_print(horizon_seconds, threshold, args.tune)


if __name__ == "__main__":
    main()


