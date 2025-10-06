import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support


# Reuse time-based split from training utilities (sibling module)
try:
    from train_baseline import time_split  # when running as `python scripts/paper_trader.py`
except ModuleNotFoundError:
    from scripts.train_baseline import time_split  # type: ignore


def compute_forward_return(series_mid: pd.Series, horizon_seconds: float) -> pd.Series:
    if float(horizon_seconds).is_integer():
        k = int(horizon_seconds)
        if k <= 0:
            raise ValueError("horizon_seconds must be positive")
        return series_mid.shift(-k) / series_mid - 1.0
    ret_1s = series_mid.shift(-1) / series_mid - 1.0
    return ret_1s * float(horizon_seconds)


def build_entry_filter(
    df: pd.DataFrame,
    index: pd.Index,
    max_spread_bps: float | None,
    min_abs_ofi: float | None,
    min_volume_total: float | None,
    min_trade_count: int | None,
) -> np.ndarray:
    filt = np.ones(len(index), dtype=bool)
    if max_spread_bps is not None:
        spread_bps = (df.loc[index, "spread"] / df.loc[index, "mid"]) * 1e4
        filt &= spread_bps.values <= float(max_spread_bps)
    if min_abs_ofi is not None and "ofi" in df.columns:
        filt &= np.abs(df.loc[index, "ofi"].values) >= float(min_abs_ofi)
    if min_volume_total is not None and "volume_total" in df.columns:
        filt &= df.loc[index, "volume_total"].values >= float(min_volume_total)
    if min_trade_count is not None and "trade_count" in df.columns:
        filt &= df.loc[index, "trade_count"].values >= int(min_trade_count)
    return filt


def train_and_simulate(
    df: pd.DataFrame,
    horizon_seconds: float,
    commission_bps_roundtrip: float,
    n_jobs: int,
    mode: str,
    threshold: float | None,
    tune_threshold: bool,
    tune_min: float,
    tune_max: float,
    tune_steps: int,
    val_days: int,
    test_days: int,
    max_spread_bps: float | None,
    min_abs_ofi: float | None,
    min_volume_total: float | None,
    min_trade_count: int | None,
) -> Tuple[pd.DataFrame, float, dict]:
    ycol_suffix = int(horizon_seconds) if float(horizon_seconds).is_integer() else horizon_seconds
    ycol = f"target_sign_dmid_{ycol_suffix}s"
    if ycol not in df.columns:
        raise KeyError(f"Target column '{ycol}' not found in dataset.")

    data = df[df[ycol] != 0].copy()
    if data.empty:
        raise ValueError("Filtered dataset is empty after removing neutral targets (0).")

    y = (data[ycol] == 1).astype(int)
    drop_cols = [c for c in data.columns if c.startswith("target_sign_dmid_") or c in ["timestamp", "symbol"]]
    X = data.drop(columns=drop_cols, errors="ignore")

    tr, va, te = time_split(data, val_days=val_days, test_days=test_days)

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

    p_val = clf.predict_proba(X.loc[va.index])[:, 1]
    p_test = clf.predict_proba(X.loc[te.index])[:, 1]

    commission = commission_bps_roundtrip / 10000.0
    ret_val = compute_forward_return(data.loc[va.index, "mid"], horizon_seconds)
    ret_test = compute_forward_return(data.loc[te.index, "mid"], horizon_seconds)

    filt_val = build_entry_filter(
        data, va.index, max_spread_bps, min_abs_ofi, min_volume_total, min_trade_count
    )
    filt_test = build_entry_filter(
        data, te.index, max_spread_bps, min_abs_ofi, min_volume_total, min_trade_count
    )

    best_thr = threshold
    if tune_threshold or threshold is None:
        thr_grid = np.linspace(tune_min, tune_max, tune_steps)
        best_pnl = -np.inf
        best_thr_local = None
        for thr in thr_grid:
            if mode == "ls":
                sig_val = np.where(p_val >= thr, 1, np.where(p_val <= 1 - thr, -1, 0))
            else:
                sig_val = (p_val >= thr).astype(int)
            sig_val[~filt_val] = 0
            pnl_val = sig_val * ret_val - (commission * (sig_val != 0).astype(int))
            total_pnl_val = float(pnl_val.sum())
            if total_pnl_val > best_pnl:
                best_pnl = total_pnl_val
                best_thr_local = float(thr)
        if best_thr_local is not None:
            best_thr = best_thr_local
    if best_thr is None:
        best_thr = 0.5

    if mode == "ls":
        signal = np.where(p_test >= best_thr, 1, np.where(p_test <= 1 - best_thr, -1, 0))
    else:
        signal = (p_test >= best_thr).astype(int)
    signal[~filt_test] = 0

    pnl = signal * ret_test - (commission * (signal != 0).astype(int))

    # Classification metrics on test (binary)
    y_pred_bin = (p_test >= best_thr).astype(int)
    y_pred_bin[~filt_test] = 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        y.loc[te.index], y_pred_bin, average="binary", zero_division=0
    )

    # Build trade log (only executed entries) — align by position to avoid mask length mismatches
    test_df = data.loc[te.index].copy()
    trades_mask = (signal != 0)
    if trades_mask.any():
        idx = np.flatnonzero(trades_mask)
        # Base arrays
        ts = test_df.index.values
        mid_arr = test_df["mid"].to_numpy()
        spread_arr = test_df["spread"].to_numpy() if "spread" in test_df.columns else np.full(len(test_df), np.nan)
        ofi_arr = test_df["ofi"].to_numpy() if "ofi" in test_df.columns else np.full(len(test_df), np.nan)
        vol_arr = (
            test_df["volume_total"].to_numpy() if "volume_total" in test_df.columns else np.full(len(test_df), np.nan)
        )
        tc_arr = test_df["trade_count"].to_numpy() if "trade_count" in test_df.columns else np.full(len(test_df), np.nan)

        trades = pd.DataFrame(
            {
                "timestamp": ts[idx],
                "prob_up": p_test[idx],
                "signal": signal[idx],
                "mid": mid_arr[idx],
                "spread": spread_arr[idx],
                "ofi": ofi_arr[idx],
                "volume_total": vol_arr[idx],
                "trade_count": tc_arr[idx],
                "fwd_return": pd.Series(ret_test).to_numpy()[idx],
                "pnl_bps": (pnl * 1e4)[idx],
                "passed_filters": filt_test.astype(int)[idx],
            }
        ).reset_index(drop=True)
    else:
        trades = pd.DataFrame(
            columns=[
                "timestamp",
                "prob_up",
                "signal",
                "mid",
                "spread",
                "ofi",
                "volume_total",
                "trade_count",
                "fwd_return",
                "pnl_bps",
                "passed_filters",
            ]
        )

    summary = {
        "used_threshold": float(best_thr),
        "num_trades": int(trades.shape[0]),
        "selected_share": float((trades_mask.mean() if len(trades_mask) else 0.0)),
        "avg_pnl_per_trade_bps": float(trades["pnl_bps"].mean() if not trades.empty else 0.0),
        "total_pnl_bps": float(trades["pnl_bps"].sum() if not trades.empty else 0.0),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    return trades, best_thr, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trading simulation on held-out test window")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to parquet dataset (built by scripts/build_dataset.py)",
    )
    parser.add_argument("--horizon", type=float, required=True, help="Prediction horizon in seconds")
    parser.add_argument("--comm_bps", type=float, default=8.0, help="Round-trip commission in bps")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["long", "ls"],
        default="long",
        help="Signal mode: long-only ('long') or long-short symmetric ('ls')",
    )
    parser.add_argument("--thr", type=float, default=None, help="Classification threshold (probability)")
    parser.add_argument("--tune", action="store_true", help="Enable threshold tuning on validation to maximize PnL")
    parser.add_argument("--th_min", type=float, default=0.50, help="Min threshold for tuning grid")
    parser.add_argument("--th_max", type=float, default=0.90, help="Max threshold for tuning grid")
    parser.add_argument("--th_steps", type=int, default=21, help="Number of thresholds in tuning grid")
    parser.add_argument("--val_days", type=int, default=3, help="Days for validation window")
    parser.add_argument("--test_days", type=int, default=3, help="Days for test window")
    parser.add_argument("--n_jobs", type=int, default=4, help="Parallel threads for XGBoost")
    # Entry filters
    parser.add_argument("--max_spread_bps", type=float, default=None, help="Max spread in bps for entry")
    parser.add_argument("--min_abs_ofi", type=float, default=None, help="Min absolute OFI for entry")
    parser.add_argument("--min_volume_total", type=float, default=None, help="Min total volume for entry")
    parser.add_argument("--min_trade_count", type=int, default=None, help="Min trade count for entry")
    parser.add_argument(
        "--out_trades",
        type=str,
        default="data/ml/paper_trades.csv",
        help="Where to save per-trade log CSV (test window only)",
    )

    args = parser.parse_args()

    df = pd.read_parquet(args.data).dropna().sort_values("timestamp")

    trades, used_thr, summary = train_and_simulate(
        df=df,
        horizon_seconds=args.horizon,
        commission_bps_roundtrip=args.comm_bps,
        n_jobs=args.n_jobs,
        mode=args.mode,
        threshold=args.thr,
        tune_threshold=bool(args.tune),
        tune_min=args.th_min,
        tune_max=args.th_max,
        tune_steps=args.th_steps,
        val_days=args.val_days,
        test_days=args.test_days,
        max_spread_bps=args.max_spread_bps,
        min_abs_ofi=args.min_abs_ofi,
        min_volume_total=args.min_volume_total,
        min_trade_count=args.min_trade_count,
    )

    trades.to_csv(args.out_trades, index=False)

    print(
        (
            f"Saved trades to {args.out_trades}\n"
            f"h={args.horizon}s thr={used_thr:.2f} mode={args.mode} comm={args.comm_bps} bps\n"
            f"trades={summary['num_trades']} selected={summary['selected_share']:.1%} "
            f"avg_pnl_per_trade={summary['avg_pnl_per_trade_bps']:.2f} bps total_pnl={summary['total_pnl_bps']:.2f} bps\n"
            f"precision={summary['precision']:.3f} recall={summary['recall']:.3f} f1={summary['f1']:.3f}"
        )
    )


if __name__ == "__main__":
    main()


