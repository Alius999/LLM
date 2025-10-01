#!/usr/bin/env python
import argparse
import os
import sqlite3
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def to_ms(s: str) -> int:
    return int(pd.to_datetime(s, utc=True).timestamp() * 1000)


def load_dataset(db_path: str, start: str, end: str, symbols: list[str] | None, deadzone_bps: float) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    where = []
    params: list[object] = []
    if start:
        where.append("timestamp >= ?"); params.append(to_ms(start))
    if end:
        where.append("timestamp <= ?"); params.append(to_ms(end))
    if symbols:
        where.append("symbol IN (%s)" % ",".join(["?"] * len(symbols)))
        params.extend(symbols)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    q = f"""
    SELECT timestamp, symbol, mid, spread, trade_count, volume_total, volume_buy, volume_sell, order_flow_imbalance AS ofi, vwap
    FROM features_1s
    {where_sql}
    ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(q, conn, params=params)

    # rolling features (compute in-memory; DB не содержит этих колонок)
    for w in (3, 5, 10):
        grp = df.groupby("symbol")
        df[f"mid_ema_{w}s"] = grp["mid"].transform(lambda s: s.ewm(span=w, adjust=False).mean())
        df[f"mid_ret_{w}s"] = grp["mid"].transform(lambda s: s.pct_change(periods=w))
        df[f"vola_{w}s"] = grp["mid"].transform(lambda s: s.pct_change().rolling(w).std())
        df[f"ofi_sum_{w}s"] = grp["ofi"].transform(lambda s: s.rolling(w, min_periods=1).sum())
        df[f"tc_sum_{w}s"] = grp["trade_count"].transform(lambda s: s.rolling(w, min_periods=1).sum())

    # build targets with deadzone
    for horizon_s in (1, 3, 5):
        future_mid = df.groupby("symbol")["mid"].shift(-horizon_s)
        dmid = future_mid - df["mid"]
        thr = (deadzone_bps / 1e4) * df["mid"].abs() if deadzone_bps > 0 else 0.0
        label = np.where(dmid > thr, 1, np.where(dmid < -thr, -1, 0))
        df[f"target_sign_dmid_{horizon_s}s"] = label

    return df.dropna().reset_index(drop=True)


def split_time(df: pd.DataFrame, test_days: int = 1, val_days: int = 1):
    df = df.sort_values("timestamp")
    max_ts = df["timestamp"].max()
    day_ms = 24 * 3600 * 1000
    test_start = max_ts - test_days * day_ms
    val_start = test_start - val_days * day_ms
    return (
        df[df["timestamp"] < val_start],
        df[(df["timestamp"] >= val_start) & (df["timestamp"] < test_start)],
        df[df["timestamp"] >= test_start],
    )


def evaluate_binary(df: pd.DataFrame, horizon: int) -> dict:
    ycol = f"target_sign_dmid_{horizon}s"
    df = df[df[ycol] != 0].copy()
    y = (df[ycol] == 1).astype(int)
    feats = [
        "mid","spread","trade_count","volume_total","volume_buy","volume_sell","ofi","vwap",
        "mid_ema_3s","mid_ema_5s","mid_ema_10s",
        "mid_ret_3s","mid_ret_5s","mid_ret_10s",
        "vola_3s","vola_5s","vola_10s",
        "ofi_sum_3s","ofi_sum_5s","ofi_sum_10s",
        "tc_sum_3s","tc_sum_5s","tc_sum_10s",
    ]
    feats = [c for c in feats if c in df.columns]
    X = df[feats].fillna(0)
    tr, va, te = split_time(df)
    Xtr, ytr = X.loc[tr.index], y.loc[tr.index]
    Xva, yva = X.loc[va.index], y.loc[va.index]
    Xte, yte = X.loc[te.index], y.loc[te.index]

    pos = (ytr == 1).sum(); neg = (ytr == 0).sum(); spw = (neg / max(pos, 1)) if neg > 0 else 1.0
    xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9,
                        objective="binary:logistic", eval_metric="logloss",
                        scale_pos_weight=spw, n_jobs=2)
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    yhat = xgb.predict(Xte)
    rep = classification_report(yte, yhat, output_dict=True)
    return {
        "accuracy": rep["accuracy"],
        "precision_pos": rep["1"]["precision"],
        "recall_pos": rep["1"]["recall"],
        "f1_pos": rep["1"]["f1-score"],
        "support": int(rep["1"]["support"] + rep["0"]["support"]),
    }


def main():
    ap = argparse.ArgumentParser(description="Grid search deadzone/horizon/symbols")
    ap.add_argument("--db", default="data/bybit_linear.sqlite3")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--deadzone", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--horizons", default="1,3,5")
    ap.add_argument("--symbols", default="ALL", help="CSV or ALL")
    args = ap.parse_args()

    # determine symbols
    if args.symbols.upper() == "ALL":
        conn = sqlite3.connect(args.db)
        sym = pd.read_sql_query("SELECT DISTINCT symbol FROM features_1s", conn)["symbol"].tolist()
    else:
        sym = [s.strip().upper() for s in args.symbols.split(",")]

    rows = []
    for dz, hz, syms in product([float(x) for x in args.deadzone.split(",")], [int(x) for x in args.horizons.split(",")], [sym, ["BTCUSDT"], ["ETHUSDT"]]):
        df = load_dataset(args.db, args.start, args.end, syms if syms != sym else None, dz)
        if df.empty:
            continue
        metrics = evaluate_binary(df, hz)
        rows.append({
            "deadzone_bps": dz,
            "horizon_s": hz,
            "symbols": ",".join(syms) if syms != sym else "ALL",
            **metrics,
        })

    out = pd.DataFrame(rows).sort_values(["accuracy","f1_pos"], ascending=False)
    os.makedirs("data/ml", exist_ok=True)
    out_path = "data/ml/grid_results.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved grid results to {out_path}\nTop 10:\n", out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()


