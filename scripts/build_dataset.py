#!/usr/bin/env python
import argparse
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd


def to_ms(dt_str: str) -> int:
    # Accept ISO like '2025-09-27 00:00:00' in UTC by default
    dt = pd.to_datetime(dt_str, utc=True)
    return int(dt.timestamp() * 1000)


def build_dataset(
    db_path: str,
    out_path: str,
    start: str | None,
    end: str | None,
    symbols: list[str] | None,
    deadzone_bps: float = 0.0,
    horizons: list[int] | None = None,
) -> None:
    conn = sqlite3.connect(db_path)
    where = []
    params: list[object] = []

    if start:
        where.append("timestamp >= ?")
        params.append(to_ms(start))
    if end:
        where.append("timestamp <= ?")
        params.append(to_ms(end))
    if symbols:
        where.append("symbol IN (%s)" % ",".join(["?"] * len(symbols)))
        params.extend(symbols)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    # Pull 1s features
    query = f"""
        SELECT timestamp, symbol, mid, spread, trade_count, volume_total, volume_buy, volume_sell,
               order_flow_imbalance AS ofi, vwap
        FROM features_1s
        {where_sql}
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        raise SystemExit("No rows in selected range. Adjust --start/--end or symbols.")

    # Targets: sign of future mid change at requested horizons
    if not horizons:
        horizons = [1.0, 3.0, 5.0]

    int_h = sorted({int(h) for h in horizons if float(h).is_integer()})
    frac_h = sorted({float(h) for h in horizons if not float(h).is_integer()})

    # Integer-second horizons from features_1s via shift
    for horizon_s in int_h:
        future_mid = df.groupby("symbol")["mid"].shift(-int(horizon_s))
        dmid = future_mid - df["mid"]
        if deadzone_bps > 0:
            threshold = (deadzone_bps / 1e4) * df["mid"].abs()
            label = dmid.apply(lambda x: 0 if x is None else (1 if x > 0 else (-1 if x < 0 else 0)))
            label[(dmid.abs() < threshold)] = 0
        else:
            label = dmid.apply(lambda x: 1 if x is not None and x > 0 else (-1 if x is not None and x < 0 else 0))
        df[f"target_sign_dmid_{horizon_s}s"] = label

    # Fractional-second horizons (e.g., 0.5s) via forward join to ticker mid
    if frac_h:
        # Load tickers once
        t_where = []
        t_params: list[object] = []
        if symbols:
            t_where.append("symbol IN (%s)" % ",".join(["?"] * len(symbols)))
            t_params.extend(symbols)
        t_where_sql = ("WHERE " + " AND ".join(t_where)) if t_where else ""
        t_query = f"""
            SELECT timestamp, symbol, bid, ask
            FROM ticker
            {t_where_sql}
            ORDER BY symbol ASC, timestamp ASC
        """
        tdf = pd.read_sql_query(t_query, conn, params=t_params)
        if not tdf.empty:
            tdf["mid_t"] = (tdf["bid"].astype(float) + tdf["ask"].astype(float)) / 2.0
            # Ensure sorted for merge_asof
            tdf = tdf.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

            for h in frac_h:
                future_times = df["timestamp"].astype("int64") + int(h * 1000)
                left = pd.DataFrame({
                    "symbol": df["symbol"].values,
                    "ts_future": future_times.values,
                })
                # merge_asof forward within symbol groups
                merged = pd.merge_asof(
                    left.sort_values(["symbol", "ts_future"]),
                    tdf.sort_values(["symbol", "timestamp"]),
                    left_on="ts_future",
                    right_on="timestamp",
                    by="symbol",
                    direction="forward",
                    allow_exact_matches=True,
                )
                future_mid = merged["mid_t"].reindex_like(df)
                dmid = future_mid.values - df["mid"].values
                if deadzone_bps > 0:
                    threshold = (deadzone_bps / 1e4) * df["mid"].abs().values
                    label = np.where(np.abs(dmid) < threshold, 0, np.where(dmid > 0, 1, -1))
                else:
                    label = np.where(dmid > 0, 1, np.where(dmid < 0, -1, 0))
                df[f"target_sign_dmid_{h}s"] = label

    # Drop rows with NaN mids only
    df = df.dropna(subset=["mid"]).reset_index(drop=True)

    # Rolling features (3s, 5s, 10s)
    for w in (3, 5, 10, 15, 30, 60):
        grp = df.groupby("symbol")
        df[f"mid_ema_{w}s"] = grp["mid"].transform(lambda s: s.ewm(span=w, adjust=False).mean())
        df[f"mid_ret_{w}s"] = grp["mid"].transform(lambda s: s.pct_change(periods=w, fill_method=None))
        df[f"vola_{w}s"] = grp["mid"].transform(lambda s: s.pct_change(fill_method=None).rolling(w).std())
        df[f"ofi_sum_{w}s"] = grp["ofi"].transform(lambda s: s.rolling(w, min_periods=1).sum())
        df[f"tc_sum_{w}s"] = grp["trade_count"].transform(lambda s: s.rolling(w, min_periods=1).sum())

    # Drop tail NaNs only for existing target columns
    label_cols = [f"target_sign_dmid_{h}s" for h in horizons]
    label_cols = [c for c in label_cols if c in df.columns]
    if label_cols:
        df = df.dropna(subset=label_cols)

    # Save to parquet
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved dataset: {out_path} rows={len(df)} symbols={sorted(df['symbol'].unique())}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ML dataset from features_1s")
    ap.add_argument("--db", default="data/bybit_linear.sqlite3", help="Path to SQLite database")
    ap.add_argument("--out", default="data/ml/dataset.parquet", help="Output parquet path")
    ap.add_argument("--start", default=None, help="Start time (e.g., 2025-09-27 00:00:00 UTC)")
    ap.add_argument("--end", default=None, help="End time (UTC)")
    ap.add_argument("--symbols", default=None, help="Comma-separated symbols filter")
    ap.add_argument("--deadzone_bps", type=float, default=0.0, help="Deadzone threshold in bps for 0-class (e.g., 1.0)")
    ap.add_argument("--horizons", default="1,3,5", help="Comma-separated horizons in seconds, supports floats like '0.5,1,2,3'")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    horizons = [float(x) for x in args.horizons.split(",") if x]
    build_dataset(
        args.db,
        args.out,
        args.start,
        args.end,
        symbols,
        deadzone_bps=float(args.deadzone_bps),
        horizons=horizons,
    )


if __name__ == "__main__":
    main()


