## Bybit Real-time Data Collector (Order Book, Trades, Ticker)

This Python project collects real-time market data from Bybit WebSocket API and stores it in a local SQLite database. It is designed with a modular architecture and can be extended to other databases (e.g., PostgreSQL) by implementing the same database interface.

### Features
- Subscribe to Order Book depth updates, Trades, and Ticker in real time
- Handle multiple symbols concurrently (e.g., `BTCUSDT`, `ETHUSDT`)
- Robust auto-reconnection on network errors
- Structured logging to console
- Periodic HTTP orderbook snapshot reconciliation
- Optional derivatives metrics (funding rate, open interest) for `linear`/`inverse`
- 1-second feature aggregation (`features_1s`): mid, spread, trade counts, volumes, OFI, VWAP
- Simple storage to SQLite with tables:
  - `orderbook_updates (timestamp, symbol, bids, asks)`
  - `trades (timestamp, symbol, price, size, side)`
  - `ticker (timestamp, symbol, bid, ask)`
  - `features_1s (timestamp, symbol, mid, spread, trade_count, volume_total, volume_buy, volume_sell, order_flow_imbalance, vwap)`
  - `derivatives_metrics (timestamp, symbol, funding_rate, open_interest)`

### Project Structure
```
.
├── db/
│   ├── __init__.py
│   └── database.py
├── src/
│   ├── __init__.py
│   └── bybit_collector.py
├── utils/
│   ├── __init__.py
│   └── logging_setup.py
├── data/               # SQLite database file will be created here
├── main.py
├── requirements.txt
└── README.md
```

### Install
```bash
python -m pip install -r requirements.txt
```

### Run
```bash
python main.py
```

By default, it will connect to Bybit Spot public WebSocket (`wss://stream.bybit.com/v5/public/spot`) and subscribe to `orderbook.200`, `publicTrade`, and `tickers` for the symbols specified in `main.py`.

### Configure Symbols / Market
- Edit `main.py` to adjust:
  - `symbols = ["BTCUSDT", "ETHUSDT"]`
  - `category = "spot"` (alternatives: `linear`, `inverse`, `option`, depending on your needs)
  - `depth_level = 200` for order book depth (50/100/200 supported)

### Database Schema
- Implemented in `db/database.py`. To switch to PostgreSQL later, implement a new client class with the same interface as `DatabaseClient` and use it in `main.py`.

### Notes
- Only widely-used libraries are used: `websockets` and standard library modules `sqlite3`, `json`, `logging`, `asyncio`.
- Data volume can be high for order book updates. SQLite settings are tuned for reliability; for production-grade throughput, consider batching or PostgreSQL.

### ML (optional)
- Install extra deps (separately from runtime):
```bash
python -m pip install -r requirements-ml.txt
```
- Build dataset from `features_1s` with targets (example: last 48h, all symbols):
```bash
python scripts/build_dataset.py --db data/bybit_linear.sqlite3 --out data/ml/dataset.parquet --start "2025-09-28 00:00:00" --end "2025-09-30 00:00:00"
```
- Train baselines (logreg + XGBoost) for 1s horizon:
```bash
python scripts/train_baseline.py --data data/ml/dataset.parquet --out data/ml/models --horizon 1
```


# LLM
