import json
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class DatabaseClient(ABC):
    """Abstract database client interface to allow easy swapping of backends."""

    @abstractmethod
    def insert_orderbook_update(self, timestamp_ms: int, symbol: str, bids: List[List[str]], asks: List[List[str]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert_trade(self, timestamp_ms: int, symbol: str, price: float, size: float, side: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert_ticker(self, timestamp_ms: int, symbol: str, bid: Optional[float], ask: Optional[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    # Reads for feature aggregation
    @abstractmethod
    def get_trades_between(self, symbol: str, start_ms: int, end_ms: int) -> List[Tuple[int, float, float, str]]:
        """Return list of (timestamp, price, size, side) for trades in [start_ms, end_ms]."""
        raise NotImplementedError

    @abstractmethod
    def get_latest_ticker(self, symbol: str, since_ms: int) -> Optional[Tuple[int, Optional[float], Optional[float]]]:
        """Return latest (timestamp, bid, ask) with timestamp >= since_ms, else None."""
        raise NotImplementedError

    @abstractmethod
    def insert_features_1s(
        self,
        timestamp_ms: int,
        symbol: str,
        mid: Optional[float],
        spread: Optional[float],
        trade_count: int,
        volume_total: float,
        volume_buy: float,
        volume_sell: float,
        order_flow_imbalance: Optional[float],
        vwap: Optional[float],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert_derivatives_metrics(
        self,
        timestamp_ms: int,
        symbol: str,
        funding_rate: Optional[float],
        open_interest: Optional[float],
    ) -> None:
        raise NotImplementedError


class SQLiteClient(DatabaseClient):
    """SQLite implementation of DatabaseClient.

    Notes:
        - Uses a thread-safe lock since sqlite3 connections are not thread-safe by default.
        - PRAGMA tuned conservatively for reliability over speed.
    """

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL;")
        self._connection.execute("PRAGMA synchronous=NORMAL;")
        self._connection.execute("PRAGMA temp_store=MEMORY;")
        self._connection.execute("PRAGMA foreign_keys=ON;")
        self._connection.execute("PRAGMA busy_timeout=5000;")
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock:
            cur = self._connection.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS orderbook_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    bids TEXT NOT NULL,
                    asks TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    side TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ticker (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    bid REAL,
                    ask REAL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS features_1s (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    mid REAL,
                    spread REAL,
                    trade_count INTEGER NOT NULL,
                    volume_total REAL NOT NULL,
                    volume_buy REAL NOT NULL,
                    volume_sell REAL NOT NULL,
                    order_flow_imbalance REAL,
                    vwap REAL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS derivatives_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    funding_rate REAL,
                    open_interest REAL
                );
                """
            )
            # Indices for faster time-range queries per symbol
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades(symbol, timestamp);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ticker_symbol_ts ON ticker(symbol, timestamp);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ob_symbol_ts ON orderbook_updates(symbol, timestamp);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_feat_symbol_ts ON features_1s(symbol, timestamp);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_deriv_symbol_ts ON derivatives_metrics(symbol, timestamp);")
            self._connection.commit()

    def insert_orderbook_update(self, timestamp_ms: int, symbol: str, bids: List[List[str]], asks: List[List[str]]) -> None:
        bids_json = json.dumps(bids, separators=(",", ":"))
        asks_json = json.dumps(asks, separators=(",", ":"))
        with self._lock:
            self._connection.execute(
                "INSERT INTO orderbook_updates (timestamp, symbol, bids, asks) VALUES (?, ?, ?, ?);",
                (timestamp_ms, symbol, bids_json, asks_json),
            )
            self._connection.commit()

    def insert_trade(self, timestamp_ms: int, symbol: str, price: float, size: float, side: str) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO trades (timestamp, symbol, price, size, side) VALUES (?, ?, ?, ?, ?);",
                (timestamp_ms, symbol, price, size, side),
            )
            self._connection.commit()

    def insert_ticker(self, timestamp_ms: int, symbol: str, bid: Optional[float], ask: Optional[float]) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO ticker (timestamp, symbol, bid, ask) VALUES (?, ?, ?, ?);",
                (timestamp_ms, symbol, bid, ask),
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # Reads for feature aggregation
    def get_trades_between(self, symbol: str, start_ms: int, end_ms: int) -> List[Tuple[int, float, float, str]]:
        with self._lock:
            cur = self._connection.cursor()
            cur.execute(
                "SELECT timestamp, price, size, side FROM trades WHERE symbol=? AND timestamp>=? AND timestamp<=? ORDER BY timestamp ASC;",
                (symbol, start_ms, end_ms),
            )
            rows = cur.fetchall()
        return [(int(ts), float(price), float(size), str(side)) for ts, price, size, side in rows]

    def get_latest_ticker(self, symbol: str, since_ms: int) -> Optional[Tuple[int, Optional[float], Optional[float]]]:
        with self._lock:
            cur = self._connection.cursor()
            cur.execute(
                "SELECT timestamp, bid, ask FROM ticker WHERE symbol=? AND timestamp>=? ORDER BY timestamp DESC LIMIT 1;",
                (symbol, since_ms),
            )
            row = cur.fetchone()
        if not row:
            return None
        ts, bid, ask = row
        return int(ts), (float(bid) if bid is not None else None), (float(ask) if ask is not None else None)

    def insert_features_1s(
        self,
        timestamp_ms: int,
        symbol: str,
        mid: Optional[float],
        spread: Optional[float],
        trade_count: int,
        volume_total: float,
        volume_buy: float,
        volume_sell: float,
        order_flow_imbalance: Optional[float],
        vwap: Optional[float],
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO features_1s (
                    timestamp, symbol, mid, spread, trade_count, volume_total, volume_buy, volume_sell, order_flow_imbalance, vwap
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    timestamp_ms,
                    symbol,
                    mid,
                    spread,
                    trade_count,
                    volume_total,
                    volume_buy,
                    volume_sell,
                    order_flow_imbalance,
                    vwap,
                ),
            )
            self._connection.commit()

    def insert_derivatives_metrics(
        self,
        timestamp_ms: int,
        symbol: str,
        funding_rate: Optional[float],
        open_interest: Optional[float],
    ) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO derivatives_metrics (timestamp, symbol, funding_rate, open_interest) VALUES (?, ?, ?, ?);",
                (timestamp_ms, symbol, funding_rate, open_interest),
            )
            self._connection.commit()


