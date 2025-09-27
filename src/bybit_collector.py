import asyncio
import json
import logging
import time
from typing import Dict, List, Optional

import websockets

from db.database import DatabaseClient
from src.feature_aggregator import FeatureAggregator
from src.http_clients import fetch_orderbook_snapshot, fetch_derivatives_ticker


class BybitDataCollector:
    """Collects real-time data from Bybit public WebSocket and stores it to DB.

    Subscribes to order book depth (orderbook.N), public trades (publicTrade), and tickers (tickers)
    for the provided symbols. Automatically reconnects on failures.
    """

    CATEGORY_TO_URL = {
        "spot": "wss://stream.bybit.com/v5/public/spot",
        "linear": "wss://stream.bybit.com/v5/public/linear",
        "inverse": "wss://stream.bybit.com/v5/public/inverse",
        "option": "wss://stream.bybit.com/v5/public/option",
    }

    def __init__(
        self,
        symbols: List[str],
        db_client: DatabaseClient,
        category: str = "spot",
        depth_level: int = 50,
        reconnect_delay_seconds: float = 3.0,
    ) -> None:
        if category not in self.CATEGORY_TO_URL:
            raise ValueError(f"Unsupported category: {category}")

        self._symbols = [s.upper() for s in symbols]
        self._db = db_client
        self._category = category
        self._depth_level = depth_level
        self._url = self.CATEGORY_TO_URL[category]
        self._reconnect_delay_seconds = reconnect_delay_seconds
        self._running = False

        self._logger = logging.getLogger(self.__class__.__name__)
        self._orderbook_saved = 0
        self._trades_saved = 0
        self._ticker_saved = 0
        self._features = FeatureAggregator(self._db)
        self._last_features_flush_sec: Dict[str, int] = {s: 0 for s in self._symbols}
        # Periodic reconciliation via HTTP snapshots (ms)
        self._snapshot_interval_ms = 30_000
        self._last_snapshot_ms: Dict[str, int] = {s: 0 for s in self._symbols}
        # Periodic derivatives metrics (ms)
        self._deriv_metrics_interval_ms = 60_000
        self._last_deriv_metrics_ms: Dict[str, int] = {s: 0 for s in self._symbols}

    def _build_subscription_args(self) -> List[str]:
        args: List[str] = []
        for symbol in self._symbols:
            args.append(f"orderbook.{self._depth_level}.{symbol}")
            args.append(f"publicTrade.{symbol}")
            args.append(f"tickers.{symbol}")
        return args

    async def start(self) -> None:
        self._running = True
        while self._running:
            try:
                await self._connect_and_consume()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - we log and retry
                self._logger.error("WebSocket error: %s", exc, exc_info=False)
                await asyncio.sleep(self._reconnect_delay_seconds)

    async def _connect_and_consume(self) -> None:
        self._logger.info("Connecting to %s (%s)...", self._url, self._category)
        async with websockets.connect(
            self._url,
            ping_interval=20,
            ping_timeout=10,
            max_size=8 * 1024 * 1024,  # 8MB to be safe on snapshots
        ) as ws:
            self._logger.info("Connected. Subscribing to %d topics for %s", len(self._symbols) * 3, ", ".join(self._symbols))
            subscribe_msg = {"op": "subscribe", "args": self._build_subscription_args()}
            await ws.send(json.dumps(subscribe_msg))

            # Consume messages until disconnect
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    self._logger.debug("Non-JSON message: %s", raw)
                    continue

                # Handle pings or system events
                if isinstance(msg, dict) and msg.get("op") == "ping":
                    await ws.send(json.dumps({"op": "pong"}))
                    continue

                # Ignore subscription acks
                if msg.get("success") is True and msg.get("request", {}).get("op") == "subscribe":
                    continue

                await self._handle_message(msg)

                # Periodic per-second features aggregation per symbol
                now_ms = int(time.time() * 1000)
                now_sec_ms = (now_ms // 1000) * 1000
                for sym in self._symbols:
                    if self._last_features_flush_sec.get(sym, 0) < now_sec_ms:
                        # Aggregate the previous second fully closed
                        sec_to_aggregate = now_sec_ms - 1000
                        if sec_to_aggregate > 0:
                            try:
                                self._features.aggregate_second(sym, sec_to_aggregate)
                            except Exception as exc:  # noqa: BLE001
                                self._logger.debug("Feature aggregation failed for %s: %s", sym, exc)
                        self._last_features_flush_sec[sym] = now_sec_ms

                # Periodic orderbook snapshot reconciliation
                for sym in self._symbols:
                    if now_ms - self._last_snapshot_ms.get(sym, 0) >= self._snapshot_interval_ms:
                        snap = fetch_orderbook_snapshot(self._category, sym, depth=self._depth_level)
                        if snap is not None:
                            ts, bids, asks = snap
                            try:
                                loop = asyncio.get_running_loop()
                                await loop.run_in_executor(
                                    None,
                                    self._db.insert_orderbook_update,
                                    ts,
                                    sym,
                                    bids,
                                    asks,
                                )
                            except Exception as exc:  # noqa: BLE001
                                self._logger.debug("Snapshot insert failed for %s: %s", sym, exc)
                        self._last_snapshot_ms[sym] = now_ms

                # Optional derivatives metrics for linear/inverse
                if self._category in ("linear", "inverse"):
                    for sym in self._symbols:
                        if now_ms - self._last_deriv_metrics_ms.get(sym, 0) >= self._deriv_metrics_interval_ms:
                            data = fetch_derivatives_ticker(self._category, sym)
                            if data:
                                ts = int(data.get("ts", now_ms))
                                # Keys may differ across instruments; best-effort parse
                                fr = data.get("fundingRate")
                                oi = data.get("openInterest") or data.get("openInterestValue")
                                try:
                                    funding = float(fr) if fr is not None else None
                                except Exception:
                                    funding = None
                                try:
                                    open_interest = float(oi) if oi is not None else None
                                except Exception:
                                    open_interest = None
                                try:
                                    loop = asyncio.get_running_loop()
                                    await loop.run_in_executor(
                                        None,
                                        self._db.insert_derivatives_metrics,
                                        ts,
                                        sym,
                                        funding,
                                        open_interest,
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    self._logger.debug("Derivatives metrics insert failed for %s: %s", sym, exc)
                            self._last_deriv_metrics_ms[sym] = now_ms

    async def _handle_message(self, msg: Dict[str, any]) -> None:
        topic: Optional[str] = msg.get("topic")
        if not topic:
            # Some system messages don't include a topic
            return

        try:
            if topic.startswith("orderbook."):
                await self._handle_orderbook_message(msg)
            elif topic.startswith("publicTrade."):
                await self._handle_trades_message(msg)
            elif topic.startswith("tickers."):
                await self._handle_ticker_message(msg)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to handle message for %s: %s", topic, exc, exc_info=False)

    async def _handle_orderbook_message(self, msg: Dict[str, any]) -> None:
        topic = msg["topic"]  # e.g., orderbook.50.BTCUSDT
        symbol = topic.split(".")[-1]
        timestamp_ms = int(msg.get("ts", int(time.time() * 1000)))
        data = msg.get("data", {})
        # data may be snapshot or delta; both include 'b' and 'a' arrays
        bids = data.get("b", [])
        asks = data.get("a", [])

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._db.insert_orderbook_update,
            timestamp_ms,
            symbol,
            bids,
            asks,
        )
        self._orderbook_saved += 1
        if self._orderbook_saved % 100 == 0:
            self._logger.info("OrderBook updates saved: %d", self._orderbook_saved)

    async def _handle_trades_message(self, msg: Dict[str, any]) -> None:
        topic = msg["topic"]  # publicTrade.BTCUSDT
        symbol = topic.split(".")[-1]
        data_list = msg.get("data", [])
        loop = asyncio.get_running_loop()
        for t in data_list:
            # Bybit v5 trade fields: T (ts ms), S (side), p (price str), v (size str)
            timestamp_ms = int(t.get("T", int(time.time() * 1000)))
            side = str(t.get("S", "")).upper() or "UNKNOWN"
            try:
                price = float(t.get("p"))
            except Exception:
                price = None
            try:
                size = float(t.get("v"))
            except Exception:
                size = None
            if price is None or size is None:
                continue

            await loop.run_in_executor(
                None,
                self._db.insert_trade,
                timestamp_ms,
                symbol,
                price,
                size,
                side,
            )
            self._trades_saved += 1
        if self._trades_saved and self._trades_saved % 200 == 0:
            self._logger.info("Trades saved: %d", self._trades_saved)

    async def _handle_ticker_message(self, msg: Dict[str, any]) -> None:
        topic = msg["topic"]  # tickers.BTCUSDT
        symbol = topic.split(".")[-1]
        data = msg.get("data", {})
        timestamp_ms = int(data.get("ts", msg.get("ts", int(time.time() * 1000))))
        bid_str = data.get("bid1Price")
        ask_str = data.get("ask1Price")
        bid = float(bid_str) if bid_str is not None else None
        ask = float(ask_str) if ask_str is not None else None

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._db.insert_ticker,
            timestamp_ms,
            symbol,
            bid,
            ask,
        )
        self._ticker_saved += 1
        if self._ticker_saved % 100 == 0:
            self._logger.info("Ticker updates saved: %d", self._ticker_saved)

    def stop(self) -> None:
        self._running = False
        self._logger.info(
            "Stopping collector. Totals — orderbook: %d, trades: %d, ticker: %d",
            self._orderbook_saved,
            self._trades_saved,
            self._ticker_saved,
        )


