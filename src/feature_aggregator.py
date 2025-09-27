import logging
import time
from typing import Optional

from db.database import DatabaseClient


class FeatureAggregator:
    """Builds 1-second features for each symbol from raw trades and ticker.

    Features:
      - mid, spread from latest best bid/ask within the second
      - trade_count, volume_total, volume_buy, volume_sell
      - order_flow_imbalance = (volume_buy - volume_sell) / max(volume_total, 1e-9)
      - vwap over trades within the second
    """

    def __init__(self, db: DatabaseClient) -> None:
        self._db = db
        self._logger = logging.getLogger(self.__class__.__name__)

    def aggregate_second(self, symbol: str, second_epoch_ms: int) -> None:
        # Define 1-second window [t, t+999]
        start_ms = second_epoch_ms
        end_ms = second_epoch_ms + 999

        trades = self._db.get_trades_between(symbol, start_ms, end_ms)
        trade_count = len(trades)
        volume_buy = 0.0
        volume_sell = 0.0
        volume_total = 0.0
        vwap_numerator = 0.0

        for ts, price, size, side in trades:
            volume_total += size
            vwap_numerator += price * size
            if side == "BUY":
                volume_buy += size
            elif side == "SELL":
                volume_sell += size

        vwap: Optional[float] = (vwap_numerator / volume_total) if volume_total > 0 else None

        ticker = self._db.get_latest_ticker(symbol, start_ms)
        if ticker is not None:
            ts, bid, ask = ticker
            if bid is not None and ask is not None and bid > 0 and ask > 0:
                mid = 0.5 * (bid + ask)
                spread = ask - bid
            else:
                mid = None
                spread = None
        else:
            mid = None
            spread = None

        if volume_total > 0:
            order_flow_imbalance = (volume_buy - volume_sell) / max(volume_total, 1e-9)
        else:
            order_flow_imbalance = None

        self._db.insert_features_1s(
            timestamp_ms=second_epoch_ms,
            symbol=symbol,
            mid=mid,
            spread=spread,
            trade_count=trade_count,
            volume_total=volume_total,
            volume_buy=volume_buy,
            volume_sell=volume_sell,
            order_flow_imbalance=order_flow_imbalance,
            vwap=vwap,
        )


