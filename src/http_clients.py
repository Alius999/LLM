import logging
from typing import Any, Dict, List, Optional, Tuple

import requests


_logger = logging.getLogger(__name__)


BASE_HTTP = "https://api.bybit.com"


def fetch_orderbook_snapshot(category: str, symbol: str, depth: int = 200) -> Optional[Tuple[int, List[List[str]], List[List[str]]]]:
    """Fetch orderbook snapshot via HTTP. Returns (ts_ms, bids, asks) or None on error.

    Bids/asks are arrays of [price_str, size_str].
    """
    try:
        params = {"category": category, "symbol": symbol, "limit": depth}
        r = requests.get(f"{BASE_HTTP}/v5/market/orderbook", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        result = data.get("result") or {}
        ts = int(result.get("ts") or data.get("time", 0))
        bids = result.get("b") or []
        asks = result.get("a") or []
        if not isinstance(bids, list) or not isinstance(asks, list):
            return None
        return ts, bids, asks
    except Exception as exc:  # noqa: BLE001
        _logger.warning("HTTP snapshot failed for %s %s: %s", category, symbol, exc, exc_info=False)
        return None


def fetch_derivatives_ticker(category: str, symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch market tickers for derivatives to extract open interest and funding rate if available."""
    try:
        params = {"category": category, "symbol": symbol}
        r = requests.get(f"{BASE_HTTP}/v5/market/tickers", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        result = (data.get("result") or {}).get("list") or []
        if not result:
            return None
        return result[0]
    except Exception as exc:  # noqa: BLE001
        _logger.debug("Derivatives ticker fetch failed for %s %s: %s", category, symbol, exc, exc_info=False)
        return None


