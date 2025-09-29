import asyncio
import logging
import os
import signal
from typing import List

from db.database import SQLiteClient
from src.bybit_collector import BybitDataCollector
from utils.logging_setup import setup_logging


def _chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


async def run(symbols: List[str], category: str = "spot", db_path: str = "data/bybit_data.sqlite3") -> None:
    setup_logging(logging.INFO)
    logger = logging.getLogger("main")

    db = SQLiteClient(db_path)

    # Split symbols across multiple WS connections to improve stability
    max_symbols_per_ws = int(os.environ.get("MAX_SYMBOLS_PER_WS", "5"))
    symbol_groups = _chunk_list(symbols, max_symbols_per_ws) if symbols else []
    collectors = [
        BybitDataCollector(symbols=group, db_client=db, category=category, depth_level=200)
        for group in symbol_groups
    ]

    loop = asyncio.get_running_loop()

    stop_event = asyncio.Event()

    def _handle_stop_signal(*_: object) -> None:
        logger.info("Received stop signal. Shutting down...")
        collector.stop()
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _handle_stop_signal)
        loop.add_signal_handler(signal.SIGTERM, _handle_stop_signal)
    except NotImplementedError:
        # Signal handlers may not be available on some platforms (e.g., Windows)
        pass

    logger.info(
        "Starting %d WS workers (max %d symbols/WS) for symbols: %s (category=%s)",
        len(collectors),
        max_symbols_per_ws,
        ", ".join(symbols),
        category,
    )
    producers = [asyncio.create_task(c.start()) for c in collectors]

    try:
        # Wait until stop_event is set
        await stop_event.wait()
    except asyncio.CancelledError:
        # Loop shutdown (e.g., Ctrl+C). Proceed to graceful cleanup.
        logger.info("Cancellation received. Shutting down...")
    finally:
        for p in producers:
            if not p.done():
                p.cancel()
        for p in producers:
            try:
                await p
            except asyncio.CancelledError:
                pass
        db.close()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    # Configure symbols and market category here
    symbols = [
        "BTCUSDT", "ETHUSDT",
        "SOLUSDT", "PENGUUSDT", "HYPEUSDT", "PENDLEUSDT", "ETHFIUSDT", "ZROUSDT", "ASTERUSDT",
    ]
    category = "linear"  # alternatives: "spot", "inverse", "option"

    # Use a category-specific DB file to avoid mixing spot and futures data
    db_path = f"data/bybit_{category}.sqlite3"
    try:
        asyncio.run(run(symbols=symbols, category=category, db_path=db_path))
    except KeyboardInterrupt:
        # Swallow extra traceback on Windows/3.13 during runner teardown
        pass


