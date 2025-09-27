import asyncio
import logging
import os
import signal
from typing import List

from db.database import SQLiteClient
from src.bybit_collector import BybitDataCollector
from utils.logging_setup import setup_logging


async def run(symbols: List[str], category: str = "spot", db_path: str = "data/bybit_data.sqlite3") -> None:
    setup_logging(logging.INFO)
    logger = logging.getLogger("main")

    db = SQLiteClient(db_path)
    collector = BybitDataCollector(symbols=symbols, db_client=db, category=category, depth_level=200)

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

    logger.info("Starting BybitDataCollector for symbols: %s (category=%s)", ", ".join(symbols), category)
    producer = asyncio.create_task(collector.start())

    try:
        # Wait until stop_event is set
        await stop_event.wait()
    except asyncio.CancelledError:
        # Loop shutdown (e.g., Ctrl+C). Proceed to graceful cleanup.
        logger.info("Cancellation received. Shutting down...")
    finally:
        if not producer.done():
            producer.cancel()
            try:
                await producer
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


