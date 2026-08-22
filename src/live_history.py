"""Live per-sender transaction history for single-row inference.

SQLite store scoped strictly to the live-inference path: every transaction
scored through ``PredictionEngine.predict`` is recorded after scoring, and the
accumulated rows provide each sender's own growing amount/location/payee/
velocity history. The training pipeline never touches this database.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import DATA_DIR, get_logger


UPI_ABSOLUTE_MAX_AMOUNT = 2_000_000.0
DEFAULT_LIVE_HISTORY_DB_PATH = DATA_DIR / "live_history.db"
RAPID_WINDOW_MINUTES = 5.0

LOGGER = get_logger(__name__)

_SCHEMA_SCRIPT = """
CREATE TABLE IF NOT EXISTS live_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT,
    sender_id TEXT NOT NULL,
    receiver_id TEXT,
    amount REAL NOT NULL,
    location TEXT,
    timestamp TEXT,
    fraud_probability REAL,
    fraud_prediction INTEGER,
    recorded_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_live_transactions_sender
    ON live_transactions (sender_id);
"""


def record_transaction(
    sender_id: str,
    receiver_id: str,
    amount: float,
    location: str,
    timestamp: Any,
    transaction_id: str | None = None,
    fraud_probability: float | None = None,
    fraud_prediction: int | None = None,
    db_path: Path | str = DEFAULT_LIVE_HISTORY_DB_PATH,
) -> bool:
    """Persist one scored transaction; returns True when stored."""
    try:
        with closing(_connect(db_path)) as connection, connection:
            connection.executescript(_SCHEMA_SCRIPT)
            connection.execute(
                """
                INSERT INTO live_transactions (
                    transaction_id, sender_id, receiver_id, amount, location,
                    timestamp, fraud_probability, fraud_prediction, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    None if transaction_id is None else str(transaction_id),
                    str(sender_id),
                    None if receiver_id is None else str(receiver_id),
                    float(amount),
                    None if location is None else str(location),
                    _iso_timestamp(timestamp),
                    None if fraud_probability is None else float(fraud_probability),
                    None if fraud_prediction is None else int(fraud_prediction),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return True
    except (sqlite3.Error, TypeError, ValueError) as exc:
        LOGGER.warning("Live history recording failed: %s", exc)
        return False


def fetch_sender_summary(
    sender_id: str,
    db_path: Path | str = DEFAULT_LIVE_HISTORY_DB_PATH,
) -> dict[str, Any] | None:
    """Aggregate one sender's recorded history; None when nothing is stored.

    Returns ``count``, ``amount_mean``, ``amount_std`` (population std, None
    for a single row), ``usual_location`` (most frequent, most recent on
    ties), ``last_timestamp`` (ISO string), and ``known_receivers`` (set).
    """
    try:
        with closing(_connect(db_path)) as connection:
            connection.executescript(_SCHEMA_SCRIPT)
            cursor = connection.execute(
                """
                SELECT receiver_id, amount, location, timestamp
                FROM live_transactions
                WHERE sender_id = ?
                ORDER BY datetime(timestamp), id
                """,
                (str(sender_id),),
            )
            rows = cursor.fetchall()
    except sqlite3.Error as exc:
        LOGGER.warning("Live history lookup failed: %s", exc)
        return None
    if not rows:
        return None

    amounts = [float(row[1]) for row in rows]
    count = len(amounts)
    mean = sum(amounts) / count
    variance = sum((value - mean) ** 2 for value in amounts) / count
    location_counts: dict[str, int] = {}
    for row in rows:
        key = str(row[2])
        location_counts[key] = location_counts.get(key, 0) + 1
    usual_location = max(
        rows,
        key=lambda row: (location_counts[str(row[2])], row[0] or 0),
    )[2]
    receivers = {str(row[0]) for row in rows if row[0] is not None}
    last_row = rows[-1]
    return {
        "count": count,
        "amount_mean": mean,
        "amount_std": None if count < 2 else variance ** 0.5,
        "usual_location": str(usual_location),
        "last_amount": amounts[-1],
        "last_timestamp": None if last_row[3] is None else str(last_row[3]),
        "known_receivers": receivers,
    }


def clear_live_history(db_path: Path | str = DEFAULT_LIVE_HISTORY_DB_PATH) -> bool:
    """Delete all recorded rows (used by tests)."""
    try:
        with closing(_connect(db_path)) as connection, connection:
            connection.executescript(_SCHEMA_SCRIPT)
            connection.execute("DELETE FROM live_transactions")
        return True
    except sqlite3.Error as exc:
        LOGGER.warning("Live history clear failed: %s", exc)
        return False


def _connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def _iso_timestamp(value: Any) -> str | None:
    """Normalize a timestamp-like value to a tz-naive ISO-8601 string.

    Aware values are converted to UTC and stripped of their offset, matching
    the ``safe_datetime`` convention used by the feature pipeline, so every
    stored timestamp compares cleanly against scoring-frame timestamps.
    """
    if value is None:
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if parsed is pd.NaT or pd.isna(parsed):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed.isoformat()
