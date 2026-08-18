"""Feature engineering for offline transaction fraud and anomaly analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema_mapping import validate_common_schema
from src.utils import get_logger, safe_datetime


RAPID_TRANSACTION_WINDOW_MINUTES = 5
LOGGER = get_logger(__name__)


def build_feature_context(df: pd.DataFrame) -> dict[str, object]:
    """Build historical aggregates used when scoring a single transaction.

    Training feature engineering operates on batches. A manual UI transaction
    has only one row, so calculating sender history from that row would create
    misleading values such as ``transaction_frequency=1``. This compact
    context keeps the reference aggregates from the preprocessing fit sample
    and is reused consistently for one-row inference.
    """
    working = df.copy()
    working["timestamp"] = safe_datetime(working["timestamp"])
    working["amount"] = pd.to_numeric(working["amount"], errors="coerce").fillna(0.0)
    working["hour_of_day"] = working["timestamp"].dt.hour.fillna(0).astype(int)
    working["sender_id"] = working["sender_id"].astype(str)
    working["receiver_id"] = working["receiver_id"].astype(str)
    working["device_type"] = working["device_type"].astype(str)
    working["merchant_category"] = working["merchant_category"].astype(str)
    working["location"] = working["location"].astype(str)

    grouped = working.groupby("sender_id", sort=False, dropna=False)
    sender_stats: dict[str, dict[str, object]] = {}
    for sender, group in grouped:
        amounts = group["amount"].to_numpy(dtype=np.float64)
        sender_stats[str(sender)] = {
            "average_amount": float(np.mean(amounts)) if amounts.size else 0.0,
            "amount_std": float(np.std(amounts)) if amounts.size > 1 else 0.0,
            "transaction_frequency": int(len(group)),
            "merchant_diversity": int(group["merchant_category"].nunique()),
            "device_switching_frequency": int(group["device_type"].nunique()),
            "usual_location": str(group["location"].mode().iloc[0]) if not group["location"].mode().empty else "Unknown",
            "hours": group["hour_of_day"].value_counts().astype(int).to_dict(),
        }

    pairs = set(
        zip(
            working["sender_id"].astype(str),
            working["receiver_id"].astype(str),
        )
    )
    hourly_counts = working.groupby(["sender_id", "hour_of_day"], sort=False).size()
    hourly_values = hourly_counts.to_numpy(dtype=np.float64)
    return {
        "sender_stats": sender_stats,
        "known_sender_receiver_pairs": pairs,
        "global_amount_mean": float(working["amount"].mean()) if len(working) else 0.0,
        "global_hourly_frequency_p95": float(np.quantile(hourly_values, 0.95)) if hourly_values.size else 1.0,
    }


def _apply_feature_context(df: pd.DataFrame, context: dict[str, object]) -> pd.DataFrame:
    """Replace one-row behavioural defaults with reference-history values."""
    enriched = df.copy()
    sender_stats = context.get("sender_stats", {})
    global_mean = float(context.get("global_amount_mean", 0.0))
    hourly_p95 = float(context.get("global_hourly_frequency_p95", 1.0))
    known_pairs = context.get("known_sender_receiver_pairs", set())

    averages = []
    frequencies = []
    merchant_diversity = []
    device_switching = []
    hourly_counts = []
    amount_spikes = []
    unusual_locations = []
    new_payees = []
    for row in enriched.itertuples(index=False):
        sender = str(row.sender_id)
        receiver = str(row.receiver_id)
        stats = sender_stats.get(sender, {})
        average = float(stats.get("average_amount", global_mean))
        std = float(stats.get("amount_std", 0.0))
        frequency = int(stats.get("transaction_frequency", 0))
        hour_counts = stats.get("hours", {})
        current_hour_count = int(hour_counts.get(int(row.hour_of_day), 0))
        usual_location = str(stats.get("usual_location", "Unknown"))

        averages.append(average)
        frequencies.append(frequency)
        merchant_diversity.append(int(stats.get("merchant_diversity", 0)))
        device_switching.append(int(stats.get("device_switching_frequency", 0)))
        hourly_counts.append(current_hour_count)
        amount_spikes.append(int(float(row.amount) > average + (3.0 * std) if frequency > 1 else 0))
        unusual_locations.append(int(frequency > 0 and str(row.location) != usual_location))
        new_payees.append(int((sender, receiver) not in known_pairs))

    enriched["avg_transaction_amount"] = np.asarray(averages, dtype="float32")
    enriched["transaction_frequency"] = np.asarray(frequencies, dtype="int32")
    enriched["merchant_diversity"] = np.asarray(merchant_diversity, dtype="int16")
    enriched["device_switching_frequency"] = np.asarray(device_switching, dtype="int16")
    enriched["transactions_per_hour"] = np.asarray(hourly_counts, dtype="int16")
    enriched["amount_spike"] = np.asarray(amount_spikes, dtype="int8")
    enriched["high_frequency_payments"] = (
        enriched["transactions_per_hour"] >= max(2.0, hourly_p95)
    ).astype("int8")
    enriched["new_payee_flag"] = np.asarray(new_payees, dtype="int8")
    enriched["unusual_location_flag"] = np.asarray(unusual_locations, dtype="int8")
    enriched["rapid_transactions"] = (
        enriched["minutes_since_previous_sender_txn"].notna()
        & (enriched["minutes_since_previous_sender_txn"] <= RAPID_TRANSACTION_WINDOW_MINUTES)
    ).astype("int8")
    return enriched


def engineer_features(df: pd.DataFrame, context: dict[str, object] | None = None) -> pd.DataFrame:
    """Generate behavioral, velocity, risk, and temporal features."""
    LOGGER.info("Starting feature engineering on %s rows", len(df))
    validate_common_schema(df)
    features = df.copy()
    features["timestamp"] = safe_datetime(features["timestamp"])
    features["timestamp"] = features["timestamp"].fillna(pd.Timestamp("2024-01-01"))
    features["amount"] = pd.to_numeric(features["amount"], errors="coerce").fillna(0.0)
    features["sender_id"] = features["sender_id"].astype("category")
    features["receiver_id"] = features["receiver_id"].astype("category")
    features["merchant_category"] = features["merchant_category"].astype("category")
    features["device_type"] = features["device_type"].astype("category")
    features["location"] = features["location"].astype("category")

    LOGGER.info("Adding temporal features")
    features = add_temporal_features(features)
    LOGGER.info("Adding behavioral features")
    features = add_behavioral_features(features)
    LOGGER.info("Adding velocity features")
    features = add_velocity_features(features)
    LOGGER.info("Adding risk features")
    features = add_risk_features(features)
    if context is not None:
        features = _apply_feature_context(features, context)
    LOGGER.info("Feature engineering complete with %s columns", len(features.columns))
    return features


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day-of-week, and weekend indicators."""
    df["hour_of_day"] = df["timestamp"].dt.hour.astype("int8")
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
    return df


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sender-level behavioral aggregates."""
    sender_group = df.groupby("sender_id", dropna=False, sort=False, observed=True)

    df["avg_transaction_amount"] = sender_group["amount"].transform("mean").astype("float32")
    df["transaction_frequency"] = sender_group["transaction_id"].transform("count").astype("int32")
    df["merchant_diversity"] = sender_group["merchant_category"].transform("nunique").astype("int16")
    df["device_switching_frequency"] = sender_group["device_type"].transform("nunique").astype("int16")

    sender_hour_group = df.groupby(["sender_id", "hour_of_day"], dropna=False, sort=False, observed=True)
    df["transactions_per_hour"] = sender_hour_group["transaction_id"].transform("count").astype("int16")
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rapid transaction, amount spike, and high-frequency payment features."""
    df = df.sort_values(["sender_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    df["minutes_since_previous_sender_txn"] = (
        df.groupby("sender_id", sort=False, observed=True)["timestamp"].diff().dt.total_seconds().div(60)
    ).astype("float32")
    df["rapid_transactions"] = (
        df["minutes_since_previous_sender_txn"].fillna(np.inf)
        <= RAPID_TRANSACTION_WINDOW_MINUTES
    ).astype("int8")

    sender_group = df.groupby("sender_id", dropna=False, sort=False, observed=True)["amount"]
    sender_mean = sender_group.transform("mean")
    sender_std = sender_group.transform("std").fillna(0)
    df["amount_spike"] = (df["amount"] > sender_mean + (3 * sender_std)).astype("int8")

    hourly_threshold = df["transactions_per_hour"].quantile(0.95)
    if pd.isna(hourly_threshold):
        hourly_threshold = 1
    df["high_frequency_payments"] = (
        df["transactions_per_hour"] >= max(2, hourly_threshold)
    ).astype("int8")
    return df


def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add unusual timing, new payee, and unusual location flags."""
    df["unusual_transaction_timing"] = df["hour_of_day"].between(0, 5).astype("int8")
    df["new_payee_flag"] = (
        ~df.duplicated(subset=["sender_id", "receiver_id"], keep="first")
    ).astype("int8")

    LOGGER.info("Calculating usual location per sender")
    usual_location = _most_frequent_value_by_group(df, group_column="sender_id", value_column="location")
    mapped_location = df["sender_id"].astype(str).map(usual_location.astype(str).to_dict())
    df["usual_location"] = mapped_location.fillna("Unknown")
    df["unusual_location_flag"] = (df["location"].astype(str) != df["usual_location"].astype(str)).astype("int8")
    df = df.drop(columns=["usual_location"])
    return df


def feature_columns_for_model(df: pd.DataFrame, target_column: str = "fraud_label") -> list[str]:
    """Return model feature columns after excluding identifiers and labels."""
    excluded = {
        target_column,
        "transaction_id",
        "timestamp",
    }
    return [column for column in df.columns if column not in excluded]


def _most_frequent_value_by_group(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.Series:
    """Return the most frequent value for each group without per-group Python mode calls."""
    counts = (
        df.groupby([group_column, value_column], sort=False, observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    if counts.empty:
        return pd.Series(dtype="object")

    winners = counts.loc[counts.groupby(group_column, sort=False, observed=True)["count"].idxmax()]
    return winners.set_index(group_column)[value_column]
