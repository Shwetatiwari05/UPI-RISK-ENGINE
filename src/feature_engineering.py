"""Point-in-time feature engineering for offline transaction fraud analysis.

All behavioral aggregates are strictly backward-looking: a derived value for a
transaction is computed only from that sender's earlier transactions, so train
and evaluation rows carry exactly the information a deployed model would have
at scoring time. Fitted constants (usual location per sender, high-frequency
threshold, cold-start fallbacks) are learned from training-period rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema_mapping import validate_common_schema
from src.utils import get_logger, safe_datetime


RAPID_TRANSACTION_WINDOW_MINUTES = 5
HIGH_FREQUENCY_QUANTILE = 0.95
LOGGER = get_logger(__name__)


def engineer_features(
    df: pd.DataFrame,
    context: dict[str, object] | None = None,
    constants: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Generate causal behavioral, velocity, risk, and temporal features.

    Behavioral aggregates use only each sender's earlier transactions within
    ``df``, so callers must pass complete history for the senders involved
    (the pipeline passes one timestamp-sorted source block at a time).

    ``constants`` are fitted values from the training period (see
    :func:`fit_feature_constants`). When omitted, constants are fitted on the
    incoming batch itself, which is only appropriate for standalone use on a
    frame that already contains full history.

    ``context`` overrides behavioral values with reference history for
    single-row inference (see :func:`build_feature_context`).
    """
    LOGGER.info("Starting point-in-time feature engineering on %s rows", len(df))
    validate_common_schema(df)
    features = df.copy()
    features["timestamp"] = safe_datetime(features["timestamp"])
    features["timestamp"] = features["timestamp"].fillna(pd.Timestamp("2024-01-01"))
    features["amount"] = pd.to_numeric(features["amount"], errors="coerce").fillna(0.0)
    for column in ["sender_id", "receiver_id", "merchant_category", "device_type", "location"]:
        features[column] = features[column].astype(str)

    features = add_temporal_features(features)
    features = apply_point_in_time_features(features)

    if context is not None:
        features = _apply_feature_context(features, context)
        LOGGER.info("Feature engineering complete with %s columns", len(features.columns))
        return features

    if constants is None:
        constants = fit_feature_constants(features)
        LOGGER.info("No constants supplied; fitted fallback constants on the given batch")
    features = apply_fitted_features(features, constants)
    LOGGER.info("Feature engineering complete with %s columns", len(features.columns))
    return features


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day-of-week, and weekend indicators from the row itself."""
    df["hour_of_day"] = df["timestamp"].dt.hour.astype("int8")
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
    return df


def apply_point_in_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly backward-looking behavioral and velocity features.

    The frame is sorted by ``[sender_id, timestamp]``; every aggregate for a
    row uses only that sender's earlier rows. First-ever transactions receive
    neutral defaults (frequency 0, no average, no spike, not rapid).
    """
    df = df.sort_values(["sender_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    grouped = df.groupby("sender_id", sort=False, observed=True)

    prior_count = grouped.cumcount().astype("int64")
    amount = df["amount"].astype("float64")
    df["_amount_sq"] = (amount**2).astype("float64")

    prior_sum = grouped["amount"].cumsum().to_numpy(dtype="float64") - amount.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        prior_mean = np.where(prior_count > 0, prior_sum / np.maximum(prior_count, 1), np.nan)

    squared_sum = grouped["_amount_sq"].cumsum().to_numpy(dtype="float64") - df["_amount_sq"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        prior_var = np.where(prior_count > 0, squared_sum / np.maximum(prior_count, 1), np.nan)
    prior_var = np.clip(prior_var - np.nan_to_num(prior_mean, nan=0.0) ** 2, 0.0, None)
    prior_std = np.sqrt(prior_var)

    df["transaction_frequency"] = prior_count.astype("int32")
    df["avg_transaction_amount"] = pd.Series(prior_mean, index=df.index).astype("float32")
    df["amount_spike"] = (
        (prior_count >= 2) & (amount.to_numpy() > np.nan_to_num(prior_mean) + 3.0 * prior_std)
    ).astype("int8")

    hour_group = df.groupby(["sender_id", "hour_of_day"], sort=False, observed=True)
    df["transactions_per_hour"] = hour_group.cumcount().astype("int16")

    df["minutes_since_previous_sender_txn"] = (
        grouped["timestamp"].diff().dt.total_seconds().div(60)
    ).astype("float32")
    df["rapid_transactions"] = (
        df["minutes_since_previous_sender_txn"].fillna(np.inf) <= RAPID_TRANSACTION_WINDOW_MINUTES
    ).astype("int8")

    merchant_new = ~df.duplicated(subset=["sender_id", "merchant_category"], keep="first")
    device_new = ~df.duplicated(subset=["sender_id", "device_type"], keep="first")
    df["merchant_diversity"] = (
        merchant_new.groupby(df["sender_id"], sort=False, observed=True).cumsum() - merchant_new.astype(int)
    ).astype("int16")
    df["device_switching_frequency"] = (
        device_new.groupby(df["sender_id"], sort=False, observed=True).cumsum() - device_new.astype(int)
    ).astype("int16")

    df["new_payee_flag"] = (~df.duplicated(subset=["sender_id", "receiver_id"], keep="first")).astype("int8")
    df["unusual_transaction_timing"] = df["hour_of_day"].between(0, 5).astype("int8")
    return df.drop(columns=["_amount_sq"])


def fit_feature_constants(train_rows: pd.DataFrame) -> dict[str, object]:
    """Learn deployment constants from training-period rows only.

    Returns the usual-location map per sender, the high-frequency threshold,
    and global amount statistics used as cold-start fallbacks.
    """
    usual_location = _most_frequent_value_by_group(
        train_rows, group_column="sender_id", value_column="location"
    )
    threshold = train_rows["transactions_per_hour"].quantile(HIGH_FREQUENCY_QUANTILE)
    if pd.isna(threshold):
        threshold = 1
    return {
        "usual_location_map": usual_location.astype(str).to_dict(),
        "high_frequency_threshold": float(max(2, threshold)),
        "global_amount_mean": float(train_rows["amount"].mean()) if len(train_rows) else 0.0,
    }


def apply_fitted_features(df: pd.DataFrame, constants: dict[str, object]) -> pd.DataFrame:
    """Apply train-period constants to any frame without recomputing them."""
    result = df.copy()
    usual_map = dict(constants.get("usual_location_map", {}))
    mapped = result["sender_id"].map(usual_map).fillna("Unknown").astype(str)
    result["unusual_location_flag"] = (
        result["location"].astype(str) != mapped
    ).astype("int8")
    threshold = float(constants.get("high_frequency_threshold", 2))
    result["high_frequency_payments"] = (
        result["transactions_per_hour"] >= threshold
    ).astype("int8")
    fallback_mean = float(constants.get("global_amount_mean", 0.0))
    result["avg_transaction_amount"] = (
        result["avg_transaction_amount"].fillna(fallback_mean).astype("float32")
    )
    return result


def build_feature_context(df: pd.DataFrame) -> dict[str, object]:
    """Build reference history used when scoring a single transaction.

    Callers pass training-period rows so one-row inference reproduces the same
    aggregates the models saw during training. The returned context mirrors
    the point-in-time definitions in :func:`apply_point_in_time_features`
    evaluated at the end of the training period.
    """
    working = df.copy()
    working["timestamp"] = safe_datetime(working["timestamp"])
    working["amount"] = pd.to_numeric(working["amount"], errors="coerce").fillna(0.0)
    working["hour_of_day"] = working["timestamp"].dt.hour.fillna(0).astype(int)
    for column in ["sender_id", "receiver_id", "device_type", "merchant_category", "location"]:
        working[column] = working[column].astype(str)

    grouped = working.groupby("sender_id", sort=False, dropna=False)
    sender_stats: dict[str, dict[str, object]] = {}
    for sender, group in grouped:
        amounts = group["amount"].to_numpy(dtype=np.float64)
        mode = group["location"].mode()
        sender_stats[str(sender)] = {
            "average_amount": float(np.mean(amounts)) if amounts.size else 0.0,
            "amount_std": float(np.std(amounts)) if amounts.size > 1 else 0.0,
            "transaction_frequency": int(len(group)),
            "merchant_diversity": int(group["merchant_category"].nunique()),
            "device_switching_frequency": int(group["device_type"].nunique()),
            "usual_location": str(mode.iloc[0]) if not mode.empty else "Unknown",
            "hours": group["hour_of_day"].value_counts().astype(int).to_dict(),
        }

    pairs = set(zip(working["sender_id"], working["receiver_id"]))
    hourly_counts = working.groupby(["sender_id", "hour_of_day"], sort=False).size()
    hourly_values = hourly_counts.to_numpy(dtype=np.float64)
    threshold = float(np.quantile(hourly_values, HIGH_FREQUENCY_QUANTILE)) if hourly_values.size else 1.0
    return {
        "sender_stats": sender_stats,
        "known_sender_receiver_pairs": pairs,
        "global_amount_mean": float(working["amount"].mean()) if len(working) else 0.0,
        "global_hourly_frequency_p95": max(2.0, threshold),
    }


def _apply_feature_context(df: pd.DataFrame, context: dict[str, object]) -> pd.DataFrame:
    """Override behavioral columns with reference-history values for serving.

    Single-row inference has no intra-batch history, so values mirror the
    training-period aggregates a scoring service would hold: frequency is the
    sender's stored history size, averages come from stored stats, and no
    previous-transaction gap exists.
    """
    enriched = df.copy()
    sender_stats = context.get("sender_stats", {})
    global_mean = float(context.get("global_amount_mean", 0.0))
    hourly_p95 = float(context.get("global_hourly_frequency_p95", 2.0))
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
        unusual_locations.append(int(str(row.location) != usual_location))
        new_payees.append(int((sender, receiver) not in known_pairs))

    enriched["avg_transaction_amount"] = np.asarray(averages, dtype="float32")
    enriched["transaction_frequency"] = np.asarray(frequencies, dtype="int32")
    enriched["merchant_diversity"] = np.asarray(merchant_diversity, dtype="int16")
    enriched["device_switching_frequency"] = np.asarray(device_switching, dtype="int16")
    enriched["transactions_per_hour"] = np.asarray(hourly_counts, dtype="int16")
    enriched["amount_spike"] = np.asarray(amount_spikes, dtype="int8")
    enriched["high_frequency_payments"] = (
        enriched["transactions_per_hour"] >= hourly_p95
    ).astype("int8")
    enriched["new_payee_flag"] = np.asarray(new_payees, dtype="int8")
    enriched["unusual_location_flag"] = np.asarray(unusual_locations, dtype="int8")
    enriched["rapid_transactions"] = np.zeros(len(enriched), dtype="int8")
    return enriched


def feature_columns_for_model(df: pd.DataFrame, target_column: str = "fraud_label") -> list[str]:
    """Return model feature columns after excluding identifiers and labels."""
    excluded = {
        target_column,
        "transaction_id",
        "timestamp",
        "period",
        "source_dataset",
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
