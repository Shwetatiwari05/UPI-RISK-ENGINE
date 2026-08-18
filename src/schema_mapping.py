"""Schema mapping utilities for supported fraud and digital payment datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.utils import make_transaction_ids, safe_datetime, validate_columns


COMMON_SCHEMA = [
    "transaction_id",
    "timestamp",
    "amount",
    "sender_id",
    "receiver_id",
    "device_type",
    "merchant_category",
    "location",
    "transaction_type",
    "fraud_label",
]


DATASET_KEYS = {
    "paysim": "PaySim Dataset",
    "upi_2024": "UPI Transaction 2024",
    "ieee": "IEEE Fraud Detection",
    "zenodo_digital_payment": "Digital Payment Transactions",
}


COLUMN_ALIASES = {
    "transaction_id": [
        "transaction_id",
        "transactionid",
        "transaction id",
        "transaction_no",
        "trans_id",
        "id",
        "TransactionID",
    ],
    "timestamp": [
        "timestamp",
        "transaction_time",
        "transaction date",
        "transaction_date",
        "paying_at",
        "created_at",
        "updated_at",
        "date",
        "datetime",
        "time",
        "TransactionDT",
    ],
    "amount": [
        "amount",
        "amount inr",
        "amount_inr",
        "amount rs",
        "amount_rs",
        "amount rupees",
        "transaction_amount",
        "transaction amount",
        "transaction amount inr",
        "transaction_amount_inr",
        "transaction value",
        "transaction_value",
        "net_amount",
        "gross_amount",
        "total_amount",
        "TransactionAmt",
        "amt",
        "value",
        "payment_amount",
        "payment amount",
    ],
    "sender_id": [
        "sender_id",
        "sender upi id",
        "sender_upi_id",
        "payer_id",
        "payer",
        "customer_id",
        "user_id",
        "nameOrig",
        "card1",
    ],
    "receiver_id": [
        "receiver_id",
        "receiver upi id",
        "receiver_upi_id",
        "payee_id",
        "payee",
        "merchant_id",
        "nameDest",
        "card2",
    ],
    "device_type": [
        "device_type",
        "device type",
        "device",
        "device_id",
        "DeviceType",
        "ProductCD",
    ],
    "merchant_category": [
        "merchant_category",
        "merchant category",
        "category",
        "merchant_type",
        "merchant",
        "ProductCD",
        "M4",
    ],
    "location": [
        "location",
        "city",
        "state",
        "country",
        "addr1",
        "addr2",
        "ip_address",
    ],
    "transaction_type": [
        "transaction_type",
        "transaction type",
        "type",
        "payment_type",
        "payment method",
        "payment_method",
    ],
    "fraud_label": [
        "fraud_label",
        "isFraud",
        "is_fraud",
        "fraud",
        "fraudulent",
        "label",
        "class",
    ],
}


@dataclass(frozen=True)
class SchemaReport:
    """Compact schema inspection result."""

    dataset_key: str
    row_count: int
    column_count: int
    columns: list[str]
    missing_common_columns: list[str]


def _normalize_column_name(column: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower())
    return normalized.strip("_")


def _compact_column_name(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(column).strip().lower())


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized_columns = {_normalize_column_name(column): column for column in df.columns}
    compact_columns = {_compact_column_name(column): column for column in df.columns}
    for alias in aliases:
        normalized_alias = _normalize_column_name(alias)
        if normalized_alias in normalized_columns:
            return normalized_columns[normalized_alias]
        compact_alias = _compact_column_name(alias)
        if compact_alias in compact_columns:
            return compact_columns[compact_alias]
    return None


def infer_dataset_key(path: str | Path | None = None, columns: list[str] | None = None) -> str:
    """Infer a dataset key from a file path or recognizable columns."""
    text = str(path or "").lower()
    column_text = " ".join(columns or []).lower()
    combined = f"{text} {column_text}"

    if "zenodo" in text or "digital" in text or "digital_payment" in text:
        return "zenodo_digital_payment"
    if "paysim" in combined or {"nameorig", "namedest"}.issubset(set(combined.split())):
        return "paysim"
    if "ieee" in combined or "transactionamt" in combined or "transactiondt" in combined:
        return "ieee"
    if "upi" in combined:
        return "upi_2024"
    if "zenodo" in combined or "digital" in combined:
        return "zenodo_digital_payment"

    return "upi_2024"


def inspect_schema(df: pd.DataFrame, dataset_key: str) -> SchemaReport:
    """Inspect raw dataframe schema against the common schema aliases."""
    mapped_columns = {
        target: _find_column(df, aliases)
        for target, aliases in COLUMN_ALIASES.items()
    }
    missing = [target for target, source in mapped_columns.items() if source is None]
    return SchemaReport(
        dataset_key=dataset_key,
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        missing_common_columns=missing,
    )


def map_dataset_schema(
    df: pd.DataFrame,
    dataset_key: str,
    source_name: str | None = None,
    id_offset: int = 0,
) -> pd.DataFrame:
    """Map a supported raw dataset into the unified common schema.

    ``id_offset`` keeps generated identifiers unique when a source is read in
    multiple chunks instead of as one in-memory dataframe.
    """
    if df.empty:
        raise ValueError("Cannot map an empty dataframe.")

    mapped = pd.DataFrame(index=df.index)
    source_prefix = source_name or dataset_key

    for target_column in COMMON_SCHEMA:
        source_column = _find_column(df, COLUMN_ALIASES[target_column])
        if source_column is not None:
            mapped[target_column] = df[source_column]
        else:
            mapped[target_column] = _default_column_value(
                target_column,
                len(df),
                source_prefix,
                id_offset=id_offset,
                index=df.index,
            )

    mapped = _apply_dataset_specific_mapping(
        df,
        mapped,
        dataset_key,
        source_prefix,
        id_offset=id_offset,
    )
    mapped = _clean_common_schema(mapped)
    mapped["transaction_id"] = _make_unique_transaction_ids(
        mapped["transaction_id"],
        source_prefix=source_prefix,
        row_offset=id_offset,
    )
    validate_common_schema(mapped, source_name=source_prefix)
    return mapped


def map_all_datasets(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Map and concatenate multiple raw datasets."""
    mapped_frames = []
    for dataset_key, df in datasets.items():
        mapped_frames.append(map_dataset_schema(df, dataset_key=dataset_key))

    if not mapped_frames:
        raise ValueError("No datasets were provided for schema mapping.")

    merged = pd.concat(mapped_frames, ignore_index=True)
    validate_common_schema(merged)
    return merged


def validate_common_schema(df: pd.DataFrame, source_name: str | None = None) -> None:
    """Validate that a dataframe follows the common schema."""
    validate_columns(df, COMMON_SCHEMA)
    if df["amount"].isna().all():
        source_text = f" for {source_name}" if source_name else ""
        raise ValueError(
            f"The mapped dataframe{source_text} has no valid amount values. "
            "Check the raw amount column name and add it to COLUMN_ALIASES['amount'] if needed."
        )
    if not set(df["fraud_label"].dropna().unique()).issubset({0, 1}):
        raise ValueError("fraud_label must contain binary values 0 and 1.")
    if df["transaction_id"].duplicated().any():
        source_text = f" for {source_name}" if source_name else ""
        duplicate_count = int(df["transaction_id"].duplicated().sum())
        raise ValueError(
            f"The mapped dataframe{source_text} contains {duplicate_count} duplicate transaction IDs."
        )


def _default_column_value(
    column: str,
    length: int,
    source_prefix: str,
    id_offset: int = 0,
    index: pd.Index | None = None,
) -> object:
    series_index = index if index is not None else range(length)
    if column == "transaction_id":
        return make_transaction_ids(source_prefix, length, start=id_offset).set_axis(series_index)
    if column == "timestamp":
        return pd.Series(pd.NaT, index=series_index)
    if column == "amount":
        return pd.Series(np.nan, index=series_index)
    if column == "fraud_label":
        return pd.Series(0, index=series_index)
    return pd.Series("Unknown", index=series_index)


def _apply_dataset_specific_mapping(
    raw_df: pd.DataFrame,
    mapped: pd.DataFrame,
    dataset_key: str,
    source_prefix: str,
    id_offset: int = 0,
) -> pd.DataFrame:
    """Apply known dataset-specific conversions after alias mapping."""
    if dataset_key == "paysim":
        if "step" in raw_df.columns:
            base_date = pd.Timestamp("2024-01-01")
            mapped["timestamp"] = base_date + pd.to_timedelta(raw_df["step"], unit="h")
        if "type" in raw_df.columns:
            mapped["transaction_type"] = raw_df["type"]
        mapped["device_type"] = "Unknown"
        mapped["merchant_category"] = mapped["transaction_type"]
        mapped["location"] = "Unknown"

    if dataset_key == "ieee":
        if "TransactionID" in raw_df.columns:
            mapped["transaction_id"] = raw_df["TransactionID"].astype(str)
        if "TransactionDT" in raw_df.columns:
            base_date = pd.Timestamp("2024-01-01")
            mapped["timestamp"] = base_date + pd.to_timedelta(raw_df["TransactionDT"], unit="s")
        if "ProductCD" in raw_df.columns:
            mapped["transaction_type"] = raw_df["ProductCD"].astype(str)
            mapped["merchant_category"] = raw_df["ProductCD"].astype(str)
        if "DeviceType" in raw_df.columns:
            mapped["device_type"] = raw_df["DeviceType"].fillna("Unknown")
        if "addr1" in raw_df.columns:
            mapped["location"] = raw_df["addr1"].astype(str)

    if mapped["transaction_id"].isna().any():
        mapped.loc[mapped["transaction_id"].isna(), "transaction_id"] = make_transaction_ids(
            source_prefix,
            len(mapped),
            start=id_offset,
        )

    return mapped


def _clean_common_schema(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df[COMMON_SCHEMA].copy()
    cleaned["transaction_id"] = cleaned["transaction_id"].astype(str)
    cleaned["timestamp"] = safe_datetime(cleaned["timestamp"])
    cleaned["amount"] = cleaned["amount"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    cleaned.loc[cleaned["amount"] == "", "amount"] = np.nan
    cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="coerce")

    text_columns = [
        "sender_id",
        "receiver_id",
        "device_type",
        "merchant_category",
        "location",
        "transaction_type",
    ]
    for column in text_columns:
        cleaned[column] = cleaned[column].fillna("Unknown").astype(str).str.strip()
        cleaned.loc[cleaned[column] == "", column] = "Unknown"

    cleaned["fraud_label"] = cleaned["fraud_label"].fillna(0)
    cleaned["fraud_label"] = cleaned["fraud_label"].replace(
        {
            "yes": 1,
            "true": 1,
            "fraud": 1,
            "fraudulent": 1,
            "no": 0,
            "false": 0,
            "legitimate": 0,
            "normal": 0,
        }
    )
    cleaned["fraud_label"] = pd.to_numeric(cleaned["fraud_label"], errors="coerce").fillna(0)
    cleaned["fraud_label"] = cleaned["fraud_label"].astype(int).clip(0, 1)
    return cleaned


def _make_unique_transaction_ids(
    values: pd.Series,
    source_prefix: str,
    row_offset: int,
) -> pd.Series:
    """Create stable globally unique IDs while retaining the raw ID as context."""
    raw_values = values.astype("string").fillna("unknown")
    raw_values = raw_values.str.strip()
    raw_values = raw_values.mask(raw_values.str.lower().isin(["", "nan", "none", "nat"]), "unknown")
    row_numbers = pd.Series(
        np.arange(row_offset, row_offset + len(values), dtype=np.int64),
        index=values.index,
    )
    return (
        source_prefix
        + "__"
        + raw_values.astype(str)
        + "__row_"
        + row_numbers.astype(str).str.zfill(8)
    )
