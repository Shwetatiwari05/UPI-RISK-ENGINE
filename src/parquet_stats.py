"""Streaming statistics for Parquet-backed pipeline stages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.parquet_pipeline import DEFAULT_CHUNK_SIZE, iter_parquet_chunks


def parquet_label_counts(
    parquet_path: Path | str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    target_column: str = "fraud_label",
) -> dict[int, int]:
    """Count binary labels without materializing the complete Parquet file."""
    counts = {0: 0, 1: 0}
    for chunk in iter_parquet_chunks(parquet_path, chunk_size, [target_column]):
        labels = pd.to_numeric(chunk[target_column], errors="coerce").fillna(0).astype(int).clip(0, 1)
        values = labels.value_counts()
        counts[0] += int(values.get(0, 0))
        counts[1] += int(values.get(1, 0))
    return counts


def parquet_period_label_counts(
    parquet_path: Path | str,
    chunk_size: int,
    period_value: str,
    target_column: str = "fraud_label",
) -> dict[int, int]:
    """Count binary labels restricted to one ``period`` value."""
    counts = {0: 0, 1: 0}
    for chunk in iter_parquet_chunks(parquet_path, chunk_size, [target_column, "period"]):
        if "period" in chunk.columns:
            chunk = chunk.loc[chunk["period"].astype(str) == period_value]
        if chunk.empty:
            continue
        labels = pd.to_numeric(chunk[target_column], errors="coerce").fillna(0).astype(int).clip(0, 1)
        values = labels.value_counts()
        counts[0] += int(values.get(0, 0))
        counts[1] += int(values.get(1, 0))
    return counts
