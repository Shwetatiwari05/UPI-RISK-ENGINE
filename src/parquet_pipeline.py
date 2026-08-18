"""Bounded Parquet and chunk-processing helpers for the offline pipeline.

The project models are intentionally kept unchanged. This module makes data
ingestion and preprocessing streaming-friendly, then hands a bounded sample to
the existing batch-only estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_preprocessing import UPITransactionPreprocessor
from src.feature_engineering import build_feature_context, engineer_features
from src.schema_mapping import (
    COMMON_SCHEMA,
    COLUMN_ALIASES,
    _find_column,
    infer_dataset_key,
    map_dataset_schema,
)
from src.utils import (
    MERGED_DATA_DIR,
    PROCESSED_DATA_DIR,
    get_logger,
    save_joblib,
)


LOGGER = get_logger(__name__)
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_FIT_ROWS = 100_000
DEFAULT_PARQUET_COMPRESSION = "snappy"
MAPPED_PARQUET_PATH = MERGED_DATA_DIR / "mapped_common_schema.parquet"
PROCESSED_FEATURES_PARQUET_PATH = PROCESSED_DATA_DIR / "processed_features.parquet"


@dataclass(frozen=True)
class ParquetPipelineConfig:
    """Runtime settings for bounded ingestion and transformation."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    fit_rows: int = DEFAULT_FIT_ROWS
    compression: str = DEFAULT_PARQUET_COMPRESSION
    mapped_path: Path = MAPPED_PARQUET_PATH
    processed_path: Path = PROCESSED_FEATURES_PARQUET_PATH

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if self.fit_rows <= 0:
            raise ValueError("fit_rows must be greater than zero.")


def iter_mapped_chunks(
    raw_dir: Path | str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield unified-schema chunks from every supported raw file exactly once."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    raw_paths = sorted(
        path
        for path in Path(raw_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".parquet"}
    )
    if not raw_paths:
        raise FileNotFoundError(f"No CSV or Parquet datasets found in {raw_dir}.")

    used_keys: set[str] = set()
    for path in raw_paths:
        columns = _available_columns(path)
        dataset_key = infer_dataset_key(path=path, columns=columns)
        dataset_key = _deduplicate_key(dataset_key, used_keys)
        used_keys.add(dataset_key)
        base_key = dataset_key.split("__")[0]
        source_offset = 0

        for chunk_number, raw_chunk in enumerate(_iter_source_chunks(path, chunk_size, columns)):
            if raw_chunk.empty:
                continue
            mapped = map_dataset_schema(
                raw_chunk,
                dataset_key=base_key,
                source_name=dataset_key,
                id_offset=source_offset,
            )
            source_offset += len(raw_chunk)
            LOGGER.info(
                "Mapped %s chunk %s: %s rows",
                path.name,
                chunk_number + 1,
                len(mapped),
            )
            yield mapped
            del raw_chunk, mapped


def write_mapped_parquet(
    raw_dir: Path | str,
    output_path: Path | str = MAPPED_PARQUET_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    compression: str = DEFAULT_PARQUET_COMPRESSION,
) -> dict[str, int | str]:
    """Map raw files and write one compressed Parquet file row-group by row-group."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    writer: pq.ParquetWriter | None = None
    output_schema: pa.Schema | None = None
    rows_read = 0
    rows_removed = 0
    total_rows = 0
    chunk_count = 0
    try:
        for mapped in iter_mapped_chunks(raw_dir, chunk_size=chunk_size):
            input_rows = len(mapped)
            duplicate_rows = int(mapped["transaction_id"].duplicated().sum())
            if duplicate_rows:
                raise ValueError(
                    f"Mapped chunk contains {duplicate_rows} duplicate transaction IDs."
                )
            table = pa.Table.from_pandas(mapped[COMMON_SCHEMA], preserve_index=False)
            if writer is None:
                output_schema = table.schema
                writer = pq.ParquetWriter(
                    destination,
                    output_schema,
                    compression=compression,
                    use_dictionary=True,
                )
            else:
                table = table.cast(output_schema, safe=False)
            writer.write_table(table, row_group_size=len(mapped))
            rows_read += input_rows
            total_rows += len(mapped)
            chunk_count += 1
            LOGGER.info(
                "Wrote mapped Parquet chunk %s: rows_read=%s rows_removed=%s rows_written=%s",
                chunk_count,
                rows_read,
                rows_removed,
                total_rows,
            )
            del table, mapped
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError("No non-empty records were available for Parquet conversion.")

    return {
        "path": str(destination),
        "rows": total_rows,
        "rows_read": rows_read,
        "rows_removed": rows_removed,
        "rows_written": total_rows,
        "chunks": chunk_count,
    }


def iter_parquet_chunks(
    parquet_path: Path | str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    columns: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield selected Parquet columns without materializing the full dataset."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    parquet_file = pq.ParquetFile(parquet_path)
    available = set(parquet_file.schema.names)
    selected = [column for column in (columns or parquet_file.schema.names) if column in available]
    if not selected:
        raise ValueError(f"None of the requested columns exist in {parquet_path}.")

    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=selected):
        yield batch.to_pandas()


def sample_parquet_rows(
    parquet_path: Path | str,
    max_rows: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    target_column: str = "fraud_label",
    random_state: int = 42,
    columns: list[str] | None = None,
    strategy: str = "fraud_preserving",
    legitimate_ratio: float = 3.0,
    stratify_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return a bounded, reproducible Parquet sample using two passes.

    ``fraud_preserving`` keeps every fraud row when the row budget permits it,
    then samples legitimate rows at the requested ratio. Legitimate rows are
    allocated proportionally across available strata. ``uniform`` is intended
    for unsupervised models and does not use fraud labels.
    """
    if max_rows <= 0:
        raise ValueError("max_rows must be greater than zero.")
    if strategy not in {"fraud_preserving", "uniform"}:
        raise ValueError("strategy must be 'fraud_preserving' or 'uniform'.")
    if legitimate_ratio <= 0:
        raise ValueError("legitimate_ratio must be greater than zero.")

    requested_columns = None if columns is None else list(dict.fromkeys(columns))
    if requested_columns is not None and target_column not in requested_columns:
        requested_columns.append(target_column)
    strata = stratify_columns or ["hour_of_day", "day_of_week"]
    for column in strata:
        if requested_columns is not None and column not in requested_columns:
            requested_columns.append(column)
    selected_columns = requested_columns

    label_counts = {0: 0, 1: 0}
    legitimate_strata_counts: dict[str, int] = {}
    total_rows = 0
    for chunk in iter_parquet_chunks(parquet_path, chunk_size, selected_columns):
        total_rows += len(chunk)
        if strategy == "fraud_preserving":
            values = pd.to_numeric(chunk[target_column], errors="coerce").fillna(0).astype(int)
            counts = values.value_counts()
            for label in label_counts:
                label_counts[label] += int(counts.get(label, 0))
            legitimate = chunk.loc[values == 0]
            legitimate_keys = _build_strata_keys(legitimate, strata)
            for key, count in legitimate_keys.value_counts().items():
                legitimate_strata_counts[str(key)] = legitimate_strata_counts.get(str(key), 0) + int(count)
        del chunk

    if total_rows == 0:
        return pd.DataFrame(columns=selected_columns or [])

    if strategy == "fraud_preserving" and label_counts.get(1, 0) == 0:
        LOGGER.warning("No fraud rows found; falling back to uniform sampling.")
        strategy = "uniform"

    if strategy == "uniform":
        target_rows = min(max_rows, total_rows)
        fraud_target = legitimate_target = 0
        legitimate_targets: dict[str, int] = {}
    else:
        fraud_count = label_counts.get(1, 0)
        legitimate_count = label_counts.get(0, 0)
        fraud_target = min(fraud_count, max_rows)
        legitimate_target = min(
            legitimate_count,
            max(0, max_rows - fraud_target),
            int(round(fraud_target * legitimate_ratio)),
        )
        target_rows = fraud_target + legitimate_target
        legitimate_targets = _allocate_stratified_targets(
            legitimate_strata_counts,
            legitimate_target,
        )
        LOGGER.info(
            "Fraud-preserving sample targets: fraud=%s legitimate=%s ratio=%.2f",
            fraud_target,
            legitimate_target,
            legitimate_ratio,
        )

    sampled_chunks: list[pd.DataFrame] = []
    selected_rows = 0
    for chunk_number, chunk in enumerate(iter_parquet_chunks(parquet_path, chunk_size, selected_columns)):
        if strategy == "uniform":
            remaining = target_rows - selected_rows
            selected = chunk.sample(
                n=min(remaining, len(chunk)),
                random_state=random_state + chunk_number,
                replace=False,
            ) if remaining > 0 else chunk.iloc[0:0]
        else:
            values = pd.to_numeric(chunk[target_column], errors="coerce").fillna(0).astype(int)
            pieces = []
            fraud_subset = chunk.loc[values == 1]
            if fraud_target >= label_counts.get(1, 0):
                if not fraud_subset.empty:
                    pieces.append(fraud_subset)
            elif not fraud_subset.empty:
                n_rows = min(
                    len(fraud_subset),
                    int(round(fraud_target * len(fraud_subset) / max(label_counts.get(1, 1), 1))),
                )
                if n_rows:
                    pieces.append(fraud_subset.sample(n=n_rows, random_state=random_state + chunk_number))

            legitimate_subset = chunk.loc[values == 0].copy()
            legitimate_keys = _build_strata_keys(legitimate_subset, strata)
            for key, group in legitimate_subset.groupby(legitimate_keys, sort=False, dropna=False):
                target = legitimate_targets.get(str(key), 0)
                global_count = legitimate_strata_counts.get(str(key), 0)
                n_rows = min(len(group), int(round(target * len(group) / max(global_count, 1))))
                if n_rows:
                    pieces.append(group.sample(n=n_rows, random_state=random_state + chunk_number + len(pieces)))
            selected = pd.concat(pieces, ignore_index=True) if pieces else chunk.iloc[0:0]

        if not selected.empty:
            sampled_chunks.append(selected)
            selected_rows += len(selected)
        del chunk

    if not sampled_chunks:
        return pd.DataFrame(columns=selected_columns or [])
    sampled = pd.concat(sampled_chunks, ignore_index=True)
    if strategy == "fraud_preserving":
        fraud_part = sampled.loc[pd.to_numeric(sampled[target_column], errors="coerce").fillna(0).astype(int) == 1]
        legitimate_part = sampled.loc[pd.to_numeric(sampled[target_column], errors="coerce").fillna(0).astype(int) == 0]
        if len(legitimate_part) > legitimate_target:
            legitimate_part = legitimate_part.sample(
                n=legitimate_target,
                random_state=random_state,
                replace=False,
            )
        sampled = pd.concat([fraud_part, legitimate_part], ignore_index=True)
    elif len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state, replace=False)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    LOGGER.info(
        "Bounded Parquet sample: rows=%s of total=%s strategy=%s",
        len(sampled),
        total_rows,
        strategy,
    )
    return sampled


def _build_strata_keys(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Build deterministic stratum keys from available engineered columns."""
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series("all", index=df.index, dtype="object")
    values = df[available].copy()
    for column in available:
        values[column] = values[column].astype(str).fillna("Unknown")
    return values.astype(str).agg("|".join, axis=1)


def _allocate_stratified_targets(counts: dict[str, int], target: int) -> dict[str, int]:
    """Allocate a target count proportionally across strata."""
    if target <= 0 or not counts:
        return {key: 0 for key in counts}
    total = sum(counts.values())
    raw = {key: target * count / total for key, count in counts.items()}
    allocated = {key: min(counts[key], int(value)) for key, value in raw.items()}
    remaining = target - sum(allocated.values())
    order = sorted(raw, key=lambda key: raw[key] - allocated[key], reverse=True)
    for key in order:
        if remaining <= 0:
            break
        if allocated[key] < counts[key]:
            allocated[key] += 1
            remaining -= 1
    return allocated


def stream_preprocess_to_parquet(
    mapped_parquet_path: Path | str,
    output_path: Path | str = PROCESSED_FEATURES_PARQUET_PATH,
    preprocessor: UPITransactionPreprocessor | None = None,
    fit_rows: int = DEFAULT_FIT_ROWS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[UPITransactionPreprocessor, dict[str, int | str]]:
    """Fit once on a bounded sample, then transform every Parquet chunk once."""
    fitted = preprocessor or UPITransactionPreprocessor()
    fit_sample = sample_parquet_rows(
        mapped_parquet_path,
        max_rows=fit_rows,
        chunk_size=chunk_size,
        random_state=42,
        strategy="uniform",
    )
    engineered_fit = engineer_features(fit_sample)
    fitted.fit_transform(engineered_fit)
    feature_context = build_feature_context(fit_sample)
    save_joblib(feature_context, PROCESSED_DATA_DIR.parent.parent / "models" / "feature_context.pkl")
    del fit_sample, engineered_fit

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    writer: pq.ParquetWriter | None = None
    output_schema: pa.Schema | None = None
    rows_read = 0
    rows_removed = 0
    total_rows = 0
    chunk_count = 0
    feature_names = fitted.get_feature_names()
    try:
        for chunk in iter_parquet_chunks(mapped_parquet_path, chunk_size, COMMON_SCHEMA):
            input_rows = len(chunk)
            duplicate_rows = int(chunk["transaction_id"].duplicated().sum())
            if duplicate_rows:
                raise ValueError(
                    f"Processed chunk contains {duplicate_rows} duplicate transaction IDs."
                )
            engineered = engineer_features(chunk)
            transformed = fitted.transform(engineered)
            output = pd.DataFrame(transformed, columns=feature_names)
            output.insert(0, "transaction_id", engineered["transaction_id"].astype(str).to_numpy())
            output["fraud_label"] = engineered["fraud_label"].astype("int8").to_numpy()
            table = pa.Table.from_pandas(output, preserve_index=False)
            if writer is None:
                output_schema = table.schema
                writer = pq.ParquetWriter(
                    destination,
                    output_schema,
                    compression=DEFAULT_PARQUET_COMPRESSION,
                    use_dictionary=True,
                )
            else:
                table = table.cast(output_schema, safe=False)
            writer.write_table(table, row_group_size=len(output))
            rows_read += input_rows
            total_rows += len(output)
            chunk_count += 1
            if chunk_count == 1 or chunk_count % 10 == 0:
                LOGGER.info(
                    "Preprocessed Parquet chunks=%s rows_read=%s rows_removed=%s rows_written=%s",
                    chunk_count,
                    rows_read,
                    rows_removed,
                    total_rows,
                )
            del chunk, engineered, transformed, output, table
    finally:
        if writer is not None:
            writer.close()

    save_joblib(fitted, PROCESSED_DATA_DIR / "chunk_preprocessor.pkl")
    if writer is None:
        raise ValueError("No rows were available for streamed preprocessing.")
    return fitted, {
        "path": str(destination),
        "rows": total_rows,
        "rows_read": rows_read,
        "rows_removed": rows_removed,
        "rows_written": total_rows,
        "chunks": chunk_count,
    }


def _available_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        return list(pq.ParquetFile(path).schema.names)
    return list(pd.read_csv(path, nrows=0).columns)


def _iter_source_chunks(path: Path, chunk_size: int, columns: list[str]) -> Iterator[pd.DataFrame]:
    selected = _required_columns(columns)
    if path.suffix.lower() == ".parquet":
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema.names)
        selected = [column for column in selected if column in available]
        for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=selected):
            yield batch.to_pandas()
        return

    selected = [column for column in selected if column in set(columns)]
    yield from pd.read_csv(
        path,
        usecols=selected,
        chunksize=chunk_size,
        low_memory=False,
    )


def _required_columns(columns: list[str]) -> list[str]:
    selected = set()
    for aliases in COLUMN_ALIASES.values():
        source = _find_column(pd.DataFrame(columns=columns), aliases)
        if source is not None:
            selected.add(source)
    selected.update(
        column
        for column in ["step", "type", "TransactionID", "TransactionDT", "ProductCD", "DeviceType", "addr1"]
        if column in columns
    )
    return [column for column in columns if column in selected]


def _deduplicate_key(dataset_key: str, used_keys: set[str]) -> str:
    if dataset_key not in used_keys:
        return dataset_key
    index = 2
    while f"{dataset_key}__{index}" in used_keys:
        index += 1
    return f"{dataset_key}__{index}"
