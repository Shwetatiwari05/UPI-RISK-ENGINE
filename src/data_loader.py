"""Dataset loading and validation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.schema_mapping import infer_dataset_key, inspect_schema, map_dataset_schema
from src.utils import RAW_DATA_DIR, get_logger, load_dataframe


LOGGER = get_logger(__name__)
SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls"}


def discover_dataset_files(raw_dir: Path | str = RAW_DATA_DIR) -> list[Path]:
    """Find supported dataset files under the raw data directory."""
    directory = Path(raw_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {directory}")

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_dataset(path: Path | str, dataset_key: str | None = None) -> tuple[str, pd.DataFrame]:
    """Load one dataset file and return its inferred key with the dataframe."""
    input_path = Path(path)
    df = load_dataframe(input_path)
    key = dataset_key or infer_dataset_key(path=input_path, columns=list(df.columns))
    LOGGER.info("Loaded %s as %s with shape %s", input_path.name, key, df.shape)
    return key, df


def load_all_datasets(raw_dir: Path | str = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all supported datasets found in data/raw."""
    datasets: dict[str, pd.DataFrame] = {}
    for path in discover_dataset_files(raw_dir):
        dataset_key, df = load_dataset(path)
        key = _deduplicate_key(dataset_key, datasets)
        datasets[key] = df
    if not datasets:
        raise FileNotFoundError(
            f"No supported dataset files found in {raw_dir}. "
            "Place the Kaggle/Zenodo files in data/raw/ first."
        )
    return datasets


def generate_schema_reports(datasets: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    """Create schema inspection reports for loaded datasets."""
    reports = {}
    for dataset_key, df in datasets.items():
        base_key = dataset_key.split("__")[0]
        report = inspect_schema(df, dataset_key=base_key)
        reports[dataset_key] = {
            "dataset_key": report.dataset_key,
            "row_count": report.row_count,
            "column_count": report.column_count,
            "columns": report.columns,
            "missing_common_columns": report.missing_common_columns,
        }
    return reports


def load_and_map_all(raw_dir: Path | str = RAW_DATA_DIR) -> pd.DataFrame:
    """Load all raw datasets and map each one into the common schema."""
    mapped_frames = []
    datasets = load_all_datasets(raw_dir)
    for dataset_key, df in datasets.items():
        base_key = dataset_key.split("__")[0]
        mapped_frames.append(
            map_dataset_schema(df, dataset_key=base_key, source_name=dataset_key)
        )

    merged = pd.concat(mapped_frames, ignore_index=True)
    LOGGER.info("Mapped merged dataframe shape: %s", merged.shape)
    return merged


def _deduplicate_key(dataset_key: str, datasets: dict[str, pd.DataFrame]) -> str:
    if dataset_key not in datasets:
        return dataset_key

    index = 2
    while f"{dataset_key}__{index}" in datasets:
        index += 1
    return f"{dataset_key}__{index}"
