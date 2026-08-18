"""Shared utilities for the offline UPI fraud detection project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MERGED_DATA_DIR = DATA_DIR / "merged"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def get_logger(name: str) -> logging.Logger:
    """Return a configured module logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def ensure_project_dirs() -> None:
    """Create expected project directories if they are missing."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MERGED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "app" / "components",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a clear error when a dataframe is missing required columns."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def save_dataframe(df: pd.DataFrame, path: Path | str) -> None:
    """Save a dataframe as CSV or Parquet based on the file extension."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


def load_dataframe(path: Path | str) -> pd.DataFrame:
    """Load a dataframe from CSV, Parquet, or Excel."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)

    raise ValueError(f"Unsupported file type: {input_path.suffix}")


def save_joblib(obj: object, path: Path | str) -> None:
    """Persist a Python object with joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, output_path)


def load_joblib(path: Path | str) -> object:
    """Load a joblib object."""
    return joblib.load(Path(path))


def safe_datetime(series: pd.Series) -> pd.Series:
    """Convert a series to pandas datetime while tolerating invalid values."""
    converted = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    return converted.dt.tz_localize(None)


def make_transaction_ids(prefix: str, length: int, start: int = 0) -> pd.Series:
    """Create deterministic transaction ids for datasets that do not provide one."""
    return pd.Series([f"{prefix}_{idx:08d}" for idx in range(start, start + length)])


def numeric_columns(df: pd.DataFrame, exclude: Iterable[str] | None = None) -> list[str]:
    """Return numeric columns, excluding any provided names."""
    excluded = set(exclude or [])
    return [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]


def categorical_columns(df: pd.DataFrame, exclude: Iterable[str] | None = None) -> list[str]:
    """Return categorical columns, excluding any provided names."""
    excluded = set(exclude or [])
    return [
        column
        for column in df.select_dtypes(include=["object", "category", "bool"]).columns
        if column not in excluded
    ]
