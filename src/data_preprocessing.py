"""Reusable preprocessing pipeline for mapped UPI transaction data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from src.schema_mapping import COMMON_SCHEMA, validate_common_schema
from src.utils import categorical_columns, numeric_columns, safe_datetime


ScalerName = Literal["standard", "minmax", "robust"]
EncoderName = Literal["onehot", "label"]


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing mapped and engineered transaction data."""

    scaler: ScalerName = "robust"
    encoder: EncoderName = "onehot"
    outlier_clip_quantile: float = 0.995
    target_column: str = "fraud_label"
    max_onehot_cardinality: int = 50


class UPITransactionPreprocessor:
    """Preprocess transaction features for supervised and anomaly models."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()
        self.scaler = None
        self.numeric_imputer: SimpleImputer | None = None
        self.onehot_imputer: SimpleImputer | None = None
        self.onehot_encoder: OneHotEncoder | None = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_columns_: list[str] = []
        self.numeric_features_: list[str] = []
        self.low_cardinality_features_: list[str] = []
        self.high_cardinality_features_: list[str] = []
        self.output_feature_names_: list[str] = []
        self.outlier_bounds_: dict[str, tuple[float, float]] = {}

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.Series | None]:
        """Fit preprocessing steps and return transformed features with labels."""
        prepared, y = self._prepare_dataframe(df, fit=True)
        if self.config.encoder == "label":
            transformed = self._fit_transform_label_encoding(prepared)
        else:
            transformed = self._fit_transform_mixed_encoding(prepared)
        return transformed.astype(np.float32, copy=False), y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted preprocessing steps."""
        prepared, _ = self._prepare_dataframe(df, fit=False)
        if self.config.encoder == "label":
            transformed = self._transform_label_encoding(prepared)
        else:
            transformed = self._transform_mixed_encoding(prepared)
        return transformed.astype(np.float32, copy=False)

    def get_feature_names(self) -> list[str]:
        """Return feature names after preprocessing where available."""
        return self.output_feature_names_ or self.feature_columns_

    def _prepare_dataframe(
        self,
        df: pd.DataFrame,
        fit: bool,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        working = df.copy()

        if set(COMMON_SCHEMA).issubset(working.columns):
            validate_common_schema(working[COMMON_SCHEMA])

        working = remove_duplicates(working)
        working = format_timestamp(working)
        working = handle_missing_values(working)
        excluded = [self.config.target_column]
        if fit:
            self.outlier_bounds_ = calculate_outlier_bounds(
                working,
                quantile=self.config.outlier_clip_quantile,
                exclude=excluded,
            )
        working = normalize_outliers(
            working,
            quantile=self.config.outlier_clip_quantile,
            exclude=excluded,
            bounds=self.outlier_bounds_ if self.outlier_bounds_ else None,
        )

        y = None
        if self.config.target_column in working.columns:
            y = working[self.config.target_column].astype(int)

        drop_columns = [
            self.config.target_column,
            "transaction_id",
            "timestamp",
            "period",
            "source_dataset",
        ]
        features = working.drop(columns=[col for col in drop_columns if col in working.columns])

        if fit:
            self.feature_columns_ = list(features.columns)
        else:
            for column in self.feature_columns_:
                if column not in features.columns:
                    features[column] = 0
            features = features[self.feature_columns_]

        self.numeric_features_ = numeric_columns(features)
        categorical_features = categorical_columns(features)
        if fit:
            self.low_cardinality_features_ = [
                column
                for column in categorical_features
                if features[column].nunique(dropna=False) <= self.config.max_onehot_cardinality
            ]
            self.high_cardinality_features_ = [
                column for column in categorical_features if column not in self.low_cardinality_features_
            ]
        return features, y

    def _fit_transform_mixed_encoding(self, features: pd.DataFrame) -> np.ndarray:
        numeric_array = self._fit_numeric_block(features)
        onehot_array = self._fit_onehot_block(features)
        high_card_array = self._fit_high_cardinality_block(features)
        self.output_feature_names_ = (
            list(self.numeric_features_)
            + self._onehot_feature_names()
            + [f"{column}_label" for column in self.high_cardinality_features_]
        )
        return _combine_feature_blocks([numeric_array, onehot_array, high_card_array])

    def _transform_mixed_encoding(self, features: pd.DataFrame) -> np.ndarray:
        numeric_array = self._transform_numeric_block(features)
        onehot_array = self._transform_onehot_block(features)
        high_card_array = self._transform_high_cardinality_block(features)
        return _combine_feature_blocks([numeric_array, onehot_array, high_card_array])

    def _fit_numeric_block(self, features: pd.DataFrame) -> np.ndarray:
        if not self.numeric_features_:
            return np.empty((len(features), 0), dtype=np.float32)
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.scaler = _get_scaler(self.config.scaler)
        numeric_values = self.numeric_imputer.fit_transform(features[self.numeric_features_])
        return self.scaler.fit_transform(numeric_values)

    def _transform_numeric_block(self, features: pd.DataFrame) -> np.ndarray:
        if not self.numeric_features_:
            return np.empty((len(features), 0), dtype=np.float32)
        if self.numeric_imputer is None or self.scaler is None:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        numeric_values = self.numeric_imputer.transform(features[self.numeric_features_])
        return self.scaler.transform(numeric_values)

    def _fit_onehot_block(self, features: pd.DataFrame) -> np.ndarray:
        if not self.low_cardinality_features_:
            return np.empty((len(features), 0), dtype=np.float32)
        self.onehot_imputer = SimpleImputer(strategy="most_frequent")
        self.onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        onehot_values = self.onehot_imputer.fit_transform(features[self.low_cardinality_features_])
        return self.onehot_encoder.fit_transform(onehot_values)

    def _transform_onehot_block(self, features: pd.DataFrame) -> np.ndarray:
        if not self.low_cardinality_features_:
            return np.empty((len(features), 0), dtype=np.float32)
        if self.onehot_imputer is None or self.onehot_encoder is None:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        onehot_values = self.onehot_imputer.transform(features[self.low_cardinality_features_])
        return self.onehot_encoder.transform(onehot_values)

    def _fit_high_cardinality_block(self, features: pd.DataFrame) -> np.ndarray:
        if not self.high_cardinality_features_:
            return np.empty((len(features), 0), dtype=np.float32)
        encoded = []
        for column in self.high_cardinality_features_:
            encoder = LabelEncoder()
            values = features[column].astype(str).fillna("Unknown")
            encoded.append(encoder.fit_transform(values).astype(np.float32))
            self.label_encoders[column] = encoder
        return np.column_stack(encoded)

    def _transform_high_cardinality_block(self, features: pd.DataFrame) -> np.ndarray:
        if not self.high_cardinality_features_:
            return np.empty((len(features), 0), dtype=np.float32)
        encoded = []
        for column in self.high_cardinality_features_:
            encoder = self.label_encoders[column]
            known = set(encoder.classes_)
            values = features[column].astype(str).fillna("Unknown")
            fallback = "Unknown" if "Unknown" in known else encoder.classes_[0]
            sanitized = values.where(values.isin(known), fallback)
            encoded.append(encoder.transform(sanitized).astype(np.float32))
        return np.column_stack(encoded)

    def _onehot_feature_names(self) -> list[str]:
        if self.onehot_encoder is None or not self.low_cardinality_features_:
            return []
        return list(self.onehot_encoder.get_feature_names_out(self.low_cardinality_features_))

    def _fit_transform_label_encoding(self, features: pd.DataFrame) -> np.ndarray:
        encoded = features.copy()
        for column in categorical_columns(encoded):
            encoder = LabelEncoder()
            encoded[column] = encoder.fit_transform(encoded[column].astype(str))
            self.label_encoders[column] = encoder
        scaler = _get_scaler(self.config.scaler)
        self.scaler = scaler
        self.output_feature_names_ = list(encoded.columns)
        return self.scaler.fit_transform(encoded)

    def _transform_label_encoding(self, features: pd.DataFrame) -> np.ndarray:
        encoded = features.copy()
        for column, encoder in self.label_encoders.items():
            if column in encoded.columns:
                known = set(encoder.classes_)
                encoded[column] = encoded[column].astype(str).apply(
                    lambda value: value if value in known else "Unknown"
                )
                if "Unknown" not in known:
                    encoded[column] = encoded[column].where(encoded[column].isin(known), encoder.classes_[0])
                encoded[column] = encoder.transform(encoded[column])
        if self.scaler is None:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        return self.scaler.transform(encoded)


def _combine_feature_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    valid_blocks = [block for block in blocks if block.size > 0]
    if not valid_blocks:
        return np.empty((0, 0), dtype=np.float32)
    return np.hstack(valid_blocks)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Validate transaction identity without silently dropping records."""
    if "transaction_id" in df.columns:
        duplicate_count = int(df["transaction_id"].duplicated().sum())
        if duplicate_count:
            raise ValueError(
                f"Found {duplicate_count} duplicate transaction IDs during preprocessing."
            )
        return df.reset_index(drop=True)
    return df.reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with simple research-friendly defaults."""
    filled = df.copy()
    for column in filled.columns:
        if pd.api.types.is_numeric_dtype(filled[column]):
            # A one-row prediction has no previous-transaction value, so some
            # engineered numeric columns can be entirely NaN. Avoid calling
            # median on an empty valid series because NumPy emits a warning.
            valid_values = filled[column].dropna()
            fallback = valid_values.median() if not valid_values.empty else 0.0
            filled[column] = filled[column].fillna(fallback)
        elif pd.api.types.is_datetime64_any_dtype(filled[column]):
            filled[column] = filled[column].fillna(pd.Timestamp("2024-01-01"))
        elif pd.api.types.is_categorical_dtype(filled[column]):
            if "Unknown" not in filled[column].cat.categories:
                filled[column] = filled[column].cat.add_categories(["Unknown"])
            filled[column] = filled[column].fillna("Unknown")
        else:
            filled[column] = filled[column].fillna("Unknown")
    return filled


def format_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp values are datetime and add basic timestamp fallbacks."""
    formatted = df.copy()
    if "timestamp" in formatted.columns:
        formatted["timestamp"] = safe_datetime(formatted["timestamp"])
        formatted["timestamp"] = formatted["timestamp"].fillna(pd.Timestamp("2024-01-01"))
    return formatted


def normalize_outliers(
    df: pd.DataFrame,
    quantile: float = 0.995,
    exclude: list[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Clip numeric outliers using fit-time bounds when supplied."""
    clipped = df.copy()
    excluded = set(exclude or [])
    active_bounds = bounds or calculate_outlier_bounds(clipped, quantile, excluded)
    for column in numeric_columns(clipped, exclude=excluded):
        lower, upper = active_bounds.get(column, (None, None))
        if pd.notna(lower) and pd.notna(upper) and lower < upper:
            clipped[column] = clipped[column].clip(lower, upper)
    return clipped


def calculate_outlier_bounds(
    df: pd.DataFrame,
    quantile: float = 0.995,
    exclude: list[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Calculate numeric clipping bounds once for consistent chunk transforms."""
    excluded = set(exclude or [])
    result: dict[str, tuple[float, float]] = {}
    for column in numeric_columns(df, exclude=excluded):
        lower = df[column].quantile(1 - quantile)
        upper = df[column].quantile(quantile)
        if pd.notna(lower) and pd.notna(upper):
            result[column] = (float(lower), float(upper))
    return result


def _get_scaler(name: ScalerName):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError(f"Unsupported scaler: {name}")
