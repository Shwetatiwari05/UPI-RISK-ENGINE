"""Unsupervised anomaly detection with Isolation Forest and LOF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.data_preprocessing import PreprocessingConfig, UPITransactionPreprocessor
from src.feature_engineering import engineer_features
from src.sampling import stratified_sample
from src.utils import MODELS_DIR, REPORTS_DIR, get_logger, save_joblib


LOGGER = get_logger(__name__)


def train_anomaly_models(
    df: pd.DataFrame,
    model_dir: Path | str = MODELS_DIR,
    report_dir: Path | str = REPORTS_DIR,
    contamination: float = 0.05,
    random_state: int = 42,
    max_rows: int = 200_000,
    preprocessor: UPITransactionPreprocessor | None = None,
    preprocessed: bool = False,
) -> dict[str, pd.DataFrame]:
    """Train Isolation Forest and LOF once on a bounded batch."""
    training_df = df.copy() if preprocessed else sample_anomaly_data(
        df,
        max_rows=max_rows,
        random_state=random_state,
    )
    LOGGER.info("Anomaly training dataset size: %s rows", len(training_df))
    if preprocessed:
        feature_frame = training_df.drop(
            columns=["transaction_id", "fraud_label", "period"],
            errors="ignore",
        )
        x = feature_frame.to_numpy(dtype=np.float32, copy=False)
        if preprocessor is None:
            raise ValueError("A fitted preprocessor is required for preprocessed training.")
    else:
        engineered = engineer_features(training_df)
        preprocessor = preprocessor or UPITransactionPreprocessor(
            PreprocessingConfig(scaler="robust", encoder="onehot")
        )
        LOGGER.info("Fitting anomaly preprocessor")
        x, _ = preprocessor.fit_transform(engineered)
    LOGGER.info("Anomaly feature matrix shape: %s", x.shape)

    isolation_forest = IsolationForest(
        n_estimators=250,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    lof = LocalOutlierFactor(
        n_neighbors=35,
        contamination=contamination,
        novelty=True,
        n_jobs=-1,
    )

    LOGGER.info("Training isolation_forest")
    isolation_forest.fit(x)
    LOGGER.info("Training lof")
    lof.fit(x)

    results = {
        "isolation_forest": score_anomaly_model(isolation_forest, x, training_df["transaction_id"]),
        "lof": score_anomaly_model(lof, x, training_df["transaction_id"]),
    }

    model_path = Path(model_dir)
    report_path = Path(report_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    save_joblib(isolation_forest, model_path / "isolation_forest.pkl")
    save_joblib(lof, model_path / "lof_model.pkl")
    save_joblib(preprocessor, model_path / "anomaly_preprocessor.pkl")

    save_anomaly_score_plot(results["isolation_forest"], report_path / "isolation_forest_scores.png")
    save_anomaly_score_plot(results["lof"], report_path / "lof_scores.png")
    save_anomaly_comparison_plot(results, report_path / "anomaly_model_comparison.png")
    return results


def sample_anomaly_data(
    df: pd.DataFrame,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    """Use a bounded sample for anomaly training on very large datasets."""
    if "fraud_label" in df.columns and df["fraud_label"].nunique() > 1:
        sampled = stratified_sample(
            df,
            target_column="fraud_label",
            max_rows=max_rows,
            random_state=random_state,
        )
        counts = sampled["fraud_label"].value_counts().to_dict()
        LOGGER.info("Stratified anomaly sample class counts: %s", counts)
        return sampled
    return df.sample(n=min(max_rows, len(df)), random_state=random_state, replace=False).reset_index(drop=True)


def score_anomaly_model(model: Any, x: np.ndarray, transaction_ids: pd.Series) -> pd.DataFrame:
    """Generate anomaly score and anomaly label for a fitted model."""
    raw_scores = model.score_samples(x)
    anomaly_scores = -raw_scores
    labels = model.predict(x)
    output = pd.DataFrame(
        {
            "transaction_id": transaction_ids.values,
            "anomaly_score": anomaly_scores,
            "anomaly_label": np.where(labels == -1, "Anomaly", "Normal"),
        }
    )
    output["anomaly_confidence"] = _min_max_confidence(output["anomaly_score"])
    return output


def predict_anomaly(
    model: Any,
    preprocessor: UPITransactionPreprocessor,
    df: pd.DataFrame,
    feature_context: dict[str, object] | None = None,
    live_context: dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Generate anomaly output for new transactions."""
    engineered = engineer_features(df, context=feature_context, live_context=live_context)
    x = preprocessor.transform(engineered)
    return score_anomaly_model(model, x, df["transaction_id"])


def save_anomaly_score_plot(scores: pd.DataFrame, path: Path) -> None:
    """Save an anomaly score distribution plot."""
    plt.figure(figsize=(7, 4))
    sns.histplot(scores["anomaly_score"], bins=40, kde=True)
    plt.xlabel("Anomaly score")
    plt.ylabel("Transaction count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_anomaly_comparison_plot(results: dict[str, pd.DataFrame], path: Path) -> None:
    """Save a model comparison chart based on anomaly label counts."""
    rows = []
    for model_name, frame in results.items():
        counts = frame["anomaly_label"].value_counts()
        rows.extend(
            {
                "model": model_name,
                "label": label,
                "count": count,
            }
            for label, count in counts.items()
        )
    comparison = pd.DataFrame(rows)

    plt.figure(figsize=(7, 4))
    sns.barplot(data=comparison, x="model", y="count", hue="label")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _min_max_confidence(scores: pd.Series) -> pd.Series:
    minimum = scores.min()
    maximum = scores.max()
    if minimum == maximum:
        return pd.Series(np.ones(len(scores)), index=scores.index)
    return (scores - minimum) / (maximum - minimum)
