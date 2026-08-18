"""Supervised fraud detection with XGBoost and Random Forest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.data_preprocessing import PreprocessingConfig, UPITransactionPreprocessor
from src.feature_engineering import engineer_features
from src.sampling import balanced_binary_sample
from src.utils import MODELS_DIR, REPORTS_DIR, get_logger, save_joblib


try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - handled at runtime for friendly errors
    XGBClassifier = None


LOGGER = get_logger(__name__)


def train_supervised_models(
    df: pd.DataFrame,
    model_dir: Path | str = MODELS_DIR,
    report_dir: Path | str = REPORTS_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    max_rows: int = 500_000,
    legitimate_ratio: float = 3.0,
    preprocessor: UPITransactionPreprocessor | None = None,
    preprocessed: bool = False,
) -> dict[str, Any]:
    """Train XGBoost and Random Forest models once on a bounded batch.

    When ``preprocessed=True``, ``df`` must contain ``transaction_id``,
    ``fraud_label``, and already transformed feature columns. This path lets
    the Parquet pipeline reuse one fit-time preprocessor consistently.
    """
    training_df = df.copy() if preprocessed else sample_training_data(
        df,
        max_rows=max_rows,
        random_state=random_state,
        legitimate_ratio=legitimate_ratio,
    )
    LOGGER.info("Supervised training dataset size: %s rows", len(training_df))
    if preprocessed:
        feature_frame = training_df.drop(columns=["transaction_id", "fraud_label"], errors="ignore")
        x = feature_frame.to_numpy(dtype=np.float32, copy=False)
        y = pd.to_numeric(training_df["fraud_label"], errors="coerce").fillna(0).astype(int)
        feature_names = list(feature_frame.columns)
        if preprocessor is None:
            raise ValueError("A fitted preprocessor is required for preprocessed training.")
    else:
        engineered = engineer_features(training_df)
        preprocessor = preprocessor or UPITransactionPreprocessor(
            PreprocessingConfig(scaler="robust", encoder="onehot")
        )
        LOGGER.info("Fitting supervised preprocessor")
        x, y = preprocessor.fit_transform(engineered)
        feature_names = preprocessor.get_feature_names()
    if y is None:
        raise ValueError("fraud_label is required for supervised training.")
    LOGGER.info("Supervised feature matrix shape: %s", x.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
    }
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Install requirements.txt first.")

    models["xgboost"] = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        scale_pos_weight=_scale_pos_weight(y_train),
        random_state=random_state,
        n_jobs=-1,
    )

    results: dict[str, Any] = {}
    model_path = Path(model_dir)
    report_path = Path(report_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        LOGGER.info("Training %s", name)
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        metrics = evaluate_classifier(y_test, prediction, probability)
        results[name] = metrics

        save_path = model_path / ("xgboost_model.pkl" if name == "xgboost" else "random_forest.pkl")
        save_joblib(model, save_path)
        save_confusion_matrix(y_test, prediction, report_path / f"{name}_confusion_matrix.png")
        save_roc_curve(y_test, probability, report_path / f"{name}_roc_curve.png")
        save_feature_importance(
            model,
            feature_names,
            report_path / f"{name}_feature_importance.png",
        )

    save_joblib(preprocessor, model_path / "scaler.pkl")
    save_joblib(preprocessor, model_path / "preprocessor.pkl")
    return results


def sample_training_data(
    df: pd.DataFrame,
    max_rows: int,
    random_state: int,
    legitimate_ratio: float = 3.0,
) -> pd.DataFrame:
    """Preserve fraud and stratify the legitimate majority for training."""
    if "fraud_label" in df.columns and df["fraud_label"].nunique() > 1:
        sampled = balanced_binary_sample(
            df,
            target_column="fraud_label",
            max_rows=max_rows,
            random_state=random_state,
            legitimate_ratio=legitimate_ratio,
        )
        counts = sampled["fraud_label"].value_counts().to_dict()
        LOGGER.info(
            "Fraud-preserving supervised sample class counts: %s, legitimate_ratio=%.2f",
            counts,
            legitimate_ratio,
        )
        return sampled
    return df.sample(n=min(max_rows, len(df)), random_state=random_state, replace=False).reset_index(drop=True)


def evaluate_classifier(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, Any]:
    """Calculate required supervised fraud metrics."""
    roc_auc = roc_auc_score(y_true, y_probability) if y_true.nunique() > 1 else np.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def predict_fraud_probability(
    model: Any,
    preprocessor: UPITransactionPreprocessor,
    df: pd.DataFrame,
    feature_context: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Generate fraud probability and binary prediction for new transactions."""
    engineered = engineer_features(df, context=feature_context)
    x = preprocessor.transform(engineered)
    probability = model.predict_proba(x)[:, 1]
    output = df[["transaction_id"]].copy()
    output["fraud_probability"] = probability
    output["fraud_prediction"] = (probability >= 0.5).astype(int)
    output["confidence_score"] = np.maximum(probability, 1 - probability)
    return output


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, path: Path) -> None:
    """Save a confusion matrix plot."""
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_roc_curve(y_true: pd.Series, y_probability: np.ndarray, path: Path) -> None:
    """Save a ROC curve plot."""
    if y_true.nunique() < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_probability)
    auc = roc_auc_score(y_true, y_probability)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_feature_importance(model: Any, feature_names: list[str], path: Path, top_n: int = 20) -> None:
    """Save feature importance plot when the model exposes importances."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=importances, x="importance", y="feature")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _scale_pos_weight(y: pd.Series) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0:
        return 1.0
    return max(1.0, negatives / positives)
