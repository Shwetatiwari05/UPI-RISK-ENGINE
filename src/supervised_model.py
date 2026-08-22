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
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.data_preprocessing import PreprocessingConfig, UPITransactionPreprocessor
from src.feature_engineering import engineer_features
from src.probability_calibration import (
    MIN_CALIBRATION_POSITIVES,
    apply_isotonic_calibration,
    calibration_diagnostics,
    choose_alert_threshold,
    fit_isotonic_calibrator,
    save_isotonic_calibrator,
    save_reliability_diagram,
)
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
    evaluation_frame: pd.DataFrame | None = None,
    calibration_frame: pd.DataFrame | None = None,
    alert_min_precision: float = 0.25,
) -> dict[str, Any]:
    """Train XGBoost and Random Forest models once on a bounded batch.

    When ``preprocessed=True``, ``df`` must contain ``transaction_id``,
    ``fraud_label``, and already transformed feature columns. This path lets
    the Parquet pipeline reuse one fit-time preprocessor consistently.

    When ``evaluation_frame`` is provided (the processed evaluation-period
    rows), models train on the full training frame and are scored strictly
    out-of-time; otherwise an in-sample random split is used.

    When ``calibration_frame`` is provided, it must hold natural-prevalence
    training-period rows that are disjoint from the training sample. Each
    model is then wrapped with an isotonic calibrator mapping raw scores to
    real-world fraud probabilities, and a supervised alert threshold is chosen
    from the out-of-time precision/recall trade-off.
    """
    training_df = df.copy() if preprocessed else sample_training_data(
        df,
        max_rows=max_rows,
        random_state=random_state,
        legitimate_ratio=legitimate_ratio,
    )
    LOGGER.info("Supervised training dataset size: %s rows", len(training_df))
    if preprocessed:
        feature_frame = training_df.drop(
            columns=["transaction_id", "fraud_label", "period"],
            errors="ignore",
        )
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

    if evaluation_frame is not None:
        x_train, y_train = x, y
        x_test = _feature_matrix(evaluation_frame, feature_names)
        y_test = pd.to_numeric(
            evaluation_frame["fraud_label"], errors="coerce"
        ).fillna(0).astype(int)
        LOGGER.info(
            "Out-of-time evaluation: train=%s rows (fraud=%s), "
            "test=%s rows (fraud=%s)",
            len(x_train),
            int(y_train.sum()),
            len(x_test),
            int(y_test.sum()),
        )
    else:
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
        scale_pos_weight=1,
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
        calibration_info = _calibrate_model(
            name=name,
            model=model,
            raw_probability=probability,
            y_test=y_test,
            calibration_frame=calibration_frame,
            feature_names=feature_names,
            alert_min_precision=alert_min_precision,
            model_path=model_path,
            report_path=report_path,
        )
        if calibration_info:
            metrics["calibration"] = calibration_info
        results[name] = metrics

        save_path = model_path / ("xgboost_model.pkl" if name == "xgboost" else "random_forest.pkl")
        save_joblib(model, save_path)
        save_confusion_matrix(y_test, prediction, report_path / f"{name}_confusion_matrix.png")
        save_roc_curve(y_test, probability, report_path / f"{name}_roc_curve.png")
        save_pr_curve(y_test, probability, report_path / f"{name}_pr_curve.png")
        save_feature_importance(
            model,
            feature_names,
            report_path / f"{name}_feature_importance.png",
        )

    save_joblib(preprocessor, model_path / "scaler.pkl")
    save_joblib(preprocessor, model_path / "preprocessor.pkl")
    return results


def _feature_matrix(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Select engineered feature columns in training order as a float matrix."""
    features = frame.drop(
        columns=["transaction_id", "fraud_label", "period"],
        errors="ignore",
    )
    missing = [column for column in feature_names if column not in features.columns]
    if missing:
        raise ValueError(f"Frame is missing engineered columns: {missing}")
    return features[feature_names].to_numpy(dtype=np.float32, copy=False)


def _calibrate_model(
    name: str,
    model: Any,
    raw_probability: np.ndarray,
    y_test: pd.Series,
    calibration_frame: pd.DataFrame | None,
    feature_names: list[str],
    alert_min_precision: float,
    model_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Fit and evaluate an isotonic calibrator plus alert threshold for one model."""
    if calibration_frame is None:
        return {}
    if "fraud_label" not in calibration_frame.columns:
        raise ValueError("calibration_frame must contain fraud_label.")
    x_cal = _feature_matrix(calibration_frame, feature_names)
    y_cal = pd.to_numeric(calibration_frame["fraud_label"], errors="coerce").fillna(0).astype(int)
    positive_count = int(y_cal.sum())
    if len(np.unique(y_cal)) < 2 or positive_count < MIN_CALIBRATION_POSITIVES:
        LOGGER.warning(
            "Skipping %s isotonic calibration: only %s positives in the "
            "calibration frame (need at least %s).",
            name,
            positive_count,
            MIN_CALIBRATION_POSITIVES,
        )
        return {}

    LOGGER.info(
        "Fitting isotonic calibrator for %s on %s rows (%s fraud)",
        name,
        len(x_cal),
        positive_count,
    )
    calibrator = fit_isotonic_calibrator(model, x_cal, y_cal.to_numpy())
    calibrated_probability = apply_isotonic_calibration(raw_probability, calibrator)
    diagnostics = calibration_diagnostics(y_test, raw_probability, calibrated_probability)
    if not diagnostics["monotonicity_preserved"]:
        LOGGER.warning(
            "%s isotonic calibrator is not monotone in the raw scores on "
            "evaluation rows; calibration output is suspect.",
            name,
        )
    save_reliability_diagram(
        y_test,
        raw_probability,
        calibrated_probability,
        report_path / f"{name}_reliability_diagram.png",
    )

    artifact = (
        "xgboost_model_calibrator.pkl" if name == "xgboost" else f"{name}_calibrator.pkl"
    )
    save_isotonic_calibrator(calibrator, model_path / artifact)

    alert = choose_alert_threshold(
        y_test.to_numpy(),
        calibrated_probability,
        min_precision=alert_min_precision,
    )
    LOGGER.info(
        "%s alert threshold %.4f -> precision=%.3f recall=%.3f (floor met=%s)",
        name,
        alert["alert_threshold"],
        alert["precision_at_alert"],
        alert["recall_at_alert"],
        alert["met_precision_floor"],
    )
    return {
        **diagnostics,
        **alert,
        "calibration_rows": int(len(x_cal)),
        "calibration_fraud_rows": positive_count,
        "calibrator_artifact": artifact,
    }


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
    pr_auc = (
        average_precision_score(y_true, y_probability) if y_true.nunique() > 1 else np.nan
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def predict_fraud_probability(
    model: Any,
    preprocessor: UPITransactionPreprocessor,
    df: pd.DataFrame,
    feature_context: dict[str, object] | None = None,
    live_context: dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Generate fraud probability and binary prediction for new transactions."""
    engineered = engineer_features(df, context=feature_context, live_context=live_context)
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


def save_pr_curve(y_true: pd.Series, y_probability: np.ndarray, path: Path) -> None:
    """Save a precision-recall curve plot."""
    if y_true.nunique() < 2:
        return
    precision, recall, _ = precision_recall_curve(y_true, y_probability)
    pr_auc = average_precision_score(y_true, y_probability)
    prevalence = float(np.mean(y_true))
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    plt.axhline(prevalence, linestyle="--", color="gray", label=f"Baseline = {prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
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
