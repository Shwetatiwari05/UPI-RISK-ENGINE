"""Probability calibration for class-imbalanced fraud models.

Isotonic calibration maps raw model scores to real-world fraud probabilities
using a held-out, natural-prevalence calibration set. It replaces the legacy
analytic prior correction, which squashed probabilities so aggressively that
a 0.5 decision threshold sat at the extreme tail of the score distribution.

The legacy analytic functions are retained only as a fallback for deployments
whose model artifacts predate isotonic calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.utils import load_joblib, save_joblib


MIN_CALIBRATION_POSITIVES = 500
DEFAULT_RANK_GRID_BINS = 999


def correct_prior_probability(
    probability: float,
    population_fraud_rate: float,
    effective_training_fraud_rate: float = 0.50,
) -> float:
    """Legacy analytic odds correction from sampled prior to population prior.

    Kept for backward compatibility with artifacts trained before isotonic
    calibration existed. Prefer :func:`apply_isotonic_calibration` whenever a
    fitted calibrator artifact is available.
    """
    p = float(np.clip(probability, 1e-7, 1.0 - 1e-7))
    target = float(np.clip(population_fraud_rate, 1e-7, 1.0 - 1e-7))
    effective = float(np.clip(effective_training_fraud_rate, 1e-7, 1.0 - 1e-7))
    odds = p / (1.0 - p)
    corrected_odds = odds * ((target / (1.0 - target)) / (effective / (1.0 - effective)))
    return float(corrected_odds / (1.0 + corrected_odds))


def calibrate_probability_array(values: Any, metadata: dict[str, Any]) -> np.ndarray:
    """Apply the legacy analytic prior correction to a probability array."""
    target = float(metadata.get("population_fraud_rate", 0.0))
    effective = float(metadata.get("effective_training_fraud_rate", 0.50))
    return np.asarray(
        [correct_prior_probability(value, target, effective) for value in values],
        dtype=np.float64,
    )


def fit_isotonic_calibrator(
    model: Any,
    x_cal: np.ndarray,
    y_cal: np.ndarray | "Any",
) -> IsotonicRegression:
    """Fit an isotonic mapping from ``model`` scores to observed fraud rates.

    Uses ``CalibratedClassifierCV(method="isotonic", cv="prefit")`` so the base
    model is never refitted, then extracts the underlying monotone regressor.
    Falls back to an equivalent direct fit if the internal layout differs.
    """
    y = np.asarray(y_cal)
    if y.ndim != 1 or len(np.unique(y)) < 2:
        raise ValueError("Isotonic calibration needs binary labels with both classes.")
    calibrated = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
    try:
        calibrated.fit(x_cal, y)
        return _extract_isotonic_regressor(calibrated)
    except (AttributeError, IndexError, KeyError):
        raw_scores = np.asarray(model.predict_proba(x_cal))[:, 1]
        isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        isotonic.fit(raw_scores, y)
        return isotonic


def _extract_isotonic_regressor(calibrated: CalibratedClassifierCV) -> IsotonicRegression:
    """Pull the fitted isotonic step function out of the calibration wrapper."""
    wrapped_classifiers = list(getattr(calibrated, "calibrated_classifiers_", []))
    if not wrapped_classifiers:
        raise AttributeError("CalibratedClassifierCV exposed no calibrated classifiers.")
    calibrators = getattr(wrapped_classifiers[0], "calibrators", [])
    for candidate in calibrators:
        if isinstance(candidate, IsotonicRegression):
            return candidate
    raise AttributeError("No IsotonicRegression calibrator found on the wrapper.")


def apply_isotonic_calibration(values: Any, calibrator: IsotonicRegression) -> np.ndarray:
    """Map raw scores through a fitted isotonic calibrator."""
    raw = np.asarray(values, dtype=np.float64).ravel()
    mapped = np.asarray(calibrator.predict(raw), dtype=np.float64)
    return np.clip(mapped, 0.0, 1.0)


def save_isotonic_calibrator(calibrator: IsotonicRegression, path: Path | str) -> None:
    """Persist one fitted calibrator artifact."""
    save_joblib(calibrator, path)


def load_isotonic_calibrator(path: Path | str) -> IsotonicRegression | None:
    """Load a calibrator artifact, returning None when absent or unreadable."""
    if not Path(path).exists():
        return None
    loaded = load_joblib(path)
    return loaded if isinstance(loaded, IsotonicRegression) else None


def expected_calibration_error(
    y_true: Any,
    probabilities: Any,
    n_bins: int = 10,
) -> float:
    """Return ECE: probability-weighted absolute gap between confidence and accuracy."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    total = float(len(p))
    error = 0.0
    for bin_index in range(n_bins):
        mask = indices == bin_index
        if not mask.any():
            continue
        confidence = float(p[mask].mean())
        accuracy = float(y[mask].mean())
        error += (mask.sum() / total) * abs(confidence - accuracy)
    return float(error)


def calibration_diagnostics(
    y_true: Any,
    prob_raw: Any,
    prob_calibrated: Any,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compare raw versus calibrated probabilities on identical evaluation rows."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    raw = np.asarray(prob_raw, dtype=np.float64).ravel()
    calibrated = np.asarray(prob_calibrated, dtype=np.float64).ravel()
    unique_labels = np.unique(y)
    roc_auc_raw = roc_auc_score(y, raw) if len(unique_labels) > 1 else float("nan")
    roc_auc_calibrated = (
        roc_auc_score(y, calibrated) if len(unique_labels) > 1 else float("nan")
    )
    # The hard invariant is that calibration be a monotone non-decreasing map
    # of the raw score; flat steps may merge scores into ties.
    order = np.argsort(raw, kind="mergesort")
    monotonicity_preserved = bool(
        np.all(np.diff(calibrated[order]) >= -1e-9)
    )
    return {
        "calibration_method": "isotonic",
        "observed_fraud_rate": round(float(y.mean()), 8),
        "mean_predicted_raw": round(float(raw.mean()), 8),
        "mean_predicted_isotonic": round(float(calibrated.mean()), 8),
        "brier_raw": round(float(brier_score_loss(y, raw)), 8),
        "brier_isotonic": round(float(brier_score_loss(y, calibrated)), 8),
        "ece_raw": round(expected_calibration_error(y, raw, n_bins), 6),
        "ece_isotonic": round(expected_calibration_error(y, calibrated, n_bins), 6),
        "roc_auc_raw": round(float(roc_auc_raw), 6),
        "roc_auc_isotonic": round(float(roc_auc_calibrated), 6),
        "roc_auc_drift": round(
            float(abs(roc_auc_raw - roc_auc_calibrated)), 6
        ) if len(unique_labels) > 1 else float("nan"),
        "monotonicity_preserved": monotonicity_preserved,
    }


def save_reliability_diagram(
    y_true: Any,
    prob_raw: Any,
    prob_calibrated: Any,
    path: Path | str,
    n_bins: int = 10,
) -> None:
    """Plot observed fraud rate per predicted-probability bin, raw vs isotonic."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    curves = [
        ("Raw model output", np.asarray(prob_raw, dtype=np.float64).ravel(), "#d9534f"),
        ("Isotonic calibrated", np.asarray(prob_calibrated, dtype=np.float64).ravel(), "#0275d8"),
    ]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    for label, probabilities, color in curves:
        indices = np.clip(np.digitize(probabilities, bins[1:-1]), 0, n_bins - 1)
        observed = np.full(n_bins, np.nan)
        for bin_index in range(n_bins):
            mask = indices == bin_index
            if mask.any():
                observed[bin_index] = y[mask].mean()
        plt.plot(centers, observed, marker="o", color=color, label=label)
    prevalence = float(y.mean())
    plt.axhline(prevalence, linestyle=":", color="#5cb85c", label=f"Prevalence = {prevalence:.4f}")
    plt.xlabel("Mean predicted fraud probability")
    plt.ylabel("Observed fraud rate")
    plt.title("Reliability diagram")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def choose_alert_threshold(
    y_true: Any,
    probabilities: Any,
    min_precision: float = 0.25,
) -> dict[str, Any]:
    """Pick the lowest threshold meeting a precision floor, else best F1.

    Lower thresholds maximize recall, so among all feasible points the one with
    the smallest threshold (and therefore largest recall) is selected. When no
    threshold reaches ``min_precision``, the best-F1 operating point is returned
    and ``met_precision_floor`` is False.
    """
    y = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    positives = max(int(y.sum()), 1)

    order = np.argsort(-p, kind="mergesort")
    sorted_labels = y[order]
    true_positive = np.cumsum(sorted_labels)
    false_positive = np.cumsum(1 - sorted_labels)
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    recall = true_positive / positives
    f1 = np.where(
        precision + recall > 0,
        2 * precision * recall / np.maximum(precision + recall, 1e-12),
        0.0,
    )

    sorted_probs = p[order]

    def _threshold_at(index: int) -> float:
        boundary = sorted_probs[index]
        upper = sorted_probs[index - 1] if index > 0 else boundary + 1.0
        return float(round((upper + boundary) / 2.0, 8))

    def _report(index: int, met_floor: bool) -> dict[str, Any]:
        flagged = index + 1
        return {
            "alert_threshold": _threshold_at(index),
            "precision_at_alert": round(float(precision[index]), 6),
            "recall_at_alert": round(float(recall[index]), 6),
            "alerts_per_10k_rows": round(float(flagged / max(len(y), 1)) * 10_000, 2),
            "met_precision_floor": met_floor,
            "min_precision_target": round(float(min_precision), 4),
        }

    feasible = np.flatnonzero(precision >= min_precision)
    if feasible.size:
        return _report(int(feasible[-1]), True)
    return _report(int(np.argmax(f1)), False)


def build_calibration_metadata(
    per_model_results: dict[str, dict[str, Any]],
    population_fraud_rate: float,
    label_counts: dict[int, int],
    effective_training_fraud_rate: float = 0.25,
) -> dict[str, Any]:
    """Assemble the persisted fraud_probability_calibration.json payload."""
    per_model: dict[str, dict[str, Any]] = {}
    preferred_threshold: float | None = None
    for name in ("xgboost", "random_forest"):
        info = per_model_results.get(name, {}).get("calibration")
        if not info:
            continue
        artifact = "xgboost_model_calibrator.pkl" if name == "xgboost" else f"{name}_calibrator.pkl"
        per_model[name] = {
            "calibrator_artifact": artifact,
            **info,
        }
        if preferred_threshold is None:
            preferred_threshold = info.get("alert_threshold")

    metadata: dict[str, Any] = {
        "method": "isotonic",
        "population_fraud_rate": population_fraud_rate,
        "effective_training_fraud_rate": effective_training_fraud_rate,
        "label_counts": {str(key): value for key, value in label_counts.items()},
        "per_model": per_model,
    }
    if preferred_threshold is not None:
        metadata["supervised_alert_threshold"] = preferred_threshold
    return metadata


def build_score_rank_grid(
    scores: Any, bins: int = DEFAULT_RANK_GRID_BINS
) -> np.ndarray:
    """Build a quantile grid representing the reference score distribution.

    The grid lets serving code map a new score to its population percentile
    without keeping the full calibration sample in memory. Quantile spacing
    keeps resolution where the data actually lives.
    """
    values = np.asarray(scores, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot build a rank grid from an empty score array.")
    quantiles = np.linspace(0.0, 1.0, num=max(int(bins), 2))
    grid = np.unique(np.quantile(values, quantiles))
    return np.clip(grid, 0.0, 1.0)


def score_to_percentile(grid: Any, scores: Any) -> np.ndarray:
    """Map scores to their percentile within the persisted reference grid."""
    reference = np.sort(np.asarray(grid, dtype=np.float64).ravel())
    if reference.size < 2:
        raise ValueError("Rank grid needs at least two points.")
    values = np.clip(np.asarray(scores, dtype=np.float64).ravel(), 0.0, 1.0)
    positions = np.searchsorted(reference, values, side="right")
    return np.clip(positions / float(reference.size), 0.0, 1.0)


def save_score_rank_grid(
    grid: Any,
    output_path: Path | str,
    artifact: str,
    calibrator: str,
    reference_rows: int,
) -> Path:
    """Persist the supervised-score rank grid alongside model artifacts."""
    path = Path(output_path)
    payload = {
        "artifact": artifact,
        "calibrator_artifact": calibrator,
        "reference_rows": int(reference_rows),
        "grid": [round(float(value), 8) for value in np.asarray(grid).ravel()],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_score_rank_grid(path: Path | str) -> dict[str, Any] | None:
    """Load the persisted rank grid; returns None when absent or unreadable."""
    grid_path = Path(path)
    if not grid_path.exists():
        return None
    try:
        payload = json.loads(grid_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    grid = payload.get("grid")
    if not isinstance(grid, list) or len(grid) < 2:
        return None
    return payload
