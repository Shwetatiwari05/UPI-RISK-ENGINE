"""Prior correction for class-weighted fraud probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def correct_prior_probability(
    probability: float,
    population_fraud_rate: float,
    effective_training_fraud_rate: float = 0.50,
) -> float:
    """Convert a weighted-sample probability to the population prior.

    The supervised models use class weighting, so their probability output is
    learned under an approximately balanced effective class prior. This odds
    correction restores the fraud prevalence of the merged dataset.
    """
    p = float(np.clip(probability, 1e-7, 1.0 - 1e-7))
    target = float(np.clip(population_fraud_rate, 1e-7, 1.0 - 1e-7))
    effective = float(np.clip(effective_training_fraud_rate, 1e-7, 1.0 - 1e-7))
    odds = p / (1.0 - p)
    corrected_odds = odds * ((target / (1.0 - target)) / (effective / (1.0 - effective)))
    return float(corrected_odds / (1.0 + corrected_odds))


def calibrate_probability_array(values: Any, metadata: dict[str, Any]) -> np.ndarray:
    """Apply prior correction to a probability array."""
    target = float(metadata.get("population_fraud_rate", 0.0))
    effective = float(metadata.get("effective_training_fraud_rate", 0.50))
    return np.asarray(
        [correct_prior_probability(value, target, effective) for value in values],
        dtype=np.float64,
    )
