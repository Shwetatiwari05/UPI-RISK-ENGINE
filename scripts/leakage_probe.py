"""Leakage probe: legacy random-split evaluation vs honest out-of-time evaluation.

Trains two model pairs on identical training-period rows:
  Pair A (honest):      full training-period sample -> scored on the later
                        evaluation period (what deployment would face).
  Pair B (legacy):      random stratified 80/20 split of the same rows ->
                        scored on its own held-out 20% (the protocol that
                        produced the inflated historical metrics).

Run AFTER stage 2 (--preprocess) has written period labels:

    ./venv/bin/python3 scripts/leakage_probe.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.parquet_pipeline import (
    DEFAULT_CHUNK_SIZE,
    PROCESSED_DATA_DIR,
    PROCESSED_FEATURES_PARQUET_PATH,
    load_period_frame,
    sample_parquet_rows,
)
from src.supervised_model import train_supervised_models


def describe(name: str, frame_count: int, fraud_count: int) -> None:
    print(
        f"{name}: {frame_count:>9,} rows | fraud {fraud_count:>7,} "
        f"({fraud_count / frame_count:.2%})"
    )


def metric_line(label: str, metrics: dict) -> str:
    return (
        f"  {label:<28} ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
        f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  "
        f"f1={metrics['f1_score']:.4f}"
    )


def main() -> int:
    chunk_size = DEFAULT_CHUNK_SIZE
    preprocessor = joblib.load(PROCESSED_DATA_DIR / "chunk_preprocessor.pkl")

    train_sample = sample_parquet_rows(
        PROCESSED_FEATURES_PARQUET_PATH,
        max_rows=500_000,
        chunk_size=chunk_size,
        strategy="fraud_preserving",
        legitimate_ratio=3.0,
        period_value="train",
    )
    train_fraud = int((train_sample["fraud_label"] == 1).sum())
    describe("Training-period sample", len(train_sample), train_fraud)

    test_frame = load_period_frame(PROCESSED_FEATURES_PARQUET_PATH, "test", chunk_size)
    test_fraud = int((test_frame["fraud_label"] == 1).sum())
    describe("Evaluation period (out-of-time)", len(test_frame), test_fraud)

    with tempfile.TemporaryDirectory(prefix="leak_probe_") as temp_dir:
        tmp = Path(temp_dir)

        print("\n=== Pair A: trained on full training period -> evaluated out-of-time ===")
        out_of_time = train_supervised_models(
            train_sample,
            model_dir=tmp / "oot_models",
            report_dir=tmp / "oot_reports",
            max_rows=len(train_sample),
            preprocessor=preprocessor,
            preprocessed=True,
            evaluation_frame=test_frame,
        )
        del test_frame

        print("\n=== Pair B: legacy random 80/20 split of the same training rows ===")
        legacy = train_supervised_models(
            train_sample,
            model_dir=tmp / "legacy_models",
            report_dir=tmp / "legacy_reports",
            max_rows=len(train_sample),
            preprocessor=preprocessor,
            preprocessed=True,
        )

    print("\n================ LEAKAGE PROBE SUMMARY ================")
    for name in ["random_forest", "xgboost"]:
        legacy_metrics = legacy[name]
        oot_metrics = out_of_time[name]
        gap_roc = legacy_metrics["roc_auc"] - oot_metrics["roc_auc"]
        gap_pr = legacy_metrics["pr_auc"] - oot_metrics["pr_auc"]
        print(f"\n{name}:")
        print(metric_line("legacy random split (inflated)", legacy_metrics))
        print(metric_line("out-of-time evaluation (honest)", oot_metrics))
        print(
            f"  {'inflation gap':<28} ROC-AUC -{gap_roc:.4f}  PR-AUC -{gap_pr:.4f}"
        )
    print("\nInterpretation: the gap is the amount of reported performance that was")
    print("an artifact of evaluating on statistically overlapping transactions")
    print("(same senders/aggregates on both sides of the split), not generalization.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
