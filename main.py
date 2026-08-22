"""Offline batch pipeline entry point for UPI fraud and anomaly research."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.anomaly_detection import train_anomaly_models
from src.fusion_model import tune_fusion_thresholds
from src.parquet_pipeline import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FIT_ROWS,
    DEFAULT_TEST_FRACTION,
    MAPPED_PARQUET_PATH,
    PROCESSED_FEATURES_PARQUET_PATH,
    ParquetPipelineConfig,
    load_period_frame,
    sample_parquet_rows,
    stream_preprocess_to_parquet,
    write_mapped_parquet,
)
from src.parquet_stats import (
    parquet_label_counts,
    parquet_period_label_counts,
)
from src.probability_calibration import build_calibration_metadata
from src.supervised_model import train_supervised_models
from src.utils import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    ensure_project_dirs,
    save_joblib,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for batch pipeline stages."""
    parser = argparse.ArgumentParser(description="Offline UPI fraud detection pipeline")
    parser.add_argument("--all", action="store_true", help="Run every implemented stage")
    parser.add_argument("--load", action="store_true", help="Load and map raw datasets")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--features", action="store_true", help="Run feature engineering")
    parser.add_argument("--train-supervised", action="store_true", help="Train supervised models")
    parser.add_argument("--train-anomaly", action="store_true", help="Train anomaly models")
    parser.add_argument(
        "--tune-thresholds",
        action="store_true",
        help="Tune fusion FRAUD_LIKELY thresholds on out-of-time data",
    )
    parser.add_argument(
        "--calibration-fraud-holdout",
        type=float,
        default=0.25,
        help=(
            "Fraction of train-period fraud rows reserved for isotonic "
            "calibration instead of model training (default: 0.25)"
        ),
    )
    parser.add_argument(
        "--alert-min-precision",
        type=float,
        default=0.25,
        help=(
            "Precision floor for the supervised alert threshold chosen after "
            "calibration (default: 0.25)"
        ),
    )
    parser.add_argument(
        "--fraud-likely-min-precision",
        type=float,
        default=0.40,
        help=(
            "Precision floor for the fused FRAUD_LIKELY operating point "
            "(default: 0.40)"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per Parquet/CSV processing chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--fit-rows",
        type=int,
        default=DEFAULT_FIT_ROWS,
        help=f"Bounded rows used to fit preprocessing state (default: {DEFAULT_FIT_ROWS})",
    )
    parser.add_argument(
        "--supervised-rows",
        type=int,
        default=500_000,
        help="Maximum supervised rows after fraud-preserving sampling",
    )
    parser.add_argument(
        "--legitimate-ratio",
        type=float,
        default=3.0,
        help="Legitimate rows selected per fraud row (default: 3.0)",
    )
    parser.add_argument(
        "--anomaly-rows",
        type=int,
        default=200_000,
        help="Bounded stratified rows for anomaly batch training",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help=(
            "Fraction of each source's most recent transactions held out as the "
            f"out-of-time evaluation period (default: {DEFAULT_TEST_FRACTION})"
        ),
    )
    parser.add_argument(
        "--mapped-parquet",
        type=Path,
        default=MAPPED_PARQUET_PATH,
        help="Output path for the unified mapped Parquet file",
    )
    return parser.parse_args()


def tune_and_save_fusion_thresholds(
    processed_path: Path,
    chunk_size: int,
    min_fraud_precision: float,
) -> dict[str, object] | None:
    """Sweep fused-score operating points on out-of-time rows and persist them.

    Scores the evaluation period with the best available supervised model plus
    its isotonic calibrator and the trained isolation forest (percentile-mapped
    against saved training scores), then writes models/fusion_thresholds.json
    for the API to pass into fuse_signals.
    """
    supervised_pairs = [
        ("xgboost_model.pkl", "xgboost_model_calibrator.pkl"),
        ("random_forest.pkl", "random_forest_calibrator.pkl"),
    ]
    pair = next(
        (
            (model_name, calibrator_name)
            for model_name, calibrator_name in supervised_pairs
            if (MODELS_DIR / model_name).exists() and (MODELS_DIR / calibrator_name).exists()
        ),
        None,
    )
    isolation_path = MODELS_DIR / "isolation_forest.pkl"
    if pair is None or not isolation_path.exists():
        return None

    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    feature_names = preprocessor.get_feature_names()
    frame = load_period_frame(processed_path, "test", chunk_size)
    y_test = pd.to_numeric(frame["fraud_label"], errors="coerce").fillna(0).astype(int)
    features = frame.drop(
        columns=["transaction_id", "fraud_label", "period"],
        errors="ignore",
    )
    x_test = features[feature_names].to_numpy(dtype=np.float32, copy=False)

    model = joblib.load(MODELS_DIR / pair[0])
    calibrator = joblib.load(MODELS_DIR / pair[1])
    raw_probability = model.predict_proba(x_test)[:, 1]
    calibrated_probability = np.clip(
        np.asarray(calibrator.predict(raw_probability), dtype=np.float64),
        0.0,
        1.0,
    )

    isolation_forest = joblib.load(isolation_path)
    anomaly_scores = -isolation_forest.score_samples(x_test)
    reference_scores_path = PROCESSED_DATA_DIR / "isolation_forest_anomaly_scores.csv"
    if reference_scores_path.exists():
        reference = pd.read_csv(reference_scores_path, usecols=["anomaly_score"])[
            "anomaly_score"
        ]
        reference = np.sort(pd.to_numeric(reference, errors="coerce").dropna().to_numpy())
        unusualness_percentile = np.searchsorted(
            reference, anomaly_scores, side="right"
        ) / max(len(reference), 1)
    else:
        ranks = pd.Series(anomaly_scores).rank(method="average").to_numpy()
        unusualness_percentile = (ranks - 0.5) / max(len(ranks), 1)
        print("Anomaly training scores missing; using within-test percentiles.")

    tuned = tune_fusion_thresholds(
        calibrated_probability,
        unusualness_percentile,
        y_test.to_numpy(),
        min_fraud_precision=min_fraud_precision,
    )
    tuned["tuned_at_utc"] = datetime.now(timezone.utc).isoformat()
    tuned["supervised_artifact"] = pair[0]
    output_path = MODELS_DIR / "fusion_thresholds.json"
    output_path.write_text(json.dumps(tuned, indent=2), encoding="utf-8")

    fraud_band = tuned["achieved"]["fraud_likely"]
    print(
        "FRAUD_LIKELY operating point: fusion>="
        f"{tuned['thresholds']['fraud_likely_fusion_threshold']:.4f}, "
        f"supervised gate={tuned['thresholds']['fraud_likely_requires_supervised_probability']:.4f} "
        f"-> precision={fraud_band['precision']:.3f}, recall={fraud_band['recall']:.3f}, "
        f"{fraud_band['alerts_per_10k_rows']:.2f} alerts/10k rows"
    )
    if not tuned["target_precision_met"]:
        print(
            f"WARNING: no operating point reached the {min_fraud_precision:.0%} precision "
            "floor on out-of-time data; best-F1 point was kept instead."
        )
    print(f"Saved fusion thresholds: {output_path}")
    return tuned


def _build_calibration_frame(
    processed_path: Path,
    config: ParquetPipelineConfig,
    calibration_fraud: pd.DataFrame | None,
    supervised_sample: pd.DataFrame,
    population_fraud_rate: float,
) -> pd.DataFrame | None:
    """Compose a natural-prevalence calibration frame disjoint from training.

    Combines the held-out train-period fraud rows with a uniform legitimate
    pool sized so the frame's prevalence matches the population fraud rate.
    Legitimate rows overlapping the training sample are excluded at sampling
    time via transaction ids.
    """
    if calibration_fraud is None or len(calibration_fraud) == 0:
        return None
    fraud_count = len(calibration_fraud)
    legit_needed = int(
        round(fraud_count * (1.0 - population_fraud_rate) / max(population_fraud_rate, 1e-9))
    )
    draw_rows = int(legit_needed / 0.97) + 10_000
    training_ids = (
        frozenset(supervised_sample["transaction_id"])
        if "transaction_id" in supervised_sample.columns
        else frozenset()
    )
    legit_draw = sample_parquet_rows(
        processed_path,
        max_rows=draw_rows,
        chunk_size=config.chunk_size,
        target_column="fraud_label",
        random_state=43,
        columns=None,
        strategy="uniform",
        period_value="train",
        exclude_transaction_ids=training_ids or None,
    )
    draw_labels = pd.to_numeric(
        legit_draw["fraud_label"], errors="coerce"
    ).fillna(0).astype(int)
    legitimate = legit_draw.loc[draw_labels == 0]
    if len(legitimate) > legit_needed:
        legitimate = legitimate.sample(n=legit_needed, random_state=43)
    frame = pd.concat([calibration_fraud, legitimate], ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
    actual_prevalence = float(
        pd.to_numeric(frame["fraud_label"], errors="coerce").mean()
    )
    print(
        f"Isotonic calibration frame: {frame.shape} "
        f"(fraud={fraud_count}, legitimate={len(legitimate)}), "
        f"prevalence target={population_fraud_rate:.4%}, actual={actual_prevalence:.4%}"
    )
    return frame


def main() -> int:
    """Run selected offline batch stages."""
    ensure_project_dirs()
    args = parse_args()

    run_all = args.all or not any(
        [
            args.load,
            args.preprocess,
            args.features,
            args.train_supervised,
            args.train_anomaly,
            args.tune_thresholds,
        ]
    )

    config = ParquetPipelineConfig(
        chunk_size=args.chunk_size,
        fit_rows=args.fit_rows,
        mapped_path=args.mapped_parquet,
        processed_path=PROCESSED_FEATURES_PARQUET_PATH,
    )

    mapped_path = config.mapped_path
    preprocessor = None
    pipeline_report_path = REPORTS_DIR / "pipeline_row_counts.json"
    pipeline_report: dict[str, object] = {}
    if pipeline_report_path.exists():
        try:
            pipeline_report = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pipeline_report = {}

    def ensure_mapped_parquet() -> Path:
        if not mapped_path.exists():
            print("Creating unified compressed Parquet dataset...")
            stats = write_mapped_parquet(
                raw_dir=PROJECT_ROOT / "data" / "raw",
                output_path=mapped_path,
                chunk_size=config.chunk_size,
                compression=config.compression,
            )
            print(f"Mapped Parquet: {stats['rows']} rows in {stats['chunks']} chunks")
        return mapped_path

    def ensure_processed_parquet() -> Path:
        nonlocal preprocessor
        ensure_mapped_parquet()
        saved_preprocessor = PROCESSED_DATA_DIR / "chunk_preprocessor.pkl"
        if (
            preprocessor is None
            and config.processed_path.exists()
            and saved_preprocessor.exists()
        ):
            preprocessor = joblib.load(saved_preprocessor)
            print("Reusing existing processed Parquet and fitted preprocessor.")
            return config.processed_path
        if preprocessor is None or not config.processed_path.exists():
            preprocessor, stats = stream_preprocess_to_parquet(
                mapped_parquet_path=mapped_path,
                output_path=config.processed_path,
                preprocessor=preprocessor,
                fit_rows=config.fit_rows,
                chunk_size=config.chunk_size,
                test_fraction=args.test_fraction,
            )
            save_joblib(preprocessor, PROJECT_ROOT / "models" / "preprocessor.pkl")
            save_joblib(preprocessor, PROJECT_ROOT / "models" / "scaler.pkl")
            print(
                "Processed Parquet: "
                f"read={stats['rows_read']} removed={stats['rows_removed']} "
                f"written={stats['rows_written']} chunks={stats['chunks']}"
            )
            pipeline_report["processed_parquet"] = stats
        return config.processed_path

    if run_all or args.load:
        print("Stage 1/6: Converting raw datasets to compressed Parquet chunks...")
        stats = write_mapped_parquet(
            raw_dir=PROJECT_ROOT / "data" / "raw",
            output_path=mapped_path,
            chunk_size=config.chunk_size,
            compression=config.compression,
        )
        reports = {
            "mapped_parquet": {
                "path": str(mapped_path),
                "row_count": stats["rows"],
                "chunk_count": stats["chunks"],
                "columns": [
                    "transaction_id",
                    "timestamp",
                    "amount",
                    "sender_id",
                    "receiver_id",
                    "device_type",
                    "merchant_category",
                    "location",
                    "transaction_type",
                    "fraud_label",
                ],
                "compression": config.compression,
                "chunk_size": config.chunk_size,
                "rows_read": stats["rows_read"],
                "rows_removed": stats["rows_removed"],
                "rows_written": stats["rows_written"],
            }
        }
        pipeline_report["mapped_parquet"] = stats
        schema_report_path = REPORTS_DIR / "schema_reports.json"
        schema_report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")

    if run_all or args.preprocess or args.features:
        print("Stage 2/6: Fitting once and preprocessing Parquet chunks...")
        ensure_processed_parquet()

    if run_all or args.features:
        print("Stage 3/6: Feature engineering was applied during streaming preprocessing.")
        ensure_mapped_parquet()

    if run_all or args.train_supervised:
        print("Stage 4/6: Training supervised models...")
        processed_path = ensure_processed_parquet()
        label_counts = parquet_label_counts(processed_path, config.chunk_size)
        total_label_rows = max(1, sum(label_counts.values()))
        population_fraud_rate = label_counts[1] / total_label_rows
        train_label_counts = parquet_period_label_counts(
            processed_path, config.chunk_size, "train"
        )
        holdout_frauds = int(
            round(train_label_counts.get(1, 0) * args.calibration_fraud_holdout)
        )
        if holdout_frauds > 0:
            # Draw the calibration fraud holdout first so training can exclude
            # these exact rows; otherwise fraud-preserving training consumes
            # every train-period fraud and no honest calibration pool exists.
            calibration_fraud = sample_parquet_rows(
                processed_path,
                max_rows=holdout_frauds,
                chunk_size=config.chunk_size,
                target_column="fraud_label",
                random_state=42,
                columns=None,
                strategy="fraud_preserving",
                legitimate_ratio=1e-9,
                period_value="train",
                max_fraud_rows=holdout_frauds,
            )
            calibration_fraud = calibration_fraud.loc[
                pd.to_numeric(
                    calibration_fraud["fraud_label"], errors="coerce"
                ).fillna(0).astype(int)
                == 1
            ].reset_index(drop=True)
            holdout_ids = frozenset(calibration_fraud["transaction_id"])
            print(f"Calibration fraud holdout: {len(calibration_fraud)} rows")
        else:
            calibration_fraud = None
            holdout_ids = frozenset()
        supervised_sample = sample_parquet_rows(
            processed_path,
            max_rows=args.supervised_rows,
            chunk_size=config.chunk_size,
            target_column="fraud_label",
            random_state=42,
            columns=None,
            strategy="fraud_preserving",
            legitimate_ratio=args.legitimate_ratio,
            period_value="train",
            exclude_transaction_ids=holdout_ids or None,
        )
        print(
            "Using fraud-preserving supervised sample (training period only): "
            f"{supervised_sample.shape}, legitimate_ratio={args.legitimate_ratio:.2f}"
        )
        calibration_sample = _build_calibration_frame(
            processed_path=processed_path,
            config=config,
            calibration_fraud=calibration_fraud,
            supervised_sample=supervised_sample,
            population_fraud_rate=population_fraud_rate,
        )
        evaluation_frame = load_period_frame(processed_path, "test", config.chunk_size)
        print(f"Out-of-time evaluation frame: {evaluation_frame.shape}")
        supervised_results = train_supervised_models(
            supervised_sample,
            max_rows=len(supervised_sample),
            preprocessor=preprocessor,
            preprocessed=True,
            evaluation_frame=evaluation_frame,
            calibration_frame=calibration_sample,
            alert_min_precision=args.alert_min_precision,
        )
        del evaluation_frame
        del calibration_sample
        del supervised_sample
        calibration_metadata = build_calibration_metadata(
            supervised_results,
            population_fraud_rate=population_fraud_rate,
            label_counts=label_counts,
            effective_training_fraud_rate=0.25,
        )
        (PROJECT_ROOT / "models" / "fraud_probability_calibration.json").write_text(
            json.dumps(calibration_metadata, indent=2),
            encoding="utf-8",
        )
        print(
            "Saved fraud probability calibration: "
            f"population fraud rate={population_fraud_rate:.6%}"
        )
        for name in ("xgboost", "random_forest"):
            info = supervised_results.get(name, {}).get("calibration")
            if info:
                print(
                    f"{name}: isotonic Brier={info['brier_raw']:.4f}->"
                    f"{info['brier_isotonic']:.4f}, ECE={info['ece_raw']:.4f}->"
                    f"{info['ece_isotonic']:.4f}, alert threshold="
                    f"{info['alert_threshold']:.4f} "
                    f"(precision={info['precision_at_alert']:.3f}, "
                    f"recall={info['recall_at_alert']:.3f})"
                )
        results_path = REPORTS_DIR / "supervised_metrics.json"
        results_path.write_text(json.dumps(supervised_results, indent=2, default=str), encoding="utf-8")
        print("Supervised model training complete.")

    if run_all or args.train_anomaly:
        print("Stage 5/6: Training anomaly models...")
        processed_path = ensure_processed_parquet()
        anomaly_sample = sample_parquet_rows(
            processed_path,
            max_rows=args.anomaly_rows,
            chunk_size=config.chunk_size,
            target_column="fraud_label",
            random_state=42,
            columns=None,
            strategy="uniform",
            period_value="train",
        )
        print(f"Using uniform anomaly sample (training period only): {anomaly_sample.shape}")
        anomaly_results = train_anomaly_models(
            anomaly_sample,
            max_rows=len(anomaly_sample),
            preprocessor=preprocessor,
            preprocessed=True,
        )
        for name, frame in anomaly_results.items():
            frame.to_csv(PROJECT_ROOT / "data" / "processed" / f"{name}_anomaly_scores.csv", index=False)
        del anomaly_sample
        print("Anomaly model training complete.")

    if run_all or args.tune_thresholds:
        print("Stage 6/6: Tuning fusion FRAUD_LIKELY thresholds on out-of-time data...")
        processed_path = ensure_processed_parquet()
        tuned = tune_and_save_fusion_thresholds(
            processed_path=processed_path,
            chunk_size=config.chunk_size,
            min_fraud_precision=args.fraud_likely_min_precision,
        )
        if tuned is None:
            print(
                "Fusion threshold tuning skipped: run supervised training with "
                "calibration and anomaly training first."
            )

    if pipeline_report:
        pipeline_report_path.write_text(
            json.dumps(pipeline_report, indent=2, default=str),
            encoding="utf-8",
        )
    print("Pipeline completed with Parquet-backed chunk processing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
