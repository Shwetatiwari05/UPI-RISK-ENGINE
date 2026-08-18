"""Streamlit dashboard layout for offline model testing."""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.prediction_engine import PredictionEngine
from src.utils import MODELS_DIR, REPORTS_DIR


SAMPLE_PRESETS = {
    "Normal small transfer": {
        "amount": 450.0,
        "transaction_type": "TRANSFER",
        "device_type": "Android",
        "merchant_category": "Personal",
        "sender_id": "user_1024",
        "receiver_id": "user_2048",
        "location": "Mumbai",
        "hour": 13,
    },
    "Large late-night merchant payment": {
        "amount": 85000.0,
        "transaction_type": "PAYMENT",
        "device_type": "Web",
        "merchant_category": "Electronics",
        "sender_id": "user_1024",
        "receiver_id": "merchant_991",
        "location": "Delhi",
        "hour": 2,
    },
    "Rapid cash-out style transfer": {
        "amount": 125000.0,
        "transaction_type": "CASH_OUT",
        "device_type": "Unknown",
        "merchant_category": "Unknown",
        "sender_id": "user_7777",
        "receiver_id": "receiver_3333",
        "location": "Unknown",
        "hour": 1,
    },
}


def render_dashboard() -> None:
    """Render the Streamlit testing dashboard."""
    st.set_page_config(page_title="UPI Fraud Testing Dashboard", layout="wide")
    st.title("Hybrid Anomaly Detection for UPI Transactions")
    st.caption("Offline batch-model testing dashboard. Supervised and anomaly outputs are shown separately.")

    engine = PredictionEngine(MODELS_DIR)
    if not engine.is_ready:
        st.warning(
            "Models are not trained yet. Place datasets in data/raw/ and run `python main.py --all`."
        )

    transaction = _transaction_form()
    if st.button("Run model test", type="primary"):
        _run_prediction(engine, transaction)

    st.divider()
    _render_performance_summary()
    _render_prediction_logs()


def _transaction_form() -> dict[str, object]:
    preset_name = st.selectbox("Sample transaction preset", list(SAMPLE_PRESETS))
    preset = SAMPLE_PRESETS[preset_name]

    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Amount", min_value=0.0, value=float(preset["amount"]), step=100.0)
        transaction_type = st.selectbox(
            "Transaction type",
            ["PAYMENT", "TRANSFER", "CASH_IN", "CASH_OUT", "DEBIT", "UPI"],
            index=_index_or_zero(["PAYMENT", "TRANSFER", "CASH_IN", "CASH_OUT", "DEBIT", "UPI"], preset["transaction_type"]),
        )
        device_type = st.selectbox(
            "Device type",
            ["Android", "iOS", "Web", "POS", "Unknown"],
            index=_index_or_zero(["Android", "iOS", "Web", "POS", "Unknown"], preset["device_type"]),
        )
    with col2:
        merchant_category = st.text_input("Merchant category", str(preset["merchant_category"]))
        sender_id = st.text_input("Sender ID", str(preset["sender_id"]))
        receiver_id = st.text_input("Receiver ID", str(preset["receiver_id"]))
    with col3:
        location = st.text_input("Location", str(preset["location"]))
        transaction_date = st.date_input("Transaction date", value=datetime.now().date())
        transaction_time = st.time_input("Transaction time", value=time(int(preset["hour"]), 0))

    timestamp = datetime.combine(transaction_date, transaction_time)
    return {
        "transaction_id": f"manual_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "amount": amount,
        "transaction_type": transaction_type,
        "device_type": device_type,
        "merchant_category": merchant_category,
        "sender_id": sender_id,
        "receiver_id": receiver_id,
        "location": location,
        "timestamp": timestamp,
    }


def _run_prediction(engine: PredictionEngine, transaction: dict[str, object]) -> None:
    if not engine.is_ready:
        st.error("Prediction cannot run until the trained model files exist.")
        return

    try:
        outputs = engine.predict(transaction)
    except Exception as exc:  # pragma: no cover - UI safety
        st.exception(exc)
        return

    supervised = outputs["supervised"].iloc[0]
    anomaly = outputs["anomaly"].iloc[0]

    left, right = st.columns(2)
    with left:
        st.subheader("Supervised Model Output")
        probability = float(supervised["fraud_probability"])
        st.metric("Fraud probability", f"{probability:.2%}")
        st.metric("Fraud prediction", "Fraud" if int(supervised["fraud_prediction"]) else "Legitimate")
        st.metric("Confidence score", f"{float(supervised['confidence_score']):.2%}")
        st.plotly_chart(_probability_gauge(probability), use_container_width=True)

    with right:
        st.subheader("Unsupervised Model Output")
        anomaly_score = float(anomaly["anomaly_score"])
        st.metric("Anomaly score", f"{anomaly_score:.4f}")
        st.metric("Anomaly label", str(anomaly["anomaly_label"]))
        st.metric("Anomaly confidence", f"{float(anomaly['anomaly_confidence']):.2%}")
        st.plotly_chart(_anomaly_score_chart(anomaly_score), use_container_width=True)

    _append_prediction_log(transaction, supervised, anomaly)
    _render_transaction_comparison(transaction, probability, anomaly_score)


def _probability_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#334155"},
                "steps": [
                    {"range": [0, 40], "color": "#d1fae5"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#fee2e2"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def _anomaly_score_chart(score: float) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Current transaction"],
                y=[score],
                marker_color=["#0f766e"],
            )
        ]
    )
    fig.update_layout(
        yaxis_title="Anomaly score",
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def _render_transaction_comparison(
    transaction: dict[str, object],
    fraud_probability: float,
    anomaly_score: float,
) -> None:
    st.subheader("Transaction Comparison Plot")
    comparison = pd.DataFrame(
        {
            "metric": ["Amount", "Fraud probability", "Anomaly score"],
            "value": [
                float(transaction["amount"]),
                fraud_probability * 100,
                anomaly_score,
            ],
        }
    )
    fig = go.Figure(data=[go.Bar(x=comparison["metric"], y=comparison["value"])])
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _render_performance_summary() -> None:
    st.subheader("Model Performance Summary")
    report_images = sorted(Path(REPORTS_DIR).glob("*feature_importance.png"))
    if not report_images:
        st.info("Performance plots will appear after training the models.")
        return

    for image_path in report_images:
        st.image(str(image_path), caption=image_path.name)


def _render_prediction_logs() -> None:
    st.subheader("Prediction Logs")
    logs = st.session_state.get("prediction_logs", [])
    if not logs:
        st.info("No test transactions have been run in this session.")
        return
    st.dataframe(pd.DataFrame(logs), use_container_width=True)


def _append_prediction_log(
    transaction: dict[str, object],
    supervised: pd.Series,
    anomaly: pd.Series,
) -> None:
    logs = st.session_state.setdefault("prediction_logs", [])
    logs.append(
        {
            "timestamp": transaction["timestamp"],
            "amount": transaction["amount"],
            "transaction_type": transaction["transaction_type"],
            "fraud_probability": float(supervised["fraud_probability"]),
            "fraud_prediction": int(supervised["fraud_prediction"]),
            "anomaly_score": float(anomaly["anomaly_score"]),
            "anomaly_label": anomaly["anomaly_label"],
        }
    )


def _index_or_zero(options: list[str], value: object) -> int:
    try:
        return options.index(str(value))
    except ValueError:
        return 0
