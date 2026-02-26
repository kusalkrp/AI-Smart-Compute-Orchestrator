"""CPU/GPU utilization heatmap and gauges."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.data_fetcher import fetch_metrics_raw, parse_prometheus_metrics


def render() -> None:
    st.subheader("Live Resource Utilization")

    raw = fetch_metrics_raw()
    metrics = parse_prometheus_metrics(raw)

    cpu = metrics.get("orchestrator_cpu_percent", 0.0)
    gpu = metrics.get("orchestrator_gpu_percent", 0.0)
    ram = metrics.get("orchestrator_ram_percent", 0.0)

    col1, col2, col3 = st.columns(3)

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cpu,
            title={"text": "CPU %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2196F3"},
                "steps": [
                    {"range": [0, 70], "color": "#e0f2f1"},
                    {"range": [70, 85], "color": "#fff3e0"},
                    {"range": [85, 100], "color": "#ffebee"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90},
            },
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gpu,
            title={"text": "GPU %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4CAF50"},
                "steps": [
                    {"range": [0, 70], "color": "#e8f5e9"},
                    {"range": [70, 85], "color": "#fff3e0"},
                    {"range": [85, 100], "color": "#ffebee"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 85},
            },
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ram,
            title={"text": "RAM %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#FF9800"},
                "steps": [
                    {"range": [0, 70], "color": "#fff8e1"},
                    {"range": [70, 85], "color": "#fff3e0"},
                    {"range": [85, 100], "color": "#ffebee"},
                ],
            },
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
