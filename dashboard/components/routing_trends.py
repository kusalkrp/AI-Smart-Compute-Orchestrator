"""Chart: routing decisions over time."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.data_fetcher import decisions_to_df, fetch_decision_log


def render(hours: int = 24) -> None:
    st.subheader("Routing Decisions Over Time")

    data = fetch_decision_log(page=1, page_size=200)
    decisions = data.get("decisions", [])

    if not decisions:
        st.info("No routing decisions recorded yet.")
        return

    df = decisions_to_df(decisions)
    if df.empty or "created_at" not in df.columns:
        st.info("No time-series data available.")
        return

    df["hour"] = df["created_at"].dt.floor("H")
    trend = df.groupby(["hour", "target"]).size().reset_index(name="count")

    fig = px.bar(
        trend,
        x="hour",
        y="count",
        color="target",
        barmode="stack",
        title="Routing Decisions per Hour by Target",
        labels={"count": "Tasks", "hour": "Time", "target": "Target"},
        color_discrete_map={
            "GPU": "#4CAF50",
            "CPU": "#2196F3",
            "QUANTIZED": "#FF9800",
            "CLOUD": "#F44336",
        },
    )
    st.plotly_chart(fig, use_container_width=True)
