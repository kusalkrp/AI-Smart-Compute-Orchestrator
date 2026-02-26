"""Pie chart: task type and routing target breakdown."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.data_fetcher import fetch_routing_stats, fetch_decision_log, decisions_to_df


def render(hours: int = 24) -> None:
    st.subheader("Task Distribution")

    stats = fetch_routing_stats(hours=hours)
    by_target = stats.get("by_target", {})

    col1, col2 = st.columns(2)

    with col1:
        if by_target:
            fig = px.pie(
                names=list(by_target.keys()),
                values=list(by_target.values()),
                title="Routing Target Distribution",
                color_discrete_map={
                    "GPU": "#4CAF50",
                    "CPU": "#2196F3",
                    "QUANTIZED": "#FF9800",
                    "CLOUD": "#F44336",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No routing data yet.")

    with col2:
        data = fetch_decision_log(page=1, page_size=200)
        decisions = data.get("decisions", [])
        df = decisions_to_df(decisions)

        if not df.empty and "decision_stage" in df.columns:
            stage_counts = df["decision_stage"].value_counts().reset_index()
            stage_counts.columns = ["stage", "count"]
            fig = px.pie(
                stage_counts,
                names="stage",
                values="count",
                title="Decision Stage Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No stage data yet.")
