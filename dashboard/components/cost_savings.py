"""Chart: cloud vs local cost comparison."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.data_fetcher import fetch_routing_stats


def render(hours: int = 24) -> None:
    st.subheader("Cost Analysis")

    stats = fetch_routing_stats(hours=hours)
    if not stats:
        st.info("No cost data available.")
        return

    actual = stats.get("total_cost_usd", 0.0)
    cloud = stats.get("estimated_cloud_cost_usd", 0.0)
    saved = stats.get("cost_saved_usd", 0.0)
    pct = (saved / cloud * 100) if cloud > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Actual Cost (USD)", f"${actual:.4f}")
    col2.metric("If All-Cloud Cost (USD)", f"${cloud:.4f}")
    col3.metric("Cost Saved", f"${saved:.4f}", delta=f"{pct:.1f}%")

    fig = go.Figure(data=[
        go.Bar(name="Actual Cost", x=["Cost"], y=[actual], marker_color="#4CAF50"),
        go.Bar(name="If All-Cloud", x=["Cost"], y=[cloud], marker_color="#F44336"),
    ])
    fig.update_layout(
        barmode="group",
        title=f"Cost Comparison (last {hours}h)",
        yaxis_title="Cost (USD)",
    )
    st.plotly_chart(fig, use_container_width=True)
