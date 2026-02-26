"""Streamlit Dashboard — AI Smart Compute Orchestrator."""
from __future__ import annotations

import time

import streamlit as st

st.set_page_config(
    page_title="AI Smart Compute Orchestrator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🧠 AI Orchestrator")
st.sidebar.markdown("---")
page = st.sidebar.selectbox(
    "Navigate",
    ["Overview", "Routing Intelligence", "Cost Analysis", "Resource Monitor", "Task Explorer"],
)
hours = st.sidebar.slider("Time Window (hours)", min_value=1, max_value=168, value=24)
auto_refresh = st.sidebar.checkbox("Auto-refresh (2s)", value=False)

if auto_refresh:
    time.sleep(2)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**API Status**")

from dashboard.data_fetcher import fetch_health, fetch_readiness

health = fetch_health()
ready = fetch_readiness()

if health.get("status") == "ok":
    st.sidebar.success("API: Online")
else:
    st.sidebar.error("API: Offline")

checks = ready.get("checks", {})
for svc, ok in checks.items():
    icon = "✅" if ok else "❌"
    st.sidebar.markdown(f"{icon} {svc.title()}")

# ─── Pages ────────────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Orchestrator Overview")

    from dashboard.data_fetcher import fetch_routing_stats, fetch_metrics_raw, parse_prometheus_metrics

    stats = fetch_routing_stats(hours=hours)
    raw = fetch_metrics_raw()
    metrics = parse_prometheus_metrics(raw)

    col1, col2, col3, col4 = st.columns(4)

    total_tasks = stats.get("total_tasks", 0)
    cost_saved = stats.get("cost_saved_usd", 0.0)
    avg_lat = stats.get("avg_latency_ms", 0.0)
    actual_cost = stats.get("total_cost_usd", 0.0)

    col1.metric("Total Tasks", total_tasks)
    col2.metric("Cost Saved (USD)", f"${cost_saved:.4f}")
    col3.metric("Avg Latency (ms)", f"{avg_lat:.0f}")
    col4.metric("Actual Cost (USD)", f"${actual_cost:.6f}")

    st.markdown("---")
    from dashboard.components import resource_heatmap, task_distribution
    resource_heatmap.render()
    st.markdown("---")
    task_distribution.render(hours=hours)

elif page == "Routing Intelligence":
    st.title("Routing Intelligence")

    from dashboard.components import routing_trends, task_distribution
    routing_trends.render(hours=hours)
    st.markdown("---")

    st.subheader("Recent Routing Decisions")
    from dashboard.data_fetcher import fetch_decision_log, decisions_to_df

    data = fetch_decision_log(page=1, page_size=50)
    df = decisions_to_df(data.get("decisions", []))
    if not df.empty:
        display_cols = [c for c in [
            "created_at", "target", "model_name", "decision_stage",
            "confidence", "estimated_cost_usd", "reasoning"
        ] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)
    else:
        st.info("No routing decisions yet. Submit some tasks!")

    st.markdown("---")
    st.subheader("Active Routing Policies")
    from dashboard.data_fetcher import fetch_policies
    policies = fetch_policies()
    if policies:
        import pandas as pd
        pdf = pd.DataFrame(policies)
        st.dataframe(pdf[["name", "target", "priority_order", "is_active"]], use_container_width=True)
    else:
        st.info("No policies loaded.")

elif page == "Cost Analysis":
    st.title("Cost Analysis")
    from dashboard.components import cost_savings
    cost_savings.render(hours=hours)

elif page == "Resource Monitor":
    st.title("Resource Monitor")
    from dashboard.components import resource_heatmap
    resource_heatmap.render()

    st.markdown("---")
    st.subheader("Queue Depths")
    from dashboard.data_fetcher import fetch_metrics_raw, parse_prometheus_metrics
    raw = fetch_metrics_raw()
    metrics = parse_prometheus_metrics(raw)

    import plotly.graph_objects as go
    targets = ["GPU", "CPU", "QUANTIZED", "CLOUD"]
    depths = [
        metrics.get(f'orchestrator_tasks_by_target{{target="{t}"}}', 0)
        for t in targets
    ]
    fig = go.Figure(data=[
        go.Bar(x=targets, y=depths, marker_color=["#4CAF50", "#2196F3", "#FF9800", "#F44336"])
    ])
    fig.update_layout(title="Tasks by Target", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Task Explorer":
    st.title("Task Explorer")
    from dashboard.data_fetcher import fetch_decision_log, decisions_to_df

    page_num = st.number_input("Page", min_value=1, value=1, step=1)
    data = fetch_decision_log(page=page_num, page_size=20)
    total = data.get("total", 0)
    st.caption(f"Total decisions: {total}")

    df = decisions_to_df(data.get("decisions", []))
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No data found.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("AI Smart Compute Orchestrator v0.1.0 — Production System Demo")
