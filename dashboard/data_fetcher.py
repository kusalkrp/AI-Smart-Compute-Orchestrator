"""Data fetcher for Streamlit dashboard — queries API and DB."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import httpx
import pandas as pd

from config.settings import settings

BASE_URL = settings.DASHBOARD_API_BASE_URL
_API_KEY = settings.api_key_list[0] if settings.api_key_list else ""
_HEADERS = {"X-API-Key": _API_KEY}
_TIMEOUT = 10.0


def _get(path: str, params: dict | None = None) -> dict | list | None:
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.get(f"{BASE_URL}{path}", headers=_HEADERS, params=params)
            r.raise_for_status()
            return r.json()
    except Exception:
        return None


def fetch_health() -> dict:
    return _get("/v1/health") or {"status": "unreachable"}


def fetch_readiness() -> dict:
    return _get("/v1/health/ready") or {"status": "unreachable", "checks": {}}


def fetch_routing_stats(hours: int = 24) -> dict:
    return _get("/v1/routing/stats", {"hours": hours}) or {}


def fetch_decision_log(page: int = 1, page_size: int = 50) -> dict:
    return _get("/v1/routing/decision-log", {"page": page, "page_size": page_size}) or {"decisions": [], "total": 0}


def fetch_metrics_raw() -> str:
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.get(f"{BASE_URL}/v1/metrics", headers=_HEADERS)
            r.raise_for_status()
            return r.text
    except Exception:
        return ""


def fetch_policies() -> list[dict]:
    result = _get("/v1/routing/policy") or {"policies": []}
    return result.get("policies", [])


def parse_prometheus_metrics(raw: str) -> dict[str, float]:
    """Parse Prometheus text format into key→value dict."""
    metrics = {}
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(" ")
        if len(parts) >= 2:
            try:
                key = parts[0]
                value = float(parts[-1])
                metrics[key] = value
            except ValueError:
                pass
    return metrics


def decisions_to_df(decisions: list[dict]) -> pd.DataFrame:
    if not decisions:
        return pd.DataFrame()
    df = pd.DataFrame(decisions)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df
