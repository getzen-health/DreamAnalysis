"""Datadog reporter for Neural Dream Workshop ML backend.

Sends custom metrics and error events to the Datadog API.
All functions are no-ops when DD_API_KEY is not set, so the
service runs safely in environments without Datadog configured.
"""

import json
import logging
import os
import time
import traceback
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DD_API_KEY = os.environ.get("DD_API_KEY", "")
_DD_SITE = os.environ.get("DD_SITE", "datadoghq.com")
_SERVICE = os.environ.get("DD_SERVICE", "neural-dream-ml")
_ENV = os.environ.get("DD_ENV", "production")

_METRICS_URL = f"https://api.{_DD_SITE}/api/v2/series"
_EVENTS_URL = f"https://api.{_DD_SITE}/api/v1/events"


def _post_json(url: str, payload: Dict) -> bool:
    """POST JSON to Datadog API. Returns True on success."""
    if not _DD_API_KEY:
        return False
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "DD-API-KEY": _DD_API_KEY,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status in (200, 202)
    except Exception as exc:
        logger.debug(f"[datadog] POST failed: {exc}")
        return False


def report_metric(
    name: str,
    value: float,
    tags: Optional[List[str]] = None,
    metric_type: str = "gauge",
) -> bool:
    """Report a custom metric to Datadog.

    Args:
        name: Metric name, e.g. "neural_dream.retrain.accuracy"
        value: Numeric value
        tags: Optional list of "key:value" tag strings
        metric_type: "gauge" | "count" | "rate"

    Returns:
        True if successfully sent, False otherwise.
    """
    if not _DD_API_KEY:
        logger.debug(f"[datadog] metric (no key) {name}={value}")
        return False

    base_tags = [f"service:{_SERVICE}", f"env:{_ENV}"]
    all_tags = base_tags + (tags or [])

    payload = {
        "series": [
            {
                "metric": name,
                "type": metric_type,
                "points": [{"timestamp": int(time.time()), "value": value}],
                "tags": all_tags,
            }
        ]
    }
    ok = _post_json(_METRICS_URL, payload)
    if ok:
        logger.debug(f"[datadog] metric sent: {name}={value}")
    return ok


def report_error(
    error_type: str,
    message: str,
    exc: Optional[Exception] = None,
    tags: Optional[List[str]] = None,
) -> bool:
    """Report an error event to Datadog Events API and increment error counter metric.

    Args:
        error_type: Short camel-case or snake_case label, e.g. "model_inference_error"
        message: Human-readable error description
        exc: Optional exception instance for traceback
        tags: Additional tags

    Returns:
        True if event was sent successfully, False otherwise.
    """
    base_tags = [f"service:{_SERVICE}", f"env:{_ENV}", f"error_type:{error_type}"]
    all_tags = base_tags + (tags or [])

    # Increment the error counter metric (fire-and-forget)
    report_metric(f"neural_dream.errors.{error_type}", 1.0, tags=all_tags, metric_type="count")

    if not _DD_API_KEY:
        logger.warning(f"[datadog] error (no key) {error_type}: {message}")
        return False

    tb_text = ""
    if exc is not None:
        tb_text = "\n\n```\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) + "```"

    payload = {
        "title": f"[{_SERVICE}] {error_type}",
        "text": f"%%% \n{message}{tb_text}\n %%%",
        "priority": "normal",
        "tags": all_tags,
        "alert_type": "error",
        "source_type_name": "python",
    }
    ok = _post_json(_EVENTS_URL, payload)
    if ok:
        logger.debug(f"[datadog] error event sent: {error_type}")
    return ok
