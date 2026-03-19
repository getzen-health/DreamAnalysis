"""Emotion OS -- open API platform for emotion-as-a-service.

Provides a unified emotion representation (standardized EmotionVector),
multi-source fusion (EEG, voice, text, physiological, self-report),
API key management simulation, emotion event streaming model,
a plugin system for custom emotion processors, and webhook definitions
for threshold-based external notifications.

Issue #442.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASIC_EMOTIONS: Tuple[str, ...] = (
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "neutral",
)

VALID_SOURCES: Tuple[str, ...] = (
    "eeg",
    "voice",
    "text",
    "physiological",
    "self_report",
)

# Default source weights for fusion (higher = more trusted).
DEFAULT_SOURCE_WEIGHTS: Dict[str, float] = {
    "eeg": 0.30,
    "voice": 0.25,
    "text": 0.20,
    "physiological": 0.15,
    "self_report": 0.10,
}

# Rate limit: requests per minute per app.
DEFAULT_RATE_LIMIT = 60

# Maximum webhooks per app.
MAX_WEBHOOKS_PER_APP = 10

# Maximum plugins per platform instance.
MAX_PLUGINS = 50


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EmotionVector:
    """Standardized emotion representation.

    Combines dimensional (valence/arousal/dominance) and categorical
    (6 basic emotion probabilities) representations with metadata.
    """

    valence: float = 0.0        # -1 .. +1
    arousal: float = 0.5        # 0 .. 1
    dominance: float = 0.5      # 0 .. 1
    probabilities: Dict[str, float] = field(default_factory=lambda: {
        e: 1.0 / len(BASIC_EMOTIONS) for e in BASIC_EMOTIONS
    })
    confidence: float = 0.5     # 0 .. 1
    source: str = "unknown"
    timestamp: float = 0.0

    def dominant_emotion(self) -> str:
        """Return the emotion label with highest probability."""
        return max(self.probabilities, key=self.probabilities.get)  # type: ignore[arg-type]

    def to_array(self) -> np.ndarray:
        """Convert to a 9-element numpy vector for numerical operations.

        Layout: [valence, arousal, dominance, happy, sad, angry, fear, surprise, neutral]
        """
        probs = [self.probabilities.get(e, 0.0) for e in BASIC_EMOTIONS]
        return np.array(
            [self.valence, self.arousal, self.dominance] + probs,
            dtype=np.float64,
        )


@dataclass
class RegisteredApp:
    """An application registered on the Emotion OS platform."""

    app_id: str
    name: str
    api_key: str
    created_at: float = 0.0
    rate_limit: int = DEFAULT_RATE_LIMIT
    request_count: int = 0
    last_request_at: float = 0.0
    active: bool = True


@dataclass
class WebhookDefinition:
    """Webhook that fires when an emotion threshold is crossed."""

    webhook_id: str
    app_id: str
    url: str
    emotion: str                # key from BASIC_EMOTIONS or "valence"/"arousal"
    threshold: float            # 0..1 for probabilities/arousal, -1..1 for valence
    direction: str = "above"    # "above" or "below"
    created_at: float = 0.0
    last_triggered_at: float = 0.0
    trigger_count: int = 0


@dataclass
class PluginRegistration:
    """A custom emotion processor plugin."""

    plugin_id: str
    name: str
    description: str = ""
    processor_fn: Optional[str] = None   # name reference (real fn in-memory only)
    registered_at: float = 0.0
    invocation_count: int = 0


@dataclass
class EmotionOSPlatform:
    """Top-level platform state."""

    apps: Dict[str, RegisteredApp] = field(default_factory=dict)
    webhooks: Dict[str, WebhookDefinition] = field(default_factory=dict)
    plugins: Dict[str, PluginRegistration] = field(default_factory=dict)
    total_fusions: int = 0
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Module-level singleton platform
# ---------------------------------------------------------------------------

_platform = EmotionOSPlatform()


def _get_platform() -> EmotionOSPlatform:
    """Return the module-level platform singleton."""
    return _platform


def _reset_platform() -> None:
    """Reset the platform to a clean state (for testing)."""
    global _platform
    _platform = EmotionOSPlatform()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def create_emotion_vector(
    valence: float = 0.0,
    arousal: float = 0.5,
    dominance: float = 0.5,
    probabilities: Optional[Dict[str, float]] = None,
    confidence: float = 0.5,
    source: str = "unknown",
    timestamp: Optional[float] = None,
) -> EmotionVector:
    """Create a validated EmotionVector.

    Clamps dimensional values to valid ranges and normalizes probabilities
    so they sum to 1.0.
    """
    valence = float(np.clip(valence, -1.0, 1.0))
    arousal = float(np.clip(arousal, 0.0, 1.0))
    dominance = float(np.clip(dominance, 0.0, 1.0))
    confidence = float(np.clip(confidence, 0.0, 1.0))

    if probabilities is None:
        probs = {e: 1.0 / len(BASIC_EMOTIONS) for e in BASIC_EMOTIONS}
    else:
        # Ensure all 6 emotions present, fill missing with 0.
        probs = {e: max(0.0, probabilities.get(e, 0.0)) for e in BASIC_EMOTIONS}
        total = sum(probs.values())
        if total > 0:
            probs = {e: v / total for e, v in probs.items()}
        else:
            probs = {e: 1.0 / len(BASIC_EMOTIONS) for e in BASIC_EMOTIONS}

    ts = timestamp if timestamp is not None else time.time()

    return EmotionVector(
        valence=valence,
        arousal=arousal,
        dominance=dominance,
        probabilities=probs,
        confidence=confidence,
        source=source,
        timestamp=ts,
    )


def fuse_emotion_sources(
    sources: List[EmotionVector],
    weights: Optional[Dict[str, float]] = None,
) -> EmotionVector:
    """Fuse multiple EmotionVectors from different sources into one.

    Uses confidence-weighted averaging. Optionally accepts per-source
    weight overrides; defaults to DEFAULT_SOURCE_WEIGHTS.
    """
    if not sources:
        return create_emotion_vector(source="fused", confidence=0.0)

    if len(sources) == 1:
        vec = sources[0]
        return create_emotion_vector(
            valence=vec.valence,
            arousal=vec.arousal,
            dominance=vec.dominance,
            probabilities=dict(vec.probabilities),
            confidence=vec.confidence,
            source="fused",
        )

    w_map = weights if weights is not None else DEFAULT_SOURCE_WEIGHTS

    # Compute effective weight per source: source_weight * confidence.
    effective_weights: List[float] = []
    for vec in sources:
        base = w_map.get(vec.source, 0.1)
        effective_weights.append(base * max(vec.confidence, 0.01))

    total_w = sum(effective_weights)
    if total_w == 0:
        total_w = 1.0

    normed = [w / total_w for w in effective_weights]

    # Weighted average of dimensional values.
    fused_valence = sum(n * v.valence for n, v in zip(normed, sources))
    fused_arousal = sum(n * v.arousal for n, v in zip(normed, sources))
    fused_dominance = sum(n * v.dominance for n, v in zip(normed, sources))

    # Weighted average of probabilities.
    fused_probs: Dict[str, float] = {e: 0.0 for e in BASIC_EMOTIONS}
    for n, vec in zip(normed, sources):
        for e in BASIC_EMOTIONS:
            fused_probs[e] += n * vec.probabilities.get(e, 0.0)

    # Normalize probabilities.
    prob_total = sum(fused_probs.values())
    if prob_total > 0:
        fused_probs = {e: v / prob_total for e, v in fused_probs.items()}

    # Fused confidence: weighted average of source confidences.
    fused_confidence = sum(n * v.confidence for n, v in zip(normed, sources))

    platform = _get_platform()
    platform.total_fusions += 1

    return create_emotion_vector(
        valence=fused_valence,
        arousal=fused_arousal,
        dominance=fused_dominance,
        probabilities=fused_probs,
        confidence=fused_confidence,
        source="fused",
    )


def register_app(
    name: str,
    rate_limit: int = DEFAULT_RATE_LIMIT,
) -> RegisteredApp:
    """Register a new application on the platform.

    Returns the RegisteredApp with a generated API key.
    """
    platform = _get_platform()
    now = time.time()
    raw = f"{name}-{now}-{len(platform.apps)}"
    app_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
    api_key = hashlib.sha256(f"key-{raw}".encode()).hexdigest()[:32]

    app = RegisteredApp(
        app_id=app_id,
        name=name,
        api_key=api_key,
        created_at=now,
        rate_limit=max(1, rate_limit),
    )
    platform.apps[app_id] = app
    logger.info("Registered app '%s' with id %s", name, app_id)
    return app


def register_webhook(
    app_id: str,
    url: str,
    emotion: str,
    threshold: float,
    direction: str = "above",
) -> Dict[str, Any]:
    """Register a webhook for threshold-based emotion notifications.

    Returns a dict with webhook_id and metadata, or an error dict.
    """
    platform = _get_platform()

    if app_id not in platform.apps:
        return {"error": f"Unknown app_id: {app_id}"}

    valid_emotions = list(BASIC_EMOTIONS) + ["valence", "arousal", "dominance"]
    if emotion not in valid_emotions:
        return {"error": f"Invalid emotion key: {emotion}. Must be one of {valid_emotions}"}

    if direction not in ("above", "below"):
        return {"error": f"Invalid direction: {direction}. Must be 'above' or 'below'."}

    # Check per-app webhook limit.
    app_hooks = [w for w in platform.webhooks.values() if w.app_id == app_id]
    if len(app_hooks) >= MAX_WEBHOOKS_PER_APP:
        return {"error": f"Maximum webhooks ({MAX_WEBHOOKS_PER_APP}) reached for app {app_id}."}

    now = time.time()
    raw = f"{app_id}-{url}-{emotion}-{now}"
    webhook_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    hook = WebhookDefinition(
        webhook_id=webhook_id,
        app_id=app_id,
        url=url,
        emotion=emotion,
        threshold=float(threshold),
        direction=direction,
        created_at=now,
    )
    platform.webhooks[webhook_id] = hook
    logger.info("Registered webhook %s for app %s on %s %s %.2f",
                webhook_id, app_id, emotion, direction, threshold)

    return {
        "webhook_id": webhook_id,
        "app_id": app_id,
        "url": url,
        "emotion": emotion,
        "threshold": threshold,
        "direction": direction,
    }


def check_webhook_triggers(
    vector: EmotionVector,
) -> List[Dict[str, Any]]:
    """Check which webhooks would fire given an EmotionVector.

    Returns a list of triggered webhook dicts (webhook_id, emotion, value, threshold).
    Does NOT actually make HTTP calls -- the caller is responsible for dispatch.
    """
    platform = _get_platform()
    triggered: List[Dict[str, Any]] = []

    for hook in platform.webhooks.values():
        # Resolve the value to check.
        if hook.emotion in BASIC_EMOTIONS:
            value = vector.probabilities.get(hook.emotion, 0.0)
        elif hook.emotion == "valence":
            value = vector.valence
        elif hook.emotion == "arousal":
            value = vector.arousal
        elif hook.emotion == "dominance":
            value = vector.dominance
        else:
            continue

        fired = False
        if hook.direction == "above" and value >= hook.threshold:
            fired = True
        elif hook.direction == "below" and value <= hook.threshold:
            fired = True

        if fired:
            hook.trigger_count += 1
            hook.last_triggered_at = time.time()
            triggered.append({
                "webhook_id": hook.webhook_id,
                "app_id": hook.app_id,
                "url": hook.url,
                "emotion": hook.emotion,
                "value": round(value, 4),
                "threshold": hook.threshold,
                "direction": hook.direction,
            })

    return triggered


def register_plugin(
    name: str,
    description: str = "",
) -> Dict[str, Any]:
    """Register a custom emotion processor plugin.

    Returns a dict with plugin_id and metadata, or an error dict.
    """
    platform = _get_platform()

    if len(platform.plugins) >= MAX_PLUGINS:
        return {"error": f"Maximum plugins ({MAX_PLUGINS}) reached."}

    now = time.time()
    raw = f"plugin-{name}-{now}"
    plugin_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

    plug = PluginRegistration(
        plugin_id=plugin_id,
        name=name,
        description=description,
        registered_at=now,
    )
    platform.plugins[plugin_id] = plug
    logger.info("Registered plugin '%s' with id %s", name, plugin_id)

    return {
        "plugin_id": plugin_id,
        "name": name,
        "description": description,
    }


def compute_platform_stats() -> Dict[str, Any]:
    """Compute aggregate platform statistics."""
    platform = _get_platform()

    total_requests = sum(a.request_count for a in platform.apps.values())
    total_webhook_triggers = sum(w.trigger_count for w in platform.webhooks.values())
    total_plugin_invocations = sum(p.invocation_count for p in platform.plugins.values())
    active_apps = sum(1 for a in platform.apps.values() if a.active)

    return {
        "total_apps": len(platform.apps),
        "active_apps": active_apps,
        "total_webhooks": len(platform.webhooks),
        "total_plugins": len(platform.plugins),
        "total_fusions": platform.total_fusions,
        "total_requests": total_requests,
        "total_webhook_triggers": total_webhook_triggers,
        "total_plugin_invocations": total_plugin_invocations,
        "uptime_seconds": round(time.time() - platform.created_at, 2),
    }


def platform_to_dict() -> Dict[str, Any]:
    """Serialize the full platform state to a dict."""
    platform = _get_platform()

    apps = {}
    for aid, app in platform.apps.items():
        apps[aid] = {
            "app_id": app.app_id,
            "name": app.name,
            "api_key": app.api_key[:8] + "..." if len(app.api_key) > 8 else app.api_key,
            "created_at": app.created_at,
            "rate_limit": app.rate_limit,
            "request_count": app.request_count,
            "active": app.active,
        }

    webhooks = {}
    for wid, hook in platform.webhooks.items():
        webhooks[wid] = {
            "webhook_id": hook.webhook_id,
            "app_id": hook.app_id,
            "url": hook.url,
            "emotion": hook.emotion,
            "threshold": hook.threshold,
            "direction": hook.direction,
            "trigger_count": hook.trigger_count,
        }

    plugins = {}
    for pid, plug in platform.plugins.items():
        plugins[pid] = {
            "plugin_id": plug.plugin_id,
            "name": plug.name,
            "description": plug.description,
            "invocation_count": plug.invocation_count,
        }

    return {
        "apps": apps,
        "webhooks": webhooks,
        "plugins": plugins,
        "stats": compute_platform_stats(),
    }
