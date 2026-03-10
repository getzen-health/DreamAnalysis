"""Multi-temporal emotion buffer — hierarchical 0.5s / 2s / 10s / 60s windows.

Implements the 4-level temporal hierarchy from EmotionTFN (MDPI 2025):
  Level 1 (fast):    0.5–2 s   — micro-changes, startle, anger flashes
  Level 2 (medium):  2–10 s    — current emotional state
  Level 3 (slow):    10–60 s   — mood / trend
  Level 4 (context): 1–24 h    — daily pattern

Usage
-----
    buf = MultiTemporalBuffer(user_id="u1", modality="voice")
    buf.push({"valence": 0.3, "arousal": 0.6, "stress_index": 0.4})
    fused = buf.fuse()
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


# ---------------------------------------------------------------------------
# Time constants (seconds)
# ---------------------------------------------------------------------------
_LEVEL1_WINDOW  =   2.0    # fast: 2 s rolling window
_LEVEL2_WINDOW  =  10.0    # medium: 10 s rolling window
_LEVEL3_WINDOW  =  60.0    # slow: 60 s rolling window
_LEVEL4_WINDOW  = 86400.0  # context: 24 h rolling window

# Attention weights learned from EmotionTFN defaults (can be tuned)
# Order: [fast, medium, slow, context]
_DEFAULT_ATTN_WEIGHTS = {
    "valence":     [0.10, 0.35, 0.40, 0.15],
    "arousal":     [0.20, 0.40, 0.30, 0.10],
    "stress_index":[0.15, 0.40, 0.35, 0.10],
    "focus_index": [0.10, 0.35, 0.40, 0.15],
}
_DEFAULT_ATTN_WEIGHTS_FALLBACK = [0.15, 0.40, 0.35, 0.10]  # for unknown keys


@dataclass
class _Sample:
    ts: float          # unix timestamp
    values: Dict[str, float]


@dataclass
class TemporalLevel:
    """One temporal scale — rolling window of samples."""
    window_seconds: float
    label: str
    _buf: Deque[_Sample] = field(default_factory=deque, repr=False)

    def push(self, values: Dict[str, float], ts: float) -> None:
        self._buf.append(_Sample(ts=ts, values=values))
        cutoff = ts - self.window_seconds
        while self._buf and self._buf[0].ts < cutoff:
            self._buf.popleft()

    def stats(self) -> Dict[str, Any]:
        """Return mean ± std per key across the window."""
        if not self._buf:
            return {}
        keys = list(self._buf[-1].values.keys())
        out: Dict[str, float] = {}
        for k in keys:
            vals = [s.values[k] for s in self._buf if k in s.values]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            var  = sum((v - mean) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 0.0
            out[f"{k}_mean"] = round(mean, 4)
            out[f"{k}_std"]  = round(math.sqrt(var), 4)
            out[f"{k}_min"]  = round(min(vals), 4)
            out[f"{k}_max"]  = round(max(vals), 4)
        return out

    def mean(self, key: str) -> Optional[float]:
        vals = [s.values[key] for s in self._buf if key in s.values]
        return sum(vals) / len(vals) if vals else None

    @property
    def n_samples(self) -> int:
        return len(self._buf)

    @property
    def oldest_ts(self) -> Optional[float]:
        return self._buf[0].ts if self._buf else None

    @property
    def newest_ts(self) -> Optional[float]:
        return self._buf[-1].ts if self._buf else None


class MultiTemporalBuffer:
    """Maintains 4 temporal levels for a single user + modality.

    Parameters
    ----------
    user_id:  Arbitrary string user identifier.
    modality: "voice" | "eeg" | "hrv" | "multimodal" (label only).
    """

    def __init__(self, user_id: str = "default", modality: str = "voice") -> None:
        self.user_id  = user_id
        self.modality = modality
        self.levels: List[TemporalLevel] = [
            TemporalLevel(_LEVEL1_WINDOW,  "fast"),
            TemporalLevel(_LEVEL2_WINDOW,  "medium"),
            TemporalLevel(_LEVEL3_WINDOW,  "slow"),
            TemporalLevel(_LEVEL4_WINDOW,  "context"),
        ]
        self._push_count = 0
        self._last_ts: Optional[float] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def push(self, values: Dict[str, float], ts: Optional[float] = None) -> None:
        """Push a new emotion reading into all 4 levels."""
        if ts is None:
            ts = time.time()
        for lvl in self.levels:
            lvl.push(values, ts)
        self._push_count += 1
        self._last_ts = ts

    def fuse(self, emotion_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Attention-weighted fusion across all 4 levels.

        For each emotion dimension (valence, arousal, …) the fused value is:
            fused = Σ w_i * mean_i  (where w_i is attention weight for level i,
                                     zeroed-out if level has no samples)

        Returns a dict with:
            fused_<key>         — attention-weighted fused value
            level_<label>_<key> — per-level mean for transparency
            attention_weights   — effective weights used
            temporal_coverage   — which levels had data
        """
        if emotion_keys is None:
            # Infer from most recent sample
            if self.levels[0].n_samples > 0:
                emotion_keys = [k for k in self.levels[0]._buf[-1].values]
            else:
                emotion_keys = ["valence", "arousal", "stress_index", "focus_index"]

        result: Dict[str, Any] = {
            "user_id": self.user_id,
            "modality": self.modality,
            "n_samples_total": self._push_count,
        }

        # Gather level-wise means
        level_means: List[Dict[str, Optional[float]]] = []
        for lvl in self.levels:
            means = {}
            for k in emotion_keys:
                means[k] = lvl.mean(k)
            level_means.append(means)

        # Per-key attention-weighted fusion
        fused: Dict[str, float] = {}
        effective_weights: Dict[str, List[float]] = {}
        for k in emotion_keys:
            raw_w = _DEFAULT_ATTN_WEIGHTS.get(k, _DEFAULT_ATTN_WEIGHTS_FALLBACK)
            # Mask levels with no data
            active = [lm[k] is not None for lm in level_means]
            masked = [w if act else 0.0 for w, act in zip(raw_w, active)]
            total  = sum(masked)
            if total == 0:
                fused[k] = 0.0
                effective_weights[k] = [0.0] * 4
                continue
            norm_w = [w / total for w in masked]
            fused[k] = round(
                sum(norm_w[i] * (level_means[i][k] or 0.0) for i in range(4)),
                4,
            )
            effective_weights[k] = [round(w, 4) for w in norm_w]

        # Build output
        for k, v in fused.items():
            result[f"fused_{k}"] = v

        # Per-level breakdown
        for i, lvl in enumerate(self.levels):
            for k in emotion_keys:
                v = level_means[i][k]
                result[f"level_{lvl.label}_{k}"] = round(v, 4) if v is not None else None

        result["attention_weights"] = effective_weights
        result["temporal_coverage"] = {
            lvl.label: lvl.n_samples > 0
            for lvl in self.levels
        }
        result["level_sample_counts"] = {
            lvl.label: lvl.n_samples
            for lvl in self.levels
        }
        if self._last_ts:
            result["last_updated"] = self._last_ts

        return result

    def level_stats(self) -> Dict[str, Any]:
        """Return per-level descriptive stats (mean/std/min/max per key)."""
        return {
            lvl.label: {
                "n_samples": lvl.n_samples,
                "window_seconds": lvl.window_seconds,
                "stats": lvl.stats(),
            }
            for lvl in self.levels
        }

    def clear(self) -> None:
        for lvl in self.levels:
            lvl._buf.clear()
        self._push_count = 0
        self._last_ts = None


# ---------------------------------------------------------------------------
# Global registry — one buffer per (user_id, modality) pair
# ---------------------------------------------------------------------------

_registry: Dict[str, MultiTemporalBuffer] = {}


def get_buffer(user_id: str, modality: str = "voice") -> MultiTemporalBuffer:
    """Get-or-create a buffer for the given user + modality."""
    key = f"{user_id}:{modality}"
    if key not in _registry:
        _registry[key] = MultiTemporalBuffer(user_id=user_id, modality=modality)
    return _registry[key]


def list_buffers() -> List[Dict[str, Any]]:
    """Return summary of all active buffers."""
    return [
        {
            "user_id":  buf.user_id,
            "modality": buf.modality,
            "n_samples_total": buf._push_count,
            "level_sample_counts": {lvl.label: lvl.n_samples for lvl in buf.levels},
        }
        for buf in _registry.values()
    ]
