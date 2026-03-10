"""JITAI (Just-In-Time Adaptive Intervention) engine.

Provides:
  - InterventionBandit: Thompson Sampling contextual bandit for personalized
    intervention selection. Each arm = an intervention type. Per-user
    Beta(alpha, beta) priors. Cold start uses evidence-based intensity defaults.
  - HRVTriggerDetector: Detects stress episodes from HRV RMSSD drops relative
    to a per-user exponential moving average baseline.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Evidence-based defaults by intensity band ────────────────────────────────
# Used during cold start (< COLD_START_THRESHOLD outcomes).
# Maps intensity range to ordered list of preferred intervention types.
INTENSITY_DEFAULTS: List[Tuple[float, float, List[str]]] = [
    # (min_intensity, max_intensity, [preferred interventions in priority order])
    (0.0, 0.4, ["cognitive_reappraisal", "body_scan", "slow_breathing"]),
    (0.4, 0.7, ["slow_breathing", "body_scan", "breathing", "music_calm"]),
    (0.7, 1.0, ["cyclic_sighing", "grounding_54321", "slow_breathing", "breathing"]),
]

COLD_START_THRESHOLD = 10  # outcomes needed before bandit takes over


class InterventionBandit:
    """Thompson Sampling bandit for personalized intervention selection.

    Each arm = an intervention type. Per-user Beta(alpha, beta) priors.
    Context: emotion_type, intensity, time_of_day.
    Reward: normalized HRV/stress recovery + optional self-report.

    Cold start: use evidence-based defaults for first 10 interventions.
    After 10+: bandit selection dominates.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # user_id -> {intervention_type -> [alpha, beta]}
        self._user_priors: Dict[str, Dict[str, List[float]]] = {}
        # user_id -> total outcome count
        self._outcome_counts: Dict[str, int] = {}

    def _ensure_user(self, user_id: str, available: List[str]) -> None:
        """Lazily initialize priors for a user if missing."""
        with self._lock:
            if user_id not in self._user_priors:
                self._user_priors[user_id] = {}
                self._outcome_counts[user_id] = 0
            for arm in available:
                if arm not in self._user_priors[user_id]:
                    # Uniform prior: Beta(1, 1)
                    self._user_priors[user_id][arm] = [1.0, 1.0]

    def select(
        self,
        user_id: str,
        available_interventions: List[str],
        intensity: float,
        emotion_type: Optional[str] = None,
    ) -> str:
        """Select best intervention via Thompson Sampling.

        If < COLD_START_THRESHOLD outcomes recorded, use intensity-based defaults.
        After threshold: sample from Beta posteriors, pick highest.

        Parameters
        ----------
        user_id : str
            User identifier.
        available_interventions : list[str]
            Intervention type keys that are currently eligible (post-cooldown).
        intensity : float
            Distress intensity in [0, 1] (e.g. stress_index or emotion_intensity).
        emotion_type : str, optional
            Current dominant emotion label.

        Returns
        -------
        str
            Selected intervention type key.
        """
        if not available_interventions:
            return "breathing"  # absolute fallback

        self._ensure_user(user_id, available_interventions)

        with self._lock:
            total_outcomes = self._outcome_counts.get(user_id, 0)

        if total_outcomes < COLD_START_THRESHOLD:
            return self._cold_start_select(available_interventions, intensity)

        return self._thompson_select(user_id, available_interventions)

    def _cold_start_select(
        self, available: List[str], intensity: float
    ) -> str:
        """Select intervention based on evidence-based intensity defaults."""
        intensity = max(0.0, min(1.0, intensity))

        for low, high, preferred in INTENSITY_DEFAULTS:
            if low <= intensity < high or (high == 1.0 and intensity == 1.0):
                for pref in preferred:
                    if pref in available:
                        return pref
                break

        # Fallback: return first available
        return available[0]

    def _thompson_select(self, user_id: str, available: List[str]) -> str:
        """Sample from Beta posteriors and select highest sample."""
        best_arm = available[0]
        best_sample = -1.0

        with self._lock:
            priors = self._user_priors.get(user_id, {})

        for arm in available:
            alpha, beta = priors.get(arm, [1.0, 1.0])
            sample = np.random.beta(alpha, beta)
            if sample > best_sample:
                best_sample = sample
                best_arm = arm

        return best_arm

    def update(self, user_id: str, intervention_type: str, reward: float) -> None:
        """Update Beta posterior with outcome.

        Parameters
        ----------
        user_id : str
            User identifier.
        intervention_type : str
            The intervention that was delivered.
        reward : float
            Outcome in [0, 1]. Higher = better.
            Typically: stress_delta normalized to [0,1] + self-report bonus.
        """
        reward = max(0.0, min(1.0, reward))

        with self._lock:
            if user_id not in self._user_priors:
                self._user_priors[user_id] = {}
                self._outcome_counts[user_id] = 0

            if intervention_type not in self._user_priors[user_id]:
                self._user_priors[user_id][intervention_type] = [1.0, 1.0]

            # Beta-Bernoulli update: treat reward as probability of success
            # Fractional update for continuous rewards
            self._user_priors[user_id][intervention_type][0] += reward
            self._user_priors[user_id][intervention_type][1] += (1.0 - reward)
            self._outcome_counts[user_id] += 1

        logger.info(
            "Bandit update: user=%s, type=%s, reward=%.3f, new_prior=%s",
            user_id,
            intervention_type,
            reward,
            self._user_priors[user_id][intervention_type],
        )

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Return current bandit priors and selection stats for a user.

        Returns
        -------
        dict
            Keys: total_outcomes, is_cold_start, arms (dict of arm stats).
        """
        with self._lock:
            total = self._outcome_counts.get(user_id, 0)
            priors = dict(self._user_priors.get(user_id, {}))

        arms: Dict[str, Dict[str, float]] = {}
        for arm, (alpha, beta) in priors.items():
            mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            arms[arm] = {
                "alpha": round(alpha, 3),
                "beta": round(beta, 3),
                "estimated_reward": round(mean, 3),
                "total_updates": round(alpha + beta - 2.0, 1),  # subtract initial prior
            }

        return {
            "user_id": user_id,
            "total_outcomes": total,
            "is_cold_start": total < COLD_START_THRESHOLD,
            "cold_start_threshold": COLD_START_THRESHOLD,
            "arms": arms,
        }


class HRVTriggerDetector:
    """Detect stress episodes from HRV features.

    Trigger: RMSSD drop > 1.5 SD from rolling baseline (non-metabolic context).
    Rolling baseline: exponential moving average over last 30 minutes.

    RMSSD (Root Mean Square of Successive Differences) is the gold-standard
    short-term HRV metric. Lower RMSSD = less parasympathetic tone = more stress.
    """

    # EMA smoothing factor — approximates ~30 min window at 1 reading/30 sec
    EMA_ALPHA = 0.05
    # Trigger threshold: RMSSD drop > N standard deviations below baseline
    TRIGGER_SD = 1.5
    # Minimum readings before baseline is considered stable
    MIN_READINGS = 5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # user_id -> {rmssd_mean, rmssd_var, count, last_update}
        self._baselines: Dict[str, Dict[str, float]] = {}

    def add_reading(
        self, user_id: str, rmssd: float, heart_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """Add HRV reading and update rolling baseline.

        Parameters
        ----------
        user_id : str
            User identifier.
        rmssd : float
            RMSSD value in milliseconds.
        heart_rate : float, optional
            Heart rate in BPM (logged but not used for trigger).

        Returns
        -------
        dict
            Baseline status: mean, std, count, baseline_ready.
        """
        with self._lock:
            if user_id not in self._baselines:
                self._baselines[user_id] = {
                    "rmssd_mean": rmssd,
                    "rmssd_var": 0.0,
                    "count": 1,
                    "last_update": time.time(),
                }
                return self._baseline_status(user_id)

            bl = self._baselines[user_id]
            bl["count"] += 1

            # Welford-like online update with EMA weighting
            old_mean = bl["rmssd_mean"]
            bl["rmssd_mean"] = (1 - self.EMA_ALPHA) * old_mean + self.EMA_ALPHA * rmssd
            # Update variance estimate (EMA of squared deviations)
            deviation_sq = (rmssd - bl["rmssd_mean"]) ** 2
            bl["rmssd_var"] = (
                (1 - self.EMA_ALPHA) * bl["rmssd_var"] + self.EMA_ALPHA * deviation_sq
            )
            bl["last_update"] = time.time()

            return self._baseline_status(user_id)

    def check_trigger(self, user_id: str, current_rmssd: float) -> Dict[str, Any]:
        """Check if current RMSSD indicates a stress episode.

        Returns True if RMSSD dropped > 1.5 SD below the rolling baseline mean.

        Parameters
        ----------
        user_id : str
            User identifier.
        current_rmssd : float
            Current RMSSD reading in milliseconds.

        Returns
        -------
        dict
            Keys: triggered (bool), rmssd_z_score (float), baseline_ready (bool),
            baseline_mean, baseline_std, current_rmssd.
        """
        with self._lock:
            bl = self._baselines.get(user_id)

        if bl is None or bl["count"] < self.MIN_READINGS:
            return {
                "triggered": False,
                "reason": "insufficient_baseline",
                "baseline_ready": False,
                "current_rmssd": current_rmssd,
                "readings_collected": bl["count"] if bl else 0,
                "readings_needed": self.MIN_READINGS,
            }

        mean = bl["rmssd_mean"]
        std = math.sqrt(bl["rmssd_var"]) if bl["rmssd_var"] > 0 else 1.0

        # Avoid division by near-zero std
        if std < 0.5:
            std = 0.5

        z_score = (current_rmssd - mean) / std
        triggered = z_score < -self.TRIGGER_SD  # negative z = drop

        return {
            "triggered": triggered,
            "reason": "rmssd_drop" if triggered else "normal",
            "baseline_ready": True,
            "rmssd_z_score": round(z_score, 3),
            "baseline_mean": round(mean, 2),
            "baseline_std": round(std, 2),
            "current_rmssd": round(current_rmssd, 2),
            "threshold_sd": self.TRIGGER_SD,
        }

    def get_baseline(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Return current baseline stats for a user, or None if no data."""
        with self._lock:
            bl = self._baselines.get(user_id)
        if bl is None:
            return None
        return self._baseline_status(user_id)

    def _baseline_status(self, user_id: str) -> Dict[str, Any]:
        """Build baseline status dict (must hold lock or have local copy)."""
        bl = self._baselines[user_id]
        std = math.sqrt(bl["rmssd_var"]) if bl["rmssd_var"] > 0 else 0.0
        return {
            "user_id": user_id,
            "rmssd_mean": round(bl["rmssd_mean"], 2),
            "rmssd_std": round(std, 2),
            "count": bl["count"],
            "baseline_ready": bl["count"] >= self.MIN_READINGS,
            "last_update": bl["last_update"],
        }
