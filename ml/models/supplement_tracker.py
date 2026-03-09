"""Supplement/Medication/Vitamin Tracker with EEG correlation analysis.

Tracks user supplement intake and correlates it with EEG-derived emotional
states (valence, arousal, stress, focus) and brain signal features
(alpha/beta ratio, theta power, FAA) over time.

Answers: "Is what I'm consuming affecting my brain/emotions positively or
negatively?"

Usage:
    tracker = SupplementTracker()
    entry_id = tracker.log_supplement("user1", "Omega-3", "supplement", 1000, "mg")
    tracker.log_brain_state("user1", time.time(), {...})
    report = tracker.get_supplement_report("user1")
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Storage caps per user to prevent unbounded memory growth.
_MAX_SUPPLEMENT_ENTRIES = 5000
_MAX_BRAIN_STATES = 10000

# Minimum post-supplement readings required for a meaningful correlation.
_MIN_READINGS_FOR_CORRELATION = 5

# Verdict thresholds
_POSITIVE_VALENCE_THRESHOLD = 0.05
_POSITIVE_STRESS_THRESHOLD = -0.03
_NEGATIVE_VALENCE_THRESHOLD = -0.05
_NEGATIVE_STRESS_THRESHOLD = 0.05

# Valid supplement types
VALID_SUPPLEMENT_TYPES = frozenset({
    "vitamin",
    "supplement",
    "medication",
    "food_supplement",
})


@dataclass
class SupplementEntry:
    """A single supplement intake record."""
    entry_id: str
    name: str
    supplement_type: str
    dosage: float
    unit: str
    timestamp: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BrainStateSnapshot:
    """A brain state reading at a point in time."""
    timestamp: float
    valence: float          # -1 to 1
    arousal: float          # 0 to 1
    stress_index: float     # 0 to 1
    focus_index: float      # 0 to 1
    alpha_beta_ratio: float = 0.0
    theta_power: float = 0.0
    faa: float = 0.0        # frontal alpha asymmetry

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SupplementTracker:
    """Tracks supplement intake and correlates with EEG-derived brain states.

    Per-user in-memory storage. Thread-safe for single-process FastAPI use.

    Args:
        max_supplements: Max supplement entries per user.
        max_brain_states: Max brain state snapshots per user.
    """

    def __init__(
        self,
        max_supplements: int = _MAX_SUPPLEMENT_ENTRIES,
        max_brain_states: int = _MAX_BRAIN_STATES,
    ) -> None:
        self._max_supplements = max_supplements
        self._max_brain_states = max_brain_states
        # Per-user storage
        self._supplements: Dict[str, List[SupplementEntry]] = {}
        self._brain_states: Dict[str, List[BrainStateSnapshot]] = {}

    def _ensure_user(self, user_id: str) -> None:
        """Initialize storage for a user if not already present."""
        if user_id not in self._supplements:
            self._supplements[user_id] = []
        if user_id not in self._brain_states:
            self._brain_states[user_id] = []

    def log_supplement(
        self,
        user_id: str,
        name: str,
        supplement_type: str,
        dosage: float,
        unit: str,
        timestamp: Optional[float] = None,
        notes: str = "",
    ) -> str:
        """Log a supplement intake.

        Args:
            user_id: User identifier.
            name: Supplement name (e.g. "Omega-3", "Vitamin D").
            supplement_type: One of: vitamin, supplement, medication, food_supplement.
            dosage: Amount taken.
            unit: Unit of measurement (e.g. "mg", "IU", "mcg").
            timestamp: Unix timestamp. Defaults to now.
            notes: Optional notes.

        Returns:
            entry_id: Unique identifier for the logged entry.
        """
        self._ensure_user(user_id)

        if timestamp is None:
            timestamp = time.time()

        entry_id = str(uuid.uuid4())[:8]
        entry = SupplementEntry(
            entry_id=entry_id,
            name=name.strip(),
            supplement_type=supplement_type,
            dosage=dosage,
            unit=unit,
            timestamp=timestamp,
            notes=notes,
        )
        self._supplements[user_id].append(entry)

        # Enforce cap — keep most recent entries
        if len(self._supplements[user_id]) > self._max_supplements:
            self._supplements[user_id] = self._supplements[user_id][
                -self._max_supplements:
            ]

        logger.info(
            "Logged supplement %s for user %s: %s %s %s",
            entry_id, user_id, name, dosage, unit,
        )
        return entry_id

    def log_brain_state(
        self,
        user_id: str,
        timestamp: float,
        emotion_data: Dict[str, Any],
    ) -> None:
        """Log a brain state snapshot from EEG analysis.

        Args:
            user_id: User identifier.
            timestamp: Unix timestamp of the reading.
            emotion_data: Dict with keys: valence, arousal, stress_index,
                focus_index, and optionally alpha_beta_ratio, theta_power, faa.
        """
        self._ensure_user(user_id)

        snapshot = BrainStateSnapshot(
            timestamp=timestamp,
            valence=float(emotion_data.get("valence", 0.0)),
            arousal=float(emotion_data.get("arousal", 0.0)),
            stress_index=float(emotion_data.get("stress_index", 0.0)),
            focus_index=float(emotion_data.get("focus_index", 0.0)),
            alpha_beta_ratio=float(emotion_data.get("alpha_beta_ratio", 0.0)),
            theta_power=float(emotion_data.get("theta_power", 0.0)),
            faa=float(emotion_data.get("faa", 0.0)),
        )
        self._brain_states[user_id].append(snapshot)

        # Enforce cap
        if len(self._brain_states[user_id]) > self._max_brain_states:
            self._brain_states[user_id] = self._brain_states[user_id][
                -self._max_brain_states:
            ]

    def get_log(
        self,
        user_id: str,
        last_n: int = 50,
        supplement_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve supplement log entries.

        Args:
            user_id: User identifier.
            last_n: Number of most recent entries to return.
            supplement_name: If provided, filter by supplement name (case-insensitive).

        Returns:
            List of supplement entry dicts, most recent last.
        """
        self._ensure_user(user_id)
        entries = self._supplements[user_id]

        if supplement_name:
            name_lower = supplement_name.strip().lower()
            entries = [e for e in entries if e.name.lower() == name_lower]

        return [e.to_dict() for e in entries[-last_n:]]

    def analyze_correlations(
        self,
        user_id: str,
        supplement_name: str,
        window_hours: float = 4.0,
    ) -> Dict[str, Any]:
        """Correlate a supplement with brain state changes in the hours after intake.

        Compares brain states in the window_hours after taking the supplement
        vs brain states when the supplement was NOT taken (control readings).

        Args:
            user_id: User identifier.
            supplement_name: Name of the supplement to analyze.
            window_hours: Hours after intake to look for brain state readings.

        Returns:
            Dict with avg shifts in valence/arousal/stress/focus, EEG-specific
            metrics (alpha_beta_ratio, theta, faa), sample counts, and verdict.
        """
        self._ensure_user(user_id)

        name_lower = supplement_name.strip().lower()
        window_sec = window_hours * 3600

        # Find all intake timestamps for this supplement
        intake_times = [
            e.timestamp
            for e in self._supplements[user_id]
            if e.name.lower() == name_lower
        ]

        if not intake_times:
            return {
                "supplement_name": supplement_name,
                "verdict": "insufficient_data",
                "reason": "no_supplement_entries",
                "sample_count_post": 0,
                "sample_count_control": 0,
            }

        brain_states = self._brain_states[user_id]
        if not brain_states:
            return {
                "supplement_name": supplement_name,
                "verdict": "insufficient_data",
                "reason": "no_brain_states",
                "sample_count_post": 0,
                "sample_count_control": 0,
            }

        # Partition brain states into post-supplement and control
        post_states: List[BrainStateSnapshot] = []
        control_states: List[BrainStateSnapshot] = []

        for bs in brain_states:
            is_post = False
            for intake_t in intake_times:
                if 0 <= (bs.timestamp - intake_t) <= window_sec:
                    is_post = True
                    break
            if is_post:
                post_states.append(bs)
            else:
                control_states.append(bs)

        if len(post_states) < _MIN_READINGS_FOR_CORRELATION:
            return {
                "supplement_name": supplement_name,
                "verdict": "insufficient_data",
                "reason": "too_few_post_supplement_readings",
                "sample_count_post": len(post_states),
                "sample_count_control": len(control_states),
            }

        # Compute mean metrics for post and control groups
        def _mean_metric(
            states: List[BrainStateSnapshot], attr: str
        ) -> float:
            if not states:
                return 0.0
            return sum(getattr(s, attr) for s in states) / len(states)

        metrics = [
            "valence", "arousal", "stress_index", "focus_index",
            "alpha_beta_ratio", "theta_power", "faa",
        ]

        post_means: Dict[str, float] = {}
        control_means: Dict[str, float] = {}
        shifts: Dict[str, float] = {}

        for m in metrics:
            post_means[m] = _mean_metric(post_states, m)
            control_means[m] = _mean_metric(control_states, m)
            shifts[m] = post_means[m] - control_means[m]

        # Determine verdict
        valence_shift = shifts["valence"]
        stress_shift = shifts["stress_index"]

        if (
            valence_shift > _POSITIVE_VALENCE_THRESHOLD
            and stress_shift < _POSITIVE_STRESS_THRESHOLD
        ):
            verdict = "positive"
        elif (
            valence_shift < _NEGATIVE_VALENCE_THRESHOLD
            or stress_shift > _NEGATIVE_STRESS_THRESHOLD
        ):
            verdict = "negative"
        else:
            verdict = "neutral"

        return {
            "supplement_name": supplement_name,
            "window_hours": window_hours,
            "verdict": verdict,
            "sample_count_post": len(post_states),
            "sample_count_control": len(control_states),
            "avg_valence_shift": round(valence_shift, 4),
            "avg_arousal_shift": round(shifts["arousal"], 4),
            "avg_stress_shift": round(stress_shift, 4),
            "avg_focus_shift": round(shifts["focus_index"], 4),
            "eeg_insights": {
                "alpha_beta_ratio_shift": round(shifts["alpha_beta_ratio"], 4),
                "theta_power_shift": round(shifts["theta_power"], 4),
                "faa_shift": round(shifts["faa"], 4),
            },
            "post_means": {k: round(v, 4) for k, v in post_means.items()},
            "control_means": {k: round(v, 4) for k, v in control_means.items()},
        }

    def get_supplement_report(self, user_id: str) -> Dict[str, Any]:
        """Generate a full correlation report for all supplements a user takes.

        Args:
            user_id: User identifier.

        Returns:
            Dict with per-supplement correlation verdicts and overall summary.
        """
        self._ensure_user(user_id)

        # Collect unique supplement names
        supplement_names = sorted(set(
            e.name for e in self._supplements[user_id]
        ))

        if not supplement_names:
            return {
                "user_id": user_id,
                "supplements": [],
                "total_supplements": 0,
                "total_brain_states": len(self._brain_states[user_id]),
            }

        reports = []
        for name in supplement_names:
            correlation = self.analyze_correlations(user_id, name)
            # Count entries for this supplement
            entry_count = sum(
                1 for e in self._supplements[user_id]
                if e.name.lower() == name.lower()
            )
            reports.append({
                "name": name,
                "entry_count": entry_count,
                "correlation": correlation,
            })

        return {
            "user_id": user_id,
            "supplements": reports,
            "total_supplements": len(supplement_names),
            "total_brain_states": len(self._brain_states[user_id]),
        }

    def get_active_supplements(
        self,
        user_id: str,
        hours: float = 24.0,
    ) -> List[Dict[str, Any]]:
        """List supplements taken in the last N hours.

        Args:
            user_id: User identifier.
            hours: Look-back window in hours.

        Returns:
            List of supplement entry dicts taken within the window.
        """
        self._ensure_user(user_id)
        cutoff = time.time() - (hours * 3600)
        active = [
            e.to_dict()
            for e in self._supplements[user_id]
            if e.timestamp >= cutoff
        ]
        return active

    def reset(self, user_id: str) -> None:
        """Clear all supplement and brain state data for a user.

        Args:
            user_id: User identifier.
        """
        self._supplements.pop(user_id, None)
        self._brain_states.pop(user_id, None)
        logger.info("Reset all supplement data for user %s", user_id)
