"""Pharmacological interaction tracking for EEG emotion analysis.

Tracks how medications reshape emotional baseline over time, computes
medication-conditioned emotion models, detects emotional blunting,
onset curves, and withdrawal/rebound effects.

Usage:
    tracker = PharmacologicalTracker()
    tracker.log_medication("user1", "sertraline", "ssri", 50.0, start_date=t)
    effect = tracker.compute_medication_effect("user1", base_emotion, t)
    profile = tracker.compute_pharmacological_profile("user1")
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Storage caps per user.
_MAX_MEDICATION_ENTRIES = 500
_MAX_EMOTION_READINGS = 10000

# Minimum readings for meaningful analysis.
_MIN_READINGS_FOR_ANALYSIS = 5
_MIN_READINGS_FOR_BLUNTING = 10
_MIN_READINGS_FOR_ONSET = 8
_MIN_READINGS_FOR_WITHDRAWAL = 5


# ── Drug effect database ────────────────────────────────────────────

# Each entry: category -> { effect_type, onset_hours, peak_hours, duration_hours,
#   emotional_effects: { valence_shift, arousal_shift, range_compression } }
# range_compression: 0 = no effect, 1 = complete flattening

DRUG_EFFECT_DB: Dict[str, Dict[str, Any]] = {
    "ssri": {
        "display_name": "SSRIs (e.g. sertraline, fluoxetine, escitalopram)",
        "effect_type": "blunting",
        "onset_hours": 336.0,       # 2 weeks
        "peak_hours": 1008.0,       # 6 weeks
        "duration_hours": 2016.0,   # 12 weeks sustained
        "emotional_effects": {
            "valence_shift": 0.05,
            "arousal_shift": -0.10,
            "range_compression": 0.25,
        },
    },
    "snri": {
        "display_name": "SNRIs (e.g. venlafaxine, duloxetine)",
        "effect_type": "blunting",
        "onset_hours": 336.0,
        "peak_hours": 1008.0,
        "duration_hours": 2016.0,
        "emotional_effects": {
            "valence_shift": 0.05,
            "arousal_shift": -0.05,
            "range_compression": 0.20,
        },
    },
    "stimulant": {
        "display_name": "Stimulants (e.g. methylphenidate, amphetamine)",
        "effect_type": "stimulant",
        "onset_hours": 0.5,
        "peak_hours": 2.0,
        "duration_hours": 12.0,
        "emotional_effects": {
            "valence_shift": 0.10,
            "arousal_shift": 0.25,
            "range_compression": 0.05,
        },
    },
    "benzodiazepine": {
        "display_name": "Benzodiazepines (e.g. diazepam, lorazepam, alprazolam)",
        "effect_type": "anxiolytic",
        "onset_hours": 0.25,
        "peak_hours": 1.0,
        "duration_hours": 8.0,
        "emotional_effects": {
            "valence_shift": 0.05,
            "arousal_shift": -0.30,
            "range_compression": 0.15,
        },
    },
    "buspirone": {
        "display_name": "Buspirone (non-benzo anxiolytic)",
        "effect_type": "anxiolytic",
        "onset_hours": 168.0,      # 1 week
        "peak_hours": 504.0,       # 3 weeks
        "duration_hours": 1008.0,
        "emotional_effects": {
            "valence_shift": 0.03,
            "arousal_shift": -0.15,
            "range_compression": 0.08,
        },
    },
    "mood_stabilizer": {
        "display_name": "Mood stabilizers (e.g. lithium, valproate, lamotrigine)",
        "effect_type": "mood_stabilizing",
        "onset_hours": 168.0,
        "peak_hours": 672.0,       # 4 weeks
        "duration_hours": 2016.0,
        "emotional_effects": {
            "valence_shift": 0.0,
            "arousal_shift": -0.10,
            "range_compression": 0.30,
        },
    },
    "beta_blocker": {
        "display_name": "Beta-blockers (e.g. propranolol, atenolol)",
        "effect_type": "anxiolytic",
        "onset_hours": 1.0,
        "peak_hours": 3.0,
        "duration_hours": 12.0,
        "emotional_effects": {
            "valence_shift": 0.0,
            "arousal_shift": -0.20,
            "range_compression": 0.10,
        },
    },
    "antipsychotic": {
        "display_name": "Atypical antipsychotics (e.g. quetiapine, aripiprazole)",
        "effect_type": "sedating",
        "onset_hours": 1.0,
        "peak_hours": 4.0,
        "duration_hours": 24.0,
        "emotional_effects": {
            "valence_shift": -0.05,
            "arousal_shift": -0.25,
            "range_compression": 0.35,
        },
    },
    "gabapentinoid": {
        "display_name": "Gabapentinoids (e.g. gabapentin, pregabalin)",
        "effect_type": "anxiolytic",
        "onset_hours": 1.0,
        "peak_hours": 3.0,
        "duration_hours": 12.0,
        "emotional_effects": {
            "valence_shift": 0.02,
            "arousal_shift": -0.15,
            "range_compression": 0.10,
        },
    },
    "trazodone": {
        "display_name": "Trazodone (sedating antidepressant)",
        "effect_type": "sedating",
        "onset_hours": 1.0,
        "peak_hours": 2.0,
        "duration_hours": 8.0,
        "emotional_effects": {
            "valence_shift": 0.02,
            "arousal_shift": -0.25,
            "range_compression": 0.12,
        },
    },
    "mirtazapine": {
        "display_name": "Mirtazapine (NaSSA antidepressant)",
        "effect_type": "sedating",
        "onset_hours": 1.0,
        "peak_hours": 168.0,       # 1 week for full effect
        "duration_hours": 1008.0,
        "emotional_effects": {
            "valence_shift": 0.05,
            "arousal_shift": -0.20,
            "range_compression": 0.15,
        },
    },
    "bupropion": {
        "display_name": "Bupropion (NDRI antidepressant)",
        "effect_type": "stimulant",
        "onset_hours": 168.0,
        "peak_hours": 504.0,
        "duration_hours": 1344.0,
        "emotional_effects": {
            "valence_shift": 0.08,
            "arousal_shift": 0.10,
            "range_compression": 0.05,
        },
    },
}

# Valid medication categories (keys of DRUG_EFFECT_DB).
VALID_CATEGORIES = frozenset(DRUG_EFFECT_DB.keys())

# Valid effect types across all categories.
VALID_EFFECT_TYPES = frozenset({
    "blunting",
    "anxiolytic",
    "stimulant",
    "sedating",
    "mood_stabilizing",
})


# ── Data classes ────────────────────────────────────────────────────


@dataclass
class Medication:
    """A medication record for a user."""
    name: str
    category: str
    dosage_mg: float
    start_date: float          # Unix timestamp
    end_date: Optional[float] = None  # None = still active
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_active(self) -> bool:
        return self.end_date is None


@dataclass
class MedicationEffect:
    """Computed effect of a medication at a point in time."""
    medication_name: str
    category: str
    effect_type: str
    hours_since_start: float
    onset_fraction: float       # 0-1, how far into onset curve
    valence_modifier: float
    arousal_modifier: float
    range_compression: float    # 0 = none, 1 = full flattening

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PharmacologicalProfile:
    """Aggregate pharmacological profile for a user."""
    user_id: str
    active_medications: List[Dict[str, Any]]
    total_valence_modifier: float
    total_arousal_modifier: float
    total_range_compression: float
    dominant_effect_type: str
    blunting_detected: bool
    blunting_score: float       # 0-1
    withdrawal_risk: bool
    onset_status: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Emotion reading storage ─────────────────────────────────────────


@dataclass
class EmotionReading:
    """A single emotion measurement at a point in time."""
    timestamp: float
    valence: float          # -1 to 1
    arousal: float          # 0 to 1
    stress_index: float     # 0 to 1
    source: str = "eeg"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Core tracker ────────────────────────────────────────────────────


class PharmacologicalTracker:
    """Tracks medications and their effects on EEG-derived emotions.

    Per-user in-memory storage.

    Args:
        max_medications: Max medication entries per user.
        max_emotion_readings: Max emotion readings per user.
    """

    def __init__(
        self,
        max_medications: int = _MAX_MEDICATION_ENTRIES,
        max_emotion_readings: int = _MAX_EMOTION_READINGS,
    ) -> None:
        self._max_medications = max_medications
        self._max_emotion_readings = max_emotion_readings
        self._medications: Dict[str, List[Medication]] = {}
        self._emotion_readings: Dict[str, List[EmotionReading]] = {}

    def _ensure_user(self, user_id: str) -> None:
        if user_id not in self._medications:
            self._medications[user_id] = []
        if user_id not in self._emotion_readings:
            self._emotion_readings[user_id] = []

    # ── Medication logging ──────────────────────────────────────────

    def log_medication(
        self,
        user_id: str,
        name: str,
        category: str,
        dosage_mg: float,
        start_date: Optional[float] = None,
        end_date: Optional[float] = None,
        notes: str = "",
    ) -> Medication:
        """Log a medication start, stop, or change.

        Args:
            user_id: User identifier.
            name: Medication name (e.g. "sertraline").
            category: One of VALID_CATEGORIES.
            dosage_mg: Dosage in milligrams.
            start_date: Unix timestamp of medication start. Defaults to now.
            end_date: Unix timestamp of stop. None if still active.
            notes: Optional notes.

        Returns:
            The created Medication record.
        """
        self._ensure_user(user_id)

        if start_date is None:
            start_date = time.time()

        med = Medication(
            name=name.strip().lower(),
            category=category.strip().lower(),
            dosage_mg=dosage_mg,
            start_date=start_date,
            end_date=end_date,
            notes=notes,
        )
        self._medications[user_id].append(med)

        if len(self._medications[user_id]) > self._max_medications:
            self._medications[user_id] = self._medications[user_id][
                -self._max_medications:
            ]

        logger.info(
            "Logged medication %s (%s, %s mg) for user %s",
            name, category, dosage_mg, user_id,
        )
        return med

    def log_emotion_reading(
        self,
        user_id: str,
        timestamp: float,
        emotion_data: Dict[str, Any],
    ) -> None:
        """Log an emotion reading for pharmacological correlation.

        Args:
            user_id: User identifier.
            timestamp: Unix timestamp.
            emotion_data: Dict with valence, arousal, stress_index, source.
        """
        self._ensure_user(user_id)

        reading = EmotionReading(
            timestamp=timestamp,
            valence=float(emotion_data.get("valence", 0.0)),
            arousal=float(emotion_data.get("arousal", 0.0)),
            stress_index=float(emotion_data.get("stress_index", 0.0)),
            source=str(emotion_data.get("source", "eeg")),
        )
        self._emotion_readings[user_id].append(reading)

        if len(self._emotion_readings[user_id]) > self._max_emotion_readings:
            self._emotion_readings[user_id] = self._emotion_readings[user_id][
                -self._max_emotion_readings:
            ]

    # ── Medication effect computation ───────────────────────────────

    @staticmethod
    def _compute_onset_fraction(
        hours_since_start: float,
        onset_hours: float,
        peak_hours: float,
    ) -> float:
        """Compute how far along the onset curve a medication is.

        Returns a value between 0.0 (not yet active) and 1.0 (full effect).
        Uses a sigmoid-like ramp between onset and peak.
        """
        if hours_since_start <= 0:
            return 0.0
        if hours_since_start >= peak_hours:
            return 1.0
        if hours_since_start < onset_hours * 0.5:
            # Early phase: very little effect
            return 0.1 * (hours_since_start / (onset_hours * 0.5))
        # Ramp from onset to peak
        progress = (hours_since_start - onset_hours * 0.5) / (
            peak_hours - onset_hours * 0.5
        )
        return min(1.0, 0.1 + 0.9 * progress)

    def compute_medication_effect(
        self,
        user_id: str,
        base_emotion: Dict[str, float],
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute medication-conditioned emotion: emotion = f(base, medication_state).

        Args:
            user_id: User identifier.
            base_emotion: Dict with valence (-1 to 1), arousal (0 to 1).
            current_time: Unix timestamp. Defaults to now.

        Returns:
            Dict with adjusted_valence, adjusted_arousal, effects list,
            and total modifiers.
        """
        self._ensure_user(user_id)

        if current_time is None:
            current_time = time.time()

        base_valence = float(base_emotion.get("valence", 0.0))
        base_arousal = float(base_emotion.get("arousal", 0.5))

        active_meds = [
            m for m in self._medications[user_id]
            if m.is_active and m.start_date <= current_time
        ]

        if not active_meds:
            return {
                "adjusted_valence": base_valence,
                "adjusted_arousal": base_arousal,
                "effects": [],
                "total_valence_modifier": 0.0,
                "total_arousal_modifier": 0.0,
                "total_range_compression": 0.0,
                "medication_count": 0,
            }

        effects: List[MedicationEffect] = []
        total_val_mod = 0.0
        total_aro_mod = 0.0
        total_range_comp = 0.0

        for med in active_meds:
            hours_since = (current_time - med.start_date) / 3600.0
            db_entry = DRUG_EFFECT_DB.get(med.category)
            if db_entry is None:
                continue

            onset_frac = self._compute_onset_fraction(
                hours_since,
                db_entry["onset_hours"],
                db_entry["peak_hours"],
            )

            emo_effects = db_entry["emotional_effects"]
            val_mod = emo_effects["valence_shift"] * onset_frac
            aro_mod = emo_effects["arousal_shift"] * onset_frac
            range_comp = emo_effects["range_compression"] * onset_frac

            effect = MedicationEffect(
                medication_name=med.name,
                category=med.category,
                effect_type=db_entry["effect_type"],
                hours_since_start=round(hours_since, 2),
                onset_fraction=round(onset_frac, 4),
                valence_modifier=round(val_mod, 4),
                arousal_modifier=round(aro_mod, 4),
                range_compression=round(range_comp, 4),
            )
            effects.append(effect)

            total_val_mod += val_mod
            total_aro_mod += aro_mod
            # Range compression combines multiplicatively (each med compresses further)
            total_range_comp = 1.0 - (1.0 - total_range_comp) * (1.0 - range_comp)

        # Apply modifiers to base emotion
        adjusted_valence = base_valence + total_val_mod
        # Apply range compression: compress toward 0 (neutral)
        adjusted_valence = adjusted_valence * (1.0 - total_range_comp)
        adjusted_valence = max(-1.0, min(1.0, adjusted_valence))

        adjusted_arousal = base_arousal + total_aro_mod
        adjusted_arousal = max(0.0, min(1.0, adjusted_arousal))

        return {
            "adjusted_valence": round(adjusted_valence, 4),
            "adjusted_arousal": round(adjusted_arousal, 4),
            "effects": [e.to_dict() for e in effects],
            "total_valence_modifier": round(total_val_mod, 4),
            "total_arousal_modifier": round(total_aro_mod, 4),
            "total_range_compression": round(total_range_comp, 4),
            "medication_count": len(effects),
        }

    # ── Emotional blunting detection ────────────────────────────────

    def detect_emotional_blunting(
        self,
        user_id: str,
        window_days: float = 14.0,
    ) -> Dict[str, Any]:
        """Detect emotional blunting: narrowing of emotional range after medication start.

        Compares the emotional range (max-min valence, arousal variance) in
        the window after medication start vs before.

        Args:
            user_id: User identifier.
            window_days: Number of days to look back for comparison.

        Returns:
            Dict with blunting_detected, blunting_score, pre/post range stats.
        """
        self._ensure_user(user_id)

        readings = self._emotion_readings[user_id]
        meds = self._medications[user_id]

        if not meds or len(readings) < _MIN_READINGS_FOR_BLUNTING:
            return {
                "blunting_detected": False,
                "blunting_score": 0.0,
                "reason": "insufficient_data",
                "pre_range": 0.0,
                "post_range": 0.0,
                "sample_count_pre": 0,
                "sample_count_post": 0,
            }

        # Find earliest active blunting-type medication start
        blunting_meds = [
            m for m in meds
            if DRUG_EFFECT_DB.get(m.category, {}).get("effect_type") in (
                "blunting", "mood_stabilizing", "sedating"
            )
        ]

        if not blunting_meds:
            return {
                "blunting_detected": False,
                "blunting_score": 0.0,
                "reason": "no_blunting_medications",
                "pre_range": 0.0,
                "post_range": 0.0,
                "sample_count_pre": 0,
                "sample_count_post": 0,
            }

        earliest_start = min(m.start_date for m in blunting_meds)
        window_sec = window_days * 86400.0

        pre_readings = [
            r for r in readings
            if earliest_start - window_sec <= r.timestamp < earliest_start
        ]
        post_readings = [
            r for r in readings
            if r.timestamp >= earliest_start
        ]

        if (
            len(pre_readings) < _MIN_READINGS_FOR_BLUNTING // 2
            or len(post_readings) < _MIN_READINGS_FOR_BLUNTING // 2
        ):
            return {
                "blunting_detected": False,
                "blunting_score": 0.0,
                "reason": "insufficient_data",
                "pre_range": 0.0,
                "post_range": 0.0,
                "sample_count_pre": len(pre_readings),
                "sample_count_post": len(post_readings),
            }

        def _valence_range(rdgs: List[EmotionReading]) -> float:
            vals = [r.valence for r in rdgs]
            return max(vals) - min(vals) if vals else 0.0

        def _arousal_std(rdgs: List[EmotionReading]) -> float:
            if len(rdgs) < 2:
                return 0.0
            vals = [r.arousal for r in rdgs]
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            return math.sqrt(variance)

        pre_range = _valence_range(pre_readings)
        post_range = _valence_range(post_readings)
        pre_arousal_std = _arousal_std(pre_readings)
        post_arousal_std = _arousal_std(post_readings)

        # Blunting score: how much range narrowed
        if pre_range > 0.01:
            range_reduction = max(0.0, (pre_range - post_range) / pre_range)
        else:
            range_reduction = 0.0

        if pre_arousal_std > 0.01:
            arousal_reduction = max(
                0.0, (pre_arousal_std - post_arousal_std) / pre_arousal_std
            )
        else:
            arousal_reduction = 0.0

        blunting_score = 0.6 * range_reduction + 0.4 * arousal_reduction
        blunting_score = min(1.0, blunting_score)

        blunting_detected = blunting_score > 0.2

        return {
            "blunting_detected": blunting_detected,
            "blunting_score": round(blunting_score, 4),
            "reason": "detected" if blunting_detected else "not_detected",
            "pre_range": round(pre_range, 4),
            "post_range": round(post_range, 4),
            "pre_arousal_std": round(pre_arousal_std, 4),
            "post_arousal_std": round(post_arousal_std, 4),
            "sample_count_pre": len(pre_readings),
            "sample_count_post": len(post_readings),
            "medications_checked": [m.name for m in blunting_meds],
        }

    # ── Onset curve tracking ────────────────────────────────────────

    def compute_onset_curve(
        self,
        user_id: str,
        medication_name: str,
        bucket_hours: float = 24.0,
    ) -> Dict[str, Any]:
        """Track how long until medication effects appear in emotion data.

        Buckets emotion readings by time since medication start and computes
        mean valence/arousal per bucket to show the onset curve.

        Args:
            user_id: User identifier.
            medication_name: Name of medication to track.
            bucket_hours: Time bucket width in hours.

        Returns:
            Dict with onset_curve (list of time-bucketed emotion averages).
        """
        self._ensure_user(user_id)

        name_lower = medication_name.strip().lower()
        meds = [
            m for m in self._medications[user_id]
            if m.name == name_lower
        ]

        if not meds:
            return {
                "medication_name": medication_name,
                "onset_curve": [],
                "reason": "no_medication_entries",
            }

        readings = self._emotion_readings[user_id]
        if len(readings) < _MIN_READINGS_FOR_ONSET:
            return {
                "medication_name": medication_name,
                "onset_curve": [],
                "reason": "insufficient_emotion_data",
            }

        # Use earliest start date
        start_date = min(m.start_date for m in meds)
        bucket_sec = bucket_hours * 3600.0

        # Collect readings after medication start
        post_readings = [r for r in readings if r.timestamp >= start_date]
        if len(post_readings) < _MIN_READINGS_FOR_ONSET:
            return {
                "medication_name": medication_name,
                "onset_curve": [],
                "reason": "insufficient_post_medication_data",
            }

        # Bucket by time since start
        buckets: Dict[int, List[EmotionReading]] = {}
        for r in post_readings:
            bucket_idx = int((r.timestamp - start_date) / bucket_sec)
            if bucket_idx not in buckets:
                buckets[bucket_idx] = []
            buckets[bucket_idx].append(r)

        curve = []
        for idx in sorted(buckets.keys()):
            bucket_readings = buckets[idx]
            n = len(bucket_readings)
            mean_valence = sum(r.valence for r in bucket_readings) / n
            mean_arousal = sum(r.arousal for r in bucket_readings) / n
            mean_stress = sum(r.stress_index for r in bucket_readings) / n
            curve.append({
                "bucket_index": idx,
                "hours_since_start": round(idx * bucket_hours, 1),
                "mean_valence": round(mean_valence, 4),
                "mean_arousal": round(mean_arousal, 4),
                "mean_stress": round(mean_stress, 4),
                "sample_count": n,
            })

        return {
            "medication_name": medication_name,
            "bucket_hours": bucket_hours,
            "onset_curve": curve,
            "total_readings": len(post_readings),
        }

    # ── Withdrawal / rebound detection ──────────────────────────────

    def detect_withdrawal(
        self,
        user_id: str,
        window_days: float = 7.0,
    ) -> Dict[str, Any]:
        """Detect withdrawal or rebound effects from dose changes or stops.

        Looks for recently stopped medications and checks whether emotion
        data shows a rebound pattern (increased arousal, valence swing).

        Args:
            user_id: User identifier.
            window_days: Days after stop to check for withdrawal.

        Returns:
            Dict with withdrawal_detected, affected medications, severity.
        """
        self._ensure_user(user_id)

        readings = self._emotion_readings[user_id]
        meds = self._medications[user_id]

        stopped_meds = [m for m in meds if m.end_date is not None]
        if not stopped_meds:
            return {
                "withdrawal_detected": False,
                "affected_medications": [],
                "reason": "no_stopped_medications",
            }

        window_sec = window_days * 86400.0
        affected = []

        for med in stopped_meds:
            end_t = med.end_date
            assert end_t is not None  # type narrowing

            # Pre-stop readings (while on medication)
            pre_readings = [
                r for r in readings
                if med.start_date <= r.timestamp < end_t
            ]
            # Post-stop readings (withdrawal window)
            post_readings = [
                r for r in readings
                if end_t <= r.timestamp <= end_t + window_sec
            ]

            if (
                len(pre_readings) < _MIN_READINGS_FOR_WITHDRAWAL
                or len(post_readings) < _MIN_READINGS_FOR_WITHDRAWAL
            ):
                continue

            pre_mean_arousal = sum(r.arousal for r in pre_readings) / len(pre_readings)
            post_mean_arousal = sum(r.arousal for r in post_readings) / len(post_readings)
            pre_mean_stress = sum(r.stress_index for r in pre_readings) / len(pre_readings)
            post_mean_stress = sum(r.stress_index for r in post_readings) / len(post_readings)

            arousal_rebound = post_mean_arousal - pre_mean_arousal
            stress_rebound = post_mean_stress - pre_mean_stress

            # Withdrawal detected if arousal or stress spike after stopping
            severity = max(0.0, arousal_rebound) + max(0.0, stress_rebound)
            if severity > 0.1:
                affected.append({
                    "medication_name": med.name,
                    "category": med.category,
                    "stopped_at": med.end_date,
                    "arousal_rebound": round(arousal_rebound, 4),
                    "stress_rebound": round(stress_rebound, 4),
                    "severity": round(min(1.0, severity), 4),
                    "sample_count_pre": len(pre_readings),
                    "sample_count_post": len(post_readings),
                })

        return {
            "withdrawal_detected": len(affected) > 0,
            "affected_medications": affected,
            "medications_checked": len(stopped_meds),
            "reason": "detected" if affected else "not_detected",
        }

    # ── Full profile ────────────────────────────────────────────────

    def compute_pharmacological_profile(
        self,
        user_id: str,
        current_time: Optional[float] = None,
    ) -> PharmacologicalProfile:
        """Compute a full pharmacological profile for a user.

        Args:
            user_id: User identifier.
            current_time: Unix timestamp. Defaults to now.

        Returns:
            PharmacologicalProfile with all computed fields.
        """
        self._ensure_user(user_id)

        if current_time is None:
            current_time = time.time()

        active_meds = [
            m for m in self._medications[user_id]
            if m.is_active and m.start_date <= current_time
        ]

        # Compute effect for a neutral baseline
        effect_result = self.compute_medication_effect(
            user_id,
            {"valence": 0.0, "arousal": 0.5},
            current_time,
        )

        # Determine dominant effect type
        effect_type_counts: Dict[str, int] = {}
        for e in effect_result["effects"]:
            etype = e["effect_type"]
            effect_type_counts[etype] = effect_type_counts.get(etype, 0) + 1

        dominant = "none"
        if effect_type_counts:
            dominant = max(effect_type_counts, key=effect_type_counts.get)  # type: ignore[arg-type]

        # Blunting detection
        blunting = self.detect_emotional_blunting(user_id)

        # Withdrawal check
        withdrawal = self.detect_withdrawal(user_id)

        # Onset status for active meds
        onset_status = []
        for med in active_meds:
            db_entry = DRUG_EFFECT_DB.get(med.category)
            if db_entry is None:
                continue
            hours_since = (current_time - med.start_date) / 3600.0
            onset_frac = self._compute_onset_fraction(
                hours_since,
                db_entry["onset_hours"],
                db_entry["peak_hours"],
            )
            onset_status.append({
                "medication_name": med.name,
                "category": med.category,
                "hours_since_start": round(hours_since, 2),
                "onset_fraction": round(onset_frac, 4),
                "fully_active": onset_frac >= 1.0,
            })

        return PharmacologicalProfile(
            user_id=user_id,
            active_medications=[m.to_dict() for m in active_meds],
            total_valence_modifier=effect_result["total_valence_modifier"],
            total_arousal_modifier=effect_result["total_arousal_modifier"],
            total_range_compression=effect_result["total_range_compression"],
            dominant_effect_type=dominant,
            blunting_detected=blunting["blunting_detected"],
            blunting_score=blunting["blunting_score"],
            withdrawal_risk=withdrawal["withdrawal_detected"],
            onset_status=onset_status,
        )

    def profile_to_dict(
        self,
        user_id: str,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convenience: compute profile and return as dict.

        Args:
            user_id: User identifier.
            current_time: Unix timestamp. Defaults to now.

        Returns:
            Profile as a plain dict.
        """
        profile = self.compute_pharmacological_profile(user_id, current_time)
        return profile.to_dict()

    def get_active_medications(self, user_id: str) -> List[Dict[str, Any]]:
        """List currently active medications for a user.

        Args:
            user_id: User identifier.

        Returns:
            List of medication dicts.
        """
        self._ensure_user(user_id)
        return [
            m.to_dict() for m in self._medications[user_id]
            if m.is_active
        ]

    def get_medication_log(
        self,
        user_id: str,
        last_n: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve medication log for a user.

        Args:
            user_id: User identifier.
            last_n: Number of most recent entries.

        Returns:
            List of medication dicts.
        """
        self._ensure_user(user_id)
        return [m.to_dict() for m in self._medications[user_id][-last_n:]]

    def reset(self, user_id: str) -> None:
        """Clear all data for a user."""
        self._medications.pop(user_id, None)
        self._emotion_readings.pop(user_id, None)
        logger.info("Reset all pharmacological data for user %s", user_id)
