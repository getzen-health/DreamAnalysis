"""Interoception training engine — body-awareness exercises and scoring (#422).

Guides users through structured interoceptive training exercises and
scores their ability to detect internal body signals. Core tasks:

1. **Heartbeat counting task** (Schandry, 1981): User silently counts
   heartbeats over a fixed interval, result compared to actual HR sensor
   reading. Computes Interoceptive Accuracy Score (IAS).

2. **Body scan scoring**: User reports sensations at specific body regions,
   compared against physiological ground truth (EEG/HRV/skin conductance
   proxy signals).

3. **MAIA-like profiling**: Tracks 8 dimensions of interoceptive awareness
   based on the Multidimensional Assessment of Interoceptive Awareness
   (Mehling et al., 2012): noticing, not-distracting, not-worrying,
   attention_regulation, emotional_awareness, self_regulation,
   body_listening, trusting.

4. **Progressive exercise generation**: Adaptive difficulty that starts
   with easy tasks (heartbeat at rest) and increases challenge
   (heartbeat during movement, body scan with distractors).

References:
    Schandry (1981) — Heart beat perception and emotional experience
    Garfinkel et al. (2015) — Knowing your own heart: interoceptive accuracy
    Mehling et al. (2012) — MAIA: Multidimensional Assessment of
        Interoceptive Awareness
    Craig (2002) — How do you feel? Interoception: the sense of the body
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────

BODY_REGIONS: List[str] = [
    "head", "face", "neck", "chest",
    "upper_back", "lower_back", "stomach",
    "left_arm", "right_arm",
    "left_hand", "right_hand",
    "left_leg", "right_leg", "feet",
]

MAIA_DIMENSIONS: List[str] = [
    "noticing",
    "not_distracting",
    "not_worrying",
    "attention_regulation",
    "emotional_awareness",
    "self_regulation",
    "body_listening",
    "trusting",
]

# Difficulty levels for progressive exercise generation.
DIFFICULTY_LEVELS: List[str] = [
    "beginner",
    "easy",
    "moderate",
    "hard",
    "expert",
]

# Maximum history entries per user.
_MAX_HISTORY = 500


# ── Dataclasses ────────────────────────────────────────────────────────


@dataclass
class HeartbeatTask:
    """Result of a single heartbeat counting task."""

    counted_beats: int
    actual_beats: int
    duration_seconds: float
    ias: float  # Interoceptive Accuracy Score: 1 - |counted - actual| / actual
    condition: str = "rest"  # rest, movement, post_exercise, paced_breathing


@dataclass
class BodyScanResult:
    """Result of a body scan exercise."""

    region: str
    reported_sensation: str  # e.g. "warmth", "tingling", "none"
    ground_truth_active: bool  # whether physiological signal was present
    correct: bool
    confidence: float = 0.5  # user's confidence in their report, 0-1


@dataclass
class InteroceptiveProfile:
    """Full interoceptive profile for a user."""

    user_id: str
    heartbeat_accuracy: float  # mean IAS across heartbeat tasks
    body_scan_accuracy: float  # fraction of correct body scan responses
    maia_scores: Dict[str, float]  # 8 MAIA dimensions, each 0-1
    overall_score: float  # weighted composite, 0-1
    n_heartbeat_tasks: int
    n_body_scans: int
    difficulty_level: str
    exercise_history_length: int


# ── Engine ─────────────────────────────────────────────────────────────


class InteroceptionEngine:
    """Interoception training engine.

    Maintains per-user history of heartbeat counting tasks and body scan
    results, computes MAIA-like profiles, and generates adaptive exercises.
    """

    def __init__(self) -> None:
        self._heartbeat_history: Dict[str, List[HeartbeatTask]] = {}
        self._body_scan_history: Dict[str, List[BodyScanResult]] = {}

    # ── Heartbeat counting ─────────────────────────────────────────

    def score_heartbeat_counting(
        self,
        counted_beats: int,
        actual_beats: int,
        duration_seconds: float,
        condition: str = "rest",
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Score a heartbeat counting task.

        IAS formula (Schandry 1981):
            IAS = 1 - |counted - actual| / actual

        Clamped to [0, 1]. An IAS of 1.0 means perfect accuracy.

        Args:
            counted_beats: Number of heartbeats the user reported.
            actual_beats: Ground-truth heartbeat count from sensor.
            duration_seconds: Duration of the counting interval.
            condition: Task condition (rest, movement, etc.).
            user_id: User identifier.

        Returns:
            Dict with ias, counted, actual, duration, condition, hr_bpm,
            and interpretation.
        """
        if actual_beats <= 0:
            ias = 0.0
        else:
            ias = 1.0 - abs(counted_beats - actual_beats) / actual_beats
            ias = float(np.clip(ias, 0.0, 1.0))

        hr_bpm = 0.0
        if duration_seconds > 0:
            hr_bpm = round(actual_beats / duration_seconds * 60.0, 1)

        task = HeartbeatTask(
            counted_beats=counted_beats,
            actual_beats=actual_beats,
            duration_seconds=duration_seconds,
            ias=ias,
            condition=condition,
        )

        if user_id not in self._heartbeat_history:
            self._heartbeat_history[user_id] = []
        self._heartbeat_history[user_id].append(task)
        if len(self._heartbeat_history[user_id]) > _MAX_HISTORY:
            self._heartbeat_history[user_id] = self._heartbeat_history[user_id][-_MAX_HISTORY:]

        # Interpretation
        if ias >= 0.85:
            interpretation = "excellent"
        elif ias >= 0.70:
            interpretation = "good"
        elif ias >= 0.50:
            interpretation = "moderate"
        elif ias >= 0.30:
            interpretation = "below_average"
        else:
            interpretation = "poor"

        return {
            "ias": round(ias, 4),
            "counted_beats": counted_beats,
            "actual_beats": actual_beats,
            "duration_seconds": duration_seconds,
            "condition": condition,
            "hr_bpm": hr_bpm,
            "interpretation": interpretation,
        }

    # ── Body scan scoring ──────────────────────────────────────────

    def score_body_scan(
        self,
        reports: List[Dict[str, Any]],
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Score a body scan exercise.

        Each report dict should contain:
            - region (str): body region name
            - reported_sensation (str): what the user felt
            - ground_truth_active (bool): whether a signal was present
            - confidence (float, optional): user confidence 0-1

        A report is "correct" if:
            - ground_truth_active=True and reported_sensation != "none"
            - ground_truth_active=False and reported_sensation == "none"

        Args:
            reports: List of per-region report dicts.
            user_id: User identifier.

        Returns:
            Dict with accuracy, n_correct, n_total, per_region results,
            and sensitivity/specificity.
        """
        results: List[BodyScanResult] = []
        n_correct = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for report in reports:
            region = report.get("region", "unknown")
            sensation = report.get("reported_sensation", "none")
            ground_truth = bool(report.get("ground_truth_active", False))
            confidence = float(report.get("confidence", 0.5))
            confidence = float(np.clip(confidence, 0.0, 1.0))

            user_detected = sensation.lower() != "none"

            if ground_truth and user_detected:
                correct = True
                true_positives += 1
            elif not ground_truth and not user_detected:
                correct = True
                true_negatives += 1
            elif ground_truth and not user_detected:
                correct = False
                false_negatives += 1
            else:
                correct = False
                false_positives += 1

            if correct:
                n_correct += 1

            result = BodyScanResult(
                region=region,
                reported_sensation=sensation,
                ground_truth_active=ground_truth,
                correct=correct,
                confidence=confidence,
            )
            results.append(result)

        n_total = len(results)
        accuracy = n_correct / n_total if n_total > 0 else 0.0

        # Sensitivity (true positive rate) and specificity (true negative rate)
        sensitivity = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        specificity = (
            true_negatives / (true_negatives + false_positives)
            if (true_negatives + false_positives) > 0
            else 0.0
        )

        # Store results
        if user_id not in self._body_scan_history:
            self._body_scan_history[user_id] = []
        self._body_scan_history[user_id].extend(results)
        if len(self._body_scan_history[user_id]) > _MAX_HISTORY:
            self._body_scan_history[user_id] = self._body_scan_history[user_id][-_MAX_HISTORY:]

        per_region = []
        for r in results:
            per_region.append({
                "region": r.region,
                "reported_sensation": r.reported_sensation,
                "ground_truth_active": r.ground_truth_active,
                "correct": r.correct,
                "confidence": round(r.confidence, 4),
            })

        return {
            "accuracy": round(accuracy, 4),
            "n_correct": n_correct,
            "n_total": n_total,
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "per_region": per_region,
        }

    # ── Interoceptive profile ──────────────────────────────────────

    def compute_interoceptive_profile(
        self,
        user_id: str = "default",
    ) -> InteroceptiveProfile:
        """Compute a full interoceptive profile for a user.

        Combines heartbeat accuracy, body scan accuracy, and infers
        MAIA-like dimension scores from task performance history.

        Returns:
            InteroceptiveProfile dataclass.
        """
        hb_history = self._heartbeat_history.get(user_id, [])
        bs_history = self._body_scan_history.get(user_id, [])

        # Heartbeat accuracy: mean IAS
        if hb_history:
            heartbeat_accuracy = float(np.mean([t.ias for t in hb_history]))
        else:
            heartbeat_accuracy = 0.0

        # Body scan accuracy: fraction correct
        if bs_history:
            body_scan_accuracy = float(np.mean([1.0 if r.correct else 0.0 for r in bs_history]))
        else:
            body_scan_accuracy = 0.0

        # MAIA-like dimension inference from performance data
        maia_scores = self._infer_maia_scores(hb_history, bs_history)

        # Overall composite: 40% heartbeat, 30% body scan, 30% MAIA mean
        maia_mean = float(np.mean(list(maia_scores.values()))) if maia_scores else 0.0
        overall = 0.4 * heartbeat_accuracy + 0.3 * body_scan_accuracy + 0.3 * maia_mean
        overall = float(np.clip(overall, 0.0, 1.0))

        # Difficulty level based on overall performance and history length
        difficulty_level = self._compute_difficulty_level(
            overall, len(hb_history) + len(bs_history)
        )

        return InteroceptiveProfile(
            user_id=user_id,
            heartbeat_accuracy=round(heartbeat_accuracy, 4),
            body_scan_accuracy=round(body_scan_accuracy, 4),
            maia_scores={k: round(v, 4) for k, v in maia_scores.items()},
            overall_score=round(overall, 4),
            n_heartbeat_tasks=len(hb_history),
            n_body_scans=len(bs_history),
            difficulty_level=difficulty_level,
            exercise_history_length=len(hb_history) + len(bs_history),
        )

    # ── Exercise generation ────────────────────────────────────────

    def generate_next_exercise(
        self,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Generate the next progressive exercise for a user.

        Adapts difficulty based on performance history. Starts with
        easy heartbeat counting at rest, progresses through body scans
        and harder heartbeat conditions.

        Returns:
            Dict with exercise_type, instructions, difficulty, duration,
            condition (for heartbeat), regions (for body scan),
            and tip.
        """
        hb_history = self._heartbeat_history.get(user_id, [])
        bs_history = self._body_scan_history.get(user_id, [])
        total_exercises = len(hb_history) + len(bs_history)

        # Compute current difficulty level
        profile = self.compute_interoceptive_profile(user_id)
        difficulty = profile.difficulty_level

        # Alternate between heartbeat and body scan exercises,
        # with heartbeat more frequent early on.
        if total_exercises < 3:
            return self._heartbeat_exercise(difficulty, "rest")

        # Recent heartbeat accuracy
        recent_hb_ias = 0.0
        if hb_history:
            recent = hb_history[-min(5, len(hb_history)):]
            recent_hb_ias = float(np.mean([t.ias for t in recent]))

        # Decide exercise type based on pattern
        if total_exercises % 3 == 0:
            # Body scan every 3rd exercise
            n_regions = self._regions_for_difficulty(difficulty)
            regions = self._select_scan_regions(bs_history, n_regions)
            return self._body_scan_exercise(difficulty, regions)
        else:
            # Heartbeat task — pick condition based on difficulty
            condition = self._heartbeat_condition_for_difficulty(
                difficulty, recent_hb_ias
            )
            return self._heartbeat_exercise(difficulty, condition)

    # ── Serialization ──────────────────────────────────────────────

    def profile_to_dict(self, profile: InteroceptiveProfile) -> Dict[str, Any]:
        """Convert an InteroceptiveProfile to a plain dict."""
        return asdict(profile)

    # ── Private helpers ────────────────────────────────────────────

    def _infer_maia_scores(
        self,
        hb_history: List[HeartbeatTask],
        bs_history: List[BodyScanResult],
    ) -> Dict[str, float]:
        """Infer MAIA-like dimension scores from task performance.

        These are approximations based on behavioral performance, not the
        full MAIA questionnaire. Each score is in [0, 1].
        """
        scores: Dict[str, float] = {}

        # Noticing: ability to detect body signals (heartbeat accuracy + body scan sensitivity)
        hb_mean = float(np.mean([t.ias for t in hb_history])) if hb_history else 0.0
        bs_correct = [1.0 if r.correct and r.ground_truth_active else 0.0
                      for r in bs_history if r.ground_truth_active]
        bs_sensitivity = float(np.mean(bs_correct)) if bs_correct else 0.0
        scores["noticing"] = float(np.clip(0.6 * hb_mean + 0.4 * bs_sensitivity, 0.0, 1.0))

        # Not-distracting: low false-positive rate in body scan (specificity proxy)
        bs_fp = [1.0 if not r.correct and not r.ground_truth_active else 0.0
                 for r in bs_history if not r.ground_truth_active]
        fp_rate = float(np.mean(bs_fp)) if bs_fp else 0.5
        scores["not_distracting"] = float(np.clip(1.0 - fp_rate, 0.0, 1.0))

        # Not-worrying: stable confidence scores (low variance in reported confidence)
        confidences = [r.confidence for r in bs_history]
        if len(confidences) >= 3:
            conf_std = float(np.std(confidences))
            scores["not_worrying"] = float(np.clip(1.0 - conf_std * 2, 0.0, 1.0))
        else:
            scores["not_worrying"] = 0.5

        # Attention regulation: improvement trend in heartbeat accuracy
        if len(hb_history) >= 4:
            mid = len(hb_history) // 2
            first_half = float(np.mean([t.ias for t in hb_history[:mid]]))
            second_half = float(np.mean([t.ias for t in hb_history[mid:]]))
            improvement = second_half - first_half
            scores["attention_regulation"] = float(np.clip(0.5 + improvement, 0.0, 1.0))
        else:
            scores["attention_regulation"] = 0.5

        # Emotional awareness: number of distinct sensation types reported
        sensations = set(r.reported_sensation for r in bs_history
                         if r.reported_sensation.lower() != "none")
        if bs_history:
            diversity = min(len(sensations) / 6.0, 1.0)  # 6 distinct types = max
            scores["emotional_awareness"] = diversity
        else:
            scores["emotional_awareness"] = 0.0

        # Self-regulation: consistency over time (low variance in IAS)
        if len(hb_history) >= 3:
            ias_std = float(np.std([t.ias for t in hb_history]))
            scores["self_regulation"] = float(np.clip(1.0 - ias_std * 2, 0.0, 1.0))
        else:
            scores["self_regulation"] = 0.5

        # Body listening: fraction of body scan regions attempted across all scans
        scanned_regions = set(r.region for r in bs_history)
        region_coverage = min(len(scanned_regions) / len(BODY_REGIONS), 1.0) if BODY_REGIONS else 0.0
        scores["body_listening"] = region_coverage

        # Trusting: mean confidence in correct responses
        correct_confs = [r.confidence for r in bs_history if r.correct]
        scores["trusting"] = float(np.mean(correct_confs)) if correct_confs else 0.5

        return scores

    def _compute_difficulty_level(
        self, overall_score: float, n_exercises: int
    ) -> str:
        """Determine appropriate difficulty level."""
        if n_exercises < 3:
            return "beginner"
        if overall_score >= 0.80:
            return "expert"
        if overall_score >= 0.65:
            return "hard"
        if overall_score >= 0.45:
            return "moderate"
        if overall_score >= 0.25:
            return "easy"
        return "beginner"

    def _heartbeat_exercise(
        self, difficulty: str, condition: str
    ) -> Dict[str, Any]:
        """Generate a heartbeat counting exercise."""
        durations = {
            "beginner": 30,
            "easy": 25,
            "moderate": 20,
            "hard": 15,
            "expert": 10,
        }
        duration = durations.get(difficulty, 30)

        instructions = {
            "rest": "Sit quietly and count your heartbeats silently without touching your pulse.",
            "movement": "Walk slowly in place while counting your heartbeats.",
            "post_exercise": "After brief exercise, count your heartbeats while your heart rate returns to rest.",
            "paced_breathing": "Breathe at a steady pace (4 sec in, 4 sec out) while counting heartbeats.",
        }

        tips = {
            "rest": "Focus on the sensation in your chest or neck. Try not to guess.",
            "movement": "Movement makes this harder. Focus on the strongest pulse you can feel internally.",
            "post_exercise": "Your heart rate is changing — stay with the sensation moment to moment.",
            "paced_breathing": "The breathing pattern may distract. Try to separate breath from heartbeat awareness.",
        }

        return {
            "exercise_type": "heartbeat_counting",
            "instructions": instructions.get(condition, instructions["rest"]),
            "difficulty": difficulty,
            "duration_seconds": duration,
            "condition": condition,
            "tip": tips.get(condition, tips["rest"]),
        }

    def _body_scan_exercise(
        self, difficulty: str, regions: List[str]
    ) -> Dict[str, Any]:
        """Generate a body scan exercise."""
        return {
            "exercise_type": "body_scan",
            "instructions": (
                "Close your eyes. For each body region listed, report what "
                "sensation you feel (or 'none'). Rate your confidence."
            ),
            "difficulty": difficulty,
            "regions": regions,
            "tip": "Move through each region slowly. Spend at least 10 seconds per area.",
        }

    def _regions_for_difficulty(self, difficulty: str) -> int:
        """Number of body regions to scan at each difficulty level."""
        mapping = {
            "beginner": 3,
            "easy": 5,
            "moderate": 7,
            "hard": 10,
            "expert": 14,
        }
        return mapping.get(difficulty, 5)

    def _select_scan_regions(
        self, bs_history: List[BodyScanResult], n: int
    ) -> List[str]:
        """Select body regions for next scan, prioritizing under-practiced ones."""
        region_counts: Dict[str, int] = {r: 0 for r in BODY_REGIONS}
        for result in bs_history:
            if result.region in region_counts:
                region_counts[result.region] += 1

        # Sort by count ascending (least practiced first), then take n
        sorted_regions = sorted(BODY_REGIONS, key=lambda r: region_counts[r])
        return sorted_regions[:n]

    def _heartbeat_condition_for_difficulty(
        self, difficulty: str, recent_ias: float
    ) -> str:
        """Pick heartbeat task condition based on difficulty and performance."""
        if difficulty in ("beginner", "easy"):
            return "rest"
        if difficulty == "moderate":
            return "paced_breathing" if recent_ias > 0.6 else "rest"
        if difficulty == "hard":
            return "movement" if recent_ias > 0.5 else "paced_breathing"
        # expert
        return "post_exercise" if recent_ias > 0.6 else "movement"
