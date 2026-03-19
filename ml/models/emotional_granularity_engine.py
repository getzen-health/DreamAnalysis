"""Emotional granularity engine -- self-report-based emotion differentiation.

Measures and trains users to distinguish fine-grained emotional states based on
Barrett's (2006) emotional granularity research. Higher granularity = the user
differentiates many distinct emotion labels; lower granularity = they collapse
everything into "good" or "bad".

Key concepts:
- Emotion taxonomy: 6 basic -> 25 secondary -> 50+ tertiary emotions
- Granularity score: how distinctly a user differentiates between emotions
- Intra-class correlation (ICC): consistency of distinguishing similar emotions
- Emotion vocabulary: breadth and depth of labels a user actually uses
- Differentiation exercises: tasks to improve granularity over time

Scientific basis:
- Barrett (2006): emotional granularity as individual difference in affect
  differentiation, measured via self-report pattern diversity.
- Tugade, Fredrickson & Barrett (2004): higher granularity predicts better
  emotion regulation and psychological well-being.
- Kashdan et al. (2015): emotion differentiation reduces maladaptive coping.

Usage:
    engine = EmotionalGranularityEngine()
    score = engine.compute_granularity_score(emotion_history)
    exercise = engine.generate_differentiation_exercise(emotion_history)
    vocab = engine.compute_emotion_vocabulary(emotion_history)
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Emotion taxonomy: basic -> secondary -> tertiary
# Based on Shaver et al. (1987), Parrott (2001), Barrett (2006)
# ---------------------------------------------------------------------------

EMOTION_TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "joy": {
        "happiness": ["cheerfulness", "contentment", "delight", "amusement"],
        "enthusiasm": ["excitement", "eagerness", "zeal", "exhilaration"],
        "gratitude": ["thankfulness", "appreciation", "gratefulness"],
        "pride": ["triumph", "accomplishment", "confidence"],
        "serenity": ["peacefulness", "calm", "tranquility", "relief"],
    },
    "sadness": {
        "grief": ["sorrow", "mourning", "heartbreak", "anguish"],
        "melancholy": ["wistfulness", "nostalgia", "longing", "pensiveness"],
        "loneliness": ["isolation", "abandonment", "alienation"],
        "disappointment": ["dismay", "letdown", "disillusionment"],
        "hopelessness": ["despair", "helplessness", "powerlessness"],
    },
    "anger": {
        "frustration": ["exasperation", "irritation", "annoyance", "agitation"],
        "resentment": ["bitterness", "indignation", "spite"],
        "rage": ["fury", "hostility", "wrath"],
        "contempt": ["scorn", "disdain", "loathing"],
        "jealousy": ["envy", "covetousness", "possessiveness"],
    },
    "fear": {
        "anxiety": ["nervousness", "worry", "unease", "apprehension"],
        "terror": ["panic", "dread", "horror", "fright"],
        "insecurity": ["vulnerability", "self-doubt", "inadequacy"],
        "overwhelm": ["suffocation", "paralysis", "shock"],
    },
    "surprise": {
        "astonishment": ["amazement", "wonder", "awe"],
        "confusion": ["bewilderment", "disorientation", "perplexity"],
        "startle": ["alarm", "jolt"],
        "anticipation": ["expectation", "suspense", "hopefulness"],
    },
    "disgust": {
        "revulsion": ["repulsion", "abhorrence", "nausea"],
        "aversion": ["distaste", "dislike", "disapproval"],
    },
}

# Flatten for lookups
_ALL_TERTIARY: List[str] = []
_TERTIARY_TO_SECONDARY: Dict[str, str] = {}
_TERTIARY_TO_BASIC: Dict[str, str] = {}
_SECONDARY_TO_BASIC: Dict[str, str] = {}
_ALL_SECONDARY: List[str] = []
_ALL_BASIC: List[str] = list(EMOTION_TAXONOMY.keys())

for basic, secondaries in EMOTION_TAXONOMY.items():
    for secondary, tertiaries in secondaries.items():
        _ALL_SECONDARY.append(secondary)
        _SECONDARY_TO_BASIC[secondary] = basic
        for t in tertiaries:
            _ALL_TERTIARY.append(t)
            _TERTIARY_TO_SECONDARY[t] = secondary
            _TERTIARY_TO_BASIC[t] = basic

_ALL_EMOTIONS = set(_ALL_BASIC + _ALL_SECONDARY + _ALL_TERTIARY)

# Total counts for reference
_N_BASIC = len(_ALL_BASIC)
_N_SECONDARY = len(_ALL_SECONDARY)
_N_TERTIARY = len(_ALL_TERTIARY)

# Maximum trend entries per user
_MAX_TREND_ENTRIES = 200


def _resolve_emotion_level(label: str) -> str:
    """Determine which taxonomy level an emotion label belongs to."""
    label_lower = label.lower().strip()
    if label_lower in _ALL_BASIC:
        return "basic"
    if label_lower in _SECONDARY_TO_BASIC:
        return "secondary"
    if label_lower in _TERTIARY_TO_BASIC:
        return "tertiary"
    return "unknown"


def _resolve_to_basic(label: str) -> Optional[str]:
    """Map any emotion label to its basic category."""
    label_lower = label.lower().strip()
    if label_lower in _ALL_BASIC:
        return label_lower
    if label_lower in _SECONDARY_TO_BASIC:
        return _SECONDARY_TO_BASIC[label_lower]
    if label_lower in _TERTIARY_TO_BASIC:
        return _TERTIARY_TO_BASIC[label_lower]
    return None


def _get_sibling_emotions(label: str) -> List[str]:
    """Get emotions at the same taxonomy level sharing the same parent."""
    label_lower = label.lower().strip()

    # Tertiary: siblings share the same secondary parent
    if label_lower in _TERTIARY_TO_SECONDARY:
        parent = _TERTIARY_TO_SECONDARY[label_lower]
        basic = _TERTIARY_TO_BASIC[label_lower]
        siblings = EMOTION_TAXONOMY[basic][parent]
        return [s for s in siblings if s != label_lower]

    # Secondary: siblings share the same basic parent
    if label_lower in _SECONDARY_TO_BASIC:
        basic = _SECONDARY_TO_BASIC[label_lower]
        siblings = list(EMOTION_TAXONOMY[basic].keys())
        return [s for s in siblings if s != label_lower]

    return []


class EmotionalGranularityEngine:
    """Measures and trains emotional granularity from self-report data.

    Tracks per-user emotion histories, computes granularity metrics, and
    generates differentiation exercises to improve emotional vocabulary.
    """

    def __init__(self) -> None:
        self._user_histories: Dict[str, List[Dict[str, Any]]] = {}
        self._user_trends: Dict[str, List[Dict[str, Any]]] = {}

    def _ensure_user(self, user_id: str) -> None:
        if user_id not in self._user_histories:
            self._user_histories[user_id] = []
        if user_id not in self._user_trends:
            self._user_trends[user_id] = []

    def add_emotion_report(
        self,
        user_id: str,
        label: str,
        intensity: float = 0.5,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a self-reported emotion entry for a user.

        Args:
            user_id: User identifier.
            label: Emotion label (any level of taxonomy).
            intensity: Emotion intensity 0.0 to 1.0.
            context: Optional context description.

        Returns:
            Dict with status and report count.
        """
        self._ensure_user(user_id)
        level = _resolve_emotion_level(label)
        basic = _resolve_to_basic(label)
        entry: Dict[str, Any] = {
            "label": label.lower().strip(),
            "intensity": max(0.0, min(1.0, intensity)),
            "level": level,
            "basic_category": basic,
            "context": context,
        }
        self._user_histories[user_id].append(entry)
        return {
            "status": "recorded",
            "report_count": len(self._user_histories[user_id]),
            "level": level,
            "basic_category": basic,
        }

    def compute_granularity_score(
        self, emotion_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute the Emotional Granularity Score from emotion history.

        Granularity is measured by:
        1. Label diversity: how many unique labels relative to total reports
        2. Taxonomy depth: proportion of secondary/tertiary vs basic labels
        3. Intra-class correlation (ICC): consistency in distinguishing
           similar emotions (measured by spread within basic categories)
        4. Distribution evenness: how evenly labels are distributed (entropy)

        Args:
            emotion_history: List of dicts with at least a "label" key.

        Returns:
            Dict with granularity score (0-1), sub-scores, and level.
        """
        if not emotion_history:
            return {
                "granularity_score": 0.0,
                "label_diversity": 0.0,
                "taxonomy_depth": 0.0,
                "distribution_evenness": 0.0,
                "icc_score": 0.0,
                "n_reports": 0,
                "level": "insufficient_data",
            }

        labels = [
            e.get("label", "").lower().strip()
            for e in emotion_history
            if e.get("label")
        ]
        if not labels:
            return {
                "granularity_score": 0.0,
                "label_diversity": 0.0,
                "taxonomy_depth": 0.0,
                "distribution_evenness": 0.0,
                "icc_score": 0.0,
                "n_reports": 0,
                "level": "insufficient_data",
            }

        n_reports = len(labels)
        unique_labels = set(labels)
        n_unique = len(unique_labels)

        # 1. Label diversity: unique labels / total possible at used level
        # Scale by log to avoid punishing users with fewer total reports
        max_possible = max(_N_TERTIARY, 1)
        raw_diversity = n_unique / max_possible
        # Adjust for sample size: more reports needed for more labels
        sample_factor = min(1.0, math.log1p(n_reports) / math.log1p(50))
        label_diversity = min(1.0, raw_diversity * 3.0) * sample_factor

        # 2. Taxonomy depth: weighted proportion at each level
        levels = [_resolve_emotion_level(lbl) for lbl in labels]
        level_counts = Counter(levels)
        n_valid = sum(
            level_counts.get(lv, 0) for lv in ("basic", "secondary", "tertiary")
        )
        if n_valid > 0:
            basic_frac = level_counts.get("basic", 0) / n_valid
            secondary_frac = level_counts.get("secondary", 0) / n_valid
            tertiary_frac = level_counts.get("tertiary", 0) / n_valid
            taxonomy_depth = (
                0.1 * basic_frac + 0.4 * secondary_frac + 1.0 * tertiary_frac
            )
        else:
            taxonomy_depth = 0.0

        # 3. Distribution evenness (Shannon entropy normalized)
        label_counts = Counter(labels)
        if n_unique > 1:
            entropy = 0.0
            for count in label_counts.values():
                p = count / n_reports
                if p > 0:
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(n_unique)
            distribution_evenness = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            distribution_evenness = 0.0

        # 4. Intra-class correlation (ICC proxy): how many distinct labels
        #    are used within each basic category. High ICC = user distinguishes
        #    between similar emotions (e.g., uses "frustration", "resentment",
        #    "irritation" under anger, not just "anger").
        icc_score = self._compute_icc_proxy(labels)

        # Composite granularity score
        granularity_score = (
            0.25 * label_diversity
            + 0.25 * taxonomy_depth
            + 0.25 * distribution_evenness
            + 0.25 * icc_score
        )
        granularity_score = max(0.0, min(1.0, granularity_score))

        level = self.classify_granularity_level(granularity_score)

        return {
            "granularity_score": round(granularity_score, 4),
            "label_diversity": round(label_diversity, 4),
            "taxonomy_depth": round(taxonomy_depth, 4),
            "distribution_evenness": round(distribution_evenness, 4),
            "icc_score": round(icc_score, 4),
            "n_reports": n_reports,
            "level": level,
        }

    @staticmethod
    def _compute_icc_proxy(labels: List[str]) -> float:
        """Compute ICC proxy: within-category label variety.

        For each basic category that appears, count how many distinct
        sub-labels (secondary + tertiary) the user uses. Higher variety
        within a category = higher differentiation ability.
        """
        category_labels: Dict[str, set] = {}
        for label in labels:
            basic = _resolve_to_basic(label)
            if basic is None:
                continue
            if basic not in category_labels:
                category_labels[basic] = set()
            category_labels[basic].add(label)

        if not category_labels:
            return 0.0

        # For each category, compute the ratio of unique sublabels used
        # vs total sublabels available in that category
        scores = []
        for basic, used in category_labels.items():
            # Count available labels at all levels for this basic category
            available = {basic}
            if basic in EMOTION_TAXONOMY:
                for sec, tertiaries in EMOTION_TAXONOMY[basic].items():
                    available.add(sec)
                    available.update(tertiaries)
            n_available = len(available)
            n_used = len(used)
            # Ratio, but only count if they use more than just the basic label
            if n_used > 1:
                scores.append(min(1.0, (n_used - 1) / max(n_available - 1, 1)))
            else:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def classify_granularity_level(score: float) -> str:
        """Classify granularity score into a human-readable level.

        Args:
            score: Granularity score between 0 and 1.

        Returns:
            One of: "very_low", "low", "moderate", "high", "very_high".
        """
        if score >= 0.8:
            return "very_high"
        if score >= 0.6:
            return "high"
        if score >= 0.4:
            return "moderate"
        if score >= 0.2:
            return "low"
        return "very_low"

    def generate_differentiation_exercise(
        self,
        emotion_history: List[Dict[str, Any]],
        difficulty: str = "auto",
    ) -> Dict[str, Any]:
        """Generate an emotion differentiation training exercise.

        Creates a task where the user must distinguish between similar
        emotions (e.g., "Is this frustration or resentment?").

        Args:
            emotion_history: User's emotion history for personalization.
            difficulty: "easy", "medium", "hard", or "auto" (based on
                        current granularity level).

        Returns:
            Dict with exercise type, scenario, options, correct answer,
            and explanation.
        """
        # Determine difficulty from history if auto
        if difficulty == "auto":
            score_result = self.compute_granularity_score(emotion_history)
            level = score_result["level"]
            if level in ("very_low", "low"):
                difficulty = "easy"
            elif level == "moderate":
                difficulty = "medium"
            else:
                difficulty = "hard"

        if difficulty == "easy":
            return self._generate_basic_exercise()
        elif difficulty == "medium":
            return self._generate_secondary_exercise(emotion_history)
        else:
            return self._generate_tertiary_exercise(emotion_history)

    def _generate_basic_exercise(self) -> Dict[str, Any]:
        """Generate an easy exercise: distinguish between basic emotions."""
        basics = list(EMOTION_TAXONOMY.keys())
        chosen = random.sample(basics, min(2, len(basics)))
        if len(chosen) < 2:
            chosen = ["joy", "sadness"]
        correct = chosen[0]

        # Get a representative secondary for the scenario
        secondaries = list(EMOTION_TAXONOMY[correct].keys())
        scenario_emotion = secondaries[0] if secondaries else correct

        scenarios = {
            "joy": (
                "After finishing a challenging project, you feel a warm sense "
                "of accomplishment and a smile spreading across your face."
            ),
            "sadness": (
                "You learn that a childhood friend has moved far away and you "
                "will not see them for a long time. A heaviness settles in."
            ),
            "anger": (
                "Someone takes credit for your work in a meeting and your "
                "manager praises them. You feel heat rising in your chest."
            ),
            "fear": (
                "You are about to give a presentation to 200 people and your "
                "hands are shaking, your heart is racing."
            ),
            "surprise": (
                "You open your front door and all your friends are there "
                "shouting 'surprise!' for your birthday."
            ),
            "disgust": (
                "You discover that a food you just ate was spoiled and "
                "your stomach turns immediately."
            ),
        }

        scenario = scenarios.get(
            correct,
            f"You experience a strong feeling of {scenario_emotion}.",
        )

        return {
            "exercise_type": "basic_differentiation",
            "difficulty": "easy",
            "scenario": scenario,
            "question": "Which basic emotion best describes this experience?",
            "options": chosen,
            "correct_answer": correct,
            "explanation": (
                f"This scenario describes {correct}. The key signals are the "
                f"physical sensations and context that distinguish {correct} "
                f"from {chosen[1]}."
            ),
        }

    def _generate_secondary_exercise(
        self, emotion_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a medium exercise: distinguish between secondary emotions."""
        # Pick a basic category, preferably one the user reports often
        labels = [
            e.get("label", "").lower().strip()
            for e in emotion_history
            if e.get("label")
        ]
        basic_counts: Dict[str, int] = Counter()
        for lbl in labels:
            b = _resolve_to_basic(lbl)
            if b:
                basic_counts[b] += 1

        if basic_counts:
            basic = max(basic_counts, key=lambda k: basic_counts[k])
        else:
            basic = random.choice(_ALL_BASIC)

        secondaries = list(EMOTION_TAXONOMY[basic].keys())
        if len(secondaries) < 2:
            # Fallback to a category with enough secondaries
            for b in _ALL_BASIC:
                if len(EMOTION_TAXONOMY[b]) >= 2:
                    basic = b
                    secondaries = list(EMOTION_TAXONOMY[basic].keys())
                    break

        chosen = random.sample(secondaries, min(3, len(secondaries)))
        correct = chosen[0]

        tertiaries = EMOTION_TAXONOMY[basic][correct]
        example = tertiaries[0] if tertiaries else correct

        return {
            "exercise_type": "secondary_differentiation",
            "difficulty": "medium",
            "scenario": (
                f"Within the domain of {basic}, consider a feeling that "
                f"specifically involves {example}. This is more than just "
                f"general {basic}."
            ),
            "question": (
                f"Which specific type of {basic} does this best represent?"
            ),
            "options": chosen,
            "correct_answer": correct,
            "explanation": (
                f"{correct} is characterized by {example} and related feelings. "
                f"It differs from {chosen[1] if len(chosen) > 1 else basic} in "
                f"its specific quality and typical triggers."
            ),
        }

    def _generate_tertiary_exercise(
        self, emotion_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a hard exercise: distinguish between tertiary emotions."""
        # Pick a secondary category with enough tertiaries
        candidates: List[Tuple[str, str, List[str]]] = []
        for basic, secondaries in EMOTION_TAXONOMY.items():
            for sec, tertiaries in secondaries.items():
                if len(tertiaries) >= 2:
                    candidates.append((basic, sec, tertiaries))

        if not candidates:
            return self._generate_secondary_exercise(emotion_history)

        basic, secondary, tertiaries = random.choice(candidates)
        chosen = random.sample(tertiaries, min(3, len(tertiaries)))
        correct = chosen[0]

        return {
            "exercise_type": "tertiary_differentiation",
            "difficulty": "hard",
            "scenario": (
                f"You are experiencing a form of {secondary} (under {basic}). "
                f"The feeling has a very specific quality -- it is best "
                f"described as {correct}."
            ),
            "question": (
                f"Which specific shade of {secondary} does this represent?"
            ),
            "options": chosen,
            "correct_answer": correct,
            "explanation": (
                f"{correct} is a specific form of {secondary}. While all these "
                f"emotions fall under {secondary}, {correct} has a distinct "
                f"quality that separates it from "
                f"{', '.join(c for c in chosen if c != correct)}."
            ),
        }

    def compute_emotion_vocabulary(
        self, emotion_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute the user's emotion vocabulary profile.

        Analyzes which emotions the user reports and at what taxonomy depth,
        identifies gaps in their vocabulary, and suggests areas for growth.

        Args:
            emotion_history: List of dicts with at least a "label" key.

        Returns:
            Dict with vocabulary stats, most/least used, gaps, and suggestions.
        """
        if not emotion_history:
            return {
                "total_reports": 0,
                "unique_labels": 0,
                "basic_count": 0,
                "secondary_count": 0,
                "tertiary_count": 0,
                "unknown_count": 0,
                "most_used": [],
                "categories_used": [],
                "categories_missing": list(_ALL_BASIC),
                "depth_score": 0.0,
                "suggestions": [
                    "Start by labeling your emotions with basic categories "
                    "(joy, sadness, anger, fear, surprise, disgust)."
                ],
            }

        labels = [
            e.get("label", "").lower().strip()
            for e in emotion_history
            if e.get("label")
        ]
        if not labels:
            return self.compute_emotion_vocabulary([])

        label_counts = Counter(labels)
        n_unique = len(label_counts)

        level_counts = Counter(_resolve_emotion_level(lbl) for lbl in labels)
        basic_count = level_counts.get("basic", 0)
        secondary_count = level_counts.get("secondary", 0)
        tertiary_count = level_counts.get("tertiary", 0)
        unknown_count = level_counts.get("unknown", 0)

        most_used = [
            {"label": lbl, "count": cnt}
            for lbl, cnt in label_counts.most_common(10)
        ]

        # Which basic categories appear in the history
        categories_used = set()
        for lbl in labels:
            b = _resolve_to_basic(lbl)
            if b:
                categories_used.add(b)

        categories_missing = [b for b in _ALL_BASIC if b not in categories_used]

        # Depth score: ratio of secondary+tertiary to all valid labels
        valid = basic_count + secondary_count + tertiary_count
        if valid > 0:
            depth_score = (secondary_count + 2 * tertiary_count) / (2 * valid)
        else:
            depth_score = 0.0
        depth_score = min(1.0, depth_score)

        # Suggestions
        suggestions: List[str] = []
        if categories_missing:
            suggestions.append(
                f"Try noticing when you feel {categories_missing[0]}. "
                f"You have not reported any {categories_missing[0]}-family emotions yet."
            )
        if tertiary_count == 0 and secondary_count > 0:
            suggestions.append(
                "Try using more specific emotion words. For example, instead "
                "of 'anxiety', try 'nervousness', 'worry', or 'apprehension'."
            )
        if basic_count > secondary_count + tertiary_count:
            suggestions.append(
                "Most of your reports use basic labels. Try breaking down "
                "'anger' into frustration, resentment, or contempt."
            )
        if n_unique < 5 and len(labels) >= 10:
            suggestions.append(
                "Your emotion vocabulary is limited. The emotion wheel has "
                f"50+ distinct emotions -- you are using only {n_unique}."
            )
        if not suggestions:
            suggestions.append(
                "Great vocabulary depth! Keep exploring nuanced emotion labels."
            )

        return {
            "total_reports": len(labels),
            "unique_labels": n_unique,
            "basic_count": basic_count,
            "secondary_count": secondary_count,
            "tertiary_count": tertiary_count,
            "unknown_count": unknown_count,
            "most_used": most_used,
            "categories_used": sorted(categories_used),
            "categories_missing": categories_missing,
            "depth_score": round(depth_score, 4),
            "suggestions": suggestions,
        }

    def track_granularity_trend(
        self, user_id: str, emotion_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute and store a granularity score for trend tracking.

        Args:
            user_id: User identifier.
            emotion_history: Current emotion history.

        Returns:
            Dict with current score, trend data, and improvement info.
        """
        self._ensure_user(user_id)
        score_result = self.compute_granularity_score(emotion_history)

        entry = {
            "granularity_score": score_result["granularity_score"],
            "level": score_result["level"],
            "n_reports": score_result["n_reports"],
        }
        self._user_trends[user_id].append(entry)

        # Enforce cap
        if len(self._user_trends[user_id]) > _MAX_TREND_ENTRIES:
            self._user_trends[user_id] = self._user_trends[user_id][
                -_MAX_TREND_ENTRIES:
            ]

        trend = self._user_trends[user_id]
        scores = [t["granularity_score"] for t in trend]

        improvement = 0.0
        if len(scores) >= 2:
            # Compare last 3 scores to first 3 scores
            window = min(3, len(scores) // 2)
            early = sum(scores[:window]) / window
            recent = sum(scores[-window:]) / window
            improvement = recent - early

        return {
            "current": score_result,
            "trend_length": len(trend),
            "scores": scores,
            "improvement": round(improvement, 4),
            "improving": improvement > 0.01,
        }

    def profile_to_dict(self, user_id: str) -> Dict[str, Any]:
        """Get a full granularity profile for a user.

        Args:
            user_id: User identifier.

        Returns:
            Dict with history, scores, vocabulary, and trend data.
        """
        self._ensure_user(user_id)
        history = self._user_histories.get(user_id, [])
        score = self.compute_granularity_score(history)
        vocab = self.compute_emotion_vocabulary(history)
        trend = self._user_trends.get(user_id, [])

        return {
            "user_id": user_id,
            "n_reports": len(history),
            "granularity": score,
            "vocabulary": vocab,
            "trend": trend,
        }

    def get_taxonomy(self) -> Dict[str, Any]:
        """Return the full emotion taxonomy and counts."""
        return {
            "taxonomy": EMOTION_TAXONOMY,
            "counts": {
                "basic": _N_BASIC,
                "secondary": _N_SECONDARY,
                "tertiary": _N_TERTIARY,
                "total": _N_BASIC + _N_SECONDARY + _N_TERTIARY,
            },
        }
