"""Somatic marker mapping — body sensation to emotion prediction.

Maps body sensations to emotions using population priors derived from
Nummenmaa et al. (2014) body maps as a starting point. The model:

1. Takes body-map input (list of region/sensation/intensity tuples) and
   predicts valence/arousal on a continuous scale.
2. Learns personal somatic signatures over time — which body patterns
   map to which emotions for a specific user.
3. Detects somatic dissociation (mismatch between body signals and
   self-reported emotion).
4. Computes a somatic awareness score (how differentiated and detailed
   a user's body reports are over time).

Body regions (14 total):
    head, face, neck, chest, upper_back, lower_back, stomach,
    left_arm, right_arm, left_hand, right_hand, left_leg, right_leg, feet

Sensation types (12 total):
    warmth, cold, tingling, tension, pain, heaviness,
    lightness, buzzing, numbness, nausea, pressure, hollowness

References:
    Nummenmaa et al. (2014) — Bodily maps of emotions. PNAS 111(2), 646-651.
    Damasio (1994) — Descartes' Error: somatic marker hypothesis.
    Craig (2002) — How do you feel? Interoception: the sense of the body.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────

BODY_REGIONS: List[str] = [
    "head", "face", "neck", "chest",
    "upper_back", "lower_back", "stomach",
    "left_arm", "right_arm",
    "left_hand", "right_hand",
    "left_leg", "right_leg", "feet",
]

SENSATION_TYPES: List[str] = [
    "warmth", "cold", "tingling", "tension",
    "pain", "heaviness", "lightness", "buzzing",
    "numbness", "nausea", "pressure", "hollowness",
]

# Maximum history entries per user to prevent unbounded memory growth.
_MAX_HISTORY = 500

# ── Population Priors (Nummenmaa et al. 2014) ────────────────────────
#
# Valence and arousal priors per (region, sensation) pair, encoding
# population-average mappings from Nummenmaa's body map data.
#
# Format: (region, sensation) -> (valence_delta, arousal_delta)
#   valence_delta: contribution to valence, range roughly -1 to +1
#   arousal_delta: contribution to arousal, range roughly -1 to +1
#
# These are sparse — only the most significant pairings are encoded.
# Missing pairs contribute (0, 0).

_POPULATION_PRIORS: Dict[Tuple[str, str], Tuple[float, float]] = {
    # ── Chest ──
    ("chest", "warmth"):     ( 0.6,   0.3),   # warmth in chest = positive affect
    ("chest", "tension"):    (-0.5,   0.6),   # chest tightness = anxiety
    ("chest", "pressure"):   (-0.4,   0.5),   # chest pressure = anxiety / dread
    ("chest", "pain"):       (-0.6,   0.7),   # chest pain = distress
    ("chest", "hollowness"): (-0.7,  -0.3),   # hollow chest = sadness / emptiness
    ("chest", "lightness"):  ( 0.5,   0.2),   # light chest = relief / joy
    ("chest", "tingling"):   ( 0.3,   0.4),   # chest tingling = excitement
    ("chest", "cold"):       (-0.4,  -0.2),   # cold chest = withdrawal
    # ── Face ──
    ("face", "warmth"):      ( 0.5,   0.4),   # blushing / warm face = positive arousal
    ("face", "tension"):     (-0.3,   0.5),   # facial tension = stress / anger
    ("face", "tingling"):    ( 0.2,   0.3),   # tingling face = excitement
    ("face", "numbness"):    (-0.3,  -0.4),   # facial numbness = dissociation
    ("face", "cold"):        (-0.2,  -0.3),   # cold face = fear / withdrawal
    # ── Head ──
    ("head", "tension"):     (-0.3,   0.5),   # headache / tension = stress
    ("head", "pressure"):    (-0.4,   0.4),   # head pressure = overwhelm
    ("head", "warmth"):      ( 0.2,   0.2),   # warm head = engagement
    ("head", "pain"):        (-0.5,   0.6),   # head pain = distress
    ("head", "tingling"):    ( 0.1,   0.3),   # head tingling = curiosity / arousal
    ("head", "lightness"):   ( 0.4,  -0.1),   # light head = calm / relief
    ("head", "buzzing"):     (-0.1,   0.4),   # buzzing head = overstimulation
    # ── Stomach ──
    ("stomach", "tension"):  (-0.5,   0.5),   # gut tension = fear / anxiety
    ("stomach", "nausea"):   (-0.6,   0.4),   # nausea = disgust / dread
    ("stomach", "warmth"):   ( 0.4,   0.1),   # warm belly = contentment
    ("stomach", "hollowness"):(-0.5, -0.3),   # hollow stomach = sadness / hunger
    ("stomach", "pain"):     (-0.5,   0.5),   # stomach pain = distress
    ("stomach", "pressure"): (-0.3,   0.3),   # stomach pressure = nervousness
    ("stomach", "cold"):     (-0.3,  -0.1),   # cold stomach = dread
    ("stomach", "buzzing"):  (-0.1,   0.3),   # butterflies = nervousness / excitement
    # ── Hands ──
    ("left_hand", "warmth"):  ( 0.4,   0.2),  # warm hands = comfort / connection
    ("right_hand", "warmth"): ( 0.4,   0.2),
    ("left_hand", "cold"):   (-0.2,   0.3),   # cold hands = fear / stress (vasoconstriction)
    ("right_hand", "cold"):  (-0.2,   0.3),
    ("left_hand", "tingling"):( 0.3,   0.3),  # tingling hands = excitement / nervousness
    ("right_hand", "tingling"):( 0.3,  0.3),
    ("left_hand", "numbness"):(-0.3, -0.3),   # numb hands = dissociation
    ("right_hand", "numbness"):(-0.3, -0.3),
    ("left_hand", "tension"): (-0.2,   0.4),  # clenched fists = anger
    ("right_hand", "tension"):(-0.2,   0.4),
    # ── Arms ──
    ("left_arm", "warmth"):   ( 0.3,   0.2),  # warm arms = openness
    ("right_arm", "warmth"):  ( 0.3,   0.2),
    ("left_arm", "tension"):  (-0.2,   0.5),  # tense arms = anger / bracing
    ("right_arm", "tension"): (-0.2,   0.5),
    ("left_arm", "heaviness"):(-0.4, -0.3),   # heavy arms = depression / fatigue
    ("right_arm", "heaviness"):(-0.4, -0.3),
    ("left_arm", "lightness"):( 0.3,   0.2),  # light arms = joy / energy
    ("right_arm", "lightness"):( 0.3,  0.2),
    ("left_arm", "tingling"): ( 0.2,   0.3),
    ("right_arm", "tingling"):( 0.2,   0.3),
    # ── Neck ──
    ("neck", "tension"):     (-0.4,   0.5),   # neck tension = stress / anger
    ("neck", "pain"):        (-0.4,   0.4),   # neck pain = chronic stress
    ("neck", "warmth"):      ( 0.2,   0.1),
    ("neck", "cold"):        (-0.2,  -0.2),
    # ── Back ──
    ("upper_back", "tension"):(-0.3,  0.4),   # upper back tension = burden / stress
    ("upper_back", "pain"):   (-0.4,  0.4),
    ("upper_back", "warmth"): ( 0.2,  0.1),
    ("lower_back", "tension"):(-0.3,  0.3),   # lower back tension = fear / instability
    ("lower_back", "pain"):   (-0.4,  0.3),
    ("lower_back", "warmth"): ( 0.2,  0.1),
    # ── Legs ──
    ("left_leg", "tension"):  (-0.2,   0.4),  # leg tension = fight/flight readiness
    ("right_leg", "tension"): (-0.2,   0.4),
    ("left_leg", "heaviness"):(-0.4, -0.4),   # heavy legs = depression / fatigue
    ("right_leg", "heaviness"):(-0.4, -0.4),
    ("left_leg", "lightness"):( 0.3,   0.3),  # light legs = joy / energy
    ("right_leg", "lightness"):( 0.3,  0.3),
    ("left_leg", "warmth"):   ( 0.2,   0.1),
    ("right_leg", "warmth"):  ( 0.2,   0.1),
    ("left_leg", "numbness"): (-0.3, -0.3),
    ("right_leg", "numbness"):(-0.3, -0.3),
    # ── Feet ──
    ("feet", "warmth"):      ( 0.3,   0.1),   # warm feet = grounded
    ("feet", "cold"):        (-0.2,  -0.1),   # cold feet = disconnection
    ("feet", "tingling"):    ( 0.2,   0.2),
    ("feet", "numbness"):    (-0.2,  -0.3),   # numb feet = dissociation
    ("feet", "heaviness"):   (-0.2,  -0.2),
}

# ── Emotion label mapping from valence/arousal quadrants ─────────────

_EMOTION_QUADRANTS = [
    # (valence_min, valence_max, arousal_min, arousal_max, label)
    ( 0.2,  1.0,  0.2,  1.0, "excitement"),   # high V, high A
    ( 0.2,  1.0, -1.0,  0.2, "contentment"),   # high V, low A
    (-1.0, -0.2,  0.2,  1.0, "anxiety"),        # low V, high A
    (-1.0, -0.2, -1.0,  0.2, "sadness"),        # low V, low A
    (-0.2,  0.2,  0.2,  1.0, "alertness"),      # neutral V, high A
    (-0.2,  0.2, -1.0,  0.2, "calm"),           # neutral V, low A
]


def _classify_emotion(valence: float, arousal: float) -> str:
    """Map valence/arousal to a discrete emotion label."""
    for v_min, v_max, a_min, a_max, label in _EMOTION_QUADRANTS:
        if v_min <= valence <= v_max and a_min <= arousal <= a_max:
            return label
    return "neutral"


# ── Core Model ───────────────────────────────────────────────────────


class SomaticMarkerModel:
    """Predict emotions from body-sensation maps.

    Maintains per-user learned somatic signatures that override
    population priors when sufficient data has been collected.
    """

    def __init__(self) -> None:
        # Per-user learned priors: user_id -> {(region, sensation) -> [valence_deltas], [arousal_deltas]}
        self._user_priors: Dict[str, Dict[Tuple[str, str], Dict[str, List[float]]]] = {}
        # Per-user history of body-map submissions
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # ── Public API ────────────────────────────────────────────────

    def predict_emotion_from_soma(
        self,
        body_map: List[Dict[str, Any]],
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Predict valence/arousal from a body-sensation map.

        Args:
            body_map: List of dicts, each with keys:
                - region (str): one of BODY_REGIONS
                - sensation_type (str): one of SENSATION_TYPES
                - intensity (int|float): 1-5 scale
            user_id: User identifier for personalization.

        Returns:
            Dict with valence (-1 to 1), arousal (-1 to 1),
            predicted_emotion (str), confidence (0-1),
            n_activations (int), dominant_region (str|None),
            dominant_sensation (str|None).
        """
        if not body_map:
            return {
                "valence": 0.0,
                "arousal": 0.0,
                "predicted_emotion": "neutral",
                "confidence": 0.0,
                "n_activations": 0,
                "dominant_region": None,
                "dominant_sensation": None,
            }

        # Filter and normalize inputs
        activations = self._parse_body_map(body_map)
        if not activations:
            return {
                "valence": 0.0,
                "arousal": 0.0,
                "predicted_emotion": "neutral",
                "confidence": 0.0,
                "n_activations": 0,
                "dominant_region": None,
                "dominant_sensation": None,
            }

        # Compute weighted valence/arousal from priors
        valence_sum = 0.0
        arousal_sum = 0.0
        weight_sum = 0.0

        for region, sensation, intensity in activations:
            v_delta, a_delta = self._get_prior(region, sensation, user_id)
            # Intensity acts as a weight (normalized to 0-1 from 1-5 scale)
            weight = intensity / 5.0
            valence_sum += v_delta * weight
            arousal_sum += a_delta * weight
            weight_sum += weight

        if weight_sum > 0:
            valence = valence_sum / weight_sum
            arousal = arousal_sum / weight_sum
        else:
            valence = 0.0
            arousal = 0.0

        valence = float(np.clip(valence, -1.0, 1.0))
        arousal = float(np.clip(arousal, -1.0, 1.0))

        # Confidence is based on number of activations and their intensity
        # More body regions activated with higher intensity = more confident
        max_possible_weight = len(BODY_REGIONS)  # all regions at intensity 5
        confidence = float(np.clip(weight_sum / max(max_possible_weight * 0.3, 1.0), 0.0, 1.0))

        # Dominant region/sensation by intensity
        dominant = max(activations, key=lambda x: x[2])
        dominant_region = dominant[0]
        dominant_sensation = dominant[1]

        predicted_emotion = _classify_emotion(valence, arousal)

        result = {
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "predicted_emotion": predicted_emotion,
            "confidence": round(confidence, 4),
            "n_activations": len(activations),
            "dominant_region": dominant_region,
            "dominant_sensation": dominant_sensation,
        }

        # Store in history
        self._history[user_id].append(result)
        if len(self._history[user_id]) > _MAX_HISTORY:
            self._history[user_id] = self._history[user_id][-_MAX_HISTORY:]

        return result

    def learn_somatic_pairing(
        self,
        body_map: List[Dict[str, Any]],
        reported_valence: float,
        reported_arousal: float,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Learn a personal somatic-emotion pairing.

        Records the association between the given body sensations and
        the user's reported emotional state. Over time, this overrides
        population priors for this user.

        Args:
            body_map: Same format as predict_emotion_from_soma.
            reported_valence: User-reported valence (-1 to 1).
            reported_arousal: User-reported arousal (-1 to 1).
            user_id: User identifier.

        Returns:
            Dict with status, n_pairings_learned, total_pairings.
        """
        reported_valence = float(np.clip(reported_valence, -1.0, 1.0))
        reported_arousal = float(np.clip(reported_arousal, -1.0, 1.0))

        activations = self._parse_body_map(body_map)
        if not activations:
            return {
                "status": "no_valid_activations",
                "n_pairings_learned": 0,
                "total_pairings": self._count_user_pairings(user_id),
            }

        if user_id not in self._user_priors:
            self._user_priors[user_id] = {}

        n_learned = 0
        for region, sensation, intensity in activations:
            key = (region, sensation)
            if key not in self._user_priors[user_id]:
                self._user_priors[user_id][key] = {
                    "valence": [],
                    "arousal": [],
                }
            # Weight the observation by intensity
            weight = intensity / 5.0
            self._user_priors[user_id][key]["valence"].append(
                reported_valence * weight
            )
            self._user_priors[user_id][key]["arousal"].append(
                reported_arousal * weight
            )
            n_learned += 1

        return {
            "status": "learned",
            "n_pairings_learned": n_learned,
            "total_pairings": self._count_user_pairings(user_id),
        }

    def compute_somatic_signature(
        self,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Compute the personal somatic signature for a user.

        The signature describes which body-sensation patterns are most
        strongly associated with positive/negative emotions for this user.

        Returns:
            Dict with user_id, has_signature (bool), n_pairings (int),
            top_positive (list), top_negative (list), signature_strength (float).
        """
        priors = self._user_priors.get(user_id, {})
        if not priors:
            return {
                "user_id": user_id,
                "has_signature": False,
                "n_pairings": 0,
                "top_positive": [],
                "top_negative": [],
                "signature_strength": 0.0,
            }

        # Compute mean valence per (region, sensation)
        scored: List[Tuple[str, str, float, int]] = []
        for (region, sensation), data in priors.items():
            vals = data["valence"]
            if vals:
                mean_v = float(np.mean(vals))
                scored.append((region, sensation, mean_v, len(vals)))

        if not scored:
            return {
                "user_id": user_id,
                "has_signature": False,
                "n_pairings": 0,
                "top_positive": [],
                "top_negative": [],
                "signature_strength": 0.0,
            }

        scored.sort(key=lambda x: x[2], reverse=True)
        top_positive = [
            {"region": r, "sensation": s, "mean_valence": round(v, 4), "n_observations": n}
            for r, s, v, n in scored[:5] if v > 0
        ]
        top_negative = [
            {"region": r, "sensation": s, "mean_valence": round(v, 4), "n_observations": n}
            for r, s, v, n in scored if v < 0
        ]
        top_negative.sort(key=lambda x: x["mean_valence"])
        top_negative = top_negative[:5]

        total_pairings = self._count_user_pairings(user_id)
        # Signature strength based on number of learned pairings
        # (more data = stronger / more reliable signature)
        strength = float(np.clip(total_pairings / 50.0, 0.0, 1.0))

        return {
            "user_id": user_id,
            "has_signature": True,
            "n_pairings": total_pairings,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "signature_strength": round(strength, 4),
        }

    def compute_somatic_awareness_score(
        self,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Compute a somatic awareness score for a user.

        The score measures how differentiated and detailed the user's
        body reports are, based on:
        1. Number of distinct regions reported
        2. Number of distinct sensation types used
        3. Variation in intensity ratings (not always 5 or always 1)
        4. Total number of reports

        Returns:
            Dict with user_id, awareness_score (0-1), level (str),
            n_reports, distinct_regions, distinct_sensations,
            intensity_variance.
        """
        history = self._history.get(user_id, [])
        priors = self._user_priors.get(user_id, {})

        # Collect all (region, sensation) keys from learned priors
        all_regions: set = set()
        all_sensations: set = set()
        all_intensities: List[float] = []

        for (region, sensation), data in priors.items():
            all_regions.add(region)
            all_sensations.add(sensation)
            all_intensities.extend([abs(v) for v in data.get("valence", [])])

        n_reports = len(history)
        n_regions = len(all_regions)
        n_sensations = len(all_sensations)

        if n_reports == 0 and not priors:
            return {
                "user_id": user_id,
                "awareness_score": 0.0,
                "level": "minimal",
                "n_reports": 0,
                "distinct_regions": 0,
                "distinct_sensations": 0,
                "intensity_variance": 0.0,
            }

        # Component scores (each 0-1)
        # 1. Region diversity: how many of 14 regions have been used
        region_score = min(n_regions / 7.0, 1.0)  # 7+ regions = max score

        # 2. Sensation diversity: how many of 12 types have been used
        sensation_score = min(n_sensations / 6.0, 1.0)  # 6+ types = max score

        # 3. Intensity variance: measures nuance in ratings
        if len(all_intensities) >= 2:
            intensity_var = float(np.var(all_intensities))
            intensity_score = min(intensity_var / 0.5, 1.0)
        else:
            intensity_var = 0.0
            intensity_score = 0.0

        # 4. Report count: more reports = more data = more awareness
        report_score = min(n_reports / 20.0, 1.0)

        # Weighted composite
        awareness = (
            0.30 * region_score
            + 0.25 * sensation_score
            + 0.20 * intensity_score
            + 0.25 * report_score
        )
        awareness = float(np.clip(awareness, 0.0, 1.0))

        if awareness >= 0.7:
            level = "high"
        elif awareness >= 0.4:
            level = "moderate"
        elif awareness >= 0.15:
            level = "low"
        else:
            level = "minimal"

        return {
            "user_id": user_id,
            "awareness_score": round(awareness, 4),
            "level": level,
            "n_reports": n_reports,
            "distinct_regions": n_regions,
            "distinct_sensations": n_sensations,
            "intensity_variance": round(intensity_var, 4),
        }

    def detect_somatic_dissociation(
        self,
        body_map: List[Dict[str, Any]],
        reported_valence: float,
        reported_arousal: float,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Detect somatic dissociation.

        Dissociation occurs when the body's signals suggest one emotional
        state but the user reports a different one. For example, the body
        shows high tension and arousal patterns but the user reports
        feeling calm.

        Args:
            body_map: Body sensation map.
            reported_valence: User-reported valence (-1 to 1).
            reported_arousal: User-reported arousal (-1 to 1).
            user_id: User identifier.

        Returns:
            Dict with dissociation_detected (bool), dissociation_score (0-1),
            valence_mismatch (float), arousal_mismatch (float),
            somatic_valence (float), somatic_arousal (float),
            reported_valence (float), reported_arousal (float).
        """
        reported_valence = float(np.clip(reported_valence, -1.0, 1.0))
        reported_arousal = float(np.clip(reported_arousal, -1.0, 1.0))

        prediction = self.predict_emotion_from_soma(body_map, user_id)
        # Remove the stored history entry since this is a detection call,
        # not a real prediction submission — pop the last entry we just added.
        if self._history.get(user_id):
            self._history[user_id].pop()

        somatic_v = prediction["valence"]
        somatic_a = prediction["arousal"]

        valence_mismatch = abs(somatic_v - reported_valence)
        arousal_mismatch = abs(somatic_a - reported_arousal)

        # Dissociation score: average of valence and arousal mismatch,
        # scaled to 0-1 (max mismatch is 2.0 on each axis)
        dissociation_score = (valence_mismatch + arousal_mismatch) / 4.0
        dissociation_score = float(np.clip(dissociation_score, 0.0, 1.0))

        # Threshold for flagging dissociation
        dissociation_detected = dissociation_score > 0.3

        return {
            "dissociation_detected": dissociation_detected,
            "dissociation_score": round(dissociation_score, 4),
            "valence_mismatch": round(valence_mismatch, 4),
            "arousal_mismatch": round(arousal_mismatch, 4),
            "somatic_valence": somatic_v,
            "somatic_arousal": somatic_a,
            "reported_valence": round(reported_valence, 4),
            "reported_arousal": round(reported_arousal, 4),
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Return prediction history for a user."""
        history = self._history.get(user_id, [])
        if last_n is not None and last_n > 0:
            return list(history[-last_n:])
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Clear all data for a user."""
        self._user_priors.pop(user_id, None)
        if user_id in self._history:
            del self._history[user_id]

    # ── Private Helpers ───────────────────────────────────────────

    def _parse_body_map(
        self, body_map: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, float]]:
        """Validate and parse body-map input into (region, sensation, intensity) tuples."""
        activations: List[Tuple[str, str, float]] = []
        for entry in body_map:
            region = entry.get("region", "")
            sensation = entry.get("sensation_type", "")
            intensity = entry.get("intensity", 0)

            if region not in BODY_REGIONS:
                continue
            if sensation not in SENSATION_TYPES:
                continue
            try:
                intensity = float(intensity)
            except (TypeError, ValueError):
                continue
            intensity = max(1.0, min(5.0, intensity))
            activations.append((region, sensation, intensity))
        return activations

    def _get_prior(
        self, region: str, sensation: str, user_id: str
    ) -> Tuple[float, float]:
        """Get the valence/arousal prior for a (region, sensation) pair.

        Uses personal priors when available (>= 3 observations),
        blending with population priors using a sigmoid ramp.
        """
        pop_v, pop_a = _POPULATION_PRIORS.get((region, sensation), (0.0, 0.0))

        user_data = self._user_priors.get(user_id, {}).get((region, sensation))
        if user_data is None:
            return pop_v, pop_a

        n_obs = len(user_data["valence"])
        if n_obs == 0:
            return pop_v, pop_a

        user_v = float(np.mean(user_data["valence"]))
        user_a = float(np.mean(user_data["arousal"]))

        # Blend: personal weight increases with observations.
        # At 3 obs: ~50% personal, at 10 obs: ~90% personal.
        personal_weight = 1.0 / (1.0 + math.exp(-0.7 * (n_obs - 3)))
        pop_weight = 1.0 - personal_weight

        blended_v = pop_weight * pop_v + personal_weight * user_v
        blended_a = pop_weight * pop_a + personal_weight * user_a

        return blended_v, blended_a

    def _count_user_pairings(self, user_id: str) -> int:
        """Count total learned observations for a user."""
        priors = self._user_priors.get(user_id, {})
        return sum(len(d["valence"]) for d in priors.values())
