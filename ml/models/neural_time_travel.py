"""Neural time travel -- replay and re-experience past emotional states.

Stores snapshots of notable emotional states with full EEG feature vectors
and context metadata.  Later, given a target emotion, finds the closest
stored state via cosine similarity, generates a neurofeedback target
parameter set, and plans a guided replay session that gradually shifts the
user from their current state toward the stored target.

Core capabilities:
    1. Emotional state library   -- store / retrieve / search snapshots
    2. State matching            -- cosine similarity on feature vectors
    3. Neurofeedback targeting   -- EEG parameter targets for guided shift
    4. Replay session planning   -- stepwise transition from current to target
    5. Visualization data        -- time-lapse emotional journey rendering

Issue #459.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical EEG feature names expected in every snapshot vector.
FEATURE_NAMES: List[str] = [
    "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
    "alpha_beta_ratio", "theta_beta_ratio", "frontal_asymmetry",
    "valence", "arousal",
]

# Number of default interpolation steps in a replay session.
DEFAULT_REPLAY_STEPS = 8

# Minimum cosine similarity to consider a match useful.
MIN_SIMILARITY_THRESHOLD = 0.10

# Emotion label -> typical (valence, arousal) anchors on [-1, 1] scale.
_EMOTION_ANCHORS: Dict[str, Tuple[float, float]] = {
    "happy":    ( 0.8,  0.6),
    "calm":     ( 0.6, -0.4),
    "excited":  ( 0.7,  0.9),
    "focused":  ( 0.3,  0.5),
    "sad":      (-0.7, -0.3),
    "anxious":  (-0.5,  0.7),
    "angry":    (-0.6,  0.8),
    "neutral":  ( 0.0,  0.0),
    "fearful":  (-0.7,  0.6),
    "relaxed":  ( 0.5, -0.5),
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EmotionalSnapshot:
    """A stored emotional state with full EEG features + context."""

    snapshot_id: str
    user_id: str
    emotion_label: str
    valence: float
    arousal: float
    feature_vector: List[float]
    context: str = ""
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class NeurofeedbackTarget:
    """Target EEG parameters for guiding a user toward a stored state."""

    target_alpha_power: float
    target_theta_power: float
    target_beta_power: float
    target_frontal_asymmetry: float
    target_valence: float
    target_arousal: float
    difficulty: float  # 0-1 how far from current state


@dataclass
class ReplayStep:
    """A single step in a guided replay session."""

    step_number: int
    total_steps: int
    target: NeurofeedbackTarget
    instruction: str
    duration_seconds: float
    progress_fraction: float  # 0.0 -> 1.0


@dataclass
class ReplaySession:
    """A full guided replay session plan."""

    session_id: str
    user_id: str
    source_label: str
    target_snapshot_id: str
    target_emotion: str
    steps: List[ReplayStep]
    total_duration_seconds: float
    estimated_difficulty: float
    created_at: float = field(default_factory=time.time)


@dataclass
class TravelProfile:
    """Summary of a user's time-travel journey (current -> target)."""

    current_valence: float
    current_arousal: float
    target_valence: float
    target_arousal: float
    emotional_distance: float
    estimated_steps: int
    journey_arc: str  # "ascending", "descending", "lateral", "complex"


# ---------------------------------------------------------------------------
# In-memory emotional state library
# ---------------------------------------------------------------------------

# user_id -> list of snapshots
_library: Dict[str, List[EmotionalSnapshot]] = {}


def _get_user_library(user_id: str) -> List[EmotionalSnapshot]:
    """Return the snapshot list for a user, creating if absent."""
    if user_id not in _library:
        _library[user_id] = []
    return _library[user_id]


def clear_library(user_id: Optional[str] = None) -> None:
    """Clear the snapshot library.  If user_id given, clear only that user."""
    if user_id is None:
        _library.clear()
    else:
        _library.pop(user_id, None)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def store_emotional_snapshot(
    user_id: str,
    emotion_label: str,
    feature_vector: List[float],
    valence: float = 0.0,
    arousal: float = 0.0,
    context: str = "",
    tags: Optional[List[str]] = None,
) -> EmotionalSnapshot:
    """Store an emotional state snapshot in the library.

    Parameters
    ----------
    user_id : str
        Owner of the snapshot.
    emotion_label : str
        Human-readable label (e.g. "happy", "calm").
    feature_vector : list[float]
        EEG feature vector (at least 3 elements for meaningful similarity).
    valence, arousal : float
        Circumplex coordinates, each in [-1, 1].
    context : str
        Free-text description of what was happening.
    tags : list[str] | None
        Optional searchable tags.

    Returns
    -------
    EmotionalSnapshot
        The newly created snapshot.
    """
    if len(feature_vector) < 1:
        raise ValueError("feature_vector must have at least 1 element")

    valence = max(-1.0, min(1.0, valence))
    arousal = max(-1.0, min(1.0, arousal))

    snap = EmotionalSnapshot(
        snapshot_id=str(uuid.uuid4()),
        user_id=user_id,
        emotion_label=emotion_label.lower().strip(),
        valence=valence,
        arousal=arousal,
        feature_vector=list(feature_vector),
        context=context,
        tags=[t.lower().strip() for t in (tags or [])],
    )
    _get_user_library(user_id).append(snap)
    logger.info(
        "Stored snapshot %s for user %s: %s (v=%.2f, a=%.2f)",
        snap.snapshot_id, user_id, emotion_label, valence, arousal,
    )
    return snap


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors.  Returns 0 if either is zero."""
    if len(a) != len(b):
        # Pad or truncate to shorter length
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-12 or mag_b < 1e-12:
        return 0.0
    return dot / (mag_a * mag_b)


def find_similar_states(
    user_id: str,
    target_vector: List[float],
    top_k: int = 5,
    emotion_filter: Optional[str] = None,
    min_similarity: float = MIN_SIMILARITY_THRESHOLD,
) -> List[Tuple[EmotionalSnapshot, float]]:
    """Find stored snapshots most similar to a target feature vector.

    Parameters
    ----------
    user_id : str
        Whose library to search.
    target_vector : list[float]
        Query feature vector.
    top_k : int
        Maximum number of results.
    emotion_filter : str | None
        If set, only return snapshots with this emotion label.
    min_similarity : float
        Discard results below this cosine similarity.

    Returns
    -------
    list[(EmotionalSnapshot, float)]
        Matches sorted by descending similarity.
    """
    library = _get_user_library(user_id)
    scored: List[Tuple[EmotionalSnapshot, float]] = []

    for snap in library:
        if emotion_filter and snap.emotion_label != emotion_filter.lower().strip():
            continue
        sim = _cosine_similarity(target_vector, snap.feature_vector)
        if sim >= min_similarity:
            scored.append((snap, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def generate_neurofeedback_target(
    current_features: List[float],
    target_snapshot: EmotionalSnapshot,
    blend: float = 1.0,
) -> NeurofeedbackTarget:
    """Create a neurofeedback target to guide the user toward a stored state.

    Parameters
    ----------
    current_features : list[float]
        User's current EEG feature vector (same dimensionality as snapshot).
    target_snapshot : EmotionalSnapshot
        The desired emotional state.
    blend : float
        Interpolation factor 0..1 (0 = current, 1 = full target).

    Returns
    -------
    NeurofeedbackTarget
    """
    blend = max(0.0, min(1.0, blend))
    tf = target_snapshot.feature_vector

    def _interp(idx: int, default: float = 0.0) -> float:
        cur = current_features[idx] if idx < len(current_features) else default
        tgt = tf[idx] if idx < len(tf) else default
        return cur + blend * (tgt - cur)

    # Map indices to FEATURE_NAMES:  alpha=2, theta=1, beta=3, fa=7
    target_alpha = _interp(2, 0.3)
    target_theta = _interp(1, 0.2)
    target_beta = _interp(3, 0.2)
    target_fa = _interp(7, 0.0)

    # Valence / arousal from snapshot
    target_valence = target_snapshot.valence
    target_arousal = target_snapshot.arousal

    # Difficulty: Euclidean distance in valence-arousal space, normalised to 0-1
    dv = target_valence - (current_features[8] if len(current_features) > 8 else 0.0)
    da = target_arousal - (current_features[9] if len(current_features) > 9 else 0.0)
    difficulty = min(1.0, math.sqrt(dv * dv + da * da) / 2.0)

    return NeurofeedbackTarget(
        target_alpha_power=round(target_alpha, 4),
        target_theta_power=round(target_theta, 4),
        target_beta_power=round(target_beta, 4),
        target_frontal_asymmetry=round(target_fa, 4),
        target_valence=round(target_valence, 4),
        target_arousal=round(target_arousal, 4),
        difficulty=round(difficulty, 4),
    )


def plan_replay_session(
    user_id: str,
    current_features: List[float],
    target_snapshot: EmotionalSnapshot,
    num_steps: int = DEFAULT_REPLAY_STEPS,
    step_duration: float = 30.0,
) -> ReplaySession:
    """Plan a guided neurofeedback replay session.

    Creates a series of progressive steps that interpolate from the user's
    current state toward the stored target state.

    Parameters
    ----------
    user_id : str
    current_features : list[float]
        User's current EEG feature vector.
    target_snapshot : EmotionalSnapshot
        Desired emotional state.
    num_steps : int
        Number of intermediate steps (min 2).
    step_duration : float
        Duration per step in seconds.

    Returns
    -------
    ReplaySession
    """
    num_steps = max(2, num_steps)
    step_duration = max(5.0, step_duration)

    steps: List[ReplayStep] = []
    for i in range(num_steps):
        progress = (i + 1) / num_steps
        blend = progress  # linear interpolation

        target = generate_neurofeedback_target(
            current_features, target_snapshot, blend=blend,
        )

        instruction = _generate_step_instruction(
            i + 1, num_steps, progress, target_snapshot.emotion_label, target,
        )

        steps.append(ReplayStep(
            step_number=i + 1,
            total_steps=num_steps,
            target=target,
            instruction=instruction,
            duration_seconds=step_duration,
            progress_fraction=round(progress, 4),
        ))

    overall_target = generate_neurofeedback_target(
        current_features, target_snapshot, blend=1.0,
    )

    return ReplaySession(
        session_id=str(uuid.uuid4()),
        user_id=user_id,
        source_label="current",
        target_snapshot_id=target_snapshot.snapshot_id,
        target_emotion=target_snapshot.emotion_label,
        steps=steps,
        total_duration_seconds=num_steps * step_duration,
        estimated_difficulty=overall_target.difficulty,
    )


def _generate_step_instruction(
    step: int,
    total: int,
    progress: float,
    emotion: str,
    target: NeurofeedbackTarget,
) -> str:
    """Generate a human-readable instruction for a replay step."""
    if progress <= 0.25:
        phase = "settling"
        guidance = "Begin by relaxing and settling into awareness of your current state."
    elif progress <= 0.50:
        phase = "approaching"
        guidance = f"Gently begin to recall the feeling of being {emotion}."
    elif progress <= 0.75:
        phase = "deepening"
        guidance = f"Deepen your connection with the {emotion} state. Let it become vivid."
    else:
        phase = "immersion"
        guidance = f"Fully immerse in the {emotion} experience. Let it fill your awareness."

    return f"Step {step}/{total} ({phase}): {guidance}"


def generate_visualization_data(
    user_id: str,
    current_features: List[float],
    target_snapshot: EmotionalSnapshot,
    num_frames: int = 20,
) -> Dict[str, Any]:
    """Generate data for a time-lapse emotional journey visualization.

    Returns a series of frames showing the interpolated emotional state
    from current to target, suitable for rendering as an animation.

    Parameters
    ----------
    user_id : str
    current_features : list[float]
        Current EEG feature vector.
    target_snapshot : EmotionalSnapshot
        Target emotional state.
    num_frames : int
        Number of animation frames.

    Returns
    -------
    dict
        Visualization data with frames, journey metadata, and keypoints.
    """
    num_frames = max(2, min(100, num_frames))

    cur_valence = current_features[8] if len(current_features) > 8 else 0.0
    cur_arousal = current_features[9] if len(current_features) > 9 else 0.0
    tgt_valence = target_snapshot.valence
    tgt_arousal = target_snapshot.arousal

    frames: List[Dict[str, Any]] = []
    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        # Smooth interpolation (ease-in-out)
        smooth_t = 0.5 * (1 - math.cos(math.pi * t))

        v = cur_valence + smooth_t * (tgt_valence - cur_valence)
        a = cur_arousal + smooth_t * (tgt_arousal - cur_arousal)

        # Interpolate feature vector
        interp_features: List[float] = []
        tgt_vec = target_snapshot.feature_vector
        for j in range(max(len(current_features), len(tgt_vec))):
            cf = current_features[j] if j < len(current_features) else 0.0
            tf = tgt_vec[j] if j < len(tgt_vec) else 0.0
            interp_features.append(round(cf + smooth_t * (tf - cf), 4))

        frames.append({
            "frame_index": i,
            "t": round(t, 4),
            "valence": round(v, 4),
            "arousal": round(a, 4),
            "emotion_blend": round(smooth_t, 4),
            "features": interp_features,
        })

    # Compute emotional distance
    emotional_distance = math.sqrt(
        (tgt_valence - cur_valence) ** 2 + (tgt_arousal - cur_arousal) ** 2
    )

    return {
        "user_id": user_id,
        "target_snapshot_id": target_snapshot.snapshot_id,
        "target_emotion": target_snapshot.emotion_label,
        "num_frames": num_frames,
        "frames": frames,
        "journey": {
            "start_valence": round(cur_valence, 4),
            "start_arousal": round(cur_arousal, 4),
            "end_valence": round(tgt_valence, 4),
            "end_arousal": round(tgt_arousal, 4),
            "emotional_distance": round(emotional_distance, 4),
        },
        "keypoints": _compute_keypoints(cur_valence, cur_arousal, tgt_valence, tgt_arousal),
        "timestamp": time.time(),
    }


def _compute_keypoints(
    sv: float, sa: float, ev: float, ea: float,
) -> List[Dict[str, Any]]:
    """Compute notable keypoints along the emotional journey."""
    keypoints = [
        {"label": "start", "valence": round(sv, 4), "arousal": round(sa, 4), "t": 0.0},
        {
            "label": "midpoint",
            "valence": round((sv + ev) / 2, 4),
            "arousal": round((sa + ea) / 2, 4),
            "t": 0.5,
        },
        {"label": "target", "valence": round(ev, 4), "arousal": round(ea, 4), "t": 1.0},
    ]
    return keypoints


def compute_travel_profile(
    current_features: List[float],
    target_snapshot: EmotionalSnapshot,
) -> TravelProfile:
    """Summarise the journey from current state to target.

    Returns
    -------
    TravelProfile
    """
    cur_v = current_features[8] if len(current_features) > 8 else 0.0
    cur_a = current_features[9] if len(current_features) > 9 else 0.0
    tgt_v = target_snapshot.valence
    tgt_a = target_snapshot.arousal

    dist = math.sqrt((tgt_v - cur_v) ** 2 + (tgt_a - cur_a) ** 2)

    # Estimate steps proportional to distance (more distance = more steps)
    est_steps = max(2, min(20, int(dist * 10) + 2))

    # Classify journey arc
    dv = tgt_v - cur_v
    da = tgt_a - cur_a
    if abs(dv) < 0.2 and abs(da) < 0.2:
        arc = "lateral"
    elif da > 0.3 and dv > 0.1:
        arc = "ascending"
    elif da < -0.3 and dv < -0.1:
        arc = "descending"
    else:
        arc = "complex"

    return TravelProfile(
        current_valence=round(cur_v, 4),
        current_arousal=round(cur_a, 4),
        target_valence=round(tgt_v, 4),
        target_arousal=round(tgt_a, 4),
        emotional_distance=round(dist, 4),
        estimated_steps=est_steps,
        journey_arc=arc,
    )


def profile_to_dict(profile: TravelProfile) -> Dict[str, Any]:
    """Serialise a TravelProfile to a JSON-safe dict."""
    return {
        "current_valence": profile.current_valence,
        "current_arousal": profile.current_arousal,
        "target_valence": profile.target_valence,
        "target_arousal": profile.target_arousal,
        "emotional_distance": profile.emotional_distance,
        "estimated_steps": profile.estimated_steps,
        "journey_arc": profile.journey_arc,
    }
