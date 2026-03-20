"""Voice-interactive EEG-aware emotional companion.

Provides an embodied AI therapeutic companion that adapts its conversational
responses based on the user's real-time EEG emotional state. Manages a
conversation state machine (greeting through closing), selects therapeutic
stances based on context, and generates emotion-informed response templates.

Conversation state machine:
- greeting:          initial contact, establish rapport
- check_in:          assess current emotional state via EEG + self-report
- active_listening:  reflect and validate user's experience
- guidance:          offer EEG-informed therapeutic suggestions
- reflection:        help user integrate insights from the session
- closing:           summarize session, affirm progress, say goodbye

Therapeutic stances:
- supportive:        validate feelings, normalize experience
- challenging:       gently question patterns, encourage growth
- psychoeducational: explain what the EEG data reveals about their state
- reflective:        mirror back themes and emotional shifts

EEG-aware adaptation:
- High stress  -> slow pace, simpler language, validate first
- Low arousal  -> gently activate, check for dissociation
- Calm/neutral -> explore deeper, introduce psychoeducation
- Positive     -> reinforce, build on momentum

Session memory tracks: conversation themes, emotional shifts across the
session, intervention effectiveness, and cumulative session count.

CLINICAL DISCLAIMER: This is a research and educational tool only.
It is NOT a substitute for professional therapy, counseling, or clinical
care. For any genuine mental health crisis, contact a licensed
professional or emergency services immediately.

Issue #457.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clinical disclaimer
# ---------------------------------------------------------------------------

_CLINICAL_DISCLAIMER = (
    "Wellness disclaimer: This embodied companion is a wellness and educational "
    "tool only, not a medical device. It is NOT a substitute for professional "
    "support or clinical care. For any genuine mental health crisis, contact a "
    "licensed mental health professional or call the 988 Suicide & Crisis "
    "Lifeline (call or text 988 in the US), or your local equivalent."
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConversationState(str, Enum):
    """Conversation state machine states."""
    GREETING = "greeting"
    CHECK_IN = "check_in"
    ACTIVE_LISTENING = "active_listening"
    GUIDANCE = "guidance"
    REFLECTION = "reflection"
    CLOSING = "closing"


class TherapeuticStance(str, Enum):
    """Therapeutic approach for the current response."""
    SUPPORTIVE = "supportive"
    CHALLENGING = "challenging"
    PSYCHOEDUCATIONAL = "psychoeducational"
    REFLECTIVE = "reflective"


class EmotionalTone(str, Enum):
    """Detected emotional tone from EEG data."""
    STRESSED = "stressed"
    ANXIOUS = "anxious"
    CALM = "calm"
    POSITIVE = "positive"
    LOW_ENERGY = "low_energy"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_STRESS_HIGH = 0.65
_AROUSAL_HIGH = 0.70
_AROUSAL_LOW = 0.25
_VALENCE_POSITIVE = 0.20
_VALENCE_NEGATIVE = -0.20
_FOCUS_LOW = 0.30

# State transition timing (seconds)
_GREETING_MIN_DURATION = 10
_CHECK_IN_MIN_DURATION = 20
_ACTIVE_LISTENING_MIN_DURATION = 30
_GUIDANCE_MIN_DURATION = 30
_REFLECTION_MIN_DURATION = 20

# Session thresholds
_SHIFT_THRESHOLD = 0.20  # valence change to count as an emotional shift
_EFFECTIVENESS_IMPROVEMENT = 0.10  # stress/arousal improvement for "effective"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EEGState:
    """Current EEG-derived emotional state."""
    valence: float = 0.0
    arousal: float = 0.5
    stress_index: float = 0.0
    focus_index: float = 0.5
    anger_index: float = 0.0
    relaxation_index: float = 0.5
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "stress_index": round(self.stress_index, 4),
            "focus_index": round(self.focus_index, 4),
            "anger_index": round(self.anger_index, 4),
            "relaxation_index": round(self.relaxation_index, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class SessionMemory:
    """Tracks conversation themes and emotional shifts during a session."""
    session_id: str = ""
    themes: List[str] = field(default_factory=list)
    emotional_shifts: List[Dict[str, Any]] = field(default_factory=list)
    interventions_tried: List[str] = field(default_factory=list)
    interventions_effective: List[str] = field(default_factory=list)
    eeg_readings: List[EEGState] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    turn_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "themes": self.themes,
            "emotional_shifts": self.emotional_shifts,
            "interventions_tried": self.interventions_tried,
            "interventions_effective": self.interventions_effective,
            "readings_count": len(self.eeg_readings),
            "start_time": self.start_time,
            "turn_count": self.turn_count,
            "duration_seconds": round(time.time() - self.start_time, 1),
        }


@dataclass
class ResponseTemplate:
    """A generated response template with tone and content guidance."""
    text: str
    stance: TherapeuticStance
    tone: EmotionalTone
    conversation_state: ConversationState
    complexity: str  # "simple", "moderate", "detailed"
    pace: str  # "slow", "normal", "engaged"
    follow_up_prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "stance": self.stance.value,
            "tone": self.tone.value,
            "conversation_state": self.conversation_state.value,
            "complexity": self.complexity,
            "pace": self.pace,
            "follow_up_prompt": self.follow_up_prompt,
        }


@dataclass
class CompanionProfile:
    """Complete companion response profile for a given interaction."""
    conversation_state: ConversationState
    therapeutic_stance: TherapeuticStance
    emotional_tone: EmotionalTone
    response_template: ResponseTemplate
    session_summary: Dict[str, Any]
    eeg_adaptation: Dict[str, Any]
    clinical_disclaimer: str = _CLINICAL_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_state": self.conversation_state.value,
            "therapeutic_stance": self.therapeutic_stance.value,
            "emotional_tone": self.emotional_tone.value,
            "response_template": self.response_template.to_dict(),
            "session_summary": self.session_summary,
            "eeg_adaptation": self.eeg_adaptation,
            "clinical_disclaimer": self.clinical_disclaimer,
        }


# ---------------------------------------------------------------------------
# Response template library
# ---------------------------------------------------------------------------

_RESPONSE_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    # Templates keyed by conversation_state -> emotional_tone -> list of templates
    ConversationState.GREETING.value: {
        EmotionalTone.STRESSED.value: [
            "I notice you might be feeling some tension right now. "
            "That is okay -- you are here, and that matters. "
            "Let us take a moment before we begin.",
        ],
        EmotionalTone.CALM.value: [
            "Welcome. You seem settled right now, which is a good "
            "place to start. How are you feeling today?",
        ],
        EmotionalTone.LOW_ENERGY.value: [
            "Hello. I see your energy is a bit low right now. "
            "There is no pressure -- we can go at whatever pace feels right.",
        ],
        EmotionalTone.POSITIVE.value: [
            "Good to see you. You seem to be in a positive space right now. "
            "What would you like to explore today?",
        ],
        EmotionalTone.NEUTRAL.value: [
            "Welcome. I am here and ready to listen whenever you are ready to begin.",
        ],
        EmotionalTone.ANXIOUS.value: [
            "I can see some restlessness in your state right now. "
            "Let us start gently. Take a breath if that feels right.",
        ],
    },
    ConversationState.CHECK_IN.value: {
        EmotionalTone.STRESSED.value: [
            "Your body seems to be carrying some stress. "
            "Can you tell me what is on your mind right now?",
        ],
        EmotionalTone.CALM.value: [
            "You seem relatively at ease. What has been on your mind lately?",
        ],
        EmotionalTone.LOW_ENERGY.value: [
            "How have you been sleeping? Sometimes low energy "
            "tells us something important.",
        ],
        EmotionalTone.POSITIVE.value: [
            "You seem to be doing well right now. What has been going right for you?",
        ],
        EmotionalTone.NEUTRAL.value: [
            "How are you feeling right now, in this moment?",
        ],
        EmotionalTone.ANXIOUS.value: [
            "I notice some heightened activity in your state. "
            "Is there something specific you are worried about?",
        ],
    },
    ConversationState.ACTIVE_LISTENING.value: {
        EmotionalTone.STRESSED.value: [
            "I hear you. That sounds really difficult. "
            "Your feelings about this are completely valid.",
        ],
        EmotionalTone.CALM.value: [
            "Thank you for sharing that. It sounds like you have "
            "been thinking about this carefully.",
        ],
        EmotionalTone.LOW_ENERGY.value: [
            "I appreciate you telling me this, even when your energy is low. "
            "That takes effort, and I see it.",
        ],
        EmotionalTone.POSITIVE.value: [
            "It is wonderful to hear that. Tell me more about "
            "what that experience was like for you.",
        ],
        EmotionalTone.NEUTRAL.value: [
            "I understand. Can you tell me more about how that makes you feel?",
        ],
        EmotionalTone.ANXIOUS.value: [
            "I can see this is bringing up some tension for you. "
            "You are safe here. Take your time.",
        ],
    },
    ConversationState.GUIDANCE.value: {
        EmotionalTone.STRESSED.value: [
            "Given how you are feeling right now, let us try something "
            "simple. Take three slow breaths with me -- inhale for four counts, "
            "exhale for six. Your body knows how to release this tension.",
        ],
        EmotionalTone.CALM.value: [
            "Since you are in a calm space, this might be a good time to "
            "explore what patterns you have noticed in your emotional responses. "
            "Sometimes our calmest moments offer the clearest perspective.",
        ],
        EmotionalTone.LOW_ENERGY.value: [
            "I do not want to push you to do anything that feels like too much. "
            "But sometimes the smallest action -- standing up, stretching, "
            "drinking water -- can shift things just enough.",
        ],
        EmotionalTone.POSITIVE.value: [
            "You are in a good place right now. Let us use this energy to build "
            "a small toolkit: what strategies have helped you get to this state? "
            "Knowing that can help you find your way back during harder moments.",
        ],
        EmotionalTone.NEUTRAL.value: [
            "Sometimes it helps to check in with your body. Where do you "
            "feel tension, if anywhere? Where do you feel ease?",
        ],
        EmotionalTone.ANXIOUS.value: [
            "Let us ground you a little. Can you name five things you can "
            "see right now? This simple exercise helps your nervous system "
            "remember that you are safe in this moment.",
        ],
    },
    ConversationState.REFLECTION.value: {
        EmotionalTone.STRESSED.value: [
            "Even though this session was hard, you showed up and stayed present. "
            "That takes real courage. What is one thing you want to remember from today?",
        ],
        EmotionalTone.CALM.value: [
            "You have moved into a calmer place during our time together. "
            "What shifted for you? That insight is worth holding onto.",
        ],
        EmotionalTone.LOW_ENERGY.value: [
            "You made it through this session, and that counts. "
            "Be gentle with yourself after we finish.",
        ],
        EmotionalTone.POSITIVE.value: [
            "This was a productive session. You brought real openness today. "
            "What is one thing you would like to carry forward?",
        ],
        EmotionalTone.NEUTRAL.value: [
            "As we wrap up, what stood out to you from our conversation today?",
        ],
        EmotionalTone.ANXIOUS.value: [
            "Before we close, let us take a moment to settle. "
            "You did good work today, even if it did not feel easy.",
        ],
    },
    ConversationState.CLOSING.value: {
        EmotionalTone.STRESSED.value: [
            "We are done for today. Remember: you do not have to solve everything "
            "at once. If things feel overwhelming, please reach out to someone you trust.",
        ],
        EmotionalTone.CALM.value: [
            "Thank you for this session. You are leaving in a good place. "
            "Take care of yourself until next time.",
        ],
        EmotionalTone.LOW_ENERGY.value: [
            "Rest well. You did more today than you might realize. "
            "Be kind to yourself tonight.",
        ],
        EmotionalTone.POSITIVE.value: [
            "Great session today. Carry this positive energy forward. "
            "I will be here whenever you need to check in again.",
        ],
        EmotionalTone.NEUTRAL.value: [
            "Thank you for your time today. Take care, and I will be "
            "here when you are ready to come back.",
        ],
        EmotionalTone.ANXIOUS.value: [
            "Before you go: take three slow breaths. You are okay. "
            "And if you need support between sessions, do not hesitate to reach out.",
        ],
    },
}

# Follow-up prompts by stance
_FOLLOW_UP_PROMPTS: Dict[str, str] = {
    TherapeuticStance.SUPPORTIVE.value: "What else would you like to share?",
    TherapeuticStance.CHALLENGING.value: (
        "I wonder -- what would happen if you tried looking at this differently?"
    ),
    TherapeuticStance.PSYCHOEDUCATIONAL.value: (
        "Would you like to understand more about what your brain data is showing?"
    ),
    TherapeuticStance.REFLECTIVE.value: (
        "When you hear yourself say that, what comes up for you?"
    ),
}


# ---------------------------------------------------------------------------
# State transition logic
# ---------------------------------------------------------------------------

_STATE_ORDER = [
    ConversationState.GREETING,
    ConversationState.CHECK_IN,
    ConversationState.ACTIVE_LISTENING,
    ConversationState.GUIDANCE,
    ConversationState.REFLECTION,
    ConversationState.CLOSING,
]

_STATE_MIN_DURATIONS = {
    ConversationState.GREETING: _GREETING_MIN_DURATION,
    ConversationState.CHECK_IN: _CHECK_IN_MIN_DURATION,
    ConversationState.ACTIVE_LISTENING: _ACTIVE_LISTENING_MIN_DURATION,
    ConversationState.GUIDANCE: _GUIDANCE_MIN_DURATION,
    ConversationState.REFLECTION: _REFLECTION_MIN_DURATION,
    ConversationState.CLOSING: 0,  # no minimum for closing
}


# ---------------------------------------------------------------------------
# Core engine functions
# ---------------------------------------------------------------------------


def _detect_emotional_tone(eeg: EEGState) -> EmotionalTone:
    """Classify the EEG state into an emotional tone for response selection.

    Priority order:
    1. Stressed (high stress overrides everything)
    2. Anxious (high arousal + negative valence without high stress)
    3. Low energy (very low arousal)
    4. Positive (positive valence + moderate arousal)
    5. Calm (low stress + moderate arousal)
    6. Neutral (fallback)
    """
    if eeg.stress_index >= _STRESS_HIGH:
        return EmotionalTone.STRESSED
    if eeg.arousal >= _AROUSAL_HIGH and eeg.valence < _VALENCE_NEGATIVE:
        return EmotionalTone.ANXIOUS
    if eeg.arousal <= _AROUSAL_LOW:
        return EmotionalTone.LOW_ENERGY
    if eeg.valence >= _VALENCE_POSITIVE and eeg.arousal >= 0.35:
        return EmotionalTone.POSITIVE
    if eeg.stress_index < 0.35 and 0.30 <= eeg.arousal <= 0.60:
        return EmotionalTone.CALM
    return EmotionalTone.NEUTRAL


def detect_conversation_state(
    current_state: str,
    turn_count: int,
    session_duration_seconds: float,
    user_wants_to_close: bool = False,
) -> Dict[str, Any]:
    """Determine the current or next conversation state.

    The state machine progresses through: greeting -> check_in ->
    active_listening -> guidance -> reflection -> closing. Transitions
    happen based on turn count and time spent in the current state.

    Args:
        current_state: Current conversation state value.
        turn_count: Number of conversation turns so far.
        session_duration_seconds: Total elapsed time in the session.
        user_wants_to_close: Whether the user has requested to end.

    Returns:
        Dict with current_state, next_state, can_advance, and reason.
    """
    try:
        state = ConversationState(current_state)
    except ValueError:
        state = ConversationState.GREETING

    if user_wants_to_close:
        return {
            "current_state": state.value,
            "next_state": ConversationState.CLOSING.value,
            "can_advance": True,
            "reason": "user_requested_close",
        }

    state_idx = _STATE_ORDER.index(state)
    min_duration = _STATE_MIN_DURATIONS.get(state, 0)

    # Already at closing -- stay there
    if state == ConversationState.CLOSING:
        return {
            "current_state": state.value,
            "next_state": ConversationState.CLOSING.value,
            "can_advance": False,
            "reason": "session_complete",
        }

    # Check if enough time/turns have passed to advance
    can_advance = False
    reason = "not_ready"

    # Greeting advances quickly
    if state == ConversationState.GREETING and turn_count >= 1:
        can_advance = True
        reason = "greeting_complete"
    # Other states advance on time + turn count
    elif turn_count >= 2 and session_duration_seconds >= min_duration:
        can_advance = True
        reason = "sufficient_engagement"
    # Force advance if very long time in state
    elif session_duration_seconds >= min_duration * 3:
        can_advance = True
        reason = "time_based_advance"

    next_state = state
    if can_advance and state_idx < len(_STATE_ORDER) - 1:
        next_state = _STATE_ORDER[state_idx + 1]

    return {
        "current_state": state.value,
        "next_state": next_state.value,
        "can_advance": can_advance,
        "reason": reason,
    }


def select_therapeutic_stance(
    eeg: EEGState,
    conversation_state: str,
    turn_count: int,
    themes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Select the appropriate therapeutic stance based on context.

    Rules:
    - Stressed/anxious user -> always supportive first
    - Early in conversation (greeting/check_in) -> supportive or reflective
    - Calm user in guidance -> psychoeducational or challenging
    - Positive user with themes -> reflective to build insight
    - Many turns without progress -> gently challenging

    Args:
        eeg: Current EEG-derived emotional state.
        conversation_state: Current conversation state value.
        turn_count: Conversation turn count.
        themes: Identified conversation themes so far.

    Returns:
        Dict with selected stance, rationale, and alternatives.
    """
    tone = _detect_emotional_tone(eeg)
    themes_list = themes or []

    try:
        state = ConversationState(conversation_state)
    except ValueError:
        state = ConversationState.CHECK_IN

    stance = TherapeuticStance.SUPPORTIVE
    rationale = "Default supportive stance."
    alternatives: List[str] = []

    # Rule 1: stressed or anxious -> supportive always
    if tone in (EmotionalTone.STRESSED, EmotionalTone.ANXIOUS):
        stance = TherapeuticStance.SUPPORTIVE
        rationale = (
            "User shows signs of stress or anxiety. Validate and support first "
            "before introducing any other approach."
        )
        alternatives = [TherapeuticStance.REFLECTIVE.value]

    # Rule 2: early states -> supportive or reflective
    elif state in (ConversationState.GREETING, ConversationState.CHECK_IN):
        stance = TherapeuticStance.SUPPORTIVE
        rationale = "Early in conversation -- building rapport with supportive stance."
        alternatives = [TherapeuticStance.REFLECTIVE.value]

    # Rule 3: calm in guidance -> psychoeducational
    elif tone == EmotionalTone.CALM and state == ConversationState.GUIDANCE:
        stance = TherapeuticStance.PSYCHOEDUCATIONAL
        rationale = (
            "User is calm and in guidance phase -- good time for psychoeducation "
            "about what their brain data reveals."
        )
        alternatives = [
            TherapeuticStance.CHALLENGING.value,
            TherapeuticStance.REFLECTIVE.value,
        ]

    # Rule 4: positive with themes -> reflective
    elif tone == EmotionalTone.POSITIVE and len(themes_list) >= 1:
        stance = TherapeuticStance.REFLECTIVE
        rationale = (
            "User is in a positive state and has identified themes. Reflective "
            "stance to deepen insight and integration."
        )
        alternatives = [
            TherapeuticStance.CHALLENGING.value,
            TherapeuticStance.PSYCHOEDUCATIONAL.value,
        ]

    # Rule 5: many turns, neutral state -> gently challenging
    elif turn_count >= 8 and tone in (EmotionalTone.NEUTRAL, EmotionalTone.CALM):
        stance = TherapeuticStance.CHALLENGING
        rationale = (
            "Multiple turns without strong emotional direction. Gently challenging "
            "to explore new perspectives."
        )
        alternatives = [
            TherapeuticStance.REFLECTIVE.value,
            TherapeuticStance.PSYCHOEDUCATIONAL.value,
        ]

    # Rule 6: low energy -> supportive with gentle activation
    elif tone == EmotionalTone.LOW_ENERGY:
        stance = TherapeuticStance.SUPPORTIVE
        rationale = (
            "User shows low energy. Supportive stance with gentle encouragement "
            "rather than pushing."
        )
        alternatives = [TherapeuticStance.REFLECTIVE.value]

    # Default: reflective
    else:
        stance = TherapeuticStance.REFLECTIVE
        rationale = "Default reflective stance for mid-conversation engagement."
        alternatives = [
            TherapeuticStance.SUPPORTIVE.value,
            TherapeuticStance.PSYCHOEDUCATIONAL.value,
        ]

    return {
        "stance": stance.value,
        "rationale": rationale,
        "alternatives": alternatives,
        "emotional_tone": tone.value,
        "conversation_state": state.value,
    }


def adapt_response_to_eeg(eeg: EEGState) -> Dict[str, Any]:
    """Determine how to adapt response delivery based on EEG state.

    Controls three dimensions:
    - complexity: how detailed the language should be
    - pace: how fast/slow the interaction should move
    - primary_action: what the companion should do first

    Args:
        eeg: Current EEG-derived emotional state.

    Returns:
        Dict with adaptation parameters: complexity, pace, primary_action,
        tone, and reasoning.
    """
    tone = _detect_emotional_tone(eeg)

    if tone == EmotionalTone.STRESSED:
        return {
            "tone": tone.value,
            "complexity": "simple",
            "pace": "slow",
            "primary_action": "validate",
            "reasoning": (
                "High stress detected. Use simple language, slow pace, "
                "and validate the user's experience before anything else."
            ),
            "eeg_state": eeg.to_dict(),
        }

    if tone == EmotionalTone.ANXIOUS:
        return {
            "tone": tone.value,
            "complexity": "simple",
            "pace": "slow",
            "primary_action": "ground",
            "reasoning": (
                "Anxiety markers detected. Use grounding techniques, "
                "simple sentences, and a measured pace."
            ),
            "eeg_state": eeg.to_dict(),
        }

    if tone == EmotionalTone.LOW_ENERGY:
        return {
            "tone": tone.value,
            "complexity": "simple",
            "pace": "slow",
            "primary_action": "gently_activate",
            "reasoning": (
                "Low energy/arousal detected. Do not overwhelm. Use brief, "
                "warm statements and gentle invitations to engage."
            ),
            "eeg_state": eeg.to_dict(),
        }

    if tone == EmotionalTone.CALM:
        return {
            "tone": tone.value,
            "complexity": "moderate",
            "pace": "normal",
            "primary_action": "explore",
            "reasoning": (
                "User is calm and receptive. Good opportunity for "
                "moderate-depth exploration and psychoeducation."
            ),
            "eeg_state": eeg.to_dict(),
        }

    if tone == EmotionalTone.POSITIVE:
        return {
            "tone": tone.value,
            "complexity": "detailed",
            "pace": "engaged",
            "primary_action": "build",
            "reasoning": (
                "Positive emotional state. User is receptive to deeper "
                "exploration and building on current momentum."
            ),
            "eeg_state": eeg.to_dict(),
        }

    # Neutral
    return {
        "tone": tone.value,
        "complexity": "moderate",
        "pace": "normal",
        "primary_action": "inquire",
        "reasoning": (
            "Neutral emotional state. Use open-ended questions to "
            "understand what the user needs."
        ),
        "eeg_state": eeg.to_dict(),
    }


def generate_response_template(
    eeg: EEGState,
    conversation_state: str,
    stance: str,
) -> Dict[str, Any]:
    """Generate an emotion-informed response template.

    Selects an appropriate template from the library based on the
    conversation state and emotional tone, then annotates it with
    delivery guidance (complexity, pace, follow-up prompt).

    Args:
        eeg: Current EEG-derived emotional state.
        conversation_state: Current conversation state value.
        stance: Selected therapeutic stance value.

    Returns:
        Dict with the response template and delivery metadata.
    """
    tone = _detect_emotional_tone(eeg)
    adaptation = adapt_response_to_eeg(eeg)

    try:
        state = ConversationState(conversation_state)
    except ValueError:
        state = ConversationState.CHECK_IN

    try:
        therapeutic_stance = TherapeuticStance(stance)
    except ValueError:
        therapeutic_stance = TherapeuticStance.SUPPORTIVE

    # Look up template
    state_templates = _RESPONSE_TEMPLATES.get(state.value, {})
    tone_templates = state_templates.get(tone.value, [])

    if not tone_templates:
        # Fallback to neutral templates for the state
        tone_templates = state_templates.get(EmotionalTone.NEUTRAL.value, [])

    if not tone_templates:
        text = "I am here with you. Tell me what is on your mind."
    else:
        text = tone_templates[0]

    follow_up = _FOLLOW_UP_PROMPTS.get(therapeutic_stance.value)

    template = ResponseTemplate(
        text=text,
        stance=therapeutic_stance,
        tone=tone,
        conversation_state=state,
        complexity=adaptation["complexity"],
        pace=adaptation["pace"],
        follow_up_prompt=follow_up,
    )

    return {
        "template": template.to_dict(),
        "adaptation": adaptation,
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


def track_session(
    session_memory: Dict[str, Any],
    eeg: EEGState,
    theme: Optional[str] = None,
    intervention: Optional[str] = None,
) -> Dict[str, Any]:
    """Track session data: emotional shifts, themes, and intervention effectiveness.

    Updates the session memory with the current EEG reading, detects
    emotional shifts compared to the previous reading, and records
    any themes or interventions.

    Args:
        session_memory: Current session memory dict (from previous track_session
            call or empty dict for first call).
        eeg: Current EEG state.
        theme: Optional conversation theme to record.
        intervention: Optional intervention name being tried.

    Returns:
        Updated session memory dict with new data incorporated.
    """
    # Initialize memory structure if needed
    if "themes" not in session_memory:
        session_memory = {
            "session_id": session_memory.get("session_id", ""),
            "themes": [],
            "emotional_shifts": [],
            "interventions_tried": [],
            "interventions_effective": [],
            "readings": [],
            "start_time": session_memory.get("start_time", time.time()),
            "turn_count": session_memory.get("turn_count", 0),
        }

    # Record EEG reading
    reading_dict = eeg.to_dict()
    readings = session_memory.get("readings", [])
    readings.append(reading_dict)
    session_memory["readings"] = readings

    # Increment turn count
    session_memory["turn_count"] = session_memory.get("turn_count", 0) + 1

    # Record theme
    if theme and theme not in session_memory["themes"]:
        session_memory["themes"].append(theme)

    # Record intervention
    if intervention:
        if intervention not in session_memory["interventions_tried"]:
            session_memory["interventions_tried"].append(intervention)

        # Check effectiveness: compare current stress/arousal to pre-intervention
        if len(readings) >= 2:
            prev = readings[-2]
            stress_improved = (
                prev.get("stress_index", 0) - eeg.stress_index
                >= _EFFECTIVENESS_IMPROVEMENT
            )
            arousal_improved = (
                prev.get("arousal", 0) - eeg.arousal >= _EFFECTIVENESS_IMPROVEMENT
                if eeg.arousal > _AROUSAL_HIGH
                else False
            )
            if stress_improved or arousal_improved:
                if intervention not in session_memory["interventions_effective"]:
                    session_memory["interventions_effective"].append(intervention)

    # Detect emotional shift
    if len(readings) >= 2:
        prev = readings[-2]
        valence_change = eeg.valence - prev.get("valence", 0.0)
        if abs(valence_change) >= _SHIFT_THRESHOLD:
            shift = {
                "from_valence": round(prev.get("valence", 0.0), 4),
                "to_valence": round(eeg.valence, 4),
                "change": round(valence_change, 4),
                "direction": "positive" if valence_change > 0 else "negative",
                "timestamp": eeg.timestamp,
            }
            session_memory["emotional_shifts"].append(shift)

    # Compute session summary
    duration = time.time() - session_memory.get("start_time", time.time())
    session_memory["duration_seconds"] = round(duration, 1)

    return session_memory


def compute_companion_profile(
    eeg: EEGState,
    conversation_state: str = "greeting",
    turn_count: int = 0,
    session_duration: float = 0.0,
    themes: Optional[List[str]] = None,
    session_memory: Optional[Dict[str, Any]] = None,
) -> CompanionProfile:
    """Compute a complete companion response profile.

    Combines conversation state detection, therapeutic stance selection,
    EEG adaptation, and response template generation into a single
    coherent profile for the companion to use.

    Args:
        eeg: Current EEG-derived emotional state.
        conversation_state: Current conversation state value.
        turn_count: Number of conversation turns.
        session_duration: Elapsed session time in seconds.
        themes: Identified conversation themes.
        session_memory: Current session memory dict.

    Returns:
        CompanionProfile with all response parameters.
    """
    # Detect conversation state transition
    state_result = detect_conversation_state(
        current_state=conversation_state,
        turn_count=turn_count,
        session_duration_seconds=session_duration,
    )
    active_state = ConversationState(state_result["next_state"])

    # Select therapeutic stance
    stance_result = select_therapeutic_stance(
        eeg=eeg,
        conversation_state=active_state.value,
        turn_count=turn_count,
        themes=themes,
    )
    stance = TherapeuticStance(stance_result["stance"])

    # Detect emotional tone
    tone = _detect_emotional_tone(eeg)

    # Get EEG adaptation
    adaptation = adapt_response_to_eeg(eeg)

    # Generate response template
    template_result = generate_response_template(
        eeg=eeg,
        conversation_state=active_state.value,
        stance=stance.value,
    )
    template_data = template_result["template"]
    response_template = ResponseTemplate(
        text=template_data["text"],
        stance=stance,
        tone=tone,
        conversation_state=active_state,
        complexity=template_data["complexity"],
        pace=template_data["pace"],
        follow_up_prompt=template_data.get("follow_up_prompt"),
    )

    # Build session summary
    memory = session_memory or {}
    session_summary = {
        "turn_count": turn_count,
        "duration_seconds": round(session_duration, 1),
        "themes": memory.get("themes", themes or []),
        "emotional_shifts_count": len(memory.get("emotional_shifts", [])),
        "interventions_tried": memory.get("interventions_tried", []),
        "interventions_effective": memory.get("interventions_effective", []),
    }

    return CompanionProfile(
        conversation_state=active_state,
        therapeutic_stance=stance,
        emotional_tone=tone,
        response_template=response_template,
        session_summary=session_summary,
        eeg_adaptation=adaptation,
    )


def profile_to_dict(profile: CompanionProfile) -> Dict[str, Any]:
    """Convert a CompanionProfile to a serializable dict.

    Args:
        profile: CompanionProfile instance.

    Returns:
        Dict representation of the profile.
    """
    return profile.to_dict()
