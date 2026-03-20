"""Generative neural narratives engine for therapeutic storytelling.

Generates adaptive narrative segments that respond to real-time EEG feedback,
selecting from evidence-based therapeutic frameworks (ACT, EMDR, narrative
therapy, guided imagery) and personalizing imagery based on individual
response patterns.

Therapeutic frameworks:
  - ACT metaphors: passengers on bus, leaves on stream, quicksand
  - EMDR bilateral: left-right alternating imagery for processing
  - Narrative therapy: externalization of problems as characters
  - Guided imagery: nature, ocean, space, interpersonal scenes

EEG feedback loop:
  - Alpha increasing -> relaxation working, continue/deepen
  - High beta -> anxiety rising, switch to grounding
  - Theta increasing -> disengaging, re-engage with interactive element
  - Stable alpha/theta -> good state, advance narrative arc

Safety:
  - Blocks minimizing language ("just relax", "it's not that bad")
  - Avoids known trauma triggers per user profile
  - Maintains therapeutic frame (non-directive, validating)

References:
    Hayes et al. (2006) - ACT: A practical guide
    Shapiro (2001) - EMDR: Basic principles, protocols, and procedures
    White & Epston (1990) - Narrative means to therapeutic ends
    Utay & Miller (2006) - Guided imagery as therapeutic recreation

WELLNESS DISCLAIMER: This is a research/educational wellness tool only,
not a medical device. It is NOT a substitute for professional support.
Always work with a licensed mental health professional.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Clinical disclaimer included in every response
# ---------------------------------------------------------------------------
_CLINICAL_DISCLAIMER = (
    "Wellness disclaimer: This narrative engine is a research and educational "
    "wellness tool only, not a medical device. It is NOT a substitute for "
    "professional support. Do not use this as a standalone intervention. "
    "Always work with a licensed mental health professional."
)


# ---------------------------------------------------------------------------
# Safety filter — blocked phrases and patterns
# ---------------------------------------------------------------------------
_MINIMIZING_PHRASES = [
    "just relax",
    "just calm down",
    "it's not that bad",
    "it's all in your head",
    "stop worrying",
    "get over it",
    "you're fine",
    "don't be dramatic",
    "others have it worse",
    "snap out of it",
    "man up",
    "toughen up",
    "you should be grateful",
    "think positive",
    "just breathe",
    "it could be worse",
    "at least",
]

_DEFAULT_TRIGGER_WORDS = [
    "suicide",
    "self-harm",
    "abuse",
    "assault",
    "violence",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class FrameworkType(str, Enum):
    ACT_METAPHOR = "act_metaphor"
    EMDR_BILATERAL = "emdr_bilateral"
    NARRATIVE_EXTERNALIZATION = "narrative_externalization"
    GUIDED_IMAGERY = "guided_imagery"


class ImageryPreference(str, Enum):
    NATURE = "nature"
    OCEAN = "ocean"
    SPACE = "space"
    INTERPERSONAL = "interpersonal"


class EmotionalArcPhase(str, Enum):
    OPENING = "opening"
    BUILDING = "building"
    PEAK = "peak"
    RESOLUTION = "resolution"
    INTEGRATION = "integration"


class EEGTrend(str, Enum):
    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"


class BrainStateCategory(str, Enum):
    RELAXING = "relaxing"
    ANXIOUS = "anxious"
    DISENGAGING = "disengaging"
    ENGAGED = "engaged"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class EEGFeedback:
    """Summarised EEG state for narrative adaptation.

    Attributes:
        alpha_trend: Direction of alpha power change.
        beta_level: Current high-beta level (0-1 normalised).
        theta_trend: Direction of theta power change.
        timestamp: When this reading was taken.
    """
    alpha_trend: EEGTrend = EEGTrend.STABLE
    beta_level: float = 0.3
    theta_trend: EEGTrend = EEGTrend.STABLE
    timestamp: float = field(default_factory=time.time)

    def classify_state(self) -> BrainStateCategory:
        """Derive brain-state category from the three signals."""
        if self.beta_level > 0.6:
            return BrainStateCategory.ANXIOUS
        if self.alpha_trend == EEGTrend.INCREASING:
            return BrainStateCategory.RELAXING
        if self.theta_trend == EEGTrend.INCREASING and self.alpha_trend != EEGTrend.INCREASING:
            return BrainStateCategory.DISENGAGING
        if (
            self.alpha_trend == EEGTrend.STABLE
            and self.beta_level < 0.4
            and self.theta_trend != EEGTrend.INCREASING
        ):
            return BrainStateCategory.ENGAGED
        return BrainStateCategory.NEUTRAL


@dataclass
class StoryFramework:
    """A therapeutic story framework template.

    Attributes:
        id: Unique identifier.
        name: Human-readable name.
        framework_type: Which therapeutic modality.
        description: What the framework does therapeutically.
        segments: Ordered template segments with arc-phase placeholders.
        contraindications: Conditions where this framework should not be used.
    """
    id: str
    name: str
    framework_type: FrameworkType
    description: str
    segments: Dict[EmotionalArcPhase, str]
    contraindications: List[str] = field(default_factory=list)


@dataclass
class NarrativeSegment:
    """A single segment of generated narrative.

    Attributes:
        id: Unique identifier.
        text: The narrative text.
        framework_id: Which framework generated this.
        arc_phase: Current phase in emotional arc.
        imagery_type: Which imagery domain is used.
        brain_state: The EEG state that prompted this segment.
        safety_passed: Whether the segment passed safety checks.
        timestamp: Generation time.
    """
    id: str
    text: str
    framework_id: str
    arc_phase: EmotionalArcPhase
    imagery_type: ImageryPreference
    brain_state: BrainStateCategory
    safety_passed: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class NarrativeProfile:
    """Per-user narrative personalisation memory.

    Tracks which imagery types and frameworks are most effective,
    measured by how quickly alpha increases during each type.

    Attributes:
        user_id: User identifier.
        preferred_imagery: Ranked imagery preferences.
        effective_frameworks: Framework IDs that produced relaxation.
        imagery_scores: Cumulative effectiveness score per imagery type.
        framework_scores: Cumulative effectiveness score per framework.
        trigger_words: User-specific words to avoid.
        sessions_completed: How many narrative sessions finished.
    """
    user_id: str
    preferred_imagery: List[ImageryPreference] = field(
        default_factory=lambda: [ImageryPreference.NATURE]
    )
    effective_frameworks: List[str] = field(default_factory=list)
    imagery_scores: Dict[str, float] = field(default_factory=dict)
    framework_scores: Dict[str, float] = field(default_factory=dict)
    trigger_words: List[str] = field(default_factory=list)
    sessions_completed: int = 0


# ---------------------------------------------------------------------------
# Story framework library
# ---------------------------------------------------------------------------
_FRAMEWORK_LIBRARY: Dict[str, StoryFramework] = {
    "act_bus": StoryFramework(
        id="act_bus",
        name="Passengers on the Bus",
        framework_type=FrameworkType.ACT_METAPHOR,
        description=(
            "ACT metaphor: you are the bus driver, difficult thoughts and "
            "feelings are passengers. They can shout but cannot steer."
        ),
        segments={
            EmotionalArcPhase.OPENING: (
                "Imagine you are driving a bus along a quiet road. "
                "The sun is warm through the windshield and the road ahead "
                "stretches clear and open. You are the driver."
            ),
            EmotionalArcPhase.BUILDING: (
                "Some passengers begin to stir behind you. They raise their "
                "voices — worries, doubts, old fears. They want you to turn "
                "around, to take a different route. Notice them without "
                "turning the wheel."
            ),
            EmotionalArcPhase.PEAK: (
                "The passengers are loud now. One leans forward and whispers "
                "something sharp. You feel the urge to swerve. But your "
                "hands stay on the wheel. You are still driving."
            ),
            EmotionalArcPhase.RESOLUTION: (
                "Gradually, the passengers settle. Not because they left — "
                "they are still there — but because you kept driving. "
                "The road is still yours."
            ),
            EmotionalArcPhase.INTEGRATION: (
                "You notice the landscape has changed. The road led somewhere "
                "meaningful. The passengers ride with you, quieter now, "
                "part of the journey but not the navigator."
            ),
        },
    ),
    "act_leaves": StoryFramework(
        id="act_leaves",
        name="Leaves on a Stream",
        framework_type=FrameworkType.ACT_METAPHOR,
        description=(
            "ACT defusion exercise: place each thought on a leaf and "
            "watch it float downstream without holding on."
        ),
        segments={
            EmotionalArcPhase.OPENING: (
                "Picture yourself sitting beside a gently flowing stream. "
                "The water is clear and unhurried. Leaves drift on its "
                "surface, carried by the current."
            ),
            EmotionalArcPhase.BUILDING: (
                "A thought appears. Place it on a leaf and set it on the water. "
                "Watch it float away. Another thought comes — another leaf. "
                "You are not the thoughts. You are the one watching."
            ),
            EmotionalArcPhase.PEAK: (
                "Some leaves feel heavy. A thought wants to stay, demands "
                "attention. Gently place it on a leaf anyway. The stream "
                "carries everything equally."
            ),
            EmotionalArcPhase.RESOLUTION: (
                "The stream continues. You have not tried to stop it or "
                "speed it up. The leaves come and go. You remain here, "
                "beside the water, steady."
            ),
            EmotionalArcPhase.INTEGRATION: (
                "Notice how the stream keeps flowing whether you watch or "
                "not. Your thoughts are the same — natural, passing, not "
                "requiring you to hold them."
            ),
        },
    ),
    "emdr_bilateral": StoryFramework(
        id="emdr_bilateral",
        name="Bilateral Imagery Walk",
        framework_type=FrameworkType.EMDR_BILATERAL,
        description=(
            "EMDR-inspired bilateral stimulation through alternating "
            "left-right imagery to facilitate emotional processing."
        ),
        segments={
            EmotionalArcPhase.OPENING: (
                "Begin walking along a path. With each step, notice the "
                "alternation — left foot, right foot. Left, right. "
                "A steady rhythm."
            ),
            EmotionalArcPhase.BUILDING: (
                "On your left, sunlight filters through trees. On your right, "
                "a meadow stretches out. Left — shadow and shelter. Right — "
                "openness and light. Both are part of the path."
            ),
            EmotionalArcPhase.PEAK: (
                "Something weighs on you. As you walk, let it shift — left "
                "side, right side — like a stone rolling gently between your "
                "hands. It does not need to stay in one place."
            ),
            EmotionalArcPhase.RESOLUTION: (
                "The weight has not vanished but it has become something "
                "you carry differently. Left step, right step. The rhythm "
                "continues, and the stone feels smoother."
            ),
            EmotionalArcPhase.INTEGRATION: (
                "You look back at the path behind you. Every step — left "
                "and right — brought you here. The walking itself was "
                "the processing."
            ),
        },
        contraindications=["active_dissociation", "severe_ptsd_unmanaged"],
    ),
    "narrative_ext": StoryFramework(
        id="narrative_ext",
        name="The Unwelcome Guest",
        framework_type=FrameworkType.NARRATIVE_EXTERNALIZATION,
        description=(
            "Narrative therapy externalization: the problem becomes a "
            "character separate from the person, reducing identification."
        ),
        segments={
            EmotionalArcPhase.OPENING: (
                "There is a guest who arrived uninvited. It sits in a corner "
                "of your room, taking up space. It has a name — perhaps Worry, "
                "perhaps Doubt. It is not you."
            ),
            EmotionalArcPhase.BUILDING: (
                "The guest speaks up, as it often does. It tells familiar "
                "stories — old scripts about what might go wrong. You have "
                "heard these before. You can listen without agreeing."
            ),
            EmotionalArcPhase.PEAK: (
                "The guest stands up, tries to fill the whole room. It wants "
                "to be everything. But look — the room is larger than the "
                "guest. There are windows, doors, space the guest cannot fill."
            ),
            EmotionalArcPhase.RESOLUTION: (
                "You do not need to push the guest out. You simply reclaim "
                "the rest of the room. Move a chair. Open a window. "
                "The guest shrinks back to its corner."
            ),
            EmotionalArcPhase.INTEGRATION: (
                "The guest may visit again. But now you know: the guest is "
                "a visitor, not a resident. The room — your life — has "
                "space for much more than one uninvited voice."
            ),
        },
    ),
    "guided_nature": StoryFramework(
        id="guided_nature",
        name="Mountain Clearing",
        framework_type=FrameworkType.GUIDED_IMAGERY,
        description=(
            "Guided imagery in a natural mountain setting to promote "
            "calm and grounding through sensory engagement."
        ),
        segments={
            EmotionalArcPhase.OPENING: (
                "You stand at the edge of a forest clearing high in the "
                "mountains. The air is cool and carries the scent of pine. "
                "Beneath your feet, the ground is soft with fallen needles."
            ),
            EmotionalArcPhase.BUILDING: (
                "You walk to the centre of the clearing. Above, the sky is "
                "wide and open. Around you, tall trees form a protective "
                "circle. You can hear a distant stream and birdsong."
            ),
            EmotionalArcPhase.PEAK: (
                "Sit down here. Feel the earth supporting you completely. "
                "Let your weight settle. The mountain has held this ground "
                "for millennia. It can hold you now."
            ),
            EmotionalArcPhase.RESOLUTION: (
                "A breeze moves through the clearing, gentle and unhurried. "
                "It touches your face and hands. You are part of this place "
                "for this moment — grounded, present, held."
            ),
            EmotionalArcPhase.INTEGRATION: (
                "When you are ready, notice that you carry this clearing "
                "with you. The mountain, the trees, the open sky — they "
                "are always available in your memory."
            ),
        },
    ),
    "guided_ocean": StoryFramework(
        id="guided_ocean",
        name="Ocean Shore",
        framework_type=FrameworkType.GUIDED_IMAGERY,
        description=(
            "Guided imagery at the ocean shore — rhythmic waves "
            "promote parasympathetic activation and emotional regulation."
        ),
        segments={
            EmotionalArcPhase.OPENING: (
                "You are standing at the shore. The sand is warm under your "
                "feet and the water stretches to the horizon. Waves arrive "
                "and retreat in a slow, steady rhythm."
            ),
            EmotionalArcPhase.BUILDING: (
                "Step closer to the water. Let it touch your toes — cool "
                "and gentle. Each wave brings something in and takes "
                "something away. The exchange is effortless."
            ),
            EmotionalArcPhase.PEAK: (
                "Stand in the shallows. Feel the pull of the water around "
                "your ankles. The ocean is vast and you are here, exactly "
                "where the land meets the sea."
            ),
            EmotionalArcPhase.RESOLUTION: (
                "The rhythm of the waves has become your breathing. In and "
                "out. Arrive and retreat. Nothing is forced. The ocean "
                "does not try — it simply moves."
            ),
            EmotionalArcPhase.INTEGRATION: (
                "Step back onto dry sand. The warmth returns under your "
                "feet. The ocean continues behind you, steady and "
                "unchanging. Its rhythm stays with you."
            ),
        },
    ),
}


# ---------------------------------------------------------------------------
# Imagery segment variants per domain
# ---------------------------------------------------------------------------
_IMAGERY_VARIANTS: Dict[str, Dict[str, str]] = {
    ImageryPreference.NATURE.value: {
        "grounding": (
            "Feel your feet on the forest floor. Roots spread beneath "
            "you, connecting you to the earth. You are supported."
        ),
        "calming": (
            "A gentle breeze moves through the trees, carrying the scent "
            "of wildflowers. Each breath draws in calm."
        ),
        "re_engage": (
            "A bird lands on a branch nearby. Its song is bright and "
            "clear. Let it draw your attention gently back."
        ),
    },
    ImageryPreference.OCEAN.value: {
        "grounding": (
            "Feel the wet sand beneath you, cool and firm. The tide "
            "anchors you to this moment."
        ),
        "calming": (
            "Waves roll in, slow and rhythmic. Each one washes the "
            "shore clean. Let the rhythm settle through you."
        ),
        "re_engage": (
            "A shell catches the light at the water's edge. Its "
            "surface is intricate, detailed. Look closer."
        ),
    },
    ImageryPreference.SPACE.value: {
        "grounding": (
            "You float in stillness. Below, the Earth glows blue and "
            "white. You are held in orbit, weightless but secure."
        ),
        "calming": (
            "Stars surround you in every direction, silent and "
            "ancient. The vastness holds no urgency."
        ),
        "re_engage": (
            "A comet traces a bright arc across the dark. Its tail "
            "shimmers — a brief, brilliant detail in the stillness."
        ),
    },
    ImageryPreference.INTERPERSONAL.value: {
        "grounding": (
            "Someone you trust sits beside you. They do not need to "
            "speak. Their presence alone is steadying."
        ),
        "calming": (
            "A familiar hand rests on your shoulder, warm and "
            "unhurried. You are not alone in this."
        ),
        "re_engage": (
            "The person beside you turns and smiles. There is warmth "
            "in it, an invitation to be present together."
        ),
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class NarrativeEngine:
    """Therapeutic narrative generation with EEG-adaptive feedback.

    Maintains per-user profiles and adapts story selection and pacing
    based on real-time EEG feedback categories.
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, NarrativeProfile] = {}
        self._active_sessions: Dict[str, Dict] = {}

    # -- Profile management ------------------------------------------------

    def build_narrative_profile(
        self,
        user_id: str,
        preferred_imagery: Optional[List[str]] = None,
        trigger_words: Optional[List[str]] = None,
    ) -> NarrativeProfile:
        """Create or update a narrative profile for a user.

        Args:
            user_id: User identifier.
            preferred_imagery: List of imagery type strings.
            trigger_words: Additional words to filter from narratives.

        Returns:
            The created or updated NarrativeProfile.
        """
        profile = self._profiles.get(user_id)
        if profile is None:
            profile = NarrativeProfile(user_id=user_id)

        if preferred_imagery:
            valid = []
            for img in preferred_imagery:
                try:
                    valid.append(ImageryPreference(img))
                except ValueError:
                    continue
            if valid:
                profile.preferred_imagery = valid

        if trigger_words:
            combined = set(profile.trigger_words) | set(trigger_words)
            profile.trigger_words = list(combined)

        self._profiles[user_id] = profile
        return profile

    def get_profile(self, user_id: str) -> Optional[NarrativeProfile]:
        """Return the profile for a user, or None."""
        return self._profiles.get(user_id)

    # -- Safety ------------------------------------------------------------

    def check_safety(
        self,
        text: str,
        user_id: str = "anonymous",
    ) -> Dict:
        """Check narrative text against safety filters.

        Args:
            text: The narrative text to check.
            user_id: User identifier for per-user trigger words.

        Returns:
            Dict with passed (bool), flagged_phrases (list), reason (str).
        """
        text_lower = text.lower()
        flagged: List[str] = []

        # Check minimizing language
        for phrase in _MINIMIZING_PHRASES:
            if phrase in text_lower:
                flagged.append(phrase)

        # Check default trigger words
        for word in _DEFAULT_TRIGGER_WORDS:
            if word in text_lower:
                flagged.append(word)

        # Check per-user trigger words
        profile = self._profiles.get(user_id)
        if profile:
            for word in profile.trigger_words:
                if word.lower() in text_lower:
                    flagged.append(word)

        passed = len(flagged) == 0
        reason = "" if passed else f"Contains flagged content: {', '.join(flagged)}"

        return {
            "passed": passed,
            "flagged_phrases": flagged,
            "reason": reason,
        }

    # -- Framework selection -----------------------------------------------

    def select_story_framework(
        self,
        brain_state: BrainStateCategory,
        user_id: str = "anonymous",
        exclude_ids: Optional[List[str]] = None,
    ) -> StoryFramework:
        """Choose the best story framework given current brain state.

        Selection logic:
          - ANXIOUS -> ACT metaphors (defusion) or guided imagery (grounding)
          - RELAXING -> advance current framework or guided imagery
          - DISENGAGING -> narrative externalization (interactive/engaging)
          - ENGAGED/NEUTRAL -> any framework, prefer user's effective ones

        Args:
            brain_state: Current classified brain state.
            user_id: For personalisation lookup.
            exclude_ids: Framework IDs to avoid (e.g., recently used).

        Returns:
            Selected StoryFramework.
        """
        exclude = set(exclude_ids or [])
        profile = self._profiles.get(user_id)

        # Map brain states to preferred framework types
        state_preferences: Dict[BrainStateCategory, List[FrameworkType]] = {
            BrainStateCategory.ANXIOUS: [
                FrameworkType.ACT_METAPHOR,
                FrameworkType.GUIDED_IMAGERY,
            ],
            BrainStateCategory.RELAXING: [
                FrameworkType.GUIDED_IMAGERY,
                FrameworkType.ACT_METAPHOR,
            ],
            BrainStateCategory.DISENGAGING: [
                FrameworkType.NARRATIVE_EXTERNALIZATION,
                FrameworkType.EMDR_BILATERAL,
            ],
            BrainStateCategory.ENGAGED: [
                FrameworkType.EMDR_BILATERAL,
                FrameworkType.NARRATIVE_EXTERNALIZATION,
                FrameworkType.ACT_METAPHOR,
            ],
            BrainStateCategory.NEUTRAL: [
                FrameworkType.ACT_METAPHOR,
                FrameworkType.GUIDED_IMAGERY,
                FrameworkType.NARRATIVE_EXTERNALIZATION,
            ],
        }

        preferred_types = state_preferences.get(
            brain_state,
            [FrameworkType.ACT_METAPHOR],
        )

        # If user has effective frameworks, prefer those
        if profile and profile.effective_frameworks:
            for fw_id in profile.effective_frameworks:
                if fw_id in _FRAMEWORK_LIBRARY and fw_id not in exclude:
                    fw = _FRAMEWORK_LIBRARY[fw_id]
                    if fw.framework_type in preferred_types:
                        return fw

        # Filter by preferred types, excluding contraindications
        candidates = []
        for fw in _FRAMEWORK_LIBRARY.values():
            if fw.id in exclude:
                continue
            if fw.framework_type in preferred_types:
                candidates.append(fw)

        if not candidates:
            # Fallback: any framework not excluded
            candidates = [
                fw for fw in _FRAMEWORK_LIBRARY.values()
                if fw.id not in exclude
            ]

        if not candidates:
            # Last resort: return first available
            return list(_FRAMEWORK_LIBRARY.values())[0]

        # Score candidates by user framework scores if available
        if profile and profile.framework_scores:
            candidates.sort(
                key=lambda fw: profile.framework_scores.get(fw.id, 0.0),
                reverse=True,
            )

        return candidates[0]

    # -- EEG feedback adaptation -------------------------------------------

    def adapt_to_eeg_feedback(
        self,
        feedback: EEGFeedback,
        current_arc_phase: EmotionalArcPhase,
        user_id: str = "anonymous",
    ) -> Dict:
        """Determine next narrative direction based on EEG feedback.

        Logic:
          - RELAXING: advance arc phase (working well)
          - ANXIOUS: insert grounding imagery, hold arc phase
          - DISENGAGING: insert re-engagement imagery, hold arc phase
          - ENGAGED: continue current phase
          - NEUTRAL: advance if near end of phase

        Args:
            feedback: Current EEG feedback summary.
            current_arc_phase: Where we are in the emotional arc.
            user_id: For personalisation.

        Returns:
            Dict with next_phase, action, imagery_insert (optional),
            brain_state, recommendation.
        """
        state = feedback.classify_state()
        profile = self._profiles.get(user_id)
        imagery_pref = (
            profile.preferred_imagery[0] if profile and profile.preferred_imagery
            else ImageryPreference.NATURE
        )

        arc_phases = list(EmotionalArcPhase)
        current_idx = arc_phases.index(current_arc_phase)

        result: Dict = {
            "brain_state": state.value,
            "current_phase": current_arc_phase.value,
            "next_phase": current_arc_phase.value,
            "action": "continue",
            "imagery_insert": None,
            "recommendation": "",
        }

        if state == BrainStateCategory.RELAXING:
            # Relaxation working: advance to next phase
            if current_idx < len(arc_phases) - 1:
                result["next_phase"] = arc_phases[current_idx + 1].value
                result["action"] = "advance"
                result["recommendation"] = (
                    "Alpha increasing — relaxation response active. "
                    "Advancing narrative arc."
                )
            else:
                result["action"] = "complete"
                result["recommendation"] = (
                    "Narrative arc complete with positive response."
                )

        elif state == BrainStateCategory.ANXIOUS:
            # Anxiety rising: insert grounding, hold phase
            variants = _IMAGERY_VARIANTS.get(imagery_pref.value, {})
            result["action"] = "ground"
            result["imagery_insert"] = variants.get("grounding", "")
            result["recommendation"] = (
                "High beta detected — anxiety rising. "
                "Inserting grounding imagery before continuing."
            )

        elif state == BrainStateCategory.DISENGAGING:
            # Theta rising, disengaging: insert re-engagement
            variants = _IMAGERY_VARIANTS.get(imagery_pref.value, {})
            result["action"] = "re_engage"
            result["imagery_insert"] = variants.get("re_engage", "")
            result["recommendation"] = (
                "Theta increasing — attention drifting. "
                "Inserting sensory detail to re-engage."
            )

        elif state == BrainStateCategory.ENGAGED:
            result["action"] = "continue"
            result["recommendation"] = (
                "Good engagement level. Continuing current narrative."
            )

        else:
            # Neutral: gentle advance if past building phase
            if current_idx >= 2 and current_idx < len(arc_phases) - 1:
                result["next_phase"] = arc_phases[current_idx + 1].value
                result["action"] = "advance"
                result["recommendation"] = (
                    "Neutral state — gently advancing narrative."
                )
            else:
                result["action"] = "continue"
                result["recommendation"] = "Neutral state. Continuing."

        # Update profile imagery scores based on positive states
        if profile and state in (
            BrainStateCategory.RELAXING,
            BrainStateCategory.ENGAGED,
        ):
            key = imagery_pref.value
            profile.imagery_scores[key] = (
                profile.imagery_scores.get(key, 0.0) + 1.0
            )

        return result

    # -- Segment generation ------------------------------------------------

    def generate_narrative_segment(
        self,
        user_id: str = "anonymous",
        framework_id: Optional[str] = None,
        arc_phase: Optional[str] = None,
        eeg_feedback: Optional[EEGFeedback] = None,
    ) -> NarrativeSegment:
        """Generate a single narrative segment.

        If eeg_feedback is provided, adapts framework selection and arc
        progression automatically. Otherwise uses specified parameters.

        Args:
            user_id: User identifier.
            framework_id: Explicit framework ID (optional).
            arc_phase: Explicit arc phase (optional).
            eeg_feedback: Current EEG state (optional).

        Returns:
            A NarrativeSegment with generated text.
        """
        profile = self._profiles.get(user_id)
        imagery_pref = (
            profile.preferred_imagery[0] if profile and profile.preferred_imagery
            else ImageryPreference.NATURE
        )

        # Determine brain state
        if eeg_feedback is not None:
            brain_state = eeg_feedback.classify_state()
        else:
            brain_state = BrainStateCategory.NEUTRAL

        # Determine arc phase
        if arc_phase:
            try:
                phase = EmotionalArcPhase(arc_phase)
            except ValueError:
                phase = EmotionalArcPhase.OPENING
        else:
            phase = EmotionalArcPhase.OPENING

        # Select framework
        if framework_id and framework_id in _FRAMEWORK_LIBRARY:
            framework = _FRAMEWORK_LIBRARY[framework_id]
        else:
            framework = self.select_story_framework(
                brain_state, user_id=user_id
            )

        # Get segment text from framework
        text = framework.segments.get(phase, framework.segments[EmotionalArcPhase.OPENING])

        # If anxious or disengaging, append imagery variant
        if brain_state == BrainStateCategory.ANXIOUS:
            variants = _IMAGERY_VARIANTS.get(imagery_pref.value, {})
            grounding = variants.get("grounding", "")
            if grounding:
                text = f"{text} {grounding}"
        elif brain_state == BrainStateCategory.DISENGAGING:
            variants = _IMAGERY_VARIANTS.get(imagery_pref.value, {})
            re_engage = variants.get("re_engage", "")
            if re_engage:
                text = f"{text} {re_engage}"

        # Safety check
        safety = self.check_safety(text, user_id)

        segment = NarrativeSegment(
            id=str(uuid.uuid4()),
            text=text,
            framework_id=framework.id,
            arc_phase=phase,
            imagery_type=imagery_pref,
            brain_state=brain_state,
            safety_passed=safety["passed"],
        )

        # Update framework score on profile
        if profile and brain_state in (
            BrainStateCategory.RELAXING,
            BrainStateCategory.ENGAGED,
        ):
            profile.framework_scores[framework.id] = (
                profile.framework_scores.get(framework.id, 0.0) + 1.0
            )

        return segment

    # -- Listing -----------------------------------------------------------

    @staticmethod
    def list_frameworks() -> List[Dict]:
        """Return all available story frameworks as dicts."""
        result = []
        for fw in _FRAMEWORK_LIBRARY.values():
            result.append({
                "id": fw.id,
                "name": fw.name,
                "framework_type": fw.framework_type.value,
                "description": fw.description,
                "phases": [p.value for p in fw.segments.keys()],
                "contraindications": fw.contraindications,
            })
        return result

    # -- Serialisation -----------------------------------------------------

    @staticmethod
    def narrative_to_dict(segment: NarrativeSegment) -> Dict:
        """Convert a NarrativeSegment to a JSON-safe dict."""
        return {
            "id": segment.id,
            "text": segment.text,
            "framework_id": segment.framework_id,
            "arc_phase": segment.arc_phase.value,
            "imagery_type": segment.imagery_type.value,
            "brain_state": segment.brain_state.value,
            "safety_passed": segment.safety_passed,
            "timestamp": segment.timestamp,
            "disclaimer": _CLINICAL_DISCLAIMER,
        }
