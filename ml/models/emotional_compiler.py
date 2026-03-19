"""Emotional Compiler — framework-agnostic translation between therapy modalities.

Translates emotional states described in one therapeutic framework into
equivalent descriptions in all others. Supports bidirectional mapping across
five evidence-based modalities:

  - CBT  (Cognitive Behavioral Therapy — cognitive distortions)
  - DBT  (Dialectical Behavior Therapy — emotion regulation skills)
  - ACT  (Acceptance and Commitment Therapy — psychological flexibility)
  - Somatic (body-based / sensorimotor approaches)
  - IFS  (Internal Family Systems — parts work)

Each framework carries its own emotion ontology, vocabulary, and intervention
strategy. The compiler maintains a bidirectional mapping table and can
produce cross-framework compilations, vocabulary lookups, and per-framework
intervention suggestions.

GitHub issue: #454
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ── Supported frameworks ─────────────────────────────────────────────────────

FRAMEWORKS: List[str] = ["CBT", "DBT", "ACT", "Somatic", "IFS"]

FRAMEWORK_DESCRIPTIONS: Dict[str, str] = {
    "CBT": (
        "Cognitive Behavioral Therapy — identifies and restructures cognitive "
        "distortions (automatic negative thoughts) that drive emotional distress."
    ),
    "DBT": (
        "Dialectical Behavior Therapy — teaches distress tolerance, emotion "
        "regulation, interpersonal effectiveness, and mindfulness skills."
    ),
    "ACT": (
        "Acceptance and Commitment Therapy — cultivates psychological "
        "flexibility through acceptance, defusion, present-moment awareness, "
        "and values-based action."
    ),
    "Somatic": (
        "Somatic / Sensorimotor approaches — works with body sensations, "
        "posture, breath, and movement to process and release held emotions."
    ),
    "IFS": (
        "Internal Family Systems — maps the psyche as a system of parts "
        "(protectors, exiles, firefighters) led by the core Self."
    ),
}


# ── Emotion ontologies per framework ────────────────────────────────────────
# Each framework names and categorizes emotional phenomena differently.

FRAMEWORK_VOCABULARY: Dict[str, Dict[str, str]] = {
    "CBT": {
        "catastrophizing": "Assuming the worst-case outcome is certain.",
        "black_and_white_thinking": "Seeing situations in absolute terms — all good or all bad.",
        "mind_reading": "Assuming you know what others are thinking, usually negative.",
        "emotional_reasoning": "Treating feelings as facts: 'I feel it, so it must be true.'",
        "should_statements": "Rigid rules about how you or others 'should' behave.",
        "personalization": "Blaming yourself for events outside your control.",
        "overgeneralization": "Drawing broad conclusions from a single negative event.",
        "magnification": "Exaggerating the importance of negative events or mistakes.",
        "disqualifying_the_positive": "Dismissing positive experiences as flukes.",
        "labeling": "Attaching a fixed, global label to yourself or others based on one event.",
    },
    "DBT": {
        "emotion_mind": "Decisions and perceptions driven entirely by current emotional state.",
        "reasonable_mind": "Decisions driven purely by logic, ignoring emotional data.",
        "wise_mind": "Integration of emotion mind and reasonable mind — intuitive knowing.",
        "distress_intolerance": "Inability to endure painful emotions without impulsive action.",
        "emotional_vulnerability": "Heightened sensitivity to emotional stimuli.",
        "emotional_dysregulation": "Difficulty modulating the intensity or duration of emotions.",
        "interpersonal_ineffectiveness": "Struggles maintaining relationships under emotional stress.",
        "radical_acceptance": "Fully acknowledging reality as it is without judgment or resistance.",
        "opposite_action": "Acting contrary to the emotion urge when the emotion is unjustified.",
        "accumulating_positives": "Intentionally building positive experiences to counterbalance distress.",
    },
    "ACT": {
        "cognitive_fusion": "Being entangled with thoughts, treating them as literal truths.",
        "experiential_avoidance": "Attempting to suppress or escape unwanted internal experiences.",
        "psychological_inflexibility": "Rigid attachment to thoughts, avoidance, and disconnection from values.",
        "defusion": "Stepping back from thoughts, observing them without being controlled by them.",
        "present_moment_awareness": "Full contact with the here and now, without judgment.",
        "values_clarity": "Knowing what truly matters and aligning behavior accordingly.",
        "committed_action": "Taking concrete steps toward values even in the presence of discomfort.",
        "self_as_context": "Experiencing yourself as the observer of thoughts rather than their content.",
        "acceptance": "Willingness to have unwanted experiences without fighting or fleeing.",
        "contact_with_present": "Flexible attention to what is happening right now.",
    },
    "Somatic": {
        "chest_tightness": "Constriction in the chest area, often linked to anxiety or grief.",
        "shallow_breathing": "Rapid, upper-chest breathing indicating sympathetic activation.",
        "gut_sinking": "Dropping sensation in the abdomen, common with fear or dread.",
        "jaw_clenching": "Tension in the masseter, linked to anger or suppressed expression.",
        "shoulder_tension": "Elevated, rigid shoulders — chronic stress or guarding pattern.",
        "throat_constriction": "Tightness in the throat, often with suppressed emotions or tears.",
        "heat_in_face": "Flushing or warmth in the face — anger, shame, or embarrassment.",
        "numbness": "Absence of body sensation — dissociation or freeze response.",
        "trembling": "Involuntary shaking — discharge of fight/flight energy.",
        "heaviness_in_limbs": "Feeling of weight in arms/legs — depression or exhaustion.",
    },
    "IFS": {
        "protector_activated": "A protective part has taken over to prevent pain or vulnerability.",
        "exile_triggered": "A wounded inner part carrying pain from the past has been activated.",
        "firefighter_response": "An emergency part using impulsive behavior to numb exile pain.",
        "self_energy": "Access to the core Self — calm, curious, compassionate, connected.",
        "blended_state": "A part has merged with the Self, so the person IS the part's feelings.",
        "polarization": "Two parts in conflict, each pulling behavior in opposite directions.",
        "burdened_part": "A part carrying extreme beliefs or emotions from past wounds.",
        "unburdened_part": "A part that has released its extreme role after therapeutic work.",
        "manager_part": "A proactive protector that controls, plans, or criticizes to prevent pain.",
        "trailhead": "The initial sensation, thought, or trigger that leads to a part.",
    },
}


# ── Cross-framework mapping table ───────────────────────────────────────────
# Each entry maps a concept from one framework to its closest equivalents in
# all other frameworks. The key is (source_framework, concept_key).

_MAPPING_TABLE: Dict[str, Dict[str, str]] = {
    # CBT catastrophizing
    "CBT:catastrophizing": {
        "CBT": "catastrophizing",
        "DBT": "emotion_mind",
        "ACT": "cognitive_fusion",
        "Somatic": "chest_tightness",
        "IFS": "protector_activated",
    },
    "CBT:black_and_white_thinking": {
        "CBT": "black_and_white_thinking",
        "DBT": "emotion_mind",
        "ACT": "psychological_inflexibility",
        "Somatic": "jaw_clenching",
        "IFS": "polarization",
    },
    "CBT:mind_reading": {
        "CBT": "mind_reading",
        "DBT": "emotional_vulnerability",
        "ACT": "cognitive_fusion",
        "Somatic": "gut_sinking",
        "IFS": "manager_part",
    },
    "CBT:emotional_reasoning": {
        "CBT": "emotional_reasoning",
        "DBT": "emotion_mind",
        "ACT": "cognitive_fusion",
        "Somatic": "heat_in_face",
        "IFS": "blended_state",
    },
    "CBT:should_statements": {
        "CBT": "should_statements",
        "DBT": "emotional_dysregulation",
        "ACT": "psychological_inflexibility",
        "Somatic": "shoulder_tension",
        "IFS": "manager_part",
    },
    "CBT:personalization": {
        "CBT": "personalization",
        "DBT": "emotional_vulnerability",
        "ACT": "cognitive_fusion",
        "Somatic": "gut_sinking",
        "IFS": "exile_triggered",
    },
    "CBT:overgeneralization": {
        "CBT": "overgeneralization",
        "DBT": "emotion_mind",
        "ACT": "cognitive_fusion",
        "Somatic": "heaviness_in_limbs",
        "IFS": "burdened_part",
    },
    "CBT:magnification": {
        "CBT": "magnification",
        "DBT": "emotional_dysregulation",
        "ACT": "cognitive_fusion",
        "Somatic": "shallow_breathing",
        "IFS": "protector_activated",
    },
    "CBT:disqualifying_the_positive": {
        "CBT": "disqualifying_the_positive",
        "DBT": "emotional_vulnerability",
        "ACT": "experiential_avoidance",
        "Somatic": "numbness",
        "IFS": "exile_triggered",
    },
    "CBT:labeling": {
        "CBT": "labeling",
        "DBT": "emotion_mind",
        "ACT": "cognitive_fusion",
        "Somatic": "throat_constriction",
        "IFS": "burdened_part",
    },
    # DBT concepts
    "DBT:emotion_mind": {
        "CBT": "emotional_reasoning",
        "DBT": "emotion_mind",
        "ACT": "cognitive_fusion",
        "Somatic": "shallow_breathing",
        "IFS": "blended_state",
    },
    "DBT:reasonable_mind": {
        "CBT": "disqualifying_the_positive",
        "DBT": "reasonable_mind",
        "ACT": "experiential_avoidance",
        "Somatic": "numbness",
        "IFS": "manager_part",
    },
    "DBT:wise_mind": {
        "CBT": "disqualifying_the_positive",  # inverse — wise mind is the absence of distortion
        "DBT": "wise_mind",
        "ACT": "present_moment_awareness",
        "Somatic": "chest_tightness",  # relaxed chest = wise mind access
        "IFS": "self_energy",
    },
    "DBT:distress_intolerance": {
        "CBT": "catastrophizing",
        "DBT": "distress_intolerance",
        "ACT": "experiential_avoidance",
        "Somatic": "trembling",
        "IFS": "firefighter_response",
    },
    "DBT:emotional_vulnerability": {
        "CBT": "personalization",
        "DBT": "emotional_vulnerability",
        "ACT": "experiential_avoidance",
        "Somatic": "gut_sinking",
        "IFS": "exile_triggered",
    },
    "DBT:emotional_dysregulation": {
        "CBT": "magnification",
        "DBT": "emotional_dysregulation",
        "ACT": "psychological_inflexibility",
        "Somatic": "heat_in_face",
        "IFS": "protector_activated",
    },
    "DBT:interpersonal_ineffectiveness": {
        "CBT": "mind_reading",
        "DBT": "interpersonal_ineffectiveness",
        "ACT": "experiential_avoidance",
        "Somatic": "throat_constriction",
        "IFS": "manager_part",
    },
    "DBT:radical_acceptance": {
        "CBT": "disqualifying_the_positive",  # inverse
        "DBT": "radical_acceptance",
        "ACT": "acceptance",
        "Somatic": "heaviness_in_limbs",  # release of tension
        "IFS": "self_energy",
    },
    "DBT:opposite_action": {
        "CBT": "emotional_reasoning",  # counters emotional reasoning
        "DBT": "opposite_action",
        "ACT": "committed_action",
        "Somatic": "trembling",  # body completing the action cycle
        "IFS": "unburdened_part",
    },
    "DBT:accumulating_positives": {
        "CBT": "disqualifying_the_positive",  # inverse
        "DBT": "accumulating_positives",
        "ACT": "values_clarity",
        "Somatic": "chest_tightness",  # open chest = positive accumulation
        "IFS": "self_energy",
    },
    # ACT concepts
    "ACT:cognitive_fusion": {
        "CBT": "emotional_reasoning",
        "DBT": "emotion_mind",
        "ACT": "cognitive_fusion",
        "Somatic": "chest_tightness",
        "IFS": "blended_state",
    },
    "ACT:experiential_avoidance": {
        "CBT": "disqualifying_the_positive",
        "DBT": "distress_intolerance",
        "ACT": "experiential_avoidance",
        "Somatic": "numbness",
        "IFS": "protector_activated",
    },
    "ACT:psychological_inflexibility": {
        "CBT": "should_statements",
        "DBT": "emotional_dysregulation",
        "ACT": "psychological_inflexibility",
        "Somatic": "jaw_clenching",
        "IFS": "polarization",
    },
    "ACT:defusion": {
        "CBT": "labeling",  # inverse — defusion undoes labeling
        "DBT": "wise_mind",
        "ACT": "defusion",
        "Somatic": "shallow_breathing",  # slowed breath = defused
        "IFS": "self_energy",
    },
    "ACT:present_moment_awareness": {
        "CBT": "emotional_reasoning",  # inverse
        "DBT": "wise_mind",
        "ACT": "present_moment_awareness",
        "Somatic": "gut_sinking",  # grounded in body
        "IFS": "self_energy",
    },
    "ACT:values_clarity": {
        "CBT": "should_statements",  # inverse — values vs shoulds
        "DBT": "accumulating_positives",
        "ACT": "values_clarity",
        "Somatic": "chest_tightness",  # open, expansive
        "IFS": "self_energy",
    },
    "ACT:committed_action": {
        "CBT": "overgeneralization",  # inverse
        "DBT": "opposite_action",
        "ACT": "committed_action",
        "Somatic": "heaviness_in_limbs",  # moving despite heaviness
        "IFS": "unburdened_part",
    },
    "ACT:self_as_context": {
        "CBT": "labeling",  # inverse
        "DBT": "wise_mind",
        "ACT": "self_as_context",
        "Somatic": "numbness",  # not numb — observing body
        "IFS": "self_energy",
    },
    "ACT:acceptance": {
        "CBT": "magnification",  # inverse
        "DBT": "radical_acceptance",
        "ACT": "acceptance",
        "Somatic": "shoulder_tension",  # dropping the shoulders
        "IFS": "self_energy",
    },
    "ACT:contact_with_present": {
        "CBT": "catastrophizing",  # inverse — not in future
        "DBT": "wise_mind",
        "ACT": "contact_with_present",
        "Somatic": "shallow_breathing",  # slow, grounded breath
        "IFS": "self_energy",
    },
    # Somatic concepts
    "Somatic:chest_tightness": {
        "CBT": "catastrophizing",
        "DBT": "distress_intolerance",
        "ACT": "experiential_avoidance",
        "Somatic": "chest_tightness",
        "IFS": "protector_activated",
    },
    "Somatic:shallow_breathing": {
        "CBT": "catastrophizing",
        "DBT": "emotion_mind",
        "ACT": "cognitive_fusion",
        "Somatic": "shallow_breathing",
        "IFS": "blended_state",
    },
    "Somatic:gut_sinking": {
        "CBT": "mind_reading",
        "DBT": "emotional_vulnerability",
        "ACT": "experiential_avoidance",
        "Somatic": "gut_sinking",
        "IFS": "exile_triggered",
    },
    "Somatic:jaw_clenching": {
        "CBT": "should_statements",
        "DBT": "emotional_dysregulation",
        "ACT": "psychological_inflexibility",
        "Somatic": "jaw_clenching",
        "IFS": "protector_activated",
    },
    "Somatic:shoulder_tension": {
        "CBT": "should_statements",
        "DBT": "emotional_dysregulation",
        "ACT": "psychological_inflexibility",
        "Somatic": "shoulder_tension",
        "IFS": "manager_part",
    },
    "Somatic:throat_constriction": {
        "CBT": "labeling",
        "DBT": "interpersonal_ineffectiveness",
        "ACT": "experiential_avoidance",
        "Somatic": "throat_constriction",
        "IFS": "exile_triggered",
    },
    "Somatic:heat_in_face": {
        "CBT": "emotional_reasoning",
        "DBT": "emotional_dysregulation",
        "ACT": "cognitive_fusion",
        "Somatic": "heat_in_face",
        "IFS": "firefighter_response",
    },
    "Somatic:numbness": {
        "CBT": "disqualifying_the_positive",
        "DBT": "reasonable_mind",
        "ACT": "experiential_avoidance",
        "Somatic": "numbness",
        "IFS": "protector_activated",
    },
    "Somatic:trembling": {
        "CBT": "catastrophizing",
        "DBT": "distress_intolerance",
        "ACT": "experiential_avoidance",
        "Somatic": "trembling",
        "IFS": "firefighter_response",
    },
    "Somatic:heaviness_in_limbs": {
        "CBT": "overgeneralization",
        "DBT": "emotional_vulnerability",
        "ACT": "experiential_avoidance",
        "Somatic": "heaviness_in_limbs",
        "IFS": "burdened_part",
    },
    # IFS concepts
    "IFS:protector_activated": {
        "CBT": "catastrophizing",
        "DBT": "emotional_dysregulation",
        "ACT": "experiential_avoidance",
        "Somatic": "chest_tightness",
        "IFS": "protector_activated",
    },
    "IFS:exile_triggered": {
        "CBT": "personalization",
        "DBT": "emotional_vulnerability",
        "ACT": "experiential_avoidance",
        "Somatic": "gut_sinking",
        "IFS": "exile_triggered",
    },
    "IFS:firefighter_response": {
        "CBT": "emotional_reasoning",
        "DBT": "distress_intolerance",
        "ACT": "experiential_avoidance",
        "Somatic": "heat_in_face",
        "IFS": "firefighter_response",
    },
    "IFS:self_energy": {
        "CBT": "disqualifying_the_positive",  # inverse
        "DBT": "wise_mind",
        "ACT": "self_as_context",
        "Somatic": "chest_tightness",  # open chest, grounded
        "IFS": "self_energy",
    },
    "IFS:blended_state": {
        "CBT": "emotional_reasoning",
        "DBT": "emotion_mind",
        "ACT": "cognitive_fusion",
        "Somatic": "shallow_breathing",
        "IFS": "blended_state",
    },
    "IFS:polarization": {
        "CBT": "black_and_white_thinking",
        "DBT": "emotional_dysregulation",
        "ACT": "psychological_inflexibility",
        "Somatic": "jaw_clenching",
        "IFS": "polarization",
    },
    "IFS:burdened_part": {
        "CBT": "overgeneralization",
        "DBT": "emotional_vulnerability",
        "ACT": "cognitive_fusion",
        "Somatic": "heaviness_in_limbs",
        "IFS": "burdened_part",
    },
    "IFS:unburdened_part": {
        "CBT": "disqualifying_the_positive",  # inverse
        "DBT": "opposite_action",
        "ACT": "committed_action",
        "Somatic": "trembling",  # release/discharge
        "IFS": "unburdened_part",
    },
    "IFS:manager_part": {
        "CBT": "should_statements",
        "DBT": "interpersonal_ineffectiveness",
        "ACT": "psychological_inflexibility",
        "Somatic": "shoulder_tension",
        "IFS": "manager_part",
    },
    "IFS:trailhead": {
        "CBT": "magnification",
        "DBT": "emotional_vulnerability",
        "ACT": "contact_with_present",
        "Somatic": "gut_sinking",
        "IFS": "trailhead",
    },
}


# ── Per-framework intervention suggestions ──────────────────────────────────

INTERVENTIONS: Dict[str, Dict[str, str]] = {
    "CBT": {
        "catastrophizing": "Thought record: list evidence for and against the catastrophe. Calculate realistic probability.",
        "black_and_white_thinking": "Find the grey: rate the situation on a 0-100 scale instead of good/bad.",
        "mind_reading": "Behavioral experiment: ask the person what they actually think. Compare to assumption.",
        "emotional_reasoning": "Label the feeling separately from the fact. 'I feel anxious' != 'I am in danger.'",
        "should_statements": "Replace 'should' with 'I would prefer.' Notice the difference in emotional intensity.",
        "personalization": "Responsibility pie chart: list all contributing factors, assign percentages.",
        "overgeneralization": "Find three counter-examples from your own experience.",
        "magnification": "Decatastrophize: what is the actual worst outcome, best outcome, and most likely outcome?",
        "disqualifying_the_positive": "Keep a daily positives log. Write down 3 things that went well, no matter how small.",
        "labeling": "Separate behavior from identity. 'I made a mistake' != 'I am a failure.'",
    },
    "DBT": {
        "emotion_mind": "Use the STOP skill: Stop, Take a step back, Observe, Proceed mindfully.",
        "reasonable_mind": "Check the facts: is there emotional data you are ignoring? What does your gut say?",
        "wise_mind": "Practice the wise mind meditation: breathe and ask 'What does my inner wisdom say?'",
        "distress_intolerance": "TIPP skills: Temperature (ice water), Intense exercise, Paced breathing, Progressive relaxation.",
        "emotional_vulnerability": "PLEASE skills: treat PhysicaL illness, balance Eating, avoid mood-Altering substances, balance Sleep, get Exercise.",
        "emotional_dysregulation": "Opposite action: identify the emotion urge, then do the opposite if the emotion is unjustified.",
        "interpersonal_ineffectiveness": "DEAR MAN: Describe, Express, Assert, Reinforce, be Mindful, Appear confident, Negotiate.",
        "radical_acceptance": "Practice the half-smile and willing hands posture. Say: 'This is how it is right now.'",
        "opposite_action": "If afraid and safe, approach. If angry and unjustified, be gentle. If sad and no loss, get active.",
        "accumulating_positives": "Schedule one pleasant event per day. Build mastery through small, achievable tasks.",
    },
    "ACT": {
        "cognitive_fusion": "Defusion exercise: add 'I am having the thought that...' before the thought.",
        "experiential_avoidance": "Willingness practice: rate willingness 0-10 to have this feeling. Aim to increase by 1 point.",
        "psychological_inflexibility": "Values compass: identify which value this situation connects to. Act from that value.",
        "defusion": "Leaves on a stream: place each thought on a leaf and watch it float downstream.",
        "present_moment_awareness": "Five senses check-in: what can you see, hear, smell, taste, and touch right now?",
        "values_clarity": "Complete a values card sort. Rank your top 5 values. How does today align?",
        "committed_action": "Set one small, values-aligned action for the next 24 hours. Do it regardless of feelings.",
        "self_as_context": "The chessboard metaphor: you are the board, not the pieces. Observe without choosing sides.",
        "acceptance": "Expansion: breathe into the sensation. Make room for it. Let it be there without fighting.",
        "contact_with_present": "Drop anchor: push feet into floor, straighten spine, name 5 things you see.",
    },
    "Somatic": {
        "chest_tightness": "Place hand on chest. Breathe into the hand. Exhale slowly, imagining space opening.",
        "shallow_breathing": "4-7-8 breathing: inhale 4 counts, hold 7, exhale 8. Repeat 3-4 cycles.",
        "gut_sinking": "Grounding through feet: press feet firmly into the floor. Feel the support of the ground.",
        "jaw_clenching": "Open mouth wide, then gently close. Massage the masseter muscle. Sigh audibly.",
        "shoulder_tension": "Progressive release: shrug shoulders to ears for 5 seconds, then drop completely. Repeat 3x.",
        "throat_constriction": "Gentle humming or vocal toning. Let sound vibrate through the throat.",
        "heat_in_face": "Cool compress on face. Splash cold water. Activate the dive reflex.",
        "numbness": "Bilateral tapping: alternate tapping knees or crossing arms and tapping shoulders.",
        "trembling": "Allow the trembling to continue. It is the nervous system completing a survival response.",
        "heaviness_in_limbs": "Gentle movement: small circles with wrists or ankles. Gradually increase range.",
    },
    "IFS": {
        "protector_activated": "Ask the protector: 'What are you afraid will happen if you step back?' Listen without judgment.",
        "exile_triggered": "Acknowledge the exile: 'I see you. I know you are hurting.' Do not rush to fix.",
        "firefighter_response": "Thank the firefighter for trying to help. Ask: 'What do you need right now?'",
        "self_energy": "Strengthen Self-energy with the 8 Cs: calm, curiosity, clarity, compassion, confidence, courage, creativity, connectedness.",
        "blended_state": "Ask: 'Is this me, or is this a part?' Create separation by visualizing the part stepping back.",
        "polarization": "Invite both parts to a negotiation. Let each speak without the other interrupting.",
        "burdened_part": "Witness the part's story. Ask: 'Would you be willing to release this burden if you could?'",
        "unburdened_part": "Celebrate the unburdening. Ask the part: 'What would you like to do now that you are free?'",
        "manager_part": "Appreciate the manager's effort: 'You have worked so hard to keep things together.'",
        "trailhead": "Follow the sensation inward. Ask: 'Where do I feel this in my body? What part is here?'",
    },
}


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class TranslationResult:
    """Result of translating a single concept between frameworks."""

    source_framework: str
    source_concept: str
    source_description: str
    target_framework: str
    target_concept: str
    target_description: str
    target_intervention: str


@dataclass
class FrameworkProfile:
    """A concept described across all five frameworks simultaneously."""

    source_framework: str
    source_concept: str
    translations: Dict[str, str] = field(default_factory=dict)
    descriptions: Dict[str, str] = field(default_factory=dict)
    interventions: Dict[str, str] = field(default_factory=dict)


# ── Core functions ───────────────────────────────────────────────────────────


def _normalize_framework(name: str) -> Optional[str]:
    """Return canonical framework name (case-insensitive), or None."""
    lookup = {f.lower(): f for f in FRAMEWORKS}
    return lookup.get(name.strip().lower())


def _normalize_concept(framework: str, concept: str) -> Optional[str]:
    """Return canonical concept key for a framework, or None."""
    vocab = FRAMEWORK_VOCABULARY.get(framework, {})
    key = concept.strip().lower().replace(" ", "_").replace("-", "_")
    if key in vocab:
        return key
    # Try partial match
    for k in vocab:
        if key in k or k in key:
            return k
    return None


def translate_emotion(
    source_framework: str,
    concept: str,
    target_framework: str,
) -> Dict[str, Any]:
    """Translate a single concept from one framework to another.

    Parameters
    ----------
    source_framework : str
        The originating framework (e.g. "CBT").
    concept : str
        The concept key (e.g. "catastrophizing").
    target_framework : str
        The destination framework (e.g. "ACT").

    Returns
    -------
    dict with keys: source_framework, source_concept, source_description,
    target_framework, target_concept, target_description, target_intervention,
    or an error dict.
    """
    src_fw = _normalize_framework(source_framework)
    tgt_fw = _normalize_framework(target_framework)

    if src_fw is None:
        return {"error": f"Unknown source framework: {source_framework}"}
    if tgt_fw is None:
        return {"error": f"Unknown target framework: {target_framework}"}

    src_concept = _normalize_concept(src_fw, concept)
    if src_concept is None:
        return {"error": f"Unknown concept '{concept}' in framework {src_fw}"}

    map_key = f"{src_fw}:{src_concept}"
    mapping = _MAPPING_TABLE.get(map_key)
    if mapping is None:
        return {"error": f"No mapping found for {map_key}"}

    tgt_concept = mapping.get(tgt_fw)
    if tgt_concept is None:
        return {"error": f"No mapping from {map_key} to {tgt_fw}"}

    src_desc = FRAMEWORK_VOCABULARY.get(src_fw, {}).get(src_concept, "")
    tgt_desc = FRAMEWORK_VOCABULARY.get(tgt_fw, {}).get(tgt_concept, "")
    tgt_intervention = INTERVENTIONS.get(tgt_fw, {}).get(tgt_concept, "")

    result = TranslationResult(
        source_framework=src_fw,
        source_concept=src_concept,
        source_description=src_desc,
        target_framework=tgt_fw,
        target_concept=tgt_concept,
        target_description=tgt_desc,
        target_intervention=tgt_intervention,
    )
    return asdict(result)


def compile_across_frameworks(
    source_framework: str,
    concept: str,
) -> Dict[str, Any]:
    """Compile a concept into all five frameworks.

    Parameters
    ----------
    source_framework : str
        The originating framework.
    concept : str
        The concept key within that framework.

    Returns
    -------
    dict with keys: source_framework, source_concept, translations (dict of
    framework -> concept), descriptions, interventions, or error dict.
    """
    src_fw = _normalize_framework(source_framework)
    if src_fw is None:
        return {"error": f"Unknown framework: {source_framework}"}

    src_concept = _normalize_concept(src_fw, concept)
    if src_concept is None:
        return {"error": f"Unknown concept '{concept}' in framework {src_fw}"}

    map_key = f"{src_fw}:{src_concept}"
    mapping = _MAPPING_TABLE.get(map_key)
    if mapping is None:
        return {"error": f"No mapping found for {map_key}"}

    translations: Dict[str, str] = {}
    descriptions: Dict[str, str] = {}
    interventions: Dict[str, str] = {}

    for fw in FRAMEWORKS:
        tgt_concept = mapping.get(fw, "")
        translations[fw] = tgt_concept
        descriptions[fw] = FRAMEWORK_VOCABULARY.get(fw, {}).get(tgt_concept, "")
        interventions[fw] = INTERVENTIONS.get(fw, {}).get(tgt_concept, "")

    return {
        "source_framework": src_fw,
        "source_concept": src_concept,
        "translations": translations,
        "descriptions": descriptions,
        "interventions": interventions,
    }


def get_framework_vocabulary(framework: str) -> Dict[str, Any]:
    """Return the full vocabulary for a framework.

    Parameters
    ----------
    framework : str
        Framework name (case-insensitive).

    Returns
    -------
    dict with keys: framework, description, vocabulary (dict of concept -> description),
    concept_count, or error dict.
    """
    fw = _normalize_framework(framework)
    if fw is None:
        return {"error": f"Unknown framework: {framework}"}

    vocab = FRAMEWORK_VOCABULARY.get(fw, {})
    return {
        "framework": fw,
        "description": FRAMEWORK_DESCRIPTIONS.get(fw, ""),
        "vocabulary": vocab,
        "concept_count": len(vocab),
    }


def suggest_intervention_per_framework(
    source_framework: str,
    concept: str,
) -> Dict[str, Any]:
    """Suggest what each framework would recommend for the same emotional state.

    Parameters
    ----------
    source_framework : str
        The framework the concept originates from.
    concept : str
        The concept key.

    Returns
    -------
    dict with keys: source_framework, source_concept, interventions (dict of
    framework -> intervention text), or error dict.
    """
    compiled = compile_across_frameworks(source_framework, concept)
    if "error" in compiled:
        return compiled

    return {
        "source_framework": compiled["source_framework"],
        "source_concept": compiled["source_concept"],
        "interventions": compiled["interventions"],
    }


def compute_framework_profile(
    source_framework: str,
    concept: str,
) -> Optional[FrameworkProfile]:
    """Build a FrameworkProfile dataclass for a concept.

    Returns None if the framework or concept is invalid.
    """
    compiled = compile_across_frameworks(source_framework, concept)
    if "error" in compiled:
        return None

    return FrameworkProfile(
        source_framework=compiled["source_framework"],
        source_concept=compiled["source_concept"],
        translations=compiled["translations"],
        descriptions=compiled["descriptions"],
        interventions=compiled["interventions"],
    )


def profile_to_dict(profile: Optional[FrameworkProfile]) -> Optional[Dict[str, Any]]:
    """Serialize a FrameworkProfile to a plain dict, or return None."""
    if profile is None:
        return None
    return asdict(profile)
