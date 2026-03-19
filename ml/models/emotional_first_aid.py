"""EEG-guided emotional first-aid protocols for acute crises.

Provides step-by-step de-escalation guidance for acute emotional crises
detected from real-time EEG/emotion data. Each protocol has 4-6 timed
steps with expected EEG response targets; the engine advances or repeats
steps based on whether brain metrics improve.

Crisis types detected:
- Panic: very high arousal + very negative valence
- Acute stress: sudden stress spike above threshold
- Dissociation: very low arousal + flat affect (near-zero valence range)
- Rage: high arousal + anger indicators

Protocol families:
- Grounding: 5-4-3-2-1 sensory, cold water, bilateral tapping
- Panic: slow exhale, physiological sigh, body scan
- Dissociation: feet on floor, name objects, strong sensation
- Rage: progressive relaxation, count backward, leave situation

CLINICAL DISCLAIMER: This is a research and educational tool only.
It is NOT a substitute for professional clinical care. For any genuine
mental health crisis, contact a licensed professional or emergency
services immediately.

Issue #438.
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
    "Clinical disclaimer: This emotional first-aid protocol is a research "
    "and educational tool only. It is NOT a substitute for professional "
    "clinical care. For any genuine mental health crisis, contact a licensed "
    "mental health professional or call emergency services (988 Suicide & "
    "Crisis Lifeline in the US, or your local equivalent)."
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CrisisType(str, Enum):
    PANIC = "panic"
    ACUTE_STRESS = "acute_stress"
    DISSOCIATION = "dissociation"
    RAGE = "rage"
    NONE = "none"


class ProtocolCategory(str, Enum):
    GROUNDING = "grounding"
    PANIC = "panic"
    DISSOCIATION = "dissociation"
    RAGE = "rage"


# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

_PANIC_AROUSAL_THRESHOLD = 0.85
_PANIC_VALENCE_THRESHOLD = -0.5
_STRESS_THRESHOLD = 0.80
_DISSOCIATION_AROUSAL_CEILING = 0.20
_DISSOCIATION_VALENCE_RANGE = 0.15  # flat affect = valence near zero
_RAGE_AROUSAL_THRESHOLD = 0.75
_RAGE_ANGER_THRESHOLD = 0.60

# Step evaluation thresholds
_IMPROVEMENT_THRESHOLD = 0.10  # minimum delta to count as improvement
_SEVERE_EPISODE_THRESHOLD = 0.90  # arousal or stress above this = severe


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CrisisState:
    """Snapshot of the user's current emotional/physiological state."""

    valence: float = 0.0
    arousal: float = 0.0
    stress_index: float = 0.0
    anger_index: float = 0.0
    focus_index: float = 0.5
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "stress_index": round(self.stress_index, 4),
            "anger_index": round(self.anger_index, 4),
            "focus_index": round(self.focus_index, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class ProtocolStep:
    """A single step within a de-escalation protocol."""

    step_number: int
    instruction: str
    duration_seconds: int
    expected_response: str  # description of expected EEG/metric change
    arousal_target_delta: float = -0.05  # expected arousal decrease per step
    stress_target_delta: float = -0.05  # expected stress decrease per step

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "instruction": self.instruction,
            "duration_seconds": self.duration_seconds,
            "expected_response": self.expected_response,
            "arousal_target_delta": self.arousal_target_delta,
            "stress_target_delta": self.stress_target_delta,
        }


@dataclass
class Protocol:
    """A complete de-escalation protocol with ordered steps."""

    id: str
    name: str
    category: ProtocolCategory
    description: str
    crisis_types: List[CrisisType]
    steps: List[ProtocolStep]
    severity_range: tuple = (0.0, 1.0)  # (min, max) severity this protocol handles

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "crisis_types": [ct.value for ct in self.crisis_types],
            "steps": [s.to_dict() for s in self.steps],
            "severity_range": list(self.severity_range),
            "total_steps": len(self.steps),
        }


@dataclass
class StepEvaluation:
    """Result of evaluating whether a step was effective."""

    step_number: int
    arousal_before: float
    arousal_after: float
    stress_before: float
    stress_after: float
    arousal_improved: bool
    stress_improved: bool
    effective: bool
    recommendation: str  # "advance" or "repeat"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "arousal_before": round(self.arousal_before, 4),
            "arousal_after": round(self.arousal_after, 4),
            "stress_before": round(self.stress_before, 4),
            "stress_after": round(self.stress_after, 4),
            "arousal_improved": self.arousal_improved,
            "stress_improved": self.stress_improved,
            "effective": self.effective,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Protocol library
# ---------------------------------------------------------------------------

PROTOCOL_LIBRARY: List[Protocol] = [
    # -- Grounding protocols --
    Protocol(
        id="grounding_54321",
        name="5-4-3-2-1 Sensory Grounding",
        category=ProtocolCategory.GROUNDING,
        description=(
            "Engage all five senses to anchor yourself in the present moment. "
            "Activates the prefrontal cortex and reduces amygdala hyperactivation."
        ),
        crisis_types=[CrisisType.PANIC, CrisisType.ACUTE_STRESS, CrisisType.DISSOCIATION],
        steps=[
            ProtocolStep(1, "Name 5 things you can SEE. Look around slowly and identify each one aloud.", 30, "Increased alpha, reduced high-beta as attention shifts from internal to external"),
            ProtocolStep(2, "Name 4 things you can TOUCH. Feel textures around you — fabric, surface, skin.", 30, "Somatosensory cortex activation, arousal beginning to decrease"),
            ProtocolStep(3, "Name 3 things you can HEAR. Listen for distant sounds, close sounds, ambient noise.", 25, "Auditory engagement, further alpha increase"),
            ProtocolStep(4, "Name 2 things you can SMELL. Breathe in slowly through your nose.", 20, "Limbic regulation via olfactory pathway, stress index decreasing"),
            ProtocolStep(5, "Name 1 thing you can TASTE. Notice any taste in your mouth right now.", 15, "Full sensory grounding achieved, arousal notably lower"),
        ],
        severity_range=(0.3, 0.85),
    ),
    Protocol(
        id="grounding_cold_water",
        name="Cold Water Reset",
        category=ProtocolCategory.GROUNDING,
        description=(
            "Mammalian dive reflex activation. Cold water on wrists or face triggers "
            "vagal tone increase and rapid parasympathetic engagement."
        ),
        crisis_types=[CrisisType.PANIC, CrisisType.RAGE],
        steps=[
            ProtocolStep(1, "Find cold water — a sink, a cold bottle, ice. Hold it in your hands or against your wrists.", 15, "Initial orienting response, brief arousal spike then rapid drop"),
            ProtocolStep(2, "Splash cold water on your face or hold a cold object to your cheeks and forehead.", 20, "Dive reflex activation: heart rate drops, vagal tone increases", -0.10, -0.08),
            ProtocolStep(3, "Breathe slowly while keeping the cold sensation. In for 4, out for 6.", 30, "Sustained parasympathetic engagement, alpha increasing", -0.08, -0.06),
            ProtocolStep(4, "Remove the cold stimulus. Notice how your body feels different now.", 20, "Post-stimulus rebound: lower baseline arousal, stress decreased"),
        ],
        severity_range=(0.5, 1.0),
    ),
    Protocol(
        id="grounding_bilateral_tapping",
        name="Bilateral Tapping",
        category=ProtocolCategory.GROUNDING,
        description=(
            "Alternating left-right tactile stimulation. Based on EMDR bilateral "
            "stimulation research — engages both hemispheres and reduces emotional intensity."
        ),
        crisis_types=[CrisisType.ACUTE_STRESS, CrisisType.PANIC],
        steps=[
            ProtocolStep(1, "Cross your arms over your chest, each hand on the opposite shoulder (butterfly hug).", 10, "Posture establishes bilateral readiness"),
            ProtocolStep(2, "Alternately tap left shoulder, then right shoulder. Slow, steady rhythm — one tap per second.", 30, "Bilateral activation, frontal alpha asymmetry normalizing", -0.06, -0.05),
            ProtocolStep(3, "Continue tapping. Focus on the rhythm. Let thoughts pass without engaging them.", 30, "Theta increase (processing), high-beta decreasing", -0.06, -0.05),
            ProtocolStep(4, "Slow the tapping gradually. Take 3 deep breaths between the last few taps.", 20, "Alpha dominance returning, arousal significantly lower"),
            ProtocolStep(5, "Stop tapping. Rest your hands. Notice the calm.", 15, "Parasympathetic rebound, stress and arousal lowered"),
        ],
        severity_range=(0.3, 0.80),
    ),
    # -- Panic protocols --
    Protocol(
        id="panic_slow_exhale",
        name="Extended Exhale Breathing",
        category=ProtocolCategory.PANIC,
        description=(
            "Deliberately longer exhales activate the vagus nerve and shift "
            "autonomic balance toward parasympathetic dominance."
        ),
        crisis_types=[CrisisType.PANIC, CrisisType.ACUTE_STRESS],
        steps=[
            ProtocolStep(1, "Sit or stand with feet flat on the ground. Place one hand on your chest, one on your belly.", 10, "Orienting and body awareness, preparing for breath work"),
            ProtocolStep(2, "Breathe IN through your nose for 4 counts. Feel your belly expand.", 20, "Diaphragmatic activation, slight arousal from inhalation"),
            ProtocolStep(3, "Breathe OUT through pursed lips for 6 counts. Slow and steady.", 25, "Vagal activation on exhale, heart rate decreasing, alpha rising", -0.08, -0.06),
            ProtocolStep(4, "Repeat: IN for 4, OUT for 6. Do this 4 more times.", 50, "Sustained parasympathetic drive, high-beta decreasing", -0.10, -0.08),
            ProtocolStep(5, "Return to natural breathing. Notice the difference in your body.", 15, "Post-exercise calm, arousal and stress notably reduced"),
        ],
        severity_range=(0.4, 1.0),
    ),
    Protocol(
        id="panic_physiological_sigh",
        name="Physiological Sigh",
        category=ProtocolCategory.PANIC,
        description=(
            "Double inhale followed by long exhale — the fastest known voluntary "
            "method to reduce arousal. Reinflates collapsed alveoli and maximizes "
            "CO2 offloading on exhale."
        ),
        crisis_types=[CrisisType.PANIC],
        steps=[
            ProtocolStep(1, "Take a quick inhale through your nose to fill lungs halfway.", 5, "First inhale — partial lung inflation"),
            ProtocolStep(2, "Immediately take a second short inhale on top of the first — a quick 'sip' of air.", 5, "Double inhale reinflates alveoli, maximizing surface area for gas exchange"),
            ProtocolStep(3, "Now exhale very slowly through your mouth for as long as you can.", 15, "Maximum CO2 offload, strong vagal activation, rapid arousal drop", -0.12, -0.10),
            ProtocolStep(4, "Repeat the double-inhale + long exhale 3 more times.", 60, "Cumulative parasympathetic activation, panic subsiding", -0.15, -0.12),
            ProtocolStep(5, "Breathe normally. Notice: heart rate is lower, muscles are softer.", 15, "Arousal significantly reduced, alpha rhythm stabilizing"),
        ],
        severity_range=(0.6, 1.0),
    ),
    Protocol(
        id="panic_body_scan",
        name="Rapid Body Scan",
        category=ProtocolCategory.PANIC,
        description=(
            "Quick top-to-bottom body awareness scan. Redirects attention from "
            "catastrophic thoughts to bodily sensations, engaging somatosensory cortex."
        ),
        crisis_types=[CrisisType.PANIC, CrisisType.ACUTE_STRESS],
        steps=[
            ProtocolStep(1, "Close your eyes or soften your gaze. Notice the top of your head — any tension there?", 15, "Attention redirection begins, frontal activity shifting"),
            ProtocolStep(2, "Move attention to your face: jaw, forehead, eyes. Deliberately relax each area.", 20, "Frontalis EMG decreasing, alpha emerging at frontal sites", -0.05, -0.04),
            ProtocolStep(3, "Notice your shoulders and arms. Let them drop. Unclench your hands.", 20, "Muscle tension releasing, arousal decreasing", -0.06, -0.05),
            ProtocolStep(4, "Feel your torso: chest, belly, lower back. Breathe into any tight spots.", 20, "Diaphragmatic engagement, deeper relaxation", -0.06, -0.05),
            ProtocolStep(5, "Notice your legs and feet. Feel the ground beneath you.", 15, "Full body grounding, proprioceptive anchoring"),
            ProtocolStep(6, "Open your eyes slowly. You are here, in this moment, and you are safe.", 10, "Reorientation complete, arousal and stress markedly reduced"),
        ],
        severity_range=(0.3, 0.85),
    ),
    # -- Dissociation protocols --
    Protocol(
        id="dissociation_feet_on_floor",
        name="Feet on Floor",
        category=ProtocolCategory.DISSOCIATION,
        description=(
            "Physical grounding through feet — the simplest and fastest "
            "dissociation interrupt. Activates proprioceptive and somatosensory "
            "pathways to re-establish body awareness."
        ),
        crisis_types=[CrisisType.DISSOCIATION],
        steps=[
            ProtocolStep(1, "Press your feet firmly into the floor. Feel the pressure. Push down.", 15, "Proprioceptive activation, beginning to re-engage somatosensory cortex", 0.03, -0.02),
            ProtocolStep(2, "Stamp your feet gently — left, right, left, right. Feel the impact.", 20, "Motor cortex activation, arousal increasing toward normal range", 0.05, -0.03),
            ProtocolStep(3, "Wiggle your toes inside your shoes. Spread them wide, then scrunch them.", 15, "Fine motor engagement, sensory detail increasing", 0.03, -0.02),
            ProtocolStep(4, "Stand up if you can. Feel your full weight on your feet. You are HERE.", 20, "Postural control engaging, vestibular activation, arousal normalizing", 0.04, -0.03),
        ],
        severity_range=(0.0, 0.80),
    ),
    Protocol(
        id="dissociation_name_objects",
        name="Name and Describe Objects",
        category=ProtocolCategory.DISSOCIATION,
        description=(
            "Verbal engagement with the environment to reconnect "
            "prefrontal-language circuits and re-establish reality contact."
        ),
        crisis_types=[CrisisType.DISSOCIATION],
        steps=[
            ProtocolStep(1, "Look around. Pick one object. Say its name aloud: 'That is a [chair].'", 15, "Language centers activating, left hemisphere engagement", 0.03, -0.02),
            ProtocolStep(2, "Describe it: color, shape, texture. 'It is brown, wooden, has four legs.'", 20, "Prefrontal engagement increasing, analytical processing", 0.04, -0.02),
            ProtocolStep(3, "Pick another object. Name it and describe it in detail.", 20, "Sustained prefrontal activation, reality contact strengthening", 0.03, -0.02),
            ProtocolStep(4, "Now name where you are: 'I am in [location]. Today is [day]. I am [name].'", 15, "Autobiographical memory activation, self-reference restoring", 0.03, -0.02),
            ProtocolStep(5, "Take a slow breath. Notice you are more present. You are here.", 15, "Full re-engagement, arousal normalizing to healthy range"),
        ],
        severity_range=(0.0, 0.70),
    ),
    Protocol(
        id="dissociation_strong_sensation",
        name="Strong Sensation Anchor",
        category=ProtocolCategory.DISSOCIATION,
        description=(
            "Intense but safe sensory input to break through dissociative numbness. "
            "Strong taste, smell, or touch to activate the insula and reconnect "
            "interoceptive awareness."
        ),
        crisis_types=[CrisisType.DISSOCIATION],
        steps=[
            ProtocolStep(1, "Find something with a strong taste or smell — peppermint, lemon, strong coffee, hot sauce.", 15, "Seeking sensory stimulus, preparing interoceptive pathway", 0.02, -0.01),
            ProtocolStep(2, "Taste or smell it. Focus entirely on the sensation. Describe it to yourself.", 20, "Insula activation, interoceptive awareness returning", 0.05, -0.03),
            ProtocolStep(3, "Squeeze an ice cube, snap a rubber band on your wrist, or hold something very cold.", 20, "Nociceptive input (safe), strong somatosensory activation", 0.06, -0.03),
            ProtocolStep(4, "Name what you feel: 'I feel cold. I feel sharp. I am here.'", 15, "Verbal labeling + sensation = dual-pathway re-engagement", 0.03, -0.02),
        ],
        severity_range=(0.3, 1.0),
    ),
    # -- Rage protocols --
    Protocol(
        id="rage_progressive_relaxation",
        name="Rapid Progressive Relaxation",
        category=ProtocolCategory.RAGE,
        description=(
            "Tense-and-release muscle groups to discharge physical tension "
            "from anger. Engages the reciprocal inhibition principle: muscles "
            "cannot be tense and relaxed simultaneously."
        ),
        crisis_types=[CrisisType.RAGE],
        steps=[
            ProtocolStep(1, "Make tight fists with both hands. Squeeze as hard as you can for 5 seconds.", 10, "Voluntary muscle tension, acknowledging the anger in the body"),
            ProtocolStep(2, "Release your fists. Let your fingers go completely limp. Feel the contrast.", 15, "Reciprocal inhibition: post-tetanic relaxation, EMG dropping", -0.06, -0.05),
            ProtocolStep(3, "Shrug your shoulders up to your ears. Hold 5 seconds. Then drop them.", 15, "Trapezius tension-release cycle, upper body decompression", -0.06, -0.05),
            ProtocolStep(4, "Tighten your whole body — arms, legs, core, face. Hold 5 seconds. Release everything.", 15, "Full-body tension-release, maximum reciprocal inhibition effect", -0.08, -0.07),
            ProtocolStep(5, "Take 3 slow breaths. Notice: your body has let go of some of the anger.", 20, "Parasympathetic rebound, arousal and anger measurably reduced"),
        ],
        severity_range=(0.4, 0.90),
    ),
    Protocol(
        id="rage_count_backward",
        name="Count Backward from 100",
        category=ProtocolCategory.RAGE,
        description=(
            "Cognitive distraction technique. Counting backward by 7s requires "
            "working memory load that competes with anger rumination for "
            "prefrontal resources."
        ),
        crisis_types=[CrisisType.RAGE, CrisisType.ACUTE_STRESS],
        steps=[
            ProtocolStep(1, "Start counting backward from 100 by 7s: 100, 93, 86, 79... Say each number.", 30, "Working memory engagement, prefrontal cortex competing with amygdala", -0.05, -0.04),
            ProtocolStep(2, "If you lose track, start again from the last number you remember. Keep going.", 30, "Sustained cognitive load, anger rumination circuits disrupted", -0.06, -0.05),
            ProtocolStep(3, "Slow down. Continue counting but breathe between each number.", 30, "Dual-task: counting + breathing = maximum prefrontal engagement", -0.06, -0.05),
            ProtocolStep(4, "Stop counting. Notice: the intensity of the anger has shifted.", 15, "Cognitive reappraisal window opening, arousal reduced"),
        ],
        severity_range=(0.3, 0.85),
    ),
    Protocol(
        id="rage_leave_situation",
        name="Strategic Withdrawal",
        category=ProtocolCategory.RAGE,
        description=(
            "Remove yourself from the triggering environment. Not avoidance — "
            "a deliberate timeout to prevent escalation while your prefrontal "
            "cortex regains regulatory control over the amygdala."
        ),
        crisis_types=[CrisisType.RAGE],
        steps=[
            ProtocolStep(1, "Say: 'I need a moment.' Walk away from the situation. Go to a different room or outside.", 20, "Physical removal from trigger, reducing sensory input driving anger"),
            ProtocolStep(2, "Walk briskly for 60 seconds. Move your body — the anger needs a physical outlet.", 60, "Motor discharge of arousal energy, catecholamine metabolism", -0.10, -0.08),
            ProtocolStep(3, "Stop walking. Plant your feet. Take 5 slow, deep breaths.", 30, "Transition from sympathetic to parasympathetic, heart rate dropping", -0.08, -0.07),
            ProtocolStep(4, "Ask yourself: 'What do I actually need right now?' Wait for the answer.", 30, "Prefrontal reappraisal, mentalizing circuits re-engaging", -0.05, -0.04),
            ProtocolStep(5, "When you feel ready — not before — you can return. You are in control.", 15, "Regulatory control restored, anger intensity significantly reduced"),
        ],
        severity_range=(0.6, 1.0),
    ),
]

# Index by ID for fast lookup
_PROTOCOL_BY_ID: Dict[str, Protocol] = {p.id: p for p in PROTOCOL_LIBRARY}

# Map crisis types to suitable protocol categories
_CRISIS_TO_CATEGORIES: Dict[CrisisType, List[ProtocolCategory]] = {
    CrisisType.PANIC: [ProtocolCategory.PANIC, ProtocolCategory.GROUNDING],
    CrisisType.ACUTE_STRESS: [ProtocolCategory.GROUNDING, ProtocolCategory.PANIC],
    CrisisType.DISSOCIATION: [ProtocolCategory.DISSOCIATION],
    CrisisType.RAGE: [ProtocolCategory.RAGE, ProtocolCategory.GROUNDING],
}


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


def detect_crisis_type(state: CrisisState) -> Dict[str, Any]:
    """Detect crisis type from current emotional state.

    Priority order: panic > dissociation > rage > acute_stress > none.
    Panic is highest priority because it is the most physiologically
    dangerous acute state.

    Args:
        state: Current emotional/physiological state.

    Returns:
        Dict with crisis_type, severity (0-1), indicators, and safety note.
    """
    severity = 0.0
    indicators: List[str] = []

    # Panic: very high arousal + very negative valence
    if state.arousal >= _PANIC_AROUSAL_THRESHOLD and state.valence <= _PANIC_VALENCE_THRESHOLD:
        severity = min(1.0, (state.arousal + abs(state.valence)) / 2.0)
        indicators = ["very_high_arousal", "very_negative_valence"]
        return _crisis_result(CrisisType.PANIC, severity, indicators, state)

    # Dissociation: very low arousal + flat affect
    if state.arousal <= _DISSOCIATION_AROUSAL_CEILING and abs(state.valence) <= _DISSOCIATION_VALENCE_RANGE:
        severity = min(1.0, (1.0 - state.arousal) * 0.7 + (1.0 - abs(state.valence)) * 0.3)
        indicators = ["very_low_arousal", "flat_affect"]
        return _crisis_result(CrisisType.DISSOCIATION, severity, indicators, state)

    # Rage: high arousal + anger
    if state.arousal >= _RAGE_AROUSAL_THRESHOLD and state.anger_index >= _RAGE_ANGER_THRESHOLD:
        severity = min(1.0, (state.arousal + state.anger_index) / 2.0)
        indicators = ["high_arousal", "high_anger"]
        return _crisis_result(CrisisType.RAGE, severity, indicators, state)

    # Acute stress: high stress spike
    if state.stress_index >= _STRESS_THRESHOLD:
        severity = min(1.0, state.stress_index)
        indicators = ["high_stress_spike"]
        return _crisis_result(CrisisType.ACUTE_STRESS, severity, indicators, state)

    return _crisis_result(CrisisType.NONE, 0.0, [], state)


def _crisis_result(
    crisis_type: CrisisType,
    severity: float,
    indicators: List[str],
    state: CrisisState,
) -> Dict[str, Any]:
    """Build standardized crisis detection result."""
    is_severe = severity >= _SEVERE_EPISODE_THRESHOLD
    safety_note = ""
    if is_severe:
        safety_note = (
            "This appears to be a severe episode. Please consider contacting "
            "a mental health professional or crisis line (988 in the US). "
            "These exercises can help in the moment, but professional support "
            "is strongly recommended."
        )
    elif crisis_type != CrisisType.NONE:
        safety_note = (
            "Remember: these protocols are supportive tools, not clinical "
            "treatment. If episodes are frequent or intensifying, please "
            "consult a mental health professional."
        )

    return {
        "crisis_type": crisis_type.value,
        "detected": crisis_type != CrisisType.NONE,
        "severity": round(severity, 4),
        "indicators": indicators,
        "is_severe": is_severe,
        "safety_note": safety_note,
        "state": state.to_dict(),
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


def select_protocol(
    crisis_type: CrisisType,
    severity: float = 0.5,
) -> Dict[str, Any]:
    """Select the best protocol for a given crisis type and severity.

    Filters protocols by crisis type compatibility and severity range,
    then picks the one whose severity range best matches the current severity.

    Args:
        crisis_type: Detected crisis type.
        severity: Crisis severity (0-1).

    Returns:
        Dict with selected protocol details or indication that no protocol matches.
    """
    if crisis_type == CrisisType.NONE:
        return {
            "selected": False,
            "reason": "no_crisis_detected",
            "message": "No crisis detected. No protocol needed at this time.",
            "clinical_disclaimer": _CLINICAL_DISCLAIMER,
        }

    candidates: List[Protocol] = []
    for protocol in PROTOCOL_LIBRARY:
        if crisis_type in protocol.crisis_types:
            lo, hi = protocol.severity_range
            if lo <= severity <= hi:
                candidates.append(protocol)

    if not candidates:
        # Fall back to any protocol matching the crisis type
        candidates = [p for p in PROTOCOL_LIBRARY if crisis_type in p.crisis_types]

    if not candidates:
        return {
            "selected": False,
            "reason": "no_matching_protocol",
            "message": "No suitable protocol found for this crisis type.",
            "clinical_disclaimer": _CLINICAL_DISCLAIMER,
        }

    # Pick the protocol whose severity midpoint is closest to current severity
    def _severity_fit(p: Protocol) -> float:
        mid = (p.severity_range[0] + p.severity_range[1]) / 2.0
        return abs(mid - severity)

    best = min(candidates, key=_severity_fit)

    return {
        "selected": True,
        "protocol": best.to_dict(),
        "crisis_type": crisis_type.value,
        "severity": round(severity, 4),
        "alternatives": [p.id for p in candidates if p.id != best.id],
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


def get_current_step(
    protocol_id: str,
    current_step: int = 1,
) -> Dict[str, Any]:
    """Get details for a specific step in a protocol.

    Args:
        protocol_id: ID of the active protocol.
        current_step: Step number (1-indexed).

    Returns:
        Dict with step details, progress info, and safety reminders.
    """
    protocol = _PROTOCOL_BY_ID.get(protocol_id)
    if protocol is None:
        return {"error": f"Unknown protocol: {protocol_id}"}

    total = len(protocol.steps)
    if current_step < 1 or current_step > total:
        return {"error": f"Step {current_step} out of range (1-{total})"}

    step = protocol.steps[current_step - 1]

    return {
        "protocol_id": protocol_id,
        "protocol_name": protocol.name,
        "step": step.to_dict(),
        "progress": {
            "current": current_step,
            "total": total,
            "percent_complete": round((current_step - 1) / total * 100, 1),
        },
        "is_last_step": current_step == total,
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


def evaluate_step_effectiveness(
    arousal_before: float,
    arousal_after: float,
    stress_before: float,
    stress_after: float,
    step_number: int = 1,
) -> StepEvaluation:
    """Evaluate whether a protocol step was effective based on metric changes.

    A step is effective if EITHER arousal OR stress improved by at least
    the improvement threshold.

    Args:
        arousal_before: Arousal level before the step.
        arousal_after: Arousal level after the step.
        stress_before: Stress level before the step.
        stress_after: Stress level after the step.
        step_number: Which step was just completed.

    Returns:
        StepEvaluation with improvement flags and recommendation.
    """
    arousal_delta = arousal_before - arousal_after  # positive = improved
    stress_delta = stress_before - stress_after  # positive = improved

    arousal_improved = arousal_delta >= _IMPROVEMENT_THRESHOLD
    stress_improved = stress_delta >= _IMPROVEMENT_THRESHOLD
    effective = arousal_improved or stress_improved

    recommendation = "advance" if effective else "repeat"

    return StepEvaluation(
        step_number=step_number,
        arousal_before=arousal_before,
        arousal_after=arousal_after,
        stress_before=stress_before,
        stress_after=stress_after,
        arousal_improved=arousal_improved,
        stress_improved=stress_improved,
        effective=effective,
        recommendation=recommendation,
    )


def advance_or_repeat(
    protocol_id: str,
    current_step: int,
    arousal_before: float,
    arousal_after: float,
    stress_before: float,
    stress_after: float,
    max_repeats: int = 2,
    repeat_count: int = 0,
) -> Dict[str, Any]:
    """Decide whether to advance to the next step or repeat the current one.

    Uses EEG-guided progression: advance when metrics improve, repeat if not.
    After max_repeats without improvement, advance anyway to avoid getting stuck.

    Args:
        protocol_id: Active protocol ID.
        current_step: Current step number (1-indexed).
        arousal_before: Arousal before the step.
        arousal_after: Arousal after the step.
        stress_before: Stress before the step.
        stress_after: Stress after the step.
        max_repeats: Maximum times to repeat a step.
        repeat_count: How many times this step has been repeated already.

    Returns:
        Dict with evaluation, next step info, and guidance.
    """
    protocol = _PROTOCOL_BY_ID.get(protocol_id)
    if protocol is None:
        return {"error": f"Unknown protocol: {protocol_id}"}

    total = len(protocol.steps)
    evaluation = evaluate_step_effectiveness(
        arousal_before, arousal_after, stress_before, stress_after, current_step
    )

    # Decide action
    if evaluation.effective:
        action = "advance"
    elif repeat_count >= max_repeats:
        action = "advance"  # don't get stuck
        evaluation.recommendation = "advance"
    else:
        action = "repeat"

    if action == "advance" and current_step >= total:
        # Protocol complete
        return {
            "action": "complete",
            "evaluation": evaluation.to_dict(),
            "protocol_complete": True,
            "message": (
                "Protocol complete. Take a moment to notice how you feel. "
                "If you still feel distressed, you can restart or try a different protocol. "
                "If episodes continue, please reach out to a mental health professional."
            ),
            "clinical_disclaimer": _CLINICAL_DISCLAIMER,
        }

    if action == "advance":
        next_step = current_step + 1
        next_step_info = get_current_step(protocol_id, next_step)
        return {
            "action": "advance",
            "evaluation": evaluation.to_dict(),
            "protocol_complete": False,
            "next_step": next_step,
            "next_step_info": next_step_info,
            "message": "Good progress. Moving to the next step.",
            "clinical_disclaimer": _CLINICAL_DISCLAIMER,
        }

    # Repeat
    step_info = get_current_step(protocol_id, current_step)
    return {
        "action": "repeat",
        "evaluation": evaluation.to_dict(),
        "protocol_complete": False,
        "next_step": current_step,
        "repeat_count": repeat_count + 1,
        "max_repeats": max_repeats,
        "next_step_info": step_info,
        "message": (
            "Not enough improvement yet — that is okay. "
            "Let's try this step again. Take your time."
        ),
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


def protocol_to_dict(protocol_id: str) -> Optional[Dict[str, Any]]:
    """Serialize a protocol by ID.

    Args:
        protocol_id: The protocol ID to look up.

    Returns:
        Protocol dict or None if not found.
    """
    protocol = _PROTOCOL_BY_ID.get(protocol_id)
    if protocol is None:
        return None
    return protocol.to_dict()
