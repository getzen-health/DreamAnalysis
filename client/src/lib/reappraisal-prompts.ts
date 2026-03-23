/**
 * reappraisal-prompts.ts — Cognitive reappraisal prompts for neurofeedback (#526)
 *
 * When stress is detected during a neurofeedback session, the system can surface
 * a cognitive reappraisal prompt. These are evidence-based reframing suggestions
 * drawn from CBT and emotion regulation research (Gross, 1998; McRae et al., 2012).
 *
 * Each prompt includes the reframe text and a short rationale.
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface ReappraisalPrompt {
  /** Unique identifier */
  id: string;
  /** The reframe suggestion shown to the user */
  text: string;
  /** Brief scientific rationale (1 sentence) */
  rationale: string;
  /** Category of reappraisal strategy */
  category: "reframe" | "distance" | "perspective" | "growth";
}

// ── Prompt library ─────────────────────────────────────────────────────────

export const REAPPRAISAL_PROMPTS: readonly ReappraisalPrompt[] = [
  {
    id: "growth-stress",
    text: "Try reframing: instead of 'this is stressful', think 'this is helping me grow'.",
    rationale: "Reappraising stress as enhancing (vs. debilitating) improves cardiovascular responses and performance (Jamieson et al., 2012).",
    category: "growth",
  },
  {
    id: "observer",
    text: "Step back and observe: 'I notice I'm feeling tense' rather than 'I am tense'.",
    rationale: "Self-distancing reduces emotional reactivity and prefrontal cortex load (Kross et al., 2014).",
    category: "distance",
  },
  {
    id: "temporary",
    text: "Remind yourself: this feeling is temporary. It will pass, like every feeling before it.",
    rationale: "Temporal reappraisal activates ventrolateral prefrontal cortex and down-regulates amygdala (Diekhof et al., 2011).",
    category: "perspective",
  },
  {
    id: "challenge",
    text: "Reframe this as a challenge, not a threat. Your body is preparing you to perform.",
    rationale: "Challenge appraisals produce more efficient cardiac output and better task performance than threat appraisals (Blascovich, 2008).",
    category: "reframe",
  },
  {
    id: "curious",
    text: "Get curious about the tension: where do you feel it? What does it want to tell you?",
    rationale: "Mindful curiosity reduces experiential avoidance and increases distress tolerance (Kashdan & Rottenberg, 2010).",
    category: "distance",
  },
  {
    id: "wise-friend",
    text: "What would you say to a close friend feeling this way? Offer yourself that same advice.",
    rationale: "Self-compassion and perspective-taking activate the caregiving system, reducing cortisol (Neff & Germer, 2013).",
    category: "perspective",
  },
  {
    id: "body-signal",
    text: "Your body's stress response means your brain is engaged. Channel that energy toward focus.",
    rationale: "Arousal reappraisal redirects sympathetic activation from anxiety to attentional resources (Mendes & Park, 2014).",
    category: "growth",
  },
  {
    id: "bigger-picture",
    text: "Zoom out: how important will this feel in a week? In a year? Let the bigger picture settle you.",
    rationale: "Temporal distancing reduces negative affect by 25-40% in lab studies (Bruehlman-Senecal & Ayduk, 2015).",
    category: "perspective",
  },
  {
    id: "acceptance",
    text: "You don't have to fix this feeling. Just let it be here without fighting it.",
    rationale: "Acceptance-based strategies reduce emotional suppression costs and paradoxical amplification (Campbell-Sills et al., 2006).",
    category: "distance",
  },
  {
    id: "learning",
    text: "Every difficult moment is data. You're learning what your brain does under pressure.",
    rationale: "Framing experiences as learning opportunities activates growth mindset neural pathways (Moser et al., 2011).",
    category: "growth",
  },
] as const;

// ── Selection logic ────────────────────────────────────────────────────────

let _lastPromptIndex = -1;

/**
 * Select a reappraisal prompt, avoiding repeats.
 * Optionally filter by category.
 */
export function selectReappraisalPrompt(
  category?: ReappraisalPrompt["category"],
): ReappraisalPrompt {
  const candidates = category
    ? REAPPRAISAL_PROMPTS.filter((p) => p.category === category)
    : [...REAPPRAISAL_PROMPTS];

  if (candidates.length === 0) {
    return REAPPRAISAL_PROMPTS[0]; // fallback
  }

  // Avoid repeating the last shown prompt
  let idx: number;
  if (candidates.length === 1) {
    idx = 0;
  } else {
    do {
      idx = Math.floor(Math.random() * candidates.length);
    } while (
      candidates.length > 1 &&
      REAPPRAISAL_PROMPTS.indexOf(candidates[idx]) === _lastPromptIndex
    );
  }

  _lastPromptIndex = REAPPRAISAL_PROMPTS.indexOf(candidates[idx]);
  return candidates[idx];
}

/**
 * Determine whether to show a reappraisal prompt based on stress level.
 * Returns a prompt if stress exceeds the threshold, null otherwise.
 *
 * @param stressLevel 0-1 stress index
 * @param threshold minimum stress to trigger (default 0.6)
 */
export function shouldShowReappraisal(
  stressLevel: number,
  threshold: number = 0.6,
): ReappraisalPrompt | null {
  if (stressLevel >= threshold) {
    return selectReappraisalPrompt();
  }
  return null;
}
