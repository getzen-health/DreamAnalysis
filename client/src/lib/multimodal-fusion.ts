import { sbGetSetting, sbSaveGeneric } from "./supabase-store";
/**
 * Adaptive multimodal emotion fusion — merges EEG, voice, and health signals
 * into a single best estimate using confidence-weighted blending.
 *
 * Instead of fixed weights (65% voice, 35% health), each modality's weight
 * is proportional to its confidence score, normalized to sum to 1.
 *
 * User-adaptive learning: when a user corrects an emotion, the system learns
 * which modality was most accurate for THIS user and adjusts future weights
 * via per-modality confidence multipliers stored in localStorage.
 *
 * Works with any subset of modalities (1, 2, or all 3).
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface ModalityInput {
  valence: number;       // -1 to 1
  arousal: number;       // 0 to 1
  stress: number;        // 0 to 1
  confidence: number;    // 0 to 1
  emotion: string;
  source: "eeg" | "voice" | "health";
}

export interface FusedResult {
  emotion: string;
  valence: number;
  arousal: number;
  stress: number;
  focus: number;
  confidence: number;
  sources: string[];                 // which modalities contributed
  weights: Record<string, number>;   // actual weights used
  agreement: number;                 // 0-1, how much sources agree
  model_type: "multimodal_fusion";
}

type LearnedMultipliers = Record<string, number>;

export interface ModalityAccuracyEntry {
  correct: number;
  total: number;
  accuracy: number;
}

export interface ModalityAccuracy {
  eeg: ModalityAccuracyEntry;
  voice: ModalityAccuracyEntry;
  health: ModalityAccuracyEntry;
  lastUpdated: string;
}

const STORAGE_KEY = "ndw_fusion_weights";
const ACCURACY_STORAGE_KEY = "ndw_modality_accuracy";
const ACCURACY_MIN_SAMPLES = 5;
const BOOST_STEP = 0.05;
const DECAY_STEP = 0.02;

// ── Helpers ────────────────────────────────────────────────────────────────

function clip(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// ── Learned multiplier persistence ─────────────────────────────────────────

export function loadLearnedMultipliers(): LearnedMultipliers {
  try {
    const raw = sbGetSetting(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      return {
        eeg: parsed.eeg ?? 1.0,
        voice: parsed.voice ?? 1.0,
        health: parsed.health ?? 1.0,
      };
    }
  } catch { /* corrupted — start fresh */ }
  return { eeg: 1.0, voice: 1.0, health: 1.0 };
}

function saveLearnedMultipliers(m: LearnedMultipliers): void {
  try {
    sbSaveGeneric(STORAGE_KEY, m);
  } catch { /* localStorage full or unavailable */ }
}

// ── Learned modality accuracy tracking ─────────────────────────────────────

function emptyAccuracyEntry(): ModalityAccuracyEntry {
  return { correct: 0, total: 0, accuracy: 0 };
}

/**
 * Load per-modality accuracy stats from localStorage.
 * Returns zeroed entries if no history exists.
 */
export function getModalityAccuracies(): ModalityAccuracy {
  try {
    const raw = sbGetSetting(ACCURACY_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      return {
        eeg: parsed.eeg ?? emptyAccuracyEntry(),
        voice: parsed.voice ?? emptyAccuracyEntry(),
        health: parsed.health ?? emptyAccuracyEntry(),
        lastUpdated: parsed.lastUpdated ?? new Date().toISOString(),
      };
    }
  } catch { /* corrupted — start fresh */ }
  return {
    eeg: emptyAccuracyEntry(),
    voice: emptyAccuracyEntry(),
    health: emptyAccuracyEntry(),
    lastUpdated: new Date().toISOString(),
  };
}

/**
 * Update the running accuracy for a single modality.
 *
 * @param source - "eeg" | "voice" | "health"
 * @param wasCorrect - whether that modality's individual prediction matched the user's correction
 */
export function updateModalityAccuracy(source: string, wasCorrect: boolean): void {
  const acc = getModalityAccuracies();
  const entry = acc[source as keyof Pick<ModalityAccuracy, "eeg" | "voice" | "health">];
  if (!entry) return;

  entry.total += 1;
  if (wasCorrect) entry.correct += 1;
  entry.accuracy = entry.total > 0 ? entry.correct / entry.total : 0;

  acc.lastUpdated = new Date().toISOString();

  try {
    sbSaveGeneric(ACCURACY_STORAGE_KEY, acc);
  } catch { /* localStorage full or unavailable */ }
}

// ── Agreement score ────────────────────────────────────────────────────────

/**
 * Compute how much the modalities agree on emotion.
 *
 * For N modalities:
 *   - Count the emotion that appears most often (maxCount)
 *   - agreement = maxCount / N
 *
 * Examples:
 *   3 agree  → 3/3 = 1.0
 *   2 of 3   → 2/3 ≈ 0.67
 *   all different → 1/3 ≈ 0.33
 *   2 agree  → 2/2 = 1.0
 *   2 disagree → 1/2 = 0.5
 */
function computeAgreement(inputs: ModalityInput[]): number {
  if (inputs.length <= 1) return 1.0;

  const counts: Record<string, number> = {};
  for (const input of inputs) {
    counts[input.emotion] = (counts[input.emotion] ?? 0) + 1;
  }
  const maxCount = Math.max(...Object.values(counts));
  return maxCount / inputs.length;
}

/**
 * Find the majority emotion: the emotion most modalities agree on.
 * Returns null if no clear majority (all different).
 */
function findMajorityEmotion(inputs: ModalityInput[]): string | null {
  const counts: Record<string, number> = {};
  for (const input of inputs) {
    counts[input.emotion] = (counts[input.emotion] ?? 0) + 1;
  }

  let maxEmotion: string | null = null;
  let maxCount = 0;
  for (const [emotion, count] of Object.entries(counts)) {
    if (count > maxCount) {
      maxCount = count;
      maxEmotion = emotion;
    }
  }

  // Only a majority if at least 2 sources agree (and it's more than any other)
  if (maxCount >= 2) return maxEmotion;
  return null;
}

// ── Core fusion ────────────────────────────────────────────────────────────

/**
 * Fuse 1-3 modality inputs into a single emotion estimate.
 *
 * Weighting: each modality's weight = confidence * learned_multiplier,
 * normalized to sum to 1.
 *
 * Emotion selection: highest-confidence modality's emotion, UNLESS
 * 2+ modalities agree on a different emotion (majority wins).
 *
 * Agreement modulates confidence: low agreement → lower fused confidence.
 *
 * Returns null if inputs is empty.
 */
export function fuseModalities(inputs: ModalityInput[]): FusedResult | null {
  if (inputs.length === 0) return null;

  const multipliers = loadLearnedMultipliers();
  const learned = getModalityAccuracies();

  // Compute effective confidence for each modality.
  // Blend learned accuracy into confidence: 70% learned accuracy + 30% raw confidence.
  // Only activates once a modality has >= ACCURACY_MIN_SAMPLES corrections.
  const effective: { input: ModalityInput; effConf: number; blendedConf: number }[] = inputs.map((input) => {
    const entry = learned[input.source as keyof Pick<ModalityAccuracy, "eeg" | "voice" | "health">];
    const blendedConf = (entry && entry.total >= ACCURACY_MIN_SAMPLES)
      ? 0.7 * entry.accuracy + 0.3 * input.confidence
      : input.confidence;
    return {
      input,
      blendedConf,
      effConf: blendedConf * (multipliers[input.source] ?? 1.0),
    };
  });

  // Normalize weights to sum to 1
  const totalEffConf = effective.reduce((sum, e) => sum + e.effConf, 0);
  const weighted = effective.map((e) => ({
    ...e,
    weight: totalEffConf > 0 ? e.effConf / totalEffConf : 1 / inputs.length,
  }));

  // Weighted averages for continuous values
  let fusedValence = 0;
  let fusedArousal = 0;
  let fusedStress = 0;
  let fusedConfidence = 0;
  const weights: Record<string, number> = {};

  for (const { input, weight } of weighted) {
    fusedValence += weight * input.valence;
    fusedArousal += weight * input.arousal;
    fusedStress += weight * input.stress;
    fusedConfidence += weight * input.confidence;
    weights[input.source] = weight;
  }

  // Clamp to valid ranges
  fusedValence = clip(fusedValence, -1, 1);
  fusedArousal = clip(fusedArousal, 0, 1);
  fusedStress = clip(fusedStress, 0, 1);

  // Agreement score
  const agreement = computeAgreement(inputs);

  // Modulate confidence by agreement: low agreement → lower confidence
  fusedConfidence = clip(fusedConfidence * agreement, 0, 1);

  // Emotion selection: majority wins, else highest effective confidence
  const majorityEmotion = findMajorityEmotion(inputs);
  let fusedEmotion: string;

  if (majorityEmotion) {
    fusedEmotion = majorityEmotion;
  } else {
    // Take emotion from the highest effective confidence modality
    const best = weighted.reduce((a, b) => (a.effConf >= b.effConf ? a : b));
    fusedEmotion = best.input.emotion;
  }

  // Focus: derived from inverse of stress
  const focus = clip(1 - fusedStress, 0, 1);

  return {
    emotion: fusedEmotion,
    valence: fusedValence,
    arousal: fusedArousal,
    stress: fusedStress,
    focus,
    confidence: fusedConfidence,
    sources: inputs.map((i) => i.source),
    weights,
    agreement,
    model_type: "multimodal_fusion",
  };
}

// ── User-adaptive feedback ─────────────────────────────────────────────────

/**
 * A simple mapping from emotion labels to approximate valence values,
 * used to determine which modality was "closest" to the user's correction.
 */
const EMOTION_VALENCE_MAP: Record<string, number> = {
  happy: 0.7,
  excited: 0.6,
  calm: 0.4,
  relaxed: 0.3,
  neutral: 0.0,
  focused: 0.2,
  sad: -0.6,
  angry: -0.5,
  fear: -0.7,
  fearful: -0.7,
  anxious: -0.4,
  surprise: 0.1,
};

/**
 * Record user feedback to adapt per-modality confidence multipliers.
 *
 * When a user corrects an emotion:
 *   1. Find which modality's emotion was closest to the corrected emotion
 *      (by comparing valence distance in the emotion-valence map).
 *   2. Boost the closest modality's multiplier by BOOST_STEP (0.05).
 *   3. Decrease the furthest modality's multiplier by DECAY_STEP (0.02).
 *   4. Persist to localStorage.
 */
export function recordFusionFeedback(
  _fusedEmotion: string,
  userCorrectedEmotion: string,
  sources: ModalityInput[],
): void {
  if (sources.length === 0) return;

  // Update per-modality accuracy: was each source's individual emotion correct?
  for (const s of sources) {
    const wasCorrect = s.emotion === userCorrectedEmotion;
    updateModalityAccuracy(s.source, wasCorrect);
  }

  const targetValence = EMOTION_VALENCE_MAP[userCorrectedEmotion] ?? 0;

  // Score each modality by how close it was to the correction
  const scored = sources.map((s) => {
    const emotionValence = EMOTION_VALENCE_MAP[s.emotion] ?? 0;
    const distance = Math.abs(emotionValence - targetValence);
    return { source: s.source, distance };
  });

  // Sort: closest first
  scored.sort((a, b) => a.distance - b.distance);
  const closest = scored[0].source;
  const furthest = scored[scored.length - 1].source;

  const multipliers = loadLearnedMultipliers();

  // Boost closest, decay furthest
  multipliers[closest] = (multipliers[closest] ?? 1.0) + BOOST_STEP;
  if (closest !== furthest) {
    multipliers[furthest] = Math.max(0.1, (multipliers[furthest] ?? 1.0) - DECAY_STEP);
  }

  saveLearnedMultipliers(multipliers);
}
