/**
 * confidence-calculator.ts — Compute holistic emotion confidence from multiple factors.
 *
 * Research context: Showing confidence scores changes user behavior dramatically.
 * Physician AI override dropped from 87% to 33% when confidence was displayed.
 * "I don't know" builds trust.
 *
 * The confidence score fuses:
 *   - ML model output confidence (primary signal)
 *   - EEG signal quality (from signal-quality.ts)
 *   - Session duration (more data = more reliable)
 *   - Artifact rejection percentage (noisy session = less reliable)
 *   - Multimodal agreement score (sources agree = higher confidence)
 */

export interface ConfidenceFactors {
  modelConfidence: number;     // from ONNX/ML model (0-1)
  signalQuality?: number;      // from signal-quality.ts (0-1)
  sessionDuration?: number;    // seconds of data collected
  artifactPercentage?: number; // fraction of epochs rejected (0-1)
  agreementScore?: number;     // from multimodal fusion (0-1)
}

export interface ConfidenceResult {
  confidence: number;    // 0-1
  label: string;         // "High confidence" | "Moderate" | "Low — take with grain of salt"
  showEmotion: boolean;  // false if confidence < 0.3 (still hides emotion for very low values)
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Compute a holistic confidence score from multiple quality factors.
 *
 * Algorithm:
 *   1. Start with modelConfidence as the base
 *   2. Multiply by signalQuality if available (poor signal = lower confidence)
 *   3. Apply session duration factor: boost if >60s, reduce if <30s
 *   4. Reduce if artifactPercentage > 0.3 (noisy session)
 *   5. Multiply by agreementScore if provided (multimodal sources agree = higher)
 *   6. Clamp to [0, 1]
 */
export function calculateEmotionConfidence(
  factors: ConfidenceFactors,
): ConfidenceResult {
  let confidence = factors.modelConfidence;

  // Signal quality: multiply directly (poor signal = lower confidence)
  if (factors.signalQuality != null) {
    confidence *= factors.signalQuality;
  }

  // Session duration factor:
  //   < 30s:  penalty (multiply by 0.7-1.0 linearly)
  //   30-60s: neutral (no change)
  //   > 60s:  boost (multiply by 1.0-1.15 linearly, capped at 1.15 for 120s+)
  if (factors.sessionDuration != null) {
    if (factors.sessionDuration < 30) {
      const factor = 0.7 + (factors.sessionDuration / 30) * 0.3;
      confidence *= factor;
    } else if (factors.sessionDuration > 60) {
      const boostSeconds = Math.min(factors.sessionDuration - 60, 60); // cap at 120s total
      const factor = 1.0 + (boostSeconds / 60) * 0.15;
      confidence *= factor;
    }
  }

  // Artifact percentage: reduce confidence when too many epochs rejected
  // No penalty below 0.3; linear penalty from 0.3 to 1.0
  if (factors.artifactPercentage != null && factors.artifactPercentage > 0.3) {
    const penalty = 1.0 - (factors.artifactPercentage - 0.3) * 0.7; // at 1.0 artifacts: 0.51x
    confidence *= Math.max(0.3, penalty);
  }

  // Agreement score: multiply directly (low agreement = lower confidence)
  if (factors.agreementScore != null) {
    confidence *= factors.agreementScore;
  }

  confidence = clamp(confidence, 0, 1);

  const label = getConfidenceLabel(confidence);
  const showEmotion = confidence >= 0.3;

  return { confidence, label, showEmotion };
}

function getConfidenceLabel(confidence: number): string {
  if (confidence > 0.7) return "High confidence";
  if (confidence >= 0.4) return "Moderate";
  return "Low — take with grain of salt";
}
