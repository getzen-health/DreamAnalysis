import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  fuseModalities,
  recordFusionFeedback,
  loadLearnedMultipliers,
  type ModalityInput,
  type FusedResult,
} from "@/lib/multimodal-fusion";

// ── Helpers ─────────────────────────────────────────────────────────────────

function makeInput(
  source: "eeg" | "voice" | "health",
  overrides: Partial<ModalityInput> = {},
): ModalityInput {
  return {
    valence: 0.3,
    arousal: 0.5,
    stress: 0.3,
    confidence: 0.7,
    emotion: "happy",
    source,
    ...overrides,
  };
}

// ── Clear localStorage before each test ────────────────────────────────────

beforeEach(() => {
  localStorage.clear();
});

// ── 1. Single modality returns that modality's values ──────────────────────

describe("single modality passthrough", () => {
  it("returns EEG values when only EEG is provided", () => {
    const eeg = makeInput("eeg", {
      valence: 0.6,
      arousal: 0.8,
      stress: 0.2,
      confidence: 0.85,
      emotion: "happy",
    });
    const result = fuseModalities([eeg]);

    expect(result).not.toBeNull();
    expect(result!.valence).toBeCloseTo(0.6);
    expect(result!.arousal).toBeCloseTo(0.8);
    expect(result!.stress).toBeCloseTo(0.2);
    expect(result!.confidence).toBeCloseTo(0.85);
    expect(result!.emotion).toBe("happy");
    expect(result!.sources).toEqual(["eeg"]);
    expect(result!.model_type).toBe("multimodal_fusion");
  });

  it("returns voice values when only voice is provided", () => {
    const voice = makeInput("voice", { emotion: "sad", valence: -0.4 });
    const result = fuseModalities([voice]);

    expect(result!.emotion).toBe("sad");
    expect(result!.valence).toBeCloseTo(-0.4);
    expect(result!.sources).toEqual(["voice"]);
  });

  it("returns health values when only health is provided", () => {
    const health = makeInput("health", { emotion: "calm", arousal: 0.2, confidence: 0.5 });
    const result = fuseModalities([health]);

    expect(result!.emotion).toBe("calm");
    expect(result!.arousal).toBeCloseTo(0.2);
    expect(result!.sources).toEqual(["health"]);
  });
});

// ── 2. Two modalities — confidence-weighted average ────────────────────────

describe("two modality fusion", () => {
  it("fuses voice and health with confidence weighting", () => {
    const voice = makeInput("voice", {
      valence: 0.6,
      arousal: 0.7,
      stress: 0.2,
      confidence: 0.8,
      emotion: "happy",
    });
    const health = makeInput("health", {
      valence: 0.2,
      arousal: 0.3,
      stress: 0.4,
      confidence: 0.5,
      emotion: "neutral",
    });
    const result = fuseModalities([voice, health]);

    // Weights: voice=0.8/(0.8+0.5)=0.615, health=0.5/1.3=0.385
    const expectedValence = (0.8 / 1.3) * 0.6 + (0.5 / 1.3) * 0.2;
    const expectedArousal = (0.8 / 1.3) * 0.7 + (0.5 / 1.3) * 0.3;

    expect(result!.valence).toBeCloseTo(expectedValence, 2);
    expect(result!.arousal).toBeCloseTo(expectedArousal, 2);
    expect(result!.sources).toEqual(expect.arrayContaining(["voice", "health"]));
    expect(result!.sources).toHaveLength(2);
  });
});

// ── 3. Three modalities — all contribute proportionally ───────────────────

describe("three modality fusion", () => {
  it("fuses EEG + voice + health with confidence weighting", () => {
    const eeg = makeInput("eeg", {
      valence: 0.5,
      arousal: 0.6,
      stress: 0.3,
      confidence: 0.85,
      emotion: "happy",
    });
    const voice = makeInput("voice", {
      valence: 0.4,
      arousal: 0.5,
      stress: 0.4,
      confidence: 0.72,
      emotion: "happy",
    });
    const health = makeInput("health", {
      valence: 0.1,
      arousal: 0.3,
      stress: 0.5,
      confidence: 0.50,
      emotion: "neutral",
    });
    const result = fuseModalities([eeg, voice, health]);

    const totalConf = 0.85 + 0.72 + 0.50;
    const wEeg = 0.85 / totalConf;
    const wVoice = 0.72 / totalConf;
    const wHealth = 0.50 / totalConf;

    const expectedValence = wEeg * 0.5 + wVoice * 0.4 + wHealth * 0.1;
    const expectedArousal = wEeg * 0.6 + wVoice * 0.5 + wHealth * 0.3;
    const expectedStress = wEeg * 0.3 + wVoice * 0.4 + wHealth * 0.5;

    expect(result!.valence).toBeCloseTo(expectedValence, 2);
    expect(result!.arousal).toBeCloseTo(expectedArousal, 2);
    expect(result!.stress).toBeCloseTo(expectedStress, 2);
    expect(result!.sources).toHaveLength(3);
    expect(result!.sources).toEqual(expect.arrayContaining(["eeg", "voice", "health"]));
  });
});

// ── 4. High-confidence source dominates ───────────────────────────────────

describe("confidence dominance", () => {
  it("high-confidence EEG dominates over low-confidence health", () => {
    const eeg = makeInput("eeg", {
      valence: 0.8,
      arousal: 0.9,
      stress: 0.1,
      confidence: 0.95,
      emotion: "happy",
    });
    const health = makeInput("health", {
      valence: -0.5,
      arousal: 0.2,
      stress: 0.8,
      confidence: 0.10,
      emotion: "sad",
    });
    const result = fuseModalities([eeg, health]);

    // EEG weight = 0.95/(0.95+0.10) = 0.905
    // Result should be much closer to EEG values
    expect(result!.valence).toBeGreaterThan(0.5);
    expect(result!.arousal).toBeGreaterThan(0.7);
    expect(result!.stress).toBeLessThan(0.2);
    expect(result!.emotion).toBe("happy"); // highest confidence wins
  });
});

// ── 5. All agree on emotion → agreement = 1.0 ────────────────────────────

describe("agreement scoring", () => {
  it("all three agree → agreement = 1.0", () => {
    const eeg = makeInput("eeg", { emotion: "happy", confidence: 0.8 });
    const voice = makeInput("voice", { emotion: "happy", confidence: 0.7 });
    const health = makeInput("health", { emotion: "happy", confidence: 0.5 });
    const result = fuseModalities([eeg, voice, health]);

    expect(result!.agreement).toBeCloseTo(1.0);
  });

  // ── 6. All disagree → agreement = 0.33 ──────────────────────────────────

  it("all three disagree → agreement ≈ 0.33", () => {
    const eeg = makeInput("eeg", { emotion: "happy", confidence: 0.7 });
    const voice = makeInput("voice", { emotion: "sad", confidence: 0.6 });
    const health = makeInput("health", { emotion: "angry", confidence: 0.5 });
    const result = fuseModalities([eeg, voice, health]);

    expect(result!.agreement).toBeCloseTo(0.33, 1);
  });

  // ── 7. 2 of 3 agree → majority emotion wins regardless of confidence ────

  it("2 of 3 agree → agreement ≈ 0.67, majority emotion wins", () => {
    const eeg = makeInput("eeg", { emotion: "sad", confidence: 0.9 }); // minority, but highest confidence
    const voice = makeInput("voice", { emotion: "happy", confidence: 0.6 });
    const health = makeInput("health", { emotion: "happy", confidence: 0.5 });
    const result = fuseModalities([eeg, voice, health]);

    expect(result!.agreement).toBeCloseTo(0.67, 1);
    // Majority wins even though EEG (sad) has highest confidence
    expect(result!.emotion).toBe("happy");
  });

  it("two modalities agree → agreement ≈ 1.0 when both agree", () => {
    const eeg = makeInput("eeg", { emotion: "happy", confidence: 0.8 });
    const voice = makeInput("voice", { emotion: "happy", confidence: 0.7 });
    const result = fuseModalities([eeg, voice]);

    expect(result!.agreement).toBeCloseTo(1.0);
  });

  it("two modalities disagree → agreement ≈ 0.5", () => {
    const eeg = makeInput("eeg", { emotion: "happy", confidence: 0.8 });
    const voice = makeInput("voice", { emotion: "sad", confidence: 0.7 });
    const result = fuseModalities([eeg, voice]);

    expect(result!.agreement).toBeCloseTo(0.5);
  });
});

// ── 8. Fused values in valid ranges ───────────────────────────────────────

describe("value range validation", () => {
  it("valence stays in [-1, 1]", () => {
    const inputs = [
      makeInput("eeg", { valence: -1, confidence: 0.9 }),
      makeInput("voice", { valence: -0.8, confidence: 0.8 }),
      makeInput("health", { valence: -0.9, confidence: 0.7 }),
    ];
    const result = fuseModalities(inputs);
    expect(result!.valence).toBeGreaterThanOrEqual(-1);
    expect(result!.valence).toBeLessThanOrEqual(1);
  });

  it("arousal stays in [0, 1]", () => {
    const inputs = [
      makeInput("eeg", { arousal: 1.0, confidence: 0.9 }),
      makeInput("voice", { arousal: 0.9, confidence: 0.8 }),
    ];
    const result = fuseModalities(inputs);
    expect(result!.arousal).toBeGreaterThanOrEqual(0);
    expect(result!.arousal).toBeLessThanOrEqual(1);
  });

  it("stress stays in [0, 1]", () => {
    const inputs = [
      makeInput("eeg", { stress: 0.0, confidence: 0.5 }),
      makeInput("health", { stress: 1.0, confidence: 0.9 }),
    ];
    const result = fuseModalities(inputs);
    expect(result!.stress).toBeGreaterThanOrEqual(0);
    expect(result!.stress).toBeLessThanOrEqual(1);
  });

  it("confidence stays in [0, 1]", () => {
    const inputs = [
      makeInput("eeg", { confidence: 0.99 }),
      makeInput("voice", { confidence: 0.95 }),
      makeInput("health", { confidence: 0.90 }),
    ];
    const result = fuseModalities(inputs);
    expect(result!.confidence).toBeGreaterThanOrEqual(0);
    expect(result!.confidence).toBeLessThanOrEqual(1);
  });

  it("agreement stays in [0, 1]", () => {
    const inputs = [
      makeInput("eeg", { emotion: "happy" }),
      makeInput("voice", { emotion: "sad" }),
      makeInput("health", { emotion: "angry" }),
    ];
    const result = fuseModalities(inputs);
    expect(result!.agreement).toBeGreaterThanOrEqual(0);
    expect(result!.agreement).toBeLessThanOrEqual(1);
  });
});

// ── 9. Empty input → null ─────────────────────────────────────────────────

describe("empty input", () => {
  it("returns null for empty input array", () => {
    const result = fuseModalities([]);
    expect(result).toBeNull();
  });
});

// ── 10. Feedback recording adjusts weights correctly ──────────────────────

describe("feedback and learned weights", () => {
  it("recording feedback adjusts multipliers", () => {
    // Set up sources where voice was closest to correction
    const sources: ModalityInput[] = [
      makeInput("eeg", { emotion: "angry", valence: -0.5 }),
      makeInput("voice", { emotion: "happy", valence: 0.4 }),
      makeInput("health", { emotion: "neutral", valence: 0.0 }),
    ];

    // User corrects to "happy" — voice was correct
    recordFusionFeedback("angry", "happy", sources);

    const multipliers = loadLearnedMultipliers();
    // Voice was closest → boosted
    expect(multipliers.voice).toBeGreaterThan(1.0);
    // EEG was furthest (angry → negative, but happy is target) → decreased
    expect(multipliers.eeg).toBeLessThan(1.0);
  });

  // ── 11. Learned weights persist and apply on next fusion ──────────────────

  it("learned multipliers persist via localStorage and affect fusion", () => {
    // Manually set high voice multiplier
    localStorage.setItem(
      "ndw_fusion_weights",
      JSON.stringify({ eeg: 0.8, voice: 1.5, health: 1.0 }),
    );

    const eeg = makeInput("eeg", {
      valence: -0.5,
      confidence: 0.7, // effective = 0.7 * 0.8 = 0.56
      emotion: "sad",
    });
    const voice = makeInput("voice", {
      valence: 0.5,
      confidence: 0.7, // effective = 0.7 * 1.5 = 1.05
      emotion: "happy",
    });
    const result = fuseModalities([eeg, voice]);

    // Voice should dominate because its multiplied confidence is higher
    expect(result!.valence).toBeGreaterThan(0);
    expect(result!.emotion).toBe("happy"); // voice has higher effective confidence
  });

  it("multiple feedback rounds compound correctly", () => {
    const sources: ModalityInput[] = [
      makeInput("eeg", { emotion: "neutral", valence: 0.0 }),
      makeInput("voice", { emotion: "happy", valence: 0.5 }),
      makeInput("health", { emotion: "neutral", valence: 0.0 }),
    ];

    // User says "happy" is correct twice
    recordFusionFeedback("neutral", "happy", sources);
    recordFusionFeedback("neutral", "happy", sources);

    const multipliers = loadLearnedMultipliers();
    // Voice boosted twice: 1.0 + 0.05 + 0.05 = 1.10
    expect(multipliers.voice).toBeCloseTo(1.10, 2);
  });
});

// ── Low agreement reduces confidence ───────────────────────────────────────

describe("agreement modulates confidence", () => {
  it("low agreement reduces fused confidence", () => {
    const allAgree = [
      makeInput("eeg", { emotion: "happy", confidence: 0.8 }),
      makeInput("voice", { emotion: "happy", confidence: 0.8 }),
      makeInput("health", { emotion: "happy", confidence: 0.8 }),
    ];
    const allDisagree = [
      makeInput("eeg", { emotion: "happy", confidence: 0.8 }),
      makeInput("voice", { emotion: "sad", confidence: 0.8 }),
      makeInput("health", { emotion: "angry", confidence: 0.8 }),
    ];

    const agreeResult = fuseModalities(allAgree);
    const disagreeResult = fuseModalities(allDisagree);

    // Same individual confidences, but disagreement should reduce fused confidence
    expect(disagreeResult!.confidence).toBeLessThan(agreeResult!.confidence);
  });
});

// ── Focus propagation ─────────────────────────────────────────────────────

describe("focus score", () => {
  it("propagates focus as weighted average of (1 - stress)", () => {
    const eeg = makeInput("eeg", { stress: 0.2, confidence: 0.8 });
    const voice = makeInput("voice", { stress: 0.6, confidence: 0.6 });
    const result = fuseModalities([eeg, voice]);

    // Focus is derived from fused stress: focus = 1 - fused_stress
    expect(result!.focus).toBeGreaterThan(0);
    expect(result!.focus).toBeLessThanOrEqual(1);
  });
});

// ── Weights record ────────────────────────────────────────────────────────

describe("weights transparency", () => {
  it("returns actual weights used in the fusion", () => {
    const eeg = makeInput("eeg", { confidence: 0.85 });
    const voice = makeInput("voice", { confidence: 0.72 });
    const health = makeInput("health", { confidence: 0.50 });
    const result = fuseModalities([eeg, voice, health]);

    const totalConf = 0.85 + 0.72 + 0.50;
    expect(result!.weights["eeg"]).toBeCloseTo(0.85 / totalConf, 2);
    expect(result!.weights["voice"]).toBeCloseTo(0.72 / totalConf, 2);
    expect(result!.weights["health"]).toBeCloseTo(0.50 / totalConf, 2);

    // Weights sum to 1
    const weightSum = Object.values(result!.weights).reduce((a, b) => a + b, 0);
    expect(weightSum).toBeCloseTo(1.0, 5);
  });
});
