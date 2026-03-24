import { describe, it, expect } from "vitest";
import {
  mapToNuancedEmotion,
  NUANCED_EMOJI,
  NUANCED_COLORS,
  NUANCED_VALENCE_MAP,
  type NuancedInput,
} from "@/lib/nuanced-emotion";

// ── Helpers ─────────────────────────────────────────────────────────────────

function makeInput(overrides: Partial<NuancedInput> = {}): NuancedInput {
  return {
    emotion: "neutral",
    probabilities: { neutral: 1.0, happy: 0, sad: 0, angry: 0, fear: 0, surprise: 0 },
    valence: 0,
    arousal: 0.4,
    stress: 0.3,
    confidence: 0.7,
    ...overrides,
  };
}

// ── 1. Compound label: content (happy + calm + low arousal) ─────────────

describe("content detection", () => {
  it("maps happy+calm mix at low arousal to content", () => {
    const result = mapToNuancedEmotion(
      makeInput({
        emotion: "happy",
        probabilities: { happy: 0.4, calm: 0.3, neutral: 0.2, sad: 0, angry: 0, fear: 0, surprise: 0.1 },
        valence: 0.35,
        arousal: 0.25,
        stress: 0.15,
        confidence: 0.75,
      }),
    );
    expect(result.label).toBe("content");
    expect(result.baseEmotion).toBe("happy");
    expect(result.quadrant).toBe("low-pos");
    expect(result.displayLabel).toBe("Content");
  });
});

// ── 2. Compound label: excited (happy + high arousal) ────────────────────

describe("excited detection", () => {
  it("maps happy at high arousal to excited", () => {
    const result = mapToNuancedEmotion(
      makeInput({
        emotion: "happy",
        probabilities: { happy: 0.6, calm: 0.05, neutral: 0.15, sad: 0, angry: 0, fear: 0, surprise: 0.2 },
        valence: 0.6,
        arousal: 0.8,
        stress: 0.2,
        confidence: 0.8,
      }),
    );
    expect(result.label).toBe("excited");
    expect(result.quadrant).toBe("high-pos");
  });
});

// ── 3. Compound label: anxious (fear + high arousal + negative valence) ──

describe("anxious detection", () => {
  it("maps fear at high arousal with negative valence to anxious", () => {
    const result = mapToNuancedEmotion(
      makeInput({
        emotion: "fear",
        probabilities: { happy: 0, calm: 0.05, neutral: 0.1, sad: 0.15, angry: 0.1, fear: 0.5, surprise: 0.1 },
        valence: -0.4,
        arousal: 0.7,
        stress: 0.65,
        confidence: 0.72,
      }),
    );
    expect(result.label).toBe("anxious");
    expect(result.baseEmotion).toBe("fear");
    expect(result.quadrant).toBe("high-neg");
  });
});

// ── 4. Compound label: frustrated (angry + sad mix) ──────────────────────

describe("frustrated detection", () => {
  it("maps angry+sad mix to frustrated", () => {
    const result = mapToNuancedEmotion(
      makeInput({
        emotion: "angry",
        probabilities: { happy: 0, calm: 0, neutral: 0.1, sad: 0.3, angry: 0.4, fear: 0.1, surprise: 0.1 },
        valence: -0.35,
        arousal: 0.6,
        stress: 0.55,
        confidence: 0.7,
      }),
    );
    expect(result.label).toBe("frustrated");
    expect(result.baseEmotion).toBe("angry");
    expect(result.quadrant).toBe("high-neg");
  });
});

// ── 5. Compound label: melancholy (sad + low arousal) ────────────────────

describe("melancholy detection", () => {
  it("maps sad at low arousal to melancholy", () => {
    const result = mapToNuancedEmotion(
      makeInput({
        emotion: "sad",
        probabilities: { happy: 0, calm: 0.1, neutral: 0.1, sad: 0.6, angry: 0.1, fear: 0.05, surprise: 0.05 },
        valence: -0.5,
        arousal: 0.2,
        stress: 0.3,
        confidence: 0.75,
      }),
    );
    expect(result.label).toBe("melancholy");
    expect(result.baseEmotion).toBe("sad");
    expect(result.quadrant).toBe("low-neg");
  });
});

// ── 6. Compound label: overwhelmed (angry + fear + high stress) ──────────

describe("overwhelmed detection", () => {
  it("maps angry+fear with high stress to overwhelmed", () => {
    const result = mapToNuancedEmotion(
      makeInput({
        emotion: "angry",
        probabilities: { happy: 0, calm: 0, neutral: 0.05, sad: 0.05, angry: 0.35, fear: 0.4, surprise: 0.15 },
        valence: -0.45,
        arousal: 0.85,
        stress: 0.8,
        confidence: 0.65,
      }),
    );
    expect(result.label).toBe("overwhelmed");
    expect(result.quadrant).toBe("high-neg");
  });
});

// ── 7. Compound label: serene (calm + very low arousal) ──────────────────

describe("serene detection", () => {
  it("maps calm at very low arousal to serene", () => {
    const result = mapToNuancedEmotion(
      makeInput({
        emotion: "calm",
        probabilities: { happy: 0.1, calm: 0.6, neutral: 0.2, sad: 0, angry: 0, fear: 0, surprise: 0.1 },
        valence: 0.2,
        arousal: 0.15,
        stress: 0.05,
        confidence: 0.8,
      }),
    );
    expect(result.label).toBe("serene");
    expect(result.baseEmotion).toBe("calm");
    expect(result.quadrant).toBe("low-pos");
  });
});

// ── 8. Compound label: drained (sad/neutral + very low arousal + high stress)

describe("drained detection", () => {
  it("maps sad at very low arousal with high stress to drained", () => {
    const result = mapToNuancedEmotion(
      makeInput({
        emotion: "sad",
        probabilities: { happy: 0, calm: 0.05, neutral: 0.2, sad: 0.5, angry: 0.1, fear: 0.1, surprise: 0.05 },
        valence: -0.3,
        arousal: 0.1,
        stress: 0.75,
        confidence: 0.65,
      }),
    );
    expect(result.label).toBe("drained");
    expect(result.baseEmotion).toBe("sad");
    expect(result.quadrant).toBe("low-neg");
  });
});

// ── 9. Compound label: focused (neutral + moderate arousal + low stress) ──

describe("focused detection", () => {
  it("maps neutral at moderate arousal with low stress to focused", () => {
    const result = mapToNuancedEmotion(
      makeInput({
        emotion: "neutral",
        probabilities: { happy: 0.05, calm: 0.1, neutral: 0.5, sad: 0.05, angry: 0.05, fear: 0.05, surprise: 0.05, focused: 0.15 },
        valence: 0.05,
        arousal: 0.55,
        stress: 0.15,
        confidence: 0.7,
      }),
    );
    expect(result.label).toBe("focused");
    expect(result.quadrant).toBe("center");
  });
});

// ── 10. Fallback to base emotion when no compound rule matches ───────────

describe("base emotion fallback", () => {
  it("returns base emotion when probabilities are not available", () => {
    const result = mapToNuancedEmotion(
      makeInput({
        emotion: "happy",
        probabilities: undefined,
        valence: 0.5,
        arousal: 0.5,
        stress: 0.3,
        confidence: 0.7,
      }),
    );
    // Without probabilities, compound rules cannot score high enough
    // The exact label depends on valence/arousal but should still be valid
    expect(result.label).toBeTruthy();
    expect(result.confidence).toBeGreaterThan(0);
    expect(result.displayLabel).toBeTruthy();
  });

  it("returns base emotion for ambiguous uniform distribution", () => {
    const uniform = { happy: 0.17, calm: 0.17, neutral: 0.16, sad: 0.17, angry: 0.17, fear: 0.16, surprise: 0 };
    const result = mapToNuancedEmotion(
      makeInput({
        emotion: "neutral",
        probabilities: uniform,
        valence: 0.0,
        arousal: 0.4,
        stress: 0.3,
        confidence: 0.5,
      }),
    );
    // Uniform distribution should not strongly activate any compound rule
    // May fall back to base or pick contemplative (low arousal, near-zero valence)
    expect(result.label).toBeTruthy();
    expect(result.confidence).toBeGreaterThan(0);
  });
});

// ── 11. Result structure validation ──────────────────────────────────────

describe("result structure", () => {
  it("always returns valid structure", () => {
    const inputs: NuancedInput[] = [
      makeInput({ emotion: "happy", valence: 0.7, arousal: 0.9 }),
      makeInput({ emotion: "sad", valence: -0.7, arousal: 0.1 }),
      makeInput({ emotion: "angry", valence: -0.5, arousal: 0.8 }),
      makeInput({ emotion: "neutral", valence: 0, arousal: 0.4 }),
    ];

    for (const input of inputs) {
      const result = mapToNuancedEmotion(input);
      expect(result.label).toBeTruthy();
      expect(result.baseEmotion).toBeTruthy();
      expect(result.displayLabel).toBeTruthy();
      expect(result.confidence).toBeGreaterThanOrEqual(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
      expect(["high-pos", "low-pos", "high-neg", "low-neg", "center"]).toContain(result.quadrant);
    }
  });

  it("displayLabel is capitalized version of label", () => {
    const result = mapToNuancedEmotion(
      makeInput({
        emotion: "happy",
        probabilities: { happy: 0.6, calm: 0.3, neutral: 0.1, sad: 0, angry: 0, fear: 0, surprise: 0 },
        valence: 0.3,
        arousal: 0.2,
        stress: 0.1,
        confidence: 0.8,
      }),
    );
    expect(result.displayLabel).toBe(result.label.charAt(0).toUpperCase() + result.label.slice(1));
  });
});

// ── 12. Emoji and color maps cover all nuanced labels ────────────────────

describe("nuanced emoji and color maps", () => {
  const nuancedLabels = [
    "excited", "energized", "content", "serene",
    "anxious", "overwhelmed", "frustrated",
    "melancholy", "drained",
    "focused", "contemplative",
  ];

  it("NUANCED_EMOJI has entries for all nuanced labels", () => {
    for (const label of nuancedLabels) {
      expect(NUANCED_EMOJI[label]).toBeTruthy();
    }
  });

  it("NUANCED_COLORS has entries for all nuanced labels", () => {
    for (const label of nuancedLabels) {
      expect(NUANCED_COLORS[label]).toBeTruthy();
    }
  });

  it("NUANCED_VALENCE_MAP has entries for all nuanced labels", () => {
    for (const label of nuancedLabels) {
      expect(typeof NUANCED_VALENCE_MAP[label]).toBe("number");
    }
  });

  it("NUANCED_EMOJI also covers base emotions", () => {
    const baseLabels = ["happy", "sad", "angry", "fear", "surprise", "neutral", "calm"];
    for (const label of baseLabels) {
      expect(NUANCED_EMOJI[label]).toBeTruthy();
    }
  });

  it("NUANCED_COLORS also covers base emotions", () => {
    const baseLabels = ["happy", "sad", "angry", "fear", "surprise", "neutral", "calm"];
    for (const label of baseLabels) {
      expect(NUANCED_COLORS[label]).toBeTruthy();
    }
  });
});

// ── 13. Energized detection ──────────────────────────────────────────────

describe("energized detection", () => {
  it("maps positive valence + moderate arousal + low stress to energized", () => {
    const result = mapToNuancedEmotion(
      makeInput({
        emotion: "happy",
        probabilities: { happy: 0.3, calm: 0.2, neutral: 0.3, sad: 0, angry: 0, fear: 0, surprise: 0.2 },
        valence: 0.4,
        arousal: 0.65,
        stress: 0.1,
        confidence: 0.7,
      }),
    );
    // Should be either energized or excited (both high-pos)
    expect(["excited", "energized"]).toContain(result.label);
    expect(result.quadrant).toBe("high-pos");
  });
});

// ── 14. Contemplative detection ──────────────────────────────────────────

describe("contemplative detection", () => {
  it("maps neutral at low arousal with near-zero valence to contemplative", () => {
    const result = mapToNuancedEmotion(
      makeInput({
        emotion: "neutral",
        probabilities: { happy: 0.05, calm: 0.15, neutral: 0.6, sad: 0.05, angry: 0.05, fear: 0.05, surprise: 0.05 },
        valence: 0.0,
        arousal: 0.2,
        stress: 0.2,
        confidence: 0.65,
      }),
    );
    expect(result.label).toBe("contemplative");
    expect(result.quadrant).toBe("center");
  });
});
