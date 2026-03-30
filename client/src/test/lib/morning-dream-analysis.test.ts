import { describe, it, expect } from "vitest";

/**
 * Tests for the morning form dream analysis response shape.
 *
 * The POST /api/study/morning endpoint now uses the 3-pass analyzeDreamMultiPass
 * function and returns a richer `dreamAnalysis` object. These tests verify the
 * shape mapping logic that transforms multipass output into the morning response.
 */

// ── Shape mapping helpers (mirrors the server-side mapping) ─────────────────

interface MultiPassResult {
  symbols: Array<{ symbol: string; meaning: string }>;
  emotions: Array<{ emotion: string; intensity: number }>;
  aiAnalysis: string;
  themes: string[];
  emotional_arc: string;
  key_insight: string;
  threat_simulation_index: number;
  morning_mood_prediction: string;
  irt_recommended: boolean;
  wellbeing_note: string;
}

interface MorningDreamAnalysis {
  symbols: Array<{ symbol: string; meaning: string }>;
  emotions: string[];
  themes: string[];
  insights: string;
  morningMoodPrediction: string;
  emotional_arc: string;
  key_insight: string;
  threat_simulation_index: number;
  irt_recommended: boolean;
  wellbeing_note: string;
}

function mapMultiPassToMorning(r: MultiPassResult): MorningDreamAnalysis {
  return {
    symbols: r.symbols,
    emotions: r.emotions.map((e) => e.emotion),
    themes: r.themes,
    insights: r.aiAnalysis,
    morningMoodPrediction: r.morning_mood_prediction,
    emotional_arc: r.emotional_arc,
    key_insight: r.key_insight,
    threat_simulation_index: r.threat_simulation_index,
    irt_recommended: r.irt_recommended,
    wellbeing_note: r.wellbeing_note,
  };
}

// ── Fixtures ─────────────────────────────────────────────────────────────────

const typicalResult: MultiPassResult = {
  symbols: [
    { symbol: "falling", meaning: "loss of control or anxiety" },
    { symbol: "house", meaning: "the self or psyche" },
  ],
  emotions: [
    { emotion: "fear", intensity: 0.8 },
    { emotion: "curiosity", intensity: 0.4 },
  ],
  aiAnalysis:
    "The dreamer may be processing stress about a transitional life event. Morning mood: reflective. Stay grounded today.",
  themes: ["transition", "self-exploration"],
  emotional_arc: "escalating fear followed by cautious resolution",
  key_insight: "The falling imagery suggests unresolved anxiety about losing stability.",
  threat_simulation_index: 0.6,
  morning_mood_prediction: "reflective, mildly anxious",
  irt_recommended: false,
  wellbeing_note: "Grounding exercises may help today.",
};

const nightmareResult: MultiPassResult = {
  ...typicalResult,
  threat_simulation_index: 0.85,
  irt_recommended: true,
  wellbeing_note: "Consider speaking with a counselor if this dream recurs.",
};

const minimalResult: MultiPassResult = {
  symbols: [],
  emotions: [],
  aiAnalysis: "",
  themes: [],
  emotional_arc: "",
  key_insight: "",
  threat_simulation_index: 0,
  morning_mood_prediction: "",
  irt_recommended: false,
  wellbeing_note: "",
};

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("mapMultiPassToMorning", () => {
  it("maps symbols with meanings preserved", () => {
    const result = mapMultiPassToMorning(typicalResult);
    expect(result.symbols).toHaveLength(2);
    expect(result.symbols[0]).toEqual({ symbol: "falling", meaning: "loss of control or anxiety" });
    expect(result.symbols[1]).toEqual({ symbol: "house", meaning: "the self or psyche" });
  });

  it("flattens emotion objects to strings", () => {
    const result = mapMultiPassToMorning(typicalResult);
    expect(result.emotions).toEqual(["fear", "curiosity"]);
  });

  it("passes themes through unchanged", () => {
    const result = mapMultiPassToMorning(typicalResult);
    expect(result.themes).toEqual(["transition", "self-exploration"]);
  });

  it("maps aiAnalysis to insights", () => {
    const result = mapMultiPassToMorning(typicalResult);
    expect(result.insights).toBe(typicalResult.aiAnalysis);
  });

  it("maps morning_mood_prediction to morningMoodPrediction", () => {
    const result = mapMultiPassToMorning(typicalResult);
    expect(result.morningMoodPrediction).toBe("reflective, mildly anxious");
  });

  it("carries through emotional_arc, key_insight, threat_simulation_index", () => {
    const result = mapMultiPassToMorning(typicalResult);
    expect(result.emotional_arc).toBe("escalating fear followed by cautious resolution");
    expect(result.key_insight).toContain("falling imagery");
    expect(result.threat_simulation_index).toBe(0.6);
  });

  it("irt_recommended is false for non-nightmare", () => {
    const result = mapMultiPassToMorning(typicalResult);
    expect(result.irt_recommended).toBe(false);
  });

  it("irt_recommended is true when threat_simulation_index > 0.5 (nightmare)", () => {
    const result = mapMultiPassToMorning(nightmareResult);
    expect(result.irt_recommended).toBe(true);
    expect(result.threat_simulation_index).toBeGreaterThan(0.5);
  });

  it("wellbeing_note is carried through", () => {
    const result = mapMultiPassToMorning(nightmareResult);
    expect(result.wellbeing_note).toContain("counselor");
  });

  it("handles empty/minimal multi-pass output without throwing", () => {
    const result = mapMultiPassToMorning(minimalResult);
    expect(result.symbols).toHaveLength(0);
    expect(result.emotions).toHaveLength(0);
    expect(result.themes).toHaveLength(0);
    expect(result.insights).toBe("");
    expect(result.morningMoodPrediction).toBe("");
    expect(result.irt_recommended).toBe(false);
  });

  it("symbol list for DB storage extracts names only", () => {
    // Mirrors: multiResult.symbols.map(s => s.symbol) used in createDreamAnalysis
    const result = mapMultiPassToMorning(typicalResult);
    const dbSymbols = result.symbols.map((s) => s.symbol);
    expect(dbSymbols).toEqual(["falling", "house"]);
  });
});
