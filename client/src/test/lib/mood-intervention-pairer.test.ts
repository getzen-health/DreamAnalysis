import { describe, it, expect } from "vitest";
import { pairIntervention, type MoodIntervention } from "@/lib/mood-intervention-pairer";

describe("pairIntervention", () => {
  // 1. High stress returns breathing
  it("returns breathing intervention for high stress", () => {
    const result = pairIntervention({ stress: 0.8, valence: 0, focus: 0.5 });
    expect(result.trigger).toBe("high_stress");
    expect(result.suggestion).toContain("breathing");
    expect(result.href).toBe("/biofeedback");
  });

  // 2. Low mood (negative valence) returns companion
  it("returns companion for low mood (negative valence)", () => {
    const result = pairIntervention({ stress: 0.3, valence: -0.5, focus: 0.5 });
    expect(result.trigger).toBe("low_mood");
    expect(result.href).toBe("/ai-companion");
  });

  // 3. Low focus returns neurofeedback
  it("returns neurofeedback for low focus", () => {
    const result = pairIntervention({ stress: 0.2, valence: 0.1, focus: 0.2 });
    expect(result.trigger).toBe("low_focus");
    expect(result.suggestion).toContain("focus");
    expect(result.href).toBe("/neurofeedback");
  });

  // 4. Angry returns grounding
  it("returns grounding for angry emotion", () => {
    const result = pairIntervention({ stress: 0.4, valence: -0.1, focus: 0.5, emotion: "angry" });
    expect(result.trigger).toBe("anger");
    expect(result.suggestion).toContain("Ground");
    expect(result.href).toBe("/biofeedback");
  });

  // 5. Fear/anxiety returns box breathing
  it("returns box breathing for fear/anxiety", () => {
    for (const emotion of ["fear", "anxiety", "anxious", "fearful"]) {
      const result = pairIntervention({ stress: 0.4, focus: 0.5, emotion });
      expect(result.trigger).toBe("anxiety");
      expect(result.suggestion).toContain("Box breathing");
      expect(result.href).toBe("/biofeedback");
    }
  });

  // 6. Sad returns AI companion
  it("returns AI companion for sadness", () => {
    const result = pairIntervention({ stress: 0.3, valence: -0.1, focus: 0.5, emotion: "sad" });
    expect(result.trigger).toBe("sadness");
    expect(result.suggestion).toContain("companion");
    expect(result.href).toBe("/ai-companion");
  });

  // 7. Neutral returns positive action
  it("returns positive action for neutral state", () => {
    const result = pairIntervention({ stress: 0.3, valence: 0.1, focus: 0.6, emotion: "neutral" });
    expect(result.trigger).toBe("neutral");
    expect(result.suggestion).toContain("gratitude");
  });

  // 8. Always returns something (never null/undefined)
  it("always returns an intervention — never null", () => {
    const states = [
      {},
      { stress: 0 },
      { valence: 0 },
      { focus: 0 },
      { emotion: "happy" },
      { emotion: "surprise" },
      { emotion: "" },
      { stress: 0.5, valence: 0.5, focus: 0.5 },
      { stress: 1, valence: -1, focus: 0, emotion: "angry" },
    ];
    for (const state of states) {
      const result = pairIntervention(state);
      expect(result).toBeTruthy();
      expect(result.trigger).toBeTruthy();
      expect(result.suggestion).toBeTruthy();
      expect(result.action).toBeTruthy();
      expect(result.href).toBeTruthy();
      expect(result.duration).toBeTruthy();
    }
  });

  // 9. High stress takes priority over emotion
  it("prioritizes high stress over emotion label", () => {
    const result = pairIntervention({ stress: 0.9, emotion: "sad" });
    expect(result.trigger).toBe("high_stress");
  });

  // 10. Frustrated maps to anger
  it("treats frustrated as anger", () => {
    const result = pairIntervention({ emotion: "frustrated" });
    expect(result.trigger).toBe("anger");
  });
});
