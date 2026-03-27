import { describe, it, expect, beforeEach, vi } from "vitest";

const mockFrom = vi.fn();
const mockUpsert = vi.fn().mockResolvedValue({ error: null });
mockFrom.mockReturnValue({ upsert: mockUpsert });
const mockSupabase = { from: mockFrom };

vi.mock("@/lib/supabase-browser", () => ({
  getSupabase: vi.fn().mockResolvedValue(null),
}));

import { getSupabase } from "@/lib/supabase-browser";
import { EmotionTaxonomy } from "@/lib/insight-engine/emotion-taxonomy";

beforeEach(() => { localStorage.clear(); vi.clearAllMocks(); vi.mocked(getSupabase).mockResolvedValue(null); });

describe("EmotionTaxonomy.getQuadrant", () => {
  it("returns ha_pos for high arousal + positive valence", () => {
    const taxonomy = new EmotionTaxonomy("user1");
    expect(taxonomy.getQuadrant(0.6, 0.7)).toBe("ha_pos");
  });
  it("returns ha_neg for high arousal + negative valence", () => {
    const taxonomy = new EmotionTaxonomy("user1");
    expect(taxonomy.getQuadrant(0.3, 0.8)).toBe("ha_neg");
  });
  it("returns la_pos for low arousal + positive valence", () => {
    const taxonomy = new EmotionTaxonomy("user1");
    expect(taxonomy.getQuadrant(0.7, 0.3)).toBe("la_pos");
  });
  it("returns la_neg for low arousal + negative valence", () => {
    const taxonomy = new EmotionTaxonomy("user1");
    expect(taxonomy.getQuadrant(0.4, 0.3)).toBe("la_neg");
  });
});

describe("EmotionTaxonomy.getPresetsForQuadrant", () => {
  it("returns 16 presets for ha_pos quadrant", () => {
    const taxonomy = new EmotionTaxonomy("user1");
    expect(taxonomy.getPresetsForQuadrant("ha_pos")).toHaveLength(16);
    expect(taxonomy.getPresetsForQuadrant("ha_pos")).toContain("excited");
  });
  it("returns 16 presets for la_neg quadrant", () => {
    const taxonomy = new EmotionTaxonomy("user1");
    expect(taxonomy.getPresetsForQuadrant("la_neg")).toContain("sad");
  });
});

describe("EmotionTaxonomy.labelEmotion", () => {
  it("saves personal fingerprint to localStorage when Supabase unavailable", async () => {
    const taxonomy = new EmotionTaxonomy("user1");
    const fp = await taxonomy.labelEmotion("scattered", {
      valence: 0.3, arousal: 0.8, stress_index: 0.7, focus_index: 0.3,
      alpha_power: null, beta_power: null, theta_power: null, frontal_asymmetry: null,
    });
    expect(fp.label).toBe("scattered");
    expect(fp.quadrant).toBe("ha_neg");
    expect(fp.isPersonal).toBe(true);
    const stored = JSON.parse(localStorage.getItem("ndw_emotion_fingerprints") || "[]");
    expect(stored.length).toBe(1);
  });

  it("updates centroid via running average on second label for same emotion", async () => {
    const taxonomy = new EmotionTaxonomy("user1");
    await taxonomy.labelEmotion("scattered", { valence: 0.3, arousal: 0.8, stress_index: 0.7, focus_index: 0.3, alpha_power: null, beta_power: null, theta_power: null, frontal_asymmetry: null });
    await taxonomy.labelEmotion("scattered", { valence: 0.4, arousal: 0.9, stress_index: 0.6, focus_index: 0.4, alpha_power: null, beta_power: null, theta_power: null, frontal_asymmetry: null });
    const stored = JSON.parse(localStorage.getItem("ndw_emotion_fingerprints") || "[]");
    expect(stored).toHaveLength(1); // same label merged
    expect(stored[0].sampleCount).toBe(2);
    expect(stored[0].centroid.valence).toBeCloseTo(0.35); // avg of 0.3 and 0.4
  });

  it("calls Supabase upsert when Supabase available", async () => {
    vi.mocked(getSupabase).mockResolvedValue(mockSupabase);
    const taxonomy = new EmotionTaxonomy("user1");
    await taxonomy.labelEmotion("scattered", { valence: 0.3, arousal: 0.8, stress_index: 0.7, focus_index: 0.3, alpha_power: null, beta_power: null, theta_power: null, frontal_asymmetry: null });
    expect(mockFrom).toHaveBeenCalledWith("emotion_fingerprints");
    expect(mockUpsert).toHaveBeenCalled();
  });
});

describe("EmotionTaxonomy.suggestFromEEG", () => {
  it("returns null when fewer than 3 confirmed fingerprints for a label", async () => {
    const taxonomy = new EmotionTaxonomy("user1");
    await taxonomy.labelEmotion("scattered", { valence: 0.3, arousal: 0.8, stress_index: 0.7, focus_index: 0.3, alpha_power: null, beta_power: null, theta_power: null, frontal_asymmetry: null });
    const suggestion = taxonomy.suggestFromEEG({ valence: 0.3, arousal: 0.8, stress_index: 0.7, focus_index: 0.3, alpha_power: null, beta_power: null, theta_power: null, frontal_asymmetry: null });
    expect(suggestion).toBeNull(); // only 1 sample, needs 3
  });

  it("returns label suggestion when EEG is close to stored fingerprint with 3+ samples", async () => {
    const taxonomy = new EmotionTaxonomy("user1");
    const snapshot = { valence: 0.3, arousal: 0.8, stress_index: 0.7, focus_index: 0.3, alpha_power: null, beta_power: null, theta_power: null, frontal_asymmetry: null };
    for (let i = 0; i < 3; i++) {
      await taxonomy.labelEmotion("scattered", snapshot);
    }
    const suggestion = taxonomy.suggestFromEEG(snapshot);
    expect(suggestion).toBe("scattered");
  });
});
