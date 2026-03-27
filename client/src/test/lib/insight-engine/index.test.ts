// client/src/test/lib/insight-engine/index.test.ts
import { describe, it, expect, beforeEach } from "vitest";
import { InsightEngine } from "@/lib/insight-engine/index";

beforeEach(() => localStorage.clear());

describe("InsightEngine.ingest + getRealTimeInsights", () => {
  it("returns empty array when reading is within baseline", () => {
    const engine = new InsightEngine("user1");
    engine.ingest({ stress: 0.4, focus: 0.55, valence: 0.55, arousal: 0.5, source: "eeg", timestamp: new Date().toISOString() });
    expect(engine.getRealTimeInsights()).toHaveLength(0);
  });

  it("returns deviation events for out-of-range readings", () => {
    const engine = new InsightEngine("user1");
    engine.ingest({ stress: 0.90, focus: 0.55, valence: 0.55, arousal: 0.5, source: "eeg", timestamp: new Date().toISOString() });
    const events = engine.getRealTimeInsights();
    // stress 0.90 vs population mean 0.40/std 0.15 → z = 3.3 → fires
    expect(events.some(e => e.metric === "stress")).toBe(true);
  });
});

describe("InsightEngine.labelEmotion", () => {
  it("delegates to EmotionTaxonomy and returns fingerprint", async () => {
    const engine = new InsightEngine("user1");
    const fp = await engine.labelEmotion("scattered", {
      valence: 0.3, arousal: 0.8, stress_index: 0.7, focus_index: 0.3,
      alpha_power: null, beta_power: null, theta_power: null, frontal_asymmetry: null,
    });
    expect(fp.label).toBe("scattered");
  });
});

describe("InsightEngine.getMorningBriefing", () => {
  it("returns null when no briefing cached", () => {
    const engine = new InsightEngine("user1");
    expect(engine.getMorningBriefing()).toBeNull();
  });

  it("returns cached briefing when date matches today", () => {
    const engine = new InsightEngine("user1");
    const today = new Date().toISOString().slice(0, 10);
    const cached = { stateSummary: "test", actions: ["a", "b", "c"] as [string, string, string], forecast: { label: "ok", probability: 0.7, reason: "test" } };
    localStorage.setItem(`ndw_morning_briefing_user1`, JSON.stringify({ date: today, content: cached }));
    expect(engine.getMorningBriefing()).toEqual(cached);
  });
});

describe("InsightEngine.recordInterventionTap", () => {
  it("records tap in localStorage via InterventionLibrary when active deviation exists", () => {
    const engine = new InsightEngine("user1");
    // Must have an active deviation before a tap can be recorded
    engine.ingest({ stress: 0.90, focus: 0.55, valence: 0.55, arousal: 0.5, source: "eeg", timestamp: new Date().toISOString() });
    engine.recordInterventionTap("box_breathing", "stress");
    const pending = JSON.parse(localStorage.getItem("ndw_intervention_pending") || "{}");
    expect(pending["box_breathing"]).toBeDefined();
  });

  it("does not record tap when no active deviation exists for the metric", () => {
    const engine = new InsightEngine("user1");
    // No ingest — no active deviation
    engine.recordInterventionTap("box_breathing", "stress");
    const pending = JSON.parse(localStorage.getItem("ndw_intervention_pending") || "{}");
    expect(pending["box_breathing"]).toBeUndefined();
  });
});
