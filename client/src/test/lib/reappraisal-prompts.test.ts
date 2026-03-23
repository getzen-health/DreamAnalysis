import { describe, it, expect } from "vitest";
import {
  REAPPRAISAL_PROMPTS,
  selectReappraisalPrompt,
  shouldShowReappraisal,
  type ReappraisalPrompt,
} from "@/lib/reappraisal-prompts";

// ── Prompt library ─────────────────────────────────────────────────────

describe("REAPPRAISAL_PROMPTS", () => {
  it("has at least 10 prompts", () => {
    expect(REAPPRAISAL_PROMPTS.length).toBeGreaterThanOrEqual(10);
  });

  it("each prompt has required fields", () => {
    for (const p of REAPPRAISAL_PROMPTS) {
      expect(p.id.length).toBeGreaterThan(0);
      expect(p.text.length).toBeGreaterThan(0);
      expect(p.rationale.length).toBeGreaterThan(0);
      expect(["reframe", "distance", "perspective", "growth"]).toContain(p.category);
    }
  });

  it("all IDs are unique", () => {
    const ids = REAPPRAISAL_PROMPTS.map((p) => p.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("covers multiple categories", () => {
    const categories = new Set(REAPPRAISAL_PROMPTS.map((p) => p.category));
    expect(categories.size).toBeGreaterThanOrEqual(3);
  });
});

// ── selectReappraisalPrompt ────────────────────────────────────────────

describe("selectReappraisalPrompt", () => {
  it("returns a valid prompt", () => {
    const prompt = selectReappraisalPrompt();
    expect(prompt).toBeDefined();
    expect(prompt.text.length).toBeGreaterThan(0);
  });

  it("filters by category", () => {
    const prompt = selectReappraisalPrompt("growth");
    expect(prompt.category).toBe("growth");
  });

  it("returns prompts for all valid categories", () => {
    for (const cat of ["reframe", "distance", "perspective", "growth"] as const) {
      const prompt = selectReappraisalPrompt(cat);
      expect(prompt.category).toBe(cat);
    }
  });

  it("avoids immediate repeats over multiple calls", () => {
    // Call many times and check that we get at least 2 different prompts
    const ids = new Set<string>();
    for (let i = 0; i < 20; i++) {
      ids.add(selectReappraisalPrompt().id);
    }
    expect(ids.size).toBeGreaterThan(1);
  });
});

// ── shouldShowReappraisal ──────────────────────────────────────────────

describe("shouldShowReappraisal", () => {
  it("returns null when stress is below threshold", () => {
    expect(shouldShowReappraisal(0.3)).toBeNull();
    expect(shouldShowReappraisal(0.5)).toBeNull();
    expect(shouldShowReappraisal(0.59)).toBeNull();
  });

  it("returns a prompt when stress is at or above threshold", () => {
    const prompt = shouldShowReappraisal(0.6);
    expect(prompt).not.toBeNull();
    expect(prompt!.text.length).toBeGreaterThan(0);
  });

  it("returns a prompt for high stress", () => {
    const prompt = shouldShowReappraisal(0.9);
    expect(prompt).not.toBeNull();
  });

  it("respects custom threshold", () => {
    expect(shouldShowReappraisal(0.4, 0.5)).toBeNull();
    const prompt = shouldShowReappraisal(0.5, 0.5);
    expect(prompt).not.toBeNull();
  });

  it("returns prompt at exact threshold", () => {
    const prompt = shouldShowReappraisal(0.6, 0.6);
    expect(prompt).not.toBeNull();
  });
});
