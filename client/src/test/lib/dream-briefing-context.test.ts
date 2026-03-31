import { describe, it, expect } from "vitest";
import {
  extractDreamBriefingContext,
  isDreamRecent,
  type DreamBriefingContext,
} from "@/lib/dream-briefing-context";

// ── isDreamRecent ─────────────────────────────────────────────────────────────

describe("isDreamRecent", () => {
  const NOW = new Date("2026-03-30T08:00:00Z").getTime();

  it("returns true for a timestamp 2 hours ago", () => {
    const ts = new Date(NOW - 2 * 60 * 60 * 1000).toISOString();
    expect(isDreamRecent(ts, NOW)).toBe(true);
  });

  it("returns true for a timestamp exactly 23h 59m ago", () => {
    const ts = new Date(NOW - (24 * 60 * 60 * 1000 - 60_000)).toISOString();
    expect(isDreamRecent(ts, NOW)).toBe(true);
  });

  it("returns false for a timestamp exactly 24h ago", () => {
    const ts = new Date(NOW - 24 * 60 * 60 * 1000).toISOString();
    expect(isDreamRecent(ts, NOW)).toBe(false);
  });

  it("returns false for a timestamp 2 days ago", () => {
    const ts = new Date(NOW - 48 * 60 * 60 * 1000).toISOString();
    expect(isDreamRecent(ts, NOW)).toBe(false);
  });

  it("returns false for undefined", () => {
    expect(isDreamRecent(undefined, NOW)).toBe(false);
  });

  it("returns false for an invalid timestamp", () => {
    expect(isDreamRecent("not-a-date", NOW)).toBe(false);
  });
});

// ── extractDreamBriefingContext ───────────────────────────────────────────────

describe("extractDreamBriefingContext", () => {
  it("returns null for null input", () => {
    expect(extractDreamBriefingContext(null)).toBeNull();
  });

  it("returns null for undefined input", () => {
    expect(extractDreamBriefingContext(undefined)).toBeNull();
  });

  it("returns null when all useful fields are empty", () => {
    expect(extractDreamBriefingContext({
      keyInsight: null,
      themes: [],
      emotionalArc: null,
      threatSimulationIndex: 0.1,
    })).toBeNull();
  });

  it("returns null when themes is null/undefined", () => {
    expect(extractDreamBriefingContext({
      keyInsight: null,
      themes: undefined as unknown as string[],
      emotionalArc: null,
    })).toBeNull();
  });

  it("extracts keyInsight", () => {
    const result = extractDreamBriefingContext({ keyInsight: "Fear of change" });
    expect(result?.keyInsight).toBe("Fear of change");
  });

  it("trims whitespace from keyInsight", () => {
    const result = extractDreamBriefingContext({ keyInsight: "  insight  " });
    expect(result?.keyInsight).toBe("insight");
  });

  it("extracts themes array", () => {
    const result = extractDreamBriefingContext({ themes: ["transition", "self-exploration"] });
    expect(result?.themes).toEqual(["transition", "self-exploration"]);
  });

  it("filters empty strings from themes", () => {
    const result = extractDreamBriefingContext({ themes: ["", "transition", ""] });
    expect(result?.themes).toEqual(["transition"]);
  });

  it("extracts emotionalArc", () => {
    const result = extractDreamBriefingContext({ emotionalArc: "fear → relief", themes: ["transition"] });
    expect(result?.emotionalArc).toBe("fear → relief");
  });

  it("sets isSleepDistress=true when threatSimulationIndex > 0.5", () => {
    const result = extractDreamBriefingContext({
      keyInsight: "nightmare",
      threatSimulationIndex: 0.8,
    });
    expect(result?.isSleepDistress).toBe(true);
  });

  it("sets isSleepDistress=false when threatSimulationIndex is 0.5 exactly", () => {
    const result = extractDreamBriefingContext({
      keyInsight: "edge case",
      threatSimulationIndex: 0.5,
    });
    expect(result?.isSleepDistress).toBe(false);
  });

  it("sets isSleepDistress=false when threatSimulationIndex is null", () => {
    const result = extractDreamBriefingContext({
      keyInsight: "calm dream",
      threatSimulationIndex: null,
    });
    expect(result?.isSleepDistress).toBe(false);
  });

  it("returns a complete context object when all fields present", () => {
    const result = extractDreamBriefingContext({
      keyInsight: "Facing the unknown",
      themes: ["transition", "threat-simulation"],
      emotionalArc: "anxiety → acceptance",
      threatSimulationIndex: 0.75,
    });
    expect(result).toEqual<DreamBriefingContext>({
      keyInsight: "Facing the unknown",
      themes: ["transition", "threat-simulation"],
      emotionalArc: "anxiety → acceptance",
      isSleepDistress: true,
    });
  });
});

// ── buildDreamContextSection (server logic mirrored as pure function for test) ─

function buildDreamContextSection(ctx?: {
  keyInsight: string | null;
  themes: string[];
  emotionalArc: string | null;
  isSleepDistress: boolean;
}): string {
  if (!ctx) return "";
  const parts: string[] = [];
  if (ctx.themes.length > 0) {
    parts.push(`Dream themes last night: ${ctx.themes.join(", ")}.`);
  }
  if (ctx.emotionalArc) {
    parts.push(`Emotional arc: ${ctx.emotionalArc}.`);
  }
  if (ctx.keyInsight) {
    parts.push(`Key insight from dream: "${ctx.keyInsight}".`);
  }
  if (ctx.isSleepDistress) {
    parts.push("Note: last night's dream involved stress/threat content — factor in emotional recovery needs.");
  }
  return parts.length > 0 ? parts.join(" ") : "";
}

describe("buildDreamContextSection", () => {
  it("returns empty string when ctx is undefined", () => {
    expect(buildDreamContextSection(undefined)).toBe("");
  });

  it("includes themes line when themes present", () => {
    const result = buildDreamContextSection({
      keyInsight: null, themes: ["transition", "achievement"],
      emotionalArc: null, isSleepDistress: false,
    });
    expect(result).toContain("Dream themes last night: transition, achievement.");
  });

  it("includes emotional arc when present", () => {
    const result = buildDreamContextSection({
      keyInsight: null, themes: ["transition"],
      emotionalArc: "fear → relief", isSleepDistress: false,
    });
    expect(result).toContain("Emotional arc: fear → relief.");
  });

  it("includes key insight when present", () => {
    const result = buildDreamContextSection({
      keyInsight: "Need to let go", themes: [],
      emotionalArc: null, isSleepDistress: false,
    });
    expect(result).toContain('Key insight from dream: "Need to let go".');
  });

  it("includes sleep distress note when isSleepDistress is true", () => {
    const result = buildDreamContextSection({
      keyInsight: null, themes: [],
      emotionalArc: null, isSleepDistress: true,
    });
    expect(result).toContain("stress/threat content");
  });

  it("returns empty string when all fields are empty/false", () => {
    const result = buildDreamContextSection({
      keyInsight: null, themes: [],
      emotionalArc: null, isSleepDistress: false,
    });
    expect(result).toBe("");
  });

  it("concatenates all four sections in order", () => {
    const result = buildDreamContextSection({
      keyInsight: "Growth",
      themes: ["transition"],
      emotionalArc: "fear → hope",
      isSleepDistress: true,
    });
    expect(result.indexOf("Dream themes")).toBeLessThan(result.indexOf("Emotional arc"));
    expect(result.indexOf("Emotional arc")).toBeLessThan(result.indexOf("Key insight"));
    expect(result.indexOf("Key insight")).toBeLessThan(result.indexOf("stress/threat"));
  });
});
