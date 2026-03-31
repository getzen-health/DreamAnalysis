import { describe, it, expect } from "vitest";

/**
 * Tests for IRT (Image Rehearsal Therapy) session logic.
 *
 * Covers pure validation logic mirroring the POST /api/irt-session handler:
 *  - required-field checking (userId, originalDreamText, rewrittenEnding)
 *  - optional rehearsalNote
 *  - field truncation behaviour
 *  - IrtWorkflowCard step-sequencing logic (step machine)
 */

// ── Validation mirror ────────────────────────────────────────────────────────

interface IrtInput {
  userId?: unknown;
  originalDreamText?: unknown;
  rewrittenEnding?: unknown;
  rehearsalNote?: unknown;
}

interface IrtValidationResult {
  ok: boolean;
  error?: string;
}

function validateIrtInput(input: IrtInput): IrtValidationResult {
  if (!input.userId || typeof input.userId !== "string") {
    return { ok: false, error: "userId is required" };
  }
  if (!input.originalDreamText || typeof input.originalDreamText !== "string") {
    return { ok: false, error: "originalDreamText is required" };
  }
  if (!input.rewrittenEnding || typeof input.rewrittenEnding !== "string") {
    return { ok: false, error: "rewrittenEnding is required" };
  }
  return { ok: true };
}

const MAX_TEXT = 5000;
const MAX_NOTE = 1000;

function truncateIrtFields(input: {
  originalDreamText: string;
  rewrittenEnding: string;
  rehearsalNote?: string | null;
}) {
  return {
    originalDreamText: input.originalDreamText.slice(0, MAX_TEXT),
    rewrittenEnding: input.rewrittenEnding.slice(0, MAX_TEXT),
    rehearsalNote: input.rehearsalNote ? input.rehearsalNote.slice(0, MAX_NOTE) : null,
  };
}

describe("validateIrtInput", () => {
  it("rejects missing userId", () => {
    const r = validateIrtInput({ originalDreamText: "dream", rewrittenEnding: "new" });
    expect(r.ok).toBe(false);
    expect(r.error).toMatch(/userId/);
  });

  it("rejects null userId", () => {
    expect(validateIrtInput({ userId: null, originalDreamText: "d", rewrittenEnding: "r" }).ok).toBe(false);
  });

  it("rejects numeric userId", () => {
    expect(validateIrtInput({ userId: 42, originalDreamText: "d", rewrittenEnding: "r" }).ok).toBe(false);
  });

  it("rejects missing originalDreamText", () => {
    const r = validateIrtInput({ userId: "u1", rewrittenEnding: "new ending" });
    expect(r.ok).toBe(false);
    expect(r.error).toMatch(/originalDreamText/);
  });

  it("rejects missing rewrittenEnding", () => {
    const r = validateIrtInput({ userId: "u1", originalDreamText: "scary dream" });
    expect(r.ok).toBe(false);
    expect(r.error).toMatch(/rewrittenEnding/);
  });

  it("accepts valid complete input", () => {
    const r = validateIrtInput({ userId: "u1", originalDreamText: "scary", rewrittenEnding: "safe" });
    expect(r.ok).toBe(true);
    expect(r.error).toBeUndefined();
  });

  it("accepts valid input with rehearsalNote", () => {
    const r = validateIrtInput({
      userId: "u1",
      originalDreamText: "nightmare text",
      rewrittenEnding: "calm resolution",
      rehearsalNote: "felt peaceful",
    });
    expect(r.ok).toBe(true);
  });
});

describe("truncateIrtFields", () => {
  it("passes through short fields unchanged", () => {
    const result = truncateIrtFields({ originalDreamText: "dream", rewrittenEnding: "ending" });
    expect(result.originalDreamText).toBe("dream");
    expect(result.rewrittenEnding).toBe("ending");
    expect(result.rehearsalNote).toBeNull();
  });

  it("truncates originalDreamText at 5000 chars", () => {
    const long = "a".repeat(6000);
    const result = truncateIrtFields({ originalDreamText: long, rewrittenEnding: "e" });
    expect(result.originalDreamText).toHaveLength(5000);
  });

  it("truncates rewrittenEnding at 5000 chars", () => {
    const long = "b".repeat(6000);
    const result = truncateIrtFields({ originalDreamText: "d", rewrittenEnding: long });
    expect(result.rewrittenEnding).toHaveLength(5000);
  });

  it("truncates rehearsalNote at 1000 chars", () => {
    const long = "c".repeat(2000);
    const result = truncateIrtFields({ originalDreamText: "d", rewrittenEnding: "e", rehearsalNote: long });
    expect(result.rehearsalNote).toHaveLength(1000);
  });

  it("sets rehearsalNote to null when undefined", () => {
    const result = truncateIrtFields({ originalDreamText: "d", rewrittenEnding: "e" });
    expect(result.rehearsalNote).toBeNull();
  });

  it("sets rehearsalNote to null when null", () => {
    const result = truncateIrtFields({ originalDreamText: "d", rewrittenEnding: "e", rehearsalNote: null });
    expect(result.rehearsalNote).toBeNull();
  });
});

// ── IrtWorkflowCard step-machine logic ───────────────────────────────────────

type IrtStep = "read" | "rewrite" | "rehearse" | "done";

function nextStep(current: IrtStep, rewrittenEnding: string): IrtStep {
  if (current === "read") return "rewrite";
  if (current === "rewrite" && rewrittenEnding.trim()) return "rehearse";
  if (current === "rehearse") return "done";
  return current;
}

function canAdvance(current: IrtStep, rewrittenEnding: string): boolean {
  if (current === "read") return true;
  if (current === "rewrite") return rewrittenEnding.trim().length > 0;
  if (current === "rehearse") return true;
  return false;
}

describe("IrtWorkflowCard step machine", () => {
  it("read → rewrite", () => {
    expect(nextStep("read", "")).toBe("rewrite");
  });

  it("rewrite → rehearse when text present", () => {
    expect(nextStep("rewrite", "new calm ending")).toBe("rehearse");
  });

  it("rewrite stays when text empty", () => {
    // canAdvance gate prevents advancement
    expect(canAdvance("rewrite", "")).toBe(false);
    expect(canAdvance("rewrite", "  ")).toBe(false);
  });

  it("rewrite can advance when text non-empty", () => {
    expect(canAdvance("rewrite", "something")).toBe(true);
  });

  it("rehearse → done", () => {
    expect(nextStep("rehearse", "anything")).toBe("done");
  });

  it("done stays done", () => {
    expect(nextStep("done", "")).toBe("done");
  });

  it("canAdvance is true for read step regardless of rewrittenEnding", () => {
    expect(canAdvance("read", "")).toBe(true);
  });

  it("canAdvance is true for rehearse step", () => {
    expect(canAdvance("rehearse", "")).toBe(true);
  });

  it("canAdvance is false for done step", () => {
    expect(canAdvance("done", "")).toBe(false);
  });
});
