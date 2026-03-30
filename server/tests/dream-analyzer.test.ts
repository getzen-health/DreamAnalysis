/**
 * Unit tests for the multi-pass dream analyzer — Issue #546.
 *
 * Mocks the LLM calls to test:
 * - JSON parsing and validation for each pass
 * - Graceful handling of malformed LLM responses
 * - Full pipeline integration with mock LLM
 * - recentDreamThemes continuity parameter
 */
import { describe, it, expect, vi } from "vitest";
import {
  analyzeDreamMultiPass,
  type DreamAnalysisResult,
  type LLMClient,
} from "../lib/dream-analyzer";

// ── Mock LLM factory ─────────────────────────────────────────────────────────

/**
 * Creates a mock Anthropic-shaped LLM client that returns predefined responses
 * for each sequential call (pass 1, pass 2, pass 3).
 */
function createMockLLM(responses: string[]): LLMClient {
  let callIndex = 0;
  return {
    type: "anthropic",
    client: {
      messages: {
        create: vi.fn(async () => {
          const text = responses[callIndex] ?? "{}";
          callIndex++;
          return {
            content: [{ type: "text" as const, text }],
          };
        }),
      },
    } as unknown as import("@anthropic-ai/sdk").default,
  };
}

// ── Test data ─────────────────────────────────────────────────────────────────

const SAMPLE_DREAM =
  "I was flying over a vast ocean at night. The moon was enormous and red. " +
  "I looked down and saw a whale surface, then it spoke to me in my mother's voice. " +
  "I realized I was dreaming and tried to control the direction of flight.";

const VALID_PASS1 = JSON.stringify({
  themes: ["flight", "ocean", "communication"],
  symbols: ["moon", "whale", "ocean", "flying"],
  summary:
    "The dreamer flies over a moonlit ocean and encounters a speaking whale. " +
    "The dream contains lucid elements where the dreamer becomes aware of dreaming.",
});

const VALID_PASS2 = JSON.stringify({
  symbols: [
    { symbol: "moon", meaning: "Unconscious feminine energy, emotional cycles" },
    { symbol: "whale", meaning: "Deep emotions, ancestral wisdom" },
    { symbol: "ocean", meaning: "The unconscious mind, vast emotional landscape" },
    { symbol: "flying", meaning: "Freedom, transcendence, desire for control" },
  ],
  emotionalTone: "awe",
  connections: [
    "Flying may reflect a desire for freedom in a current life situation",
    "The mother's voice through the whale suggests unresolved maternal feelings",
  ],
  lucidityIndicators: [
    "Dreamer realized they were dreaming",
    "Attempted to control flight direction",
  ],
});

const VALID_PASS3 = JSON.stringify({
  actionableInsight:
    "Consider what freedoms you feel restricted from in waking life — " +
    "the dream's lucid flight and maternal voice suggest a tension between autonomy and connection.",
});

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("analyzeDreamMultiPass", () => {
  it("returns a fully populated DreamAnalysisResult with valid LLM responses", async () => {
    const llm = createMockLLM([VALID_PASS1, VALID_PASS2, VALID_PASS3]);

    const result = await analyzeDreamMultiPass(SAMPLE_DREAM, undefined, llm);

    expect(result.summary).toContain("flies over");
    expect(result.themes).toEqual(["flight", "ocean", "communication"]);
    expect(result.symbols).toHaveLength(4);
    expect(result.symbols[0]).toEqual({
      symbol: "moon",
      meaning: "Unconscious feminine energy, emotional cycles",
    });
    expect(result.emotionalTone).toBe("awe");
    expect(result.connections).toHaveLength(2);
    expect(result.lucidityIndicators).toHaveLength(2);
    expect(result.actionableInsight).toContain("freedoms");
  });

  it("makes exactly 3 LLM calls (one per pass)", async () => {
    const llm = createMockLLM([VALID_PASS1, VALID_PASS2, VALID_PASS3]);

    await analyzeDreamMultiPass(SAMPLE_DREAM, undefined, llm);

    const createFn = (llm.client as unknown as { messages: { create: ReturnType<typeof vi.fn> } }).messages.create;
    expect(createFn).toHaveBeenCalledTimes(3);
  });

  it("includes recentDreamThemes in the pass 1 prompt", async () => {
    const llm = createMockLLM([VALID_PASS1, VALID_PASS2, VALID_PASS3]);

    await analyzeDreamMultiPass(
      SAMPLE_DREAM,
      ["water", "family", "transformation"],
      llm,
    );

    const createFn = (llm.client as unknown as { messages: { create: ReturnType<typeof vi.fn> } }).messages.create;
    const firstCallArgs = createFn.mock.calls[0][0];
    const userMsg = firstCallArgs.messages[0].content as string;
    expect(userMsg).toContain("water, family, transformation");
  });

  it("returns safe defaults when pass 1 returns invalid JSON", async () => {
    const llm = createMockLLM([
      "this is not json at all",
      VALID_PASS2,
      VALID_PASS3,
    ]);

    const result = await analyzeDreamMultiPass(SAMPLE_DREAM, undefined, llm);

    // Pass 1 defaults: empty themes, symbols, summary
    expect(result.summary).toBe("");
    expect(result.themes).toEqual([]);
    // Pass 2 still runs with empty input from pass 1
    expect(result.symbols).toHaveLength(4);
    expect(result.emotionalTone).toBe("awe");
  });

  it("returns safe defaults when pass 2 returns invalid JSON", async () => {
    const llm = createMockLLM([
      VALID_PASS1,
      "not-json",
      VALID_PASS3,
    ]);

    const result = await analyzeDreamMultiPass(SAMPLE_DREAM, undefined, llm);

    // Pass 1 data is fine
    expect(result.themes).toEqual(["flight", "ocean", "communication"]);
    // Pass 2 defaults
    expect(result.symbols).toEqual([]);
    expect(result.emotionalTone).toBe("neutral");
    expect(result.connections).toEqual([]);
    expect(result.lucidityIndicators).toEqual([]);
  });

  it("returns safe defaults when pass 3 returns invalid JSON", async () => {
    const llm = createMockLLM([
      VALID_PASS1,
      VALID_PASS2,
      "broken json",
    ]);

    const result = await analyzeDreamMultiPass(SAMPLE_DREAM, undefined, llm);

    // Passes 1 and 2 are fine
    expect(result.themes).toHaveLength(3);
    expect(result.symbols).toHaveLength(4);
    // Pass 3 defaults
    expect(result.actionableInsight).toBe("");
  });

  it("handles all three passes returning empty JSON objects", async () => {
    const llm = createMockLLM(["{}", "{}", "{}"]);

    const result = await analyzeDreamMultiPass(SAMPLE_DREAM, undefined, llm);

    expect(result).toEqual({
      summary: "",
      themes: [],
      symbols: [],
      emotionalTone: "neutral",
      connections: [],
      lucidityIndicators: [],
      actionableInsight: "",
    });
  });

  it("handles markdown-wrapped JSON from LLM", async () => {
    const wrappedPass1 = "```json\n" + VALID_PASS1 + "\n```";
    const llm = createMockLLM([wrappedPass1, VALID_PASS2, VALID_PASS3]);

    const result = await analyzeDreamMultiPass(SAMPLE_DREAM, undefined, llm);

    expect(result.themes).toEqual(["flight", "ocean", "communication"]);
    expect(result.summary).toContain("flies over");
  });

  it("truncates excessively long summary to 500 chars", async () => {
    const longSummary = "A".repeat(1000);
    const pass1WithLongSummary = JSON.stringify({
      themes: ["test"],
      symbols: ["test"],
      summary: longSummary,
    });
    const llm = createMockLLM([pass1WithLongSummary, VALID_PASS2, VALID_PASS3]);

    const result = await analyzeDreamMultiPass(SAMPLE_DREAM, undefined, llm);

    expect(result.summary.length).toBeLessThanOrEqual(500);
  });

  it("limits themes array to 10 entries", async () => {
    const manyThemes = Array.from({ length: 20 }, (_, i) => `theme${i}`);
    const pass1 = JSON.stringify({
      themes: manyThemes,
      symbols: ["s"],
      summary: "test",
    });
    const llm = createMockLLM([pass1, VALID_PASS2, VALID_PASS3]);

    const result = await analyzeDreamMultiPass(SAMPLE_DREAM, undefined, llm);

    expect(result.themes.length).toBeLessThanOrEqual(10);
  });

  it("filters out non-string entries from themes array", async () => {
    const pass1 = JSON.stringify({
      themes: ["valid", 42, null, "also valid", { bad: true }],
      symbols: [],
      summary: "test",
    });
    const llm = createMockLLM([pass1, VALID_PASS2, VALID_PASS3]);

    const result = await analyzeDreamMultiPass(SAMPLE_DREAM, undefined, llm);

    expect(result.themes).toEqual(["valid", "also valid"]);
  });

  it("validates symbol interpretation objects in pass 2", async () => {
    const pass2WithBadSymbols = JSON.stringify({
      symbols: [
        { symbol: "moon", meaning: "cycles" },
        { symbol: "whale" },           // missing meaning
        { meaning: "orphan meaning" },  // missing symbol
        "just a string",                // wrong type
        { symbol: "ocean", meaning: "unconscious" },
      ],
      emotionalTone: "calm",
      connections: [],
      lucidityIndicators: [],
    });
    const llm = createMockLLM([VALID_PASS1, pass2WithBadSymbols, VALID_PASS3]);

    const result = await analyzeDreamMultiPass(SAMPLE_DREAM, undefined, llm);

    // Only the two valid symbol objects should survive
    expect(result.symbols).toEqual([
      { symbol: "moon", meaning: "cycles" },
      { symbol: "ocean", meaning: "unconscious" },
    ]);
  });

  it("propagates LLM errors as thrown exceptions", async () => {
    const llm: LLMClient = {
      type: "anthropic",
      client: {
        messages: {
          create: vi.fn(async () => {
            throw new Error("API rate limited");
          }),
        },
      } as unknown as import("@anthropic-ai/sdk").default,
    };

    await expect(
      analyzeDreamMultiPass(SAMPLE_DREAM, undefined, llm),
    ).rejects.toThrow("API rate limited");
  });
});

describe("DreamAnalysisResult schema shape", () => {
  it("has all required fields with correct types", async () => {
    const llm = createMockLLM([VALID_PASS1, VALID_PASS2, VALID_PASS3]);
    const result = await analyzeDreamMultiPass(SAMPLE_DREAM, undefined, llm);

    // Type-level checks via assertion — these validate the interface contract
    const keys: (keyof DreamAnalysisResult)[] = [
      "summary",
      "themes",
      "symbols",
      "emotionalTone",
      "connections",
      "lucidityIndicators",
      "actionableInsight",
    ];

    for (const key of keys) {
      expect(result).toHaveProperty(key);
    }

    expect(typeof result.summary).toBe("string");
    expect(Array.isArray(result.themes)).toBe(true);
    expect(Array.isArray(result.symbols)).toBe(true);
    expect(typeof result.emotionalTone).toBe("string");
    expect(Array.isArray(result.connections)).toBe(true);
    expect(Array.isArray(result.lucidityIndicators)).toBe(true);
    expect(typeof result.actionableInsight).toBe("string");

    // Validate symbol entries
    for (const sym of result.symbols) {
      expect(typeof sym.symbol).toBe("string");
      expect(typeof sym.meaning).toBe("string");
    }
  });
});
