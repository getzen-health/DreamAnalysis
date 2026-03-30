/**
 * Multi-pass LLM dream analysis — Issue #546.
 *
 * Runs the LLM in three sequential passes over a dream text:
 *   Pass 1: Extract themes and symbols (structured)
 *   Pass 2: Interpret symbols and emotional tone (narrative)
 *   Pass 3: Generate actionable insight (concise)
 *
 * Uses Anthropic claude-haiku-4-5-20251001 for all passes. Falls back to
 * OpenAI gpt-5 when ANTHROPIC_API_KEY is not set.
 *
 * JSON output is enforced via system prompts + validation. If the LLM
 * returns unparseable JSON, the pass returns safe defaults.
 */

import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";

// ── Public types ──────────────────────────────────────────────────────────────

export interface SymbolInterpretation {
  symbol: string;
  meaning: string;
}

export interface DreamAnalysisResult {
  /** 2-3 sentence summary of the dream */
  summary: string;
  /** Extracted themes (e.g. "loss", "transformation", "pursuit") */
  themes: string[];
  /** Symbol interpretations */
  symbols: SymbolInterpretation[];
  /** Dominant emotional tone: "anxious", "hopeful", "nostalgic", etc. */
  emotionalTone: string;
  /** Connections to recent life events (generic prompts) */
  connections: string[];
  /** Signs of lucidity if any */
  lucidityIndicators: string[];
  /** One thing to reflect on */
  actionableInsight: string;
}

// ── Internal pass types ───────────────────────────────────────────────────────

interface Pass1Result {
  themes: string[];
  symbols: string[];
  summary: string;
}

interface Pass2Result {
  symbols: SymbolInterpretation[];
  emotionalTone: string;
  connections: string[];
  lucidityIndicators: string[];
}

interface Pass3Result {
  actionableInsight: string;
}

// ── LLM abstraction ──────────────────────────────────────────────────────────

type LLMClient =
  | { type: "anthropic"; client: Anthropic }
  | { type: "openai"; client: OpenAI };

async function llmComplete(
  llm: LLMClient,
  systemPrompt: string,
  userPrompt: string,
  maxTokens: number,
): Promise<string> {
  if (llm.type === "anthropic") {
    const r = await llm.client.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: maxTokens,
      system: systemPrompt,
      messages: [{ role: "user", content: userPrompt }],
    });
    return r.content[0].type === "text" ? r.content[0].text : "{}";
  }

  // OpenAI path
  const r = await llm.client.chat.completions.create({
    model: "gpt-5",
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    response_format: { type: "json_object" },
  });
  return r.choices[0].message.content || "{}";
}

// ── JSON parsing with defaults ────────────────────────────────────────────────

function safeParseJSON<T>(raw: string, fallback: T): T {
  // Strip markdown code fences if the LLM wraps the JSON
  const cleaned = raw
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();
  try {
    return JSON.parse(cleaned) as T;
  } catch {
    return fallback;
  }
}

// ── Validators ────────────────────────────────────────────────────────────────

function validatePass1(raw: unknown): Pass1Result {
  const obj = raw as Record<string, unknown>;
  return {
    themes: Array.isArray(obj?.themes)
      ? (obj.themes as string[]).filter((t) => typeof t === "string").slice(0, 10)
      : [],
    symbols: Array.isArray(obj?.symbols)
      ? (obj.symbols as string[]).filter((s) => typeof s === "string").slice(0, 15)
      : [],
    summary:
      typeof obj?.summary === "string" ? (obj.summary as string).slice(0, 500) : "",
  };
}

function validatePass2(raw: unknown): Pass2Result {
  const obj = raw as Record<string, unknown>;

  const rawSymbols = Array.isArray(obj?.symbols) ? obj.symbols : [];
  const symbols: SymbolInterpretation[] = rawSymbols
    .filter(
      (s: unknown) =>
        typeof s === "object" &&
        s !== null &&
        typeof (s as Record<string, unknown>).symbol === "string" &&
        typeof (s as Record<string, unknown>).meaning === "string",
    )
    .map((s: unknown) => {
      const entry = s as Record<string, string>;
      return { symbol: entry.symbol, meaning: entry.meaning };
    })
    .slice(0, 15);

  return {
    symbols,
    emotionalTone:
      typeof obj?.emotionalTone === "string"
        ? (obj.emotionalTone as string).slice(0, 50)
        : "neutral",
    connections: Array.isArray(obj?.connections)
      ? (obj.connections as string[]).filter((c) => typeof c === "string").slice(0, 5)
      : [],
    lucidityIndicators: Array.isArray(obj?.lucidityIndicators)
      ? (obj.lucidityIndicators as string[])
          .filter((l) => typeof l === "string")
          .slice(0, 5)
      : [],
  };
}

function validatePass3(raw: unknown): Pass3Result {
  const obj = raw as Record<string, unknown>;
  return {
    actionableInsight:
      typeof obj?.actionableInsight === "string"
        ? (obj.actionableInsight as string).slice(0, 500)
        : "",
  };
}

// ── Core multi-pass function ──────────────────────────────────────────────────

/**
 * Multi-pass dream analysis:
 *   Pass 1: Extract themes and symbols (structured)
 *   Pass 2: Interpret symbols and emotional tone (narrative)
 *   Pass 3: Generate actionable insight (concise)
 *
 * @param dreamText - The dream narrative to analyze
 * @param recentDreamThemes - Optional themes from recent dreams for continuity
 * @param llmOverride - Injectable LLM client (for testing / custom config)
 */
export async function analyzeDreamMultiPass(
  dreamText: string,
  recentDreamThemes?: string[],
  llmOverride?: LLMClient,
): Promise<DreamAnalysisResult> {
  // Resolve the LLM client
  const llm: LLMClient = llmOverride ?? resolveDefaultLLM();

  const continuityCtx = recentDreamThemes?.length
    ? `\n\nRecent dream themes for continuity context: ${recentDreamThemes.join(", ")}`
    : "";

  // ── Pass 1: Theme & symbol extraction ─────────────────────────────────────
  const pass1Raw = await llmComplete(
    llm,
    "You are a dream analysis expert. Extract themes, symbols, and a brief summary from the dream text. Return only valid JSON.",
    `Analyze this dream and extract key themes, symbolic elements, and write a 2-3 sentence summary.

Return JSON in exactly this shape:
{
  "themes": ["theme1", "theme2"],
  "symbols": ["symbol1", "symbol2"],
  "summary": "2-3 sentence summary of the dream"
}${continuityCtx}

Dream: ${dreamText}`,
    512,
  );
  const pass1 = validatePass1(safeParseJSON(pass1Raw, {}));

  // ── Pass 2: Symbol interpretation & emotional tone ────────────────────────
  const pass2Raw = await llmComplete(
    llm,
    "You are a dream interpretation specialist combining Jungian and neuroscience perspectives. Return only valid JSON.",
    `Given these extracted themes: ${JSON.stringify(pass1.themes)}
And these symbols: ${JSON.stringify(pass1.symbols)}

For the dream below, interpret each symbol's meaning, determine the overall emotional tone, suggest connections to recent waking life, and identify any lucidity indicators.

Return JSON in exactly this shape:
{
  "symbols": [{"symbol": "name", "meaning": "interpretation"}],
  "emotionalTone": "one word like anxious, hopeful, nostalgic, etc.",
  "connections": ["possible connection to waking life"],
  "lucidityIndicators": ["any signs of awareness within the dream"]
}

Dream: ${dreamText}`,
    768,
  );
  const pass2 = validatePass2(safeParseJSON(pass2Raw, {}));

  // ── Pass 3: Actionable insight ────────────────────────────────────────────
  const pass3Raw = await llmComplete(
    llm,
    "You are a clinical dream analyst. Provide a single, specific, actionable insight. Return only valid JSON.",
    `Dream summary: ${pass1.summary}
Themes: ${pass1.themes.join(", ")}
Emotional tone: ${pass2.emotionalTone}
Symbol interpretations: ${JSON.stringify(pass2.symbols)}

Based on all of this, write one specific, actionable insight the dreamer should reflect on. Be concrete, not generic.

Return JSON in exactly this shape:
{
  "actionableInsight": "One specific thing to reflect on"
}`,
    256,
  );
  const pass3 = validatePass3(safeParseJSON(pass3Raw, {}));

  // ── Combine all passes ────────────────────────────────────────────────────
  return {
    summary: pass1.summary,
    themes: pass1.themes,
    symbols: pass2.symbols,
    emotionalTone: pass2.emotionalTone,
    connections: pass2.connections,
    lucidityIndicators: pass2.lucidityIndicators,
    actionableInsight: pass3.actionableInsight,
  };
}

// ── Default LLM resolution ──────────────────────────────────────────────────

function resolveDefaultLLM(): LLMClient {
  if (process.env.ANTHROPIC_API_KEY) {
    return {
      type: "anthropic",
      client: new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY }),
    };
  }
  if (process.env.OPENAI_API_KEY) {
    return {
      type: "openai",
      client: new OpenAI({ apiKey: process.env.OPENAI_API_KEY }),
    };
  }
  throw new Error("No LLM API key configured (ANTHROPIC_API_KEY or OPENAI_API_KEY)");
}

// ── Re-export LLMClient type for testing ────────────────────────────────────
export type { LLMClient };
