# Inner Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 3 score circles on the Today page with a single adaptive "Inner Score" (0-100) that works across voice-only, health+voice, and EEG+health+voice data tiers.

**Architecture:** Client-side computation library (`inner-score.ts`) handles tier detection, score formula, and narrative generation. A new `inner-score-card.tsx` component renders the 220px SVG arc gauge with tap-to-expand factor breakdown. Express API endpoints persist daily scores to a new `inner_scores` Drizzle table and serve history for trends.

**Tech Stack:** React 18 + TypeScript, Tailwind + shadcn/ui, Framer Motion, SVG, Drizzle ORM, Express, vitest + @testing-library/react

**Spec:** `docs/superpowers/specs/2026-03-27-inner-score-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `client/src/lib/inner-score.ts` | Create | Tier detection, score computation, normalization, weight redistribution, narrative generation |
| `client/src/test/lib/inner-score.test.ts` | Create | Unit tests for all computation logic |
| `client/src/components/inner-score-card.tsx` | Create | Hero gauge SVG, building state, tap-to-expand breakdown, factor bars, sparkline |
| `client/src/test/components/inner-score-card.test.tsx` | Create | Component render + interaction tests |
| `shared/schema.ts` | Modify | Add `innerScores` table + insert schema |
| `server/routes.ts` | Modify | Add `GET /api/inner-score/:userId` and `/api/inner-score/:userId/history` |
| `api/[...path].ts` | Modify | Add inner-score routes to Vercel catch-all |
| `client/src/pages/today.tsx` | Modify | Remove 3 ScoreCircles, add InnerScoreCard |

---

### Task 1: Score Computation Library

**Files:**
- Create: `client/src/lib/inner-score.ts`
- Test: `client/src/test/lib/inner-score.test.ts`

- [ ] **Step 1: Write failing tests for normalization helpers**

```typescript
// client/src/test/lib/inner-score.test.ts
import { describe, it, expect } from "vitest";
import {
  normalizeStress,
  normalizeValence,
  normalizeEnergy,
} from "@/lib/inner-score";

describe("normalizeStress", () => {
  it("inverts stress 0.0 → 100", () => {
    expect(normalizeStress(0)).toBe(100);
  });
  it("inverts stress 1.0 → 0", () => {
    expect(normalizeStress(1)).toBe(0);
  });
  it("inverts stress 0.3 → 70", () => {
    expect(normalizeStress(0.3)).toBe(70);
  });
});

describe("normalizeValence", () => {
  it("maps -1 → 0", () => {
    expect(normalizeValence(-1)).toBe(0);
  });
  it("maps 0 → 50", () => {
    expect(normalizeValence(0)).toBe(50);
  });
  it("maps 1 → 100", () => {
    expect(normalizeValence(1)).toBe(100);
  });
});

describe("normalizeEnergy", () => {
  it("maps arousal 0.0 → 0", () => {
    expect(normalizeEnergy({ arousal: 0 })).toBe(0);
  });
  it("maps arousal 1.0 → 100", () => {
    expect(normalizeEnergy({ arousal: 1 })).toBe(100);
  });
  it("maps mood log 1-5 scale: 3 → 60", () => {
    expect(normalizeEnergy({ moodEnergy: 3, moodScale: 5 })).toBe(60);
  });
  it("prefers arousal over mood log", () => {
    expect(normalizeEnergy({ arousal: 0.8, moodEnergy: 1, moodScale: 5 })).toBe(80);
  });
  it("returns 50 when no data", () => {
    expect(normalizeEnergy({})).toBe(50);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npm test -- --run client/src/test/lib/inner-score.test.ts`
Expected: FAIL — module not found

- [ ] **Step 3: Implement normalization helpers**

```typescript
// client/src/lib/inner-score.ts

// ─── Normalization ───────────────────────────────────────────────────────────

/** Invert stress index (0-1) to 0-100 where 100 = no stress */
export function normalizeStress(stress: number): number {
  return Math.round((1 - Math.max(0, Math.min(1, stress))) * 100);
}

/** Map valence (-1 to +1) to 0-100 where 100 = most positive */
export function normalizeValence(valence: number): number {
  return Math.round(((Math.max(-1, Math.min(1, valence)) + 1) / 2) * 100);
}

/** Normalize energy from arousal (0-1) or mood log (1-5 or 1-10). Arousal preferred. */
export function normalizeEnergy(input: {
  arousal?: number;
  moodEnergy?: number;
  moodScale?: number;
}): number {
  if (input.arousal != null) {
    return Math.round(Math.max(0, Math.min(1, input.arousal)) * 100);
  }
  if (input.moodEnergy != null && input.moodScale != null && input.moodScale > 0) {
    return Math.round((input.moodEnergy / input.moodScale) * 100);
  }
  return 50; // neutral default
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npm test -- --run client/src/test/lib/inner-score.test.ts`
Expected: PASS

- [ ] **Step 5: Write failing tests for tier detection**

Add to `client/src/test/lib/inner-score.test.ts`:

```typescript
import { detectTier } from "@/lib/inner-score";

describe("detectTier", () => {
  it("returns null when no data", () => {
    expect(detectTier({})).toBeNull();
  });
  it("returns voice when only stress present", () => {
    expect(detectTier({ stress: 0.4 })).toBe("voice");
  });
  it("returns voice when only valence present", () => {
    expect(detectTier({ valence: 0.2 })).toBe("voice");
  });
  it("returns health_voice when sleep + stress present", () => {
    expect(detectTier({ sleepQuality: 80, stress: 0.3 })).toBe("health_voice");
  });
  it("returns eeg_health_voice when brainHealth + sleep present", () => {
    expect(detectTier({ brainHealth: 70, sleepQuality: 85, stress: 0.2 })).toBe("eeg_health_voice");
  });
  it("falls back to health_voice when brainHealth missing but sleep present", () => {
    expect(detectTier({ sleepQuality: 80, stress: 0.5 })).toBe("health_voice");
  });
  it("falls back to voice when sleep missing", () => {
    expect(detectTier({ stress: 0.4, valence: 0.1 })).toBe("voice");
  });
});
```

- [ ] **Step 6: Implement detectTier**

Add to `client/src/lib/inner-score.ts`:

```typescript
// ─── Types ───────────────────────────────────────────────────────────────────

export type Tier = "voice" | "health_voice" | "eeg_health_voice";

export interface ScoreInputs {
  stress?: number | null;
  valence?: number | null;
  arousal?: number | null;
  moodEnergy?: number | null;
  moodScale?: number | null;
  sleepQuality?: number | null;
  hrvTrend?: number | null;
  activity?: number | null;
  brainHealth?: number | null;
}

export interface ScoreResult {
  score: number | null;
  tier: Tier;
  factors: Record<string, number>;
  narrative: string;
  label: string;
  color: string;
}

// ─── Tier Detection ──────────────────────────────────────────────────────────

/** Detect the highest available tier from input data */
export function detectTier(inputs: ScoreInputs): Tier | null {
  const hasStressOrValence = inputs.stress != null || inputs.valence != null;
  const hasSleep = inputs.sleepQuality != null;
  const hasBrain = inputs.brainHealth != null;

  if (hasBrain && hasSleep) return "eeg_health_voice";
  if (hasSleep && hasStressOrValence) return "health_voice";
  if (hasStressOrValence) return "voice";
  return null;
}
```

- [ ] **Step 7: Run tests — verify pass**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npm test -- --run client/src/test/lib/inner-score.test.ts`

- [ ] **Step 8: Write failing tests for computeScore**

Add to test file:

```typescript
import { computeScore } from "@/lib/inner-score";

describe("computeScore", () => {
  it("computes Tier 1 (voice only)", () => {
    const result = computeScore({ stress: 0.3, valence: 0.4 });
    expect(result.tier).toBe("voice");
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(100);
    expect(result.factors).toHaveProperty("stress_inverse");
    expect(result.factors).toHaveProperty("valence");
  });

  it("computes Tier 2 (health + voice)", () => {
    const result = computeScore({ stress: 0.3, valence: 0.4, sleepQuality: 80, hrvTrend: 60, activity: 50 });
    expect(result.tier).toBe("health_voice");
    expect(result.factors).toHaveProperty("sleep_quality");
  });

  it("computes Tier 3 (EEG + health + voice)", () => {
    const result = computeScore({ stress: 0.2, valence: 0.5, sleepQuality: 85, hrvTrend: 70, activity: 60, brainHealth: 75 });
    expect(result.tier).toBe("eeg_health_voice");
    expect(result.factors).toHaveProperty("brain_health");
  });

  it("returns building state when no data", () => {
    const result = computeScore({});
    expect(result.score).toBeNull();
    expect(result.label).toBe("Building");
  });

  it("redistributes weights when optional factors missing in Tier 2", () => {
    const withHrv = computeScore({ stress: 0.3, valence: 0.4, sleepQuality: 80, hrvTrend: 60, activity: 50 });
    const withoutHrv = computeScore({ stress: 0.3, valence: 0.4, sleepQuality: 80 });
    expect(withoutHrv.tier).toBe("health_voice");
    expect(withoutHrv.score).toBeGreaterThan(0);
  });

  it("labels Thriving for score >= 80", () => {
    const result = computeScore({ stress: 0.05, valence: 0.9, sleepQuality: 95, hrvTrend: 90, activity: 85 });
    expect(result.label).toBe("Thriving");
  });

  it("labels Low for score < 40", () => {
    const result = computeScore({ stress: 0.9, valence: -0.8 });
    expect(result.label).toBe("Low");
  });
});
```

- [ ] **Step 9: Implement computeScore + getScoreLabel + redistributeWeights**

Add to `client/src/lib/inner-score.ts`:

```typescript
// ─── Weights ─────────────────────────────────────────────────────────────────

const TIER_WEIGHTS: Record<Tier, Record<string, number>> = {
  voice: { stress_inverse: 0.4, valence: 0.4, energy: 0.2 },
  health_voice: { sleep_quality: 0.35, hrv_trend: 0.2, stress_inverse: 0.2, valence: 0.15, activity: 0.1 },
  eeg_health_voice: { sleep_quality: 0.3, brain_health: 0.25, hrv_trend: 0.15, stress_inverse: 0.15, valence: 0.1, activity: 0.05 },
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

export function getScoreLabel(score: number): string {
  if (score >= 80) return "Thriving";
  if (score >= 60) return "Good";
  if (score >= 40) return "Steady";
  return "Low";
}

export function getScoreColor(score: number): string {
  if (score >= 80) return "var(--success)";
  if (score >= 60) return "var(--primary)";
  if (score >= 40) return "var(--warning)";
  return "var(--destructive)";
}

function redistributeWeights(
  weights: Record<string, number>,
  presentKeys: string[],
): Record<string, number> {
  const present = Object.entries(weights).filter(([k]) => presentKeys.includes(k));
  const totalPresent = present.reduce((s, [, w]) => s + w, 0);
  if (totalPresent <= 0) return {};
  const result: Record<string, number> = {};
  for (const [k, w] of present) {
    result[k] = w / totalPresent;
  }
  return result;
}

// ─── Score Computation ───────────────────────────────────────────────────────

export function computeScore(inputs: ScoreInputs): ScoreResult {
  const tier = detectTier(inputs);
  if (!tier) {
    return { score: null, tier: "voice", factors: {}, narrative: "", label: "Building", color: "var(--muted)" };
  }

  // Extract normalized factor values
  const allFactors: Record<string, number> = {};
  if (inputs.stress != null) allFactors.stress_inverse = normalizeStress(inputs.stress);
  if (inputs.valence != null) allFactors.valence = normalizeValence(inputs.valence);
  allFactors.energy = normalizeEnergy({ arousal: inputs.arousal ?? undefined, moodEnergy: inputs.moodEnergy ?? undefined, moodScale: inputs.moodScale ?? undefined });
  if (inputs.sleepQuality != null) allFactors.sleep_quality = Math.round(Math.max(0, Math.min(100, inputs.sleepQuality)));
  if (inputs.hrvTrend != null) allFactors.hrv_trend = Math.round(Math.max(0, Math.min(100, inputs.hrvTrend)));
  if (inputs.activity != null) allFactors.activity = Math.round(Math.max(0, Math.min(100, inputs.activity)));
  if (inputs.brainHealth != null) allFactors.brain_health = Math.round(Math.max(0, Math.min(100, inputs.brainHealth)));

  // Get tier weights and redistribute for missing optional factors
  const baseWeights = TIER_WEIGHTS[tier];
  const presentKeys = Object.keys(allFactors).filter((k) => k in baseWeights);
  const weights = redistributeWeights(baseWeights, presentKeys);

  // Weighted sum
  let score = 0;
  const usedFactors: Record<string, number> = {};
  for (const [key, weight] of Object.entries(weights)) {
    const val = allFactors[key] ?? 50;
    score += val * weight;
    usedFactors[key] = allFactors[key] ?? 50;
  }
  score = Math.round(Math.max(0, Math.min(100, score)));

  return {
    score,
    tier,
    factors: usedFactors,
    narrative: "", // filled by computeNarrative
    label: getScoreLabel(score),
    color: getScoreColor(score),
  };
}
```

- [ ] **Step 10: Run tests — verify pass**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npm test -- --run client/src/test/lib/inner-score.test.ts`

- [ ] **Step 11: Write failing tests for narrative generation**

Add to test file:

```typescript
import { computeNarrative } from "@/lib/inner-score";

describe("computeNarrative", () => {
  it("mentions highest and lowest factor", () => {
    const narrative = computeNarrative({ sleep_quality: 90, stress_inverse: 40, valence: 60 }, 63, null);
    expect(narrative).toContain("sleep");
    expect(narrative).toContain("stress");
  });
  it("says well-balanced when all within 10 points", () => {
    const narrative = computeNarrative({ sleep_quality: 70, stress_inverse: 65, valence: 68 }, 68, null);
    expect(narrative).toContain("well-balanced");
  });
  it("mentions improvement when delta > 10", () => {
    const narrative = computeNarrative({ sleep_quality: 80 }, 80, 15);
    expect(narrative).toContain("improvement");
  });
  it("mentions dip when delta < -10", () => {
    const narrative = computeNarrative({ sleep_quality: 50 }, 50, -12);
    expect(narrative).toContain("Dip");
  });
});
```

- [ ] **Step 12: Implement computeNarrative**

Add to `client/src/lib/inner-score.ts`:

```typescript
// ─── Narrative ───────────────────────────────────────────────────────────────

const FACTOR_LABELS: Record<string, string> = {
  sleep_quality: "sleep",
  stress_inverse: "stress levels",
  valence: "mood",
  energy: "energy",
  hrv_trend: "heart rate variability",
  activity: "activity",
  brain_health: "brain health",
};

export function computeNarrative(
  factors: Record<string, number>,
  score: number,
  delta: number | null,
): string {
  const entries = Object.entries(factors);
  if (entries.length === 0) return "";

  // Check delta first
  if (delta != null && delta > 10) return "Strong improvement from yesterday.";
  if (delta != null && delta < -10) return "Dip from yesterday — check what changed.";

  const sorted = [...entries].sort((a, b) => b[1] - a[1]);
  const highest = sorted[0];
  const lowest = sorted[sorted.length - 1];

  // Check if all within 10 points
  if (highest[1] - lowest[1] <= 10) {
    return "You're well-balanced across the board today.";
  }

  const highLabel = FACTOR_LABELS[highest[0]] ?? highest[0];
  const lowLabel = FACTOR_LABELS[lowest[0]] ?? lowest[0];
  return `Good ${highLabel} is carrying you today, but ${lowLabel} could use attention.`;
}
```

- [ ] **Step 13: Run tests — verify all pass**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npm test -- --run client/src/test/lib/inner-score.test.ts`

- [ ] **Step 14: Commit**

```bash
git add client/src/lib/inner-score.ts client/src/test/lib/inner-score.test.ts
git commit -m "feat: add Inner Score computation library — 3-tier adaptive formula with narrative"
```

---

### Task 2: Inner Score Card Component

**Files:**
- Create: `client/src/components/inner-score-card.tsx`
- Test: `client/src/test/components/inner-score-card.test.tsx`

- [ ] **Step 1: Write failing component tests**

```typescript
// client/src/test/components/inner-score-card.test.tsx
import React from "react";
import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { InnerScoreCard } from "@/components/inner-score-card";

describe("InnerScoreCard", () => {
  it("renders without crashing", () => {
    renderWithProviders(<InnerScoreCard score={null} tier="voice" factors={{}} narrative="" delta={null} trend={[]} />);
    expect(screen.getByTestId("inner-score-card")).toBeInTheDocument();
  });

  it("shows building state when score is null", () => {
    renderWithProviders(<InnerScoreCard score={null} tier="voice" factors={{}} narrative="" delta={null} trend={[]} />);
    expect(screen.getByTestId("inner-score-building")).toBeInTheDocument();
    expect(screen.getByText(/voice check-in/i)).toBeInTheDocument();
  });

  it("shows score number when score is provided", () => {
    renderWithProviders(<InnerScoreCard score={72} tier="health_voice" factors={{ sleep_quality: 85 }} narrative="Good sleep." delta={5} trend={[65, 68, 72]} />);
    expect(screen.getByText("72")).toBeInTheDocument();
  });

  it("shows Inner Score label", () => {
    renderWithProviders(<InnerScoreCard score={72} tier="health_voice" factors={{}} narrative="" delta={null} trend={[]} />);
    expect(screen.getByText("Inner Score")).toBeInTheDocument();
  });

  it("shows Good label for score 72", () => {
    renderWithProviders(<InnerScoreCard score={72} tier="health_voice" factors={{}} narrative="" delta={null} trend={[]} />);
    expect(screen.getByText("Good")).toBeInTheDocument();
  });

  it("shows Thriving for score 85", () => {
    renderWithProviders(<InnerScoreCard score={85} tier="eeg_health_voice" factors={{}} narrative="" delta={null} trend={[]} />);
    expect(screen.getByText("Thriving")).toBeInTheDocument();
  });

  it("shows Low for score 30", () => {
    renderWithProviders(<InnerScoreCard score={30} tier="voice" factors={{}} narrative="" delta={null} trend={[]} />);
    expect(screen.getByText("Low")).toBeInTheDocument();
  });

  it("shows tier confidence label for health_voice", () => {
    renderWithProviders(<InnerScoreCard score={72} tier="health_voice" factors={{}} narrative="" delta={null} trend={[]} />);
    expect(screen.getByText(/sleep, body, and mood/i)).toBeInTheDocument();
  });

  it("shows tier confidence label for voice", () => {
    renderWithProviders(<InnerScoreCard score={60} tier="voice" factors={{}} narrative="" delta={null} trend={[]} />);
    expect(screen.getByText(/how you sound/i)).toBeInTheDocument();
  });

  it("shows delta when provided", () => {
    renderWithProviders(<InnerScoreCard score={72} tier="voice" factors={{}} narrative="" delta={5} trend={[]} />);
    expect(screen.getByText("+5")).toBeInTheDocument();
  });

  it("renders SVG gauge", () => {
    renderWithProviders(<InnerScoreCard score={72} tier="voice" factors={{}} narrative="" delta={null} trend={[]} />);
    expect(screen.getByTestId("inner-score-gauge")).toBeInTheDocument();
  });

  it("shows factor bars after tap", async () => {
    const { user } = renderWithProviders(
      <InnerScoreCard score={72} tier="health_voice" factors={{ sleep_quality: 85, stress_inverse: 58 }} narrative="Good sleep." delta={5} trend={[]} />
    );
    await user.click(screen.getByTestId("inner-score-card"));
    expect(screen.getByText("Good sleep.")).toBeInTheDocument();
    expect(screen.getByText(/sleep/i)).toBeInTheDocument();
  });

  it("has accessible aria-label on gauge", () => {
    renderWithProviders(<InnerScoreCard score={72} tier="voice" factors={{}} narrative="" delta={null} trend={[]} />);
    expect(screen.getByLabelText(/inner score: 72/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npm test -- --run client/src/test/components/inner-score-card.test.tsx`
Expected: FAIL — module not found

- [ ] **Step 3: Implement InnerScoreCard component**

Create `client/src/components/inner-score-card.tsx` with:
- SVG arc gauge (220px, 270-degree, emerald gradient stroke, 12px width)
- Score number centered (52px, `var(--font-sans)`)
- Label below gauge ("Inner Score")
- Score label ("Thriving"/"Good"/"Steady"/"Low")
- Tier confidence text
- Delta indicator ("+5" green or "-3" red)
- 7-day sparkline (56×20px SVG polyline)
- Building state: pulsing opacity, "—" text, CTA
- Tap-to-expand: Framer Motion `AnimatePresence`, factor bars + narrative
- Props: `score, tier, factors, narrative, delta, trend`
- Data testids: `inner-score-card`, `inner-score-gauge`, `inner-score-building`
- All colors via CSS variables, no hardcoded hex
- `aria-label` on SVG gauge: `"Inner Score: {score} out of 100, {label}"`
- Factor bars have `role="progressbar"` with `aria-valuenow`, `aria-valuemin=0`, `aria-valuemax=100`
- Tap-to-expand is keyboard accessible (Enter/Space toggles via `onClick` + `onKeyDown`)
- Import `{ motion, AnimatePresence }` from `"framer-motion"` for tap-to-expand

- [ ] **Step 4: Run tests — verify all pass**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npm test -- --run client/src/test/components/inner-score-card.test.tsx`

- [ ] **Step 5: Commit**

```bash
git add client/src/components/inner-score-card.tsx client/src/test/components/inner-score-card.test.tsx
git commit -m "feat: add Inner Score card component — hero gauge, building state, tap-to-expand"
```

---

### Task 3: Database Schema + API Endpoints

**Files:**
- Modify: `shared/schema.ts` (add innerScores table)
- Modify: `server/routes.ts` (add 2 endpoints)
- Modify: `api/[...path].ts` (add routes to Vercel catch-all)

- [ ] **Step 1: Add innerScores table to schema**

In `shared/schema.ts`, add after the existing tables:

```typescript
export const innerScores = pgTable("inner_scores", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: text("user_id").notNull(),
  score: integer("score").notNull(),
  tier: text("tier").notNull(),
  factors: jsonb("factors").notNull().default({}),
  narrative: text("narrative"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertInnerScoreSchema = createInsertSchema(innerScores);
```

- [ ] **Step 2: Push schema to database**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx drizzle-kit push`

- [ ] **Step 3: Add Express endpoints**

In `server/routes.ts`, add two new routes:

```typescript
// GET /api/inner-score/:userId — compute or return cached today's score
app.get("/api/inner-score/:userId", async (req, res) => {
  const { userId } = req.params;
  // Uses `db` imported at top of routes.ts: import { db } from "./db";

  // Check cache (< 4h old)
  const fourHoursAgo = new Date(Date.now() - 4 * 60 * 60 * 1000);
  const [cached] = await db.select().from(schema.innerScores)
    .where(and(eq(schema.innerScores.userId, userId), gte(schema.innerScores.createdAt, fourHoursAgo)))
    .orderBy(desc(schema.innerScores.createdAt)).limit(1);

  if (cached) {
    // Fetch 7-day trend
    const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
    const history = await db.select().from(schema.innerScores)
      .where(and(eq(schema.innerScores.userId, userId), gte(schema.innerScores.createdAt, sevenDaysAgo)))
      .orderBy(asc(schema.innerScores.createdAt));
    const trend = history.map(h => h.score);
    const yesterday = history.length >= 2 ? history[history.length - 2].score : null;
    const delta = yesterday != null ? cached.score - yesterday : null;

    return res.json({
      score: cached.score, label: getLabel(cached.score), color: getColor(cached.score),
      tier: cached.tier, confidence: getTierLabel(cached.tier),
      factors: cached.factors, narrative: cached.narrative,
      delta, trend,
    });
  }

  // No cache — return building state (client computes + POST to save)
  res.json({ score: null, state: "building", cta: "Do a voice check-in to get your Inner Score" });
});

// GET /api/inner-score/:userId/history?days=30
app.get("/api/inner-score/:userId/history", async (req, res) => {
  const { userId } = req.params;
  const days = parseInt(req.query.days as string) || 30;
  const since = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
  // Uses `db` imported at top of routes.ts: import { db } from "./db";
  const rows = await db.select().from(schema.innerScores)
    .where(and(eq(schema.innerScores.userId, userId), gte(schema.innerScores.createdAt, since)))
    .orderBy(desc(schema.innerScores.createdAt));
  res.json({ scores: rows.map(r => ({ date: r.createdAt.toISOString().slice(0, 10), score: r.score, tier: r.tier })) });
});
```

Helper functions (add near the route definitions):

```typescript
function getLabel(score: number) { return score >= 80 ? "Thriving" : score >= 60 ? "Good" : score >= 40 ? "Steady" : "Low"; }
function getColor(score: number) { return score >= 80 ? "#34D399" : score >= 60 ? "#2DD4BF" : score >= 40 ? "#F59E0B" : "#F87171"; }
function getTierLabel(tier: string) {
  if (tier === "eeg_health_voice") return "Based on your brain, body, and mood";
  if (tier === "health_voice") return "Based on your sleep, body, and mood";
  return "Based on how you sound today";
}
```

- [ ] **Step 4: Add POST endpoint to persist scores**

In `server/routes.ts`, add after the GET routes. Also add `innerScores` to the import from `@shared/schema` at the top of the file:

```typescript
// POST /api/inner-score/:userId — persist a computed score
app.post("/api/inner-score/:userId", async (req, res) => {
  const { userId } = req.params;
  const { score, tier, factors, narrative } = req.body;
  if (score == null || !tier) return res.status(400).json({ error: "score and tier required" });
  const [row] = await db.insert(innerScores).values({
    userId, score, tier, factors: factors ?? {}, narrative: narrative ?? null,
  }).returning();
  res.status(201).json(row);
});
```

- [ ] **Step 5: Add routes to Vercel catch-all**

In `api/[...path].ts`, find the route matching section (uses `segs` array). Add:

```typescript
// Inner Score routes
if (s0 === "inner-score" && s1) {
  if (segs.length === 3 && segs[2] === "history") return await innerScoreHistory(req, res, s1);
  if (req.method === "POST") return await innerScorePost(req, res, s1);
  return await innerScoreGet(req, res, s1);
}
```

Then add the handler functions following the existing pattern (use `_dbGetter()` for db access, import `innerScores` from schema):

```typescript
async function innerScoreGet(req: VercelRequest, res: VercelResponse, userId: string) {
  const db = _dbGetter();
  const fourHoursAgo = new Date(Date.now() - 4 * 3600_000);
  const [cached] = await db.select().from(schema.innerScores)
    .where(and(eq(schema.innerScores.userId, userId), gte(schema.innerScores.createdAt, fourHoursAgo)))
    .orderBy(desc(schema.innerScores.createdAt)).limit(1);
  if (!cached) return success(res, { score: null, state: "building", cta: "Do a voice check-in to get your Inner Score" });
  const sevenDaysAgo = new Date(Date.now() - 7 * 86400_000);
  const history = await db.select().from(schema.innerScores)
    .where(and(eq(schema.innerScores.userId, userId), gte(schema.innerScores.createdAt, sevenDaysAgo)))
    .orderBy(asc(schema.innerScores.createdAt));
  const trend = history.map(h => h.score);
  const yesterday = history.length >= 2 ? history[history.length - 2].score : null;
  const delta = yesterday != null ? cached.score - yesterday : null;
  return success(res, { score: cached.score, label: getLabel(cached.score), tier: cached.tier, confidence: getTierLabel(cached.tier), factors: cached.factors, narrative: cached.narrative, delta, trend });
}

async function innerScorePost(req: VercelRequest, res: VercelResponse, userId: string) {
  const body = await parseRequestBody(req) as any;
  if (body?.score == null || !body?.tier) return badRequest(res, "score and tier required");
  const db = _dbGetter();
  const [row] = await db.insert(schema.innerScores).values({ userId, score: body.score, tier: body.tier, factors: body.factors ?? {}, narrative: body.narrative ?? null }).returning();
  return success(res, row, 201);
}

async function innerScoreHistory(req: VercelRequest, res: VercelResponse, userId: string) {
  const url = new URL(req.url!, `http://${req.headers.host}`);
  const days = parseInt(url.searchParams.get("days") ?? "30");
  const since = new Date(Date.now() - days * 86400_000);
  const db = _dbGetter();
  const rows = await db.select().from(schema.innerScores)
    .where(and(eq(schema.innerScores.userId, userId), gte(schema.innerScores.createdAt, since)))
    .orderBy(desc(schema.innerScores.createdAt));
  return success(res, { scores: rows.map(r => ({ date: r.createdAt.toISOString().slice(0, 10), score: r.score, tier: r.tier })) });
}

function getLabel(s: number) { return s >= 80 ? "Thriving" : s >= 60 ? "Good" : s >= 40 ? "Steady" : "Low"; }
function getTierLabel(t: string) { return t === "eeg_health_voice" ? "Based on your brain, body, and mood" : t === "health_voice" ? "Based on your sleep, body, and mood" : "Based on how you sound today"; }
```

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npm test -- --run`
Expected: All 152+ files pass

- [ ] **Step 6: Commit**

```bash
git add shared/schema.ts server/routes.ts api/\[...path\].ts
git commit -m "feat: add Inner Score API endpoints and database schema"
```

---

### Task 4: Wire Into Today Page

**Files:**
- Modify: `client/src/pages/today.tsx`

- [ ] **Step 1: Import InnerScoreCard and computation**

Add imports at top of `client/src/pages/today.tsx`:

```typescript
import { InnerScoreCard } from "@/components/inner-score-card";
import { computeScore, computeNarrative, type ScoreInputs } from "@/lib/inner-score";
```

- [ ] **Step 2: Remove 3 ScoreCircle instances**

Find the section around line 1346-1368 that renders three `ScoreCircle` components (Recovery, Sleep, Strain) and remove it.

- [ ] **Step 3: Add Inner Score computation and rendering**

Replace the removed ScoreCircles with:

```tsx
{/* ── Inner Score Hero ── */}
{(() => {
  const inputs: ScoreInputs = {
    stress: brainTotals?.avgStress ?? null,
    valence: brainTotals?.avgValence ?? null,
    sleepQuality: latestPayload?.sleep_efficiency ?? null,
    hrvTrend: latestPayload?.hrv_sdnn ?? null,
    activity: latestPayload?.steps_today ? Math.min(100, (latestPayload.steps_today / 10000) * 100) : null,
  };
  const result = computeScore(inputs);
  const narrative = result.score != null
    ? computeNarrative(result.factors, result.score, null)
    : "";

  // Persist score to DB (fire-and-forget) when computed
  React.useEffect(() => {
    if (result.score != null && userId) {
      fetch(`/api/inner-score/${userId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ score: result.score, tier: result.tier, factors: result.factors, narrative }),
      }).catch(() => {});
    }
  }, [result.score, result.tier]);

  return (
    <InnerScoreCard
      score={result.score}
      tier={result.tier}
      factors={result.factors}
      narrative={narrative}
      delta={null}
      trend={[]}
    />
  );
})()}
```

Note: The `useEffect` for persisting must be lifted outside the IIFE into the component body (since hooks can't be called inside IIFEs). Refactor: extract the inputs/computation into a `useMemo` at the component level, and the persist logic into a `useEffect` at the component level.

- [ ] **Step 4: Run full test suite**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npm test -- --run`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add client/src/pages/today.tsx
git commit -m "feat: wire Inner Score into Today page — replace 3 score circles with single hero gauge"
```

---

### Task 5: Final Integration + Push

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npm test -- --run`
Expected: All tests pass

- [ ] **Step 2: Push to GitHub**

```bash
git push
```

- [ ] **Step 3: Deploy to Vercel**

```bash
vercel --prod
```

- [ ] **Step 4: Build APK**

```bash
npx vite build && npx cap sync android
export JAVA_HOME=$(brew --prefix openjdk@21) && export PATH="$JAVA_HOME/bin:$PATH"
cd android && ./gradlew assembleDebug
```

- [ ] **Step 5: Upload APK to release**

```bash
gh release upload v3.1.0 android/app/build/outputs/apk/debug/app-debug.apk --clobber
```
