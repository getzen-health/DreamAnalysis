# Emotional Fitness Score (EFS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a daily 0-100 Emotional Fitness Score from 5 vital signs (Resilience, Regulation, Awareness, Range, Stability), with a full-page UI, daily insights, share card, and dashboard integration.

**Architecture:** Express.js GET endpoint orchestrates scoring by querying existing ML endpoints + DB tables. New Drizzle table stores daily scores. React page with arc gauge, vital cards, insight banner, history chart, and PNG share card. No new ML models — pure orchestration + UI.

**Tech Stack:** TypeScript, Drizzle ORM (PostgreSQL/Neon), Express.js, React 18, Recharts, wouter, TanStack Query, Canvas 2D API, shadcn/ui, Tailwind CSS, lucide-react icons.

**Spec:** `docs/superpowers/specs/2026-03-19-emotional-fitness-score-design.md`

---

## File Map

### New Files

| File | Responsibility |
|---|---|
| `shared/schema.ts` (modify) | Add `emotionalFitnessScores` table + insert schema + types |
| `server/routes.ts` (modify) | Add `GET /api/brain/emotional-fitness/:userId` endpoint |
| `server/efs-compute.ts` (create) | EFS scoring logic: fetch vitals, compute composite, generate insight |
| `client/src/pages/emotional-fitness.tsx` (create) | Main EFS page |
| `client/src/components/efs-hero-score.tsx` (create) | Arc gauge hero component |
| `client/src/components/efs-vital-card.tsx` (create) | Individual vital sign card |
| `client/src/components/efs-insight-banner.tsx` (create) | Daily insight display |
| `client/src/components/efs-history-chart.tsx` (create) | 30/60/90 day timeline |
| `client/src/components/efs-share-card.tsx` (create) | Canvas PNG export |
| `client/src/components/efs-mini-card.tsx` (create) | Dashboard widget |
| `client/src/lib/ml-api.ts` (modify) | Add `getEmotionalFitness()` API helper |
| `client/src/App.tsx` (modify) | Add `/emotional-fitness` route |

---

## Task 1: Database Schema

**Files:**
- Modify: `shared/schema.ts` (after `emotionCalibration` table, ~line 820)

- [ ] **Step 1: Add `emotionalFitnessScores` table to schema**

Add after the `emotionCalibration` table block in `shared/schema.ts`:

```typescript
export const emotionalFitnessScores = pgTable("emotional_fitness_scores", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }).notNull(),
  date: date("date").notNull(),
  score: integer("score"),
  resilience: integer("resilience"),
  regulation: integer("regulation"),
  awareness: integer("awareness"),
  range: integer("range"),
  stability: integer("stability"),
  dailyInsightText: text("daily_insight_text"),
  dailyInsightType: varchar("daily_insight_type", { length: 50 }),
  confidence: varchar("confidence", { length: 20 }).notNull().default("full"),
  computedAt: timestamp("computed_at").defaultNow().notNull(),
}, (table) => [
  uniqueIndex("efs_user_date_idx").on(table.userId, table.date),
  index("efs_user_idx").on(table.userId),
]);

export const insertEmotionalFitnessScoreSchema = createInsertSchema(emotionalFitnessScores).omit({
  id: true,
  computedAt: true,
});
export type EmotionalFitnessScore = typeof emotionalFitnessScores.$inferSelect;
export type InsertEmotionalFitnessScore = z.infer<typeof insertEmotionalFitnessScoreSchema>;
```

- [ ] **Step 2: Push schema to database**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx drizzle-kit push`
Expected: Migration applied, new table created.

- [ ] **Step 3: Verify table exists**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx drizzle-kit studio` (or check Neon dashboard)
Expected: `emotional_fitness_scores` table visible with all columns.

- [ ] **Step 4: Commit**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add shared/schema.ts
git commit -m "feat(efs): add emotionalFitnessScores table schema"
```

---

## Task 2: EFS Compute Engine (Server)

**Files:**
- Create: `server/efs-compute.ts`

This is the core scoring logic. It queries existing data, computes 5 vitals, blends into composite score, and generates a daily insight.

- [ ] **Step 1: Create `server/efs-compute.ts` with type definitions and constants**

```typescript
import { db } from "./db";
import { emotionalFitnessScores, emotionCalibration, emotionReadings, type EmotionalFitnessScore } from "@shared/schema";
import { eq, and, gte, lte, desc, sql, isNotNull } from "drizzle-orm";

// ---------- Types ----------

interface VitalScore {
  score: number | null;
  status: "available" | "unavailable";
  insight: string;
  unlockHint?: string;
}

interface EFSResult {
  score: number | null;
  color: "green" | "amber" | "red" | null;
  label: "Strong" | "Developing" | "Needs Attention" | null;
  confidence: "full" | "early_estimate" | "building";
  progress?: { daysTracked: number; daysRequired: number; percentage: number; message: string };
  trend: { direction: "up" | "down" | "stable"; delta: number; period: string } | null;
  vitals: {
    resilience: VitalScore & { history: { date: string; score: number }[] };
    regulation: VitalScore & { history: { date: string; score: number }[] };
    awareness: VitalScore & { history: { date: string; score: number }[] };
    range: VitalScore & { history: { date: string; score: number }[] };
    stability: VitalScore & { history: { date: string; score: number }[] };
  };
  dailyInsight: { text: string; type: string; actionNudge: string } | null;
  computedAt: string;
}

// ---------- Constants ----------

const WEIGHTS = { resilience: 0.25, regulation: 0.20, awareness: 0.25, range: 0.15, stability: 0.15 };

const INSIGHT_NUDGES: Record<string, string> = {
  awareness_gap: "Try naming your emotion out loud during your next check-in.",
  improvement: "Keep up the momentum — consistency compounds.",
  range_expansion: "When you feel 'bad', try to be more specific: frustrated? disappointed? anxious?",
  pattern: "Notice what's different about that time of week. Small changes can shift the pattern.",
  milestone: "You've earned this. Consistency is the hardest part.",
  transformation: "Look how far you've come. This is real, measurable growth.",
};

const VITAL_EXPLANATIONS: Record<string, { explanation: string; tips: string }> = {
  resilience: {
    explanation: "How quickly your emotions return to baseline after a negative event.",
    tips: "Practice 4-7-8 breathing during stressful moments. Regular sleep improves recovery speed.",
  },
  regulation: {
    explanation: "How effectively you manage emotional intensity when it spikes.",
    tips: "Try the biofeedback exercises in the app. Even 2 minutes of guided breathing strengthens regulation.",
  },
  awareness: {
    explanation: "How accurately you perceive your own emotions compared to what your brain and voice show.",
    tips: "During check-ins, pause before answering. Notice body sensations first, then name the emotion.",
  },
  range: {
    explanation: "How many distinct emotions you can identify and differentiate. Higher range = better regulation.",
    tips: "When you feel 'bad', try to be more specific: frustrated? disappointed? anxious? lonely?",
  },
  stability: {
    explanation: "How consistent your emotional baseline is day to day. Higher stability = less volatility.",
    tips: "Consistent sleep schedule, regular meals, and daily routines all contribute to emotional stability.",
  },
};
```

- [ ] **Step 2: Add helper functions for color/label and date utilities**

Append to `server/efs-compute.ts`:

```typescript
function scoreColor(score: number): "green" | "amber" | "red" {
  if (score >= 70) return "green";
  if (score >= 40) return "amber";
  return "red";
}

function scoreLabel(score: number): "Strong" | "Developing" | "Needs Attention" {
  if (score >= 70) return "Strong";
  if (score >= 40) return "Developing";
  return "Needs Attention";
}

function todayUTC(): string {
  return new Date().toISOString().slice(0, 10);
}

function daysAgoUTC(days: number): Date {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - days);
  d.setUTCHours(0, 0, 0, 0);
  return d;
}

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}
```

- [ ] **Step 3: Add vital computation functions**

Append to `server/efs-compute.ts`:

```typescript
async function computeResilience(userId: string, mlBaseUrl: string): Promise<VitalScore> {
  try {
    // 1. Get recovery_speed from emotional genome
    const genomeRes = await fetch(`${mlBaseUrl}/api/emotional-genome/profile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, samples: [] }),
    });

    let genomeScore = 50; // default
    if (genomeRes.ok) {
      const genome = await genomeRes.json();
      if (genome.traits?.recovery_speed != null) {
        genomeScore = Math.round(genome.traits.recovery_speed * 100);
      }
    }

    // 2. Event-level recovery from emotionReadings
    const fourteenDaysAgo = daysAgoUTC(14);
    const readings = await db
      .select({ valence: emotionReadings.valence, createdAt: emotionReadings.timestamp })
      .from(emotionReadings)
      .where(and(
        eq(emotionReadings.userId, userId),
        gte(emotionReadings.timestamp, fourteenDaysAgo),
        isNotNull(emotionReadings.valence),
      ))
      .orderBy(emotionReadings.timestamp);

    if (readings.length < 5) {
      return { score: genomeScore, status: "available", insight: "Building resilience profile..." };
    }

    // Detect valence dips (below -0.3) and measure recovery time
    const valences = readings.map(r => ({ v: r.valence as number, t: r.createdAt!.getTime() }));
    let totalRecoveryMs = 0;
    let dipCount = 0;
    const baseline = valences.reduce((s, r) => s + r.v, 0) / valences.length;

    for (let i = 0; i < valences.length; i++) {
      if (valences[i].v < -0.3) {
        // Find when valence returns to baseline
        for (let j = i + 1; j < valences.length; j++) {
          if (valences[j].v >= baseline) {
            totalRecoveryMs += valences[j].t - valences[i].t;
            dipCount++;
            i = j; // skip to recovery point
            break;
          }
        }
      }
    }

    let eventScore = 75; // default if no dips detected
    if (dipCount > 0) {
      const avgRecoveryMin = (totalRecoveryMs / dipCount) / 60000;
      if (avgRecoveryMin <= 5) eventScore = 100;
      else if (avgRecoveryMin <= 10) eventScore = 75;
      else if (avgRecoveryMin <= 20) eventScore = 50;
      else eventScore = 25;
    }

    const score = clamp(Math.round(genomeScore * 0.6 + eventScore * 0.4), 0, 100);
    const insight = dipCount > 0
      ? `You recovered from ${dipCount} emotional dip${dipCount > 1 ? "s" : ""} in the last 14 days`
      : "No significant emotional dips detected — steady baseline";

    return { score, status: "available", insight };
  } catch {
    return { score: null, status: "unavailable", insight: "", unlockHint: "Track for 3+ days" };
  }
}

async function computeRegulation(userId: string, mlBaseUrl: string): Promise<VitalScore> {
  try {
    const res = await fetch(`${mlBaseUrl}/api/emotion-regulation/summary?user_id=${encodeURIComponent(userId)}`);
    if (!res.ok) throw new Error("No regulation data");
    const data = await res.json();

    if (!data.session_count || data.session_count < 1) {
      // Fallback: infer from stress-to-calm transitions in emotionReadings
      const fourteenDaysAgo = daysAgoUTC(14);
      const readings = await db
        .select({ stress: emotionReadings.stress, createdAt: emotionReadings.timestamp })
        .from(emotionReadings)
        .where(and(
          eq(emotionReadings.userId, userId),
          gte(emotionReadings.timestamp, fourteenDaysAgo),
          isNotNull(emotionReadings.stress),
        ))
        .orderBy(emotionReadings.timestamp);

      if (readings.length < 5) {
        return { score: null, status: "unavailable", insight: "", unlockHint: "Track for 3+ days" };
      }

      // Count stress→calm transitions (stress > 0.6 followed by stress < 0.3)
      let transitions = 0;
      let highStressEpisodes = 0;
      for (let i = 0; i < readings.length - 1; i++) {
        const s = readings[i].stress as number;
        if (s > 0.6) {
          highStressEpisodes++;
          for (let j = i + 1; j < readings.length; j++) {
            if ((readings[j].stress as number) < 0.3) {
              transitions++;
              i = j;
              break;
            }
          }
        }
      }

      const rate = highStressEpisodes > 0 ? transitions / highStressEpisodes : 0.5;
      const score = clamp(Math.round(rate * 100), 0, 100);
      return { score, status: "available", insight: `${transitions} stress recovery events in 14 days` };
    }

    // Use regulation summary: mean_regulation_score is 0-1
    const score = clamp(Math.round(data.mean_regulation_score * 100), 0, 100);
    return {
      score,
      status: "available",
      insight: `Regulation success rate across ${data.session_count} sessions`,
    };
  } catch {
    return { score: null, status: "unavailable", insight: "", unlockHint: "Track for 3+ days" };
  }
}

async function computeAwareness(userId: string): Promise<VitalScore> {
  try {
    const rows = await db
      .select({ awarenessScore: emotionCalibration.awarenessScore, reporterType: emotionCalibration.reporterType })
      .from(emotionCalibration)
      .where(eq(emotionCalibration.userId, userId))
      .orderBy(desc(emotionCalibration.recordedAt))
      .limit(1);

    if (!rows.length || rows[0].awarenessScore == null) {
      return { score: null, status: "unavailable", insight: "", unlockHint: "Complete a calibration session" };
    }

    const score = clamp(Math.round(rows[0].awarenessScore), 0, 100);
    const type = rows[0].reporterType || "unknown";
    const typeLabel = type === "suppressor" ? "Suppressor" : type === "amplifier" ? "Amplifier" : type === "accurate" ? "Accurate" : "Inconsistent";
    const insight = type === "accurate"
      ? "Your self-reports closely match your brain activity"
      : `You're a ${typeLabel} — your reports ${type === "suppressor" ? "understate" : "overstate"} what your brain shows`;

    return { score, status: "available", insight };
  } catch {
    return { score: null, status: "unavailable", insight: "", unlockHint: "Complete a calibration session" };
  }
}

async function computeRange(userId: string, mlBaseUrl: string): Promise<VitalScore> {
  try {
    const res = await fetch(`${mlBaseUrl}/api/emotional-granularity/compute/${encodeURIComponent(userId)}`);
    if (!res.ok) throw new Error("No granularity data");
    const data = await res.json();

    if (!data.ready || data.granularity == null) {
      return { score: null, status: "unavailable", insight: "", unlockHint: "Do 5+ check-ins" };
    }

    const score = clamp(Math.round(data.granularity * 100), 0, 100);
    const level = data.granularity_level || "moderate";
    return {
      score,
      status: "available",
      insight: `${level.charAt(0).toUpperCase() + level.slice(1)} emotional granularity (${data.episodes_collected} episodes)`,
    };
  } catch {
    return { score: null, status: "unavailable", insight: "", unlockHint: "Do 5+ check-ins" };
  }
}

async function computeStability(userId: string, mlBaseUrl: string): Promise<VitalScore> {
  try {
    const genomeRes = await fetch(`${mlBaseUrl}/api/emotional-genome/profile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, samples: [] }),
    });

    if (!genomeRes.ok) throw new Error("No genome data");
    const genome = await genomeRes.json();

    if (genome.traits?.emotional_stability == null) {
      return { score: null, status: "unavailable", insight: "", unlockHint: "Track for 7+ days" };
    }

    const score = clamp(Math.round(genome.traits.emotional_stability * 100), 0, 100);
    return {
      score,
      status: "available",
      insight: score >= 70 ? "Your emotional baseline has been consistent" : "Your emotional baseline shows some variability",
    };
  } catch {
    return { score: null, status: "unavailable", insight: "", unlockHint: "Track for 7+ days" };
  }
}
```

- [ ] **Step 4: Add insight generation and main compute function**

Append to `server/efs-compute.ts`:

```typescript
async function generateInsight(
  userId: string,
  vitals: EFSResult["vitals"],
  currentScore: number | null,
): Promise<EFSResult["dailyInsight"]> {
  if (currentScore == null) return null;

  // Check yesterday's insight type for rotation
  const yesterday = new Date();
  yesterday.setUTCDate(yesterday.getUTCDate() - 1);
  const yesterdayStr = yesterday.toISOString().slice(0, 10);

  const yesterdayRow = await db
    .select({ dailyInsightType: emotionalFitnessScores.dailyInsightType })
    .from(emotionalFitnessScores)
    .where(and(eq(emotionalFitnessScores.userId, userId), eq(emotionalFitnessScores.date, yesterdayStr)))
    .limit(1);

  const lastType = yesterdayRow[0]?.dailyInsightType || null;

  // Priority-ordered insight candidates
  const candidates: { type: string; text: string }[] = [];

  // P1: Awareness gap
  if (vitals.awareness.status === "available" && vitals.awareness.score !== null && vitals.awareness.score < 50) {
    candidates.push({ type: "awareness_gap", text: vitals.awareness.insight });
  }

  // P2: Improvement (any vital up 10+ in 30 days)
  const thirtyDaysAgo = new Date();
  thirtyDaysAgo.setUTCDate(thirtyDaysAgo.getUTCDate() - 30);
  const oldScore = await db
    .select()
    .from(emotionalFitnessScores)
    .where(and(
      eq(emotionalFitnessScores.userId, userId),
      gte(emotionalFitnessScores.date, thirtyDaysAgo.toISOString().slice(0, 10)),
    ))
    .orderBy(emotionalFitnessScores.date)
    .limit(1);

  if (oldScore.length > 0) {
    for (const [name, vital] of Object.entries(vitals)) {
      const oldVal = oldScore[0][name as keyof typeof oldScore[0]] as number | null;
      if (vital.score != null && oldVal != null && vital.score - oldVal >= 10) {
        candidates.push({
          type: "improvement",
          text: `Your ${name} jumped ${vital.score - oldVal} points in the last 30 days`,
        });
        break;
      }
    }
  }

  // P3: Range expansion
  if (vitals.range.status === "available" && vitals.range.score !== null && vitals.range.score > 65) {
    candidates.push({ type: "range_expansion", text: vitals.range.insight });
  }

  // P5: Milestone (EFS above 70 for 30+ days)
  const streakRows = await db
    .select({ score: emotionalFitnessScores.score })
    .from(emotionalFitnessScores)
    .where(and(
      eq(emotionalFitnessScores.userId, userId),
      gte(emotionalFitnessScores.date, thirtyDaysAgo.toISOString().slice(0, 10)),
    ))
    .orderBy(desc(emotionalFitnessScores.date));

  if (streakRows.length >= 30 && streakRows.every(r => r.score != null && r.score >= 70)) {
    candidates.push({ type: "milestone", text: "30-day streak: your EFS has been above 70 for a full month" });
  }

  // P6: Transformation
  const firstEver = await db
    .select({ score: emotionalFitnessScores.score })
    .from(emotionalFitnessScores)
    .where(and(eq(emotionalFitnessScores.userId, userId), isNotNull(emotionalFitnessScores.score)))
    .orderBy(emotionalFitnessScores.date)
    .limit(1);

  if (firstEver.length > 0 && firstEver[0].score != null && currentScore - firstEver[0].score >= 15) {
    candidates.push({
      type: "transformation",
      text: `You started at ${firstEver[0].score}. You're now ${currentScore}. That's a transformation.`,
    });
  }

  // Pick highest priority, respecting soft rotation
  for (const c of candidates) {
    if (c.type !== lastType || candidates.length === 1) {
      return { text: c.text, type: c.type, actionNudge: INSIGHT_NUDGES[c.type] || "" };
    }
  }

  // If all candidates matched lastType, just use the first one
  if (candidates.length > 0) {
    const c = candidates[0];
    return { text: c.text, type: c.type, actionNudge: INSIGHT_NUDGES[c.type] || "" };
  }

  return null;
}

export async function computeEmotionalFitness(userId: string, mlBaseUrl: string, force = false, historyDays = 14): Promise<EFSResult> {
  const today = todayUTC();

  // Check cache
  if (!force) {
    const cached = await db
      .select()
      .from(emotionalFitnessScores)
      .where(and(eq(emotionalFitnessScores.userId, userId), eq(emotionalFitnessScores.date, today)))
      .limit(1);

    if (cached.length > 0 && cached[0].score != null) {
      // Build response from cached row + history
      const history = await getHistory(userId, historyDays);
      const trend = await computeTrend(userId, cached[0].score);

      return {
        score: cached[0].score,
        color: scoreColor(cached[0].score),
        label: scoreLabel(cached[0].score),
        confidence: cached[0].confidence as "full" | "early_estimate",
        trend,
        vitals: buildVitalsFromCache(cached[0], history),
        dailyInsight: cached[0].dailyInsightText
          ? { text: cached[0].dailyInsightText, type: cached[0].dailyInsightType || "", actionNudge: INSIGHT_NUDGES[cached[0].dailyInsightType || ""] || "" }
          : null,
        computedAt: cached[0].computedAt.toISOString(),
      };
    }
  }

  // Check minimum data (3 distinct days with emotionReadings)
  const distinctDays = await db
    .select({ day: sql<string>`DATE(${emotionReadings.timestamp})` })
    .from(emotionReadings)
    .where(eq(emotionReadings.userId, userId))
    .groupBy(sql`DATE(${emotionReadings.timestamp})`);

  const daysTracked = distinctDays.length;

  if (daysTracked < 3) {
    return {
      score: null, color: null, label: null,
      confidence: "building",
      progress: {
        daysTracked,
        daysRequired: 3,
        percentage: Math.round((daysTracked / 3) * 100),
        message: `Keep tracking for ${3 - daysTracked} more day${3 - daysTracked > 1 ? "s" : ""} to unlock your Emotional Fitness Score`,
      },
      trend: null,
      vitals: {
        resilience: { score: null, status: "unavailable", insight: "", unlockHint: "Track for 3+ days", history: [] },
        regulation: { score: null, status: "unavailable", insight: "", unlockHint: "Track for 3+ days", history: [] },
        awareness: { score: null, status: "unavailable", insight: "", unlockHint: "Complete a calibration session", history: [] },
        range: { score: null, status: "unavailable", insight: "", unlockHint: "Do 5+ check-ins", history: [] },
        stability: { score: null, status: "unavailable", insight: "", unlockHint: "Track for 7+ days", history: [] },
      },
      dailyInsight: null,
      computedAt: new Date().toISOString(),
    };
  }

  // Compute all 5 vitals
  const [resilience, regulation, awareness, range, stability] = await Promise.all([
    computeResilience(userId, mlBaseUrl),
    computeRegulation(userId, mlBaseUrl),
    computeAwareness(userId),
    computeRange(userId, mlBaseUrl),
    computeStability(userId, mlBaseUrl),
  ]);

  // Compute weighted composite from available vitals
  const available: { name: string; score: number; weight: number }[] = [];
  const vitalEntries = { resilience, regulation, awareness, range, stability };
  for (const [name, vital] of Object.entries(vitalEntries)) {
    if (vital.status === "available" && vital.score != null) {
      available.push({ name, score: vital.score, weight: WEIGHTS[name as keyof typeof WEIGHTS] });
    }
  }

  let compositeScore: number | null = null;
  let confidence: "full" | "early_estimate" | "building" = "building";

  if (available.length >= 3) {
    // Re-normalize weights
    const totalWeight = available.reduce((s, v) => s + v.weight, 0);
    compositeScore = Math.round(available.reduce((s, v) => s + v.score * (v.weight / totalWeight), 0));
    compositeScore = clamp(compositeScore, 0, 100);
    confidence = daysTracked >= 7 ? "full" : "early_estimate";
  }

  // Get history for vitals
  const history = await getHistory(userId, historyDays);
  const trend = compositeScore != null ? await computeTrend(userId, compositeScore) : null;

  const vitalsResult: EFSResult["vitals"] = {
    resilience: { ...resilience, history: history.map(h => ({ date: h.date, score: h.resilience ?? 0 })) },
    regulation: { ...regulation, history: history.map(h => ({ date: h.date, score: h.regulation ?? 0 })) },
    awareness: { ...awareness, history: history.map(h => ({ date: h.date, score: h.awareness ?? 0 })) },
    range: { ...range, history: history.map(h => ({ date: h.date, score: h.range ?? 0 })) },
    stability: { ...stability, history: history.map(h => ({ date: h.date, score: h.stability ?? 0 })) },
  };

  // Generate insight
  const dailyInsight = await generateInsight(userId, vitalsResult, compositeScore);

  // Store
  await db
    .insert(emotionalFitnessScores)
    .values({
      userId,
      date: today,
      score: compositeScore,
      resilience: resilience.score,
      regulation: regulation.score,
      awareness: awareness.score,
      range: range.score,
      stability: stability.score,
      dailyInsightText: dailyInsight?.text || null,
      dailyInsightType: dailyInsight?.type || null,
      confidence,
    })
    .onConflictDoUpdate({
      target: [emotionalFitnessScores.userId, emotionalFitnessScores.date],
      set: {
        score: compositeScore,
        resilience: resilience.score,
        regulation: regulation.score,
        awareness: awareness.score,
        range: range.score,
        stability: stability.score,
        dailyInsightText: dailyInsight?.text || null,
        dailyInsightType: dailyInsight?.type || null,
        confidence,
        computedAt: new Date(),
      },
    });

  return {
    score: compositeScore,
    color: compositeScore != null ? scoreColor(compositeScore) : null,
    label: compositeScore != null ? scoreLabel(compositeScore) : null,
    confidence,
    trend,
    vitals: vitalsResult,
    dailyInsight,
    computedAt: new Date().toISOString(),
  };
}

// ---------- Helpers ----------

async function getHistory(userId: string, days: number) {
  return db
    .select({
      date: emotionalFitnessScores.date,
      score: emotionalFitnessScores.score,
      resilience: emotionalFitnessScores.resilience,
      regulation: emotionalFitnessScores.regulation,
      awareness: emotionalFitnessScores.awareness,
      range: emotionalFitnessScores.range,
      stability: emotionalFitnessScores.stability,
    })
    .from(emotionalFitnessScores)
    .where(and(
      eq(emotionalFitnessScores.userId, userId),
      gte(emotionalFitnessScores.date, daysAgoUTC(days).toISOString().slice(0, 10)),
    ))
    .orderBy(emotionalFitnessScores.date);
}

async function computeTrend(userId: string, currentScore: number): Promise<EFSResult["trend"]> {
  const thirtyDaysAgo = daysAgoUTC(30).toISOString().slice(0, 10);
  const oldRows = await db
    .select({ score: emotionalFitnessScores.score })
    .from(emotionalFitnessScores)
    .where(and(
      eq(emotionalFitnessScores.userId, userId),
      lte(emotionalFitnessScores.date, thirtyDaysAgo),
    ))
    .orderBy(desc(emotionalFitnessScores.date))
    .limit(1);

  if (!oldRows.length || oldRows[0].score == null) return null;

  const delta = currentScore - oldRows[0].score;
  return {
    direction: delta > 2 ? "up" : delta < -2 ? "down" : "stable",
    delta,
    period: "30d",
  };
}

function buildVitalsFromCache(row: EmotionalFitnessScore, history: Awaited<ReturnType<typeof getHistory>>): EFSResult["vitals"] {
  const makeVital = (name: string, score: number | null): VitalScore & { history: { date: string; score: number }[] } => ({
    score,
    status: score != null ? "available" : "unavailable",
    insight: VITAL_EXPLANATIONS[name]?.explanation || "",
    history: history.map(h => ({ date: h.date, score: (h[name as keyof typeof h] as number) ?? 0 })),
  });

  return {
    resilience: makeVital("resilience", row.resilience),
    regulation: makeVital("regulation", row.regulation),
    awareness: makeVital("awareness", row.awareness),
    range: makeVital("range", row.range),
    stability: makeVital("stability", row.stability),
  };
}
```

- [ ] **Step 5: Commit**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add server/efs-compute.ts
git commit -m "feat(efs): add EFS compute engine with 5 vital scoring + insights"
```

---

## Task 3: Express Route

**Files:**
- Modify: `server/routes.ts` (after weekly-summary route, ~line 1288)

- [ ] **Step 1: Add EFS GET endpoint**

Add this after the weekly-summary endpoint in `server/routes.ts`:

```typescript
// Emotional Fitness Score
app.get("/api/brain/emotional-fitness/:userId", async (req, res) => {
  try {
    const { userId } = req.params;
    const force = req.query.force === "true";
    const days = parseInt(req.query.days as string) || 14;
    const { computeEmotionalFitness } = await import("./efs-compute");
    const mlBaseUrl = process.env.VITE_ML_API_URL || "http://localhost:8080";
    const result = await computeEmotionalFitness(userId, mlBaseUrl, force, days);
    res.json(result);
  } catch (error) {
    console.error("Emotional fitness error:", error);
    res.status(500).json({ message: "Failed to compute emotional fitness score" });
  }
});
```

- [ ] **Step 2: Verify it compiles**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx tsc --noEmit`
Expected: No new errors.

- [ ] **Step 3: Commit**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add server/routes.ts
git commit -m "feat(efs): add GET /api/brain/emotional-fitness/:userId endpoint"
```

---

## Task 4: Client API Helper

**Files:**
- Modify: `client/src/lib/ml-api.ts`

- [ ] **Step 1: Add EFS type and fetch function**

Add near the other export functions in `ml-api.ts`:

```typescript
export interface EFSData {
  score: number | null;
  color: "green" | "amber" | "red" | null;
  label: string | null;
  confidence: "full" | "early_estimate" | "building";
  progress?: { daysTracked: number; daysRequired: number; percentage: number; message: string };
  trend: { direction: "up" | "down" | "stable"; delta: number; period: string } | null;
  vitals: Record<string, {
    score: number | null;
    status: "available" | "unavailable";
    insight: string;
    unlockHint?: string;
    history: { date: string; score: number }[];
  }>;
  dailyInsight: { text: string; type: string; actionNudge: string } | null;
  computedAt: string | null;
}

export async function getEmotionalFitness(userId: string, force = false): Promise<EFSData> {
  const res = await fetch(`/api/brain/emotional-fitness/${encodeURIComponent(userId)}${force ? "?force=true" : ""}`);
  if (!res.ok) throw new Error("Failed to fetch emotional fitness");
  return res.json();
}
```

Note: This goes through the Express proxy (`/api/brain/...`), not through `mlFetch`, because the EFS endpoint is on Express, not FastAPI.

- [ ] **Step 2: Commit**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add client/src/lib/ml-api.ts
git commit -m "feat(efs): add getEmotionalFitness client API helper"
```

---

## Task 5: EFS Hero Score Component

**Files:**
- Create: `client/src/components/efs-hero-score.tsx`

- [ ] **Step 1: Create arc gauge component**

Model after `readiness-score.tsx` ScoreArc pattern:

```typescript
import { cn } from "@/lib/utils";

interface EFSHeroScoreProps {
  score: number | null;
  color: "green" | "amber" | "red" | null;
  label: string | null;
  confidence: string;
  trend: { direction: "up" | "down" | "stable"; delta: number; period: string } | null;
  progress?: { percentage: number; message: string };
}

const COLOR_MAP = {
  green: { stroke: "#0891b2", text: "text-cyan-400", bg: "bg-cyan-400/10" },
  amber: { stroke: "#d4a017", text: "text-amber-400", bg: "bg-amber-400/10" },
  red: { stroke: "#e879a8", text: "text-rose-400", bg: "bg-rose-400/10" },
};

export function EFSHeroScore({ score, color, label, confidence, trend, progress }: EFSHeroScoreProps) {
  const size = 200;
  const strokeWidth = 12;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const arcLength = circumference * 0.75; // 270 degrees
  const fillLength = score != null ? arcLength * (score / 100) : 0;
  const colors = color ? COLOR_MAP[color] : COLOR_MAP.green;

  if (score == null && progress) {
    return (
      <div className="flex flex-col items-center gap-3 py-6">
        <div className="relative" style={{ width: size, height: size }}>
          <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
            <circle cx={size/2} cy={size/2} r={radius} fill="none" stroke="hsl(var(--muted))" strokeWidth={strokeWidth}
              strokeDasharray={`${arcLength} ${circumference}`} transform={`rotate(135 ${size/2} ${size/2})`} strokeLinecap="round" />
            <circle cx={size/2} cy={size/2} r={radius} fill="none" stroke="#0891b2" strokeWidth={strokeWidth}
              strokeDasharray={`${arcLength * progress.percentage / 100} ${circumference}`}
              transform={`rotate(135 ${size/2} ${size/2})`} strokeLinecap="round"
              style={{ transition: "stroke-dasharray 1s ease" }} />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-3xl font-bold text-muted-foreground">{progress.percentage}%</span>
            <span className="text-xs text-muted-foreground">Building</span>
          </div>
        </div>
        <p className="text-sm text-muted-foreground text-center max-w-[250px]">{progress.message}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-2 py-4">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          <circle cx={size/2} cy={size/2} r={radius} fill="none" stroke="hsl(var(--muted))" strokeWidth={strokeWidth}
            strokeDasharray={`${arcLength} ${circumference}`} transform={`rotate(135 ${size/2} ${size/2})`} strokeLinecap="round" />
          <circle cx={size/2} cy={size/2} r={radius} fill="none" stroke={colors.stroke} strokeWidth={strokeWidth}
            strokeDasharray={`${fillLength} ${circumference}`} transform={`rotate(135 ${size/2} ${size/2})`} strokeLinecap="round"
            style={{ transition: "stroke-dasharray 1.2s cubic-bezier(0.34, 1.56, 0.64, 1)" }} />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={cn("text-5xl font-bold", colors.text)}>{score}</span>
          <span className="text-sm text-muted-foreground">Emotional Fitness</span>
        </div>
      </div>
      {label && <span className={cn("text-sm font-medium", colors.text)}>{label}</span>}
      {confidence === "early_estimate" && <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded">Early estimate</span>}
      {trend && trend.direction !== "stable" && (
        <span className={cn("text-xs px-2 py-0.5 rounded", colors.bg, colors.text)}>
          {trend.direction === "up" ? "↑" : "↓"} {Math.abs(trend.delta)} pts this month
        </span>
      )}
      <p className="text-xs text-muted-foreground">Updated daily from your brain, voice, and self-reports</p>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add client/src/components/efs-hero-score.tsx
git commit -m "feat(efs): add EFS hero score arc gauge component"
```

---

## Task 6: Vital Card, Insight Banner, History Chart, Mini Card Components

**Files:**
- Create: `client/src/components/efs-vital-card.tsx`
- Create: `client/src/components/efs-insight-banner.tsx`
- Create: `client/src/components/efs-history-chart.tsx`
- Create: `client/src/components/efs-mini-card.tsx`

- [ ] **Step 1: Create `efs-vital-card.tsx`**

Build the vital sign card with sparkline, score, and insight. Use Recharts LineChart for sparkline. Include expand/collapse for detail view with explanation and tips.

Key structure: Card with icon + name + score + sparkline + insight. On tap, expand to show 30/60/90 day chart + explanation + tips from `VITAL_EXPLANATIONS`.

- [ ] **Step 2: Create `efs-insight-banner.tsx`**

Card with accent border, insight text, action nudge in muted text, share icon.

- [ ] **Step 3: Create `efs-history-chart.tsx`**

Recharts AreaChart with 30/60/90 day toggle buttons. Query `/api/brain/emotional-fitness/:userId?days=N` for longer ranges.

- [ ] **Step 4: Create `efs-mini-card.tsx`**

Compact dashboard card: score number + trend arrow + "Emotional Fitness" label. Links to `/emotional-fitness`. If score is null, show progress ring.

- [ ] **Step 5: Commit all components**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add client/src/components/efs-vital-card.tsx client/src/components/efs-insight-banner.tsx client/src/components/efs-history-chart.tsx client/src/components/efs-mini-card.tsx
git commit -m "feat(efs): add vital card, insight banner, history chart, mini card components"
```

---

## Task 7: EFS Page + Router

**Files:**
- Create: `client/src/pages/emotional-fitness.tsx`
- Modify: `client/src/App.tsx`

- [ ] **Step 1: Create the main page**

Structure: EFSHeroScore → 5 EFSVitalCards in 2-col grid → EFSInsightBanner → EFSHistoryChart → Share button.

Use `useQuery` with `queryKey: ["emotional-fitness", userId]` and `queryFn: () => getEmotionalFitness(userId)`.

- [ ] **Step 2: Add route to App.tsx**

Add lazy import:
```typescript
const EmotionalFitness = lazy(() => import("@/pages/emotional-fitness"));
```

Add route (after emotional-intelligence route):
```typescript
<Route path="/emotional-fitness">
  <Suspense fallback={<PageLoader />}>
    <AppLayout><EmotionalFitness /></AppLayout>
  </Suspense>
</Route>
```

- [ ] **Step 3: Verify it compiles**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add client/src/pages/emotional-fitness.tsx client/src/App.tsx
git commit -m "feat(efs): add Emotional Fitness page and route"
```

---

## Task 8: Share Card (PNG Export)

**Files:**
- Create: `client/src/components/efs-share-card.tsx`

- [ ] **Step 1: Create canvas-based PNG export**

Follow the `weekly-brain-summary.tsx` pattern: create offscreen canvas, draw dark background, score arc, 5 horizontal bars, insight text, watermark. Export via `canvas.toDataURL("image/png")`.

Function signature: `exportEFSCard(data: EFSData): void`

- [ ] **Step 2: Add share button to the page**

In `emotional-fitness.tsx`, add a share/download button that calls `exportEFSCard(data)`.

- [ ] **Step 3: Commit**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add client/src/components/efs-share-card.tsx client/src/pages/emotional-fitness.tsx
git commit -m "feat(efs): add shareable PNG export card"
```

---

## Task 9: Dashboard Integration

**Files:**
- Modify: Dashboard page (find the main dashboard in `client/src/pages/`)

- [ ] **Step 1: Add EFS mini card to dashboard**

Import `EFSMiniCard` and add it to the dashboard layout alongside existing cards (Readiness Score, etc.).

- [ ] **Step 2: Verify integration**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add client/src/pages/dashboard.tsx client/src/components/efs-mini-card.tsx
git commit -m "feat(efs): integrate EFS mini card into dashboard"
```

---

## Task 10: TypeScript Check + Final Verification

- [ ] **Step 1: Run full TypeScript check**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx tsc --noEmit`
Expected: 0 errors.

- [ ] **Step 2: Run tests**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx vitest run`
Expected: All existing tests pass. No regressions.

- [ ] **Step 3: Push to GitHub**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop && git push
```

- [ ] **Step 4: Update STATUS.md**

Add EFS to the completed features list with endpoint and page info.
