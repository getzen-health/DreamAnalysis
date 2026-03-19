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

// ---------- Helpers ----------

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

// ---------- Vital Computation Functions ----------

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
      .select({ valence: emotionReadings.valence, ts: emotionReadings.timestamp })
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
    const valences = readings.map(r => ({ v: r.valence as number, t: r.ts!.getTime() }));
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
        .select({ stress: emotionReadings.stress, ts: emotionReadings.timestamp })
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

      // Count stress->calm transitions (stress > 0.6 followed by stress < 0.3)
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

// ---------- Insight Generation ----------

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

// ---------- Main Compute Function ----------

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
