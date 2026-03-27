import { getSupabase } from "@/lib/supabase-browser";

export type PassType = "time_bucket" | "food_lag" | "sleep_cascade" | "hrv_valence" | "weekly_rhythm";

export interface StoredInsight {
  id: string;
  category: PassType;
  priority: "high" | "medium" | "low";
  headline: string;
  context: string;
  action: string;
  actionHref: string;
  correlationStrength: number;
  discoveredAt: string;
}

interface EmotionEntry {
  stress: number;
  focus: number;
  valence: number;
  arousal?: number;
  timestamp: string;
}

interface CurrentReading {
  stress: number;
  focus: number;
  valence: number;
  arousal: number;
}

const CACHE_KEY = "ndw_pattern_cache";
const CACHE_TTL_MS = 6 * 60 * 60 * 1000;

function isPrivacyModeEnabled(): boolean {
  try { return localStorage.getItem("ndw_privacy_mode") === "true"; } catch { return false; }
}

function pearsonR(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 2) return 0;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  const num = xs.reduce((acc, x, i) => acc + (x - mx) * (ys[i] - my), 0);
  const dx  = Math.sqrt(xs.reduce((acc, x) => acc + (x - mx) ** 2, 0));
  const dy  = Math.sqrt(ys.reduce((acc, y) => acc + (y - my) ** 2, 0));
  return (dx === 0 || dy === 0) ? 0 : num / (dx * dy);
}

function mean(arr: number[]): number {
  return arr.length === 0 ? 0 : arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr: number[]): number {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((acc, v) => acc + (v - m) ** 2, 0) / Math.max(arr.length, 1));
}

export class PatternDiscovery {
  constructor(private userId: string) {}

  async run(nowIso: string, current?: CurrentReading): Promise<StoredInsight[]> {
    // Check cache
    try {
      const cached = JSON.parse(localStorage.getItem(CACHE_KEY) || "null");
      if (cached && Date.now() - new Date(cached.computed).getTime() < CACHE_TTL_MS) {
        return cached.insights as StoredInsight[];
      }
    } catch {}

    const history: EmotionEntry[] = (() => {
      try { return JSON.parse(localStorage.getItem("ndw_emotion_history") || "[]"); } catch { return []; }
    })();

    const insights: StoredInsight[] = [
      ...this.timeBucketPass(history, current, nowIso),
      ...this.weeklyRhythmPass(history),
      ...this.sleepCascadePass(history),
      ...this.foodLagPass(),
      ...this.hrvValencePass(),
    ];

    // Persist cache
    try {
      localStorage.setItem(CACHE_KEY, JSON.stringify({ computed: new Date().toISOString(), insights }));
    } catch {}

    // Persist to Supabase (privacy-gated)
    if (!isPrivacyModeEnabled()) {
      const supabase = await getSupabase();
      if (supabase && insights.length > 0) {
        const rows = insights.map(insight => {
          const match = insight.context.match(/across (\d+)/);
          const sampleCount = match ? parseInt(match[1], 10) : 0;
          return {
            user_id: this.userId,
            pass_type: insight.category,
            pattern_data: { headline: insight.headline, context: insight.context },
            correlation_strength: insight.correlationStrength,
            sample_count: sampleCount,
            last_computed: new Date().toISOString(),
            is_active: true,
          };
        });
        supabase.from("user_patterns")
          .upsert(rows, { onConflict: "user_id,pass_type" })
          .then(({ error }) => { if (error) console.warn("[PatternDiscovery] upsert error:", error.message); });
      }
    }

    return insights;
  }

  // valence in emotion_history is stored as raw FAA output (−1 to +1).
  // Normalize to 0-1 before z-score comparison.
  private normalizeHistoryValence(raw: number): number {
    return (raw + 1) / 2;
  }

  private timeBucketPass(history: EmotionEntry[], current: CurrentReading | undefined, nowIso: string): StoredInsight[] {
    if (!current) return [];
    const nowBucket = new Date(nowIso).getUTCHours();
    const bucketEntries = history.filter(e => new Date(e.timestamp).getUTCHours() === nowBucket);
    if (bucketEntries.length < 7) return [];

    const insights: StoredInsight[] = [];
    for (const metric of ["stress", "focus", "valence"] as const) {
      // Normalize valence from raw −1..1 to 0..1
      const values = bucketEntries.map(e =>
        metric === "valence" ? this.normalizeHistoryValence(e[metric]) : e[metric]
      );
      const m = mean(values);
      const s = std(values);
      const currentVal = current[metric]; // already 0-1 in CurrentReading
      const z = (currentVal - m) / Math.max(s, 0.01);
      if (Math.abs(z) > 1.5) {
        const dir = z > 0 ? "elevated" : "lower";
        insights.push({
          id: `time_bucket_${metric}`,
          category: "time_bucket",
          priority: Math.abs(z) > 2 ? "high" : "medium",
          headline: `Your ${metric} at ${nowBucket}:00 is ${dir} — ${(currentVal * 100).toFixed(0)}% vs your usual ${(m * 100).toFixed(0)}%`,
          context: `Pattern found across ${bucketEntries.length} similar-hour readings`,
          action: metric === "stress" ? "Try box breathing" : "Take a short break",
          actionHref: metric === "stress" ? "/biofeedback" : "/neurofeedback",
          correlationStrength: Math.min(Math.abs(z) / 3, 1),
          discoveredAt: nowIso,
        });
      }
    }
    return insights;
  }

  private weeklyRhythmPass(history: EmotionEntry[]): StoredInsight[] {
    if (history.length < 10) return [];
    // Only process entries with valid timestamps (getUTCDay returns 0-6)
    const byDay: Record<number, number[]> = {};
    for (const entry of history) {
      const dt = new Date(entry.timestamp);
      const day = dt.getUTCDay();
      // isNaN guard: if entry.timestamp is malformed, getUTCDay() returns NaN — skip these entries
      if (isNaN(day)) continue;
      byDay[day] = byDay[day] || [];
      byDay[day].push(entry.stress);
    }
    // Require at least 5 valid entries total across all days
    const totalValid = Object.values(byDay).reduce((a, v) => a + v.length, 0);
    if (totalValid < 5) return [];
    const insights: StoredInsight[] = [];
    const dayNames = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    for (const [dayStr, values] of Object.entries(byDay)) {
      const day = Number(dayStr);
      if (isNaN(day) || values.length < 3) continue;
      const dayMean = mean(values);
      // Compare against all OTHER days' mean
      const otherValues = Object.entries(byDay)
        .filter(([k]) => Number(k) !== day && !isNaN(Number(k)))
        .flatMap(([, v]) => v);
      if (otherValues.length < 2) continue;
      const baseline = mean(otherValues);
      const ratio = baseline > 0 ? dayMean / baseline : 1;
      if (ratio > 1.3) {
        insights.push({
          id: `weekly_rhythm_${day}`,
          category: "weekly_rhythm",
          priority: ratio > 1.6 ? "high" : "medium",
          headline: `${dayNames[day]}s show elevated stress — ${ratio.toFixed(1)}x your weekday baseline`,
          context: `Pattern found across ${values.length} ${dayNames[day]}s`,
          action: "Front-load creative work before 11AM",
          actionHref: "/neurofeedback",
          correlationStrength: Math.min((ratio - 1) / 0.7, 1),
          discoveredAt: new Date().toISOString(),
        });
      }
    }
    return insights;
  }

  private sleepCascadePass(history: EmotionEntry[]): StoredInsight[] {
    // Requires sleep score data from health_samples (ndw_sleep_data localStorage key).
    // Minimum: 5 poor-sleep nights (score <60) with next-day emotion data.
    const sleepData: Array<{ lastPeriodStart?: string; score?: number; hours?: number; date?: string }> = (() => {
      try { return JSON.parse(localStorage.getItem("ndw_sleep_data") || "null") ?? []; } catch { return []; }
    })();
    if (!Array.isArray(sleepData)) return [];

    const poorNights = sleepData.filter(s => (s.score ?? 100) < 60 || (s.hours ?? 8) < 6);
    if (poorNights.length < 5) return [];

    // For each poor-sleep night, gather focus/stress/valence in next 24h
    type Pair = { sleepHours: number; nextFocus: number; nextStress: number };
    const pairs: Pair[] = [];
    for (const night of poorNights) {
      const nightDate = night.date;
      if (!nightDate) continue;
      const nightMs = new Date(nightDate).getTime();
      const nextDay = history.filter(e => {
        const eMs = new Date(e.timestamp).getTime();
        // Start at 6AM the next day to avoid including nocturnal readings on the same night
        return eMs >= nightMs + 6 * 60 * 60 * 1000 && eMs <= nightMs + 30 * 60 * 60 * 1000;
      });
      if (nextDay.length === 0) continue;
      pairs.push({
        sleepHours: night.hours ?? 6,
        nextFocus: mean(nextDay.map(e => e.focus)),
        nextStress: mean(nextDay.map(e => e.stress)),
      });
    }
    if (pairs.length < 5) return [];

    const r = pearsonR(pairs.map(p => p.sleepHours), pairs.map(p => p.nextFocus));
    if (Math.abs(r) < 0.3) return [];

    return [{
      id: "sleep_cascade",
      category: "sleep_cascade",
      priority: Math.abs(r) > 0.5 ? "high" : "medium",
      headline: `Poor sleep predicts a focus drop the next day in your data`,
      context: `Pattern found across ${pairs.length} short-sleep nights (r=${r.toFixed(2)})`,
      action: "Plan light cognitive work on post-short-sleep days",
      actionHref: "/health-analytics",
      correlationStrength: Math.abs(r),
      discoveredAt: new Date().toISOString(),
    }];
  }

  private foodLagPass(): StoredInsight[] {
    // Correlates food log entries with emotion changes at T+60/90/120/180 min.
    // Minimum: 10 paired food+emotion data points, |Pearson r| > 0.45.
    const foodLogs: Array<{ loggedAt: string; dominantMacro: string | null; calories?: number }> = (() => {
      try { return JSON.parse(localStorage.getItem("ndw_food_logs_" + this.userId) || "[]"); } catch { return []; }
    })();
    const history: EmotionEntry[] = (() => {
      try { return JSON.parse(localStorage.getItem("ndw_emotion_history") || "[]"); } catch { return []; }
    })();
    if (foodLogs.length < 10 || history.length < 10) return [];

    // For each food log, check stress at T+60, T+90, T+120, T+180 min
    const LAGS_MS = [60, 90, 120, 180].map(m => m * 60 * 1000);
    const hasMacros = foodLogs.some(f => f.dominantMacro != null);

    let bestR = 0;
    let bestLagMin = 60;
    let bestPairs: Array<[number, number]> = [];
    for (const lagMs of LAGS_MS) {
      const pairs: Array<[number, number]> = [];
      // Meal-preceded readings: x=1
      for (const food of foodLogs) {
        const foodMs = new Date(food.loggedAt).getTime();
        const window = history.filter(e => {
          const eMs = new Date(e.timestamp).getTime();
          return Math.abs(eMs - (foodMs + lagMs)) < 15 * 60 * 1000;
        });
        if (window.length === 0) continue;
        pairs.push([1, mean(window.map(e => e.stress))]);
      }
      // Non-meal control readings: x=0 (emotion readings not within lag of any meal)
      for (const e of history) {
        const eMs = new Date(e.timestamp).getTime();
        const nearMeal = foodLogs.some(f => {
          const foodMs = new Date(f.loggedAt).getTime();
          return Math.abs(eMs - (foodMs + lagMs)) < 15 * 60 * 1000;
        });
        if (!nearMeal) pairs.push([0, e.stress]);
      }
      if (pairs.length < 10) continue;
      const xs = pairs.map(p => p[0]);
      const ys = pairs.map(p => p[1]);
      const r = pearsonR(xs, ys);
      if (Math.abs(r) > Math.abs(bestR)) {
        bestR = r;
        bestLagMin = lagMs / 60000;
        bestPairs = pairs;
      }
    }

    if (Math.abs(bestR) < 0.45) return [];

    const macroLabel = hasMacros
      ? foodLogs.find(f => f.dominantMacro != null)?.dominantMacro ?? "a meal"
      : "eating";
    const direction = bestR > 0 ? "stress increase" : "stress decrease";

    return [{
      id: "food_lag",
      category: "food_lag",
      priority: Math.abs(bestR) > 0.6 ? "high" : "medium",
      headline: `${macroLabel.charAt(0).toUpperCase() + macroLabel.slice(1)} predicts a ${direction} ~${bestLagMin} minutes later`,
      context: `Pattern found across ${bestPairs.filter(p => p[0] === 1).length} food+emotion paired readings (r=${bestR.toFixed(2)})`,
      action: "Log your next meal and check how you feel 90 minutes later",
      actionHref: "/nutrition",
      correlationStrength: Math.abs(bestR),
      discoveredAt: new Date().toISOString(),
    }];
  }

  private hrvValencePass(): StoredInsight[] {
    // Correlates morning HRV (health_samples where metric='hrv_sdnn')
    // with afternoon valence (12PM-6PM). Minimum 14 paired days.
    const healthSamples: Array<{ metric: string; value: number; recordedAt: string }> = (() => {
      try { return JSON.parse(localStorage.getItem("ndw_health_samples") || "[]"); } catch { return []; }
    })();
    const history: EmotionEntry[] = (() => {
      try { return JSON.parse(localStorage.getItem("ndw_emotion_history") || "[]"); } catch { return []; }
    })();

    const hrv = healthSamples.filter(s => s.metric === "hrv_sdnn");
    if (hrv.length < 14) return [];

    type DayPair = { morningHrv: number; afternoonValence: number };
    const pairs: DayPair[] = [];
    for (const h of hrv) {
      const dayStr = new Date(h.recordedAt).toISOString().slice(0, 10);
      const afternoon = history.filter(e => {
        const d = new Date(e.timestamp);
        return d.toISOString().slice(0, 10) === dayStr
          && d.getUTCHours() >= 12 && d.getUTCHours() < 18;
      });
      if (afternoon.length === 0) continue;
      const avgValence = mean(afternoon.map(e => this.normalizeHistoryValence(e.valence)));
      pairs.push({ morningHrv: h.value, afternoonValence: avgValence });
    }
    if (pairs.length < 14) return [];

    const r = pearsonR(pairs.map(p => p.morningHrv), pairs.map(p => p.afternoonValence));
    if (Math.abs(r) < 0.4) return [];

    return [{
      id: "hrv_valence",
      category: "hrv_valence",
      priority: Math.abs(r) > 0.55 ? "high" : "medium",
      headline: `Your morning HRV predicts your afternoon mood — pattern found across ${pairs.length} days`,
      context: `Low HRV mornings correlate with mood dips after 2PM (r=${r.toFixed(2)})`,
      action: "Check HRV on waking. If low, plan light work after 2PM",
      actionHref: "/health-analytics",
      correlationStrength: Math.abs(r),
      discoveredAt: new Date().toISOString(),
    }];
  }
}
