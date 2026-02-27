import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { type SessionSummary } from "@/lib/ml-api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Moon,
  ArrowRight,
  Flame,
  BarChart2,
} from "lucide-react";
import { useLocation } from "wouter";

/* ── Types ───────────────────────────────────────────────────── */
interface SleepData {
  deepSleepMinutes?: number;
  remMinutes?: number;
  totalSleepMinutes?: number;
  sleepQuality?: number;
}

interface DreamEntry {
  id: string;
  dreamText: string;
  timestamp: string;
}

interface HealthEntry {
  id: string;
  heartRate?: number;
  stressLevel?: number;
  sleepQuality?: number;
  neuralActivity?: number;
  sleepDuration?: number;
  timestamp: string;
}

/* ── Types ───────────────────────────────────────────────────── */
interface ServerInsight {
  type: string;
  text: string;
  delta?: number;
}

interface BrainPattern {
  type: string;
  title: string;
  description: string;
  recommendation: string;
  confidence: number;
  data: Record<string, unknown>;
}

/* ── Derived / computed helpers ──────────────────────────────── */
const CURRENT_USER = "default";

function fmtMinutes(mins: number): string {
  const h = Math.floor(mins / 60);
  const m = Math.round(mins % 60);
  if (h > 0 && m > 0) return `${h}h ${m}m`;
  if (h > 0) return `${h}h`;
  return `${m}m`;
}

function greeting(): string {
  const h = new Date().getHours();
  if (h < 12) return "Good morning";
  if (h < 17) return "Good afternoon";
  return "Good evening";
}

/* ── Streak helper ───────────────────────────────────────────── */
function currentStreak(sessions: SessionSummary[]): number {
  if (sessions.length === 0) return 0;
  const dayMs = 86_400_000;
  const daySet = new Set(
    sessions.map((s) => {
      const d = new Date((s.start_time ?? 0) * 1000);
      d.setHours(0, 0, 0, 0);
      return d.getTime();
    })
  );
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const todayTs = today.getTime();
  const startTs = daySet.has(todayTs)
    ? todayTs
    : daySet.has(todayTs - dayMs)
    ? todayTs - dayMs
    : null;
  if (!startTs) return 0;
  let streak = 0;
  let checkTs = startTs;
  while (daySet.has(checkTs)) {
    streak++;
    checkTs -= dayMs;
  }
  return streak;
}

/* ── Pattern engine ──────────────────────────────────────────── */
function patternInsight(sessions: SessionSummary[], health: HealthEntry[]): string | null {
  const twoWeeksMs = 14 * 86_400_000;
  const twoWeeksAgoUnix = (Date.now() - twoWeeksMs) / 1000;
  const twoWeeksAgoDate = new Date(Date.now() - twoWeeksMs);

  const recentSessions = sessions.filter((s) => (s.start_time ?? 0) >= twoWeeksAgoUnix);
  const biofeedbackDays = new Set(
    recentSessions
      .filter((s) => s.session_type === "biofeedback")
      .map((s) => {
        const d = new Date((s.start_time ?? 0) * 1000);
        d.setHours(0, 0, 0, 0);
        return d.getTime();
      })
  );

  if (biofeedbackDays.size < 2) return null;

  const recentHealth = health.filter((h) => new Date(h.timestamp) >= twoWeeksAgoDate);
  if (recentHealth.length === 0) return null;

  const bfStress: number[] = [];
  const otherStress: number[] = [];

  for (const h of recentHealth) {
    const d = new Date(h.timestamp);
    d.setHours(0, 0, 0, 0);
    if (biofeedbackDays.has(d.getTime())) {
      bfStress.push(h.stressLevel ?? 5);
    } else {
      otherStress.push(h.stressLevel ?? 5);
    }
  }

  if (bfStress.length === 0 || otherStress.length === 0) return null;

  const avgBf    = bfStress.reduce((a, b) => a + b, 0) / bfStress.length;
  const avgOther = otherStress.reduce((a, b) => a + b, 0) / otherStress.length;
  const deltaPct = Math.round(((avgOther - avgBf) / Math.max(avgOther, 0.1)) * 100);

  if (deltaPct >= 8) {
    return `${biofeedbackDays.size} breathing sessions in 2 weeks → stress ${deltaPct}% lower on those days.`;
  }
  if (deltaPct <= -8) {
    return `${biofeedbackDays.size} breathing sessions tracked — stress similar to non-session days so far.`;
  }
  return null;
}

/** Map a stress level (0–10) to a text label. */
function stressLabel(level: number): string {
  if (level < 3) return "low";
  if (level < 6) return "moderate";
  return "high";
}

/** Find the 2-hour window with highest avg focus across all health entries.
 *  Falls back to a circadian heuristic if there's not enough data. */
function peakFocusWindow(health: HealthEntry[]): string {
  if (health.length < 6) return "9:30 am – 12:00 pm";
  const buckets: number[] = Array(24).fill(0);
  const counts: number[] = Array(24).fill(0);
  for (const h of health) {
    const hour = new Date(h.timestamp).getHours();
    buckets[hour] += h.neuralActivity ?? 5;
    counts[hour]++;
  }
  let bestHour = 9;
  let bestAvg = 0;
  for (let hr = 5; hr <= 22; hr++) {
    if (counts[hr] === 0) continue;
    const avg = buckets[hr] / counts[hr];
    if (avg > bestAvg) { bestAvg = avg; bestHour = hr; }
  }
  const fmt = (h: number) => {
    const ampm = h < 12 ? "am" : "pm";
    const h12 = h % 12 || 12;
    return `${h12}:00 ${ampm}`;
  };
  return `${fmt(bestHour)} – ${fmt(bestHour + 2)}`;
}

/** Find the 1-hour window with highest avg stress (the slump).
 *  Falls back to circadian heuristic. */
function slumpWindow(health: HealthEntry[]): string {
  if (health.length < 6) return "2:30 pm – 3:30 pm";
  const buckets: number[] = Array(24).fill(0);
  const counts: number[] = Array(24).fill(0);
  for (const h of health) {
    const hour = new Date(h.timestamp).getHours();
    buckets[hour] += h.stressLevel ?? 5;
    counts[hour]++;
  }
  let worstHour = 14;
  let worstAvg = 0;
  for (let hr = 12; hr <= 18; hr++) {
    if (counts[hr] === 0) continue;
    const avg = buckets[hr] / counts[hr];
    if (avg > worstAvg) { worstAvg = avg; worstHour = hr; }
  }
  const fmt = (h: number) => {
    const ampm = h < 12 ? "am" : "pm";
    const h12 = h % 12 || 12;
    return `${h12}:00 ${ampm}`;
  };
  return `${fmt(worstHour)} – ${fmt(worstHour + 1)}`;
}

/** Derive recommended action from latest health data. */
function recommendedAction(health: HealthEntry[]): {
  label: string;
  route: string;
  description: string;
} {
  if (!health.length) {
    return {
      label: "Start coherence breathing",
      route: "/biofeedback",
      description: "4-min session to centre your nervous system",
    };
  }
  const latest = health[0];
  const stress = latest.stressLevel ?? 0;
  const focus = latest.neuralActivity ?? 5;

  if (stress > 6) {
    return {
      label: "Start coherence breathing",
      route: "/biofeedback",
      description: "4-min session to lower cortisol and reset",
    };
  }
  if (focus < 4) {
    return {
      label: "Check your emotion state",
      route: "/emotions",
      description: "See what your brain is doing right now",
    };
  }
  return {
    label: "Review your sleep session",
    route: "/sessions",
    description: "Explore last night's EEG data in depth",
  };
}

/** Richer pattern engine: correlates time-of-day with focus/stress peaks.
 *  Returns a specific insight like "Focus peaked at 11 am, 31% above your afternoon."
 */
function yesterdayInsight(health: HealthEntry[]): string | null {
  if (health.length < 2) return null;
  const today = new Date();
  const yday = new Date(today);
  yday.setDate(today.getDate() - 1);

  const yesterdayEntries = health.filter((h) => {
    const d = new Date(h.timestamp);
    return d.getFullYear() === yday.getFullYear() &&
           d.getMonth() === yday.getMonth() &&
           d.getDate() === yday.getDate();
  });

  // If no yesterday entries, fall back to most recent single entry
  const entries = yesterdayEntries.length >= 2 ? yesterdayEntries : health.slice(0, 5);
  if (entries.length < 2) return null;

  // Find peak focus hour and compare to rest of the day
  let peakFocusEntry = entries[0];
  for (const e of entries) {
    if ((e.neuralActivity ?? 0) > (peakFocusEntry.neuralActivity ?? 0)) peakFocusEntry = e;
  }
  const peakFocus = peakFocusEntry.neuralActivity ?? 5;
  const otherFocusAvg =
    entries
      .filter((e) => e !== peakFocusEntry)
      .reduce((s, e) => s + (e.neuralActivity ?? 5), 0) /
    Math.max(1, entries.length - 1);

  const focusDeltaPct = Math.round(((peakFocus - otherFocusAvg) / Math.max(otherFocusAvg, 1)) * 100);
  const peakHour = new Date(peakFocusEntry.timestamp).getHours();
  const peakHourFmt = `${peakHour % 12 || 12} ${peakHour < 12 ? "am" : "pm"}`;

  // Find highest stress period
  let peakStressEntry = entries[0];
  for (const e of entries) {
    if ((e.stressLevel ?? 0) > (peakStressEntry.stressLevel ?? 0)) peakStressEntry = e;
  }
  const peakStress = peakStressEntry.stressLevel ?? 5;
  const stressHour = new Date(peakStressEntry.timestamp).getHours();
  const stressHourFmt = `${stressHour % 12 || 12} ${stressHour < 12 ? "am" : "pm"}`;

  // Pick the most interesting pattern
  if (focusDeltaPct >= 20) {
    return `Focus peaked at ${peakHourFmt}, ${focusDeltaPct}% above the rest of the day.`;
  }
  if (peakStress > 6) {
    return `Stress spiked around ${stressHourFmt} yesterday — ${stressLabel(peakStress)} level. Today, watch that window.`;
  }
  if (focusDeltaPct >= 10) {
    return `${peakHourFmt} was your sharpest hour yesterday — ${focusDeltaPct}% above average.`;
  }
  const avgFocus = entries.reduce((s, e) => s + (e.neuralActivity ?? 5), 0) / entries.length;
  const avgStress = entries.reduce((s, e) => s + (e.stressLevel ?? 5), 0) / entries.length;
  if (avgFocus > 6) return `Yesterday was a strong focus day — avg ${Math.round(avgFocus * 10)}%.`;
  if (avgStress > 6) return `Stress ran high yesterday — today is a fresh start.`;
  return `Yesterday looked balanced — stress ${stressLabel(avgStress)}, focus steady.`;
}

/* ── Skeleton card ───────────────────────────────────────────── */
function SkeletonCard() {
  return (
    <Card className="glass-card p-6 animate-pulse">
      <div className="h-4 w-1/3 bg-muted/40 rounded mb-4" />
      <div className="space-y-2">
        <div className="h-3 w-full bg-muted/30 rounded" />
        <div className="h-3 w-3/4 bg-muted/30 rounded" />
      </div>
    </Card>
  );
}

/* ── Weekly summary helpers ──────────────────────────────────── */
function weeklyStats(health: HealthEntry[], sessions: SessionSummary[]) {
  const cutoffMs = Date.now() - 7 * 24 * 60 * 60 * 1000;
  const cutoffUnix = cutoffMs / 1000;
  const week = health.filter((h) => new Date(h.timestamp).getTime() >= cutoffMs);
  const weekSessions = sessions.filter(
    (s) => (s.start_time ?? 0) >= cutoffUnix && (s.summary?.avg_focus ?? 0) > 0
  );
  if (week.length === 0 && weekSessions.length === 0) return null;

  // Stress from health entries (frequent signal)
  const avgStress = week.length > 0
    ? week.reduce((s, h) => s + (h.stressLevel ?? 5), 0) / week.length
    : 5;

  // Focus: prefer EEG session avg_focus (0-1 → ×100 = %) over health proxy
  const avgFocusPct = weekSessions.length > 0
    ? weekSessions.reduce((s, sess) => s + (sess.summary?.avg_focus ?? 0), 0) / weekSessions.length * 100
    : week.reduce((s, h) => s + (h.neuralActivity ?? 5), 0) / Math.max(week.length, 1) * 10;

  // Sleep from health entries
  const avgSleep = week.length > 0
    ? week.reduce((s, h) => s + (h.sleepQuality ?? 5), 0) / week.length
    : 5;

  return {
    days: Math.max(week.length, weekSessions.length),
    avgStress: Math.round(avgStress * 10),
    avgFocus: Math.round(avgFocusPct),
    avgSleep: Math.round(avgSleep * 10),
    focusSource: weekSessions.length > 0 ? "eeg" : "proxy",
  };
}

/* ── Main page ───────────────────────────────────────────────── */
export default function DailyBrainReport() {
  const [, navigate] = useLocation();
  const [copied, setCopied] = useState(false);

  /* — Data fetches — */
  const { data: sessions = [], isLoading: sessionsLoading } =
    useQuery<SessionSummary[]>({
      queryKey: ["sessions-brain-report"],
      queryFn: async () => {
        const res = await fetch("/api/ml/sessions");
        if (!res.ok) return [];
        return res.json();
      },
      staleTime: 60_000,
      retry: false,
    });

  const { data: dreams = [], isLoading: dreamsLoading } =
    useQuery<DreamEntry[]>({
      queryKey: ["dreams-brain-report", CURRENT_USER],
      queryFn: async () => {
        const res = await fetch(`/api/dream-analysis/${CURRENT_USER}`);
        if (!res.ok) return [];
        return res.json();
      },
      staleTime: 60_000,
      retry: false,
    });

  const { data: health = [], isLoading: healthLoading } =
    useQuery<HealthEntry[]>({
      queryKey: ["health-brain-report", CURRENT_USER],
      queryFn: async () => {
        const res = await fetch(`/api/health-metrics/${CURRENT_USER}`);
        if (!res.ok) return [];
        return res.json();
      },
      staleTime: 60_000,
      retry: false,
    });

  const { data: serverInsightsData } =
    useQuery<{ userId: string; insights: ServerInsight[] }>({
      queryKey: ["yesterday-insights", CURRENT_USER],
      queryFn: async () => {
        const res = await fetch(`/api/brain/yesterday-insights/${CURRENT_USER}`);
        if (!res.ok) return { userId: CURRENT_USER, insights: [] };
        return res.json();
      },
      staleTime: 5 * 60_000,
      retry: false,
    });

  const { data: patternsData } =
    useQuery<{ userId: string; dataPoints: number; patterns: BrainPattern[] }>({
      queryKey: ["brain-patterns", CURRENT_USER],
      queryFn: async () => {
        const res = await fetch(`/api/brain/patterns/${CURRENT_USER}`);
        if (!res.ok) return { userId: CURRENT_USER, dataPoints: 0, patterns: [] };
        return res.json();
      },
      staleTime: 10 * 60_000,  // re-fetch every 10 minutes
      retry: false,
    });

  const serverInsights: ServerInsight[] = serverInsightsData?.insights ?? [];
  const brainPatterns: BrainPattern[] = patternsData?.patterns ?? [];

  const isLoading = sessionsLoading || dreamsLoading || healthLoading;

  /* — Derived data — */
  const latestHealth = health[0] as HealthEntry | undefined;
  const sleepData: SleepData = {
    deepSleepMinutes: latestHealth?.sleepQuality
      ? Math.round((latestHealth.sleepQuality / 10) * 134)
      : undefined,
    remMinutes: latestHealth?.sleepQuality
      ? Math.round((latestHealth.sleepQuality / 10) * 62)
      : undefined,
    totalSleepMinutes: latestHealth?.sleepDuration
      ? latestHealth.sleepDuration * 60
      : undefined,
    sleepQuality: latestHealth?.sleepQuality,
  };

  const recentDreams = dreams.slice(0, 3);
  const action = recommendedAction(health);
  const insight = yesterdayInsight(health);
  const latestStress = latestHealth?.stressLevel ?? null;
  const weekly = weeklyStats(health, sessions);
  const streak = currentStreak(sessions);
  const pattern = patternInsight(sessions, health);

  /* — Overnight EEG session — */
  const overnightSession = sessions.find(
    (s) => s.session_type === "sleep" || (s.summary?.duration_sec ?? 0) > 3600
  );

  /* — Stress badge color — */
  const stressBadgeClass =
    latestStress !== null && latestStress > 6
      ? "bg-red-500/15 text-red-400 border-red-500/30"
      : latestStress !== null && latestStress > 3
      ? "bg-orange-500/15 text-orange-400 border-orange-500/30"
      : "bg-emerald-500/15 text-emerald-400 border-emerald-500/30";

  const stressBadgeLabel =
    latestStress !== null
      ? latestStress > 6 ? "High stress" : latestStress > 3 ? "Moderate" : "Calm"
      : null;

  /* — Top insight (1 liner) — */
  const topInsight: string | null =
    serverInsights.length > 0
      ? serverInsights[0].text
      : insight;

  return (
    <div className="max-w-lg mx-auto px-4 py-8 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div>
          <p className="text-xs text-muted-foreground">
            {new Date().toLocaleDateString([], { weekday: "long", month: "long", day: "numeric" })}
          </p>
          <h1 className="text-xl font-semibold mt-0.5">{greeting()}</h1>
        </div>
        <div className="flex flex-col items-end gap-1">
          {streak >= 2 && (
            <span className={`flex items-center gap-1 text-xs font-semibold ${streak >= 7 ? "text-orange-400" : "text-amber-400"}`}>
              <Flame className="h-3.5 w-3.5" />
              {streak}-day streak
            </span>
          )}
        </div>
      </div>

      {/* Card 1 — Right now */}
      {isLoading ? (
        <SkeletonCard />
      ) : (
        <Card className="glass-card p-5">
          <p className="text-[11px] text-muted-foreground uppercase tracking-wide mb-3">Right now</p>
          <div className="flex items-center gap-3 flex-wrap">
            {stressBadgeLabel && (
              <span className={`text-xs font-medium px-2.5 py-1 rounded-full border ${stressBadgeClass}`}>
                {stressBadgeLabel}
              </span>
            )}
            {latestHealth?.sleepQuality !== undefined && (
              <span className="text-xs px-2.5 py-1 rounded-full border border-indigo-500/30 bg-indigo-500/10 text-indigo-300">
                Sleep {Math.round(latestHealth.sleepQuality * 10)}%
              </span>
            )}
            {latestHealth?.heartRate && (
              <span className="text-xs px-2.5 py-1 rounded-full border border-border/40 text-muted-foreground">
                ♥ {latestHealth.heartRate} bpm
              </span>
            )}
          </div>
          {topInsight && (
            <p className="mt-3 text-sm text-foreground/75 leading-relaxed border-t border-border/20 pt-3">
              {topInsight}
            </p>
          )}
        </Card>
      )}

      {/* Card 2 — Last night */}
      {!isLoading && (sleepData.totalSleepMinutes !== undefined || recentDreams.length > 0 || overnightSession) && (
        <Card className="glass-card p-5">
          <div className="flex items-center gap-2 mb-3">
            <Moon className="h-3.5 w-3.5 text-indigo-400" />
            <p className="text-[11px] text-muted-foreground uppercase tracking-wide">Last night</p>
          </div>
          <div className="flex items-center gap-6">
            {sleepData.totalSleepMinutes !== undefined && (
              <div>
                <p className="text-lg font-semibold">{fmtMinutes(sleepData.totalSleepMinutes)}</p>
                <p className="text-[11px] text-muted-foreground">sleep</p>
              </div>
            )}
            {overnightSession?.summary?.duration_sec && !sleepData.totalSleepMinutes && (
              <div>
                <p className="text-lg font-semibold">{fmtMinutes(overnightSession.summary.duration_sec / 60)}</p>
                <p className="text-[11px] text-muted-foreground">session</p>
              </div>
            )}
            {recentDreams.length > 0 && (
              <div>
                <p className="text-lg font-semibold">{recentDreams.length}</p>
                <p className="text-[11px] text-muted-foreground">dream{recentDreams.length !== 1 ? "s" : ""}</p>
              </div>
            )}
          </div>
          {recentDreams.length > 0 && (
            <button
              onClick={() => navigate("/dreams")}
              className="mt-3 text-xs text-primary hover:underline flex items-center gap-1"
            >
              Open dream journal <ArrowRight className="h-3 w-3" />
            </button>
          )}
        </Card>
      )}

      {/* Card 3 — Do this now */}
      {!isLoading && (
        <Card className="glass-card p-5">
          <p className="text-[11px] text-muted-foreground uppercase tracking-wide mb-3">Do this now</p>
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-medium">{action.label}</p>
              <p className="text-xs text-muted-foreground mt-0.5">{action.description}</p>
            </div>
            <Button
              size="sm"
              className="shrink-0 min-w-[72px]"
              onClick={() => navigate(action.route)}
            >
              Start <ArrowRight className="ml-1 h-3 w-3" />
            </Button>
          </div>
        </Card>
      )}

      {/* Card 4 — Your pattern (shown only when server has a meaningful pattern) */}
      {!isLoading && brainPatterns.length > 0 && (
        <Card className="glass-card p-5">
          <div className="flex items-center gap-2 mb-2">
            <BarChart2 className="h-3.5 w-3.5 text-emerald-400" />
            <p className="text-[11px] text-muted-foreground uppercase tracking-wide">Your pattern</p>
          </div>
          <p className="text-sm font-medium">{brainPatterns[0].title}</p>
          <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{brainPatterns[0].description}</p>
          <p className="text-xs text-emerald-400/90 mt-1.5">→ {brainPatterns[0].recommendation}</p>
        </Card>
      )}
    </div>
  );
}
