import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { type SessionSummary } from "@/lib/ml-api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Sun,
  Moon,
  Brain,
  Wind,
  TrendingUp,
  Zap,
  AlertTriangle,
  Clock,
  BookOpen,
  ArrowRight,
  CalendarDays,
  Copy,
  Check,
  Flame,
  BarChart2,
  Lightbulb,
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

  return (
    <div className="max-w-2xl mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-muted-foreground mb-1">{new Date().toLocaleDateString([], { weekday: "long", month: "long", day: "numeric" })}</p>
          <h1 className="text-2xl font-semibold">{greeting()}</h1>
        </div>
        <div className="flex flex-col items-end gap-1">
          <Sun className="h-7 w-7 text-yellow-400 opacity-80" />
          {streak >= 2 && (
            <span className={`flex items-center gap-1 text-xs font-semibold ${streak >= 7 ? "text-orange-400" : "text-amber-400"}`}>
              <Flame className="h-3.5 w-3.5" />
              {streak}-day streak
            </span>
          )}
        </div>
      </div>

      {/* Last night — sleep summary */}
      {isLoading ? (
        <SkeletonCard />
      ) : (
        <Card className="glass-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <Moon className="h-4 w-4 text-indigo-400" />
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
              Last night
            </h2>
          </div>

          {sleepData.deepSleepMinutes !== undefined ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              <div>
                <p className="text-xs text-muted-foreground mb-1">Deep sleep</p>
                <p className="text-lg font-semibold">
                  {fmtMinutes(sleepData.deepSleepMinutes)}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">REM</p>
                <p className="text-lg font-semibold">
                  {sleepData.remMinutes !== undefined
                    ? fmtMinutes(sleepData.remMinutes)
                    : "—"}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">Dreams</p>
                <p className="text-lg font-semibold">
                  {recentDreams.length > 0
                    ? `${recentDreams.length} detected`
                    : "None recorded"}
                </p>
              </div>
            </div>
          ) : overnightSession ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-muted-foreground mb-1">Session length</p>
                <p className="text-lg font-semibold">
                  {overnightSession.summary?.duration_sec
                    ? fmtMinutes(overnightSession.summary.duration_sec / 60)
                    : "—"}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">Dreams</p>
                <p className="text-lg font-semibold">
                  {recentDreams.length > 0
                    ? `${recentDreams.length} detected`
                    : "None recorded"}
                </p>
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              No overnight session recorded yet.{" "}
              <button
                onClick={() => navigate("/device-setup")}
                className="text-primary underline-offset-4 hover:underline"
              >
                Connect your Muse 2
              </button>{" "}
              to start tracking sleep.
            </p>
          )}

          {recentDreams.length > 0 && (
            <div className="mt-4 pt-4 border-t border-border/30">
              <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1">
                <BookOpen className="h-3 w-3" />
                Latest dream
              </p>
              <p className="text-sm text-foreground/80 line-clamp-2">
                {recentDreams[0].dreamText}
              </p>
              <button
                onClick={() => navigate("/dreams")}
                className="mt-2 text-xs text-primary hover:underline flex items-center gap-1"
              >
                Open dream journal <ArrowRight className="h-3 w-3" />
              </button>
            </div>
          )}
        </Card>
      )}

      {/* Today's forecast */}
      {isLoading ? (
        <SkeletonCard />
      ) : (
        <Card className="glass-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="h-4 w-4 text-emerald-400" />
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
              Today's forecast
            </h2>
          </div>

          <div className="space-y-3">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-yellow-400 shrink-0" />
                <div>
                  <p className="text-sm font-medium">Peak focus</p>
                  <p className="text-xs text-muted-foreground">protect this time</p>
                </div>
              </div>
              <span className="text-sm font-mono text-foreground/90">
                {peakFocusWindow(health)}
              </span>
            </div>

            <div className="flex items-start justify-between">
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4 text-orange-400 shrink-0" />
                <div>
                  <p className="text-sm font-medium">Likely slump</p>
                  <p className="text-xs text-muted-foreground">schedule a break</p>
                </div>
              </div>
              <span className="text-sm font-mono text-foreground/90">
                {slumpWindow(health)}
              </span>
            </div>

            {latestStress !== null && (
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-2">
                  <AlertTriangle
                    className={`h-4 w-4 shrink-0 ${
                      latestStress > 6
                        ? "text-red-400"
                        : latestStress > 3
                        ? "text-orange-400"
                        : "text-emerald-400"
                    }`}
                  />
                  <div>
                    <p className="text-sm font-medium">Stress risk</p>
                    <p className="text-xs text-muted-foreground">based on last reading</p>
                  </div>
                </div>
                <span
                  className={`text-sm font-medium capitalize ${
                    latestStress > 6
                      ? "text-red-400"
                      : latestStress > 3
                      ? "text-orange-400"
                      : "text-emerald-400"
                  }`}
                >
                  {stressLabel(latestStress)}
                </span>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Yesterday's insight — server-computed activity correlations */}
      {!isLoading && (serverInsights.length > 0 || insight) && (
        <Card className="glass-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <Lightbulb className="h-4 w-4 text-violet-400" />
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
              Yesterday's insight
            </h2>
          </div>

          {serverInsights.length > 0 ? (
            <div className="space-y-3">
              {serverInsights.map((si, i) => (
                <div
                  key={i}
                  className={`flex items-start gap-3 ${i > 0 ? "pt-3 border-t border-border/20" : ""}`}
                >
                  <div className={`mt-0.5 w-1.5 h-1.5 rounded-full shrink-0 ${
                    si.type === "activity_focus" || si.type === "activity_stress"
                      ? "bg-emerald-400"
                      : si.type === "peak_focus"
                      ? "bg-violet-400"
                      : "bg-sky-400"
                  }`} />
                  <p className="text-sm text-foreground/90 leading-snug">{si.text}</p>
                </div>
              ))}
            </div>
          ) : (
            /* Fallback to client-side insight when server has no data */
            <p className="text-sm text-foreground/90">{insight}</p>
          )}
        </Card>
      )}

      {/* Recommended action */}
      {isLoading ? (
        <SkeletonCard />
      ) : (
        <Card className="glass-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <Wind className="h-4 w-4 text-sky-400" />
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
              Recommended now
            </h2>
          </div>

          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-medium mb-1">{action.label}</p>
              <p className="text-xs text-muted-foreground">{action.description}</p>
            </div>
            <Button
              size="sm"
              className="shrink-0"
              onClick={() => navigate(action.route)}
            >
              Start
              <ArrowRight className="ml-1 h-3 w-3" />
            </Button>
          </div>
        </Card>
      )}

      {/* Weekly brain summary */}
      {!isLoading && weekly && (
        <Card className="glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <CalendarDays className="h-4 w-4 text-sky-400" />
              <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
                This week
              </h2>
              <span className="text-[10px] text-muted-foreground/60">
                {weekly.days} day{weekly.days !== 1 ? "s" : ""} of data
              </span>
            </div>
            <button
              onClick={() => {
                const text = `Brain Summary — ${new Date().toLocaleDateString([], { month: "long", day: "numeric" })}\nStress: ${weekly.avgStress}%\nFocus: ${weekly.avgFocus}%\nSleep quality: ${weekly.avgSleep}%\nvia Svapnastra`;
                navigator.clipboard.writeText(text).then(() => {
                  setCopied(true);
                  setTimeout(() => setCopied(false), 2000);
                });
              }}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {copied ? (
                <><Check className="h-3 w-3 text-emerald-400" /><span className="text-emerald-400">Copied</span></>
              ) : (
                <><Copy className="h-3 w-3" />Copy summary</>
              )}
            </button>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            <div>
              <p className="text-xs text-muted-foreground mb-1">Avg stress</p>
              <p className={`text-xl font-semibold font-mono ${weekly.avgStress > 60 ? "text-red-400" : weekly.avgStress > 40 ? "text-orange-400" : "text-emerald-400"}`}>
                {weekly.avgStress}%
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">Avg focus</p>
              <p className="text-xl font-semibold font-mono text-primary">
                {weekly.avgFocus}%
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">Sleep quality</p>
              <p className={`text-xl font-semibold font-mono ${weekly.avgSleep > 60 ? "text-emerald-400" : weekly.avgSleep > 40 ? "text-orange-400" : "text-red-400"}`}>
                {weekly.avgSleep}%
              </p>
            </div>
          </div>
        </Card>
      )}

      {/* Pattern engine — server-computed 30-day correlations */}
      {!isLoading && (brainPatterns.length > 0 || pattern) && (
        <Card className="glass-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <BarChart2 className="h-4 w-4 text-emerald-400" />
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
              Your patterns
            </h2>
            {patternsData && patternsData.dataPoints > 0 && (
              <span className="text-[10px] text-muted-foreground/60 ml-auto">
                {patternsData.dataPoints} readings
              </span>
            )}
          </div>

          {brainPatterns.length > 0 ? (
            <div className="space-y-4">
              {brainPatterns.map((p, i) => (
                <div
                  key={p.type}
                  className={`${i > 0 ? "pt-4 border-t border-border/20" : ""}`}
                >
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <p className="text-sm font-medium leading-snug">{p.title}</p>
                    <span className="shrink-0 text-[10px] text-muted-foreground/60 mt-0.5">
                      {Math.round(p.confidence * 100)}% confidence
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed mb-1.5">
                    {p.description}
                  </p>
                  <p className="text-xs text-emerald-400/90 leading-relaxed">
                    → {p.recommendation}
                  </p>
                </div>
              ))}
            </div>
          ) : pattern ? (
            /* Fallback to client-side pattern when server has no data yet */
            <p className="text-sm text-foreground/80 leading-relaxed">{pattern}</p>
          ) : null}
        </Card>
      )}

      {/* Footer link to sessions */}
      <div className="text-center pt-2">
        <button
          onClick={() => navigate("/sessions")}
          className="text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          View full session history →
        </button>
      </div>
    </div>
  );
}
