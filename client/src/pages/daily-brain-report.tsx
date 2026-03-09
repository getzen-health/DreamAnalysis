import { useQuery } from "@tanstack/react-query";
import { type SessionSummary, getBrainReport, type BrainReport } from "@/lib/ml-api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Moon,
  ArrowRight,
  Flame,
  BarChart2,
  Radio,
  Mic,
  Activity,
  Heart,
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

/** Map a stress level (0–10) to a text label. */
function stressLabel(level: number): string {
  if (level < 3) return "low";
  if (level < 6) return "moderate";
  return "high";
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

/* ── Main page ───────────────────────────────────────────────── */
export default function DailyBrainReport() {
  const [, navigate] = useLocation();

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

  const { data: mlReport } = useQuery<BrainReport>({
    queryKey: ["brain-report-ml", CURRENT_USER],
    queryFn: () => getBrainReport(CURRENT_USER),
    staleTime: 5 * 60_000,
    retry: false,
  });

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

  // Use ML recommended action if health is sparse but voice data is available
  const effectiveAction = !latestHealth && mlReport?.recommended_action
    ? { label: mlReport.recommended_action, route: "/", description: "Based on your voice + health patterns" }
    : action;
  const latestStress = latestHealth?.stressLevel ?? null;
  const streak = currentStreak(sessions);

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
          {mlReport && mlReport.data_sources.length > 0 && (
            <span className="text-[10px] text-muted-foreground">
              Based on: {mlReport.data_sources.join(" + ")}
            </span>
          )}
        </div>
      </div>

      {/* ML focus + stress forecast (voice/health-based, no EEG needed) */}
      {mlReport && (mlReport.data_sources.includes("voice") || mlReport.data_sources.includes("health")) && (
        <Card className="glass-card p-4">
          <p className="text-[11px] text-muted-foreground uppercase tracking-wide mb-3">Today's Forecast</p>
          <div className="grid grid-cols-2 gap-3">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-blue-400 shrink-0" />
              <div>
                <p className="text-sm font-semibold">{mlReport.focus_forecast.toFixed(0)}<span className="text-xs text-muted-foreground">/100</span></p>
                <p className="text-[11px] text-muted-foreground">Focus</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Heart className="h-4 w-4 text-red-400 shrink-0" />
              <div>
                <p className="text-sm font-semibold">{mlReport.stress_risk.toFixed(0)}<span className="text-xs text-muted-foreground">/100</span></p>
                <p className="text-[11px] text-muted-foreground">Stress risk</p>
              </div>
            </div>
          </div>
          {mlReport.peak_focus_window && (
            <p className="text-xs text-muted-foreground mt-2">
              Peak focus window: <span className="text-foreground">{mlReport.peak_focus_window}</span>
            </p>
          )}
          {mlReport.insight && (
            <p className="mt-2 text-xs text-foreground/75 border-t border-border/20 pt-2 leading-relaxed">
              {mlReport.insight}
            </p>
          )}
        </Card>
      )}

      {/* Card 1 — Right now */}
      {isLoading ? (
        <SkeletonCard />
      ) : !latestHealth ? (
        <Card className="glass-card p-5">
          <div className="flex items-start gap-3">
            <Radio className="h-4 w-4 text-muted-foreground/50 shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-foreground/80">No health data yet</p>
              <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                Do a quick voice check-in to start your report — no EEG required.
              </p>
              <div className="flex gap-2 mt-2 flex-wrap">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 px-2 text-xs border-accent/40 text-accent hover:border-accent"
                  onClick={() => navigate("/")}
                >
                  <Mic className="h-3 w-3 mr-1" /> Voice check-in
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-xs text-muted-foreground"
                  onClick={() => navigate("/settings")}
                >
                  Sync Health <ArrowRight className="h-3 w-3 ml-1" />
                </Button>
              </div>
            </div>
          </div>
        </Card>
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
      {!isLoading && (latestHealth || mlReport) && (
        <Card className="glass-card p-5">
          <p className="text-[11px] text-muted-foreground uppercase tracking-wide mb-3">Do this now</p>
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-medium">{effectiveAction.label}</p>
              <p className="text-xs text-muted-foreground mt-0.5">{effectiveAction.description}</p>
            </div>
            <Button
              size="sm"
              className="shrink-0 min-w-[72px]"
              onClick={() => navigate(effectiveAction.route)}
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
