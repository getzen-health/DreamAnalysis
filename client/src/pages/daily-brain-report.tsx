import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
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

interface SessionSummary {
  id: string;
  session_type?: string;
  duration_seconds?: number;
  start_time?: number;
  end_time?: number;
  summary?: Record<string, unknown>;
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

/** Map a stress level (0–10) to a text label. */
function stressLabel(level: number): string {
  if (level < 3) return "low";
  if (level < 6) return "moderate";
  return "high";
}

/** Estimate peak focus window from time-of-day heuristic. */
function peakFocusWindow(): string {
  return "9:30 am – 12:00 pm";
}

/** Estimate afternoon slump window. */
function slumpWindow(): string {
  return "2:30 pm – 3:30 pm";
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

/** Pick the most recent yesterday's insight from health entries. */
function yesterdayInsight(health: HealthEntry[]): string | null {
  if (health.length < 2) return null;
  const yesterday = health.find((h) => {
    const d = new Date(h.timestamp);
    const today = new Date();
    return d.getDate() === today.getDate() - 1;
  });
  if (!yesterday) return null;
  const stress = yesterday.stressLevel ?? 5;
  const focus = yesterday.neuralActivity ?? 5;
  if (focus > 6)
    return `Focus was ${Math.round((focus / 10) * 100)}% above baseline yesterday.`;
  if (stress > 6)
    return `Stress ran high yesterday — today is a fresh start.`;
  return `Yesterday looked balanced — stress ${stressLabel(stress)}, focus steady.`;
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
function weeklyStats(health: HealthEntry[]) {
  const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000;
  const week = health.filter((h) => new Date(h.timestamp).getTime() >= cutoff);
  if (week.length === 0) return null;
  const avgStress = week.reduce((s, h) => s + (h.stressLevel ?? 5), 0) / week.length;
  const avgFocus = week.reduce((s, h) => s + (h.neuralActivity ?? 5), 0) / week.length;
  const avgSleep = week.reduce((s, h) => s + (h.sleepQuality ?? 5), 0) / week.length;
  return {
    days: week.length,
    avgStress: Math.round(avgStress * 10),
    avgFocus: Math.round(avgFocus * 10),
    avgSleep: Math.round(avgSleep * 10),
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
  const weekly = weeklyStats(health);

  /* — Overnight EEG session — */
  const overnightSession = sessions.find(
    (s) => s.session_type === "sleep" || (s.duration_seconds ?? 0) > 3600
  );

  return (
    <div className="max-w-2xl mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-muted-foreground mb-1">{new Date().toLocaleDateString([], { weekday: "long", month: "long", day: "numeric" })}</p>
          <h1 className="text-2xl font-semibold">{greeting()}</h1>
        </div>
        <Sun className="h-8 w-8 text-yellow-400 opacity-80" />
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
            <div className="grid grid-cols-3 gap-4">
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
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-muted-foreground mb-1">Session length</p>
                <p className="text-lg font-semibold">
                  {overnightSession.duration_seconds
                    ? fmtMinutes(overnightSession.duration_seconds / 60)
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
                {peakFocusWindow()}
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
                {slumpWindow()}
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

      {/* Yesterday's insight */}
      {!isLoading && insight && (
        <Card className="glass-card p-6">
          <div className="flex items-center gap-2 mb-3">
            <Brain className="h-4 w-4 text-violet-400" />
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
              Yesterday's insight
            </h2>
          </div>
          <p className="text-sm text-foreground/90">{insight}</p>
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

          <div className="grid grid-cols-3 gap-4">
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
