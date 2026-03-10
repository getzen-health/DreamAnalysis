import { useQuery } from "@tanstack/react-query";
import { type SessionSummary, type SleepMoodPrediction, predictSleepMood } from "@/lib/ml-api";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Moon,
  ArrowRight,
  Flame,
  BarChart2,
  Radio,
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

interface VoiceSnapshot {
  emotion: string;
  valence: number;
  arousal: number;
  confidence: number;
  stress_from_watch: number | null;
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

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
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
function recommendedAction(health: HealthEntry[], voice: VoiceSnapshot | null): {
  label: string;
  route: string;
  description: string;
} {
  if (!health.length && !voice) {
    return {
      label: "Run a voice check-in",
      route: "/emotions",
      description: "Capture a quick emotion snapshot to personalize today’s report",
    };
  }
  const latest = health[0];
  const stress = latest?.stressLevel ?? 0;
  const focus = latest?.neuralActivity ?? 5;
  const voiceStress = voice?.stress_from_watch ?? (voice ? clamp((voice.arousal - voice.valence + 1) * 3, 0, 10) : 0);

  if (stress > 6 || voiceStress > 6) {
    return {
      label: "Start coherence breathing",
      route: "/biofeedback",
      description: "4-min session to lower cortisol and reset",
    };
  }
  if (voice && voice.valence < -0.2) {
    return {
      label: "Check your emotion state",
      route: "/emotions",
      description: "Review your voice snapshot and reset before the day ramps up",
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
    label: "Open Daily Brain Monitor",
    route: "/",
    description: "Use your current state to protect your best focus window",
  };
}

function focusForecast(health: HealthEntry | undefined, voice: VoiceSnapshot | null): {
  score: number;
  label: string;
  window: string;
} {
  const sleepQuality = health?.sleepQuality ?? 5;
  const neuralActivity = health?.neuralActivity ?? 5;
  const voiceLift = voice ? clamp(((voice.valence + 1) / 2) * 10, 0, 10) : 5;
  const arousalLift = voice ? clamp(10 - Math.abs(voice.arousal - 0.55) * 10, 0, 10) : 5;
  const score = Math.round(clamp(sleepQuality * 0.4 + neuralActivity * 0.3 + voiceLift * 0.2 + arousalLift * 0.1, 0, 10) * 10);
  const label = score >= 75 ? "High focus potential" : score >= 55 ? "Steady focus day" : "Protect your energy";
  const window = score >= 75 ? "9:30am - 12:00pm" : score >= 55 ? "10:30am - 12:00pm" : "11:00am - 12:00pm";
  return { score, label, window };
}

function stressForecast(health: HealthEntry | undefined, voice: VoiceSnapshot | null): {
  score: number;
  label: string;
} {
  const healthStress = health?.stressLevel ?? 4;
  const sleepPenalty = health?.sleepQuality !== undefined ? clamp(10 - health.sleepQuality, 0, 10) : 5;
  const voiceStress = voice?.stress_from_watch ?? (voice ? clamp((voice.arousal - voice.valence + 1) * 3, 0, 10) : 4);
  const score = Math.round(clamp(healthStress * 0.45 + sleepPenalty * 0.2 + voiceStress * 0.35, 0, 10) * 10);
  const label = score >= 70 ? "High stress risk" : score >= 45 ? "Moderate stress risk" : "Low stress risk";
  return { score, label };
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

  const { data: latestVoice, isLoading: voiceLoading } =
    useQuery<VoiceSnapshot | null>({
      queryKey: ["voice-latest-brain-report", CURRENT_USER],
      queryFn: async () => {
        const res = await fetch(`/api/ml/voice-watch/latest/${CURRENT_USER}`);
        if (!res.ok) return null;
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

  /* — Sleep-to-mood prediction (requires health data to be present) — */
  const latestHealthForSleep = (health as HealthEntry[])[0] as HealthEntry | undefined;
  const hasSleepInput = !!(latestHealthForSleep?.sleepDuration || latestHealthForSleep?.sleepQuality);

  const { data: sleepMoodData } =
    useQuery<SleepMoodPrediction | null>({
      queryKey: ["sleep-mood-prediction", CURRENT_USER],
      queryFn: async () => {
        if (!hasSleepInput) return null;
        try {
          const sleepHours = latestHealthForSleep?.sleepDuration ?? 7;
          const quality = latestHealthForSleep?.sleepQuality ?? 7;
          // Derive deep_sleep_pct: quality 10/10 ≈ 25% deep, 5/10 ≈ 12.5%
          const deepSleepPct = (quality / 10) * 0.25;
          const sleepEfficiency = (quality / 10) * 0.95;
          return await predictSleepMood({
            total_sleep_hours: sleepHours,
            deep_sleep_pct: deepSleepPct,
            sleep_efficiency: sleepEfficiency,
            user_id: CURRENT_USER,
          });
        } catch {
          return null;
        }
      },
      enabled: hasSleepInput,
      staleTime: 15 * 60_000,
      retry: false,
    });

  const serverInsights: ServerInsight[] = serverInsightsData?.insights ?? [];
  const brainPatterns: BrainPattern[] = patternsData?.patterns ?? [];
  const voiceSnapshot = latestVoice ?? null;

  const isLoading = sessionsLoading || dreamsLoading || healthLoading || voiceLoading;

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
  const action = recommendedAction(health, voiceSnapshot);
  const insight = yesterdayInsight(health);
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

  const focusToday = focusForecast(latestHealth, voiceSnapshot);
  const stressToday = stressForecast(latestHealth, voiceSnapshot);
  const sourceLabel =
    overnightSession && voiceSnapshot && latestHealth
      ? "Based on: EEG + Voice + Health"
      : voiceSnapshot && latestHealth
      ? "Based on: Voice + Health"
      : latestHealth
      ? "Based on: Health"
      : voiceSnapshot
      ? "Based on: Voice"
      : "Based on: No inputs yet";

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
      ) : !latestHealth && !voiceSnapshot ? (
        <Card className="glass-card p-5">
          <div className="flex items-start gap-3">
            <Radio className="h-4 w-4 text-muted-foreground/50 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-foreground/80">No data yet</p>
              <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                Run a voice check-in, sync Apple Health, or connect EEG to generate today’s report.
              </p>
              <Button
                variant="ghost"
                size="sm"
                className="mt-2 h-7 px-2 text-xs text-primary"
                onClick={() => navigate("/emotions")}
              >
                Start check-in <ArrowRight className="h-3 w-3 ml-1" />
              </Button>
            </div>
          </div>
        </Card>
      ) : (
        <Card className="glass-card p-5">
          <p className="text-[11px] text-muted-foreground uppercase tracking-wide mb-3">Right now</p>
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-xs px-2.5 py-1 rounded-full border border-border/40 text-muted-foreground">
              {sourceLabel}
            </span>
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
            {voiceSnapshot && (
              <span className="text-xs px-2.5 py-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 capitalize">
                Voice: {voiceSnapshot.emotion}
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

      {/* Sleep-mood forecast card — hidden if ML backend unavailable or no sleep data */}
      {!isLoading && sleepMoodData && (
        <Card className="glass-card p-5">
          <p className="text-[11px] text-muted-foreground uppercase tracking-wide mb-3">Tonight's sleep predicts...</p>
          <div className="flex items-center gap-2 mb-2">
            <Badge
              className={
                sleepMoodData.mood_label === "positive"
                  ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30 border"
                  : sleepMoodData.mood_label === "challenging"
                  ? "bg-red-500/15 text-red-400 border-red-500/30 border"
                  : "bg-muted/40 text-muted-foreground border-border/40 border"
              }
              variant="outline"
            >
              {sleepMoodData.mood_label === "positive" ? "Positive mood" : sleepMoodData.mood_label === "challenging" ? "Challenging day" : "Neutral outlook"}
            </Badge>
          </div>
          {sleepMoodData.key_factor && (
            <p className="text-xs text-foreground/70 leading-relaxed">{sleepMoodData.key_factor}</p>
          )}
          <p className="text-xs text-muted-foreground mt-2 border-t border-border/20 pt-2">
            Best focus window: {sleepMoodData.predicted_focus_window}
          </p>
        </Card>
      )}

      {!isLoading && (latestHealth || voiceSnapshot) && (
        <Card className="glass-card p-5">
          <p className="text-[11px] text-muted-foreground uppercase tracking-wide mb-3">Today's forecast</p>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-lg font-semibold">{focusToday.score}%</p>
              <p className="text-[11px] text-muted-foreground">focus readiness</p>
              <p className="text-xs text-foreground/75 mt-1">{focusToday.label}</p>
            </div>
            <div>
              <p className="text-lg font-semibold">{stressToday.score}%</p>
              <p className="text-[11px] text-muted-foreground">stress risk</p>
              <p className="text-xs text-foreground/75 mt-1">{stressToday.label}</p>
            </div>
          </div>
          <p className="text-xs text-muted-foreground mt-3 border-t border-border/20 pt-3">
            Peak focus window: {focusToday.window}
          </p>
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
