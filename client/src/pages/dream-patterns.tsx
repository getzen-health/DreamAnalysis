import { useState, useEffect, useRef, useCallback } from "react";
import { SectionErrorBoundary } from "@/components/section-error-boundary";
import { useQuery } from "@tanstack/react-query";
import { Skeleton } from "@/components/ui/skeleton";
import { Card } from "@/components/ui/card";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useMemo } from "react";
import { Moon, TrendingUp, Brain, Activity, Sparkles, Radio, Share2 } from "lucide-react";
import { useDevice } from "@/hooks/use-device";
import { listSessions, type SessionSummary } from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";
import { resolveUrl } from "@/lib/queryClient";
import { DreamFusionCard } from "@/components/dream-fusion-card";
import { fuseDreamBiometrics, type DreamEntry, type OvernightBiometrics } from "@/lib/dream-biometric-fusion";
import { DreamPatternsCard } from "@/components/dream-patterns-card";
import { NightmareRecurrenceCard } from "@/components/nightmare-recurrence-card";
import type { NightmareRecurrenceData } from "@/lib/nightmare-recurrence";
import { DreamQualityCard } from "@/components/dream-quality-card";
import type { DreamQualityTrend } from "@/lib/dream-quality-score";
import { WeeklySynthesisCard } from "@/components/weekly-synthesis-card";
import { DreamHistoryCard } from "@/components/dream-history-card";
import { EmotionalArcTrendCard } from "@/components/emotional-arc-trend-card";
import { PresleepIntentionCard } from "@/components/presleep-intention-card";
import { DreamRecallHeatmap } from "@/components/dream-recall-heatmap";
import { DreamSymbolContextCard } from "@/components/dream-symbol-context-card";
import { DreamArchetypeCard } from "@/components/dream-archetype-card";
import { LucidityPredictorCard } from "@/components/lucidity-predictor-card";
import type { DreamEntry as ThemeTrackerEntry } from "@/lib/dream-theme-tracker";
import { renderDreamShareCard, type DreamShareData } from "@/lib/dream-share-card";
import { shareImage } from "@/lib/share-utils";

/* ---------- constants ---------- */
const PERIOD_TABS = [
  { label: "Today", days: 1 },
  { label: "Week", days: 7 },
  { label: "Month", days: 30 },
  { label: "3 Months", days: 90 },
  { label: "Year", days: 365 },
];

/* ---------- types ---------- */
interface NightlyPoint {
  time: string;
  remMinutes: number;
  dreamEpisodes: number;
  avgIntensity: number;
  avgLucidity: number;
  sleepStageIdx: number;
}

interface HypnogramPoint {
  time: string;
  stage: number;
  label: string;
}

interface SessionWellnessPoint {
  date: string;
  flow: number;
  relaxation: number;
  focus: number;
}

interface DreamPatternData {
  period: number;
  entryCount: number;
  themes: { name: string; count: number }[];
  symbols: { name: string; count: number }[];
  sentimentTrend: { date: string; valence: number }[];
  topInsights: Array<{ insight: string | null; keyInsight: string | null; date: string }>;
  nightmareCount?: number;
  nightmareDates?: string[];
  counts?: { last7: number; last30: number; last90: number };
}

interface DreamSymbol {
  id: string;
  symbol: string;
  meaning: string | null;
  frequency: number;
  firstSeen: string;
  lastSeen: string;
}

const STAGE_NAMES = ["N3 (Deep)", "N2 (Light)", "N1", "REM", "Wake"];
const STAGE_VALUES: Record<string, number> = { Wake: 4, REM: 3, N1: 2, N2: 1, N3: 0 };

/* ---------- helpers ---------- */
function avgNums(arr: number[]): number {
  return arr.length ? Math.round(arr.reduce((a, b) => a + b, 0) / arr.length) : 0;
}

function buildWellnessChartData(sessions: SessionSummary[], days: number): SessionWellnessPoint[] {
  const map: Record<
    string,
    { flow: number[]; relaxation: number[]; focus: number[]; ts: number }
  > = {};

  for (const s of sessions) {
    if (s.summary?.avg_focus == null) continue;
    const d = new Date((s.start_time ?? 0) * 1000);
    let key: string;
    let ts: number;

    if (days <= 1) {
      key = d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
      ts = d.getTime();
    } else if (days <= 7) {
      key = d.toLocaleDateString("en-US", { weekday: "short" });
      ts = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    } else if (days <= 30) {
      key = d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
      ts = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    } else if (days <= 90) {
      const ws = new Date(d);
      ws.setDate(d.getDate() - d.getDay());
      key = ws.toLocaleDateString("en-US", { month: "short", day: "numeric" });
      ts = ws.getTime();
    } else {
      key = d.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
      ts = new Date(d.getFullYear(), d.getMonth(), 1).getTime();
    }

    if (!map[key]) map[key] = { flow: [], relaxation: [], focus: [], ts };
    map[key].flow.push((s.summary.avg_flow ?? 0) * 100);
    map[key].relaxation.push((s.summary.avg_relaxation ?? 0) * 100);
    map[key].focus.push((s.summary.avg_focus ?? 0) * 100);
  }

  return Object.entries(map)
    .sort(([, a], [, b]) => a.ts - b.ts)
    .map(([date, data]) => ({
      date,
      flow: avgNums(data.flow),
      relaxation: avgNums(data.relaxation),
      focus: avgNums(data.focus),
    }));
}

/* ========== Component ========== */
export default function DreamPatterns() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;

  const dreamDetection = analysis?.dream_detection;
  const sleepStaging = analysis?.sleep_staging;

  // Period selector
  const [periodDays, setPeriodDays] = useState(1);
  const isLiveToday = periodDays === 1;

  const userId = getParticipantId();

  // Sessions query
  const { data: allSessions = [] } = useQuery<SessionSummary[]>({
    queryKey: ["sessions"],
    queryFn: () => listSessions(),
    retry: false,
    staleTime: 2 * 60 * 1000,
    refetchInterval: 60_000,
  });

  // Dream patterns query — aggregated themes/symbols/sentiment from dream journal
  const patternDays = periodDays === 1 ? 7 : periodDays;
  const { data: dreamPatterns, isLoading: patternsLoading } = useQuery<DreamPatternData>({
    queryKey: ["dream-patterns", userId, patternDays],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/dream-patterns/${userId}?days=${patternDays}`));
      if (!res.ok) throw new Error("Failed to fetch dream patterns");
      return res.json();
    },
    staleTime: 5 * 60 * 1000,
    retry: false,
  });

  // Dream symbol library — personal recurring symbols with frequency + meaning
  const { data: dreamSymbols = [] } = useQuery<DreamSymbol[]>({
    queryKey: ["dream-symbols", userId],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/dream-symbols/${userId}`));
      if (!res.ok) return [];
      return res.json();
    },
    staleTime: 10 * 60 * 1000,
    retry: false,
  });

  // Dream quality trend — 14-day composite score sparkline
  const { data: dreamQualityTrend } = useQuery<DreamQualityTrend>({
    queryKey: ["dream-quality-trend", userId],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/dream-quality-trend/${userId}?days=14`));
      if (!res.ok) throw new Error("Failed to fetch dream quality trend");
      return res.json();
    },
    staleTime: 10 * 60 * 1000,
    retry: false,
  });

  // Nightmare recurrence + IRT effectiveness — 14-day rolling window
  const { data: nightmareRecurrence } = useQuery<NightmareRecurrenceData>({
    queryKey: ["nightmare-recurrence", userId],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/nightmare-recurrence/${userId}`));
      if (!res.ok) throw new Error("Failed to fetch nightmare recurrence");
      return res.json();
    },
    staleTime: 10 * 60 * 1000,
    retry: false,
  });

  // Latest dream analysis — auto-fetched, no user action needed
  const { data: latestDreamArr } = useQuery<{ dreamText?: string; emotions?: Array<{emotion: string; intensity: number} | string>; lucidityScore?: number; sleepQuality?: number; timestamp?: string }[]>({
    queryKey: ["dream-analysis", userId],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/dream-analysis/${userId}`));
      if (!res.ok) return [];
      return res.json();
    },
    staleTime: 5 * 60 * 1000,
    retry: false,
  });

  // Latest health payload — from ML backend (EEG session data includes HRV, HR, sleep stages)
  const { data: latestPayload } = useQuery<Record<string, number | null>>({
    queryKey: ["health-latest", userId],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/health-samples/${userId}?limit=1`));
      if (!res.ok) return {};
      const rows = await res.json();
      return rows?.[0] ?? {};
    },
    staleTime: 5 * 60 * 1000,
    retry: false,
  });

  // Biometric fusion — EEG-derived narrative + overnight physiology, zero user input
  const dreamFusionInsight = useMemo(() => {
    const row = latestDreamArr?.[0];
    if (!row?.dreamText) return null;
    const dream: DreamEntry = {
      dreamText: row.dreamText,
      emotions: Array.isArray(row.emotions)
        ? row.emotions.map((e) => (typeof e === "string" ? e : e.emotion))
        : [],
      lucidityScore: row.lucidityScore != null ? row.lucidityScore / 100 : undefined,
      sleepQuality: row.sleepQuality ?? undefined,
      timestamp: row.timestamp ?? new Date().toISOString(),
    };
    const bio: OvernightBiometrics = {};
    if (latestPayload) {
      if (latestPayload.resting_heart_rate != null) bio.avgHeartRate = latestPayload.resting_heart_rate;
      if (latestPayload.hrv_sdnn != null) bio.hrvSdnn = latestPayload.hrv_sdnn;
      if (latestPayload.sleep_total_hours != null) bio.sleepDuration = latestPayload.sleep_total_hours;
      if (latestPayload.sleep_efficiency != null) bio.sleepEfficiency = latestPayload.sleep_efficiency;
      if (latestPayload.sleep_deep_hours != null && latestPayload.sleep_total_hours)
        bio.deepSleepPct = Math.round((latestPayload.sleep_deep_hours / latestPayload.sleep_total_hours) * 100);
      if (latestPayload.sleep_rem_hours != null && latestPayload.sleep_total_hours)
        bio.remSleepPct = Math.round((latestPayload.sleep_rem_hours / latestPayload.sleep_total_hours) * 100);
    }
    return fuseDreamBiometrics(dream, bio);
  }, [latestDreamArr, latestPayload]);

  // Transform dream analysis entries into ThemeTrackerEntry[] for DreamPatternsCard
  const themeTrackerDreams: ThemeTrackerEntry[] = useMemo(() => {
    if (!Array.isArray(latestDreamArr)) return [];
    return latestDreamArr
      .filter((d) => d.dreamText)
      .map((d) => ({
        dreamText: d.dreamText ?? "",
        emotions: Array.isArray(d.emotions)
          ? d.emotions.map((e) => (typeof e === "string" ? e : e.emotion))
          : [],
        symbols: Array.isArray((d as Record<string, unknown>).symbols)
          ? ((d as Record<string, unknown>).symbols as string[])
          : [],
        lucidityScore:
          d.lucidityScore != null ? d.lucidityScore / 100 : undefined,
        timestamp: d.timestamp ?? new Date().toISOString(),
      }));
  }, [latestDreamArr]);

  const cutoff = Date.now() / 1000 - periodDays * 86400;
  const periodSessions = allSessions.filter((s) => (s.start_time ?? 0) >= cutoff);
  const wellnessChartData = buildWellnessChartData(periodSessions, periodDays);
  const hasWellnessData = wellnessChartData.length >= 1;

  // Accumulate live session data for charts
  const [sessionData, setSessionData] = useState<NightlyPoint[]>([]);
  const [hypnogram, setHypnogram] = useState<HypnogramPoint[]>([]);
  const [remCycles, setRemCycles] = useState<{ cycle: string; duration: number; intensity: number; lucidity: number }[]>([]);
  const [inRem, setInRem] = useState(false);
  const [remStart, setRemStart] = useState(0);

  // Share card state
  const [shareGenerating, setShareGenerating] = useState(false);
  const shareCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!isStreaming || !sleepStaging) return;

    const now = new Date();
    const timeStr = now.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    const stage = sleepStaging.stage ?? "Wake";
    const stageIdx = STAGE_VALUES[stage] ?? 4;

    setHypnogram((prev) => [...prev.slice(-60), { time: timeStr, stage: stageIdx, label: stage }]);

    const remLikelihood = dreamDetection?.rem_likelihood ?? 0;
    const isDreaming = dreamDetection?.is_dreaming ?? false;
    const intensity = Math.round((dreamDetection?.dream_intensity ?? 0) * 100);
    const lucidity = Math.round((dreamDetection?.lucidity_estimate ?? 0) * 100);

    setSessionData((prev) => [
      ...prev.slice(-60),
      { time: timeStr, remMinutes: Math.round(remLikelihood * 100), dreamEpisodes: isDreaming ? 1 : 0, avgIntensity: intensity, avgLucidity: lucidity, sleepStageIdx: stageIdx },
    ]);

    const isCurrentlyRem = stage === "REM";
    if (isCurrentlyRem && !inRem) {
      setInRem(true);
      setRemStart(Date.now());
    } else if (!isCurrentlyRem && inRem && remStart > 0) {
      setInRem(false);
      const durationMin = Math.round((Date.now() - remStart) / 60000);
      if (durationMin > 0) {
        setRemCycles((prev) => [...prev, { cycle: `Cycle ${prev.length + 1}`, duration: Math.max(1, durationMin), intensity, lucidity }]);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  // Summary stats
  const totalDreamFrames = sessionData.filter((d) => d.dreamEpisodes > 0).length;
  const avgRem = sessionData.length > 0 ? Math.round(sessionData.reduce((s, d) => s + d.remMinutes, 0) / sessionData.length) : 0;
  const avgIntensity = sessionData.length > 0 ? Math.round(sessionData.reduce((s, d) => s + d.avgIntensity, 0) / sessionData.length) : 0;

  // Share dream card handler
  const handleShareDream = useCallback(async () => {
    if (shareGenerating) return;

    // Gather data for the share card
    const row = latestDreamArr?.[0];
    const dreamSummary = row?.dreamText ?? "A vivid dream captured by EEG overnight monitoring.";
    const emotions = Array.isArray(row?.emotions)
      ? row.emotions.map((e) => (typeof e === "string" ? e : e.emotion))
      : [];
    const emotionalTone = emotions[0] ?? "mysterious";

    const sleepHours = latestPayload?.sleep_total_hours;
    let sleepDuration = "--";
    if (sleepHours != null) {
      const h = Math.floor(sleepHours);
      const m = Math.round((sleepHours - h) * 60);
      sleepDuration = `${h}h ${m}m`;
    }

    const remPct = latestPayload?.sleep_rem_hours != null && sleepHours
      ? Math.round((latestPayload.sleep_rem_hours / sleepHours) * 100)
      : avgRem;

    const dreamCount = Array.isArray(latestDreamArr) ? latestDreamArr.filter((d) => d.dreamText).length : 0;

    const now = new Date();
    const dateStr = now.toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });

    const shareData: DreamShareData = {
      dreamSummary,
      emotionalTone,
      sleepDuration,
      remPercentage: remPct,
      dreamCount: dreamCount || 1,
      date: dateStr,
    };

    setShareGenerating(true);
    try {
      // Create or reuse offscreen canvas
      if (!shareCanvasRef.current) {
        shareCanvasRef.current = document.createElement("canvas");
      }
      const blob = await renderDreamShareCard(shareCanvasRef.current, shareData);

      // Convert blob to data URL for share-utils
      const reader = new FileReader();
      const dataUrl = await new Promise<string>((resolve, reject) => {
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });

      const filename = `antarai-dream-${Date.now()}.png`;
      await shareImage(dataUrl, filename);
    } catch {
      // Share cancelled or failed — no-op
    } finally {
      setShareGenerating(false);
    }
  }, [latestDreamArr, latestPayload, avgRem, shareGenerating]);

  // Historical summary stats
  const histAvgFlow = periodSessions.length > 0
    ? Math.round(periodSessions.reduce((s, x) => s + (x.summary?.avg_flow ?? 0), 0) / periodSessions.length * 100)
    : 0;
  const histAvgRelax = periodSessions.length > 0
    ? Math.round(periodSessions.reduce((s, x) => s + (x.summary?.avg_relaxation ?? 0), 0) / periodSessions.length * 100)
    : 0;
  const histTotalHours = Math.round(
    periodSessions.reduce((s, x) => s + (x.summary?.duration_sec ?? 0), 0) / 3600 * 10
  ) / 10;

  const TOOLTIP_STYLE = {
    cursor: { stroke: "hsl(220, 12%, 55%)", strokeWidth: 1, strokeDasharray: "4 4" },
    contentStyle: { background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 11 },
    labelStyle: { color: "var(--muted-foreground)", marginBottom: 4, fontSize: 10 },
    itemStyle: { padding: "1px 0" },
  };

  // Loading skeleton
  if (patternsLoading) {
    return (
      <main className="p-6 space-y-6 max-w-5xl">
        <div className="flex items-center gap-2">
          <Skeleton className="h-5 w-5 rounded-full" />
          <Skeleton className="h-6 w-40" />
        </div>
        {[...Array(4)].map((_, i) => (
          <div key={i} className="rounded-xl border border-border p-5 space-y-3">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-3 w-48" />
          </div>
        ))}
      </main>
    );
  }

  // Empty state — no dreams logged yet
  if (!dreamPatterns && !patternsLoading) {
    return (
      <main className="p-6 max-w-5xl">
        <div className="flex items-center gap-2 mb-6">
          <Moon className="h-5 w-5 text-secondary" />
          <span className="text-lg font-semibold">Dream Patterns</span>
        </div>
        <div className="rounded-xl border border-border bg-card p-10 flex flex-col items-center gap-4 text-center">
          <Moon className="h-10 w-10 text-muted-foreground/40" />
          <div>
            <p className="font-semibold text-foreground">No dream data yet</p>
            <p className="text-sm text-muted-foreground mt-1">Log your first dream in the Dream Journal to see patterns here.</p>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="p-6 space-y-6 max-w-5xl">
      {/* Connection Banner */}
      {!isStreaming && isLiveToday && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          Manual dream logs are shown below. Optional overnight EEG can add live sleep staging and automatic dream detection later.
        </div>
      )}

      {/* Period Selector */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Moon className="h-5 w-5 text-secondary" />
          <span className="text-lg font-semibold">Dream Patterns</span>
          {isLiveToday && isStreaming && (
            <span className="text-[10px] font-mono text-primary animate-pulse">● LIVE</span>
          )}
          <button
            onClick={handleShareDream}
            disabled={shareGenerating}
            className="ml-2 p-1.5 rounded-lg bg-primary/10 border border-primary/20 text-primary
              hover:bg-primary/20 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
            title="Share dream summary card"
          >
            <Share2 className="h-4 w-4" />
          </button>
        </div>
        <div className="flex gap-1 flex-wrap">
          {PERIOD_TABS.map((tab) => (
            <button
              key={tab.days}
              onClick={() => setPeriodDays(tab.days)}
              className={`px-3 py-1 text-xs rounded-full transition-colors ${
                periodDays === tab.days
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Summary Stats */}
      {isLiveToday ? (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: "Dream Frames", value: totalDreamFrames, sub: "this session", color: "text-secondary" },
            { label: "Avg REM %", value: `${avgRem}`, sub: "session avg", color: "text-primary" },
            { label: "Avg Intensity", value: `${avgIntensity}%`, sub: "session avg", color: "text-accent" },
            { label: "REM Cycles", value: remCycles.length, sub: "this session", color: "text-foreground" },
          ].map((stat) => (
            <Card key={stat.label} className="glass-card p-4 hover-glow text-center">
              <p className={`text-2xl font-semibold ${stat.color}`}>{stat.value}</p>
              <p className="text-xs text-muted-foreground mt-1">{stat.label}</p>
              <p className="text-[10px] text-muted-foreground/60">{stat.sub}</p>
            </Card>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: "Sessions", value: periodSessions.length, sub: "in period", color: "text-primary" },
            { label: "Total Hours", value: `${histTotalHours}h`, sub: "recorded", color: "text-secondary" },
            { label: "Avg Flow", value: `${histAvgFlow}%`, sub: "period avg", color: "text-accent" },
            { label: "Avg Relaxation", value: `${histAvgRelax}%`, sub: "period avg", color: "text-success" },
          ].map((stat) => (
            <Card key={stat.label} className="glass-card p-4 hover-glow text-center">
              <p className={`text-2xl font-semibold ${stat.color}`}>{stat.value}</p>
              <p className="text-xs text-muted-foreground mt-1">{stat.label}</p>
              <p className="text-[10px] text-muted-foreground/60">{stat.sub}</p>
            </Card>
          ))}
        </div>
      )}

      {/* ── TODAY: Live charts ─────────────────────────────── */}
      {isLiveToday && (
        <>
          {/* Sleep Architecture (Hypnogram) */}
          <Card className="glass-card p-5 hover-glow">
            <div className="flex items-center gap-2 mb-2">
              <Brain className="h-4 w-4 text-secondary" />
              <h3 className="text-sm font-medium">Sleep Architecture</h3>
              {isStreaming && (
                <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
              )}
            </div>
            {hypnogram.length < 2 ? (
              <div className="h-[200px] flex items-center justify-center text-sm text-muted-foreground">
                {isStreaming ? "Collecting sleep stage data..." : "Connect device to see hypnogram"}
              </div>
            ) : (
              <>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={hypnogram.slice(-40)}>
                  <defs>
                    <linearGradient id="hypnoGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="hsl(262, 45%, 65%)" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="hsl(262, 45%, 65%)" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
                  <YAxis
                    domain={[0, 4]}
                    ticks={[0, 1, 2, 3, 4]}
                    tickFormatter={(v: number) => STAGE_NAMES[v] || ""}
                    tick={{ fontSize: 9, fill: "hsl(220, 12%, 42%)" }}
                    width={70}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    {...TOOLTIP_STYLE}
                    formatter={(value: number) => [STAGE_NAMES[value] || "Unknown"]}
                  />
                  <Area type="stepAfter" dataKey="stage" name="Stage" stroke="hsl(262, 45%, 65%)" fill="url(#hypnoGrad)" strokeWidth={2} activeDot={{ r: 4, fill: "hsl(262, 45%, 65%)" }} />
                </AreaChart>
              </ResponsiveContainer>
              <div className="flex gap-3 mt-2 flex-wrap">
                {[
                  { label: "Wake", color: "hsl(38, 85%, 58%)" },
                  { label: "N1 Light", color: "hsl(200, 60%, 60%)" },
                  { label: "N2 Sleep", color: "hsl(220, 60%, 55%)" },
                  { label: "N3 Deep", color: "hsl(240, 60%, 50%)" },
                  { label: "REM", color: "hsl(262, 45%, 65%)" },
                ].map((s) => (
                  <div key={s.label} className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-sm" style={{ background: s.color }} />
                    <span className="text-[10px] text-muted-foreground">{s.label}</span>
                  </div>
                ))}
              </div>
              </>
            )}
          </Card>

          {/* Dream Activity + REM Likelihood */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="glass-card p-5 hover-glow">
              <div className="flex items-center gap-2 mb-2">
                <Moon className="h-4 w-4 text-secondary" />
                <h3 className="text-sm font-medium">Dream Activity</h3>
              </div>
              {sessionData.length < 2 ? (
                <div className="h-[200px] flex items-center justify-center text-sm text-muted-foreground">
                  {isStreaming ? "Collecting..." : "No data"}
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={sessionData.slice(-30)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.4} />
                    <XAxis dataKey="time" tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${v}%`} width={32} />
                    <Tooltip
                      cursor={{ fill: "hsl(262, 45%, 65%)", opacity: 0.1 }}
                      contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 11 }}
                      labelStyle={{ color: "var(--muted-foreground)", marginBottom: 4, fontSize: 10 }}
                      itemStyle={{ padding: "1px 0" }}
                      formatter={(value: number) => [`${value}%`, "Dream Intensity"]}
                    />
                    <Bar dataKey="avgIntensity" fill="hsl(262, 45%, 65%)" radius={[4, 4, 0, 0]} name="Intensity %" />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </Card>

            <Card className="glass-card p-5 hover-glow">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="h-4 w-4 text-primary" />
                <h3 className="text-sm font-medium">REM Likelihood</h3>
              </div>
              {sessionData.length < 2 ? (
                <div className="h-[200px] flex items-center justify-center text-sm text-muted-foreground">
                  {isStreaming ? "Collecting..." : "No data"}
                </div>
              ) : (
                <>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={sessionData.slice(-30)}>
                    <defs>
                      <linearGradient id="remPatternGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="hsl(152, 60%, 48%)" stopOpacity={0.3} />
                        <stop offset="100%" stopColor="hsl(152, 60%, 48%)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.4} />
                    <XAxis dataKey="time" tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0, 100]} hide />
                    <Tooltip
                      {...TOOLTIP_STYLE}
                      formatter={(value: number) => [`${value}%`]}
                    />
                    <Area type="monotone" dataKey="remMinutes" stroke="hsl(152, 60%, 48%)" fill="url(#remPatternGrad)" strokeWidth={2} dot={false} name="REM %" activeDot={{ r: 4, fill: "hsl(152, 60%, 48%)" }} />
                  </AreaChart>
                </ResponsiveContainer>
                <div className="flex items-center gap-1.5 mt-2">
                  <div className="w-3 h-0.5 rounded" style={{ background: "hsl(152, 60%, 48%)" }} />
                  <span className="text-[10px] text-muted-foreground">REM likelihood (0–100%)</span>
                </div>
                </>
              )}
            </Card>
          </div>

          {/* REM Cycle Progression */}
          {remCycles.length > 0 && (
            <Card className="glass-card p-5 hover-glow">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="h-4 w-4 text-primary" />
                <h3 className="text-sm font-medium">REM Cycle Progression</h3>
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={remCycles}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.4} />
                  <XAxis dataKey="cycle" tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
                  <Tooltip
                    {...TOOLTIP_STYLE}
                  />
                  <Line type="monotone" dataKey="intensity" stroke="hsl(38, 85%, 58%)" strokeWidth={2} dot={{ r: 4, fill: "hsl(38, 85%, 58%)" }} name="Intensity %" activeDot={{ r: 5, fill: "hsl(38, 85%, 58%)" }} />
                  <Line type="monotone" dataKey="lucidity" stroke="hsl(200, 70%, 55%)" strokeWidth={2} dot={{ r: 4, fill: "hsl(200, 70%, 55%)" }} name="Lucidity %" activeDot={{ r: 5, fill: "hsl(200, 70%, 55%)" }} />
                  <Line type="monotone" dataKey="duration" stroke="hsl(152, 60%, 48%)" strokeWidth={2} dot={{ r: 4, fill: "hsl(152, 60%, 48%)" }} name="Duration (min)" activeDot={{ r: 5, fill: "hsl(152, 60%, 48%)" }} />
                </LineChart>
              </ResponsiveContainer>
              <div className="flex justify-center gap-4 mt-3">
                {[
                  { label: "Intensity", color: "hsl(38, 85%, 58%)" },
                  { label: "Lucidity", color: "hsl(200, 70%, 55%)" },
                  { label: "Duration", color: "hsl(152, 60%, 48%)" },
                ].map((item) => (
                  <div key={item.label} className="flex items-center gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ background: item.color }} />
                    <span className="text-[10px] text-muted-foreground">{item.label}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* AI Pattern Analysis */}
          {isStreaming && sessionData.length > 5 && (
            <PatternInsightCard
              sessionLen={sessionData.length}
              dreamFrames={totalDreamFrames}
              avgRem={avgRem}
              remCycles={remCycles.length}
              frameTs={latestFrame?.timestamp}
            />
          )}
        </>
      )}

      {/* ── HISTORICAL: Session-based charts ──────────────── */}
      {!isLiveToday && (
        <>
          {/* Weekly Dream Synthesis — LLM-generated narrative from structured metadata */}
          <SectionErrorBoundary label="Weekly Synthesis"><WeeklySynthesisCard userId={userId} /></SectionErrorBoundary>

          {/* Dream Quality Score + 14-day sparkline */}
          {dreamQualityTrend && (
            <SectionErrorBoundary label="Dream Quality"><DreamQualityCard data={dreamQualityTrend} /></SectionErrorBoundary>
          )}

          {/* Pattern Insights from dream journal */}
          {/* Dream + Biometric Fusion — auto-generated from EEG, no user input */}
          <SectionErrorBoundary label="Dream Fusion"><DreamFusionCard insight={dreamFusionInsight} /></SectionErrorBoundary>

          {/* Longitudinal dream theme tracking — 7/30/90 day patterns (#549) */}
          {themeTrackerDreams.length > 0 && (
            <SectionErrorBoundary label="Dream Patterns"><DreamPatternsCard dreams={themeTrackerDreams} /></SectionErrorBoundary>
          )}

          {dreamPatterns && dreamPatterns.entryCount > 0 && (
            <Card className="glass-card p-5 hover-glow border-secondary/20">
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="h-4 w-4 text-secondary" />
                <h3 className="text-sm font-medium">Pattern Insights</h3>
                <span className="ml-auto text-[10px] text-muted-foreground">
                  {dreamPatterns.entryCount} dream{dreamPatterns.entryCount !== 1 ? "s" : ""} analyzed over {dreamPatterns.period} days
                </span>
              </div>

              {/* 7 / 30 / 90 day counts */}
              {dreamPatterns.counts && (
                <div className="flex gap-3 mb-3">
                  {(["last7", "last30", "last90"] as const).map((key) => (
                    <div key={key} className="flex-1 text-center rounded-lg bg-muted/30 py-1.5">
                      <p className="text-base font-semibold text-foreground">{dreamPatterns.counts![key]}</p>
                      <p className="text-[9px] text-muted-foreground uppercase tracking-wide">
                        {key === "last7" ? "7d" : key === "last30" ? "30d" : "90d"}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              <div className="space-y-3">
                {dreamPatterns.themes.length > 0 && (
                  <div>
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1.5">Top Themes</p>
                    <div className="flex flex-wrap gap-1.5">
                      {dreamPatterns.themes.slice(0, 3).map((t) => (
                        <span
                          key={t.name}
                          className="text-[11px] px-2.5 py-0.5 rounded-full bg-secondary/15 text-secondary capitalize"
                        >
                          {t.name} · {t.count}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {dreamPatterns.symbols.length > 0 && (
                  <div>
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1.5">Top Symbols</p>
                    <div className="flex flex-wrap gap-1.5">
                      {dreamPatterns.symbols.slice(0, 3).map((s) => (
                        <span
                          key={s.name}
                          className="text-[11px] px-2.5 py-0.5 rounded-full bg-primary/10 text-primary capitalize"
                        >
                          {s.name} · {s.count}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Nightmare frequency */}
                {(dreamPatterns.nightmareCount ?? 0) > 0 && (
                  <div className="flex items-center gap-2 rounded-lg bg-destructive/10 border border-destructive/20 px-3 py-2">
                    <Moon className="h-3.5 w-3.5 text-destructive shrink-0" />
                    <p className="text-[11px] text-destructive">
                      {dreamPatterns.nightmareCount} nightmare{dreamPatterns.nightmareCount !== 1 ? "s" : ""} in this period
                    </p>
                  </div>
                )}

                {/* Key insight from most-recent dream with one */}
                {dreamPatterns.topInsights[0] && (dreamPatterns.topInsights[0].keyInsight || dreamPatterns.topInsights[0].insight) && (
                  <blockquote className="border-l-2 border-secondary/40 pl-3 mt-2">
                    <p className="text-xs text-muted-foreground italic leading-relaxed line-clamp-3">
                      {dreamPatterns.topInsights[0].keyInsight ?? dreamPatterns.topInsights[0].insight}
                    </p>
                    {dreamPatterns.topInsights[0].date && (
                      <p className="text-[10px] text-muted-foreground/60 mt-1">{dreamPatterns.topInsights[0].date}</p>
                    )}
                  </blockquote>
                )}
              </div>
            </Card>
          )}

          {/* Dream Symbol Library — recurring symbols across all dreams */}
          {dreamSymbols.length > 0 && (
            <Card className="glass-card p-5 hover-glow border-primary/20">
              <div className="flex items-center gap-2 mb-3">
                <Brain className="h-4 w-4 text-primary" />
                <h3 className="text-sm font-medium">Symbol Library</h3>
                <span className="ml-auto text-[10px] text-muted-foreground">
                  {dreamSymbols.length} unique symbol{dreamSymbols.length !== 1 ? "s" : ""}
                </span>
              </div>
              <div className="space-y-2">
                {dreamSymbols.slice(0, 8).map((sym) => (
                  <div key={sym.id} className="flex items-start gap-2.5">
                    <span className="min-w-[22px] h-5 flex items-center justify-center rounded bg-primary/10 text-primary text-[10px] font-bold shrink-0 mt-0.5">
                      {sym.frequency}×
                    </span>
                    <div className="flex-1 min-w-0">
                      <span className="text-[12px] font-medium capitalize">{sym.symbol}</span>
                      {sym.meaning && (
                        <p className="text-[10px] text-muted-foreground leading-snug mt-0.5 line-clamp-2">{sym.meaning}</p>
                      )}
                    </div>
                    <span className="text-[9px] text-muted-foreground/50 shrink-0 mt-0.5">
                      {new Date(sym.lastSeen).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                    </span>
                  </div>
                ))}
              </div>
              {dreamSymbols.length > 8 && (
                <p className="text-[10px] text-muted-foreground mt-2">
                  +{dreamSymbols.length - 8} more symbols
                </p>
              )}
            </Card>
          )}

          {/* Nightmare Recurrence + IRT Effectiveness */}
          {nightmareRecurrence && (
            <SectionErrorBoundary label="Nightmare Recurrence"><NightmareRecurrenceCard data={nightmareRecurrence} /></SectionErrorBoundary>
          )}

          {/* Lucidity Predictor — tonight's lucid dream potential score */}
          <SectionErrorBoundary label="Lucidity Predictor"><LucidityPredictorCard userId={userId} /></SectionErrorBoundary>

          {/* Dream Recall Heatmap — 28-day calendar of recording consistency */}
          <SectionErrorBoundary label="Recall Heatmap"><DreamRecallHeatmap userId={userId} /></SectionErrorBoundary>

          {/* Presleep Intention — set tonight's intention, track alignment with dreams */}
          <SectionErrorBoundary label="Presleep Intention"><PresleepIntentionCard /></SectionErrorBoundary>

          {/* Emotional Arc Trend — valence of dream narrative arcs over time */}
          <SectionErrorBoundary label="Emotional Arc"><EmotionalArcTrendCard userId={userId} /></SectionErrorBoundary>

          {/* Dream Journal History — searchable/filterable past entries */}
          <SectionErrorBoundary label="Dream Journal"><DreamHistoryCard userId={userId} /></SectionErrorBoundary>

          {/* Personal Dream Dictionary — symbol mood/context across dream history */}
          <SectionErrorBoundary label="Symbol Dictionary"><DreamSymbolContextCard userId={userId} /></SectionErrorBoundary>

          {/* Dream Archetypes — universal patterns across dream history */}
          <SectionErrorBoundary label="Dream Archetypes"><DreamArchetypeCard userId={userId} /></SectionErrorBoundary>

          {/* Sleep Session Wellness Trend */}
          <Card className="glass-card p-5 hover-glow">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-medium">Sleep Session Wellness</h3>
              <span className="ml-auto text-[10px] text-muted-foreground">{periodSessions.length} sessions</span>
            </div>
            {!hasWellnessData ? (
              <div className="h-[220px] flex flex-col items-center justify-center text-sm text-muted-foreground gap-2">
                <Moon className="h-8 w-8 text-muted-foreground/30" />
                <p>No sessions in this period</p>
                <p className="text-xs text-muted-foreground/60">Log dreams manually or connect your EEG headband for automatic detection</p>
              </div>
            ) : (
              <>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={wellnessChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 22%)" opacity={0.6} />
                    <XAxis dataKey="date" tick={{ fontSize: 9, fill: "hsl(220, 12%, 52%)" }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                    <YAxis hide domain={[0, 100]} />
                    <Tooltip
                      cursor={{ stroke: "hsl(220, 12%, 55%)", strokeWidth: 1, strokeDasharray: "4 4" }}
                      contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 11 }}
                      labelStyle={{ color: "var(--muted-foreground)", marginBottom: 4, fontSize: 10 }}
                      itemStyle={{ padding: "1px 0" }}
                      formatter={(value: number) => [`${value}%`]}
                    />
                    <Line type="monotone" dataKey="flow" name="Flow" stroke="hsl(152, 60%, 48%)" strokeWidth={2.5} dot={false} activeDot={{ r: 4, fill: "hsl(152, 60%, 48%)" }} />
                    <Line type="monotone" dataKey="relaxation" name="Relaxation" stroke="hsl(200, 70%, 55%)" strokeWidth={2} dot={false} activeDot={{ r: 4, fill: "hsl(200, 70%, 55%)" }} />
                    <Line type="monotone" dataKey="focus" name="Focus" stroke="hsl(262, 45%, 65%)" strokeWidth={2.5} strokeDasharray="4 3" dot={false} activeDot={{ r: 4, fill: "hsl(262, 45%, 65%)" }} />
                  </LineChart>
                </ResponsiveContainer>
                <div className="flex gap-4 mt-2 flex-wrap">
                  {[
                    { label: "Flow", color: "hsl(152, 60%, 48%)" },
                    { label: "Relaxation", color: "hsl(200, 70%, 55%)" },
                    { label: "Focus", color: "hsl(262, 45%, 65%)", dashed: true },
                  ].map((l) => (
                    <div key={l.label} className="flex items-center gap-1.5">
                      <svg width="18" height="8">
                        <line x1="0" y1="4" x2="18" y2="4" stroke={l.color} strokeWidth="2" strokeDasharray={l.dashed ? "4 3" : "0"} />
                      </svg>
                      <span className="text-[10px] text-muted-foreground">{l.label}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </Card>

          {/* Session list for the period */}
          {periodSessions.length > 0 && (
            <Card className="glass-card p-5 hover-glow">
              <div className="flex items-center gap-2 mb-4">
                <Brain className="h-4 w-4 text-secondary" />
                <h3 className="text-sm font-medium">Sessions in Period</h3>
              </div>
              <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                {periodSessions.slice(0, 20).map((s) => {
                  const t = new Date((s.start_time ?? 0) * 1000);
                  const dur = s.summary?.duration_sec
                    ? `${Math.floor(s.summary.duration_sec / 60)}m ${Math.round(s.summary.duration_sec % 60)}s`
                    : "—";
                  return (
                    <div key={s.session_id} className="flex items-center justify-between py-2 border-b border-border/40 last:border-0">
                      <div>
                        <p className="text-xs font-medium">
                          {t.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" })}
                          {" · "}
                          {t.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })}
                        </p>
                        <p className="text-[10px] text-muted-foreground">{dur}</p>
                      </div>
                      <div className="flex gap-2">
                        {s.summary?.avg_flow != null && (
                          <span className="text-[10px] px-2 py-0.5 rounded-full bg-success/10 text-success">
                            Flow {Math.round(s.summary.avg_flow * 100)}%
                          </span>
                        )}
                        {s.summary?.dominant_emotion && (
                          <span className="text-[10px] px-2 py-0.5 rounded-full bg-muted text-muted-foreground capitalize">
                            {s.summary.dominant_emotion}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </Card>
          )}

          {/* Live charts note */}
          <div className="p-4 rounded-xl border border-border/40 bg-muted/20 text-sm text-muted-foreground flex items-center gap-3">
            <Brain className="h-4 w-4 shrink-0" />
            Real-time hypnogram, dream activity, and REM data are available in the <button onClick={() => setPeriodDays(1)} className="text-primary underline underline-offset-2">Today</button> view while streaming.
          </div>
        </>
      )}
    </main>
  );
}

/* Throttled pattern insight card */
function PatternInsightCard({ sessionLen, dreamFrames, avgRem, remCycles, frameTs }: {
  sessionLen: number; dreamFrames: number; avgRem: number; remCycles: number; frameTs?: number;
}) {
  const [text, setText] = useState("");
  const timerRef = useRef(0);

  useEffect(() => {
    const now = Date.now();
    if (now - timerRef.current < 10_000 && text) return;
    timerRef.current = now;
    const cycleText = remCycles > 0
      ? ` ${remCycles} complete REM cycle${remCycles > 1 ? "s" : ""} detected. Later cycles tend to show higher intensity and lucidity.`
      : " No complete REM cycles detected yet.";
    setText(`This session captured ${sessionLen} data points with ${dreamFrames} dream-active frames and an average REM likelihood of ${avgRem}%.${cycleText}`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frameTs]);

  return (
    <div className="ai-insight-card">
      <div className="flex items-start gap-3">
        <Sparkles className="h-5 w-5 text-primary mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-medium text-foreground mb-1">Pattern Analysis</p>
          <p className="text-sm text-muted-foreground leading-relaxed">{text}</p>
        </div>
      </div>
    </div>
  );
}
