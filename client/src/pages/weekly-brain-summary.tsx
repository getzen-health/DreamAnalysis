/**
 * Weekly Brain Summary — shareable card showing this week vs last week
 * for stress, focus, and sleep.  Pulls from health metrics, voice analyses,
 * food logs, and dream entries.  PNG export via Canvas 2D API (no extra deps).
 */

import { useRef, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { getParticipantId } from "@/lib/participant";
import { resolveUrl } from "@/lib/queryClient";
import { getMLApiUrl } from "@/lib/ml-api";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Download,
  Brain,
  Moon,
  Zap,
  AlertCircle,
  Utensils,
  Mic,
} from "lucide-react";

const CURRENT_USER = getParticipantId();

// ── Types ─────────────────────────────────────────────────────────────────────

interface HealthMetric {
  stressLevel?: number | null;
  neuralActivity?: number | null; // 0-100; returned as-is from /api/health-metrics
  sleepQuality?: number | null;
  sleepDuration?: number | null;
  timestamp?: string;
  createdAt?: string;
}

interface VoiceRecord {
  timestamp?: number;
  emotion?: string;
  valence?: number;
  arousal?: number;
  stress_index?: number;
  focus_index?: number;
}

interface FoodLogEntry {
  id: string;
  loggedAt: string;
  mealType: string | null;
  summary: string | null;
  totalCalories: number | null;
  dominantMacro: string | null;
  glycemicImpact: string | null;
}

interface DreamEntry {
  id: string;
  dreamText: string;
  emotions: Array<{ emotion: string; intensity: number }> | null;
  sleepQuality: number | null;
  lucidityScore: number | null;
  timestamp: string;
}

interface WeekStats {
  stress:       number | null;
  focus:        number | null;
  sleep:        number | null;
  sleepHours:   number | null;
  sampleCount:  number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function avg(nums: number[]): number | null {
  const valid = nums.filter(n => !isNaN(n) && n != null);
  if (valid.length === 0) return null;
  return valid.reduce((a, b) => a + b, 0) / valid.length;
}

function computeWeek(metrics: HealthMetric[], daysAgoStart: number, daysAgoEnd: number): WeekStats {
  const now = Date.now();
  const ms = (d: number) => d * 86_400_000;
  const rows = metrics.filter(m => {
    const t = m.timestamp ?? m.createdAt;
    if (!t) return false;
    const age = now - new Date(t).getTime();
    return age >= ms(daysAgoStart) && age < ms(daysAgoEnd);
  });
  return {
    stress:      avg(rows.map(r => r.stressLevel ?? NaN)),
    focus:       avg(rows.map(r => r.neuralActivity ?? NaN)),
    sleep:       avg(rows.map(r => r.sleepQuality ?? NaN)),
    sleepHours:  avg(rows.map(r => r.sleepDuration ?? NaN)),
    sampleCount: rows.length,
  };
}

/** Filter records from the past N days */
function withinDays<T>(items: T[], getTime: (item: T) => number, daysBack: number): T[] {
  const cutoff = Date.now() - daysBack * 86_400_000;
  return items.filter(item => getTime(item) >= cutoff);
}

function delta(now: number | null, prev: number | null): number | null {
  if (now == null || prev == null) return null;
  return now - prev;
}

/** Format a 0-10 integer as "X.X/10". Used for stressLevel and sleepQuality. */
function fmtSlash10(v: number | null): string {
  if (v == null) return "—";
  return v.toFixed(1) + "/10";
}

/** Format a 0-100 integer as "X%". Used for neuralActivity (focus). */
function fmtPct100(v: number | null): string {
  if (v == null) return "—";
  return v.toFixed(0) + "%";
}

function fmtNum(v: number | null, decimals = 1): string {
  if (v == null) return "—";
  return v.toFixed(decimals);
}

function weekLabel(daysAgoStart: number): string {
  const start = new Date(Date.now() - daysAgoStart * 86_400_000);
  const end   = new Date(Date.now() - (daysAgoStart > 0 ? (daysAgoStart - 7) * 86_400_000 : 0));
  const fmt = (d: Date) =>
    d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  return `${fmt(start)} – ${fmt(end)}`;
}

// ── Trend component ───────────────────────────────────────────────────────────

function Trend({
  d,
  positiveIsGood = true,
}: {
  d: number | null;
  positiveIsGood?: boolean;
}) {
  if (d == null || Math.abs(d) < 0.01) {
    return <Minus className="h-4 w-4 text-muted-foreground" />;
  }
  const isPositive = d > 0;
  const isGood = positiveIsGood ? isPositive : !isPositive;
  const Icon = isPositive ? TrendingUp : TrendingDown;
  return (
    <span className={`flex items-center gap-0.5 text-xs font-medium ${isGood ? "text-cyan-400" : "text-rose-400"}`}>
      <Icon className="h-4 w-4" />
      {Math.abs(d * 100).toFixed(0)}%
    </span>
  );
}

// ── PNG Export ────────────────────────────────────────────────────────────────

function exportAsPng(
  thisWeek: WeekStats,
  lastWeek: WeekStats,
  weekRange: string,
) {
  const W = 800, H = 450;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // Background
  ctx.fillStyle = "#0d0f14";
  ctx.fillRect(0, 0, W, H);

  // Subtle grid lines
  ctx.strokeStyle = "#1e2130";
  ctx.lineWidth = 1;
  for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
  for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

  // Title
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 28px system-ui, sans-serif";
  ctx.fillText("My Brain Week", 40, 60);

  ctx.fillStyle = "#6b7280";
  ctx.font = "14px system-ui, sans-serif";
  ctx.fillText(weekRange, 40, 82);

  // Accent bar
  const grad = ctx.createLinearGradient(0, 100, W, 100);
  grad.addColorStop(0, "#6366f1");
  grad.addColorStop(1, "#8b5cf6");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 100, W, 2);

  // 3 metric cards
  const cards = [
    {
      label: "Stress",
      thisVal: fmtSlash10(thisWeek.stress),
      d: delta(thisWeek.stress, lastWeek.stress),
      positiveIsGood: false,
      color: "#e879a8",
    },
    {
      label: "Focus",
      thisVal: fmtPct100(thisWeek.focus),
      d: delta(thisWeek.focus, lastWeek.focus),
      positiveIsGood: true,
      color: "#3b82f6",
    },
    {
      label: "Sleep",
      thisVal: thisWeek.sleepHours != null ? thisWeek.sleepHours.toFixed(1) + "h" : fmtSlash10(thisWeek.sleep),
      d: delta(thisWeek.sleep, lastWeek.sleep),
      positiveIsGood: true,
      color: "#7c3aed",
    },
  ];

  const cardW = 220, cardH = 130, cardY = 130, gap = 40;
  const totalW = cards.length * cardW + (cards.length - 1) * gap;
  const startX = (W - totalW) / 2;

  cards.forEach((card, i) => {
    const x = startX + i * (cardW + gap);

    // Card bg
    ctx.fillStyle = "#141720";
    ctx.beginPath();
    (ctx as CanvasRenderingContext2D).roundRect?.(x, cardY, cardW, cardH, 12);
    ctx.fill();

    // Accent border top
    ctx.fillStyle = card.color;
    ctx.fillRect(x, cardY, cardW, 3);

    // Label
    ctx.fillStyle = "#9ca3af";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText(card.label.toUpperCase(), x + 20, cardY + 30);

    // Value
    ctx.fillStyle = card.color;
    ctx.font = "bold 36px system-ui, sans-serif";
    ctx.fillText(card.thisVal, x + 20, cardY + 78);

    // Delta vs last week
    if (card.d != null) {
      const dPct = (Math.abs(card.d) * 100).toFixed(0) + "%";
      const isGood = card.positiveIsGood ? card.d > 0 : card.d < 0;
      const arrow = card.d > 0 ? "↑" : "↓";
      ctx.fillStyle = isGood ? "#0891b2" : "#e879a8";
      ctx.font = "13px system-ui, sans-serif";
      ctx.fillText(`${arrow} ${dPct} vs last week`, x + 20, cardY + 108);
    } else {
      ctx.fillStyle = "#4b5563";
      ctx.font = "13px system-ui, sans-serif";
      ctx.fillText("no prior data", x + 20, cardY + 108);
    }
  });

  // Footer
  ctx.fillStyle = "#374151";
  ctx.fillRect(0, H - 50, W, 1);

  ctx.fillStyle = "#6b7280";
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText("AntarAI", 40, H - 18);

  ctx.fillStyle = "#4b5563";
  ctx.font = "12px system-ui, sans-serif";
  const sampleText = `${thisWeek.sampleCount} data points this week`;
  const tw = ctx.measureText(sampleText).width;
  ctx.fillText(sampleText, W - tw - 40, H - 18);

  // Download
  const link = document.createElement("a");
  link.download = `brain-week-${new Date().toISOString().slice(0, 10)}.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function WeeklyBrainSummary() {
  const exportRef = useRef<HTMLDivElement>(null);

  // ── Data source 1: Health metrics (stress, focus, sleep) ────────────────────
  const { data: metrics = [], isLoading: loadingHealth } = useQuery<HealthMetric[]>({
    queryKey: ["/api/health-metrics", CURRENT_USER],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/health-metrics/${CURRENT_USER}`));
      if (!res.ok) return [];
      return res.json();
    },
    staleTime: 5 * 60 * 1000,
  });

  // ── Data source 2: Voice analysis history ───────────────────────────────────
  const { data: voiceHistory = [], isLoading: loadingVoice } = useQuery<VoiceRecord[]>({
    queryKey: ["voice-history-weekly", CURRENT_USER],
    queryFn: async () => {
      try {
        const res = await fetch(`${getMLApiUrl()}/api/voice-watch/history/${CURRENT_USER}?last_n=100`);
        if (!res.ok) return [];
        const data = await res.json();
        return (data?.history ?? []) as VoiceRecord[];
      } catch {
        return [];
      }
    },
    staleTime: 5 * 60 * 1000,
    retry: false,
  });

  // ── Data source 3: Food logs ────────────────────────────────────────────────
  const { data: foodLogs = [], isLoading: loadingFood } = useQuery<FoodLogEntry[]>({
    queryKey: ["/api/food/logs", CURRENT_USER],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/food/logs/${CURRENT_USER}`));
      if (!res.ok) return [];
      return res.json();
    },
    staleTime: 5 * 60 * 1000,
  });

  // ── Data source 4: Dream entries ────────────────────────────────────────────
  const { data: dreamEntries = [], isLoading: loadingDreams } = useQuery<DreamEntry[]>({
    queryKey: ["/api/dream-analysis", CURRENT_USER],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/dream-analysis/${CURRENT_USER}`));
      if (!res.ok) return [];
      return res.json();
    },
    staleTime: 5 * 60 * 1000,
  });

  const isLoading = loadingHealth || loadingVoice || loadingFood || loadingDreams;

  const thisWeek = computeWeek(metrics, 0, 7);
  const lastWeek = computeWeek(metrics, 7, 14);
  const weekRange = weekLabel(7);

  const stressDelta = delta(thisWeek.stress, lastWeek.stress);
  const focusDelta  = delta(thisWeek.focus,  lastWeek.focus);
  const sleepDelta  = delta(thisWeek.sleep,  lastWeek.sleep);

  // ── Voice analysis weekly stats ─────────────────────────────────────────────
  const voiceThisWeek = useMemo(() => {
    const cutoff7 = Date.now() - 7 * 86_400_000;
    const recent = voiceHistory.filter(v => (v.timestamp ?? 0) * 1000 >= cutoff7);
    if (recent.length === 0) return null;

    const emotions = recent.map(v => v.emotion).filter(Boolean) as string[];
    const emotionCounts: Record<string, number> = {};
    emotions.forEach(e => { emotionCounts[e] = (emotionCounts[e] ?? 0) + 1; });
    const dominant = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "neutral";

    return {
      count: recent.length,
      avgValence: avg(recent.map(v => v.valence ?? NaN)),
      avgArousal: avg(recent.map(v => v.arousal ?? NaN)),
      avgStress: avg(recent.map(v => v.stress_index ?? NaN)),
      avgFocus: avg(recent.map(v => v.focus_index ?? NaN)),
      dominantEmotion: dominant,
      emotionCounts,
    };
  }, [voiceHistory]);

  // ── Food log weekly stats ───────────────────────────────────────────────────
  const foodThisWeek = useMemo(() => {
    const recent = withinDays(foodLogs, f => new Date(f.loggedAt).getTime(), 7);
    if (recent.length === 0) return null;

    const mealCounts: Record<string, number> = {};
    recent.forEach(f => { mealCounts[f.mealType ?? "other"] = (mealCounts[f.mealType ?? "other"] ?? 0) + 1; });

    const cals = recent.map(f => f.totalCalories).filter((c): c is number => c != null && c > 0);
    const giCounts: Record<string, number> = {};
    recent.forEach(f => { if (f.glycemicImpact) giCounts[f.glycemicImpact] = (giCounts[f.glycemicImpact] ?? 0) + 1; });

    return {
      count: recent.length,
      avgCalories: cals.length > 0 ? Math.round(cals.reduce((a, b) => a + b, 0) / cals.length) : null,
      mealCounts,
      topGI: Object.entries(giCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? null,
    };
  }, [foodLogs]);

  // ── Dream weekly stats ──────────────────────────────────────────────────────
  const dreamsThisWeek = useMemo(() => {
    const recent = withinDays(dreamEntries, d => new Date(d.timestamp).getTime(), 7);
    if (recent.length === 0) return null;

    const sleepQuals = recent.map(d => d.sleepQuality).filter((q): q is number => q != null);
    const lucidity = recent.map(d => d.lucidityScore).filter((l): l is number => l != null);

    const allEmotions: string[] = [];
    recent.forEach(d => {
      if (Array.isArray(d.emotions)) {
        d.emotions.forEach((e: { emotion: string }) => allEmotions.push(e.emotion));
      }
    });

    return {
      count: recent.length,
      avgSleepQuality: sleepQuals.length > 0 ? Math.round((sleepQuals.reduce((a, b) => a + b, 0) / sleepQuals.length) * 10) / 10 : null,
      avgLucidity: lucidity.length > 0 ? Math.round(lucidity.reduce((a, b) => a + b, 0) / lucidity.length) : null,
      dreamEmotions: allEmotions.slice(0, 5),
    };
  }, [dreamEntries]);

  // Combined "has any data" check across all sources
  const hasData = thisWeek.sampleCount > 0;
  const hasAnyData = hasData || !!voiceThisWeek || !!foodThisWeek || !!dreamsThisWeek;
  const totalDataPoints = (thisWeek.sampleCount) + (voiceThisWeek?.count ?? 0) + (foodThisWeek?.count ?? 0) + (dreamsThisWeek?.count ?? 0);

  // ─── Metric row ─────────────────────────────────────────────────────────────

  function MetricCard({
    icon: Icon,
    label,
    thisVal,
    lastVal,
    d,
    positiveIsGood = true,
    color,
  }: {
    icon: React.ElementType;
    label: string;
    thisVal: string;
    lastVal: string;
    d: number | null;
    positiveIsGood?: boolean;
    color: string;
  }) {
    return (
      <Card className="glass-card p-5 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Icon className="h-4 w-4" style={{ color }} />
            <span className="text-sm font-medium">{label}</span>
          </div>
          <Trend d={d} positiveIsGood={positiveIsGood} />
        </div>

        <div className="flex items-end justify-between">
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">This week</p>
            <p className="text-3xl font-mono font-bold" style={{ color }}>{thisVal}</p>
          </div>
          <div className="text-right">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Last week</p>
            <p className="text-xl font-mono text-muted-foreground">{lastVal}</p>
          </div>
        </div>

        {/* Micro progress bars — comparison */}
        {d != null && (
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-muted-foreground w-16 shrink-0">This week</span>
              <div className="flex-1 bg-muted/20 rounded-full h-1.5 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700"
                  style={{
                    width: thisWeek.sampleCount > 0 ? `${Math.min(100, Math.abs(parseFloat(thisVal)) || 50)}%` : "0%",
                    background: color,
                  }}
                />
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-muted-foreground w-16 shrink-0">Last week</span>
              <div className="flex-1 bg-muted/20 rounded-full h-1.5 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700"
                  style={{
                    width: lastWeek.sampleCount > 0 ? `${Math.min(100, Math.abs(parseFloat(lastVal)) || 50)}%` : "0%",
                    background: color + "80",
                  }}
                />
              </div>
            </div>
          </div>
        )}
      </Card>
    );
  }

  return (
    <main className="p-4 md:p-6 space-y-6 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Brain className="h-6 w-6 text-primary" />
          <div>
            <h2 className="text-xl font-semibold">Weekly Brain Summary</h2>
            <p className="text-xs text-muted-foreground">{weekRange}</p>
          </div>
        </div>
        <Button
          onClick={() => exportAsPng(thisWeek, lastWeek, weekRange)}
          disabled={!hasAnyData}
          className="bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30"
          size="sm"
        >
          <Download className="h-4 w-4 mr-2" />
          Export PNG
        </Button>
      </div>

      {/* Loading skeleton */}
      {isLoading && (
        <div className="space-y-4">
          <Skeleton className="h-32 w-full rounded-2xl" />
          <div className="grid grid-cols-3 gap-3">
            <Skeleton className="h-28 rounded-xl" />
            <Skeleton className="h-28 rounded-xl" />
            <Skeleton className="h-28 rounded-xl" />
          </div>
          <Skeleton className="h-20 w-full rounded-xl" />
        </div>
      )}

      {/* No data state */}
      {!isLoading && !hasAnyData && (
        <Card className="glass-card p-6 flex items-center gap-4">
          <AlertCircle className="h-5 w-5 text-muted-foreground shrink-0" />
          <div>
            <p className="text-sm font-medium">No data yet for this week</p>
            <p className="text-xs text-muted-foreground mt-0.5">
              Your weekly summary builds from voice analyses, food logs, dream entries, and health signals.
              Do a voice analysis or log a meal to get started.
            </p>
          </div>
        </Card>
      )}

      {/* Preview card — styled to match exported PNG */}
      {hasAnyData && (
        <div ref={exportRef}>
          {/* Summary hero */}
          <Card className="glass-card p-6 rounded-2xl mb-4">
            <div className="flex items-center justify-between mb-5">
              <div>
                <p className="text-[10px] text-muted-foreground uppercase tracking-widest">
                  Week of
                </p>
                <p className="text-base font-semibold">{weekRange}</p>
              </div>
              <div className="text-right">
                <p className="text-[10px] text-muted-foreground uppercase tracking-widest">
                  Data points
                </p>
                <p className="text-base font-mono font-bold text-primary">
                  {totalDataPoints}
                </p>
              </div>
            </div>

            {/* Week-in-one-sentence */}
            <p className="text-sm text-muted-foreground border-l-2 border-primary/40 pl-3">
              {(() => {
                const parts: string[] = [];
                if (stressDelta != null) {
                  parts.push(stressDelta < -0.03
                    ? "Stress improved this week"
                    : stressDelta > 0.03
                      ? "Stress was higher this week"
                      : "Stress stayed steady");
                }
                if (focusDelta != null) {
                  parts.push(focusDelta > 0.03
                    ? "focus was up"
                    : focusDelta < -0.03
                      ? "focus dipped"
                      : "focus held stable");
                }
                if (sleepDelta != null) {
                  parts.push(sleepDelta > 0.05
                    ? "sleep quality improved"
                    : sleepDelta < -0.05
                      ? "sleep quality dropped"
                      : "sleep was consistent");
                }
                // Voice mood summary
                if (voiceThisWeek && voiceThisWeek.avgValence != null) {
                  parts.push(voiceThisWeek.avgValence > 0.1
                    ? `mood trending positive (${voiceThisWeek.dominantEmotion})`
                    : voiceThisWeek.avgValence < -0.1
                      ? `mood leaning low (${voiceThisWeek.dominantEmotion})`
                      : `mood was neutral overall`);
                }
                return parts.length > 0
                  ? parts.join(", ") + "."
                  : "Your first analyses are recorded. Keep going to build trends.";
              })()}
            </p>
          </Card>

          {/* Health metric cards — only if health data exists */}
          {hasData && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
              <MetricCard
                icon={AlertCircle}
                label="Stress"
                thisVal={fmtSlash10(thisWeek.stress)}
                lastVal={fmtSlash10(lastWeek.stress)}
                d={stressDelta}
                positiveIsGood={false}
                color="hsl(0, 70%, 65%)"
              />
              <MetricCard
                icon={Zap}
                label="Focus"
                thisVal={fmtPct100(thisWeek.focus)}
                lastVal={fmtPct100(lastWeek.focus)}
                d={focusDelta}
                positiveIsGood={true}
                color="hsl(210, 80%, 65%)"
              />
              <MetricCard
                icon={Moon}
                label="Sleep"
                thisVal={
                  thisWeek.sleepHours != null
                    ? `${fmtNum(thisWeek.sleepHours)}h`
                    : fmtSlash10(thisWeek.sleep)
                }
                lastVal={
                  lastWeek.sleepHours != null
                    ? `${fmtNum(lastWeek.sleepHours)}h`
                    : fmtSlash10(lastWeek.sleep)
                }
                d={sleepDelta}
                positiveIsGood={true}
                color="hsl(270, 70%, 65%)"
              />
            </div>
          )}

          {/* Voice analysis mood summary */}
          {voiceThisWeek && (
            <Card className="glass-card p-5 mb-4">
              <div className="flex items-center gap-2 mb-3">
                <Mic className="h-4 w-4 text-cyan-400" />
                <span className="text-sm font-medium">Voice Analyses</span>
                <span className="ml-auto text-xs text-muted-foreground">{voiceThisWeek.count} this week</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Mood</p>
                  <p className="text-lg font-medium capitalize" style={{ color: (voiceThisWeek.avgValence ?? 0) > 0 ? "hsl(152, 60%, 48%)" : (voiceThisWeek.avgValence ?? 0) < -0.1 ? "hsl(0, 70%, 65%)" : "hsl(210, 20%, 60%)" }}>
                    {voiceThisWeek.dominantEmotion}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Valence</p>
                  <p className="text-lg font-mono font-bold text-indigo-400">
                    {voiceThisWeek.avgValence != null ? (voiceThisWeek.avgValence > 0 ? "+" : "") + voiceThisWeek.avgValence.toFixed(2) : "--"}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Stress</p>
                  <p className="text-lg font-mono font-bold text-rose-400">
                    {voiceThisWeek.avgStress != null ? (voiceThisWeek.avgStress * 100).toFixed(0) + "%" : "--"}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Focus</p>
                  <p className="text-lg font-mono font-bold text-violet-400">
                    {voiceThisWeek.avgFocus != null ? (voiceThisWeek.avgFocus * 100).toFixed(0) + "%" : "--"}
                  </p>
                </div>
              </div>
              {/* Emotion breakdown */}
              {Object.keys(voiceThisWeek.emotionCounts).length > 1 && (
                <div className="mt-3 pt-3 border-t border-border/30 flex flex-wrap gap-1.5">
                  {Object.entries(voiceThisWeek.emotionCounts)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 6)
                    .map(([emotion, count]) => (
                      <span key={emotion} className="text-[10px] px-2 py-0.5 rounded-full bg-muted/40 text-muted-foreground capitalize">
                        {emotion} ({count})
                      </span>
                    ))}
                </div>
              )}
            </Card>
          )}

          {/* Food log summary */}
          {foodThisWeek && (
            <Card className="glass-card p-5 mb-4">
              <div className="flex items-center gap-2 mb-3">
                <Utensils className="h-4 w-4 text-amber-400" />
                <span className="text-sm font-medium">Food Log</span>
                <span className="ml-auto text-xs text-muted-foreground">{foodThisWeek.count} meals logged</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {foodThisWeek.avgCalories != null && (
                  <div>
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Avg Calories</p>
                    <p className="text-lg font-mono font-bold text-amber-400">{foodThisWeek.avgCalories}</p>
                  </div>
                )}
                <div>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Meals</p>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {Object.entries(foodThisWeek.mealCounts).map(([type, count]) => (
                      <span key={type} className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-300 capitalize">
                        {type} ({count})
                      </span>
                    ))}
                  </div>
                </div>
                {foodThisWeek.topGI && (
                  <div>
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Typical GI</p>
                    <p className="text-lg font-medium capitalize" style={{ color: foodThisWeek.topGI === "low" ? "hsl(152, 60%, 48%)" : foodThisWeek.topGI === "high" ? "hsl(0, 70%, 65%)" : "hsl(38, 85%, 58%)" }}>
                      {foodThisWeek.topGI}
                    </p>
                  </div>
                )}
              </div>
            </Card>
          )}

          {/* Dream summary */}
          {dreamsThisWeek && (
            <Card className="glass-card p-5 mb-4">
              <div className="flex items-center gap-2 mb-3">
                <Moon className="h-4 w-4 text-indigo-400" />
                <span className="text-sm font-medium">Dreams</span>
                <span className="ml-auto text-xs text-muted-foreground">{dreamsThisWeek.count} recorded</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {dreamsThisWeek.avgSleepQuality != null && (
                  <div>
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Sleep Quality</p>
                    <p className="text-lg font-mono font-bold text-indigo-400">{dreamsThisWeek.avgSleepQuality}/10</p>
                  </div>
                )}
                {dreamsThisWeek.avgLucidity != null && (
                  <div>
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Avg Lucidity</p>
                    <p className="text-lg font-mono font-bold text-purple-400">{dreamsThisWeek.avgLucidity}%</p>
                  </div>
                )}
                {dreamsThisWeek.dreamEmotions.length > 0 && (
                  <div>
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Dream Emotions</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {dreamsThisWeek.dreamEmotions.map((e, i) => (
                        <span key={i} className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-300 capitalize">
                          {e}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </Card>
          )}
        </div>
      )}

      {/* Last week comparison row (summary only) */}
      {hasData && lastWeek.sampleCount > 0 && (
        <Card className="glass-card p-4">
          <p className="text-xs text-muted-foreground mb-3">
            Last week ({lastWeek.sampleCount} data points)
          </p>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Stress</p>
              <p className="text-lg font-mono font-bold text-rose-400">{fmtSlash10(lastWeek.stress)}</p>
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Focus</p>
              <p className="text-lg font-mono font-bold text-indigo-400">{fmtPct100(lastWeek.focus)}</p>
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Sleep</p>
              <p className="text-lg font-mono font-bold text-violet-400">
                {lastWeek.sleepHours != null
                  ? `${fmtNum(lastWeek.sleepHours)}h`
                  : fmtSlash10(lastWeek.sleep)}
              </p>
            </div>
          </div>
        </Card>
      )}

      {/* Export hint */}
      {hasAnyData && (
        <p className="text-center text-xs text-muted-foreground">
          Tap "Export PNG" to download a shareable image of your weekly summary.
        </p>
      )}
    </main>
  );
}
