import { useState, useEffect, useRef, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChartTooltip } from "@/components/chart-tooltip";
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
import { Moon, TrendingUp, Brain, Activity, Sparkles, Radio } from "lucide-react";
import { useDevice } from "@/hooks/use-device";
import { listSessions, type SessionSummary } from "@/lib/ml-api";

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

  // Sessions query
  const { data: allSessions = [] } = useQuery<SessionSummary[]>({
    queryKey: ["sessions"],
    queryFn: () => listSessions(),
    retry: false,
    staleTime: 2 * 60 * 1000,
    refetchInterval: 60_000,
  });

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
    contentStyle: { background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 10, fontSize: 11 },
    labelStyle: { color: "hsl(220, 12%, 65%)", marginBottom: 4, fontSize: 10 },
    itemStyle: { padding: "1px 0" },
  };

  return (
    <main className="p-6 space-y-6 max-w-5xl">
      {/* Connection Banner */}
      {!isStreaming && isLiveToday && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          Connect your Muse 2 from the sidebar to see live dream pattern data.
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
                      contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 10, fontSize: 11 }}
                      labelStyle={{ color: "hsl(220, 12%, 65%)", marginBottom: 4, fontSize: 10 }}
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
                <p className="text-xs text-muted-foreground/60">Connect your Muse 2 to start recording</p>
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
                      contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 10, fontSize: 11 }}
                      labelStyle={{ color: "hsl(220, 12%, 65%)", marginBottom: 4, fontSize: 10 }}
                      itemStyle={{ padding: "1px 0" }}
                      formatter={(value: number) => [`${value}%`]}
                    />
                    <Line type="monotone" dataKey="flow" name="Flow" stroke="hsl(152, 60%, 48%)" strokeWidth={2.5} dot={false} activeDot={{ r: 4, fill: "hsl(152, 60%, 48%)" }} />
                    <Line type="monotone" dataKey="relaxation" name="Relaxation" stroke="hsl(200, 70%, 55%)" strokeWidth={2} dot={false} activeDot={{ r: 4, fill: "hsl(200, 70%, 55%)" }} />
                    <Line type="monotone" dataKey="focus" name="Focus" stroke="hsl(262, 45%, 65%)" strokeWidth={1.5} strokeDasharray="4 3" dot={false} activeDot={{ r: 4, fill: "hsl(262, 45%, 65%)" }} />
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
