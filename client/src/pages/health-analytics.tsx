import { useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChartTooltip } from "@/components/chart-tooltip";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScoreCircle } from "@/components/score-circle";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Brain,
  Sparkles,
  TrendingUp,
  Radio,
} from "lucide-react";
import { useDevice } from "@/hooks/use-device";
import { useVoiceEmotion } from "@/hooks/use-voice-emotion";
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
interface HealthPoint {
  time: string;
  stress: number;
  focus: number;
  relaxation: number;
  cogLoad: number;
  flow: number;
}

/** Per-frame live band-power point (updates every ~1.5 s) */
interface LiveBandPoint {
  time: string;
  calm: number;    // alpha % of total
  alert: number;   // beta % of total
  creative: number;// theta % of total
}

interface SessionPoint {
  date: string;
  stress: number;
  focus: number;
  relaxation: number;
  flow: number;
}

/* ---------- helpers ---------- */
function avgNums(arr: number[]): number {
  return arr.length ? Math.round(arr.reduce((a, b) => a + b, 0) / arr.length) : 0;
}

function buildHealthChartData(sessions: SessionSummary[], days: number): SessionPoint[] {
  const map: Record<
    string,
    { stress: number[]; focus: number[]; relaxation: number[]; flow: number[]; ts: number }
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

    if (!map[key]) map[key] = { stress: [], focus: [], relaxation: [], flow: [], ts };
    map[key].stress.push((s.summary.avg_stress ?? 0) * 100);
    map[key].focus.push((s.summary.avg_focus ?? 0) * 100);
    map[key].relaxation.push((s.summary.avg_relaxation ?? 0) * 100);
    map[key].flow.push((s.summary.avg_flow ?? 0) * 100);
  }

  return Object.entries(map)
    .sort(([, a], [, b]) => a.ts - b.ts)
    .map(([date, data]) => ({
      date,
      stress: avgNums(data.stress),
      focus: avgNums(data.focus),
      relaxation: avgNums(data.relaxation),
      flow: avgNums(data.flow),
    }));
}

/* ========== Component ========== */
export default function HealthAnalytics() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;
  const { lastResult: voiceResult } = useVoiceEmotion();

  // Extract live metrics from ML models
  const emotions = analysis?.emotions;
  const stress = analysis?.stress;
  const attention = analysis?.attention;
  const cognitiveLoad = analysis?.cognitive_load;
  const flowState = analysis?.flow_state;
  const creativity = analysis?.creativity;
  const drowsiness = analysis?.drowsiness;
  const memoryEncoding = analysis?.memory_encoding;

  // Voice-derived fallbacks when EEG is not streaming
  const voiceStress = voiceResult ? Math.round((voiceResult.stress_from_watch ?? Math.max(0, voiceResult.arousal - voiceResult.valence + 0.5) / 1.5) * 100) : null;
  const voiceFocus = voiceResult ? Math.round(((voiceResult.arousal * 0.6 + (voiceResult.valence + 1) / 2 * 0.4)) * 100) : null;
  const voiceRelax = voiceResult ? Math.round(((1 - voiceResult.arousal) * 0.5 + (voiceResult.valence + 1) / 2 * 0.5) * 100) : null;

  // True when we have any real measurement (EEG or voice)
  const hasRealData = isStreaming || voiceResult !== null;

  // Current metrics — EEG when streaming, voice estimates when available, null otherwise
  const stressIndex: number | null = isStreaming
    ? Math.round((stress?.stress_index ?? emotions?.stress_index ?? 0) * 100)
    : voiceStress;
  const focusScore: number | null = isStreaming
    ? Math.round((attention?.attention_score ?? emotions?.focus_index ?? 0) * 100)
    : voiceFocus;
  const relaxScore: number | null = isStreaming
    ? Math.round((emotions?.relaxation_index ?? 0) * 100)
    : voiceRelax;
  const cogLoadIndex = isStreaming ? Math.round((cognitiveLoad?.load_index ?? 0) * 100) : 0;
  const flowScore = isStreaming ? Math.round((flowState?.flow_score ?? 0) * 100) : 0;
  const creativityScore = isStreaming ? Math.round((creativity?.creativity_score ?? 0) * 100) : 0;
  const drowsinessIndex = isStreaming ? Math.round((drowsiness?.drowsiness_index ?? 0) * 100) : 0;
  const memoryScore = isStreaming ? Math.round((memoryEncoding?.encoding_score ?? 0) * 100) : 0;

  // Composite scores — null when any required input is missing
  const brainHealthScore: number | null =
    focusScore !== null && relaxScore !== null && stressIndex !== null
      ? Math.round(focusScore * 0.25 + relaxScore * 0.25 + (100 - stressIndex) * 0.25 + (isStreaming ? flowScore : relaxScore) * 0.25)
      : null;
  const cognitiveScore: number | null =
    focusScore !== null
      ? isStreaming
        ? Math.round(focusScore * 0.3 + creativityScore * 0.25 + memoryScore * 0.25 + (100 - drowsinessIndex) * 0.2)
        : relaxScore !== null ? Math.round(focusScore * 0.6 + relaxScore * 0.4) : null
      : null;
  const wellbeingScore: number | null =
    relaxScore !== null && stressIndex !== null
      ? Math.round(relaxScore * 0.35 + (100 - stressIndex) * 0.35 + (isStreaming ? flowScore : relaxScore) * 0.3)
      : null;

  // Period selector
  const [periodDays, setPeriodDays] = useState(1);
  const isLiveToday = periodDays === 1;

  // Sessions query for historical data
  const { data: allSessions = [] } = useQuery<SessionSummary[]>({
    queryKey: ["sessions"],
    queryFn: () => listSessions(),
    retry: false,
    staleTime: 2 * 60 * 1000,
    refetchInterval: 60_000,
  });

  // Accumulate live timeline (Today) — 15-s emotion window for composite scores
  const [timeline, setTimeline] = useState<HealthPoint[]>([]);

  useEffect(() => {
    if (!isStreaming || !analysis) return;
    const now = new Date().toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    setTimeline((prev) => [
      ...prev.slice(-60),
      { time: now, stress: stressIndex ?? 0, focus: focusScore ?? 0, relaxation: relaxScore ?? 0, cogLoad: cogLoadIndex, flow: flowScore },
    ]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  // Per-frame live band-power timeline (updates every ~1.5 s — no 15-s stale lag)
  const [liveBands, setLiveBands] = useState<LiveBandPoint[]>([]);
  useEffect(() => {
    if (!isStreaming || !analysis?.band_powers) return;
    const bp = analysis.band_powers;
    const total = (bp.delta ?? 0) + (bp.theta ?? 0) + (bp.alpha ?? 0) +
                  (bp.beta ?? 0) + (bp.gamma ?? 0) + 0.001;
    const calm     = Math.round(Math.min(100, (bp.alpha ?? 0) / total * 100));
    const alert    = Math.round(Math.min(100, (bp.beta  ?? 0) / total * 100));
    const creative = Math.round(Math.min(100, (bp.theta ?? 0) / total * 100));
    const now = new Date().toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    setLiveBands((prev) => [...prev.slice(-80), { time: now, calm, alert, creative }]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  // Build chart data
  const cutoff = Date.now() / 1000 - periodDays * 86400;
  const periodSessions = allSessions.filter((s) => (s.start_time ?? 0) >= cutoff);
  const sessionChartData = buildHealthChartData(periodSessions, periodDays);

  const liveNow: HealthPoint | null = isLiveToday && isStreaming
    ? { time: "Now", stress: stressIndex ?? 0, focus: focusScore ?? 0, relaxation: relaxScore ?? 0, cogLoad: cogLoadIndex, flow: flowScore }
    : null;
  const todayChartData: (HealthPoint | SessionPoint)[] = liveNow
    ? [...timeline.slice(-30), liveNow]
    : timeline.slice(-30);

  const chartData = isLiveToday ? todayChartData : sessionChartData;
  const dataKey = isLiveToday ? "time" : "date";
  const hasChartData = chartData.length >= 1;

  // Live insights (throttled to 10s)
  interface Insight { title: string; description: string; strength: number; brain: string; health: string; }
  const [insights, setInsights] = useState<Insight[]>([]);
  const insightTimerRef = useRef(0);

  useEffect(() => {
    if (!isStreaming) { setInsights([]); return; }
    const now = Date.now();
    if (now - insightTimerRef.current < 10_000 && insights.length > 0) return;
    insightTimerRef.current = now;

    const next: Insight[] = [];
    if (flowScore > 60) next.push({ title: "Flow State Active", description: `Flow score ${flowScore}%. Optimal for deep work. Minimize interruptions.`, strength: flowScore / 100, brain: "flow_score", health: "focus" });
    if ((stressIndex ?? 0) > 50) next.push({ title: "Elevated Stress", description: `Stress at ${stressIndex ?? 0}%. Consider a breathing exercise or short break.`, strength: (stressIndex ?? 0) / 100, brain: "stress_index", health: "relaxation" });
    if (creativityScore > 50) next.push({ title: "Creative State", description: `Creativity at ${creativityScore}%. Theta-alpha ratio suggests heightened divergent thinking.`, strength: creativityScore / 100, brain: "creativity", health: "cognitive" });
    if (memoryScore > 60) next.push({ title: "Strong Memory Encoding", description: `Memory encoding at ${memoryScore}%. Brain actively consolidating — great time to learn.`, strength: memoryScore / 100, brain: "memory", health: "encoding" });
    if (drowsinessIndex > 60) next.push({ title: "Drowsiness Alert", description: `Drowsiness at ${drowsinessIndex}%. Consider a break or movement to restore alertness.`, strength: drowsinessIndex / 100, brain: "drowsiness", health: "alertness" });
    if (next.length === 0) next.push({ title: "Balanced Brain State", description: "All metrics within normal ranges. Neural patterns balanced.", strength: 0.5, brain: "overall", health: "balance" });

    setInsights(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  return (
    <main className="p-6 space-y-6 max-w-5xl">
      {/* Connection Banner */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-primary/20 bg-primary/5 text-sm text-muted-foreground flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0 text-primary" />
          {voiceResult
            ? "Showing voice-derived estimates. Connect EEG for precise live brain data."
            : "Showing baseline estimates. Run a voice check-in or connect EEG for live analytics."}
        </div>
      )}

      {/* Score Gauges */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {hasRealData && brainHealthScore !== null ? (
          <div className="score-card p-4 flex flex-col items-center hover-glow">
            <ScoreCircle value={brainHealthScore} label="Brain Health" gradientId="grad-brain-health" colorFrom="hsl(152, 60%, 48%)" colorTo="hsl(200, 70%, 55%)" size="sm" />
            <p className="text-xs text-muted-foreground mt-1">Focus + Relaxation + Low Stress</p>
          </div>
        ) : (
          <div className="score-card p-4 flex flex-col items-center justify-center hover-glow h-[140px]">
            <span className="text-3xl font-semibold text-muted-foreground">—</span>
            <span className="text-xs font-medium text-muted-foreground mt-1">Brain Health</span>
            <p className="text-[10px] text-muted-foreground mt-1">No data</p>
          </div>
        )}
        {hasRealData && cognitiveScore !== null ? (
          <div className="score-card p-4 flex flex-col items-center hover-glow">
            <ScoreCircle value={cognitiveScore} label="Cognitive" gradientId="grad-cognitive" colorFrom="hsl(262, 45%, 65%)" colorTo="hsl(220, 50%, 50%)" size="sm" />
            <p className="text-xs text-muted-foreground mt-1">Focus + Creativity + Memory</p>
          </div>
        ) : (
          <div className="score-card p-4 flex flex-col items-center justify-center hover-glow h-[140px]">
            <span className="text-3xl font-semibold text-muted-foreground">—</span>
            <span className="text-xs font-medium text-muted-foreground mt-1">Cognitive</span>
            <p className="text-[10px] text-muted-foreground mt-1">No data</p>
          </div>
        )}
        {hasRealData && wellbeingScore !== null ? (
          <div className="score-card p-4 flex flex-col items-center hover-glow">
            <ScoreCircle value={wellbeingScore} label="Wellbeing" gradientId="grad-wellbeing" colorFrom="hsl(38, 85%, 58%)" colorTo="hsl(25, 85%, 55%)" size="sm" />
            <p className="text-xs text-muted-foreground mt-1">Relaxation + Low Stress + Flow</p>
          </div>
        ) : (
          <div className="score-card p-4 flex flex-col items-center justify-center hover-glow h-[140px]">
            <span className="text-3xl font-semibold text-muted-foreground">—</span>
            <span className="text-xs font-medium text-muted-foreground mt-1">Wellbeing</span>
            <p className="text-[10px] text-muted-foreground mt-1">No data</p>
          </div>
        )}
      </div>

      {/* Brain Health Trends */}
      <Card className="glass-card p-5 hover-glow">
        <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-medium">Brain Health Trends</h3>
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

        {/* Live: per-frame band-power chart — changes every 1.5 s */}
        {isLiveToday && isStreaming ? (
          liveBands.length < 2 ? (
            <div className="h-48 flex flex-col items-center justify-center text-sm text-muted-foreground gap-2">
              <Brain className="h-8 w-8 opacity-30" />
              <p>Collecting live data…</p>
            </div>
          ) : (
            <>
              <p className="text-[10px] text-muted-foreground mb-2">% of total EEG power per band — updates every 1.5 s</p>
              <ResponsiveContainer width="100%" height={192}>
                <LineChart data={liveBands}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(220,18%,14%)" opacity={0.5} />
                  <XAxis dataKey="time" tick={{ fontSize: 9, fill: "hsl(220,12%,42%)" }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                  <YAxis domain={[0, 60]} tick={{ fontSize: 9, fill: "hsl(220,12%,42%)" }} axisLine={false} tickLine={false} width={28} tickFormatter={(v) => `${v}%`} />
                  <Tooltip
                    cursor={{ stroke: "hsl(220,14%,55%)", strokeWidth: 1, strokeDasharray: "4 3" }}
                    contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 11 }}
                    formatter={(v: number, name: string) => [`${v}%`, name]}
                  />
                  <Line type="monotone" dataKey="calm"     name="Calm (α)"     stroke="hsl(152,65%,50%)" strokeWidth={2}   dot={false} isAnimationActive={false} activeDot={{ r: 4 }} />
                  <Line type="monotone" dataKey="alert"    name="Alert (β)"    stroke="hsl(200,70%,55%)" strokeWidth={2}   dot={false} isAnimationActive={false} activeDot={{ r: 4 }} />
                  <Line type="monotone" dataKey="creative" name="Creative (θ)" stroke="hsl(270,65%,62%)" strokeWidth={1.5} strokeDasharray="5 3" dot={false} isAnimationActive={false} activeDot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
              <div className="flex gap-4 mt-2 flex-wrap">
                {[
                  { label: "Calm (α — alpha)",    color: "hsl(152,65%,50%)" },
                  { label: "Alert (β — beta)",     color: "hsl(200,70%,55%)" },
                  { label: "Creative (θ — theta)", color: "hsl(270,65%,62%)", dashed: true },
                ].map((l) => (
                  <div key={l.label} className="flex items-center gap-1.5">
                    <svg width="16" height="8"><line x1="0" y1="4" x2="16" y2="4" stroke={l.color} strokeWidth="2" strokeDasharray={l.dashed ? "4 3" : "0"} /></svg>
                    <span className="text-[10px] text-muted-foreground">{l.label}</span>
                  </div>
                ))}
              </div>
            </>
          )
        ) : !hasChartData ? (
          <div className="h-48 flex flex-col items-center justify-center text-sm text-muted-foreground gap-2">
            <Brain className="h-8 w-8 opacity-30" />
            <p>{isLiveToday ? "Connect device to see trends" : "No sessions in this period"}</p>
          </div>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={192}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="focusGradH" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(200,70%,55%)" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="hsl(200,70%,55%)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(220,18%,14%)" opacity={0.5} />
                <XAxis dataKey={dataKey} tick={{ fontSize: 9, fill: "hsl(220,12%,42%)" }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                <YAxis domain={[0, 100]} tick={{ fontSize: 9, fill: "hsl(220,12%,42%)" }} axisLine={false} tickLine={false} width={24} />
                <Tooltip
                  cursor={{ stroke: "hsl(220,14%,55%)", strokeWidth: 1, strokeDasharray: "4 3" }}
                  contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 11 }}
                  formatter={(v: number) => [`${v}%`]}
                />
                <Area type="monotone" dataKey="focus" name="Focus" stroke="hsl(200,70%,55%)" fill="url(#focusGradH)" strokeWidth={2} dot={false} isAnimationActive={false} activeDot={{ r: 4 }} />
                <Line type="monotone" dataKey="stress" name="Stress" stroke="hsl(38,85%,58%)" strokeWidth={1.5} strokeDasharray="4 3" dot={false} isAnimationActive={false} activeDot={{ r: 4 }} />
                <Line type="monotone" dataKey="relaxation" name="Relax" stroke="hsl(152,60%,48%)" strokeWidth={1.5} dot={false} isAnimationActive={false} activeDot={{ r: 4 }} />
                <Line type="monotone" dataKey="flow" name="Flow" stroke="hsl(262,45%,65%)" strokeWidth={1.5} strokeDasharray="2 2" dot={false} isAnimationActive={false} activeDot={{ r: 4 }} />
              </AreaChart>
            </ResponsiveContainer>
            <div className="flex gap-4 mt-2 flex-wrap">
              {[
                { label: "Focus",   color: "hsl(200,70%,55%)" },
                { label: "Stress",  color: "hsl(38,85%,58%)",  dashed: true },
                { label: "Relax",   color: "hsl(152,60%,48%)" },
                { label: "Flow",    color: "hsl(262,45%,65%)", dashed: true },
              ].map((l) => (
                <div key={l.label} className="flex items-center gap-1.5">
                  <svg width="16" height="8"><line x1="0" y1="4" x2="16" y2="4" stroke={l.color} strokeWidth="2" strokeDasharray={l.dashed ? "4 3" : "0"} /></svg>
                  <span className="text-[10px] text-muted-foreground">{l.label}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </Card>

      {/* Brain-Health Insights (live only) */}
      {insights.length > 0 && (
        <Card className="glass-card p-5 hover-glow">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-medium">Brain-Health Insights</h3>
            </div>
            <Badge variant={isStreaming ? "default" : "secondary"} className="text-[10px]">
              {isStreaming ? "Live Data" : "No Data"}
            </Badge>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {insights.map((insight, i) => (
              <div
                key={i}
                className="p-4 rounded-xl"
                style={{ background: "hsl(220, 22%, 8%)", border: "1px solid hsl(220, 18%, 13%)" }}
              >
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-medium text-foreground">{insight.title}</h4>
                  <span
                    className="text-[10px] font-mono px-2 py-0.5 rounded-full"
                    style={{ background: `hsl(152, 60%, 48%, ${insight.strength * 0.2})`, color: "hsl(152, 60%, 48%)" }}
                  >
                    {Math.round(insight.strength * 100)}%
                  </span>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">{insight.description}</p>
                <div className="flex gap-2 mt-2">
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary">{insight.brain.replace(/_/g, " ")}</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent/10 text-accent">{insight.health}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </main>
  );
}
