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
  Heart,
  Activity,
  Brain,
  Sparkles,
  Zap,
  TrendingUp,
  Radio,
} from "lucide-react";
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
interface HealthPoint {
  time: string;
  stress: number;
  focus: number;
  relaxation: number;
  cogLoad: number;
  flow: number;
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

  // Extract live metrics from ML models
  const emotions = analysis?.emotions;
  const stress = analysis?.stress;
  const attention = analysis?.attention;
  const cognitiveLoad = analysis?.cognitive_load;
  const flowState = analysis?.flow_state;
  const creativity = analysis?.creativity;
  const drowsiness = analysis?.drowsiness;
  const memoryEncoding = analysis?.memory_encoding;

  // Current live metrics
  const stressIndex = isStreaming ? Math.round((stress?.stress_index ?? emotions?.stress_index ?? 0) * 100) : 0;
  const focusScore = isStreaming ? Math.round((attention?.attention_score ?? emotions?.focus_index ?? 0) * 100) : 0;
  const relaxScore = isStreaming ? Math.round((emotions?.relaxation_index ?? 0) * 100) : 0;
  const cogLoadIndex = isStreaming ? Math.round((cognitiveLoad?.load_index ?? 0) * 100) : 0;
  const flowScore = isStreaming ? Math.round((flowState?.flow_score ?? 0) * 100) : 0;
  const creativityScore = isStreaming ? Math.round((creativity?.creativity_score ?? 0) * 100) : 0;
  const drowsinessIndex = isStreaming ? Math.round((drowsiness?.drowsiness_index ?? 0) * 100) : 0;
  const memoryScore = isStreaming ? Math.round((memoryEncoding?.encoding_score ?? 0) * 100) : 0;

  // Composite scores (live only)
  const brainHealthScore = isStreaming
    ? Math.round(focusScore * 0.25 + relaxScore * 0.25 + (100 - stressIndex) * 0.25 + flowScore * 0.25)
    : 0;
  const cognitiveScore = isStreaming
    ? Math.round(focusScore * 0.3 + creativityScore * 0.25 + memoryScore * 0.25 + (100 - drowsinessIndex) * 0.2)
    : 0;
  const wellbeingScore = isStreaming
    ? Math.round(relaxScore * 0.35 + (100 - stressIndex) * 0.35 + flowScore * 0.3)
    : 0;

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

  // Accumulate live timeline (Today)
  const [timeline, setTimeline] = useState<HealthPoint[]>([]);

  useEffect(() => {
    if (!isStreaming || !analysis) return;
    const now = new Date().toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    setTimeline((prev) => [
      ...prev.slice(-60),
      { time: now, stress: stressIndex, focus: focusScore, relaxation: relaxScore, cogLoad: cogLoadIndex, flow: flowScore },
    ]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  // Build chart data
  const cutoff = Date.now() / 1000 - periodDays * 86400;
  const periodSessions = allSessions.filter((s) => (s.start_time ?? 0) >= cutoff);
  const sessionChartData = buildHealthChartData(periodSessions, periodDays);

  const liveNow: HealthPoint | null = isLiveToday && isStreaming
    ? { time: "Now", stress: stressIndex, focus: focusScore, relaxation: relaxScore, cogLoad: cogLoadIndex, flow: flowScore }
    : null;
  const todayChartData: (HealthPoint | SessionPoint)[] = liveNow
    ? [...timeline.slice(-30), liveNow]
    : timeline.slice(-30);

  const chartData = isLiveToday ? todayChartData : sessionChartData;
  const dataKey = isLiveToday ? "time" : "date";
  const hasData = chartData.length >= 1;

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
    if (stressIndex > 50) next.push({ title: "Elevated Stress", description: `Stress at ${stressIndex}%. Consider a breathing exercise or short break.`, strength: stressIndex / 100, brain: "stress_index", health: "relaxation" });
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
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          Connect your Muse 2 from the sidebar to see live health analytics.
        </div>
      )}

      {/* Score Gauges */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle value={brainHealthScore} label="Brain Health" gradientId="grad-brain-health" colorFrom="hsl(152, 60%, 48%)" colorTo="hsl(200, 70%, 55%)" size="sm" />
          <p className="text-xs text-muted-foreground mt-1">Focus + Relaxation + Low Stress</p>
        </div>
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle value={cognitiveScore} label="Cognitive" gradientId="grad-cognitive" colorFrom="hsl(262, 45%, 65%)" colorTo="hsl(220, 50%, 50%)" size="sm" />
          <p className="text-xs text-muted-foreground mt-1">Focus + Creativity + Memory</p>
        </div>
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle value={wellbeingScore} label="Wellbeing" gradientId="grad-wellbeing" colorFrom="hsl(38, 85%, 58%)" colorTo="hsl(25, 85%, 55%)" size="sm" />
          <p className="text-xs text-muted-foreground mt-1">Relaxation + Low Stress + Flow</p>
        </div>
      </div>

      {/* Vital Stats Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { icon: Activity, label: "Stress", value: `${stressIndex}%`, color: stressIndex > 50 ? "text-warning" : "text-success" },
          { icon: Brain, label: "Focus", value: `${focusScore}%`, color: "text-primary" },
          { icon: Zap, label: "Flow", value: `${flowScore}%`, color: "text-accent" },
          { icon: Heart, label: "Relaxation", value: `${relaxScore}%`, color: "text-success" },
          { icon: TrendingUp, label: "Creativity", value: `${creativityScore}%`, color: "text-secondary" },
          { icon: Brain, label: "Memory", value: `${memoryScore}%`, color: "text-primary" },
          { icon: Activity, label: "Cog Load", value: `${cogLoadIndex}%`, color: cogLoadIndex > 70 ? "text-warning" : "text-foreground" },
          { icon: Activity, label: "Drowsiness", value: `${drowsinessIndex}%`, color: drowsinessIndex > 60 ? "text-warning" : "text-success" },
        ].map((stat) => {
          const Icon = stat.icon;
          return (
            <Card key={stat.label} className="glass-card p-4 hover-glow">
              <div className="flex items-center gap-2 mb-2">
                <Icon className={`h-4 w-4 ${stat.color}`} />
                <span className="text-xs text-muted-foreground">{stat.label}</span>
              </div>
              <p className={`text-lg font-semibold font-mono ${stat.color}`}>{stat.value}</p>
            </Card>
          );
        })}
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

        {!hasData ? (
          <div className="h-48 flex flex-col items-center justify-center text-sm text-muted-foreground gap-2">
            <Brain className="h-8 w-8 opacity-30" />
            <p>{isLiveToday ? (isStreaming ? "Collecting data…" : "Connect device to see trends") : "No sessions in this period"}</p>
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
                  contentStyle={{ background: "hsl(220,22%,9%)", border: "1px solid hsl(220,18%,20%)", borderRadius: 10, fontSize: 11 }}
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
