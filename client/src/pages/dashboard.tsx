import { useState, useEffect, useRef } from "react";
import { Link } from "wouter";
import { ContinuousBrainTimeline } from "@/components/charts/continuous-brain-timeline";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { ScoreCircle } from "@/components/score-circle";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";
import {
  Heart,
  Brain,
  Activity,
  AlertCircle,
  ArrowRight,
  Moon,
  Headphones,
  MessageSquare,
  Sparkles,
  Radio,
  Zap,
  Eye,
  Cpu,
  MemoryStick,
  TrendingUp,
} from "lucide-react";
import { useDevice } from "@/hooks/use-device";
import {
  getHealthInsights,
  listSessions,
  type HealthInsight,
  type SessionSummary,
} from "@/lib/ml-api";
import { ChartTooltip } from "@/components/chart-tooltip";

/* ---------- types ---------- */
interface MoodPoint {
  time: string;
  mood: number;
  stress: number;
}

/* ---------- helpers ---------- */
const EMOTION_LABELS: Record<string, string> = {
  happy: "Happy",
  sad: "Sad",
  angry: "Angry",
  fearful: "Anxious",
  relaxed: "Relaxed",
  focused: "Focused",
  neutral: "Neutral",
};

function getInsightText(
  stress: number,
  focus: number,
  relaxation: number,
  hour: number,
): string {
  const isNight = hour >= 21 || hour < 6;
  const isMorning = hour >= 6 && hour < 12;

  if (stress > 60) {
    return "Your stress levels are elevated. Consider a 5-minute breathing exercise or a neurofeedback session to recalibrate.";
  }
  if (relaxation > 70 && focus > 60) {
    return "You're in a flow state — high focus with low stress. This is optimal for creative work and deep thinking.";
  }
  if (isNight && relaxation > 50) {
    return "Your body is winding down naturally. This is a good time to journal your dreams and prepare for restorative sleep.";
  }
  if (isMorning) {
    return "Morning neural patterns suggest good baseline clarity. Set an intention now to maximize your cognitive potential today.";
  }
  if (focus > 65) {
    return "High focus detected — your prefrontal cortex is highly active. Great time for analytical tasks.";
  }
  return "Your neural patterns are balanced. Regular check-ins help build emotional awareness over time.";
}

/* Emotion shift type descriptions — mirrors backend EMOTION_PRECURSORS */
const EMOTION_PRECURSORS: Record<string, { description: string; guidance: string }> = {
  approaching_anxiety: {
    description: "Rising tension — beta increasing, alpha dropping",
    guidance: "Take 3 slow breaths. Ground yourself in the present moment.",
  },
  approaching_sadness: {
    description: "Withdrawal pattern — right frontal activation increasing",
    guidance: "Notice this feeling without judgment. It's information, not identity.",
  },
  approaching_calm: {
    description: "Settling pattern — alpha rising, beta decreasing",
    guidance: "Beautiful — your nervous system is settling. Let it happen.",
  },
  approaching_focus: {
    description: "Engagement pattern — beta structured, theta dropping",
    guidance: "You're entering a focused state. Channel it toward what matters.",
  },
  approaching_joy: {
    description: "Approach pattern — left frontal activation, gamma bursts",
    guidance: "Savor this. Consciously noting positive states strengthens them.",
  },
  emotional_turbulence: {
    description: "Rapid fluctuation — emotional state is unstable",
    guidance: "Your system is processing something. Pause. Breathe. Give yourself space.",
  },
  general_shift: {
    description: "Your emotional state is shifting",
    guidance: "Pause and check in with yourself. What are you feeling right now?",
  },
};

const QUICK_ACTIONS = [
  { href: "/brain-monitor", icon: Activity, label: "Brain Monitor", color: "hsl(200, 70%, 55%)" },
  { href: "/dreams", icon: Moon, label: "Dream Journal", color: "hsl(262, 45%, 65%)" },
  { href: "/ai-companion", icon: MessageSquare, label: "AI Companion", color: "hsl(152, 60%, 48%)" },
  { href: "/neurofeedback", icon: Headphones, label: "Neurofeedback", color: "hsl(38, 85%, 58%)" },
];

const USER_ID = "default";

const PERIOD_TABS = [
  { label: "Today", days: 1 },
  { label: "Week", days: 7 },
  { label: "Month", days: 30 },
  { label: "3 Months", days: 90 },
  { label: "Year", days: 365 },
];

/* Metric card config */
const HEALTH_METRICS = [
  { key: "stress", label: "Stress", source: "stress_index", color: "var(--warning)", bgClass: "bg-warning/10", textClass: "text-warning" },
  { key: "focus", label: "Focus", source: "focus_index", color: "var(--primary)", bgClass: "bg-primary/10", textClass: "text-primary" },
  { key: "flow", label: "Flow", source: "flow_score", color: "var(--success)", bgClass: "bg-success/10", textClass: "text-success" },
  { key: "creativity", label: "Creativity", source: "creativity_score", color: "var(--secondary)", bgClass: "bg-secondary/10", textClass: "text-secondary" },
  { key: "cogLoad", label: "Cog Load", source: "load_index", color: "var(--neural-blue)", bgClass: "bg-[hsl(200,70%,55%)]/10", textClass: "text-[hsl(200,70%,55%)]" },
  { key: "memory", label: "Memory", source: "encoding_score", color: "var(--neural-purple)", bgClass: "bg-[hsl(262,45%,65%)]/10", textClass: "text-[hsl(262,45%,65%)]" },
];

const METRIC_ICONS: Record<string, typeof Activity> = {
  stress: Zap,
  focus: Eye,
  flow: Activity,
  creativity: Sparkles,
  cogLoad: Cpu,
  memory: MemoryStick,
};

/* Score color helper */
function scoreColor(score: number): string {
  if (score > 70) return "hsl(152, 60%, 48%)";
  if (score >= 40) return "hsl(38, 85%, 58%)";
  return "hsl(4, 72%, 55%)";
}

/* Build time-series chart data from sessions with the right X-axis granularity */
type ChartPoint = { date: string; focus: number; stress: number; flow: number; creativity: number; ts: number };

function avg(arr: number[]) {
  return arr.length ? Math.round(arr.reduce((a, b) => a + b, 0) / arr.length) : 0;
}

function buildChartData(sessions: SessionSummary[], days: number): ChartPoint[] {
  const map: Record<string, { focus: number[]; stress: number[]; flow: number[]; creativity: number[]; ts: number }> = {};

  for (const s of sessions) {
    if (s.summary?.avg_focus == null) continue;
    const d = new Date((s.start_time ?? 0) * 1000);
    let key: string;
    let ts: number;

    if (days <= 1) {
      // Today: group by hour
      key = d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
      ts = d.getTime();
    } else if (days <= 7) {
      // Week: group by day
      key = d.toLocaleDateString("en-US", { weekday: "short" });
      ts = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    } else if (days <= 30) {
      // Month: group by day
      key = d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
      ts = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    } else if (days <= 90) {
      // 3 Months: group by week starting day
      const dayOfWeek = d.getDay();
      const weekStart = new Date(d);
      weekStart.setDate(d.getDate() - dayOfWeek);
      key = weekStart.toLocaleDateString("en-US", { month: "short", day: "numeric" });
      ts = weekStart.getTime();
    } else {
      // Year: group by month
      key = d.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
      ts = new Date(d.getFullYear(), d.getMonth(), 1).getTime();
    }

    if (!map[key]) map[key] = { focus: [], stress: [], flow: [], creativity: [], ts };
    map[key].focus.push((s.summary.avg_focus ?? 0) * 100);
    map[key].stress.push((s.summary.avg_stress ?? 0) * 100);
    map[key].flow.push((s.summary.avg_flow ?? 0) * 100);
    map[key].creativity.push((s.summary.avg_creativity ?? 0) * 100);
  }

  return Object.entries(map)
    .sort((a, b) => a[1].ts - b[1].ts)
    .map(([date, data]) => ({
      date,
      ts: data.ts,
      focus: avg(data.focus),
      stress: avg(data.stress),
      flow: avg(data.flow),
      creativity: avg(data.creativity),
    }));
}

/* ========== Component ========== */
export default function Dashboard() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;

  // Extract live data
  const emotions = analysis?.emotions;
  const bandPowers = analysis?.band_powers ?? {};
  const sleepStaging = analysis?.sleep_staging;
  const emotionShift = latestFrame?.emotion_shift;

  // Live metrics from analysis sub-objects
  const flowState = (analysis as Record<string, any>)?.flow_state;
  const creativity = (analysis as Record<string, any>)?.creativity;
  const cognitiveLoad = (analysis as Record<string, any>)?.cognitive_load;
  const memoryEncoding = (analysis as Record<string, any>)?.memory_encoding;

  // Current metrics
  const stressIndex = (emotions?.stress_index ?? 0) * 100;
  const focusIndex = (emotions?.focus_index ?? 0) * 100;
  const relaxationIndex = (emotions?.relaxation_index ?? 0) * 100;
  // emotions?.ready is false for first 30s while buffer fills
  // ready=false only during first 30s buffer fill; undefined/true means result available
  const emotionReady = !emotions || emotions.ready !== false || emotions.emotion != null;
  const currentEmotion = emotionReady ? (emotions?.emotion ?? "—") : "Calibrating…";
  const confidence = emotions?.confidence ?? 0;
  const valence = emotions?.valence ?? 0;
  const arousal = emotions?.arousal ?? 0;

  // Derived live metrics
  const flowScore = (flowState?.flow_score ?? 0) * 100;
  const creativityScore = (creativity?.creativity_score ?? 0) * 100;
  const cogLoadIndex = (cognitiveLoad?.load_index ?? 0) * 100;
  const memoryScore = (memoryEncoding?.encoding_score ?? 0) * 100;

  // Mental Health Score composite
  const mentalHealthScore = isStreaming
    ? Math.round(
        (100 - stressIndex) * 0.25 +
        focusIndex * 0.20 +
        flowScore * 0.20 +
        relaxationIndex * 0.20 +
        creativityScore * 0.15
      )
    : 0;

  // Live metric values for cards
  const liveMetricValues: Record<string, number> = {
    stress: Math.round(stressIndex),
    focus: Math.round(focusIndex),
    flow: Math.round(flowScore),
    creativity: Math.round(creativityScore),
    cogLoad: Math.round(cogLoadIndex),
    memory: Math.round(memoryScore),
  };


  // Today time-series — sample every 30s for the mental health chart
  const [todayTimeline, setTodayTimeline] = useState<
    Array<{ time: string; focus: number; stress: number; flow: number; creativity: number }>
  >([]);
  const lastTimelineSampleRef = useRef(0);
  const TIMELINE_INTERVAL_MS = 3_000; // one data point every 3 seconds

  // Mood timeline — accumulate from live frames (for the fine-grained Mood Timeline chart)
  const [moodHistory, setMoodHistory] = useState<MoodPoint[]>([]);

  useEffect(() => {
    if (!isStreaming || !emotions) return;
    const now = new Date().toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });

    // Mood history: every frame (throttled by FRAME_THROTTLE_MS in useDevice)
    setMoodHistory((prev) => [
      ...prev.slice(-30),
      {
        time: now,
        mood: Math.round(relaxationIndex * 0.5 + (100 - stressIndex) * 0.3 + focusIndex * 0.2),
        stress: Math.round(stressIndex),
      },
    ]);

    // Today timeline: only every 30s
    const ts = Date.now();
    if (ts - lastTimelineSampleRef.current >= TIMELINE_INTERVAL_MS) {
      lastTimelineSampleRef.current = ts;
      setTodayTimeline((prev) => [
        ...prev.slice(-600), // keep last 30 min at 3s intervals = 600 points max
        {
          time: now,
          focus: Math.round(focusIndex),
          stress: Math.round(stressIndex),
          flow: Math.round(flowScore),
          creativity: Math.round(creativityScore),
        },
      ]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);


  // Shift alert from emotion_shift
  const [shift, setShift] = useState<{
    detected: boolean;
    type: string;
    description: string;
    guidance: string;
    bodyFeeling: string;
  } | null>(null);

  useEffect(() => {
    if (!emotionShift?.shift_detected) return;
    const shiftType: string = emotionShift.shift_type ?? "general_shift";
    const isCalm = shiftType === "approaching_calm" || shiftType === "approaching_joy";
    setShift({
      detected: true,
      type: isCalm ? "approaching_calm" : shiftType,
      description: emotionShift.description ?? EMOTION_PRECURSORS[shiftType]?.description ?? "Emotional state is shifting",
      guidance: emotionShift.guidance ?? EMOTION_PRECURSORS[shiftType]?.guidance ?? "Pause and check in with yourself.",
      bodyFeeling: emotionShift.body_feeling ?? "",
    });
  }, [emotionShift?.shift_detected, emotionShift?.shift_type]);

  // Dismiss shift after 8s
  useEffect(() => {
    if (shift?.detected) {
      const timer = setTimeout(() => setShift(null), 8000);
      return () => clearTimeout(timer);
    }
  }, [shift]);

  // Computed scores from live data
  const wellnessScore = isStreaming
    ? Math.round(relaxationIndex * 0.4 + (100 - stressIndex) * 0.35 + focusIndex * 0.25)
    : 0;
  const sleepScore = isStreaming && sleepStaging
    ? Math.round(sleepStaging.confidence * 100)
    : 0;
  const brainScore = isStreaming
    ? Math.round(focusIndex * 0.4 + relaxationIndex * 0.3 + (100 - arousal * 60) * 0.3)
    : 0;

  const hour = new Date().getHours();
  const [insightText, setInsightText] = useState("");
  const insightTimerRef = useRef(0);
  const INSIGHT_THROTTLE_MS = 10_000;

  useEffect(() => {
    if (!isStreaming) {
      setInsightText("");
      return;
    }
    const now = Date.now();
    if (now - insightTimerRef.current < INSIGHT_THROTTLE_MS && insightText) return;
    insightTimerRef.current = now;
    setInsightText(getInsightText(stressIndex, focusIndex, relaxationIndex, hour));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  // --- Session data queries ---
  const [trendDays, setTrendDays] = useState(1); // default to Today

  const { data: allSessions = [] } = useQuery<SessionSummary[]>({
    queryKey: ["sessions"],
    queryFn: () => listSessions(),
    retry: false,
    staleTime: 2 * 60 * 1000,
    refetchInterval: 60_000,
  });

  const { data: healthInsights } = useQuery<HealthInsight[]>({
    queryKey: ["health", "insights", USER_ID],
    queryFn: () => getHealthInsights(USER_ID),
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  // Filter sessions by selected period
  const periodSessions = allSessions.filter(
    (s) => (s.start_time ?? 0) >= Date.now() / 1000 - trendDays * 86400
  );

  // Build time-series chart data with correct granularity per period
  const sessionTrendData = buildChartData(periodSessions, trendDays);

  // Peak focus hour
  const peakFocusHour = (() => {
    const map: Record<number, { sum: number; count: number }> = {};
    for (const s of periodSessions) {
      const val = s.summary?.avg_focus;
      if (val == null) continue;
      const h = new Date((s.start_time ?? 0) * 1000).getHours();
      if (!map[h]) map[h] = { sum: 0, count: 0 };
      map[h].sum += val;
      map[h].count += 1;
    }
    const entries = Object.entries(map);
    if (!entries.length) return null;
    const best = entries.reduce((a, b) =>
      a[1].sum / a[1].count > b[1].sum / b[1].count ? a : b
    );
    const h = Number(best[0]);
    const fmt = h === 0 ? "12 AM" : h < 12 ? `${h} AM` : h === 12 ? "12 PM" : `${h - 12} PM`;
    return { label: fmt, value: Math.round((best[1].sum / best[1].count) * 100) };
  })();

  // Dominant emotion
  const dominantEmotion = (() => {
    const counts: Record<string, number> = {};
    for (const s of periodSessions) {
      const e = s.summary?.dominant_emotion;
      if (e) counts[e] = (counts[e] ?? 0) + 1;
    }
    const entries = Object.entries(counts);
    if (!entries.length) return null;
    return entries.sort((a, b) => b[1] - a[1])[0][0];
  })();

  /* Sparkline renderer */

  return (
    <main className="p-6 space-y-6 max-w-6xl">
      {/* 1. Connection Banner */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          Connect your Muse 2 from the sidebar to see live dashboard data.
        </div>
      )}

      {/* 2. Emotional Shift Alert */}
      {shift?.detected && (
        <div
          className={
            shift.type === "approaching_calm" || shift.type === "approaching_joy"
              ? "shift-alert-calm"
              : "shift-alert"
          }
        >
          <div className="flex items-start gap-3">
            <AlertCircle
              className={`h-5 w-5 mt-0.5 shrink-0 ${
                shift.type === "approaching_calm" || shift.type === "approaching_joy"
                  ? "text-success"
                  : "text-warning"
              }`}
            />
            <div>
              <p className="text-sm font-medium text-foreground">
                Pre-Conscious Shift Detected
              </p>
              <p className="text-sm text-muted-foreground mt-1">{shift.description}</p>
              {shift.bodyFeeling && (
                <p className="text-xs text-muted-foreground/70 mt-1 italic">{shift.bodyFeeling}</p>
              )}
              <p className="text-xs text-foreground/60 mt-2">{shift.guidance}</p>
              {(shift.type === "approaching_anxiety" || shift.type === "emotional_turbulence") && (
                <Link
                  href="/neurofeedback"
                  className="inline-flex items-center gap-1 mt-2 text-xs text-warning hover:text-warning/80 transition-colors"
                >
                  Start neurofeedback session <ArrowRight className="h-3 w-3" />
                </Link>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 3. Score Circles — Mental Health + Wellness + Sleep + Brain */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
        <div className="score-card p-6 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={mentalHealthScore}
            label="Mental Health"
            gradientId="grad-mental-health"
            colorFrom={scoreColor(mentalHealthScore)}
            colorTo={mentalHealthScore > 70 ? "hsl(180, 65%, 50%)" : mentalHealthScore >= 40 ? "hsl(38, 60%, 65%)" : "hsl(4, 50%, 65%)"}
            size="lg"
          />
          <p className="text-xs text-muted-foreground mt-2">
            {isStreaming
              ? `${mentalHealthScore > 70 ? "Great" : mentalHealthScore >= 40 ? "Moderate" : "Low"}`
              : "—"}
          </p>
        </div>

        <div className="score-card p-6 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={wellnessScore}
            label="Wellness"
            gradientId="grad-wellness"
            colorFrom="hsl(152, 60%, 48%)"
            colorTo="hsl(180, 65%, 50%)"
            size="lg"
          />
          <p className="text-xs text-muted-foreground mt-2">
            {isStreaming
              ? `${EMOTION_LABELS[currentEmotion] || currentEmotion} · ${Math.round(confidence * 100)}%`
              : "—"}
          </p>
        </div>

        <div className="score-card p-6 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={sleepScore}
            label="Sleep"
            gradientId="grad-sleep"
            colorFrom="hsl(200, 70%, 55%)"
            colorTo="hsl(262, 45%, 65%)"
            size="lg"
          />
          <p className="text-xs text-muted-foreground mt-2">
            {isStreaming && sleepStaging ? `${sleepStaging.stage} stage` : "—"}
          </p>
        </div>

        <div className="score-card p-6 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={brainScore}
            label="Brain"
            gradientId="grad-brain"
            colorFrom="hsl(262, 45%, 65%)"
            colorTo="hsl(320, 55%, 60%)"
            size="lg"
          />
          <p className="text-xs text-muted-foreground mt-2">
            {isStreaming
              ? `Focus ${Math.round(focusIndex)} · Stress ${Math.round(stressIndex)}`
              : "—"}
          </p>
        </div>
      </div>

      {/* 5. Health Metric Cards (6-grid) */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
        {HEALTH_METRICS.map((metric) => {
          const Icon = METRIC_ICONS[metric.key];
          const value = liveMetricValues[metric.key];
          return (
            <Card key={metric.key} className="glass-card p-4 hover-glow">
              <div className="flex items-center gap-2 mb-2">
                <div className={`w-7 h-7 rounded-lg flex items-center justify-center ${metric.bgClass}`}>
                  <Icon className={`h-3.5 w-3.5 ${metric.textClass}`} />
                </div>
                <span className="text-xs text-muted-foreground">{metric.label}</span>
              </div>
              <p className="text-xl font-semibold font-mono">
                {value}%
              </p>
              <div className="mt-2 h-1.5 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${value}%`,
                    backgroundColor: metric.color,
                  }}
                />
              </div>
            </Card>
          );
        })}
      </div>

      {/* 6. Mental Health Trend — Apple Health-style continuous timeline */}
      <ContinuousBrainTimeline
        userId="default"
        defaultMetric="focus_index"
        title="Mental Health & Emotional Analysis"
      />

      {/* Legacy session chart — hidden, kept for reference */}
      {false && <Card className="glass-card p-5 hover-glow">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-medium">Mental Health & Emotional Analysis</h3>
          </div>
          <div className="flex gap-1">
            {PERIOD_TABS.map((tab) => (
              <button
                key={tab.days}
                onClick={() => setTrendDays(tab.days)}
                className={`px-3 py-1 text-xs rounded-full transition-colors ${
                  trendDays === tab.days
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Unified scrubbable chart — Apple Health / Whoop style */}
        {(() => {
          const isLiveToday = trendDays === 1 && isStreaming;
          const baseData = isLiveToday ? todayTimeline : sessionTrendData;
          const dataKey = isLiveToday ? "time" : "date";

          // Always inject the live "Now" point when streaming Today — no collecting wait
          const liveNow = isLiveToday
            ? {
                time: "Now",
                focus: Math.round(focusIndex),
                stress: Math.round(stressIndex),
                flow: Math.round(flowScore),
                creativity: Math.round(creativityScore),
              }
            : null;
          const fullData = liveNow ? [...baseData, liveNow] : baseData;
          // For live Today view, cap to last 120 points (6 min at 3s) for readability
          const chartData = isLiveToday ? fullData.slice(-120) : fullData;
          const hasData = chartData.length >= 1;


          if (!hasData) {
            return (
              <div className="h-40 flex flex-col items-center justify-center text-sm text-muted-foreground gap-2">
                <Brain className="h-8 w-8 text-muted-foreground/40" />
                <p>{trendDays === 1 ? "No sessions recorded today yet" : "No sessions in this period"}</p>
                <p className="text-xs text-muted-foreground/60">Connect your Muse 2 to start recording</p>
              </div>
            );
          }

          // Show timestamp only while scrubbing
          const showDots = chartData.length <= 2;

          return (
            <>
              {/* ── Chart ── */}
              <div className="h-48 select-none" style={{ touchAction: "pan-y" }}>
                <ResponsiveContainer width="100%" height={192}>
                  <LineChart data={chartData}>
                    <XAxis
                      dataKey={dataKey}
                      tick={{ fontSize: 9 }}
                      axisLine={false}
                      tickLine={false}
                      interval="preserveStartEnd"
                    />
                    <YAxis hide domain={[0, 100]} />
                    <Tooltip
                      cursor={{ stroke: "hsl(220,14%,55%)", strokeWidth: 1, strokeDasharray: "4 3" }}
                      contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 10, fontSize: 11 }}
                  labelStyle={{ color: "hsl(220, 12%, 65%)", marginBottom: 4, fontSize: 10 }}
                  itemStyle={{ padding: "1px 0" }}
                  formatter={(value: number) => [`${value}%`]}
                    />
                    <Line type="monotone" dataKey="focus" name="Focus" stroke="hsl(152,60%,48%)" strokeWidth={2} dot={showDots ? { r: 3, fill: "hsl(152,60%,48%)" } : false} isAnimationActive={false} activeDot={{ r: 4, fill: "hsl(152,60%,48%)" }} />
                    <Line type="monotone" dataKey="stress" name="Stress" stroke="hsl(38,85%,58%)" strokeWidth={1.5} strokeDasharray="4 3" dot={showDots ? { r: 3, fill: "hsl(38,85%,58%)" } : false} isAnimationActive={false} activeDot={{ r: 4, fill: "hsl(38,85%,58%)" }} />
                    <Line type="monotone" dataKey="flow" name="Flow" stroke="hsl(200,70%,55%)" strokeWidth={1.5} dot={showDots ? { r: 3, fill: "hsl(200,70%,55%)" } : false} isAnimationActive={false} activeDot={{ r: 4, fill: "hsl(200,70%,55%)" }} />
                    <Line type="monotone" dataKey="creativity" name="Creativity" stroke="hsl(262,45%,65%)" strokeWidth={1.5} dot={showDots ? { r: 3, fill: "hsl(262,45%,65%)" } : false} isAnimationActive={false} activeDot={{ r: 4, fill: "hsl(262,45%,65%)" }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* ── Legend ── */}
              <div className="flex gap-4 mt-2 flex-wrap">
                {[
                  { label: "Focus", color: "hsl(152,60%,48%)" },
                  { label: "Stress", color: "hsl(38,85%,58%)", dashed: true },
                  { label: "Flow", color: "hsl(200,70%,55%)" },
                  { label: "Creativity", color: "hsl(262,45%,65%)" },
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
          );
        })()}
      </Card>}

      {/* 7. AI Insight + Brain-Health Insights (side by side) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* AI Insight */}
        <div className="ai-insight-card">
          {isStreaming ? (
            <div className="flex items-start gap-3">
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
                style={{
                  background: "linear-gradient(135deg, hsl(152,60%,48%,0.2), hsl(38,85%,58%,0.2))",
                }}
              >
                <Sparkles className="h-4 w-4 text-primary" />
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium text-foreground mb-1">AI Insight</p>
                <p className="text-sm text-muted-foreground leading-relaxed">{insightText}</p>
                <div className="flex gap-2 mt-3">
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-primary/10 text-primary">
                    Stress {Math.round(stressIndex)}
                  </span>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-secondary/10 text-secondary">
                    Focus {Math.round(focusIndex)}
                  </span>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-accent/10 text-accent">
                    Relax {Math.round(relaxationIndex)}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-sm text-muted-foreground gap-2 py-6">
              <Sparkles className="h-8 w-8 text-muted-foreground/40" />
              <p>Connect device for live AI insights</p>
            </div>
          )}
        </div>

        {/* Brain-Health Insights */}
        <Card className="glass-card p-5 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="h-4 w-4 text-accent" />
            <h3 className="text-sm font-medium">Brain-Health Insights</h3>
          </div>

          {!healthInsights || healthInsights.length === 0 || healthInsights[0]?.insight_type === "info" ? (
            <div className="py-6 flex flex-col items-center text-sm text-muted-foreground gap-2">
              <Brain className="h-8 w-8 text-muted-foreground/40" />
              <p>Insights appear after a few days of data</p>
              <p className="text-xs text-muted-foreground/60 text-center px-4">
                Complete a few EEG sessions and sync health data to unlock brain-health correlations.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {healthInsights.slice(0, 3).map((insight, i) => (
                <div key={i} className="p-3 rounded-xl bg-muted/50 border border-border">
                  <p className="text-sm font-medium text-foreground mb-1">{insight.title}</p>
                  <p className="text-xs text-muted-foreground leading-relaxed mb-2">{insight.description}</p>
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
                      insight.correlation_strength > 0.7
                        ? "bg-success/10 text-success"
                        : insight.correlation_strength > 0.4
                          ? "bg-warning/10 text-warning"
                          : "bg-muted text-muted-foreground"
                    }`}>
                      {insight.correlation_strength > 0.7 ? "Strong" : insight.correlation_strength > 0.4 ? "Moderate" : "Weak"} correlation
                    </span>
                    <span className="text-[10px] text-muted-foreground">
                      {insight.evidence_count} data points
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* 8. Mood Timeline */}
      <Card className="glass-card p-5 hover-glow">
        <div className="flex items-center gap-2 mb-4">
          <Heart className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-medium">Mood Timeline</h3>
          {isStreaming && (
            <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
          )}
        </div>
        <div className="h-40">
          {moodHistory.length < 2 ? (
            <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
              {isStreaming ? "Collecting data..." : "Connect device to see timeline"}
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={moodHistory.slice(-20)}>
                <defs>
                  <linearGradient id="moodGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(152, 60%, 48%)" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="hsl(152, 60%, 48%)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="stressGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(38, 85%, 58%)" stopOpacity={0.2} />
                    <stop offset="100%" stopColor="hsl(38, 85%, 58%)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis hide domain={[0, 100]} />
                <Tooltip
                  contentStyle={{
                    background: "var(--card)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                />
                <Area type="monotone" dataKey="mood" stroke="hsl(152, 60%, 48%)" fill="url(#moodGrad)" strokeWidth={2} dot={false} />
                <Area type="monotone" dataKey="stress" stroke="hsl(38, 85%, 58%)" fill="url(#stressGrad)" strokeWidth={1.5} dot={false} strokeDasharray="4 4" />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </Card>

      {/* 10. Quick Actions (4 items) */}
      <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-4 gap-3">
        {QUICK_ACTIONS.map((action) => {
          const Icon = action.icon;
          return (
            <Link key={action.href} href={action.href}>
              <Card className="glass-card p-4 hover-glow cursor-pointer transition-all group text-center">
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center mx-auto mb-2 transition-transform group-hover:scale-110"
                  style={{ background: `${action.color}18` }}
                >
                  <Icon className="h-5 w-5" style={{ color: action.color }} />
                </div>
                <span className="text-xs text-muted-foreground group-hover:text-foreground transition-colors">
                  {action.label}
                </span>
              </Card>
            </Link>
          );
        })}
      </div>
    </main>
  );
}
