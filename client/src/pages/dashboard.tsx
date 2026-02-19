import { useState, useEffect, useRef } from "react";
import { Link } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { ScoreCircle } from "@/components/score-circle";
import {
  AreaChart,
  Area,
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
  TrendingDown,
  Settings,
} from "lucide-react";
import { useDevice } from "@/hooks/use-device";
import {
  getHealthTrends,
  getHealthInsights,
  getWeeklyReport,
  type HealthTrend,
  type HealthInsight,
  type WeeklyReport,
} from "@/lib/ml-api";

/* ---------- types ---------- */
interface MoodPoint {
  time: string;
  mood: number;
  stress: number;
}

interface BandHistory {
  alpha: number[];
  theta: number[];
  beta: number[];
}

/* ---------- helpers ---------- */
const EMOTION_LABELS: Record<string, string> = {
  happy: "Happy",
  sad: "Sad",
  angry: "Angry",
  fearful: "Anxious",
  relaxed: "Relaxed",
  focused: "Focused",
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
  const currentEmotion = emotions?.emotion ?? "—";
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

  // Mood timeline — accumulate from live data
  const [moodHistory, setMoodHistory] = useState<MoodPoint[]>([]);

  useEffect(() => {
    if (!isStreaming || !emotions) return;
    const now = new Date().toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    setMoodHistory((prev) => [
      ...prev.slice(-30),
      {
        time: now,
        mood: Math.round(relaxationIndex * 0.5 + (100 - stressIndex) * 0.3 + focusIndex * 0.2),
        stress: Math.round(stressIndex),
      },
    ]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  // Brain wave sparklines — accumulate from band powers
  const [bandHistory, setBandHistory] = useState<BandHistory>({
    alpha: [],
    theta: [],
    beta: [],
  });

  useEffect(() => {
    if (!isStreaming || !bandPowers.alpha) return;
    setBandHistory((prev) => ({
      alpha: [...prev.alpha.slice(-30), (bandPowers.alpha ?? 0) * 100],
      theta: [...prev.theta.slice(-30), (bandPowers.theta ?? 0) * 100],
      beta: [...prev.beta.slice(-30), (bandPowers.beta ?? 0) * 100],
    }));
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

  // --- Health data queries ---
  const [trendDays, setTrendDays] = useState(7);

  const { data: weeklyReport } = useQuery<WeeklyReport>({
    queryKey: ["health", "weekly-report", USER_ID],
    queryFn: () => getWeeklyReport(USER_ID),
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  const { data: healthTrends } = useQuery<HealthTrend[]>({
    queryKey: ["health", "trends", USER_ID, trendDays],
    queryFn: () => getHealthTrends(USER_ID, trendDays),
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  const { data: healthInsights } = useQuery<HealthInsight[]>({
    queryKey: ["health", "insights", USER_ID],
    queryFn: () => getHealthInsights(USER_ID),
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  /* Sparkline renderer */
  const renderSparkline = (data: number[], color: string) => {
    if (data.length < 2) return null;
    const w = 200;
    const h = 40;
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;
    const points = data
      .map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * h}`)
      .join(" ");
    return (
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          strokeLinejoin="round"
        />
      </svg>
    );
  };

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

      {/* 3. Mental Health Score Hero */}
      <Card className="glass-card p-6 hover-glow">
        <div className="flex flex-col md:flex-row items-center gap-6">
          <div className="flex flex-col items-center">
            <ScoreCircle
              value={mentalHealthScore}
              label="Mental Health"
              gradientId="grad-mental-health"
              colorFrom={scoreColor(mentalHealthScore)}
              colorTo={mentalHealthScore > 70 ? "hsl(180, 65%, 50%)" : mentalHealthScore >= 40 ? "hsl(38, 60%, 65%)" : "hsl(4, 50%, 65%)"}
              size="lg"
            />
            <p className="text-xs text-muted-foreground mt-2">
              {isStreaming ? "Live composite score" : "Connect device to begin"}
            </p>
          </div>

          {/* Weekly change badges */}
          {weeklyReport && weeklyReport.total_sessions > 0 && (
            <div className="flex-1 grid grid-cols-2 sm:grid-cols-3 gap-3">
              {([
                { label: "Stress", change: weeklyReport.stress_change, invert: true },
                { label: "Focus", change: weeklyReport.focus_change, invert: false },
                { label: "Flow", change: weeklyReport.flow_change, invert: false },
                { label: "Relaxation", change: weeklyReport.relaxation_change, invert: false },
                { label: "Creativity", change: weeklyReport.creativity_change, invert: false },
              ]).map((item) => {
                const change = item.change ?? 0;
                const isPositive = item.invert ? change < 0 : change > 0;
                const displayChange = item.invert ? -change : change;
                return (
                  <div key={item.label} className="flex items-center gap-2 p-2 rounded-lg bg-muted/50">
                    {isPositive ? (
                      <TrendingUp className="h-3.5 w-3.5 text-success shrink-0" />
                    ) : (
                      <TrendingDown className="h-3.5 w-3.5 text-destructive shrink-0" />
                    )}
                    <div>
                      <p className="text-[10px] text-muted-foreground">{item.label}</p>
                      <p className={`text-xs font-mono font-medium ${isPositive ? "text-success" : "text-destructive"}`}>
                        {displayChange > 0 ? "+" : ""}{(displayChange * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>
                );
              })}
              <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/50">
                <Activity className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                <div>
                  <p className="text-[10px] text-muted-foreground">Sessions</p>
                  <p className="text-xs font-mono font-medium text-foreground">{weeklyReport.total_sessions}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* 4. Live Score Circles */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
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

      {/* 6. Health Trends */}
      <Card className="glass-card p-5 hover-glow">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-medium">Health Trends</h3>
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

        <div className="h-56">
          {!healthTrends || healthTrends.length < 2 ? (
            <div className="h-full flex flex-col items-center justify-center text-sm text-muted-foreground gap-2">
              <Settings className="h-8 w-8 text-muted-foreground/40" />
              <p>Connect Apple Health or Google Fit in Settings to see trends</p>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={healthTrends}>
                <defs>
                  <linearGradient id="trendFlow" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(152, 60%, 48%)" stopOpacity={0.2} />
                    <stop offset="100%" stopColor="hsl(152, 60%, 48%)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(d: string) => {
                    const date = new Date(d);
                    return `${date.getMonth() + 1}/${date.getDate()}`;
                  }}
                />
                <YAxis hide domain={[0, 1]} />
                <Tooltip
                  contentStyle={{
                    background: "var(--card)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                />
                <Area type="monotone" dataKey="flow_score" name="Flow" stroke="hsl(152, 60%, 48%)" fill="url(#trendFlow)" strokeWidth={2} dot={false} connectNulls />
                <Area type="monotone" dataKey="creativity_score" name="Creativity" stroke="hsl(262, 45%, 65%)" fill="none" strokeWidth={1.5} dot={false} connectNulls />
                <Area type="monotone" dataKey="encoding_score" name="Encoding" stroke="hsl(200, 70%, 55%)" fill="none" strokeWidth={1.5} dot={false} connectNulls />
                <Area type="monotone" dataKey="valence" name="Valence" stroke="hsl(38, 85%, 58%)" fill="none" strokeWidth={1.5} dot={false} strokeDasharray="4 4" connectNulls />
                <Area type="monotone" dataKey="arousal" name="Arousal" stroke="hsl(320, 55%, 60%)" fill="none" strokeWidth={1.5} dot={false} strokeDasharray="4 4" connectNulls />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Category pills */}
        {healthTrends && healthTrends.length >= 2 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {([
              { key: "flow_score", label: "Flow", color: "text-success" },
              { key: "creativity_score", label: "Creativity", color: "text-secondary" },
              { key: "encoding_score", label: "Encoding", color: "text-[hsl(200,70%,55%)]" },
              { key: "valence", label: "Valence", color: "text-warning" },
              { key: "arousal", label: "Arousal", color: "text-[hsl(320,55%,60%)]" },
            ] as const).map((cat) => {
              const vals = healthTrends.map((t) => t[cat.key]).filter((v): v is number => v !== null);
              const avg = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
              return (
                <span key={cat.key} className={`px-2.5 py-1 rounded-full text-[10px] font-medium bg-muted ${cat.color}`}>
                  {cat.label} {(avg * 100).toFixed(0)}%
                </span>
              );
            })}
          </div>
        )}
      </Card>

      {/* 7. AI Insight + Brain Waves (side by side) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* AI Insight */}
        {isStreaming && (
          <div className="ai-insight-card md:col-span-2">
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
          </div>
        )}

        {/* Brain Waves Sparklines */}
        <Card className={`glass-card p-5 hover-glow ${!isStreaming ? "md:col-span-3" : ""}`}>
          <div className="flex items-center gap-2 mb-4">
            <Brain className="h-4 w-4 text-secondary" />
            <h3 className="text-sm font-medium">Brain Waves</h3>
            {isStreaming && (
              <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </div>
          {bandHistory.alpha.length < 2 ? (
            <div className="h-32 flex items-center justify-center text-sm text-muted-foreground">
              {isStreaming ? "Collecting..." : "No data"}
            </div>
          ) : (
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
                  <span>Alpha</span>
                  <span className="font-mono">{Math.round(bandHistory.alpha[bandHistory.alpha.length - 1])}%</span>
                </div>
                {renderSparkline(bandHistory.alpha, "hsl(152, 60%, 48%)")}
              </div>
              <div>
                <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
                  <span>Theta</span>
                  <span className="font-mono">{Math.round(bandHistory.theta[bandHistory.theta.length - 1])}%</span>
                </div>
                {renderSparkline(bandHistory.theta, "hsl(262, 45%, 65%)")}
              </div>
              <div>
                <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
                  <span>Beta</span>
                  <span className="font-mono">{Math.round(bandHistory.beta[bandHistory.beta.length - 1])}%</span>
                </div>
                {renderSparkline(bandHistory.beta, "hsl(200, 70%, 55%)")}
              </div>
            </div>
          )}
        </Card>
      </div>

      {/* 8. Brain-Health Insights */}
      <Card className="glass-card p-5 hover-glow">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className="h-4 w-4 text-accent" />
          <h3 className="text-sm font-medium">Brain-Health Insights</h3>
        </div>

        {!healthInsights || healthInsights.length === 0 ? (
          <div className="py-8 flex flex-col items-center text-sm text-muted-foreground gap-2">
            <Brain className="h-8 w-8 text-muted-foreground/40" />
            <p>Insights will appear after a few days of data</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {healthInsights.slice(0, 3).map((insight, i) => (
              <div key={i} className="p-4 rounded-xl bg-muted/50 border border-border">
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

      {/* 9. Mood Timeline */}
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
            <ResponsiveContainer width="100%" height="100%">
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
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                />
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
