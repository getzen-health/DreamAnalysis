import { getParticipantId } from "@/lib/participant";
import { useState, useEffect, useRef } from "react";
import { Link } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { ScoreCircle } from "@/components/score-circle";
import {
  Heart,
  Brain,
  Activity,
  AlertCircle,
  ArrowRight,
  ChevronRight,
  Moon,
  MessageSquare,
  Sparkles,
  Radio,
  Zap,
  Eye,
  Cpu,
  MemoryStick,
  Wind,
  TrendingUp,
  Trophy,
  Clock,
  Flame,
} from "lucide-react";
import { useDevice } from "@/hooks/use-device";
import {
  getHealthInsights,
  listSessions,
  getBaselineStatus,
  type HealthInsight,
  type SessionSummary,
} from "@/lib/ml-api";

/* ---------- helpers ---------- */

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
  // Allow streak to still count if today has no session yet (start from yesterday)
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

/* Build a one-sentence insight from the most recent session with real data */
function generateSessionInsight(
  session: SessionSummary,
  allSessions: SessionSummary[],
): { text: string; metric: string; metricLabel: string; color: string } {
  const s = session.summary;
  const focus = Math.round((s?.avg_focus ?? 0) * 100);
  const stress = Math.round((s?.avg_stress ?? 0) * 100);
  const flow = Math.round((s?.avg_flow ?? 0) * 100);
  const relaxation = Math.round((s?.avg_relaxation ?? 0) * 100);
  const duration = Math.round((s?.duration_sec ?? 0) / 60);
  const emotion = s?.dominant_emotion ?? "neutral";

  // Compute 7-day average focus for comparison
  const recent = allSessions
    .filter((x) => (x.summary?.n_frames ?? 0) > 0)
    .slice(-7);
  const avgFocus7d =
    recent.length > 1
      ? Math.round(
          (recent.slice(0, -1).reduce((a, x) => a + (x.summary?.avg_focus ?? 0), 0) /
            (recent.length - 1)) *
            100,
        )
      : 0;
  const focusDelta = avgFocus7d > 0 ? focus - avgFocus7d : 0;

  if (flow > 55) {
    return {
      text: `You hit a flow state for much of the ${duration}-min session.`,
      metric: `${flow}%`,
      metricLabel: "Flow",
      color: "text-success",
    };
  }
  if (focusDelta >= 15 && avgFocus7d > 0) {
    return {
      text: `Focus was ${Math.abs(focusDelta)}% higher than your recent average.`,
      metric: `${focus}%`,
      metricLabel: "Focus",
      color: "text-primary",
    };
  }
  if (focusDelta <= -15 && avgFocus7d > 0) {
    return {
      text: `Focus dipped ${Math.abs(focusDelta)}% below your recent average — worth a note.`,
      metric: `${focus}%`,
      metricLabel: "Focus",
      color: "text-warning",
    };
  }
  if (stress < 20 && focus > 50) {
    return {
      text: `Low stress with strong focus — your brain was in an optimal state.`,
      metric: `${stress}%`,
      metricLabel: "Stress",
      color: "text-success",
    };
  }
  if (stress > 65) {
    return {
      text: `Stress peaked at ${stress}%. A breathing session can help bring it down.`,
      metric: `${stress}%`,
      metricLabel: "Stress",
      color: "text-destructive",
    };
  }
  if (relaxation > 70) {
    return {
      text: `Deeply relaxed session (${relaxation}%). Great for recovery and memory consolidation.`,
      metric: `${relaxation}%`,
      metricLabel: "Relax",
      color: "text-success",
    };
  }
  return {
    text: `Dominant state: ${emotion}. ${duration}m session — Focus ${focus}%, Stress ${stress}%.`,
    metric: `${focus}%`,
    metricLabel: "Focus",
    color: "text-primary",
  };
}

/* Format relative time for session label */
function relativeDay(startTime: number): string {
  const now = Date.now() / 1000;
  const diffH = (now - startTime) / 3600;
  if (diffH < 2) return "just now";
  if (diffH < 24) return `${Math.round(diffH)}h ago`;
  if (diffH < 48) return "yesterday";
  return `${Math.round(diffH / 24)} days ago`;
}

const QUICK_ACTIONS = [
  { href: "/brain-monitor", icon: Activity, label: "Brain Monitor", color: "hsl(200, 70%, 55%)" },
  { href: "/dreams", icon: Moon, label: "Dream Journal", color: "hsl(262, 45%, 65%)" },
  { href: "/ai-companion", icon: MessageSquare, label: "AI Companion", color: "hsl(152, 60%, 48%)" },
  { href: "/biofeedback", icon: Wind, label: "Breathe", color: "hsl(38, 85%, 58%)" },
];

const USER_ID = getParticipantId();

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

  // --- Health insights query ---
  const { data: healthInsights } = useQuery<HealthInsight[]>({
    queryKey: ["health", "insights", USER_ID],
    queryFn: () => getHealthInsights(USER_ID),
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  // --- Baseline calibration status (for setup banner) ---
  const { data: baselineStatus } = useQuery({
    queryKey: ["baseline-status"],
    queryFn: () => getBaselineStatus(USER_ID),
    retry: false,
    staleTime: 5 * 60 * 1000,
  });
  const baselineReady = baselineStatus?.ready ?? true; // optimistic: hide banner until data loads

  // --- Sessions for Yesterday's Insight + Personal Records ---
  const { data: allSessions = [] } = useQuery<SessionSummary[]>({
    queryKey: ["sessions"],
    queryFn: () => listSessions(),
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  const sessionsWithData = allSessions.filter((s) => (s.summary?.n_frames ?? 0) > 0);
  const lastSession = sessionsWithData.length > 0
    ? sessionsWithData[sessionsWithData.length - 1]
    : null;
  const lastInsight = lastSession
    ? generateSessionInsight(lastSession, sessionsWithData)
    : null;

  const streak = currentStreak(allSessions);

  // Personal records across all sessions
  const peakFocus = sessionsWithData.reduce(
    (m, s) => Math.max(m, Math.round((s.summary?.avg_focus ?? 0) * 100)),
    0,
  );
  const peakFlow = sessionsWithData.reduce(
    (m, s) => Math.max(m, Math.round((s.summary?.avg_flow ?? 0) * 100)),
    0,
  );
  const longestSession = sessionsWithData.reduce(
    (m, s) => Math.max(m, Math.round((s.summary?.duration_sec ?? 0) / 60)),
    0,
  );

  return (
    <main className="p-6 space-y-6 max-w-6xl">
      {/* 1. Connection Banner */}
      {!isStreaming && (
        <Link href="/device-setup">
          <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center justify-between gap-3 cursor-pointer hover:bg-warning/10 transition-colors">
            <div className="flex items-center gap-3">
              <Radio className="h-4 w-4 shrink-0" />
              Connect your Muse 2 to see live dashboard data.
            </div>
            <ChevronRight className="h-4 w-4 shrink-0 opacity-60" />
          </div>
        </Link>
      )}

      {/* 2. Baseline calibration prompt (shown until calibrated) */}
      {!baselineReady && (
        <Link href="/onboarding">
          <div className="p-4 rounded-xl border border-primary/30 bg-primary/5 text-sm flex items-center justify-between gap-3 cursor-pointer hover:bg-primary/10 transition-colors">
            <div className="flex items-center gap-3">
              <Brain className="h-4 w-4 shrink-0 text-primary" />
              <span>
                <span className="font-medium">Calibrate for accurate readings</span>
                <span className="text-muted-foreground ml-2">— takes 2 min, +15–29% accuracy</span>
              </span>
            </div>
            <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
          </div>
        </Link>
      )}

      {/* 3. Emotional Shift Alert */}
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

      {/* 3. Yesterday's Insight + Personal Records */}
      {(lastInsight || sessionsWithData.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* Yesterday's Insight */}
          {lastInsight && lastSession && (
            <Card className="glass-card p-4 hover-glow">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-lg bg-secondary/10 flex items-center justify-center shrink-0">
                  <TrendingUp className="h-4 w-4 text-secondary" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                      Last Session Insight
                    </p>
                    <span className="text-[10px] text-muted-foreground flex items-center gap-1 shrink-0">
                      <Clock className="h-3 w-3" />
                      {relativeDay(lastSession.start_time ?? 0)}
                    </span>
                  </div>
                  <p className="text-sm text-foreground leading-snug mb-2">
                    {lastInsight.text}
                  </p>
                  <div className="flex items-center gap-2">
                    <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-muted ${lastInsight.color}`}>
                      {lastInsight.metricLabel} {lastInsight.metric}
                    </span>
                    <Link href="/sessions" className="text-[10px] text-muted-foreground hover:text-foreground flex items-center gap-0.5 transition-colors">
                      View sessions <ArrowRight className="h-3 w-3" />
                    </Link>
                  </div>
                </div>
              </div>
            </Card>
          )}

          {/* Personal Records */}
          {sessionsWithData.length >= 2 && (
            <Card className="glass-card p-4 hover-glow">
              <div className="flex items-center gap-2 mb-3">
                <Trophy className="h-4 w-4 text-amber-400" />
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Personal Records</p>
                {streak >= 3 && (
                  <span className="ml-auto flex items-center gap-1 text-[10px] font-semibold text-orange-400">
                    <Flame className="h-3 w-3" />{streak}d
                  </span>
                )}
              </div>
              <div className="grid grid-cols-4 gap-2">
                <div className="text-center">
                  <p className="text-xl font-bold text-primary font-mono">{peakFocus}%</p>
                  <p className="text-[10px] text-muted-foreground">Peak Focus</p>
                </div>
                <div className="text-center">
                  <p className="text-xl font-bold text-success font-mono">{peakFlow}%</p>
                  <p className="text-[10px] text-muted-foreground">Best Flow</p>
                </div>
                <div className="text-center">
                  <p className="text-xl font-bold text-secondary font-mono">{longestSession}m</p>
                  <p className="text-[10px] text-muted-foreground">Longest</p>
                </div>
                <div className="text-center">
                  <p className={`text-xl font-bold font-mono ${streak >= 7 ? "text-orange-400" : streak >= 3 ? "text-amber-400" : "text-muted-foreground"}`}>
                    {streak}
                  </p>
                  <p className="text-[10px] text-muted-foreground flex items-center justify-center gap-0.5">
                    {streak >= 3 && <Flame className="h-2.5 w-2.5 text-orange-400" />}Streak
                  </p>
                </div>
              </div>
              <p className="text-[10px] text-muted-foreground mt-2 text-center">
                Across {sessionsWithData.length} sessions
              </p>
              {isStreaming && focusIndex > 0 && (
                <p className={`text-[10px] text-center mt-1 font-medium ${focusIndex >= peakFocus ? "text-emerald-400" : focusIndex >= peakFocus * 0.85 ? "text-amber-400" : "text-muted-foreground/50"}`}>
                  {focusIndex >= peakFocus
                    ? "Focus record this session — keep going"
                    : focusIndex >= peakFocus * 0.85
                    ? `${Math.round(peakFocus - focusIndex)}% from your focus record`
                    : `Live focus: ${Math.round(focusIndex)}%`}
                </p>
              )}
            </Card>
          )}
        </div>
      )}

      {/* 4. Score Circles — Mental Health + Wellness + Sleep + Brain */}
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
              ? `Focus ${Math.round(focusIndex)}% · Stress ${Math.round(stressIndex)}%`
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

      {/* 6. AI Insight + Brain-Health Insights (side by side) */}
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
                    Stress {Math.round(stressIndex)}%
                  </span>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-secondary/10 text-secondary">
                    Focus {Math.round(focusIndex)}%
                  </span>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-accent/10 text-accent">
                    Relax {Math.round(relaxationIndex)}%
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

      {/* 7. Quick Actions */}
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
