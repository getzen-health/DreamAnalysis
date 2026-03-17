import { getParticipantId } from "@/lib/participant";
import { useState, useEffect, useRef } from "react";
import { Link } from "wouter";
import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Brain,
  Activity,
  AlertCircle,
  ArrowRight,
  ChevronRight,
  Moon,
  MessageSquare,
  MessageCircle,
  Sparkles,
  Radio,
  Wind,
  TrendingUp,
  TrendingDown,
  Trophy,
  Clock,
  Flame,
  Star,
  Music,
  Heart,
  Sun,
  BedDouble,
  UtensilsCrossed,
  Lightbulb,
  Network,
  Zap,
  Headphones,
  BookOpen,
  X as XIcon,
  AlertTriangle,
} from "lucide-react";
import { hapticLight } from "@/lib/haptics";
import { useDevice } from "@/hooks/use-device";
import { useHealthSync } from "@/hooks/use-health-sync";
import type { BiometricPayload } from "@/lib/health-sync";
import {
  getHealthInsights,
  listSessions,
  getBaselineStatus,
  type HealthInsight,
  type SessionSummary,
} from "@/lib/ml-api";
import { HealthEmotionCard } from "@/components/health-emotion-card";
import { StreakCard } from "@/components/streak-card";
import { ReadinessScore } from "@/components/readiness-score";
import { StreakBadge } from "@/components/streak-badge";
import { Skeleton } from "@/components/ui/skeleton";
import EmotionLandscape, { type HeatmapCell } from "@/components/emotion-landscape";
import { useAuth } from "@/hooks/use-auth";
import { useTheme } from "@/hooks/use-theme";

/* ---------- helpers ---------- */

/** Check whether a BiometricPayload contains any real sensor data. */
function hasRealHealthData(p: BiometricPayload): boolean {
  return !!(p.current_heart_rate || p.hrv_sdnn || p.sleep_efficiency || p.steps_today);
}

/** Derive approximate stress/focus/relaxation from health biometrics (no EEG). */
function deriveHealthState(p: BiometricPayload) {
  const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

  // Stress: elevated HR relative to resting + low HRV + poor sleep
  let stress = 0;
  if (p.current_heart_rate && p.resting_heart_rate && p.resting_heart_rate > 40) {
    stress += clamp((p.current_heart_rate - p.resting_heart_rate) / 40, -0.15, 0.45) * 0.4;
  }
  if (p.hrv_sdnn !== undefined && p.hrv_sdnn > 0) {
    stress = stress * 0.65 + clamp(1 - p.hrv_sdnn / 70, 0, 1) * 0.35;
  }
  if (p.sleep_efficiency !== undefined) {
    stress = stress * 0.75 + clamp(1 - p.sleep_efficiency / 100, 0, 1) * 0.25;
  }
  stress = clamp(stress, 0, 1);

  // Focus: driven by sleep quality + physical activity, dampened by stress
  let focus = 0;
  if (p.sleep_efficiency !== undefined) focus = clamp(p.sleep_efficiency / 100, 0, 1) * 0.65;
  if (p.steps_today !== undefined) focus += clamp(p.steps_today / 8000, 0, 1) * 0.2;
  focus = clamp(focus * (1 - stress * 0.45), 0, 1);

  // Relaxation: inverse of stress, boosted by good sleep
  let relaxation = clamp((1 - stress) * 0.65, 0, 0.65);
  if (p.sleep_efficiency !== undefined) relaxation += clamp(p.sleep_efficiency / 100, 0, 1) * 0.35;
  relaxation = clamp(relaxation, 0, 1);

  const source = p.hrv_sdnn !== undefined ? "Apple Health" : "Health data";
  return { stress: Math.round(stress * 100), focus: Math.round(focus * 100), relaxation: Math.round(relaxation * 100), source };
}

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

/** All-time longest consecutive-day streak across all sessions. */
function longestEverStreak(sessions: SessionSummary[]): number {
  if (sessions.length === 0) return 0;
  const dayMs = 86_400_000;
  const daySet = new Set(
    sessions.map((s) => {
      const d = new Date((s.start_time ?? 0) * 1000);
      d.setHours(0, 0, 0, 0);
      return d.getTime();
    }),
  );
  const sorted = Array.from(daySet).sort((a, b) => a - b);
  let best = 1;
  let current = 1;
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] - sorted[i - 1] === dayMs) {
      current++;
      best = Math.max(best, current);
    } else {
      current = 1;
    }
  }
  return best;
}

/** Focus trend: compares avg focus of last 7 sessions vs prior 7. */
function focusTrend(sessions: SessionSummary[]): "up" | "down" | "stable" {
  const withData = sessions.filter((s) => (s.summary?.n_frames ?? 0) > 0);
  if (withData.length < 4) return "stable";
  const recent = withData.slice(-7);
  const prior  = withData.slice(-14, -7);
  const mean = (arr: SessionSummary[]) =>
    arr.reduce((s, x) => s + (x.summary?.avg_focus ?? 0), 0) / arr.length;
  const rMean = mean(recent);
  const pMean = prior.length > 0 ? mean(prior) : rMean;
  const delta = pMean > 0 ? (rMean - pMean) / pMean * 100 : 0;
  if (delta >  6) return "up";
  if (delta < -6) return "down";
  return "stable";
}

/** Next streak milestone above current streak. */
function nextMilestone(streak: number): number {
  for (const m of [3, 7, 14, 21, 30, 60, 100]) {
    if (streak < m) return m;
  }
  return streak + 10;
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

// Primary 2x2 quick-action grid — most-used features
const QUICK_ACTION_CARDS = [
  { href: "/journal",      icon: MessageSquare, label: "Voice Check-in",  subtitle: "Log how you feel",    color: "hsl(270, 60%, 60%)" },
  { href: "/dreams",       icon: Moon,          label: "Dream Journal",   subtitle: "Record and analyze",  color: "hsl(230, 60%, 60%)" },
  { href: "/biofeedback",  icon: Wind,          label: "Breathe",         subtitle: "Guided breathing",    color: "hsl(170, 55%, 48%)" },
  { href: "/ai-companion", icon: MessageCircle, label: "AI Companion",    subtitle: "Talk to your guide",  color: "hsl(152, 60%, 48%)" },
];

// Discover section — horizontal scroll, links to deeper features
const DISCOVER_CARDS = [
  { href: "/brain-monitor",      icon: Brain,          label: "Brain Monitor",   desc: "Live EEG waveforms",       color: "hsl(152, 60%, 48%)" },
  { href: "/neurofeedback",      icon: Zap,            label: "Neurofeedback",   desc: "Train your focus",         color: "hsl(38, 85%, 58%)" },
  { href: "/inner-energy",       icon: Sparkles,       label: "Inner Energy",    desc: "Chakra & energy map",      color: "hsl(270, 65%, 65%)" },
  { href: "/food",               icon: UtensilsCrossed,label: "Food & Mood",     desc: "What you eat matters",     color: "hsl(25, 80%, 55%)" },
  { href: "/sleep-session",      icon: BedDouble,      label: "Sleep Session",   desc: "Track overnight rest",     color: "hsl(217, 70%, 55%)" },
  { href: "/weekly-summary",     icon: TrendingUp,     label: "Weekly Summary",  desc: "Trends over 7 days",       color: "hsl(190, 70%, 50%)" },
  { href: "/brain-connectivity", icon: Network,        label: "Connectivity",    desc: "Brain region links",       color: "hsl(160, 55%, 50%)" },
  { href: "/insights",           icon: Lightbulb,      label: "Insights",        desc: "Pattern analysis",         color: "hsl(50, 80%, 55%)"  },
  { href: "/sleep-stories",      icon: Headphones,     label: "Sleep Stories",   desc: "EEG-triggered fade-out",   color: "hsl(248, 65%, 62%)" },
  { href: "/cbti",               icon: BookOpen,       label: "CBT-i Program",   desc: "6-week sleep restriction", color: "hsl(210, 70%, 55%)" },
];

// Keep FEATURE_CARDS for backward compat with tests that reference specific labels
const FEATURE_CARDS = QUICK_ACTION_CARDS;

const USER_ID = getParticipantId();

/* ========== Component ========== */
export default function Dashboard() {
  const { user } = useAuth();
  const [calibrationBannerDismissed, setCalibrationBannerDismissed] = useState(() => {
    return localStorage.getItem("ndw_calibration_banner_dismissed") === "true";
  });
  const showCalibrationBanner =
    !calibrationBannerDismissed &&
    localStorage.getItem("ndw_onboarding_complete") === "true" &&
    localStorage.getItem("ndw_baseline_complete") === null;

  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;

  // Health data fallback (iOS/Android only — web returns null)
  const { latestPayload, lastSyncAt } = useHealthSync();
  const healthState = !isStreaming && latestPayload && hasRealHealthData(latestPayload) ? deriveHealthState(latestPayload) : null;

  // Last emotion check-in from localStorage (persisted by emotion-lab page)
  const [lastEmotionCheckin, setLastEmotionCheckin] = useState<{
    emotion: string;
    confidence: number;
    timestamp: number;
  } | null>(null);

  useEffect(() => {
    try {
      const raw = localStorage.getItem("ndw_last_emotion");
      if (!raw) return;
      const parsed = JSON.parse(raw);
      // Only show if less than 24 hours old
      if (parsed?.timestamp && Date.now() - parsed.timestamp < 24 * 60 * 60 * 1000) {
        setLastEmotionCheckin({
          emotion: parsed.result?.emotion ?? "neutral",
          confidence: parsed.result?.confidence ?? 0,
          timestamp: parsed.timestamp,
        });
      }
    } catch {
      // Ignore parse errors
    }
  }, []);

  // Extract live data
  const emotions = analysis?.emotions;
  const emotionShift = latestFrame?.emotion_shift;

  // Live metrics from analysis sub-objects
  const flowState  = (analysis as Record<string, any>)?.flow_state;
  const stressModel = (analysis as Record<string, any>)?.stress;
  const attnModel   = (analysis as Record<string, any>)?.attention;

  // Current metrics — prefer emotion-classifier values (ready after 30s buffer);
  // fall back to dedicated stress/attention models which are available immediately.
  const stressIndex = emotions?.stress_index
    ? emotions.stress_index * 100
    : (stressModel?.stress_index ?? 0) * 100;
  const focusIndex = emotions?.focus_index
    ? emotions.focus_index * 100
    : (attnModel?.attention_score ?? 0) * 100;
  const relaxationIndex = (emotions?.relaxation_index ?? 0) * 100;
  // emotions?.ready is false for first 30s while buffer fills
  // ready=false only during first 30s buffer fill; undefined/true means result available
  const emotionReady = !emotions || emotions.ready !== false || emotions.emotion != null;
  const currentEmotion = emotionReady ? (emotions?.emotion ?? "—") : "Calibrating…";
  const confidence = emotions?.confidence ?? 0;

  // Derived live metrics
  const flowScore = (flowState?.flow_score ?? 0) * 100;


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
  const { data: healthInsights, isLoading: insightsLoading } = useQuery<HealthInsight[]>({
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
  const { data: allSessions = [], isLoading: sessionsLoading } = useQuery<SessionSummary[]>({
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
  const bestStreak  = longestEverStreak(allSessions);
  const trend       = focusTrend(sessionsWithData);
  const milestone   = nextMilestone(streak);
  const totalSessions = sessionsWithData.length;

  // New-record celebration — fires once when live focus exceeds all-time peak
  const [newFocusRecord, setNewFocusRecord] = useState(false);
  const recordTriggeredRef = useRef(false);
  useEffect(() => {
    if (!isStreaming || peakFocus === 0) { recordTriggeredRef.current = false; return; }
    if (focusIndex > peakFocus && !recordTriggeredRef.current) {
      recordTriggeredRef.current = true;
      setNewFocusRecord(true);
      setTimeout(() => setNewFocusRecord(false), 6000);
    }
    if (!isStreaming) recordTriggeredRef.current = false;
  }, [focusIndex, peakFocus, isStreaming]);

  const greetingName = user?.username ? user.username.charAt(0).toUpperCase() + user.username.slice(1) : "";
  const greetingTime = hour >= 5 && hour < 12 ? "Good morning" : hour >= 12 && hour < 17 ? "Good afternoon" : "Good evening";
  const { theme, setTheme } = useTheme();

  return (
    <main className="px-4 pt-2 pb-24 space-y-4 max-w-xl mx-auto">

      {/* ── Calibration banner — show when baseline not complete ─── */}
      {showCalibrationBanner && (
        <div className="relative flex items-start gap-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3">
          <AlertTriangle className="h-5 w-5 text-amber-500 shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-foreground leading-snug">
              EEG baseline not calibrated — emotion accuracy is reduced. A 2-minute setup improves accuracy by up to 29%.
            </p>
            <Link href="/calibration">
              <span className="inline-block mt-2 text-xs font-medium text-amber-400 hover:text-amber-300 transition-colors cursor-pointer">
                Calibrate Now &rarr;
              </span>
            </Link>
          </div>
          <button
            onClick={() => {
              localStorage.setItem("ndw_calibration_banner_dismissed", "true");
              setCalibrationBannerDismissed(true);
            }}
            className="shrink-0 p-1 rounded hover:bg-amber-500/20 transition-colors"
            aria-label="Dismiss calibration banner"
          >
            <XIcon className="h-4 w-4 text-amber-500/70" />
          </button>
        </div>
      )}

      {/* ── Greeting header — Oura-style inline row ─────────── */}
      <div
        className="flex items-center justify-between"
        style={{ paddingTop: "calc(env(safe-area-inset-top, 0px) + 12px)" }}
      >
        <div className="flex items-center gap-3 min-w-0">
          {/* Avatar */}
          {user ? (
            <div
              className="w-10 h-10 rounded-2xl flex items-center justify-center shrink-0 text-[14px] font-bold text-white"
              style={{ background: "linear-gradient(135deg, hsl(152,60%,40%), hsl(38,85%,50%))" }}
            >
              {user.username.charAt(0).toUpperCase()}
            </div>
          ) : (
            <div
              className="w-10 h-10 rounded-2xl flex items-center justify-center shrink-0 bg-muted border border-border"
            >
              <Activity className="h-4 w-4 text-muted-foreground/60" />
            </div>
          )}
          <div className="min-w-0">
            <h1 className="text-[20px] font-bold text-foreground tracking-tight leading-tight truncate">
              {greetingTime}{greetingName ? `, ${greetingName}` : ""}
            </h1>
            <p className="text-[12px] text-muted-foreground leading-tight">
              {isStreaming
                ? "EEG streaming live"
                : healthState
                ? `Connected to ${healthState.source}`
                : "How are you feeling today?"}
            </p>
          </div>
        </div>
        {/* Theme toggle */}
        <button
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          className="w-10 h-10 shrink-0 flex items-center justify-center rounded-2xl text-muted-foreground hover:text-foreground hover:bg-muted/40 active:bg-muted/60 transition-colors"
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        >
          {theme === "dark" ? <Sun className="h-4.5 w-4.5" /> : <Moon className="h-4.5 w-4.5" />}
        </button>
      </div>

      {/* ── Status pill ─────────────────────────────────────── */}
      {!isStreaming && !healthState && (
        <div className="flex items-center gap-2.5 px-4 py-3 rounded-2xl bg-muted/25 border border-border/30">
          <Radio className="h-4 w-4 shrink-0 text-primary" />
          <span className="text-[13px] text-muted-foreground">
            Voice check-in or sync health data to start. EEG optional.
          </span>
        </div>
      )}
      {!isStreaming && healthState && (
        <div className="flex items-center gap-2 px-3.5 py-2.5 rounded-2xl bg-emerald-500/6 border border-emerald-500/20">
          <Heart className="h-3.5 w-3.5 shrink-0 text-emerald-400" />
          <span className="text-xs text-emerald-400">
            Estimates from {healthState.source} · EEG adds live neural data
          </span>
        </div>
      )}

      {/* ── Last emotion check-in (from localStorage) ─────── */}
      {lastEmotionCheckin && (
        <div className="flex items-center gap-3 px-3.5 py-2.5 rounded-2xl bg-violet-500/6 border border-violet-500/20">
          <Heart className="h-3.5 w-3.5 shrink-0 text-violet-400" />
          <span className="text-xs text-foreground">
            Last check-in:{" "}
            <span className="font-semibold capitalize">{lastEmotionCheckin.emotion}</span>
            {lastEmotionCheckin.confidence > 0 && (
              <span className="text-muted-foreground">
                {" "}({Math.round(lastEmotionCheckin.confidence * 100)}%)
              </span>
            )}
            <span className="text-muted-foreground/60 ml-1">
              {(() => {
                const diffH = Math.round((Date.now() - lastEmotionCheckin.timestamp) / 3_600_000);
                if (diffH < 1) return "just now";
                if (diffH === 1) return "1h ago";
                return `${diffH}h ago`;
              })()}
            </span>
          </span>
        </div>
      )}

      {/* ── Passive health emotion (Apple Watch / Google Health) ── */}
      {!isStreaming && latestPayload && hasRealHealthData(latestPayload) && (
        <HealthEmotionCard payload={latestPayload} lastSyncAt={lastSyncAt} />
      )}

      {/* ── Empty state for brand-new users ────────────────── */}
      {!isStreaming && !healthState && sessionsWithData.length === 0 && !sessionsLoading && (
        <div
          className="rounded-2xl p-6 text-center space-y-4 bg-card/70 border border-border/60 shadow-sm"
        >
          <div className="w-14 h-14 rounded-2xl mx-auto flex items-center justify-center bg-primary/10 border border-primary/20">
            <Brain className="h-7 w-7 text-primary" />
          </div>
          <div className="space-y-1.5">
            <h2 className="text-lg font-bold text-foreground">Welcome to Neural Dream Workshop</h2>
            <p className="text-sm text-muted-foreground max-w-xs mx-auto">
              Start your first voice check-in or connect your EEG headset to begin tracking your brain health.
            </p>
          </div>
          <div className="flex items-center justify-center gap-3">
            <Link href="/emotions">
              <button
                className="px-4 py-2.5 rounded-xl text-sm font-semibold bg-primary text-primary-foreground hover:bg-primary/90 active:scale-[0.97] transition-all"
              >
                Voice Check-in
              </button>
            </Link>
            <Link href="/device-setup">
              <button
                className="px-4 py-2.5 rounded-xl text-sm font-semibold border border-border bg-muted/30 text-foreground hover:bg-muted/50 active:scale-[0.97] transition-all"
              >
                Connect Device
              </button>
            </Link>
          </div>
        </div>
      )}

      {/* ── Daily streak ────────────────────────────────────── */}
      <StreakCard userId={USER_ID} />

      {/* ── Brain Readiness Score (#353) ────────────────────── */}
      <ReadinessScore userId={USER_ID} />

      {/* ── Habit Streak Badge (#354) ────────────────────────── */}
      <StreakBadge userId={USER_ID} />

      {/* ── Quick actions — 2x2 card grid ───────────────────── */}
      <div>
        <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em] mb-3 px-0.5">
          Quick Actions
        </p>
        <div className="grid grid-cols-2 gap-2.5">
          {QUICK_ACTION_CARDS.map((card, index) => {
            const Icon = card.icon;
            return (
              <motion.div
                key={card.href}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <Link href={card.href} onClick={() => hapticLight()} aria-label={`${card.label}: ${card.subtitle}`}>
                  <div
                    className="group flex items-center gap-3 px-3.5 py-4 rounded-2xl active:scale-[0.97] transition-all duration-150 bg-card/70 border border-border/60 shadow-sm"
                    style={{ minHeight: "80px" }}
                  >
                    <div
                      className="w-10 h-10 rounded-2xl flex items-center justify-center shrink-0 transition-transform group-active:scale-95"
                      style={{ background: `${card.color}1a` }}
                    >
                      <Icon className="h-5 w-5" style={{ color: card.color }} aria-hidden="true" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-[13px] font-semibold text-foreground leading-tight truncate">{card.label}</p>
                      <p className="text-[11px] text-muted-foreground/70 mt-0.5 leading-tight truncate">{card.subtitle}</p>
                    </div>
                  </div>
                </Link>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* ── Discover — horizontal scroll chips ───────────────── */}
      <div>
        <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em] mb-3 px-0.5">
          Discover
        </p>
        <div
          className="flex gap-3 overflow-x-auto pb-1"
          style={{ scrollbarWidth: "none", WebkitOverflowScrolling: "touch" }}
        >
          {DISCOVER_CARDS.map((card) => {
            const Icon = card.icon;
            return (
              <Link
                key={card.href}
                href={card.href}
                onClick={() => hapticLight()}
                aria-label={`${card.label}: ${card.desc}`}
                className="shrink-0"
              >
                <div
                  className="flex flex-col gap-2 p-3.5 rounded-2xl active:scale-[0.97] transition-all duration-150 bg-card/70 border border-border/60 shadow-sm"
                  style={{ width: "120px", minHeight: "100px" }}
                >
                  <div
                    className="w-9 h-9 rounded-xl flex items-center justify-center"
                    style={{ background: `${card.color}1a` }}
                  >
                    <Icon className="h-4.5 w-4.5" style={{ color: card.color }} aria-hidden="true" />
                  </div>
                  <div>
                    <p className="text-[12px] font-semibold text-foreground leading-tight">{card.label}</p>
                    <p className="text-[10px] text-muted-foreground/70 mt-0.5 leading-tight">{card.desc}</p>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Weekly stress landscape — disabled: was generating fake data from single stress value */}

      {/* 2. Baseline calibration prompt (shown until calibrated) */}
      {!baselineReady && isStreaming && (
        <Link href="/onboarding">
          <div className="p-4 rounded-xl border border-primary/30 bg-primary/5 text-sm flex items-center justify-between gap-3 cursor-pointer hover:bg-primary/10 transition-colors">
            <div className="flex items-center gap-3">
              <Brain className="h-4 w-4 shrink-0 text-primary" />
              <span>
                <span className="font-medium">Finish optional EEG calibration</span>
                <span className="text-muted-foreground ml-2">— improves live headset readings</span>
              </span>
            </div>
            <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
          </div>
        </Link>
      )}

      {/* 3. Capability Badges — hidden in consumer mode, dev-only info */}
      {false && totalSessions > 0 && (
        <div className="mb-2">
          <h3 className="text-sm font-medium text-muted-foreground mb-2">Active ML Models</h3>
          <div className="flex flex-wrap gap-2">
            {[
              "Sleep", "Emotion", "Flow", "Creativity", "Spiritual", "Lucid Dream",
              "Stress", "Focus", "Cognitive Load", "Attention", "Drowsiness", "Meditation",
              "Memory", "Artifact", "Denoising", "Online Learning",
            ].map((model) => (
              <Badge key={model} variant="secondary" className="text-xs">
                {model}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* ── Last Session Snapshot ───────────────────────────── */}
      {sessionsLoading ? (
        <div className="rounded-2xl p-4 bg-card/70 border border-border/60">
          <Skeleton className="h-3.5 w-36 mb-3" />
          <div className="grid grid-cols-4 gap-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="rounded-xl bg-muted/40 p-2.5 text-center">
                <Skeleton className="h-5 w-10 mx-auto mb-1.5" />
                <Skeleton className="h-2 w-8 mx-auto" />
              </div>
            ))}
          </div>
        </div>
      ) : lastSession ? (
        <div className="rounded-2xl p-4 bg-card/70 border border-border/60 shadow-sm">
          <div className="flex items-center justify-between mb-3">
            <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em]">
              Last Session
            </p>
            <span className="text-[10px] text-muted-foreground/50">
              {lastSession.start_time ? relativeDay(lastSession.start_time) : "Recent"}
              {lastSession.summary?.duration_sec
                ? ` · ${Math.round(lastSession.summary.duration_sec / 60)}m`
                : ""}
            </span>
          </div>
          <div className="grid grid-cols-4 gap-2">
            {[
              { value: Math.round((lastSession.summary?.avg_stress ?? 0) * 100), label: "Stress", color: "text-rose-400" },
              { value: Math.round((lastSession.summary?.avg_focus ?? 0) * 100), label: "Focus", color: "text-primary" },
              { value: Math.round((lastSession.summary?.avg_flow ?? 0) * 100), label: "Flow", color: "text-emerald-400" },
              { value: lastSession.summary?.dominant_emotion ?? "—", label: "Mood", color: "text-violet-400", isText: true },
            ].map((item) => (
              <div key={item.label} className="rounded-xl bg-muted/30 px-2 py-2.5 text-center">
                <p className={`text-[15px] font-bold font-mono leading-none ${item.color} ${item.isText ? "capitalize text-[12px]" : ""}`}>
                  {item.isText ? item.value : `${item.value}%`}
                </p>
                <p className="text-[10px] text-muted-foreground/60 mt-1.5 leading-none">{item.label}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {/* 5. Emotional Shift Alert */}
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
                  href="/biofeedback"
                  className="inline-flex items-center gap-1 mt-2 text-xs text-warning hover:text-warning/80 transition-colors"
                >
                  Start breathing session <ArrowRight className="h-3 w-3" />
                </Link>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 3. Yesterday's Insight + Personal Records */}
      {sessionsLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <Card className="glass-card p-4">
            <div className="flex items-start gap-3">
              <Skeleton className="w-8 h-8 rounded-lg shrink-0" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-3 w-32" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-5 w-20 rounded-full" />
              </div>
            </div>
          </Card>
          <Card className="glass-card p-4">
            <div className="flex items-center gap-2 mb-3">
              <Skeleton className="h-4 w-4" />
              <Skeleton className="h-3 w-28" />
            </div>
            <div className="space-y-3">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="flex items-center justify-between">
                  <div>
                    <Skeleton className="h-5 w-12 mb-1" />
                    <Skeleton className="h-2.5 w-16" />
                  </div>
                  <Skeleton className="h-3 w-20" />
                </div>
              ))}
            </div>
          </Card>
        </div>
      ) : (lastInsight || sessionsWithData.length > 0) && (
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

          {/* Personal Records — gamified */}
          {sessionsWithData.length >= 1 && (
            <Card className={`glass-card p-4 hover-glow transition-all ${newFocusRecord ? "ring-2 ring-emerald-400/50" : ""}`}>
              {/* Header */}
              <div className="flex items-center gap-2 mb-3">
                <Trophy className="h-4 w-4 text-amber-400" />
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Personal Records</p>
                {trend !== "stable" && (
                  <span className="ml-auto">
                    {trend === "up"
                      ? <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />
                      : <TrendingDown className="h-3.5 w-3.5 text-red-400" />}
                  </span>
                )}
              </div>

              {/* New record celebration banner */}
              {newFocusRecord && (
                <div className="mb-3 px-3 py-1.5 rounded-lg bg-emerald-500/15 border border-emerald-500/30 flex items-center gap-2 animate-pulse">
                  <Star className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                  <p className="text-xs font-bold text-emerald-400">New focus record this session — keep going!</p>
                </div>
              )}

              {/* Record rows */}
              <div className="space-y-2.5">
                {/* Focus */}
                <div className="flex items-center justify-between">
                  <div>
                    <p className={`text-lg font-bold font-mono leading-none ${newFocusRecord ? "text-emerald-400" : "text-primary"}`}>
                      {peakFocus > 0 ? `${peakFocus}%` : "—"}
                    </p>
                    <p className="text-[10px] text-muted-foreground mt-0.5">Peak focus</p>
                  </div>
                  {peakFocus > 0 && (
                    isStreaming && focusIndex > 0 ? (
                      <span className={`text-[11px] font-medium ${
                        focusIndex >= peakFocus
                          ? "text-emerald-400 font-bold"
                          : focusIndex >= peakFocus * 0.9
                          ? "text-amber-400"
                          : "text-muted-foreground/60"
                      }`}>
                        {focusIndex >= peakFocus
                          ? "Record broken!"
                          : focusIndex >= peakFocus * 0.9
                          ? `${Math.round(peakFocus - focusIndex)}% to beat`
                          : `Beat ${peakFocus}%?`}
                      </span>
                    ) : (
                      <Link href="/brain-report"
                        className="text-[10px] text-muted-foreground/50 hover:text-primary transition-colors">
                        Build toward it →
                      </Link>
                    )
                  )}
                </div>

                {/* Flow */}
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-lg font-bold font-mono text-success leading-none">
                      {peakFlow > 0 ? `${peakFlow}%` : "—"}
                    </p>
                    <p className="text-[10px] text-muted-foreground mt-0.5">Best flow</p>
                  </div>
                  {peakFlow > 0 && (
                    <Link href="/brain-report"
                      className="text-[10px] text-muted-foreground/50 hover:text-success transition-colors">
                      Review your best day →
                    </Link>
                  )}
                </div>

                {/* Longest session */}
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-lg font-bold font-mono text-secondary leading-none">
                      {longestSession > 0 ? `${longestSession}m` : "—"}
                    </p>
                    <p className="text-[10px] text-muted-foreground mt-0.5">Longest session</p>
                  </div>
                  {longestSession > 0 && (
                    <Link href="/emotions"
                      className="text-[10px] text-muted-foreground/50 hover:text-secondary transition-colors">
                      Beat {longestSession}m? →
                    </Link>
                  )}
                </div>
              </div>

              {/* Streak section */}
              <div className="mt-3 pt-3 border-t border-border/20 flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <Flame className={`h-3.5 w-3.5 ${streak >= 7 ? "text-orange-400" : streak >= 3 ? "text-amber-400" : "text-muted-foreground/40"}`} />
                  <span className={`text-sm font-bold font-mono ${streak >= 7 ? "text-orange-400" : streak >= 3 ? "text-amber-400" : "text-muted-foreground"}`}>
                    {streak}d
                  </span>
                  <span className="text-[10px] text-muted-foreground">streak</span>
                  {bestStreak > streak && (
                    <span className="text-[10px] text-muted-foreground/50">· best {bestStreak}d</span>
                  )}
                </div>
                <span className="text-[10px] text-muted-foreground/60">
                  {streak < milestone
                    ? `${milestone - streak} more → ${milestone}d goal`
                    : `${totalSessions} sessions total`}
                </span>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ── Brain State Now card ───────────────────────────── */}
      {(isStreaming || healthState) && (() => {
        const stress = Math.round(isStreaming ? stressIndex : healthState!.stress);
        const focus  = Math.round(isStreaming ? focusIndex  : healthState!.focus);
        const flow   = Math.round(isStreaming ? flowScore   : healthState!.relaxation);
        const stressBarColor = stress > 65
          ? "hsl(4,72%,55%)"
          : stress > 35
          ? "hsl(38,85%,58%)"
          : "hsl(152,60%,48%)";
        const focusBarColor = focus > 60
          ? "hsl(152,60%,48%)"
          : focus > 30
          ? "hsl(38,85%,58%)"
          : "hsl(4,72%,55%)";
        const emotionLabel = EMOTION_LABELS[currentEmotion] || currentEmotion;
        const actionHref  = stress > 65 ? "/biofeedback" : "/brain-report";
        const actionLabel = stress > 65 ? "Breathe →" : "Daily Report →";

        /* Mini ring helper — renders a 56px arc gauge */
        const MiniRing = ({
          value,
          color,
          label,
          sublabel,
        }: {
          value: number;
          color: string;
          label: string;
          sublabel?: string;
        }) => {
          const size = 64;
          const r = 24;
          const c = size / 2;
          const circ = 2 * Math.PI * r;
          const arc = circ * 0.78;
          const gap = circ * 0.22;
          const offset = arc * (1 - value / 100);
          return (
            <div className="flex flex-col items-center gap-1">
              <svg
                width={size}
                height={size}
                viewBox={`0 0 ${size} ${size}`}
                role="img"
                aria-label={`${label}: ${value}%${sublabel ? `, ${sublabel}` : ""}`}
              >
                <circle cx={c} cy={c} r={r} fill="none"
                  stroke="hsl(var(--border))" strokeWidth={5}
                  strokeDasharray={`${arc} ${gap}`} strokeLinecap="round"
                  transform={`rotate(129 ${c} ${c})`} />
                <circle cx={c} cy={c} r={r} fill="none"
                  stroke={color} strokeWidth={5}
                  strokeDasharray={`${arc} ${gap}`} strokeDashoffset={offset}
                  strokeLinecap="round"
                  transform={`rotate(129 ${c} ${c})`}
                  style={{ transition: "stroke-dashoffset 0.9s cubic-bezier(0.34,1.56,0.64,1)" }}
                />
                <text x={c} y={c + 1} textAnchor="middle" dominantBaseline="central"
                  fill="hsl(38,20%,92%)" fontSize={13} fontWeight="700"
                  fontFamily="Inter,system-ui,sans-serif">
                  {value}
                </text>
              </svg>
              <span className="text-[11px] font-semibold text-foreground leading-none">{label}</span>
              {sublabel && <span className="text-[10px] text-muted-foreground/60 leading-none">{sublabel}</span>}
            </div>
          );
        };

        return (
          <div
            className="rounded-2xl p-4 bg-card/70 border border-border/60 shadow-sm"
          >
            {/* Card header */}
            <div className="flex items-center justify-between mb-4">
              <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em]">
                Brain State Now
              </p>
              {isStreaming ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-primary/10 text-primary">
                  <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                  Live
                </span>
              ) : (
                <span className="text-[10px] text-muted-foreground/50">from {healthState?.source}</span>
              )}
            </div>

            {/* Three metric rings */}
            <div className="flex items-center justify-around mb-4" aria-live="polite" aria-label="Live brain state metrics">
              <MiniRing value={stress} color={stressBarColor} label="Stress" sublabel={stress > 65 ? "High" : stress > 35 ? "Moderate" : "Low"} />
              <MiniRing value={focus}  color={focusBarColor}  label="Focus"  sublabel={focus > 60 ? "Sharp" : focus > 30 ? "Steady" : "Low"} />
              <MiniRing value={flow}   color="hsl(200,70%,55%)" label={isStreaming ? "Flow" : "Relax"} />
            </div>

            {/* Bottom action row */}
            <div className="flex items-center justify-between pt-3 border-t border-border/20">
              <span className="text-[11px] text-muted-foreground/60 leading-tight max-w-[180px]">
                {isStreaming && emotionLabel !== "—" && emotionLabel !== "Calibrating…"
                  ? `${emotionLabel} · ${Math.round(confidence * 100)}% confident`
                  : isStreaming
                  ? insightText || "Calibrating…"
                  : `Estimated from ${healthState?.source}`}
              </span>
              <div className="flex items-center gap-3">
                <Link
                  href={`/biofeedback?tab=music&mood=${stress > 35 ? "calm" : "focus"}`}
                  className="text-[11px] font-medium text-violet-400"
                >
                  <Music className="h-3 w-3 inline mr-1" />
                  Music
                </Link>
                <Link href={actionHref}
                  className="text-[11px] font-semibold text-primary">
                  {actionLabel}
                </Link>
              </div>
            </div>
          </div>
        );
      })()}

      {/* 5. Brain-Health Insights — hidden for brand-new users */}
      {(totalSessions > 0 || (healthInsights && healthInsights.length > 0 && healthInsights[0]?.insight_type !== "info")) && <div className="grid grid-cols-1 gap-4">
        <Card className="glass-card p-5 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="h-4 w-4 text-accent" />
            <h3 className="text-sm font-medium">Brain-Health Insights</h3>
          </div>

          {insightsLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="p-3 rounded-xl bg-muted/50 border border-border">
                  <Skeleton className="h-4 w-48 mb-2" />
                  <Skeleton className="h-3 w-full mb-1" />
                  <Skeleton className="h-3 w-2/3 mb-2" />
                  <Skeleton className="h-4 w-28 rounded-full" />
                </div>
              ))}
            </div>
          ) : !healthInsights || healthInsights.length === 0 || healthInsights[0]?.insight_type === "info" ? (
            <div className="py-6 flex flex-col items-center text-sm text-muted-foreground gap-2">
              <Brain className="h-8 w-8 text-muted-foreground/40" />
              <p>Insights appear after a few days of data</p>
              <p className="text-xs text-muted-foreground/60 text-center px-4">
                Complete a few voice check-ins or EEG sessions and sync health data to unlock correlations.
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
      </div>}

      {/* Quick Actions removed — replaced by "Your Tools" feature cards above */}
    </main>
  );
}
