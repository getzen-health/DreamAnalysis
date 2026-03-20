import { useState, useEffect, useMemo, useCallback } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { resolveUrl, apiRequest } from "@/lib/queryClient";
import { getParticipantId } from "@/lib/participant";
import { useHealthSync } from "@/hooks/use-health-sync";
import { Sparkles, Moon, Heart, Footprints, UtensilsCrossed, Share2, Music, Wind, CloudMoon, Dumbbell, TreePine, AlertTriangle, Smile, Minus, Frown, PenLine, TrendingUp, TrendingDown } from "lucide-react";
import { ScoreSplash } from "@/components/score-splash";
import { hapticWarning } from "@/lib/haptics";
import { useVoiceData, type VoiceCheckinData } from "@/hooks/use-voice-data";
import { InlineBreathe } from "@/components/inline-breathe";
import { syncMoodLogToML } from "@/lib/ml-api";

// ── Types ──────────────────────────────────────────────────────────────────

interface EmotionCheckin {
  emotion?: string;
  probabilities?: Record<string, number>;
  valence?: number;
  arousal?: number;
  stress_index?: number;
  focus_index?: number;
  relaxation_index?: number;
}

interface FoodLog {
  totalCalories?: number;
  date?: string;
  loggedAt?: string;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

function computeReadiness(checkin: EmotionCheckin | null): number {
  if (!checkin) return 0;
  const stress = checkin.stress_index ?? 0.5;
  const focus = checkin.focus_index ?? 0.5;
  const valence = checkin.valence ?? 0;
  const raw = (1 - stress) * 30 + focus * 30 + ((valence + 1) / 2) * 20 + 20;
  return clamp(Math.round(raw), 0, 100);
}

function getEmotionScoreLabel(score: number): string {
  if (score === 0) return "Record a voice note to see your score";
  if (score >= 80) return "You're thriving emotionally";
  if (score >= 60) return "Positive emotional state";
  if (score >= 40) return "Mixed emotional state";
  return "Take care of yourself today";
}

function getStressLabel(stress: number): string {
  if (stress < 0.3) return "Low";
  if (stress < 0.6) return "Moderate";
  return "High";
}

function getStressColor(stress: number): string {
  if (stress < 0.3) return "#06b6d4";
  if (stress < 0.6) return "#d4a017";
  return "#e879a8";
}

function getFocusLabel(focus: number): string {
  if (focus >= 0.7) return "Sharp";
  if (focus >= 0.45) return "Moderate";
  return "Diffuse";
}

function getMoodLabel(valence: number): string {
  if (valence > 0.3) return "Positive";
  if (valence > -0.1) return "Normal";
  return "Low";
}

function getMoodDotColor(valence: number): string {
  if (valence > 0.3) return "#06b6d4";
  if (valence > -0.1) return "#d4a017";
  return "#e879a8";
}

function getAIInsight(checkin: EmotionCheckin | null): string {
  if (!checkin) return "Record a voice note to get your personalized AI insight.";
  const stress = checkin.stress_index ?? 0.5;
  const focus = checkin.focus_index ?? 0.5;
  const valence = checkin.valence ?? 0;
  const emotion = checkin.emotion ?? "neutral";

  if (stress < 0.3 && focus > 0.6) {
    return "Your stress is low and focus is high — great conditions for deep creative or analytical work.";
  }
  if (valence > 0.3 && stress < 0.4) {
    return "Positive mood detected. This is a good window for collaborative tasks or learning something new.";
  }
  if (stress > 0.65) {
    return "Elevated stress detected. Consider a 5-minute breathing exercise before your next task.";
  }
  if (emotion === "sad" || valence < -0.2) {
    return "Your mood is leaning negative. Light movement or social connection may help shift your state.";
  }
  if (focus < 0.35) {
    return "Focus is low right now. Short focused sprints (25-min Pomodoro) may help re-engage your attention.";
  }
  return "Your brain state looks balanced. Stay consistent with your routines today.";
}

function formatDate(): string {
  const now = new Date();
  const days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
  const months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  return `${days[now.getDay()]}, ${months[now.getMonth()]} ${now.getDate()}`;
}

// ── Share Wellness Score ─────────────────────────────────────────────────

async function shareWellnessScore(score: number, emotion: string, insight: string): Promise<void> {
  const W = 1080;
  const H = 1080;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // Background gradient
  const bgGrad = ctx.createLinearGradient(0, 0, W, H);
  bgGrad.addColorStop(0, "#1a0533");
  bgGrad.addColorStop(0.5, "#0f172a");
  bgGrad.addColorStop(1, "#0a1a2a");
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, W, H);

  // Radial glow behind score
  const glowHue = score >= 70 ? "#0891b2" : score >= 40 ? "#fbbf24" : "#f472b6";
  const glow = ctx.createRadialGradient(W / 2, 400, 0, W / 2, 400, 360);
  glow.addColorStop(0, glowHue + "30");
  glow.addColorStop(0.5, glowHue + "10");
  glow.addColorStop(1, "transparent");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, W, H);

  // Date
  ctx.fillStyle = "#64748b";
  ctx.font = "400 28px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "center";
  const dateStr = new Date().toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
  ctx.fillText(dateStr, W / 2, 120);

  // "WELLNESS SCORE" label
  ctx.fillStyle = "#94a3b8";
  ctx.font = "600 26px system-ui, -apple-system, sans-serif";
  ctx.letterSpacing = "6px";
  ctx.fillText("WELLNESS SCORE", W / 2, 200);
  ctx.letterSpacing = "0px";

  // Score arc (270-degree gauge)
  const cx = W / 2;
  const cy = 440;
  const r = 160;
  const strokeW = 14;
  const startAngle = (135 * Math.PI) / 180;
  const totalArc = (270 * Math.PI) / 180;
  const filledArc = (score / 100) * totalArc;

  // Background arc
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, startAngle + totalArc);
  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = strokeW;
  ctx.lineCap = "round";
  ctx.stroke();

  // Filled arc with gradient
  if (score > 0) {
    const arcGrad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
    arcGrad.addColorStop(0, "#7c3aed");
    arcGrad.addColorStop(1, "#e879a8");
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, startAngle + filledArc);
    ctx.strokeStyle = arcGrad;
    ctx.lineWidth = strokeW;
    ctx.lineCap = "round";
    ctx.stroke();
  }

  // Score number
  const scoreColor = score >= 70 ? "#22d3ee" : score >= 40 ? "#fde68a" : "#fda4af";
  ctx.fillStyle = scoreColor;
  ctx.font = "700 96px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(score), cx, cy - 10);

  // "Wellness" label below number
  ctx.fillStyle = "#94a3b8";
  ctx.font = "400 24px system-ui, -apple-system, sans-serif";
  ctx.textBaseline = "alphabetic";
  ctx.fillText("Wellness", cx, cy + 46);

  // Emotion chip
  if (emotion && emotion !== "---") {
    const chipLabel = `Mood: ${emotion.charAt(0).toUpperCase() + emotion.slice(1)}`;
    ctx.fillStyle = "#1e293b";
    const chipW = 260;
    const chipH = 48;
    const chipX = cx - chipW / 2;
    const chipY = 660;
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") {
      ctx.roundRect(chipX, chipY, chipW, chipH, 24);
    } else {
      ctx.moveTo(chipX + 24, chipY);
      ctx.lineTo(chipX + chipW - 24, chipY);
      ctx.quadraticCurveTo(chipX + chipW, chipY, chipX + chipW, chipY + 24);
      ctx.quadraticCurveTo(chipX + chipW, chipY + chipH, chipX + chipW - 24, chipY + chipH);
      ctx.lineTo(chipX + 24, chipY + chipH);
      ctx.quadraticCurveTo(chipX, chipY + chipH, chipX, chipY + 24);
      ctx.quadraticCurveTo(chipX, chipY, chipX + 24, chipY);
      ctx.closePath();
    }
    ctx.fill();
    ctx.fillStyle = "#f8fafc";
    ctx.font = "500 22px system-ui, -apple-system, sans-serif";
    ctx.fillText(chipLabel, cx, chipY + 30);
  }

  // AI insight text (word-wrapped)
  if (insight) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "400 24px system-ui, -apple-system, sans-serif";
    const maxW = W - 160;
    const words = insight.split(" ");
    let line = "";
    let y = 760;
    for (const word of words) {
      const test = line ? `${line} ${word}` : word;
      if (ctx.measureText(test).width > maxW) {
        ctx.fillText(line, cx, y);
        line = word;
        y += 34;
      } else {
        line = test;
      }
    }
    if (line) ctx.fillText(line, cx, y);
  }

  // Branding
  ctx.fillStyle = "#475569";
  ctx.font = "500 22px system-ui, -apple-system, sans-serif";
  ctx.fillText("NeuralDreamWorkshop", cx, H - 60);

  // Export as blob and share (wrapped in Promise so caller can await)
  const blob = await new Promise<Blob | null>((resolve) => {
    try {
      canvas.toBlob((b) => resolve(b), "image/png");
    } catch {
      resolve(null);
    }
  });

  // If toBlob failed, try toDataURL as fallback
  if (!blob) {
    try {
      const link = document.createElement("a");
      link.download = "antarai-wellness.png";
      link.href = canvas.toDataURL("image/png");
      link.click();
    } catch { /* last resort failed */ }
    return;
  }

  const file = new File([blob], "antarai-wellness.png", { type: "image/png" });

  // Try Web Share API with file
  try {
    if (navigator.share && navigator.canShare?.({ files: [file] })) {
      await navigator.share({
        title: `My Wellness Score: ${score}`,
        text: `My wellness score today is ${score}/100. ${insight}`,
        files: [file],
      });
      return;
    }
  } catch {
    // User cancelled or share failed — fall through to download
  }

  // Fallback: download the image directly
  try {
    const link = document.createElement("a");
    link.download = "antarai-wellness.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
  } catch {
    // toDataURL fallback failed — try blob URL
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "antarai-wellness.png";
    a.click();
    URL.revokeObjectURL(url);
  }
}

// ── Animation variants ──────────────────────────────────────────────────

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.06,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.4,
      ease: [0.22, 1, 0.36, 1],
    },
  },
};

// ── Bevel card style ─────────────────────────────────────────────────────

const bevelCard: React.CSSProperties = {
  borderRadius: 20,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "var(--card)",
  padding: "18px 20px",
};

// ── Hero Wellness Gauge ──────────────────────────────────────────────────

function WellnessGauge({ score }: { score: number }) {
  const size = 160;
  const strokeWidth = 10;
  const r = (size - strokeWidth) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * r;
  // Use 270 degrees of arc
  const arcLength = (270 / 360) * circumference;
  const filled = (score / 100) * arcLength;
  const gradientId = "gaugeGrad";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#7c3aed" />
            <stop offset="100%" stopColor="#e879a8" />
          </linearGradient>
        </defs>
        {/* Background arc */}
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke="var(--muted)"
          strokeWidth={strokeWidth}
          strokeDasharray={`${arcLength} ${circumference - arcLength}`}
          strokeLinecap="round"
          transform={`rotate(135 ${cx} ${cy})`}
        />
        {/* Filled arc */}
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeWidth={strokeWidth}
          strokeDasharray={`${filled} ${circumference - filled}`}
          strokeLinecap="round"
          transform={`rotate(135 ${cx} ${cy})`}
          style={{
            transition: "stroke-dasharray 1.4s cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        />
        {/* Percentage text */}
        <text
          x={cx}
          y={cy - 6}
          textAnchor="middle"
          fill="var(--foreground)"
          fontSize={36}
          fontWeight={700}
          fontFamily="system-ui, -apple-system, sans-serif"
        >
          {score}
        </text>
        <text
          x={cx}
          y={cy + 16}
          textAnchor="middle"
          fill="var(--muted-foreground)"
          fontSize={12}
          fontFamily="system-ui, -apple-system, sans-serif"
          letterSpacing="0.5"
        >
          Wellness
        </text>
      </svg>
      <p
        style={{
          fontSize: 14,
          color: score === 0 ? "var(--muted-foreground)" : "#7c3aed",
          margin: 0,
          textAlign: "center",
          lineHeight: 1.5,
          maxWidth: 220,
        }}
      >
        {getEmotionScoreLabel(score)}
      </p>
    </div>
  );
}

// ── Score Card (Mood / Stress / Focus) ───────────────────────────────────

function ScoreCard({
  label,
  value,
  statusLabel,
  dotColor,
  onClick,
  delta,
}: {
  label: string;
  value: string;
  statusLabel: string;
  dotColor: string;
  onClick?: () => void;
  delta?: number | null;
}) {
  return (
    <motion.div
      variants={itemVariants}
      onClick={onClick}
      style={{
        ...bevelCard,
        cursor: onClick ? "pointer" : "default",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 6,
        padding: "18px 20px",
      }}
    >
      <span
        style={{
          fontSize: 11,
          fontWeight: 500,
          color: "var(--muted-foreground)",
          textTransform: "uppercase" as const,
          letterSpacing: "0.6px",
        }}
      >
        {label}
      </span>
      <span
        style={{
          fontSize: 28,
          fontWeight: 700,
          color: "var(--foreground)",
          lineHeight: 1,
        }}
      >
        {value}
      </span>
      <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
        <div
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: dotColor,
            flexShrink: 0,
          }}
        />
        <span
          style={{
            fontSize: 11,
            color: "var(--muted-foreground)",
          }}
        >
          {statusLabel}
        </span>
      </div>
      {delta != null && Math.abs(delta) > 0.02 && (
        <div style={{ display: "flex", alignItems: "center", gap: 3, marginTop: 2 }}>
          {delta > 0 ? (
            <TrendingUp style={{ width: 12, height: 12, color: label === "Stress" ? "#e879a8" : "#06b6d4" }} />
          ) : (
            <TrendingDown style={{ width: 12, height: 12, color: label === "Stress" ? "#06b6d4" : "#e879a8" }} />
          )}
          <span style={{ fontSize: 10, color: "var(--muted-foreground)" }}>
            {Math.abs(Math.round(delta * 100))}% vs last
          </span>
        </div>
      )}
    </motion.div>
  );
}

// ── Health Metric Card ───────────────────────────────────────────────────

function HealthMetricCard({
  label,
  value,
  unit,
  statusLabel,
  dotColor,
  onClick,
  barPercent,
  barGradient,
  accentColor,
  emptyIcon: EmptyIcon,
  emptyCta,
}: {
  label: string;
  value: string;
  unit: string;
  statusLabel: string;
  dotColor: string;
  onClick?: () => void;
  barPercent?: number;
  barGradient?: string;
  /** Identity color for this metric — used for left border accent in empty state */
  accentColor?: string;
  /** Lucide icon shown in empty state to give visual identity */
  emptyIcon?: React.ElementType;
  /** Call-to-action text shown when no data is available */
  emptyCta?: string;
}) {
  const isEmpty = value === "---";

  return (
    <motion.div
      variants={itemVariants}
      onClick={onClick}
      style={{
        ...bevelCard,
        cursor: onClick ? "pointer" : "default",
        display: "flex",
        flexDirection: "column",
        gap: 14,
        ...(isEmpty && accentColor
          ? {
              borderLeft: `3px solid ${accentColor}`,
              background: `linear-gradient(135deg, ${accentColor}08 0%, var(--card) 40%)`,
            }
          : {}),
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          {isEmpty && EmptyIcon && (
            <EmptyIcon style={{ width: 14, height: 14, color: accentColor || "var(--muted-foreground)" }} />
          )}
          <span
            style={{
              fontSize: 11,
              fontWeight: 500,
              color: isEmpty && accentColor ? accentColor : "var(--muted-foreground)",
              textTransform: "uppercase" as const,
              letterSpacing: "0.6px",
            }}
          >
            {label}
          </span>
        </div>
        {!isEmpty && (
          <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <div
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: dotColor,
                flexShrink: 0,
              }}
            />
            <span style={{ fontSize: 11, color: "var(--muted-foreground)" }}>{statusLabel}</span>
          </div>
        )}
      </div>
      {isEmpty ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span
            style={{
              fontSize: 14,
              fontWeight: 500,
              color: "var(--muted-foreground)",
              lineHeight: 1.4,
            }}
          >
            {emptyCta || "No data yet"}
          </span>
        </div>
      ) : (
        <>
          <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
            <span
              style={{
                fontSize: 28,
                fontWeight: 700,
                color: "var(--foreground)",
                lineHeight: 1,
              }}
            >
              {value}
            </span>
            <span style={{ fontSize: 14, color: "var(--muted-foreground)" }}>{unit}</span>
          </div>
          {barPercent !== undefined && (
            <div
              style={{
                height: 10,
                borderRadius: 6,
                background: "var(--muted)",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${clamp(barPercent, 0, 100)}%`,
                  background: barGradient || "linear-gradient(90deg, #7c3aed, #e879a8)",
                  borderRadius: 6,
                  transition: "width 1s cubic-bezier(0.22, 1, 0.36, 1)",
                }}
              />
            </div>
          )}
        </>
      )}
    </motion.div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function Today() {
  const { latestPayload, lastSyncAt } = useHealthSync();
  const userId = useMemo(() => getParticipantId(), []);
  const [, navigate] = useLocation();
  const voiceData = useVoiceData();
  const queryClient = useQueryClient();
  const [showBreathe, setShowBreathe] = useState(false);

  // Fetch recent brain history for trend comparison
  const { data: recentHistory } = useQuery<any[]>({
    queryKey: [`/api/brain/history/${userId}?days=7`],
    staleTime: 5 * 60_000,
  });

  // ── Log a feeling state ──
  const [feelingText, setFeelingText] = useState("");
  const [feelingTone, setFeelingTone] = useState<"positive" | "neutral" | "low">("neutral");
  const [feelingSaving, setFeelingSaving] = useState(false);
  const [feelingSaved, setFeelingSaved] = useState(false);

  const submitFeeling = useCallback(async () => {
    if (!feelingText.trim() || feelingSaving) return;
    setFeelingSaving(true);
    const moodScore = feelingTone === "positive" ? 8 : feelingTone === "low" ? 3 : 5;
    const energyLevel = feelingTone === "positive" ? 7 : feelingTone === "low" ? 3 : 5;
    try {
      await apiRequest("POST", "/api/mood", { moodScore, energyLevel, notes: feelingText.trim() });
      setFeelingSaved(true);
      setFeelingText("");
      queryClient.invalidateQueries({ queryKey: ["/api/mood"] });
      setTimeout(() => setFeelingSaved(false), 2000);

      // Also sync to Railway ML backend for session history + retraining
      syncMoodLogToML({
        user_id: userId,
        mood_score: moodScore,
        energy_level: energyLevel,
        notes: feelingText.trim() || undefined,
        emotion: checkin?.emotion,
        valence: checkin?.valence,
      });
    } catch {
      // best-effort
    } finally {
      setFeelingSaving(false);
    }
  }, [feelingText, feelingTone, feelingSaving, queryClient]);

  // Load last emotion check-in from localStorage — re-read on voice update
  const [checkin, setCheckin] = useState<EmotionCheckin | null>(null);
  useEffect(() => {
    function loadCheckin() {
      try {
        const raw = localStorage.getItem("ndw_last_emotion");
        if (raw) {
          const parsed = JSON.parse(raw);
          setCheckin(parsed?.result ?? parsed);
        }
      } catch { /* ignore */ }
    }
    loadCheckin();
    // Listen for voice analysis updates from both event sources
    window.addEventListener("ndw-voice-updated", loadCheckin);
    window.addEventListener("ndw-emotion-update", loadCheckin);
    return () => {
      window.removeEventListener("ndw-voice-updated", loadCheckin);
      window.removeEventListener("ndw-emotion-update", loadCheckin);
    };
  }, []);

  // Fetch food logs for today — API with localStorage fallback
  const { data: foodLogs } = useQuery<FoodLog[]>({
    queryKey: ["/api/food/logs", userId],
    queryFn: async () => {
      try {
        const res = await fetch(resolveUrl(`/api/food/logs/${userId}`));
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data)) return data;
        }
      } catch { /* API unavailable */ }
      try {
        return JSON.parse(localStorage.getItem(`ndw_food_logs_${userId}`) || "[]");
      } catch { return []; }
    },
    retry: false,
  });

  const today = new Date().toISOString().slice(0, 10);
  const todayCalories = useMemo(() => {
    if (!foodLogs) return 0;
    return foodLogs
      .filter((l) => (l.date ?? l.loggedAt ?? "").startsWith(today))
      .reduce((sum, l) => sum + (l.totalCalories ?? 0), 0);
  }, [foodLogs, today]);

  const readiness = useMemo(() => computeReadiness(checkin), [checkin]);
  const aiInsight = useMemo(() => getAIInsight(checkin), [checkin]);

  // Derived values
  const emotion = checkin?.emotion ?? "---";
  const stressVal = checkin?.stress_index ?? 0;
  const focusVal = checkin?.focus_index ?? 0;
  const valenceVal = checkin?.valence ?? 0;
  const topProb = checkin?.probabilities
    ? Math.max(...Object.values(checkin.probabilities))
    : 0;

  // Yesterday comparison — read from localStorage history
  const yesterday = useMemo(() => {
    try {
      const raw = localStorage.getItem("ndw_yesterday_emotion");
      if (raw) return JSON.parse(raw) as { stress_index?: number; focus_index?: number; valence?: number };
    } catch { /* ignore */ }
    return null;
  }, []);

  // Save today's data as "yesterday" at end of day (or when new data arrives)
  useEffect(() => {
    if (!checkin?.stress_index) return;
    try {
      const todayKey = new Date().toISOString().slice(0, 10);
      const savedKey = localStorage.getItem("ndw_yesterday_date");
      if (savedKey !== todayKey) {
        // Move current "today" to "yesterday"
        const prev = localStorage.getItem("ndw_today_emotion");
        if (prev) localStorage.setItem("ndw_yesterday_emotion", prev);
        localStorage.setItem("ndw_yesterday_date", todayKey);
      }
      localStorage.setItem("ndw_today_emotion", JSON.stringify({
        stress_index: stressVal, focus_index: focusVal, valence: checkin?.valence ?? 0,
      }));
    } catch { /* ignore */ }
  }, [checkin, stressVal, focusVal]);

  // Gentle haptic warning when stress is elevated
  useEffect(() => {
    if (stressVal > 0.5) hapticWarning();
  }, [stressVal]);

  const heartRate = latestPayload?.current_heart_rate ?? latestPayload?.resting_heart_rate;
  const steps = latestPayload?.steps_today ?? 0;

  const sleepTotal = latestPayload?.sleep_total_hours ?? 0;
  const sleepEfficiency = latestPayload?.sleep_efficiency ?? 0;

  const calGoal = 2000;
  const calPct = Math.min(100, Math.round((todayCalories / calGoal) * 100));

  const stepsGoal = 10000;
  const stepsPct = Math.min(100, Math.round((steps / stepsGoal) * 100));

  // Heart rate status
  const hrStatus = heartRate
    ? heartRate < 60
      ? { label: "Low", color: "#d4a017" }
      : heartRate < 100
      ? { label: "Normal", color: "#06b6d4" }
      : { label: "Elevated", color: "#e879a8" }
    : { label: "No data", color: "var(--muted-foreground)" };

  // Sleep status
  const sleepStatus = sleepTotal > 0
    ? sleepTotal >= 7
      ? { label: "Good", color: "#06b6d4" }
      : sleepTotal >= 5
      ? { label: "Fair", color: "#d4a017" }
      : { label: "Low", color: "#e879a8" }
    : { label: "No data", color: "var(--muted-foreground)" };

  // Steps status
  const stepsStatus = steps > 0
    ? stepsPct >= 80
      ? { label: "On Track", color: "#06b6d4" }
      : stepsPct >= 40
      ? { label: "Moderate", color: "#d4a017" }
      : { label: "Low", color: "#e879a8" }
    : { label: "No data", color: "var(--muted-foreground)" };

  // Nutrition status
  const nutritionStatus = todayCalories > 0
    ? calPct >= 80
      ? { label: "On Track", color: "#06b6d4" }
      : calPct >= 40
      ? { label: "Moderate", color: "#d4a017" }
      : { label: "Low", color: "#e879a8" }
    : { label: "No data", color: "var(--muted-foreground)" };

  // Mood display
  const moodDisplay = checkin?.emotion
    ? checkin.emotion.charAt(0).toUpperCase() + checkin.emotion.slice(1)
    : "---";
  const moodDotColor = checkin?.emotion ? getMoodDotColor(valenceVal) : "var(--muted-foreground)";
  const moodStatusLabel = checkin?.emotion ? getMoodLabel(valenceVal) : "No data";

  // Stress display
  const stressDisplay = stressVal > 0 ? `${Math.round(stressVal * 100)}%` : "---";
  const stressDotColor = stressVal > 0 ? getStressColor(stressVal) : "var(--muted-foreground)";
  const stressStatusLabel = stressVal > 0 ? getStressLabel(stressVal) : "No data";

  // Focus display
  const focusDisplay = focusVal > 0 ? `${Math.round(focusVal * 100)}%` : "---";
  const focusDotColor = focusVal > 0
    ? focusVal >= 0.7 ? "#06b6d4" : focusVal >= 0.45 ? "#d4a017" : "#e879a8"
    : "var(--muted-foreground)";
  const focusStatusLabel = focusVal > 0 ? getFocusLabel(focusVal) : "No data";

  // Compute deltas vs previous session
  const prevEntry = recentHistory && recentHistory.length >= 2 ? recentHistory[recentHistory.length - 2] : null;
  const stressDelta = prevEntry?.stress != null ? (stressVal - prevEntry.stress) : null;
  const focusDelta = prevEntry?.focus != null ? (focusVal - prevEntry.focus) : null;
  const moodDelta = prevEntry?.valence != null ? ((checkin?.valence ?? 0) - prevEntry.valence) : null;

  // Score splash — show once per session when data exists
  const [showSplash, setShowSplash] = useState(() => {
    if (!checkin?.emotion) return false;
    const shown = sessionStorage.getItem("ndw_splash_shown");
    return !shown;
  });
  // Re-check when checkin loads (it's async from localStorage)
  useEffect(() => {
    if (checkin?.emotion && !sessionStorage.getItem("ndw_splash_shown")) {
      setShowSplash(true);
    }
  }, [checkin]);
  const dismissSplash = useCallback(() => {
    setShowSplash(false);
    sessionStorage.setItem("ndw_splash_shown", "1");
  }, []);

  return (
    <>
      {showSplash && checkin?.emotion && (
        <ScoreSplash
          emotion={checkin.emotion}
          readiness={readiness}
          stress={stressVal}
          focus={focusVal}
          onDismiss={dismissSplash}
        />
      )}
      <motion.main
        initial={pageTransition.initial}
        animate={pageTransition.animate}
        transition={pageTransition.transition}
        style={{
          background: "var(--background)",
          padding: "16px 16px 16px 16px",
          fontFamily: "system-ui, -apple-system, sans-serif",
        }}
      >
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          style={{ maxWidth: 480, margin: "0 auto" }}
        >
          {/* ── 1. Header ── */}
          <motion.div
            variants={itemVariants}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: 24,
            }}
          >
            <div>
              <p
                style={{
                  fontSize: 11,
                  color: "var(--muted-foreground)",
                  margin: "0 0 4px 0",
                  letterSpacing: "0.3px",
                }}
              >
                {formatDate()}
              </p>
              <p
                style={{
                  fontSize: 22,
                  fontWeight: 700,
                  background: "linear-gradient(135deg, #c4b5fd, #e879a8, #7c3aed)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  backgroundClip: "text",
                  margin: 0,
                  lineHeight: 1.3,
                }}
              >
                {(() => {
                  const h = new Date().getHours();
                  const timeGreet = h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
                  const em = checkin?.emotion;
                  if (em === "sad") return "Hey, take it easy today";
                  if (em === "angry") return `${timeGreet} -- breathe`;
                  if (em === "fear") return `You're safe. ${timeGreet}`;
                  return timeGreet;
                })()}
              </p>
            </div>
            <div
              style={{
                width: 38,
                height: 38,
                borderRadius: "50%",
                background: "linear-gradient(135deg, #7c3aed, #e879a8)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 15,
                fontWeight: 700,
                color: "#fff",
                flexShrink: 0,
              }}
            >
              S
            </div>
          </motion.div>

          {/* ── 2. Hero Wellness Circle ── */}
          <motion.div
            variants={itemVariants}
            style={{
              display: "flex",
              justifyContent: "center",
              marginBottom: 24,
              position: "relative",
            }}
          >
            {/* Ambient glow behind gauge */}
            <div
              style={{
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                width: 200,
                height: 200,
                borderRadius: "50%",
                background: "radial-gradient(circle, rgba(124,58,237,0.15) 0%, rgba(232,121,168,0.08) 50%, transparent 70%)",
                filter: "blur(30px)",
                pointerEvents: "none",
              }}
            />
            <WellnessGauge score={readiness} />
          </motion.div>

          {/* ── 2b. Share Wellness Score ── */}
          {readiness > 0 && (
            <motion.div
              variants={itemVariants}
              style={{
                display: "flex",
                justifyContent: "center",
                marginBottom: 20,
                marginTop: -12,
              }}
            >
              <button
                onClick={() => shareWellnessScore(readiness, emotion, aiInsight)}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "8px 18px",
                  borderRadius: 20,
                  border: "1px solid rgba(124, 58, 237, 0.25)",
                  background: "rgba(124, 58, 237, 0.08)",
                  color: "#a78bfa",
                  fontSize: 14,
                  fontWeight: 600,
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                  letterSpacing: "0.3px",
                }}
                onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.96)")}
                onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
                onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
              >
                <Share2 size={13} />
                Share Score
              </button>
            </motion.div>
          )}

          {/* ── 3. Score Row (Mood / Stress / Focus) ── */}
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              gap: 14,
              marginBottom: 20,
            }}
          >
            <ScoreCard
              label="Mood"
              value={moodDisplay}
              statusLabel={moodStatusLabel}
              dotColor={moodDotColor}
              onClick={() => navigate("/mood")}
              delta={moodDelta}
            />
            <ScoreCard
              label="Stress"
              value={stressDisplay}
              statusLabel={stressStatusLabel}
              dotColor={stressDotColor}
              onClick={() => navigate("/stress")}
              delta={stressDelta}
            />
            <ScoreCard
              label="Focus"
              value={focusDisplay}
              statusLabel={focusStatusLabel}
              dotColor={focusDotColor}
              onClick={() => navigate("/focus")}
              delta={focusDelta}
            />
          </motion.div>

          {/* ── 4. AI Insight ── */}
          <motion.div
            variants={itemVariants}
            style={{
              ...bevelCard,
              marginBottom: 20,
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                marginBottom: 8,
              }}
            >
              <Sparkles size={13} color="#7c3aed" />
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  color: "#7c3aed",
                  textTransform: "uppercase" as const,
                  letterSpacing: "0.5px",
                }}
              >
                AI Insight
              </span>
            </div>
            <p
              style={{
                fontSize: 14,
                color: "var(--foreground)",
                margin: 0,
                lineHeight: 1.6,
              }}
            >
              {aiInsight}
            </p>
          </motion.div>

          {/* ── 4b. Log a Feeling ── */}
          <motion.div
            variants={itemVariants}
            style={{
              ...bevelCard,
              marginBottom: 20,
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                marginBottom: 10,
              }}
            >
              <PenLine size={13} color="#7c3aed" />
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  color: "#7c3aed",
                  textTransform: "uppercase" as const,
                  letterSpacing: "0.5px",
                }}
              >
                Log a Feeling
              </span>
            </div>
            <input
              type="text"
              value={feelingText}
              onChange={(e) => setFeelingText(e.target.value)}
              placeholder="What are you feeling? (e.g. proud of myself, grateful...)"
              onKeyDown={(e) => { if (e.key === "Enter") submitFeeling(); }}
              style={{
                width: "100%",
                background: "var(--muted)",
                border: "1px solid var(--border)",
                borderRadius: 12,
                padding: "10px 14px",
                fontSize: 14,
                color: "var(--foreground)",
                outline: "none",
                marginBottom: 10,
                boxSizing: "border-box",
              }}
            />
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ display: "flex", gap: 6 }}>
                {([
                  { key: "positive" as const, Icon: Smile, label: "Positive", color: "#06b6d4" },
                  { key: "neutral" as const, Icon: Minus, label: "Neutral", color: "#94a3b8" },
                  { key: "low" as const, Icon: Frown, label: "Low", color: "#e879a8" },
                ] as const).map(({ key, Icon, label, color }) => (
                  <button
                    key={key}
                    onClick={() => setFeelingTone(key)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 4,
                      padding: "6px 10px",
                      borderRadius: 16,
                      border: feelingTone === key ? `1.5px solid ${color}` : "1px solid var(--border)",
                      background: feelingTone === key ? `${color}15` : "transparent",
                      color: feelingTone === key ? color : "var(--muted-foreground)",
                      fontSize: 11,
                      fontWeight: 500,
                      cursor: "pointer",
                      transition: "all 0.15s ease",
                    }}
                  >
                    <Icon style={{ width: 13, height: 13 }} />
                    {label}
                  </button>
                ))}
              </div>
              <button
                onClick={submitFeeling}
                disabled={!feelingText.trim() || feelingSaving}
                style={{
                  padding: "6px 16px",
                  borderRadius: 16,
                  border: "none",
                  background: feelingText.trim() ? "linear-gradient(135deg, #7c3aed, #e879a8)" : "var(--muted)",
                  color: feelingText.trim() ? "#fff" : "var(--muted-foreground)",
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: feelingText.trim() ? "pointer" : "default",
                  transition: "all 0.15s ease",
                  opacity: feelingSaving ? 0.6 : 1,
                }}
              >
                {feelingSaving ? "Saving..." : "Save"}
              </button>
            </div>
            {feelingSaved && (
              <motion.p
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                style={{
                  fontSize: 12,
                  color: "#06b6d4",
                  fontWeight: 500,
                  marginTop: 8,
                  textAlign: "center",
                }}
              >
                Feeling logged!
              </motion.p>
            )}
          </motion.div>

          {/* ── Stress Warning (conditional) ── */}
          {(checkin?.stress_index ?? 0) > 0.6 && (
            <motion.div
              variants={itemVariants}
              style={{
                ...bevelCard,
                border: "1px solid rgba(232, 121, 168, 0.2)",
                marginBottom: 20,
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 12,
                  marginBottom: 12,
                }}
              >
                <AlertTriangle style={{ width: 22, height: 22, color: "#e879a8", flexShrink: 0 }} />
                <div>
                  <div
                    style={{
                      fontSize: 14,
                      fontWeight: 600,
                      color: "#e879a8",
                      marginBottom: 4,
                    }}
                  >
                    Your stress levels are elevated
                  </div>
                  <div
                    style={{
                      fontSize: 14,
                      color: "var(--muted-foreground)",
                      lineHeight: 1.6,
                    }}
                  >
                    Take a moment to breathe. Try the 4-7-8 breathing technique:
                    <br />
                    <span style={{ fontWeight: 500, color: "var(--foreground)" }}>
                      Inhale 4s &rarr; Hold 7s &rarr; Exhale 8s
                    </span>
                  </div>
                </div>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button
                  onClick={() =>
                    window.open(
                      "https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO",
                      "_blank"
                    )
                  }
                  style={{
                    flex: 1,
                    background: "linear-gradient(135deg, #1DB954, #158a3e)",
                    color: "white",
                    border: "none",
                    borderRadius: 12,
                    padding: "10px 16px",
                    fontSize: 14,
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  Listen to Calm Music
                </button>
                <button
                  onClick={() => setShowBreathe(true)}
                  style={{
                    flex: 1,
                    background: "linear-gradient(135deg, #7c3aed, #6d28d9)",
                    color: "white",
                    border: "none",
                    borderRadius: 12,
                    padding: "10px 16px",
                    fontSize: 14,
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  Breathing Exercise
                </button>
              </div>
            </motion.div>
          )}

          {/* Inline breathing exercise */}
          {showBreathe && <InlineBreathe onClose={() => setShowBreathe(false)} />}

          {/* ── 5. Health Metrics (2x2 grid) ── */}
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 14,
            }}
          >
            {/* Sleep */}
            <HealthMetricCard
              label="Sleep"
              value={sleepTotal > 0 ? sleepTotal.toFixed(1) : "---"}
              unit={sleepTotal > 0 ? "hrs" : ""}
              statusLabel={
                sleepTotal > 0 && sleepEfficiency > 0
                  ? `${Math.round(sleepEfficiency)}% quality`
                  : sleepStatus.label
              }
              dotColor={sleepStatus.color}
              onClick={() => navigate("/sleep-session")}
              barPercent={sleepTotal > 0 ? Math.min(100, (sleepTotal / 8) * 100) : undefined}
              barGradient="linear-gradient(90deg, #7c3aed, #a78bfa)"
              accentColor="#7c3aed"
              emptyIcon={Moon}
              emptyCta="Sync sleep data"
            />

            {/* Heart Rate */}
            <HealthMetricCard
              label="Heart Rate"
              value={heartRate ? `${Math.round(heartRate)}` : "---"}
              unit={heartRate ? "bpm" : ""}
              statusLabel={hrStatus.label}
              dotColor={hrStatus.color}
              onClick={() => navigate("/heart-rate")}
              accentColor="#e879a8"
              emptyIcon={Heart}
              emptyCta="Connect Health to track"
            />

            {/* Steps */}
            <HealthMetricCard
              label="Steps"
              value={steps > 0 ? steps.toLocaleString() : "---"}
              unit={steps > 0 ? `${stepsPct}%` : ""}
              statusLabel={stepsStatus.label}
              dotColor={stepsStatus.color}
              onClick={() => navigate("/steps")}
              barPercent={steps > 0 ? stepsPct : undefined}
              barGradient="linear-gradient(90deg, #06b6d4, #22d3ee)"
              accentColor="#06b6d4"
              emptyIcon={Footprints}
              emptyCta="Sync to see steps"
            />

            {/* Nutrition */}
            <HealthMetricCard
              label="Nutrition"
              value={todayCalories > 0 ? todayCalories.toLocaleString() : "---"}
              unit={todayCalories > 0 ? "kcal" : ""}
              statusLabel={nutritionStatus.label}
              dotColor={nutritionStatus.color}
              onClick={() => navigate("/nutrition")}
              barPercent={todayCalories > 0 ? calPct : undefined}
              barGradient="linear-gradient(90deg, #d4a017, #ea580c)"
              accentColor="#d4a017"
              emptyIcon={UtensilsCrossed}
              emptyCta="Log a meal to start"
            />
          </motion.div>

          {/* ── Quick Listen — Spotify Music Section ── */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.5, ease: "easeOut" }}
            style={{ marginBottom: 20 }}
          >
            <div style={{
              fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
              textTransform: "uppercase" as const, letterSpacing: "0.5px", marginBottom: 8,
            }}>
              Quick Listen
            </div>
            <div style={{
              display: "flex", gap: 14, overflowX: "auto",
              paddingBottom: 4, scrollbarWidth: "none",
              WebkitOverflowScrolling: "touch",
            }}>
              {([
                { icon: Music, color: "#6366f1", title: "Focus", url: "https://open.spotify.com/playlist/37i9dQZF1DWZeKCadgRdKQ" },
                { icon: Wind, color: "#0891b2", title: "Calm", url: "https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO" },
                { icon: CloudMoon, color: "#7c3aed", title: "Sleep", url: "https://open.spotify.com/playlist/37i9dQZF1DWZd79rJ6a7lp" },
                { icon: Dumbbell, color: "#ea580c", title: "Workout", url: "https://open.spotify.com/playlist/37i9dQZF1DX76Wlfdnj7AP" },
                { icon: TreePine, color: "#4ade80", title: "Energize", url: "https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0" },
              ] as const).map((card) => {
                const IconComp = card.icon;
                return (
                <button
                  key={card.title}
                  onClick={() => window.open(card.url, "_blank")}
                  style={{
                    flex: "0 0 auto",
                    width: 90,
                    background: "var(--card)",
                    border: "1px solid var(--border)",
                    borderRadius: 20,
                    padding: "12px 8px",
                    textAlign: "center" as const,
                    cursor: "pointer",
                    transition: "transform 0.2s ease, box-shadow 0.2s ease",
                    boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
                  }}
                  onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.96)")}
                  onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
                  onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
                >
                  <div style={{ display: "flex", justifyContent: "center", marginBottom: 4 }}>
                    <IconComp style={{ width: 24, height: 24, color: card.color }} />
                  </div>
                  <div style={{ fontSize: 11, fontWeight: 600, color: "var(--foreground)" }}>{card.title}</div>
                </button>
                );
              })}
            </div>
          </motion.div>

        </motion.div>
      </motion.main>
    </>
  );
}
