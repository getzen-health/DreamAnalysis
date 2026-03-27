import { useState, useEffect, useMemo, useCallback } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useLocation } from "wouter";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { motion } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { resolveUrl, apiRequest } from "@/lib/queryClient";
import { getParticipantId } from "@/lib/participant";
import { useHealthSync } from "@/hooks/use-health-sync";
import { Sparkles, Moon, Heart, Footprints, UtensilsCrossed, Share2, Music, Wind, CloudMoon, Dumbbell, TreePine, AlertTriangle, Smile, Minus, Frown, PenLine, TrendingUp, TrendingDown, Check, Pencil, Clock, Brain, Mic, Activity } from "lucide-react";
import { ScoreSplash } from "@/components/score-splash";
import { hapticWarning } from "@/lib/haptics";
import { useVoiceData, type VoiceCheckinData } from "@/hooks/use-voice-data";
import { InlineBreathe } from "@/components/inline-breathe";
import { syncMoodLogToML, getTodayTotals, type StoredEmotionReading } from "@/lib/ml-api";
import { BrainCoachCard } from "@/components/brain-coach-card";
import { InnerScoreCard } from "@/components/inner-score-card";
import { computeScore, computeNarrative, type ScoreInputs } from "@/lib/inner-score";
import { EEGWeekCompareCard } from "@/components/eeg-week-compare-card";
import { calculateFoodScore } from "@/lib/food-score";
import { getStoredChronotype, getBaselineAdjustment } from "@/lib/chronotype";
import { useMultimodalEmotion } from "@/hooks/use-multimodal-emotion";
import { useFusedState } from "@/hooks/use-fused-state";
import { ConfidenceMeter } from "@/components/confidence-meter";
import { calculateEmotionConfidence } from "@/lib/confidence-calculator";
import { getCycleData, getFoodLogs as sbGetFoodLogs, sbGetSetting, sbSaveGeneric, sbSaveSetting } from "../lib/supabase-store";
import { fetchWeather, buildMoodContext, type WeatherData, type WeatherMoodContext } from "@/lib/weather-context";
import { getCurrentCyclePhase, getCyclePhaseContext, type CyclePhaseContext } from "@/lib/cycle-phase-adjustment";
import { Cloud, CloudRain, Sun, Snowflake, CloudLightning, CloudFog, CloudSun, HelpCircle } from "lucide-react";
import { recordCorrection } from "@/lib/feedback-sync";
import { updateModalityAccuracy } from "@/lib/multimodal-fusion";
import { MoodPicker } from "@/components/mood-picker";
import { RecoveryInterventions } from "@/components/recovery-interventions";
import { EnergyTimeline } from "@/components/energy-timeline";
import { useScores } from "@/hooks/use-scores";

// ── Types ──────────────────────────────────────────────────────────────────

interface EmotionCheckin {
  emotion?: string;
  probabilities?: Record<string, number>;
  valence?: number;
  arousal?: number;
  confidence?: number;
  stress_index?: number;
  focus_index?: number;
  relaxation_index?: number;
  model_type?: string;
  timestamp?: number;
}

interface FoodLog {
  id?: string;
  totalCalories?: number;
  date?: string;
  loggedAt?: string;
}

interface MoodLogEntry {
  id?: string;
  moodScore?: string | number;
  energyLevel?: string | number;
  notes?: string;
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
  if (stress < 0.3) return "var(--emotion-calm-to, #06b6d4)";
  if (stress < 0.6) return "var(--warning, #d4a017)";
  return "var(--secondary, #e879a8)";
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
  if (valence > 0.3) return "var(--emotion-calm-to, #06b6d4)";
  if (valence > -0.1) return "var(--warning, #d4a017)";
  return "var(--secondary, #e879a8)";
}

function getAIInsight(checkin: EmotionCheckin | null): { headline: string; body: string; action: string } {
  if (!checkin) return {
    headline: "No reading yet today",
    body: "Tap the mic to do a voice check-in. The more you log, the smarter your insights get.",
    action: "Start your first check-in",
  };
  const stress = checkin.stress_index ?? 0.5;
  const focus = checkin.focus_index ?? 0.5;
  const valence = checkin.valence ?? 0;
  const emotion = checkin.emotion ?? "neutral";

  // Read personal baseline from emotion history
  let baselineStress = 0.5;
  let baselineFocus = 0.5;
  let baselineN = 0;
  try {
    const raw = localStorage.getItem("ndw_emotion_history");
    if (raw) {
      const hist = JSON.parse(raw) as Array<{ stress: number; focus: number }>;
      if (hist.length >= 3) {
        baselineStress = hist.reduce((s: number, e) => s + e.stress, 0) / hist.length;
        baselineFocus = hist.reduce((s: number, e) => s + e.focus, 0) / hist.length;
        baselineN = hist.length;
      }
    }
  } catch { /* ignore */ }

  const hasBaseline = baselineN >= 5;
  const stressVsBaseline = hasBaseline ? stress - baselineStress : 0;
  const focusVsBaseline = hasBaseline ? focus - baselineFocus : 0;

  // State: flow (low stress + high focus)
  if (stress < 0.3 && focus > 0.6) {
    return {
      headline: "You're primed for deep work",
      body: hasBaseline
        ? `Your stress (${Math.round(stress*100)}%) is ${Math.round(Math.abs(stressVsBaseline)*100)}pts below your usual — and focus is elevated. This combination is rare and won't last forever.`
        : "Low stress and high focus are the neural signature of a flow state. Your prefrontal cortex is firing at full capacity.",
      action: "Start your hardest task now — silence notifications and work for 45 minutes.",
    };
  }
  // State: high stress
  if (stress > 0.65) {
    return {
      headline: hasBaseline && stressVsBaseline > 0.1
        ? `Stress is ${Math.round(stressVsBaseline*100)}pts above your average`
        : "Your stress is elevated",
      body: "High-beta activity is suppressing prefrontal function — you're spending cognitive resources on threat detection instead of creative thinking.",
      action: "Try 4-7-8 breathing (4s inhale, 7s hold, 8s exhale). Two cycles measurably lower cortisol.",
    };
  }
  // State: positive
  if (valence > 0.3 && stress < 0.4) {
    return {
      headline: "Positive emotional state detected",
      body: hasBaseline
        ? `Your mood is ${Math.round(valence*100)}% positive — above your usual baseline. Positive affect enhances cognitive flexibility and memory consolidation.`
        : "Positive mood enhances working memory, creative thinking, and social engagement. A window worth using.",
      action: "Great time for learning, collaborative tasks, or connecting with someone important.",
    };
  }
  // State: low mood
  if (emotion === "sad" || valence < -0.2) {
    return {
      headline: "Your mood is leaning low today",
      body: "Negative valence narrows attentional focus — you may notice rumination or difficulty with complex decisions. This is your nervous system asking for recovery.",
      action: "5 minutes of brisk walking or natural light exposure resets the cortisol cycle faster than any other intervention.",
    };
  }
  // State: low focus
  if (focus < 0.35) {
    return {
      headline: hasBaseline && focusVsBaseline < -0.1
        ? `Focus is ${Math.round(Math.abs(focusVsBaseline)*100)}pts below your usual`
        : "Focus is diffuse right now",
      body: "Low beta activity and elevated alpha suggest mind-wandering mode. This often follows sustained cognitive effort or insufficient sleep last night.",
      action: "5-min walk before your next task, or a 25-min Pomodoro to re-engage attention circuits.",
    };
  }
  // Default: balanced
  return {
    headline: "Your brain state is balanced",
    body: hasBaseline
      ? `Stress at ${Math.round(stress*100)}% (your avg: ${Math.round(baselineStress*100)}%) and focus at ${Math.round(focus*100)}% (avg: ${Math.round(baselineFocus*100)}%) — you're tracking close to your baseline.`
      : "A balanced state across stress, focus, and mood — your brain is in a receptive, general-purpose mode.",
    action: "Good conditions for planning, reflection, or taking in new information.",
  };
}

function getMoodLogTone(moodScore: number): { label: string; color: string } {
  if (moodScore >= 7) return { label: "Positive", color: "var(--emotion-calm-to, #06b6d4)" };
  if (moodScore >= 4) return { label: "Neutral", color: "var(--muted-foreground, #94a3b8)" };
  return { label: "Low", color: "var(--secondary, #e879a8)" };
}

function formatTime(dateStr: string): string {
  try {
    const d = new Date(dateStr);
    return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  } catch {
    return "";
  }
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

  // Export as blob
  const dataUrl = canvas.toDataURL("image/png");
  let blob: Blob | null = null;
  try {
    blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob((b) => resolve(b), "image/png");
    });
  } catch { /* toBlob not supported */ }

  // Try native share (Capacitor)
  try {
    const { Capacitor } = await import("@capacitor/core");
    if (Capacitor.isNativePlatform()) {
      // Use Capacitor's native share via a temporary file
      const { Filesystem, Directory } = await import("@capacitor/filesystem" as string);
      const base64Data = dataUrl.split(",")[1];
      const saved = await Filesystem.writeFile({
        path: "antarai-wellness.png",
        data: base64Data,
        directory: Directory.Cache,
      });
      const { Share } = await import("@capacitor/share" as string);
      await Share.share({
        title: `My Wellness Score: ${score}`,
        text: `My wellness score today is ${score}/100. ${insight}`,
        url: saved.uri,
      });
      return;
    }
  } catch {
    // Capacitor share not available — fall through
  }

  // Try Web Share API
  if (blob) {
    try {
      const file = new File([blob], "antarai-wellness.png", { type: "image/png" });
      if (navigator.share && navigator.canShare?.({ files: [file] })) {
        await navigator.share({
          title: `My Wellness Score: ${score}`,
          text: `My wellness score today is ${score}/100. ${insight}`,
          files: [file],
        });
        return;
      }
    } catch { /* share cancelled or failed */ }
  }

  // Fallback: open image in new tab / download
  try {
    const w = window.open();
    if (w) {
      w.document.write(`<img src="${dataUrl}" style="max-width:100%"/>`);
      w.document.title = "Wellness Score";
      return;
    }
  } catch { /* popup blocked */ }

  // Last resort: download link
  try {
    const link = document.createElement("a");
    link.download = "antarai-wellness.png";
    link.href = dataUrl;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  } catch { /* all methods failed */ }
}

// ── Animation variants ──────────────────────────────────────────────────

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.07,
      delayChildren: 0.05,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 14, scale: 0.98 },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      duration: 0.45,
      ease: [0.22, 1, 0.36, 1],
    },
  },
};

// ── Card classes ─────────────────────────────────────────────────────────
// All cards now use premium style: `rounded-[14px] bg-card border border-border`.
// No glass-card or premiumCard style objects — clean Tailwind only.

// ── Score Circle (premium style) ─────────────────────────────────────────
// Reusable SVG arc circle: 120px diameter, 270-degree sweep, gradient stroke.
// Big number in center (32px bold), label below (12px, muted).

function ScoreCircle({
  score,
  label,
  colorFrom,
  colorTo,
  id,
}: {
  score: number;
  label: string;
  colorFrom: string;
  colorTo: string;
  id: string;
}) {
  const size = 120;
  const strokeWidth = 8;
  const r = (size - strokeWidth) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * r;
  const arcLength = (270 / 360) * circumference;
  const filled = (score / 100) * arcLength;
  const gradientId = `scoreGrad-${id}`;

  return (
    <div className="flex flex-col items-center gap-1.5">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor={colorFrom} />
            <stop offset="100%" stopColor={colorTo} />
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
          opacity={0.4}
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
          className="transition-all duration-[1400ms] ease-[cubic-bezier(0.22,1,0.36,1)]"
        />
        {/* Score number */}
        <text
          x={cx}
          y={cy + 2}
          textAnchor="middle"
          dominantBaseline="central"
          fill="var(--foreground)"
          fontSize={32}
          fontWeight={700}
          className="font-sans"
        >
          {score}
        </text>
      </svg>
      <span className="text-xs text-muted-foreground font-medium">{label}</span>
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
      className={`rounded-[14px] bg-card border border-border p-4 flex flex-col items-center gap-1.5 transition-colors ${onClick ? "cursor-pointer hover:border-foreground/15" : "cursor-default"}`}
    >
      <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
        {label}
      </span>
      <span className="text-2xl font-bold text-foreground leading-none">
        {value}
      </span>
      <div className="flex items-center gap-1.5">
        <div
          className="w-[7px] h-[7px] rounded-full shrink-0"
          style={{ background: dotColor }}
        />
        <span className="text-[11px] text-muted-foreground">
          {statusLabel}
        </span>
      </div>
      {delta != null && Math.abs(delta) > 0.02 && (
        <div className="flex items-center gap-1 mt-0.5">
          {delta > 0 ? (
            <TrendingUp className={`w-3 h-3 ${label === "Stress" ? "text-rose-400" : "text-cyan-400"}`} />
          ) : (
            <TrendingDown className={`w-3 h-3 ${label === "Stress" ? "text-cyan-400" : "text-rose-400"}`} />
          )}
          <span className="text-[10px] text-muted-foreground">
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
      className={`rounded-[14px] bg-card border border-border p-4 flex flex-col gap-3 transition-colors ${onClick ? "cursor-pointer hover:border-foreground/15" : "cursor-default"}`}
      style={isEmpty && accentColor ? {
        borderLeftWidth: 3,
        borderLeftColor: accentColor,
      } : undefined}
    >
      <div className="flex justify-between items-start">
        <div className="flex items-center gap-1.5">
          {isEmpty && EmptyIcon && (
            <EmptyIcon className="w-3.5 h-3.5" style={{ color: accentColor || "var(--muted-foreground)" }} />
          )}
          <span
            className="text-[11px] font-medium uppercase tracking-wider"
            style={{ color: isEmpty && accentColor ? accentColor : "var(--muted-foreground)" }}
          >
            {label}
          </span>
        </div>
        {!isEmpty && (
          <div className="flex items-center gap-1.5">
            <div
              className="w-[7px] h-[7px] rounded-full shrink-0"
              style={{ background: dotColor }}
            />
            <span className="text-[11px] text-muted-foreground">{statusLabel}</span>
          </div>
        )}
      </div>
      {isEmpty ? (
        <div className="flex flex-col gap-1">
          <span className="text-sm font-medium text-muted-foreground leading-snug">
            {emptyCta || "No data yet"}
          </span>
        </div>
      ) : (
        <>
          <div className="flex items-baseline gap-1">
            <span className="text-[28px] font-bold text-foreground leading-none">
              {value}
            </span>
            <span className="text-sm text-muted-foreground">{unit}</span>
          </div>
          {barPercent !== undefined && (
            <div className="progress-thick">
              <div
                className="progress-thick-fill"
                style={{
                  width: `${clamp(barPercent, 0, 100)}%`,
                  background: barGradient || "linear-gradient(90deg, var(--primary), var(--secondary))",
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
  const { scores: userScores } = useScores(userId);
  const [, navigate] = useLocation();
  const voiceData = useVoiceData();
  const { emotion: fusedEmotion, correctEmotion: correctFusedEmotion } = useMultimodalEmotion();
  const { fusedState } = useFusedState(); // Data fusion bus: auto-updates from EEG/voice/health
  const queryClient = useQueryClient();
  const [showBreathe, setShowBreathe] = useState(false);

  // ── Weather context (Issue #508) ──
  const [weatherCtx, setWeatherCtx] = useState<WeatherMoodContext | null>(null);
  useEffect(() => {
    fetchWeather().then((data) => {
      if (data) setWeatherCtx(buildMoodContext(data));
    }).catch(() => {});
  }, []);

  // ── Cycle phase context (Issue #498) ──
  const [cycleCtx, setCycleCtx] = useState<CyclePhaseContext | null>(null);
  useEffect(() => {
    getCycleData(userId).then((data) => {
      if (data?.last_period_start) {
        const phase = getCurrentCyclePhase(data.last_period_start, data.cycle_length ?? 28);
        if (phase) {
          const ctx = getCyclePhaseContext(phase.phase);
          setCycleCtx({ ...ctx, dayInCycle: phase.dayInCycle });
        }
      }
    }).catch(() => {});
  }, [userId]);

  // Fetch today's EEG totals for Brain Coach
  const { data: brainTotals } = useQuery({
    queryKey: ["brain-totals-today", userId],
    queryFn: () => getTodayTotals(userId!),
    enabled: !!userId,
    staleTime: 5 * 60_000,
  });

  // Fetch recent brain history for trend comparison
  const { data: recentHistory } = useQuery<StoredEmotionReading[]>({
    queryKey: [`/api/brain/history/${userId}?days=30`],
    staleTime: 5 * 60_000,
  });

  // Fetch mood log history for "Log a Feeling" display
  const { data: moodLogHistory } = useQuery<MoodLogEntry[]>({
    queryKey: ["/api/mood", userId],
    queryFn: async () => {
      try {
        const res = await fetch(resolveUrl(`/api/mood/${userId}?days=1`));
        if (res.ok) return await res.json();
      } catch { /* API unavailable */ }
      return [];
    },
    staleTime: 30_000,
  });

  // ── Log a feeling state ──
  const [feelingText, setFeelingText] = useState("");
  const [feelingSaving, setFeelingSaving] = useState(false);
  const [feelingSaved, setFeelingSaved] = useState(false);

  // Keyword-based mapping from text to valence/stress/focus
  const FEELING_KEYWORDS: Record<string, { valence: number; stress: number; focus: number }> = {
    happy: { valence: 0.7, stress: 0.15, focus: 0.6 },
    calm: { valence: 0.4, stress: 0.1, focus: 0.5 },
    focused: { valence: 0.3, stress: 0.3, focus: 0.85 },
    excited: { valence: 0.8, stress: 0.3, focus: 0.7 },
    grateful: { valence: 0.6, stress: 0.1, focus: 0.5 },
    frustrated: { valence: -0.4, stress: 0.7, focus: 0.4 },
    anxious: { valence: -0.3, stress: 0.8, focus: 0.3 },
    sad: { valence: -0.5, stress: 0.5, focus: 0.3 },
    tired: { valence: -0.1, stress: 0.4, focus: 0.2 },
    neutral: { valence: 0, stress: 0.35, focus: 0.45 },
    proud: { valence: 0.6, stress: 0.2, focus: 0.6 },
    disappointed: { valence: -0.4, stress: 0.5, focus: 0.3 },
    angry: { valence: -0.5, stress: 0.85, focus: 0.5 },
    loved: { valence: 0.8, stress: 0.05, focus: 0.4 },
    lonely: { valence: -0.3, stress: 0.4, focus: 0.25 },
    inspired: { valence: 0.7, stress: 0.15, focus: 0.75 },
    stressed: { valence: -0.3, stress: 0.85, focus: 0.35 },
    hopeful: { valence: 0.5, stress: 0.2, focus: 0.55 },
    overwhelmed: { valence: -0.4, stress: 0.9, focus: 0.2 },
    peaceful: { valence: 0.5, stress: 0.05, focus: 0.45 },
    nervous: { valence: -0.2, stress: 0.7, focus: 0.3 },
    content: { valence: 0.4, stress: 0.15, focus: 0.5 },
    bored: { valence: -0.1, stress: 0.2, focus: 0.15 },
    motivated: { valence: 0.6, stress: 0.25, focus: 0.8 },
    scared: { valence: -0.4, stress: 0.8, focus: 0.35 },
    joy: { valence: 0.8, stress: 0.1, focus: 0.6 },
    good: { valence: 0.4, stress: 0.2, focus: 0.5 },
    bad: { valence: -0.3, stress: 0.5, focus: 0.3 },
    great: { valence: 0.7, stress: 0.1, focus: 0.6 },
    terrible: { valence: -0.6, stress: 0.7, focus: 0.2 },
    ok: { valence: 0.1, stress: 0.3, focus: 0.4 },
    fine: { valence: 0.1, stress: 0.3, focus: 0.4 },
    meh: { valence: -0.05, stress: 0.3, focus: 0.3 },
  };

  function detectFeelingValues(text: string): { valence: number; stress: number; focus: number } {
    const lower = text.toLowerCase();
    for (const [keyword, values] of Object.entries(FEELING_KEYWORDS)) {
      if (lower.includes(keyword)) return values;
    }
    // Default neutral values if no keyword matched
    return { valence: 0, stress: 0.35, focus: 0.45 };
  }

  const submitFeeling = useCallback(async () => {
    const trimmed = feelingText.trim();
    if (!trimmed || feelingSaving) return;
    setFeelingSaving(true);
    const detected = detectFeelingValues(trimmed);
    const moodScore = Math.round((detected.valence + 1) * 5);
    const energyLevel = Math.round((1 - detected.stress) * 10);
    const emotionLabel = trimmed;
    try {
      await apiRequest("POST", "/api/mood", { moodScore, energyLevel, notes: trimmed });

      // Save as emotion data so all pages update
      const now = Date.now();
      try {
        localStorage.setItem("ndw_last_emotion", JSON.stringify({
          result: {
            emotion: emotionLabel,
            valence: detected.valence,
            arousal: 0.5,
            stress_index: detected.stress,
            focus_index: detected.focus,
            confidence: 0.9,
            model_type: "manual",
            timestamp: now / 1000,
          },
          timestamp: now,
        }));
        window.dispatchEvent(new CustomEvent("ndw-emotion-update"));
        // Sync to Supabase
        import("@/lib/supabase-store").then(({ saveEmotionHistory }) => {
          saveEmotionHistory(getParticipantId(), {
            stress: detected.stress, focus: detected.focus,
            mood: detected.valence, source: "manual",
            dominantEmotion: emotionLabel,
          }).catch(() => {});
        });
      } catch { /* ok */ }
      setFeelingSaved(true);
      setFeelingText("");
      queryClient.invalidateQueries({ queryKey: ["/api/mood"] });
      // Keep saved confirmation visible longer
      setTimeout(() => setFeelingSaved(false), 10000);
      // Manual feeling stays as the active emotion until next check-in
      try { localStorage.setItem("ndw_manual_emotion_until", String(Date.now() + 86400000)); } catch { /* ok */ }

      // Sync to Railway ML backend for session history
      syncMoodLogToML({
        user_id: userId,
        mood_score: moodScore,
        energy_level: energyLevel,
        notes: trimmed,
        emotion: emotionLabel,
        valence: detected.valence,
      });

      // Send as MODEL CORRECTION — teaches the model that this user's current
      // brain/voice state = this emotion. Used for personalization.
      import("@/lib/ml-api").then(({ submitFeedback }) => {
        // Get current EEG prediction to send as "predicted" vs user's "correct" label
        const currentPrediction = checkin?.emotion ?? "neutral";
        submitFeedback(null, currentPrediction, emotionLabel, userId)
          .catch(() => {});
      }).catch(() => {});
    } catch {
      // best-effort
    } finally {
      setFeelingSaving(false);
    }
  }, [feelingText, feelingSaving, queryClient]);

  // Load last emotion check-in from localStorage — re-read on voice update
  const [checkin, setCheckin] = useState<EmotionCheckin | null>(null);
  const [checkinTimestamp, setCheckinTimestamp] = useState<number | null>(null);
  useEffect(() => {
    function loadCheckin() {
      try {
        const raw = sbGetSetting("ndw_last_emotion");
        if (raw) {
          const parsed = JSON.parse(raw);
          const data = parsed?.result ?? parsed;
          setCheckin(data);
          // Timestamp: wrapper timestamp (ms) or result timestamp (seconds)
          const ts = parsed?.timestamp ?? (data?.timestamp ? data.timestamp * 1000 : null);
          setCheckinTimestamp(ts && ts > 1e12 ? ts : ts ? ts * 1000 : null);
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

  // Emotion correction state (Task 3: confirm or correct voice-detected emotion)
  const [emotionFeedback, setEmotionFeedback] = useState<"ask" | "correcting" | "confirmed" | "corrected">("ask");
  const [correctedEmotion, setCorrectedEmotion] = useState<string | null>(null);

  // Reset feedback state when a new voice analysis arrives
  useEffect(() => {
    setEmotionFeedback("ask");
    setCorrectedEmotion(null);
  }, [checkinTimestamp]);

  const confirmVoiceEmotion = useCallback(() => {
    setEmotionFeedback("confirmed");
    // Update per-modality accuracy tracking (prediction was correct)
    updateModalityAccuracy("voice", true);
    if (checkin?.emotion) {
      fetch(resolveUrl(`/api/readings/${userId}/correct-latest`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ correctedEmotion: checkin.emotion }),
      }).catch(() => {});
      // Persist confirmed emotion to Supabase (prediction matched reality)
      recordCorrection({
        userId,
        predictedEmotion: checkin.emotion,
        correctedEmotion: checkin.emotion,
        source: "voice",
        confidence: checkin.confidence,
        features: checkin.valence != null ? {
          voice_valence: checkin.valence,
          voice_arousal: checkin.arousal ?? 0.5,
          voice_stress: checkin.stress_index ?? 0,
        } : undefined,
      }).catch(() => {});
    }
  }, [userId, checkin]);

  const submitEmotionCorrection = useCallback((emotion: string) => {
    setCorrectedEmotion(emotion);
    setEmotionFeedback("corrected");
    // Update per-modality accuracy tracking
    updateModalityAccuracy("voice", emotion === checkin?.emotion);
    fetch(resolveUrl(`/api/readings/${userId}/correct-latest`), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ correctedEmotion: emotion }),
    }).catch((err) => console.error("Failed to save emotion correction:", err));
    // Persist correction to Supabase + ML backend with voice feature vectors
    recordCorrection({
      userId,
      predictedEmotion: checkin?.emotion ?? "unknown",
      correctedEmotion: emotion,
      source: "voice",
      confidence: checkin?.confidence,
      features: checkin?.valence != null ? {
        voice_valence: checkin.valence ?? 0,
        voice_arousal: checkin.arousal ?? 0.5,
        voice_stress: checkin.stress_index ?? 0,
      } : undefined,
    }).catch(() => {});
    // Record fusion feedback — adapts per-modality weights for this user
    correctFusedEmotion(emotion);
    // Update local emotion display so the card reflects the correction immediately
    try {
      const raw = sbGetSetting("ndw_last_emotion");
      if (raw) {
        const parsed = JSON.parse(raw);
        const data = parsed?.result ?? parsed;
        data.emotion = emotion;
        if (parsed?.result) parsed.result = data;
        sbSaveGeneric("ndw_last_emotion", parsed?.result ? parsed : { result: data, timestamp: Date.now() });
        window.dispatchEvent(new CustomEvent("ndw-emotion-update"));
      }
    } catch { /* ignore */ }
  }, [userId, correctFusedEmotion]);

  // Fetch food logs — merge API + Supabase + localStorage (same as nutrition page)
  const { data: foodLogs } = useQuery<FoodLog[]>({
    queryKey: ["/api/food/logs", userId],
    queryFn: async () => {
      let apiLogs: FoodLog[] = [];
      try {
        const res = await fetch(resolveUrl(`/api/food/logs/${userId}`));
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data)) apiLogs = data;
        }
      } catch { /* API unavailable */ }
      // Supabase
      let sbLogs: FoodLog[] = [];
      try {
        sbLogs = await sbGetFoodLogs(userId) ?? [];
      } catch { /* ok */ }
      // localStorage
      let localLogs: FoodLog[] = [];
      try {
        const raw = localStorage.getItem(`ndw_food_logs_${userId}`);
        if (raw) localLogs = JSON.parse(raw);
      } catch { /* ok */ }
      // Merge + deduplicate
      const all = [...apiLogs, ...sbLogs, ...localLogs];
      const seen = new Set<string>();
      return all.filter((l) => {
        const key = l.id ?? l.loggedAt ?? String(Math.random());
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      }).sort((a, b) => new Date(b.loggedAt ?? 0).getTime() - new Date(a.loggedAt ?? 0).getTime());
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

  // ── Inner Score computation ──────────────────────────────────────────────
  const innerScore = useMemo(() => {
    const inputs: ScoreInputs = {
      stress: brainTotals?.avgStress ?? null,
      valence: brainTotals?.avgValence ?? null,
      sleepQuality: latestPayload?.sleep_efficiency ?? null,
      hrvTrend: latestPayload?.hrv_sdnn ?? null,
      activity: latestPayload?.steps_today ? Math.min(100, Math.round((latestPayload.steps_today / 10000) * 100)) : null,
    };
    const result = computeScore(inputs);
    const narrative = result.score != null
      ? computeNarrative(result.factors, result.score, null)
      : "";
    return { ...result, narrative };
  }, [brainTotals, latestPayload]);

  // Persist inner score to DB (fire-and-forget)
  useEffect(() => {
    if (innerScore.score != null && userId) {
      fetch(`/api/inner-score/${userId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ score: innerScore.score, tier: innerScore.tier, factors: innerScore.factors, narrative: innerScore.narrative }),
      }).catch(() => {});
    }
  }, [innerScore.score, innerScore.tier, userId]);

  // 7-day daily-average trend chart data (for "This Week" mini chart)
  const weekTrendData = useMemo(() => {
    let history: Array<{ stress: number; focus: number; happiness: number; timestamp: string }> = [];
    // Use already-fetched recentHistory from TanStack Query (DB data) when available
    if (recentHistory && recentHistory.length > 0) {
      history = recentHistory.map(r => ({
        stress: r.stress ?? 0.5,
        focus: r.focus ?? 0.5,
        happiness: r.happiness ?? 0.5,
        timestamp: (r as any).timestamp ?? (r as any).created_at ?? new Date().toISOString(),
      }));
    } else {
      // Fallback to localStorage if API data not available
      try {
        const raw = localStorage.getItem("ndw_emotion_history");
        if (raw) history = JSON.parse(raw);
      } catch { /* ignore */ }
    }
    const days = Array.from({ length: 7 }, (_, i) => {
      const d = new Date();
      d.setDate(d.getDate() - (6 - i));
      return { label: d.toLocaleDateString([], { weekday: "short" }), date: d.toDateString() };
    });
    const baselineAll = history.length >= 3
      ? {
          stress: history.reduce((s, e) => s + e.stress, 0) / history.length,
          focus: history.reduce((s, e) => s + e.focus, 0) / history.length,
        }
      : null;
    const chartRows = days.map(day => {
      const entries = history.filter(e => new Date(e.timestamp).toDateString() === day.date);
      if (entries.length === 0) return { label: day.label, stress: null, focus: null };
      const n = entries.length;
      return {
        label: day.label,
        stress: Math.round(entries.reduce((s, e) => s + e.stress, 0) / n * 100),
        focus: Math.round(entries.reduce((s, e) => s + e.focus, 0) / n * 100),
      };
    });
    return { chartRows, baseline: baselineAll };
  }, [recentHistory]);

  // Map scores for recovery interventions & energy timeline
  const scores = useMemo(() => ({
    recovery: userScores?.recoveryScore ?? undefined,
    strain: userScores?.strainScore ?? undefined,
    stress: userScores?.stressScore ?? undefined,
  }), [userScores]);
  const hrvTrend: "up" | "down" | "stable" | undefined = undefined; // TODO: derive from HRV history

  // Derived values (raw from model)
  const emotion = checkin?.emotion ?? "---";
  const rawStress = checkin?.stress_index ?? 0;
  const rawFocus = checkin?.focus_index ?? 0;
  const rawValence = checkin?.valence ?? 0;
  const topProb = checkin?.probabilities
    ? Math.max(...Object.values(checkin.probabilities))
    : 0;

  // Holistic confidence for the current emotion reading
  const emotionConfidence = useMemo(() => {
    if (!checkin) return null;
    return calculateEmotionConfidence({
      modelConfidence: topProb > 0 ? topProb : 0.5,
      agreementScore: fusedEmotion?.agreement,
    });
  }, [checkin, topProb, fusedEmotion]);

  // Chronotype-aware baseline adjustment (display-level only)
  const chronotypeAdj = useMemo(() => {
    const ct = getStoredChronotype();
    if (!ct) return null;
    return getBaselineAdjustment(ct.category, new Date().getHours());
  }, [checkin]); // recompute when checkin changes (captures current hour)

  const stressVal = chronotypeAdj
    ? clamp(rawStress - chronotypeAdj.arousalOffset, 0, 1)
    : rawStress;
  const focusVal = rawFocus; // focus not adjusted by chronotype
  const valenceVal = chronotypeAdj
    ? clamp(rawValence + chronotypeAdj.valenceOffset, -1, 1)
    : rawValence;

  // Yesterday comparison — read from localStorage history
  const yesterday = useMemo(() => {
    try {
      const raw = sbGetSetting("ndw_yesterday_emotion");
      if (raw) return JSON.parse(raw) as { stress_index?: number; focus_index?: number; valence?: number };
    } catch { /* ignore */ }
    return null;
  }, []);

  // Save today's data as "yesterday" at end of day (or when new data arrives)
  useEffect(() => {
    if (!checkin?.stress_index) return;
    try {
      const todayKey = new Date().toISOString().slice(0, 10);
      const savedKey = sbGetSetting("ndw_yesterday_date");
      if (savedKey !== todayKey) {
        // Move current "today" to "yesterday"
        const prev = sbGetSetting("ndw_today_emotion");
        if (prev) sbSaveSetting("ndw_yesterday_emotion", prev);
        sbSaveSetting("ndw_yesterday_date", todayKey);
      }
      sbSaveGeneric("ndw_today_emotion", {
        stress_index: stressVal, focus_index: focusVal, valence: checkin?.valence ?? 0,
      });
    } catch { /* ignore */ }
  }, [checkin, stressVal, focusVal]);

  // Gentle haptic warning when stress is elevated
  useEffect(() => {
    if (stressVal > 0.5) hapticWarning();
  }, [stressVal]);

  const heartRate = latestPayload?.current_heart_rate ?? latestPayload?.resting_heart_rate;
  const steps = latestPayload?.steps_today ?? 0;
  const hrvSdnn = latestPayload?.hrv_sdnn;
  const spo2 = latestPayload?.spo2;
  const respiratoryRate = latestPayload?.respiratory_rate;

  const sleepTotal = latestPayload?.sleep_total_hours ?? 0;
  const sleepEfficiency = latestPayload?.sleep_efficiency ?? 0;

  const calGoal = 2000;
  const calPct = Math.min(100, Math.round((todayCalories / calGoal) * 100));

  const stepsGoal = 10000;
  const stepsPct = Math.min(100, Math.round((steps / stepsGoal) * 100));

  // Heart rate status
  const hrStatus = heartRate
    ? heartRate < 60
      ? { label: "Low", color: "var(--warning, #d4a017)" }
      : heartRate < 100
      ? { label: "Normal", color: "var(--emotion-calm-to, #06b6d4)" }
      : { label: "Elevated", color: "var(--secondary, #e879a8)" }
    : { label: "No data", color: "var(--muted-foreground)" };

  // Sleep status
  const sleepStatus = sleepTotal > 0
    ? sleepTotal >= 7
      ? { label: "Good", color: "var(--emotion-calm-to, #06b6d4)" }
      : sleepTotal >= 5
      ? { label: "Fair", color: "var(--warning, #d4a017)" }
      : { label: "Low", color: "var(--secondary, #e879a8)" }
    : { label: "No data", color: "var(--muted-foreground)" };

  // Steps status
  const stepsStatus = steps > 0
    ? stepsPct >= 80
      ? { label: "On Track", color: "var(--emotion-calm-to, #06b6d4)" }
      : stepsPct >= 40
      ? { label: "Moderate", color: "var(--warning, #d4a017)" }
      : { label: "Low", color: "var(--secondary, #e879a8)" }
    : { label: "No data", color: "var(--muted-foreground)" };

  // Nutrition status
  const nutritionStatus = todayCalories > 0
    ? calPct >= 80
      ? { label: "On Track", color: "var(--emotion-calm-to, #06b6d4)" }
      : calPct >= 40
      ? { label: "Moderate", color: "var(--warning, #d4a017)" }
      : { label: "Low", color: "var(--secondary, #e879a8)" }
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
    ? focusVal >= 0.7 ? "var(--emotion-calm-to, #06b6d4)" : focusVal >= 0.45 ? "var(--warning, #d4a017)" : "var(--secondary, #e879a8)"
    : "var(--muted-foreground)";
  const focusStatusLabel = focusVal > 0 ? getFocusLabel(focusVal) : "No data";

  // Compute deltas vs previous session — use brain history, fall back to localStorage yesterday
  const prevEntry = recentHistory && recentHistory.length >= 2 ? recentHistory[recentHistory.length - 2] : null;
  const prevStress = prevEntry?.stress ?? yesterday?.stress_index ?? null;
  const prevFocus = prevEntry?.focus ?? yesterday?.focus_index ?? null;
  const prevValence = prevEntry?.valence ?? yesterday?.valence ?? null;
  const stressDelta = (prevStress != null && stressVal > 0) ? (stressVal - prevStress) : null;
  const focusDelta = (prevFocus != null && focusVal > 0) ? (focusVal - prevFocus) : null;
  const moodDelta = (prevValence != null && checkin?.valence != null) ? (checkin.valence - prevValence) : null;

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
        className="bg-background p-4 font-sans"
      >
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="max-w-[480px] mx-auto"
        >
          {/* ── 1. Header (date + greeting left, avatar right) ── */}
          <motion.div
            variants={itemVariants}
            className="flex items-center justify-between mb-5"
          >
            <div>
              <p className="text-xs text-muted-foreground m-0 mb-0.5 font-medium">
                {formatDate()}
              </p>
              <p className="text-xl font-bold text-foreground m-0 leading-tight">
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
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center text-sm font-bold text-white shrink-0">
              S
            </div>
          </motion.div>


          {/* ── Weather badge (Issue #508) ── */}
          {weatherCtx && (
            <motion.div
              variants={itemVariants}
              className="rounded-[14px] bg-card border border-border mb-3.5 flex items-center gap-2.5 px-4 py-2.5"
            >
              {weatherCtx.condition === "sunny" || weatherCtx.condition === "mostly_clear" ? (
                <Sun className="w-[18px] h-[18px] text-amber-400 shrink-0" />
              ) : weatherCtx.condition === "partly_cloudy" ? (
                <CloudSun className="w-[18px] h-[18px] text-foreground/40 shrink-0" />
              ) : weatherCtx.condition === "cloudy" ? (
                <Cloud className="w-[18px] h-[18px] text-foreground/40 shrink-0" />
              ) : weatherCtx.condition === "rainy" || weatherCtx.condition === "drizzle" ? (
                <CloudRain className="w-[18px] h-[18px] text-blue-400 shrink-0" />
              ) : weatherCtx.condition === "snowy" ? (
                <Snowflake className="w-[18px] h-[18px] text-blue-300 shrink-0" />
              ) : weatherCtx.condition === "stormy" ? (
                <CloudLightning className="w-[18px] h-[18px] text-amber-400 shrink-0" />
              ) : weatherCtx.condition === "foggy" ? (
                <CloudFog className="w-[18px] h-[18px] text-foreground/40 shrink-0" />
              ) : (
                <HelpCircle className="w-[18px] h-[18px] text-foreground/40 shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <div className="text-xs font-semibold text-foreground">
                  {Math.round(weatherCtx.temperature)}&deg;C &middot; {weatherCtx.temperatureLabel}
                </div>
                <div className="text-[11px] text-muted-foreground overflow-hidden text-ellipsis whitespace-nowrap">
                  {weatherCtx.message}
                </div>
              </div>
            </motion.div>
          )}

          {/* ── Cycle phase context (Issue #498) ── */}
          {cycleCtx && (
            <motion.div
              variants={itemVariants}
              className="rounded-[14px] bg-card border border-border mb-3.5 flex items-center gap-2.5 px-4 py-2.5"
            >
              <div className="w-7 h-7 rounded-full bg-secondary/10 flex items-center justify-center shrink-0">
                <Moon className="w-3.5 h-3.5 text-secondary" />
              </div>
              <div className="flex-1">
                <div className="text-xs font-semibold text-foreground">
                  {cycleCtx.phase.charAt(0).toUpperCase() + cycleCtx.phase.slice(1)} phase &middot; Day {cycleCtx.dayInCycle}
                </div>
                <div className="text-[11px] text-muted-foreground">
                  {cycleCtx.message}
                </div>
              </div>
            </motion.div>
          )}

          {/* ── 2. Inner Score Hero ── */}
          <motion.div variants={itemVariants} className="mb-4">
            <InnerScoreCard
              score={innerScore.score}
              tier={innerScore.tier}
              factors={innerScore.factors}
              narrative={innerScore.narrative}
              delta={null}
              trend={[]}
            />
          </motion.div>

          {/* ── 3. Stress & Energy Row (mini cards) ── */}
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="grid grid-cols-2 gap-3 mb-4"
          >
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

          {/* ── 3b. Mood card (full width) ── */}
          <motion.div variants={itemVariants} className="mb-4">
            <ScoreCard
              label="Mood"
              value={moodDisplay}
              statusLabel={moodStatusLabel}
              dotColor={moodDotColor}
              onClick={() => navigate("/mood")}
              delta={moodDelta}
            />
          </motion.div>

          {/* ── Last analysis timestamp ── */}
          {checkinTimestamp && (
            <p className="text-[11px] text-muted-foreground text-center -mt-3 mb-3.5 opacity-70">
              Voice analyzed {(() => {
                const diff = Date.now() - checkinTimestamp;
                if (diff < 60_000) return "just now";
                if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
                if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
                return new Date(checkinTimestamp).toLocaleDateString();
              })()}
              {checkin?.model_type === "fallback" && " (no audio data)"}
              {checkin?.model_type === "on-device" && " (on-device)"}
            </p>
          )}

          {/* ── 4. AI Insight Card (green border, sparkle icon) ── */}
          <motion.div
            variants={itemVariants}
            className="mb-5 rounded-[14px] bg-card border border-emerald-500/20 p-4"
          >
            <div className="flex items-center gap-1.5 mb-2">
              <Sparkles size={13} className="text-emerald-500" />
              <span className="text-[11px] font-semibold text-emerald-500 uppercase tracking-wider">
                AI Insight
              </span>
            </div>
            <p className="text-sm font-semibold text-foreground leading-snug mb-1.5">
              {aiInsight.headline}
            </p>
            <p className="text-xs text-muted-foreground leading-relaxed mb-2.5">
              {aiInsight.body}
            </p>
            <div className="flex items-start gap-2 rounded-xl bg-emerald-500/[0.08] border border-emerald-500/15 px-3 py-2">
              <span className="text-xs font-bold text-emerald-500 mt-px">-&gt;</span>
              <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400 m-0">{aiInsight.action}</p>
            </div>
          </motion.div>

          {/* ── This Week mini trend chart ── */}
          {weekTrendData.chartRows.some(r => r.stress !== null) && (
            <motion.div
              variants={itemVariants}
              className="rounded-[14px] bg-card border border-border mb-5 p-4"
            >
              <div className="flex items-center justify-between mb-3">
                <div>
                  <p className="text-[11px] font-semibold text-foreground">This Week</p>
                  <p className="text-[10px] text-muted-foreground">Daily averages</p>
                </div>
                <div className="flex gap-3">
                  {[{ label: "Stress", color: "#f87171" }, { label: "Focus", color: "#60a5fa" }].map(({ label, color }) => (
                    <div key={label} className="flex items-center gap-1">
                      <div className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
                      <span className="text-[10px] text-muted-foreground">{label}</span>
                    </div>
                  ))}
                </div>
              </div>
              <ResponsiveContainer width="100%" height={100}>
                <AreaChart
                  data={weekTrendData.chartRows}
                  margin={{ left: -30, right: 0, top: 2, bottom: 0 }}
                >
                  <defs>
                    <linearGradient id="twStress" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f87171" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#f87171" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="twFocus" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#60a5fa" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#60a5fa" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="label"
                    tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis domain={[0, 100]} hide />
                  <Tooltip
                    contentStyle={{
                      background: "hsl(var(--card))", border: "1px solid hsl(var(--border))",
                      borderRadius: 8, fontSize: 11, padding: "4px 8px",
                      color: "hsl(var(--foreground))",
                    }}
                    formatter={(v: number, name: string) => [`${v}%`, name]}
                    labelStyle={{ color: "hsl(var(--muted-foreground))", fontSize: 10 }}
                  />
                  {weekTrendData.baseline && (
                    <>
                      <ReferenceLine y={Math.round(weekTrendData.baseline.stress * 100)} stroke="#f87171" strokeDasharray="3 2" strokeOpacity={0.35} strokeWidth={1} />
                      <ReferenceLine y={Math.round(weekTrendData.baseline.focus * 100)} stroke="#60a5fa" strokeDasharray="3 2" strokeOpacity={0.35} strokeWidth={1} />
                    </>
                  )}
                  <Area type="monotone" dataKey="stress" name="Stress" stroke="#f87171" strokeWidth={1.5} fill="url(#twStress)" dot={false} connectNulls />
                  <Area type="monotone" dataKey="focus" name="Focus" stroke="#60a5fa" strokeWidth={1.5} fill="url(#twFocus)" dot={false} connectNulls />
                </AreaChart>
              </ResponsiveContainer>
              {weekTrendData.baseline && (
                <p className="text-[10px] text-muted-foreground text-center mt-1.5">Dashed = your baseline</p>
              )}
            </motion.div>
          )}

          {/* ── 4b. How Are You Feeling — MoodPicker ── */}
          <motion.div
            variants={itemVariants}
            className="rounded-[14px] bg-card border border-border mb-5 overflow-hidden"
          >
            <MoodPicker
              userName={undefined}
              onMoodSelect={(result) => {
                // Map 2D grid position to valence/arousal/stress
                const valence = (result.pleasantness - 0.5) * 2; // -1 to 1
                const arousal = result.energy; // 0 to 1
                const stressVal = result.pleasantness < 0.5 ? 0.3 + (1 - result.pleasantness) * 0.5 : Math.max(0, 0.3 - result.pleasantness * 0.3);

                // Save to same storage as existing feeling system
                const emotionResult = {
                  emotion: result.emotionWord.toLowerCase(),
                  valence,
                  arousal,
                  stress: stressVal,
                  focus: 0.5,
                  confidence: 0.9,
                  source: "manual" as const,
                  timestamp: result.timestamp,
                };
                try {
                  localStorage.setItem("ndw_last_emotion", JSON.stringify({
                    result: emotionResult,
                    timestamp: Date.now(),
                  }));
                  window.dispatchEvent(new CustomEvent("ndw-emotion-update"));
                } catch {}

                // Sync to ML backend
                const noteText = [
                  result.emotionWord,
                  `(${result.quadrant})`,
                  ...result.tags,
                  result.note,
                ].filter(Boolean).join(" — ");

                try {
                  syncMoodLogToML({
                    user_id: userId,
                    mood_score: Math.round(result.pleasantness * 10),
                    energy_level: Math.round(result.energy * 10),
                    notes: noteText,
                  });
                } catch {}
              }}
            />
          </motion.div>

          {/* ── 4c. Text Log a Feeling (secondary) ── */}
          <motion.div
            variants={itemVariants}
            className="rounded-[14px] bg-card border border-border mb-5 p-4"
          >
            <div className="flex items-center gap-1.5 mb-2.5">
              <PenLine size={13} className="text-primary" />
              <span className="text-[11px] font-semibold text-primary uppercase tracking-wider">
                Or type how you feel
              </span>
            </div>
            <input
              type="text"
              value={feelingText}
              onChange={(e) => setFeelingText(e.target.value)}
              placeholder="proud of myself, grateful, anxious..."
              onKeyDown={(e) => { if (e.key === "Enter") submitFeeling(); }}
              className="w-full bg-foreground/[0.03] border border-foreground/8 rounded-xl px-3.5 py-2.5 text-sm text-foreground placeholder:text-foreground/30 outline-none focus:border-primary/30 focus:bg-foreground/[0.05] transition-all duration-200 mb-2.5"
            />
            <div className="flex items-center justify-end">
              <button
                onClick={submitFeeling}
                disabled={!feelingText.trim() || feelingSaving}
                className={`px-4 py-1.5 rounded-2xl border-none text-xs font-semibold transition-all duration-150 ${
                  feelingText.trim()
                    ? "bg-gradient-to-br from-primary to-secondary text-white cursor-pointer"
                    : "bg-foreground/[0.04] text-foreground/35 cursor-default"
                } ${feelingSaving ? "opacity-60" : "opacity-100"}`}
              >
                {feelingSaving ? "Saving..." : "Save"}
              </button>
            </div>
            {feelingSaved && (
              <motion.p
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-xs text-cyan-400 font-medium mt-2 text-center"
              >
                Feeling logged!
              </motion.p>
            )}
            {/* Feeling history -- show last 3 as pills with timestamps */}
            {moodLogHistory && moodLogHistory.length > 0 && (
              <div className="mt-3 border-t border-foreground/[0.06] pt-2.5">
                <div className="flex flex-wrap gap-1.5">
                  {moodLogHistory.slice(0, 3).map((entry, i) => {
                    const score = typeof entry.moodScore === "string" ? parseFloat(entry.moodScore) : (entry.moodScore ?? 5);
                    const tone = getMoodLogTone(score);
                    return (
                      <div
                        key={entry.id ?? i}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-[20px]"
                        style={{
                          background: `color-mix(in srgb, ${tone.color} 8%, transparent)`,
                          border: `1px solid color-mix(in srgb, ${tone.color} 18%, transparent)`,
                        }}
                      >
                        <span className="text-xs text-foreground leading-snug">
                          {entry.notes || tone.label}
                        </span>
                        {entry.loggedAt && (
                          <span className="text-[9px] text-muted-foreground shrink-0">
                            {formatTime(entry.loggedAt)}
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </motion.div>

          {/* ── 4c. Emotion Correction (after voice analysis) ── */}
          {checkin?.emotion && checkin.emotion !== "---" && (
            <motion.div
              variants={itemVariants}
              className="rounded-[14px] bg-card border border-border p-4 mb-5 border-l-[3px] border-l-primary"
            >
              <div className="flex items-center gap-2 mb-2">
                <span className="text-[22px]">
                  {checkin.emotion === "happy" ? "😊" : checkin.emotion === "sad" ? "😢" : checkin.emotion === "angry" ? "😠" : checkin.emotion === "fear" ? "😨" : checkin.emotion === "surprise" ? "😲" : checkin.emotion === "anxious" ? "😰" : checkin.emotion === "neutral" ? "😐" : "🧠"}
                </span>
                <div>
                  <span className="text-sm font-semibold text-foreground capitalize">
                    {correctedEmotion ?? (fusedEmotion?.emotion ?? checkin.emotion)}
                  </span>
                  <div className="flex items-center gap-1 mt-0.5">
                    {fusedEmotion && fusedEmotion.sources.length > 1 ? (
                      <>
                        <span className="text-[11px] text-muted-foreground">fused from</span>
                        {fusedEmotion.sources.includes("eeg") && (
                          <span title="EEG"><Brain className="w-3 h-3 text-primary" /></span>
                        )}
                        {fusedEmotion.sources.includes("voice") && (
                          <span title="Voice"><Mic className="w-3 h-3 text-cyan-400" /></span>
                        )}
                        {fusedEmotion.sources.includes("health") && (
                          <span title="Health"><Activity className="w-3 h-3 text-rose-400" /></span>
                        )}
                      </>
                    ) : null}
                  </div>
                </div>
              </div>

              {/* Confidence meter for emotion reading */}
              {emotionConfidence && (
                <div className="mt-2 mb-1">
                  {emotionConfidence.showEmotion ? (
                    <ConfidenceMeter
                      confidence={emotionConfidence.confidence}
                      size="sm"
                      showLabel
                    />
                  ) : (
                    <p className="text-[11px] text-muted-foreground leading-snug">
                      Not enough data to determine your emotional state. Try a voice check-in or connect your Muse headband.
                    </p>
                  )}
                </div>
              )}

              {/* Active learning prompt -- prominent when confidence < 0.4 */}
              {emotionFeedback === "ask" && (checkin.confidence ?? 1) < 0.4 && (
                <div
                  data-testid="active-learning-prompt"
                  className="rounded-lg border border-amber-500/30 bg-amber-500/[0.08] p-3"
                >
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-500 shrink-0" />
                    <span className="text-xs font-semibold text-amber-400">
                      Low confidence -- your feedback is especially valuable
                    </span>
                  </div>
                  <p className="text-[10px] text-muted-foreground leading-snug mb-2">
                    The model is uncertain about this prediction. One correction here teaches more than 5 high-confidence ones.
                  </p>
                  <div className="flex gap-2">
                    <button
                      onClick={confirmVoiceEmotion}
                      className="flex items-center gap-1 px-3.5 py-1 rounded-2xl border-none bg-amber-500/15 text-amber-400 text-xs font-medium cursor-pointer"
                    >
                      <Check className="w-3 h-3" />
                      Yes, correct
                    </button>
                    <button
                      onClick={() => setEmotionFeedback("correcting")}
                      className="flex items-center gap-1 px-3.5 py-1 rounded-2xl border-none bg-amber-500/15 text-amber-400 text-xs font-medium cursor-pointer"
                    >
                      <Pencil className="w-3 h-3" />
                      No, I felt...
                    </button>
                  </div>
                </div>
              )}

              {/* Standard "Is this right?" feedback (confidence >= 0.4) */}
              {emotionFeedback === "ask" && (checkin.confidence ?? 1) >= 0.4 && (
                <div className="flex items-center justify-between pt-2 border-t border-foreground/[0.06]">
                  <span className="text-xs text-muted-foreground">Is this right?</span>
                  <div className="flex gap-2">
                    <button
                      onClick={confirmVoiceEmotion}
                      className="flex items-center gap-1 px-3.5 py-1 rounded-2xl border-none bg-primary/10 text-primary text-xs font-medium cursor-pointer"
                    >
                      <Check className="w-3 h-3" />
                      Yes
                    </button>
                    <button
                      onClick={() => setEmotionFeedback("correcting")}
                      className="flex items-center gap-1 px-3.5 py-1 rounded-2xl border-none bg-muted text-muted-foreground text-xs font-medium cursor-pointer"
                    >
                      <Pencil className="w-3 h-3" />
                      Correct it
                    </button>
                  </div>
                </div>
              )}

              {emotionFeedback === "correcting" && (
                <div className="pt-2 border-t border-foreground/[0.06]">
                  <span className="text-xs text-muted-foreground block mb-2">
                    What were you actually feeling?
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {["happy", "sad", "angry", "fearful", "surprised", "neutral", "anxious", "peaceful", "excited", "frustrated"].map((em) => (
                      <button
                        key={em}
                        onClick={() => submitEmotionCorrection(em)}
                        className={`px-3 py-1 rounded-2xl border-none text-[11px] font-medium cursor-pointer capitalize ${
                          em === checkin.emotion
                            ? "bg-muted text-muted-foreground opacity-50"
                            : "bg-primary/10 text-primary"
                        }`}
                      >
                        {em}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {emotionFeedback === "confirmed" && (
                <div className="flex items-center gap-1.5 pt-2 border-t border-foreground/[0.06]">
                  <Check className="w-[13px] h-[13px] text-cyan-400" />
                  <span className="text-xs text-cyan-400 font-medium">Confirmed</span>
                </div>
              )}

              {emotionFeedback === "corrected" && correctedEmotion && (
                <div className="flex items-center gap-1.5 pt-2 border-t border-foreground/[0.06]">
                  <Check className="w-[13px] h-[13px] text-cyan-400" />
                  <span className="text-xs text-cyan-400 font-medium">
                    Saved as {correctedEmotion}. This helps improve accuracy!
                  </span>
                </div>
              )}
            </motion.div>
          )}


          {/* ── Stress Warning (conditional) ── */}
          {(checkin?.stress_index ?? 0) > 0.6 && (
            <motion.div
              variants={itemVariants}
              className="rounded-[14px] bg-card border border-border p-4 mb-5"
            >
              <div className="flex items-start gap-3 mb-3">
                <AlertTriangle className="w-[22px] h-[22px] text-rose-400 shrink-0" />
                <div>
                  <div className="text-sm font-semibold text-rose-400 mb-1">
                    Your stress levels are elevated
                  </div>
                  <div className="text-sm text-muted-foreground leading-relaxed">
                    Take a moment to breathe. Try the 4-7-8 breathing technique:
                    <br />
                    <span className="font-medium text-foreground">
                      Inhale 4s &rarr; Hold 7s &rarr; Exhale 8s
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() =>
                    window.open(
                      "https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO",
                      "_blank"
                    )
                  }
                  className="flex-1 bg-gradient-to-br from-green-500 to-green-700 text-white border-none rounded-xl px-4 py-2.5 text-sm font-semibold cursor-pointer active:scale-[0.97]"
                >
                  Listen to Calm Music
                </button>
                <button
                  onClick={() => setShowBreathe(true)}
                  className="flex-1 bg-gradient-to-br from-primary to-violet-700 text-white border-none rounded-xl px-4 py-2.5 text-sm font-semibold cursor-pointer active:scale-[0.97]"
                >
                  Breathing Exercise
                </button>
              </div>
            </motion.div>
          )}

          {/* Inline breathing exercise */}
          {showBreathe && <InlineBreathe onClose={() => setShowBreathe(false)} />}

          {/* ── 4b. Recovery Interventions ── */}
          <motion.div variants={itemVariants}>
            <RecoveryInterventions
              recovery={scores?.recovery}
              sleepHours={sleepTotal}
              strain={scores?.strain}
              stress={scores?.stress}
              hrvTrend={hrvTrend}
            />
          </motion.div>

          {/* ── 4c. Brain Coach — EEG + health fusion ── */}
          <motion.div variants={itemVariants}>
            <BrainCoachCard
              recoveryScore={userScores?.recoveryScore ?? null}
              sleepScore={userScores?.sleepScore ?? null}
              stressScore={userScores?.stressScore ?? null}
              strainScore={userScores?.strainScore ?? null}
              avgFocus={brainTotals?.avgFocus ?? null}
              avgValence={brainTotals?.avgValence ?? null}
            />
          </motion.div>

          {/* ── 4d. EEG Brain Trends — 7-day week-over-week comparison ── */}
          {recentHistory && recentHistory.length > 0 && (
            <motion.div variants={itemVariants}>
              <EEGWeekCompareCard history={recentHistory} />
            </motion.div>
          )}

          {/* ── 4e. Energy Timeline Forecast ── */}
          <motion.div variants={itemVariants} className="rounded-[14px] bg-card border border-border p-4">
            <EnergyTimeline
              sleepHours={sleepTotal}
              recovery={scores?.recovery}
            />
          </motion.div>

          {/* ── 5. Health Monitor (2-column grid) ── */}
          <motion.div variants={itemVariants} className="mb-3 mt-1">
            <span className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
              Health Monitor
            </span>
          </motion.div>
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="grid grid-cols-2 gap-3"
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
              barGradient="linear-gradient(90deg, var(--primary), var(--neural-purple))"
              accentColor="var(--primary)"
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
              accentColor="var(--secondary)"
              emptyIcon={Heart}
              emptyCta="Connect Health to track"
            />

            {/* HRV */}
            <HealthMetricCard
              label="HRV"
              value={hrvSdnn ? `${Math.round(hrvSdnn)}` : "---"}
              unit={hrvSdnn ? "ms" : ""}
              statusLabel={hrvSdnn ? (hrvSdnn >= 50 ? "Good" : hrvSdnn >= 30 ? "Fair" : "Low") : "No data"}
              dotColor={hrvSdnn ? (hrvSdnn >= 50 ? "var(--emotion-calm-to, #06b6d4)" : hrvSdnn >= 30 ? "var(--warning, #d4a017)" : "var(--secondary, #e879a8)") : "var(--muted-foreground)"}
              accentColor="var(--emotion-calm-to)"
              emptyIcon={Activity}
              emptyCta="Sync to see HRV"
            />

            {/* SpO2 */}
            <HealthMetricCard
              label="SpO2"
              value={spo2 ? `${Math.round(spo2)}` : "---"}
              unit={spo2 ? "%" : ""}
              statusLabel={spo2 ? (spo2 >= 95 ? "Normal" : "Low") : "No data"}
              dotColor={spo2 ? (spo2 >= 95 ? "var(--emotion-calm-to, #06b6d4)" : "var(--secondary, #e879a8)") : "var(--muted-foreground)"}
              accentColor="var(--primary)"
              emptyIcon={Heart}
              emptyCta="Sync to see SpO2"
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
              barGradient="linear-gradient(90deg, var(--emotion-calm-to), var(--neural-cyan))"
              accentColor="var(--emotion-calm-to)"
              emptyIcon={Footprints}
              emptyCta="Sync to see steps"
            />

            {/* Respiratory Rate */}
            <HealthMetricCard
              label="Resp. Rate"
              value={respiratoryRate ? `${Math.round(respiratoryRate)}` : "---"}
              unit={respiratoryRate ? "br/min" : ""}
              statusLabel={respiratoryRate ? (respiratoryRate >= 12 && respiratoryRate <= 20 ? "Normal" : "Abnormal") : "No data"}
              dotColor={respiratoryRate ? (respiratoryRate >= 12 && respiratoryRate <= 20 ? "var(--emotion-calm-to, #06b6d4)" : "var(--secondary, #e879a8)") : "var(--muted-foreground)"}
              accentColor="var(--warning)"
              emptyIcon={Wind}
              emptyCta="Sync to see rate"
            />

            {false && (() => { // Food score removed from Today — shown on nutrition page when you tap a meal
              return null;
            })()}
          </motion.div>

          {/* ── 6. Nutrition Summary (calorie progress bar) ── */}
          <motion.div
            variants={itemVariants}
            onClick={() => navigate("/nutrition")}
            className="rounded-[14px] bg-card border border-border p-4 mt-3 cursor-pointer hover:border-foreground/15 transition-colors"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5">
                <UtensilsCrossed className="w-3.5 h-3.5 text-muted-foreground" />
                <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
                  Nutrition
                </span>
              </div>
              <span className="text-xs text-muted-foreground">
                {todayCalories > 0 ? `${todayCalories.toLocaleString()} / ${calGoal.toLocaleString()} kcal` : "No meals logged"}
              </span>
            </div>
            <div className="w-full h-2 rounded-full bg-muted/40 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700 ease-out"
                style={{
                  width: `${todayCalories > 0 ? calPct : 0}%`,
                  background: todayCalories > 0
                    ? "linear-gradient(90deg, hsl(var(--warning)), hsl(var(--destructive)))"
                    : "transparent",
                }}
              />
            </div>
            {todayCalories > 0 && (
              <p className="text-[10px] text-muted-foreground mt-1.5 text-right">
                {calPct}% of daily goal
              </p>
            )}
          </motion.div>

          {/* ── Quick Listen -- Spotify Music Section ── */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.5, ease: "easeOut" }}
            className="mb-5"
          >
            <div className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">
              Quick Listen
            </div>
            <div className="flex gap-3.5 overflow-x-auto pb-1 [scrollbar-width:none] [-webkit-overflow-scrolling:touch]">
              {([
                { icon: Music, colorClass: "text-indigo-400", title: "Focus", url: "https://open.spotify.com/playlist/37i9dQZF1DWZeKCadgRdKQ" },
                { icon: Wind, colorClass: "text-cyan-500", title: "Calm", url: "https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO" },
                { icon: CloudMoon, colorClass: "text-primary", title: "Sleep", url: "https://open.spotify.com/playlist/37i9dQZF1DWZd79rJ6a7lp" },
                { icon: Dumbbell, colorClass: "text-orange-500", title: "Workout", url: "https://open.spotify.com/playlist/37i9dQZF1DX76Wlfdnj7AP" },
                { icon: TreePine, colorClass: "text-green-400", title: "Energize", url: "https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0" },
              ] as const).map((card) => {
                const IconComp = card.icon;
                return (
                <button
                  key={card.title}
                  onClick={() => window.open(card.url, "_blank")}
                  className="flex-none w-[90px] rounded-[14px] bg-card border border-border px-2 py-3 text-center cursor-pointer transition-all duration-200 active:scale-[0.96] hover:border-foreground/15"
                >
                  <div className="flex justify-center mb-1">
                    <IconComp className={`w-6 h-6 ${card.colorClass}`} />
                  </div>
                  <div className="text-[11px] font-semibold text-foreground">{card.title}</div>
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
