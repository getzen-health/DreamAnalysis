import { useState, useEffect, useMemo, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { resolveUrl } from "@/lib/queryClient";
import { getParticipantId } from "@/lib/participant";
import { useHealthSync } from "@/hooks/use-health-sync";
import { Sparkles } from "lucide-react";
import { ScoreSplash } from "@/components/score-splash";
import { hapticWarning } from "@/lib/haptics";
import { useVoiceData, type VoiceCheckinData } from "@/hooks/use-voice-data";
import { InlineBreathe } from "@/components/inline-breathe";
import { forecastMood } from "@/lib/mood-patterns";
import { StreakProtector } from "@/components/streak-protector";
import { SleepInsights } from "@/components/sleep-insights";

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
  if (stress < 0.3) return "#0891b2";
  if (stress < 0.6) return "#d4a017";
  return "#e879a8";
}

function getFocusLabel(focus: number): string {
  if (focus >= 0.7) return "Sharp";
  if (focus >= 0.45) return "Moderate";
  return "Diffuse";
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

// ── Tomorrow's Forecast Card ──────────────────────────────────────────────

const FORECAST_EMOJI: Record<string, string> = {
  happy: "😊", sad: "😢", angry: "😠", fear: "😨",
  surprise: "😲", neutral: "😐",
};

function ForecastCard({ userId }: { userId: string }) {
  const { data } = useQuery<Array<{ dominantEmotion: string; timestamp: string }>>({
    queryKey: [`/api/brain/history/${userId}?days=7`],
    retry: false,
    staleTime: 10 * 60 * 1000,
  });

  if (!data || data.length < 3) return null;

  const forecast = forecastMood(data);
  if (!forecast) return null;

  const emoji = FORECAST_EMOJI[forecast.emotion] ?? "🔮";
  const confPct = Math.round(forecast.confidence * 100);

  return (
    <div style={{
      background: "var(--card)", border: "1px solid var(--border)",
      borderRadius: 14, padding: 14, marginTop: 14,
    }}>
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6,
      }}>
        <div style={{
          fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
          textTransform: "uppercase" as const, letterSpacing: "0.5px",
        }}>
          Tomorrow's forecast
        </div>
        <span style={{ fontSize: 10, color: "var(--muted-foreground)" }}>{confPct}% likely</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 28 }}>{emoji}</span>
        <div>
          <div style={{ fontSize: 15, fontWeight: 600, color: "var(--foreground)", textTransform: "capitalize" as const }}>
            {forecast.emotion}
          </div>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>
            {forecast.reason}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Weekly Summary Card ────────────────────────────────────────────────────

function WeeklySummaryCard({ userId }: { userId: string }) {
  const dayOfWeek = new Date().getDay();
  // Only show on Sunday (0) and Monday (1)
  if (dayOfWeek !== 0 && dayOfWeek !== 1) return null;

  const { data } = useQuery<{
    available: boolean;
    summary?: {
      total_readings: number;
      checkin_days: number;
      avg_stress: number;
      avg_focus: number;
      avg_happiness: number;
      avg_energy: number;
      dominant_emotion: string;
      stress_trend: string;
    };
    insight?: string;
  }>({
    queryKey: [`/api/brain/weekly-summary/${userId}`],
    staleTime: 60 * 60 * 1000, // Cache for 1 hour
    retry: false,
  });

  if (!data?.available || !data.summary) return null;
  const s = data.summary;

  const trendIcon = s.stress_trend === "decreasing" ? "↓" : s.stress_trend === "increasing" ? "↑" : "→";
  const trendColor = s.stress_trend === "decreasing" ? "#4ade80" : s.stress_trend === "increasing" ? "#e87676" : "var(--muted-foreground)";

  return (
    <div style={{
      background: "var(--card)", border: "1px solid var(--border)",
      borderRadius: 14, padding: 14, marginTop: 14,
    }}>
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
        textTransform: "uppercase" as const, letterSpacing: "0.5px", marginBottom: 8,
      }}>
        Weekly Wellness Report
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>
        <div style={{ background: "var(--muted)", borderRadius: 10, padding: "8px 10px" }}>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)" }}>Avg Stress</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: "var(--foreground)" }}>
            {s.avg_stress}%
            <span style={{ fontSize: 12, color: trendColor, marginLeft: 4 }}>{trendIcon}</span>
          </div>
        </div>
        <div style={{ background: "var(--muted)", borderRadius: 10, padding: "8px 10px" }}>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)" }}>Avg Focus</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: "var(--foreground)" }}>{s.avg_focus}%</div>
        </div>
        <div style={{ background: "var(--muted)", borderRadius: 10, padding: "8px 10px" }}>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)" }}>Check-in Days</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: "var(--foreground)" }}>{s.checkin_days}/7</div>
        </div>
        <div style={{ background: "var(--muted)", borderRadius: 10, padding: "8px 10px" }}>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)" }}>Top Mood</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: "var(--foreground)", textTransform: "capitalize" as const }}>{s.dominant_emotion}</div>
        </div>
      </div>
      {data.insight && (
        <div style={{
          fontSize: 12, color: "var(--muted-foreground)", lineHeight: 1.5,
          padding: "8px 0", borderTop: "1px solid var(--border)",
        }}>
          {data.insight}
        </div>
      )}
    </div>
  );
}

// ── Daily Wellness Tip ────────────────────────────────────────────────────

const TIPS_BY_STATE: Record<string, string[]> = {
  happy: [
    "🌟 You're in a great mood! This is the perfect time for creative projects or learning something new.",
    "😊 Positive energy detected. Share it with someone — positivity is contagious.",
    "🎨 Your brain is in a receptive state. Try journaling or expressing yourself creatively.",
    "💡 Happy state = better problem solving. Tackle that thing you've been putting off.",
  ],
  sad: [
    "💙 It's okay to feel low. A 10-minute walk outside can boost serotonin naturally.",
    "🫂 Be gentle with yourself today. Reaching out to a friend can help more than you think.",
    "🎵 Music therapy works — listening to uplifting music for 15 minutes can shift your mood.",
    "☕ Warm drinks activate comfort pathways. Take a moment with your favorite tea.",
  ],
  angry: [
    "🌊 Anger carries energy. Channel it into a workout or power walk — physical movement helps.",
    "🧊 Try the cold water trick: splash cold water on your wrists to activate the dive reflex and calm down.",
    "📝 Write it out. Putting anger into words reduces its emotional intensity by up to 50%.",
    "⏸️ The 4-7-8 breath: inhale 4s, hold 7s, exhale 8s. Repeat 3 times.",
  ],
  fear: [
    "🤗 Anxiety often feels bigger than the situation warrants. Name what you're afraid of — naming reduces fear.",
    "🌳 Grounding technique: name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.",
    "💪 Remind yourself: you've handled hard things before. You can handle this too.",
    "🕯️ Progressive muscle relaxation: tense each muscle group for 5s, then release. Start from your toes.",
  ],
  neutral: [
    "⚖️ Balanced state — a good baseline for mindfulness practice or setting intentions.",
    "🎯 Neutral is underrated. Use this calm moment to plan your day with clarity.",
    "📚 Your brain is receptive right now. Great time for reading or learning.",
    "🧘 Try a 5-minute meditation to deepen this peaceful state.",
  ],
  default: [
    "🎙️ Record a voice note to get personalized wellness tips based on your emotional state.",
    "💫 Your emotional data shapes your wellness journey. The more you track, the better the insights.",
  ],
};

function DailyTip({ emotion, stress, focus }: { emotion: string; stress: number; focus: number }) {
  // Pick a consistent daily tip (changes once per day, not on every render)
  const dayIndex = new Date().getDate();
  const state = emotion === "—" ? "default" : emotion;
  const tips = TIPS_BY_STATE[state] ?? TIPS_BY_STATE.default;
  const tip = tips[dayIndex % tips.length];

  return (
    <div style={{
      background: "var(--card)", border: "1px solid var(--border)",
      borderRadius: 14, padding: 14, marginTop: 14,
    }}>
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
        textTransform: "uppercase" as const, letterSpacing: "0.5px", marginBottom: 6,
      }}>
        Daily tip
      </div>
      <p style={{ fontSize: 13, color: "var(--foreground)", margin: 0, lineHeight: 1.5 }}>
        {tip}
      </p>
    </div>
  );
}

// ── Mental Fitness Score ───────────────────────────────────────────────────

function MentalFitnessCard({ voice }: { voice: VoiceCheckinData | null }) {
  if (!voice?.stress_index) return null;

  const stress = voice.stress_index ?? 0.5;
  const focus = voice.focus_index ?? 0.5;
  const valence = voice.valence ?? 0;
  const arousal = voice.arousal ?? 0.5;

  // Mental Fitness = composite of low stress + high focus + positive valence + balanced arousal
  const stressScore = (1 - stress) * 30;      // 0-30: lower stress = better
  const focusScore = focus * 25;               // 0-25: higher focus = better
  const valenceScore = ((valence + 1) / 2) * 25; // 0-25: positive = better
  const arousalBalance = (1 - Math.abs(arousal - 0.5) * 2) * 20; // 0-20: moderate = best
  const raw = stressScore + focusScore + valenceScore + arousalBalance;
  const score = Math.min(100, Math.max(0, Math.round(raw)));

  const color = score >= 70 ? "#4ade80" : score >= 45 ? "#e8b94a" : "#e87676";
  const label = score >= 80 ? "Excellent" : score >= 65 ? "Good" : score >= 45 ? "Fair" : "Low";
  const desc = score >= 70
    ? "Your mental fitness is strong — keep it up"
    : score >= 45
    ? "Room for improvement — try a breathing exercise"
    : "Your mind needs rest — prioritize self-care today";

  // Simple bar visualization
  return (
    <div style={{
      background: "var(--card)", border: "1px solid var(--border)",
      borderRadius: 14, padding: 14, marginBottom: 14,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <div>
          <div style={{
            fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
            textTransform: "uppercase" as const, letterSpacing: "0.5px",
          }}>
            Mental Fitness
          </div>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>
            From voice biomarkers
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 28, fontWeight: 700, color, lineHeight: 1 }}>{score}</div>
          <div style={{ fontSize: 10, color }}>{label}</div>
        </div>
      </div>

      {/* Score bar */}
      <div style={{ height: 6, background: "var(--border)", borderRadius: 3, overflow: "hidden", marginBottom: 8 }}>
        <div style={{
          width: `${score}%`, height: "100%", background: color,
          borderRadius: 3, transition: "width 1.2s cubic-bezier(0.22, 1, 0.36, 1)",
        }} />
      </div>

      {/* Breakdown mini bars */}
      <div style={{ display: "flex", gap: 8, fontSize: 9, color: "var(--muted-foreground)" }}>
        <div style={{ flex: 1 }}>
          <div style={{ marginBottom: 2 }}>Calm {Math.round((1 - stress) * 100)}%</div>
          <div style={{ height: 3, background: "var(--border)", borderRadius: 2 }}>
            <div style={{ width: `${(1 - stress) * 100}%`, height: "100%", background: "#4ade80", borderRadius: 2 }} />
          </div>
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ marginBottom: 2 }}>Focus {Math.round(focus * 100)}%</div>
          <div style={{ height: 3, background: "var(--border)", borderRadius: 2 }}>
            <div style={{ width: `${focus * 100}%`, height: "100%", background: "#3b82f6", borderRadius: 2 }} />
          </div>
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ marginBottom: 2 }}>Mood {Math.round(((valence + 1) / 2) * 100)}%</div>
          <div style={{ height: 3, background: "var(--border)", borderRadius: 2 }}>
            <div style={{ width: `${((valence + 1) / 2) * 100}%`, height: "100%", background: "#e8b94a", borderRadius: 2 }} />
          </div>
        </div>
      </div>

      <p style={{ fontSize: 10, color: "var(--muted-foreground)", margin: "8px 0 0 0", fontStyle: "italic" }}>
        {desc}
      </p>
    </div>
  );
}

// ── Weekly Mood Strip ──────────────────────────────────────────────────────

const MOOD_COLORS: Record<string, string> = {
  happy: "#4ade80", sad: "#7ba7d9", angry: "#e87676", fear: "#b49ae0",
  surprise: "#e8b94a", neutral: "#a09890",
};

function WeeklyMoodStrip({ userId }: { userId: string }) {
  const { data } = useQuery<Array<{ dominantEmotion: string; timestamp: string }>>({
    queryKey: [`/api/brain/history/${userId}?days=7`],
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  if (!data || data.length === 0) return null;

  // Group by day, take last emotion per day
  const dayMap = new Map<string, { emotion: string; label: string }>();
  for (const r of data) {
    const d = new Date(r.timestamp);
    const key = d.toISOString().slice(0, 10);
    const label = d.toLocaleDateString(undefined, { weekday: "narrow" });
    dayMap.set(key, { emotion: r.dominantEmotion, label });
  }

  const days = Array.from(dayMap.entries()).sort(([a], [b]) => a.localeCompare(b)).slice(-7);
  if (days.length < 2) return null;

  return (
    <div
      onClick={() => window.location.href = "/discover"}
      style={{
        background: "var(--card)", border: "1px solid var(--border)",
        borderRadius: 14, padding: "12px 14px", marginBottom: 14, cursor: "pointer",
      }}
    >
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8,
      }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)", textTransform: "uppercase" as const, letterSpacing: "0.5px" }}>
          This week
        </span>
        <span style={{ fontSize: 16, color: "var(--muted-foreground)" }}>›</span>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        {days.map(([key, { emotion, label }]) => (
          <div key={key} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
            <div style={{
              width: 24, height: 24, borderRadius: "50%",
              background: MOOD_COLORS[emotion] ?? "#94a3b8", opacity: 0.85,
            }} />
            <span style={{ fontSize: 8, color: "var(--muted-foreground)" }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Emotion emojis ────────────────────────────────────────────────────────

const EMOTION_EMOJI: Record<string, string> = {
  happy: "😊", sad: "😢", angry: "😠", fear: "😨",
  surprise: "😲", neutral: "😐",
};

const EMOTION_COLOR: Record<string, string> = {
  happy: "#4ade80", sad: "#7ba7d9", angry: "#e87676", fear: "#b49ae0",
  surprise: "#e8b94a", neutral: "#a09890",
};

// ── Hero Section: Emotion + Readiness ─────────────────────────────────────

function EmotionHero({ checkin, score }: { checkin: EmotionCheckin | null; score: number }) {
  const emotion = checkin?.emotion ?? "neutral";
  const emoji = EMOTION_EMOJI[emotion] ?? "😐";
  const color = EMOTION_COLOR[emotion] ?? "#94a3b8";
  // Read confidence from the raw localStorage result
  const confidence = (() => {
    try {
      const raw = localStorage.getItem("ndw_last_emotion");
      if (raw) {
        const parsed = JSON.parse(raw);
        return Math.round((parsed?.result?.confidence ?? 0) * 100);
      }
    } catch { /* ignore */ }
    return 0;
  })();
  const label = getEmotionScoreLabel(score);
  const hasData = !!checkin?.emotion;

  // Arc params
  const r = 52;
  const cx = 65;
  const cy = 65;
  const totalArc = (270 / 360) * 2 * Math.PI * r; // ~245
  const circumference = 2 * Math.PI * r;
  const filled = (score / 100) * totalArc;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, margin: "8px 0 4px" }}>
      {/* Emotion emoji + label */}
      {hasData ? (
        <>
          <div style={{
            fontSize: 52, lineHeight: 1,
            animation: "gentleFloat 4s ease-in-out infinite",
          }}>{emoji}</div>
          <style>{`@keyframes gentleFloat { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }`}</style>
          <div style={{ fontSize: 22, fontWeight: 700, color, textTransform: "capitalize" as const }}>{emotion}</div>
          <div style={{ fontSize: 11, color: "var(--muted-foreground)" }}>
            {confidence > 0 ? `${confidence}% confidence` : "via voice"} · valence {(checkin?.valence ?? 0) >= 0 ? "+" : ""}{(checkin?.valence ?? 0).toFixed(1)}
          </div>
          {/* Time since check-in */}
          {(() => {
            try {
              const raw = localStorage.getItem("ndw_last_emotion");
              if (raw) {
                const ts = JSON.parse(raw)?.timestamp;
                if (ts) {
                  const mins = Math.floor((Date.now() - ts) / 60000);
                  const label = mins < 1 ? "just now" : mins < 60 ? `${mins}m ago` : `${Math.floor(mins / 60)}h ago`;
                  return <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2, opacity: 0.7 }}>Detected {label}</div>;
                }
              }
            } catch { /* ignore */ }
            return null;
          })()}
        </>
      ) : (
        <>
          <div style={{ fontSize: 16, fontWeight: 600, color: "var(--foreground)", marginBottom: 4 }}>Detecting your emotional state</div>
          <div style={{ fontSize: 12, color: "var(--muted-foreground)", marginBottom: 4, lineHeight: 1.5 }}>
            Record a voice note to detect your emotional state automatically.
          </div>
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", opacity: 0.7 }}>
            Emotions are detected from voice analysis, health data, and EEG signals.
          </div>
        </>
      )}

      {/* Readiness arc (smaller, below emotion) */}
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 8 }}>
        <svg width={130} height={130} viewBox="0 0 130 130">
          <defs>
            <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#0891b2" />
              <stop offset="100%" stopColor="#0e7490" />
            </linearGradient>
          </defs>
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--muted)" strokeWidth={7}
            strokeDasharray={`${totalArc} ${circumference - totalArc}`}
            strokeLinecap="round" transform={`rotate(135 ${cx} ${cy})`} />
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="url(#arcGrad)" strokeWidth={7}
            strokeDasharray={`${filled} ${circumference - filled}`}
            strokeLinecap="round" transform={`rotate(135 ${cx} ${cy})`}
            style={{ transition: "stroke-dasharray 1.2s cubic-bezier(0.22, 1, 0.36, 1)" }} />
          <text x={cx} y={cy - 4} textAnchor="middle" fill="var(--foreground)" fontSize={32} fontWeight={700}
            fontFamily="system-ui, -apple-system, sans-serif">{score}</text>
          <text x={cx} y={cy + 14} textAnchor="middle" fill="var(--muted-foreground)" fontSize={10}
            fontFamily="system-ui, -apple-system, sans-serif">Score</text>
        </svg>
      </div>

      <p style={{ fontSize: 12, color: score === 0 ? "var(--muted-foreground)" : "#0891b2", margin: 0, textAlign: "center" }}>
        {label}
      </p>
    </div>
  );
}

// ── Mini Score Card ────────────────────────────────────────────────────────

function MiniCard({
  label,
  value,
  sub,
  valueColor,
  onClick,
  trend,
}: {
  label: string;
  value: string;
  sub: string;
  valueColor: string;
  onClick?: () => void;
  /** "up" = improved, "down" = declined, "same" = unchanged, undefined = no data */
  trend?: "up" | "down" | "same";
}) {
  return (
    <div
      onClick={onClick}
      style={{
        background: "var(--card)",
        border: "1px solid var(--border)",
        borderRadius: 14,
        padding: "14px 10px",
        textAlign: "center",
        position: "relative",
        cursor: onClick ? "pointer" : "default",
      }}
    >
      <p style={{ fontSize: 11, color: "var(--muted-foreground)", margin: "0 0 6px 0" }}>{label}</p>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 4 }}>
        <p style={{ fontSize: 22, fontWeight: 700, color: valueColor, margin: "0", lineHeight: 1 }}>
          {value}
        </p>
        {trend && (
          <span style={{
            fontSize: 12,
            color: trend === "up" ? "#4ade80" : trend === "down" ? "#e87676" : "var(--muted-foreground)",
            lineHeight: 1,
          }}>
            {trend === "up" ? "↑" : trend === "down" ? "↓" : "→"}
          </span>
        )}
      </div>
      <p style={{ fontSize: 10, color: "var(--muted-foreground)", margin: "4px 0 0 0" }}>{sub}</p>
      {onClick && (
        <span style={{ color: "var(--muted-foreground)", fontSize: 16, position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)" }}>›</span>
      )}
    </div>
  );
}

// ── Sleep Stage Bar ────────────────────────────────────────────────────────

function SleepStageBar({
  deep,
  rem,
  light,
  awake,
  total,
}: {
  deep: number;
  rem: number;
  light: number;
  awake: number;
  total: number;
}) {
  const safeTotal = total || 1;
  const pDeep = (deep / safeTotal) * 100;
  const pRem = (rem / safeTotal) * 100;
  const pLight = (light / safeTotal) * 100;
  const pAwake = (awake / safeTotal) * 100;

  return (
    <div>
      <div
        style={{
          display: "flex",
          height: 6,
          borderRadius: 3,
          overflow: "hidden",
          background: "var(--border)",
          marginBottom: 6,
        }}
      >
        <div style={{ width: `${pDeep}%`, background: "#6366f1" }} />
        <div style={{ width: `${pLight}%`, background: "#818cf8" }} />
        <div style={{ width: `${pRem}%`, background: "#c084fc" }} />
        <div style={{ width: `${pAwake}%`, background: "#374151" }} />
      </div>
      <div style={{ display: "flex", gap: 10 }}>
        {[
          { label: "Deep", color: "#6366f1" },
          { label: "Light", color: "#818cf8" },
          { label: "REM", color: "#c084fc" },
          { label: "Awake", color: "#6b7280" },
        ].map(({ label, color }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: color }} />
            <span style={{ fontSize: 9, color: "var(--muted-foreground)" }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function Today() {
  const { latestPayload, lastSyncAt } = useHealthSync();
  const userId = useMemo(() => getParticipantId(), []);
  const [, navigate] = useLocation();
  const voiceData = useVoiceData();
  const [showBreathe, setShowBreathe] = useState(false);

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

  // Fetch food logs for today
  const { data: foodLogs } = useQuery<FoodLog[]>({
    queryKey: [resolveUrl(`/api/food/logs/${userId}`)],
    retry: false,
  });

  const today = new Date().toISOString().slice(0, 10);
  const todayCalories = useMemo(() => {
    if (!foodLogs) return 0;
    return foodLogs
      .filter((l) => l.date?.startsWith(today))
      .reduce((sum, l) => sum + (l.totalCalories ?? 0), 0);
  }, [foodLogs, today]);

  const readiness = useMemo(() => computeReadiness(checkin), [checkin]);
  const aiInsight = useMemo(() => getAIInsight(checkin), [checkin]);

  // Derived values
  const emotion = checkin?.emotion ?? "—";
  const stressVal = checkin?.stress_index ?? 0;
  const focusVal = checkin?.focus_index ?? 0;
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

  // Compute trends
  function getTrend(current: number, prev: number | undefined, lowerIsBetter = false): "up" | "down" | "same" | undefined {
    if (prev === undefined) return undefined;
    const diff = current - prev;
    if (Math.abs(diff) < 0.05) return "same";
    if (lowerIsBetter) return diff < 0 ? "up" : "down"; // less stress = improvement
    return diff > 0 ? "up" : "down";
  }
  const stressTrend = getTrend(stressVal, yesterday?.stress_index, true);
  const focusTrend = getTrend(focusVal, yesterday?.focus_index);
  const moodTrend = getTrend(checkin?.valence ?? 0, yesterday?.valence);

  // Gentle haptic warning when stress is elevated
  useEffect(() => {
    if (stressVal > 0.5) hapticWarning();
  }, [stressVal]);

  const heartRate = latestPayload?.current_heart_rate ?? latestPayload?.resting_heart_rate;
  const steps = latestPayload?.steps_today ?? 0;

  const sleepTotal = latestPayload?.sleep_total_hours ?? 0;
  const sleepRem = latestPayload?.sleep_rem_hours ?? 0;
  const sleepDeep = latestPayload?.sleep_deep_hours ?? 0;
  const sleepLight = Math.max(0, sleepTotal - sleepRem - sleepDeep - 0.3);
  const sleepAwake = 0.3;
  const sleepEfficiency = latestPayload?.sleep_efficiency ?? 0;

  const calGoal = 2000;
  const calPct = Math.min(1, todayCalories / calGoal);

  const stepsGoal = 10000;
  const stepsPct = Math.min(100, Math.round((steps / stepsGoal) * 100));

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
        minHeight: "100vh",
        padding: 16,
        paddingBottom: 100,
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}
    >
      {/* ── Header ── */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 20,
        }}
      >
        <div>
          <p style={{ fontSize: 11, color: "var(--muted-foreground)", margin: "0 0 2px 0" }}>{formatDate()}</p>
          <p style={{ fontSize: 18, fontWeight: 600, color: "var(--foreground)", margin: 0 }}>
            {(() => {
              const h = new Date().getHours();
              const timeGreet = h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
              const em = checkin?.emotion;
              if (em === "happy") return `${timeGreet} 😊`;
              if (em === "sad") return `Hey, take it easy today 💙`;
              if (em === "angry") return `${timeGreet} — breathe 🌊`;
              if (em === "fear") return `You're safe. ${timeGreet} 🤗`;
              return timeGreet;
            })()}
          </p>
        </div>
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: "50%",
            background: "linear-gradient(135deg, #1db88a, #0d9668)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 15,
            fontWeight: 700,
            color: "#0a0e17",
            flexShrink: 0,
          }}
        >
          S
        </div>
      </div>

      {/* ── Readiness Score ── */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
        <EmotionHero checkin={checkin} score={readiness} />
      </div>

      {/* ── Mini Score Cards ── */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: 10,
          marginBottom: 14,
        }}
      >
        <MiniCard
          label="Mood"
          value={emotion === "—" ? "—" : emotion.charAt(0).toUpperCase() + emotion.slice(1)}
          sub={topProb > 0 ? `${Math.round(topProb * 100)}% confidence` : "No data"}
          valueColor="#0891b2"
          onClick={() => navigate("/emotions")}
          trend={moodTrend}
        />
        <MiniCard
          label="Stress"
          value={stressVal > 0 ? `${Math.round(stressVal * 100)}%` : "—"}
          sub={stressVal > 0 ? getStressLabel(stressVal) : "No data"}
          valueColor={stressVal > 0 ? getStressColor(stressVal) : "var(--muted-foreground)"}
          onClick={() => navigate("/emotions")}
          trend={stressTrend}
        />
        <MiniCard
          label="Focus"
          value={focusVal > 0 ? `${Math.round(focusVal * 100)}%` : "—"}
          sub={focusVal > 0 ? getFocusLabel(focusVal) : "No data"}
          valueColor="#3b82f6"
          onClick={() => navigate("/emotions")}
          trend={focusTrend}
        />
      </div>

      {/* ── Streak Protection Nudge ── */}
      <StreakProtector />

      {/* ── Mental Fitness Score ── */}
      <MentalFitnessCard voice={voiceData} />

      {/* ── Weekly Mood Strip ── */}
      <WeeklyMoodStrip userId={userId} />

      {/* ── AI Insight ── */}
      <div
        style={{
          background: "var(--card)",
          border: "1px solid #1f3a2e",
          borderRadius: 14,
          padding: 14,
          marginBottom: 14,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
          <Sparkles size={13} color="#0891b2" />
          <span style={{ fontSize: 11, fontWeight: 600, color: "#0891b2" }}>AI Insight</span>
        </div>
        <p style={{ fontSize: 13, color: "var(--foreground)", margin: 0, lineHeight: 1.5 }}>
          {aiInsight}
        </p>
      </div>

      {/* ── Stress Warning — appears when stress > 60% ── */}
      {(checkin?.stress_index ?? 0) > 0.6 && (
        <div
          style={{
            background: "var(--card)",
            border: "1px solid #3d1f1f",
            borderRadius: 14,
            padding: 16,
            marginBottom: 14,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
            <span style={{ fontSize: 24 }}>&#x26A0;&#xFE0F;</span>
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#e879a8" }}>Your stress levels are elevated</div>
              <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 4, lineHeight: 1.6 }}>
                Take a moment to breathe. Try the 4-7-8 breathing technique:<br />
                <span style={{ fontWeight: 500, color: "var(--foreground)" }}>
                  Inhale 4 seconds &rarr; Hold 7 seconds &rarr; Exhale 8 seconds
                </span>
              </div>
            </div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={() => window.open("https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO", "_blank")}
              style={{
                flex: 1,
                background: "linear-gradient(135deg, #1DB954, #158a3e)",
                color: "white",
                border: "none",
                borderRadius: 10,
                padding: "10px 16px",
                fontSize: 13,
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
                background: "linear-gradient(135deg, #0891b2, #0e7490)",
                color: "white",
                border: "none",
                borderRadius: 10,
                padding: "10px 16px",
                fontSize: 13,
                fontWeight: 600,
                cursor: "pointer",
              }}
            >
              Breathing Exercise
            </button>
          </div>
        </div>
      )}

      {/* Inline breathing exercise */}
      {showBreathe && (
        <InlineBreathe onClose={() => setShowBreathe(false)} />
      )}

      {/* ── Sleep Card ── */}
      <div
        onClick={() => navigate("/sleep-session")}
        style={{
          background: "var(--card)",
          border: "1px solid var(--border)",
          borderRadius: 14,
          padding: 14,
          marginBottom: 14,
          cursor: "pointer",
          position: "relative",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            marginBottom: 12,
          }}
        >
          <div>
            <p style={{ fontSize: 11, color: "var(--muted-foreground)", margin: "0 0 4px 0" }}>Sleep</p>
            <p style={{ fontSize: 20, fontWeight: 700, color: "#7c3aed", margin: 0 }}>
              {sleepTotal > 0 ? `${sleepTotal.toFixed(1)}h` : "—"}
            </p>
          </div>
          <div style={{ textAlign: "right", display: "flex", alignItems: "center", gap: 6 }}>
            <div>
              <p style={{ fontSize: 11, color: "var(--muted-foreground)", margin: "0 0 4px 0" }}>Quality</p>
              <p style={{ fontSize: 20, fontWeight: 700, color: "#7c3aed", margin: 0 }}>
                {sleepEfficiency > 0 ? `${Math.round(sleepEfficiency)}%` : "—"}
              </p>
            </div>
            <span style={{ color: "var(--muted-foreground)", fontSize: 18, lineHeight: 1 }}>›</span>
          </div>
        </div>
        <SleepStageBar
          deep={sleepDeep}
          rem={sleepRem}
          light={sleepLight}
          awake={sleepAwake}
          total={sleepTotal || 1}
        />
      </div>

      {/* ── Sleep Insights ── */}
      <SleepInsights
        sleepHours={sleepTotal > 0 ? sleepTotal : null}
        deepHours={sleepDeep > 0 ? sleepDeep : null}
        remHours={sleepRem > 0 ? sleepRem : null}
        efficiency={sleepEfficiency > 0 ? sleepEfficiency : null}
      />

      {/* ── Health Metrics ── */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 10,
          marginBottom: 14,
        }}
      >
        {/* Heart Rate */}
        <div
          onClick={() => navigate("/health-analytics")}
          style={{
            background: "var(--card)",
            border: "1px solid var(--border)",
            borderRadius: 14,
            padding: 14,
            cursor: "pointer",
            position: "relative",
          }}
        >
          <p style={{ fontSize: 11, color: "var(--muted-foreground)", margin: "0 0 6px 0" }}>Heart Rate</p>
          <p style={{ fontSize: 22, fontWeight: 700, color: "var(--foreground)", margin: "0 0 4px 0" }}>
            {heartRate ? `${Math.round(heartRate)} bpm` : "—"}
          </p>
          <p style={{ fontSize: 10, color: "#0891b2", margin: 0 }}>
            {heartRate
              ? heartRate < 60
                ? "Low — rest well"
                : heartRate < 100
                ? "Normal"
                : "Elevated"
              : "No data"}
          </p>
          <span style={{ color: "var(--muted-foreground)", fontSize: 16, position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)" }}>›</span>
        </div>

        {/* Steps */}
        <div
          onClick={() => navigate("/health-analytics")}
          style={{
            background: "var(--card)",
            border: "1px solid var(--border)",
            borderRadius: 14,
            padding: 14,
            cursor: "pointer",
            position: "relative",
          }}
        >
          <p style={{ fontSize: 11, color: "var(--muted-foreground)", margin: "0 0 6px 0" }}>Steps</p>
          <p style={{ fontSize: 22, fontWeight: 700, color: "var(--foreground)", margin: "0 0 4px 0" }}>
            {steps > 0 ? steps.toLocaleString() : "—"}
          </p>
          <p style={{ fontSize: 10, color: "var(--muted-foreground)", margin: 0 }}>
            {steps > 0 ? `${stepsPct}% of goal` : "No data"}
          </p>
          <span style={{ color: "var(--muted-foreground)", fontSize: 16, position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)" }}>›</span>
        </div>
      </div>

      {/* ── Nutrition Summary ── */}
      <div
        onClick={() => navigate("/nutrition")}
        style={{
          background: "var(--card)",
          border: "1px solid var(--border)",
          borderRadius: 14,
          padding: 14,
          cursor: "pointer",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 10,
          }}
        >
          <span style={{ fontSize: 11, color: "var(--muted-foreground)" }}>Today's Nutrition</span>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 11, color: "#d4a017" }}>
              {todayCalories > 0
                ? `${todayCalories.toLocaleString()} / ${calGoal.toLocaleString()} kcal`
                : `— / ${calGoal.toLocaleString()} kcal`}
            </span>
            <span style={{ color: "var(--muted-foreground)", fontSize: 16, lineHeight: 1 }}>›</span>
          </div>
        </div>
        <div
          style={{
            height: 6,
            borderRadius: 3,
            background: "var(--border)",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${calPct * 100}%`,
              background: "linear-gradient(90deg, #d4a017, #ea580c)",
              borderRadius: 3,
              transition: "width 0.8s cubic-bezier(0.22, 1, 0.36, 1)",
            }}
          />
        </div>
      </div>

      {/* ── Quick Listen — Music Section ── */}
      <div style={{ marginBottom: 14 }}>
        <div style={{
          fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
          textTransform: "uppercase" as const, letterSpacing: "0.5px", marginBottom: 8,
        }}>
          Quick Listen
        </div>
        <div style={{
          display: "flex", gap: 10, overflowX: "auto",
          paddingBottom: 4, scrollbarWidth: "none",
          WebkitOverflowScrolling: "touch",
        }}>
          {[
            { emoji: "\uD83C\uDFB5", title: "Focus", url: "https://open.spotify.com/playlist/37i9dQZF1DWZeKCadgRdKQ" },
            { emoji: "\uD83E\uDDD8", title: "Calm", url: "https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO" },
            { emoji: "\uD83C\uDF19", title: "Sleep", url: "https://open.spotify.com/playlist/37i9dQZF1DWZd79rJ6a7lp" },
            { emoji: "\uD83C\uDFC3", title: "Workout", url: "https://open.spotify.com/playlist/37i9dQZF1DX76Wlfdnj7AP" },
          ].map((card) => (
            <button
              key={card.title}
              onClick={() => window.open(card.url, "_blank")}
              style={{
                flex: "0 0 auto",
                width: 90,
                background: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: 14,
                padding: "12px 8px",
                textAlign: "center",
                cursor: "pointer",
                transition: "transform 0.15s",
              }}
              onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.96)")}
              onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
              onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
            >
              <div style={{ fontSize: 24, marginBottom: 4 }}>{card.emoji}</div>
              <div style={{ fontSize: 11, fontWeight: 600, color: "var(--foreground)" }}>{card.title}</div>
            </button>
          ))}
        </div>
      </div>

      {/* ── Tomorrow's Forecast ── */}
      <ForecastCard userId={userId} />

      {/* ── Weekly Summary Card (shows on Sun/Mon or when 3+ days data) ── */}
      <WeeklySummaryCard userId={userId} />

      {/* ── Daily Wellness Tip ── */}
      <DailyTip emotion={emotion} stress={stressVal} focus={focusVal} />
    </motion.main>
    </>
  );
}
