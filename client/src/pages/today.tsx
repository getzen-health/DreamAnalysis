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
  padding: 16,
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
          fontSize: 13,
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
}: {
  label: string;
  value: string;
  statusLabel: string;
  dotColor: string;
  onClick?: () => void;
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
        padding: "16px 8px",
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
          fontSize: 24,
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
}: {
  label: string;
  value: string;
  unit: string;
  statusLabel: string;
  dotColor: string;
  onClick?: () => void;
  barPercent?: number;
  barGradient?: string;
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
        gap: 10,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
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
          <span style={{ fontSize: 10, color: "var(--muted-foreground)" }}>{statusLabel}</span>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
        <span
          style={{
            fontSize: 24,
            fontWeight: 700,
            color: "var(--foreground)",
            lineHeight: 1,
          }}
        >
          {value}
        </span>
        <span style={{ fontSize: 12, color: "var(--muted-foreground)" }}>{unit}</span>
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
    </motion.div>
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
          padding: "16px 16px 100px 16px",
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
                  fontSize: 12,
                  color: "var(--muted-foreground)",
                  margin: "0 0 4px 0",
                  letterSpacing: "0.3px",
                }}
              >
                {formatDate()}
              </p>
              <p
                style={{
                  fontSize: 20,
                  fontWeight: 600,
                  color: "var(--foreground)",
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
            }}
          >
            <WellnessGauge score={readiness} />
          </motion.div>

          {/* ── 3. Score Row (Mood / Stress / Focus) ── */}
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              gap: 10,
              marginBottom: 20,
            }}
          >
            <ScoreCard
              label="Mood"
              value={moodDisplay}
              statusLabel={moodStatusLabel}
              dotColor={moodDotColor}
              onClick={() => navigate("/mood")}
            />
            <ScoreCard
              label="Stress"
              value={stressDisplay}
              statusLabel={stressStatusLabel}
              dotColor={stressDotColor}
              onClick={() => navigate("/stress")}
            />
            <ScoreCard
              label="Focus"
              value={focusDisplay}
              statusLabel={focusStatusLabel}
              dotColor={focusDotColor}
              onClick={() => navigate("/focus")}
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
                fontSize: 13,
                color: "var(--foreground)",
                margin: 0,
                lineHeight: 1.6,
              }}
            >
              {aiInsight}
            </p>
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
                <span style={{ fontSize: 22, lineHeight: 1, flexShrink: 0 }}>&#x26A0;&#xFE0F;</span>
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
                      fontSize: 12,
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
                    background: "linear-gradient(135deg, #7c3aed, #6d28d9)",
                    color: "white",
                    border: "none",
                    borderRadius: 12,
                    padding: "10px 16px",
                    fontSize: 13,
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
              gap: 10,
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
            />

            {/* Heart Rate */}
            <HealthMetricCard
              label="Heart Rate"
              value={heartRate ? `${Math.round(heartRate)}` : "---"}
              unit={heartRate ? "bpm" : ""}
              statusLabel={hrStatus.label}
              dotColor={hrStatus.color}
              onClick={() => navigate("/health-analytics")}
            />

            {/* Steps */}
            <HealthMetricCard
              label="Steps"
              value={steps > 0 ? steps.toLocaleString() : "---"}
              unit={steps > 0 ? `${stepsPct}%` : ""}
              statusLabel={stepsStatus.label}
              dotColor={stepsStatus.color}
              onClick={() => navigate("/health-analytics")}
              barPercent={steps > 0 ? stepsPct : undefined}
              barGradient="linear-gradient(90deg, #06b6d4, #22d3ee)"
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
            />
          </motion.div>
        </motion.div>
      </motion.main>
    </>
  );
}
