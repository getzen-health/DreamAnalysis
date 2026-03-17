import { useState, useEffect, useMemo, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { resolveUrl } from "@/lib/queryClient";
import { getParticipantId } from "@/lib/participant";
import { useHealthSync } from "@/hooks/use-health-sync";
import { Sparkles } from "lucide-react";
import { ScoreSplash } from "@/components/score-splash";

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

function getReadinessLabel(score: number): string {
  if (score === 0) return "Do a voice check-in to see your score";
  if (score >= 80) return "You're at peak performance today";
  if (score >= 60) return "You're feeling good today";
  if (score >= 40) return "Moderate readiness — pace yourself";
  return "Rest and recover today";
}

function getStressLabel(stress: number): string {
  if (stress < 0.3) return "Low";
  if (stress < 0.6) return "Moderate";
  return "High";
}

function getStressColor(stress: number): string {
  if (stress < 0.3) return "#34d399";
  if (stress < 0.6) return "#fbbf24";
  return "#f87171";
}

function getFocusLabel(focus: number): string {
  if (focus >= 0.7) return "Sharp";
  if (focus >= 0.45) return "Moderate";
  return "Diffuse";
}

function getAIInsight(checkin: EmotionCheckin | null): string {
  if (!checkin) return "Complete a voice check-in to get your personalized AI insight.";
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

// ── Emotion emojis ────────────────────────────────────────────────────────

const EMOTION_EMOJI: Record<string, string> = {
  happy: "😊", sad: "😢", angry: "😠", fear: "😨",
  surprise: "😲", neutral: "😐",
};

const EMOTION_COLOR: Record<string, string> = {
  happy: "#34d399", sad: "#60a5fa", angry: "#f87171", fear: "#a78bfa",
  surprise: "#fbbf24", neutral: "#94a3b8",
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
  const label = getReadinessLabel(score);
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
          <div style={{ fontSize: 52, lineHeight: 1 }}>{emoji}</div>
          <div style={{ fontSize: 22, fontWeight: 700, color, textTransform: "capitalize" as const }}>{emotion}</div>
          <div style={{ fontSize: 11, color: "#8b8578" }}>
            {confidence > 0 ? `${confidence}% confidence` : "via voice"} · valence {(checkin?.valence ?? 0) >= 0 ? "+" : ""}{(checkin?.valence ?? 0).toFixed(1)}
          </div>
        </>
      ) : (
        <>
          <div style={{ fontSize: 48, lineHeight: 1, opacity: 0.4 }}>🎙️</div>
          <div style={{ fontSize: 16, fontWeight: 600, color: "#8b8578" }}>How are you feeling?</div>
          <div style={{ fontSize: 11, color: "#6b7280" }}>Tap the mic button to check in</div>
        </>
      )}

      {/* Readiness arc (smaller, below emotion) */}
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 8 }}>
        <svg width={130} height={130} viewBox="0 0 130 130">
          <defs>
            <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#2dd4a0" />
              <stop offset="100%" stopColor="#059669" />
            </linearGradient>
          </defs>
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="#1a1f2e" strokeWidth={7}
            strokeDasharray={`${totalArc} ${circumference - totalArc}`}
            strokeLinecap="round" transform={`rotate(135 ${cx} ${cy})`} />
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="url(#arcGrad)" strokeWidth={7}
            strokeDasharray={`${filled} ${circumference - filled}`}
            strokeLinecap="round" transform={`rotate(135 ${cx} ${cy})`}
            style={{ transition: "stroke-dasharray 0.8s ease" }} />
          <text x={cx} y={cy - 4} textAnchor="middle" fill="#e8e0d4" fontSize={32} fontWeight={700}
            fontFamily="system-ui, -apple-system, sans-serif">{score}</text>
          <text x={cx} y={cy + 14} textAnchor="middle" fill="#8b8578" fontSize={10}
            fontFamily="system-ui, -apple-system, sans-serif">Readiness</text>
        </svg>
      </div>

      <p style={{ fontSize: 12, color: score === 0 ? "#8b8578" : "#2dd4a0", margin: 0, textAlign: "center" }}>
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
}: {
  label: string;
  value: string;
  sub: string;
  valueColor: string;
}) {
  return (
    <div
      style={{
        background: "#111827",
        border: "1px solid #1f2937",
        borderRadius: 14,
        padding: "14px 10px",
        textAlign: "center",
      }}
    >
      <p style={{ fontSize: 11, color: "#8b8578", margin: "0 0 6px 0" }}>{label}</p>
      <p style={{ fontSize: 22, fontWeight: 700, color: valueColor, margin: "0 0 4px 0", lineHeight: 1 }}>
        {value}
      </p>
      <p style={{ fontSize: 10, color: "#6b7280", margin: 0 }}>{sub}</p>
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
          background: "#1f2937",
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
            <span style={{ fontSize: 9, color: "#6b7280" }}>{label}</span>
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

  // Load last emotion check-in from localStorage
  const [checkin, setCheckin] = useState<EmotionCheckin | null>(null);
  useEffect(() => {
    try {
      const raw = localStorage.getItem("ndw_last_emotion");
      if (raw) setCheckin(JSON.parse(raw));
    } catch {
      // ignore
    }
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
    <main
      style={{
        background: "#0a0e17",
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
          <p style={{ fontSize: 11, color: "#8b8578", margin: "0 0 2px 0" }}>{formatDate()}</p>
          <p style={{ fontSize: 18, fontWeight: 600, color: "#e8e0d4", margin: 0 }}>
            Good{" "}
            {new Date().getHours() < 12
              ? "morning"
              : new Date().getHours() < 17
              ? "afternoon"
              : "evening"}
            , Sravya
          </p>
        </div>
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: "50%",
            background: "linear-gradient(135deg, #2dd4a0, #059669)",
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
          valueColor="#34d399"
        />
        <MiniCard
          label="Stress"
          value={stressVal > 0 ? `${Math.round(stressVal * 100)}%` : "—"}
          sub={stressVal > 0 ? getStressLabel(stressVal) : "No data"}
          valueColor={stressVal > 0 ? getStressColor(stressVal) : "#8b8578"}
        />
        <MiniCard
          label="Focus"
          value={focusVal > 0 ? `${Math.round(focusVal * 100)}%` : "—"}
          sub={focusVal > 0 ? getFocusLabel(focusVal) : "No data"}
          valueColor="#60a5fa"
        />
      </div>

      {/* ── AI Insight ── */}
      <div
        style={{
          background: "linear-gradient(135deg, #0f1f1a, #111827)",
          border: "1px solid #1f3a2e",
          borderRadius: 14,
          padding: 14,
          marginBottom: 14,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
          <Sparkles size={13} color="#2dd4a0" />
          <span style={{ fontSize: 11, fontWeight: 600, color: "#2dd4a0" }}>AI Insight</span>
        </div>
        <p style={{ fontSize: 13, color: "#d1cdc4", margin: 0, lineHeight: 1.5 }}>
          {aiInsight}
        </p>
      </div>

      {/* ── Stress Relief — appears when stress > 50% ── */}
      {(checkin?.stress_index ?? 0) > 0.5 && (
        <div
          style={{
            background: "linear-gradient(135deg, #1f1210, #111827)",
            border: "1px solid #2d1f18",
            borderRadius: 14,
            padding: 16,
            marginBottom: 14,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
            <span style={{ fontSize: 24 }}>😮‍💨</span>
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#f87171" }}>Your stress is elevated</div>
              <div style={{ fontSize: 11, color: "#8b8578", marginTop: 2 }}>
                A quick breathing exercise can lower stress by up to 40%
              </div>
            </div>
          </div>
          <button
            onClick={() => window.location.href = "/biofeedback"}
            style={{
              width: "100%",
              background: "linear-gradient(135deg, #2dd4a0, #059669)",
              color: "#0a0e17",
              border: "none",
              borderRadius: 10,
              padding: "10px 16px",
              fontSize: 13,
              fontWeight: 600,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 6,
            }}
          >
            🧘 Try a breathing exercise
          </button>
        </div>
      )}

      {/* ── Sleep Card ── */}
      <div
        style={{
          background: "#111827",
          border: "1px solid #1f2937",
          borderRadius: 14,
          padding: 14,
          marginBottom: 14,
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
            <p style={{ fontSize: 11, color: "#8b8578", margin: "0 0 4px 0" }}>Sleep</p>
            <p style={{ fontSize: 20, fontWeight: 700, color: "#a78bfa", margin: 0 }}>
              {sleepTotal > 0 ? `${sleepTotal.toFixed(1)}h` : "—"}
            </p>
          </div>
          <div style={{ textAlign: "right" }}>
            <p style={{ fontSize: 11, color: "#8b8578", margin: "0 0 4px 0" }}>Quality</p>
            <p style={{ fontSize: 20, fontWeight: 700, color: "#a78bfa", margin: 0 }}>
              {sleepEfficiency > 0 ? `${Math.round(sleepEfficiency)}%` : "—"}
            </p>
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
          style={{
            background: "#111827",
            border: "1px solid #1f2937",
            borderRadius: 14,
            padding: 14,
          }}
        >
          <p style={{ fontSize: 11, color: "#8b8578", margin: "0 0 6px 0" }}>Heart Rate</p>
          <p style={{ fontSize: 22, fontWeight: 700, color: "#e8e0d4", margin: "0 0 4px 0" }}>
            {heartRate ? `${Math.round(heartRate)} bpm` : "—"}
          </p>
          <p style={{ fontSize: 10, color: "#34d399", margin: 0 }}>
            {heartRate
              ? heartRate < 60
                ? "Low — rest well"
                : heartRate < 100
                ? "Normal"
                : "Elevated"
              : "No data"}
          </p>
        </div>

        {/* Steps */}
        <div
          style={{
            background: "#111827",
            border: "1px solid #1f2937",
            borderRadius: 14,
            padding: 14,
          }}
        >
          <p style={{ fontSize: 11, color: "#8b8578", margin: "0 0 6px 0" }}>Steps</p>
          <p style={{ fontSize: 22, fontWeight: 700, color: "#e8e0d4", margin: "0 0 4px 0" }}>
            {steps > 0 ? steps.toLocaleString() : "—"}
          </p>
          <p style={{ fontSize: 10, color: "#8b8578", margin: 0 }}>
            {steps > 0 ? `${stepsPct}% of goal` : "No data"}
          </p>
        </div>
      </div>

      {/* ── Nutrition Summary ── */}
      <div
        style={{
          background: "#111827",
          border: "1px solid #1f2937",
          borderRadius: 14,
          padding: 14,
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
          <span style={{ fontSize: 11, color: "#8b8578" }}>Today's Nutrition</span>
          <span style={{ fontSize: 11, color: "#f59e0b" }}>
            {todayCalories > 0
              ? `${todayCalories.toLocaleString()} / ${calGoal.toLocaleString()} kcal`
              : `— / ${calGoal.toLocaleString()} kcal`}
          </span>
        </div>
        <div
          style={{
            height: 6,
            borderRadius: 3,
            background: "#1f2937",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${calPct * 100}%`,
              background: "linear-gradient(90deg, #f59e0b, #f97316)",
              borderRadius: 3,
              transition: "width 0.6s ease",
            }}
          />
        </div>
      </div>
    </main>
    </>
  );
}
