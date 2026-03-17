import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { resolveUrl } from "@/lib/queryClient";
import { getParticipantId } from "@/lib/participant";
import { useHealthSync } from "@/hooks/use-health-sync";

// ── Emotion data from localStorage ──────────────────────────────────────

interface CheckinData {
  emotion?: string;
  valence?: number;
  stress_index?: number;
  focus_index?: number;
  relaxation_index?: number;
  confidence?: number;
}

const EMOTION_COLOR: Record<string, string> = {
  happy: "#34d399", sad: "#60a5fa", angry: "#f87171", fear: "#a78bfa",
  surprise: "#fbbf24", neutral: "#94a3b8",
};

function useCheckinData(): CheckinData | null {
  const [data, setData] = useState<CheckinData | null>(null);
  useEffect(() => {
    try {
      const raw = localStorage.getItem("ndw_last_emotion");
      if (raw) {
        const parsed = JSON.parse(raw);
        setData(parsed?.result ?? parsed);
      }
    } catch { /* ignore */ }
    const handler = () => {
      try {
        const raw = localStorage.getItem("ndw_last_emotion");
        if (raw) setData(JSON.parse(raw)?.result ?? JSON.parse(raw));
      } catch { /* ignore */ }
    };
    window.addEventListener("ndw-voice-updated", handler);
    return () => window.removeEventListener("ndw-voice-updated", handler);
  }, []);
  return data;
}

// ── Types ──────────────────────────────────────────────────────────────────

interface FeatureCard {
  emoji: string;
  title: string;
  subtitle: string;
  route: string;
}

interface ChipItem {
  emoji: string;
  label: string;
  route: string;
}

// ── Data ───────────────────────────────────────────────────────────────────

const GRID_CARDS: FeatureCard[] = [
  {
    emoji: "🧘",
    title: "Inner Energy",
    subtitle: "Chakras & aura vitality",
    route: "/inner-energy",
  },
  {
    emoji: "🤖",
    title: "AI Companion",
    subtitle: "Your wellness coach",
    route: "/ai-companion",
  },
  {
    emoji: "🧠",
    title: "Brain Monitor",
    subtitle: "Live EEG waveforms",
    route: "/brain-monitor",
  },
  {
    emoji: "🌃",
    title: "Dreams",
    subtitle: "Record & analyze dreams",
    route: "/dreams",
  },
];

const ROW_CARDS: FeatureCard[] = [
  {
    emoji: "🎯",
    title: "Neurofeedback",
    subtitle: "Train your focus",
    route: "/neurofeedback",
  },
  {
    emoji: "😴",
    title: "Sleep Session",
    subtitle: "Guided sleep protocol",
    route: "/sleep-session",
  },
];

const CHIP_ITEMS: ChipItem[] = [
  { emoji: "💡", label: "Insights", route: "/insights" },
  { emoji: "📊", label: "Weekly Summary", route: "/weekly-summary" },
  { emoji: "🎵", label: "Sleep Stories", route: "/sleep-stories" },
  { emoji: "📖", label: "CBT-i", route: "/cbti" },
];

// Sample sparkline points (normalized 0–40 in Y space, 0–280 in X)
const SPARKLINE_POINTS = [
  [0, 32],
  [40, 24],
  [80, 30],
  [120, 14],
  [160, 22],
  [200, 10],
  [240, 18],
  [280, 8],
] as [number, number][];

function pointsToPolyline(pts: [number, number][]): string {
  return pts.map(([x, y]) => `${x},${y}`).join(" ");
}

function pointsToArea(pts: [number, number][]): string {
  if (pts.length === 0) return "";
  const first = pts[0];
  const last = pts[pts.length - 1];
  const line = pts.map(([x, y]) => `${x},${y}`).join(" L ");
  return `M ${first[0]},${first[1]} L ${line} L ${last[0]},40 L ${first[0]},40 Z`;
}

// ── Sub-components ─────────────────────────────────────────────────────────

function GridCard({
  card,
  onClick,
}: {
  card: FeatureCard;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        background: "var(--card)",
        border: "1px solid var(--border)",
        borderRadius: 14,
        padding: 16,
        textAlign: "left",
        cursor: "pointer",
        width: "100%",
        WebkitTapHighlightColor: "transparent",
      }}
    >
      <div style={{ fontSize: 28, marginBottom: 8 }}>{card.emoji}</div>
      <p
        style={{
          fontSize: 13,
          fontWeight: 600,
          color: "var(--foreground)",
          margin: "0 0 3px 0",
        }}
      >
        {card.title}
      </p>
      <p style={{ fontSize: 10, color: "var(--muted-foreground)", margin: 0 }}>{card.subtitle}</p>
    </button>
  );
}

function RowCard({
  card,
  onClick,
}: {
  card: FeatureCard;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: 1,
        background: "var(--card)",
        border: "1px solid var(--border)",
        borderRadius: 14,
        padding: 16,
        textAlign: "left",
        cursor: "pointer",
        WebkitTapHighlightColor: "transparent",
      }}
    >
      <div style={{ fontSize: 28, marginBottom: 8 }}>{card.emoji}</div>
      <p
        style={{
          fontSize: 13,
          fontWeight: 600,
          color: "var(--foreground)",
          margin: "0 0 3px 0",
        }}
      >
        {card.title}
      </p>
      <p style={{ fontSize: 10, color: "var(--muted-foreground)", margin: 0 }}>{card.subtitle}</p>
    </button>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function Discover() {
  const [, navigate] = useLocation();
  const checkin = useCheckinData();
  const { latestPayload } = useHealthSync();
  const userId = getParticipantId();

  // Fetch today's food logs for calorie sum
  const { data: foodLogs } = useQuery<{ calories?: number }[]>({
    queryKey: [resolveUrl(`/api/food/logs/${userId}`)],
    retry: false,
  });

  const emotion = checkin?.emotion ?? "—";
  const emoColor = EMOTION_COLOR[emotion] ?? "var(--muted-foreground)";
  const stress = checkin?.stress_index ?? 0;
  const focus = checkin?.focus_index ?? 0;
  const relaxation = checkin?.relaxation_index ?? (1 - stress);
  const valence = checkin?.valence ?? 0;
  const hasData = !!checkin?.emotion;

  // ── Health metric derived values ─────────────────────────────────────────
  const heartRate = latestPayload?.current_heart_rate ?? latestPayload?.resting_heart_rate ?? null;
  const steps = latestPayload?.steps_today ?? null;
  const stepsGoal = 10000;
  const stepsPercent = steps != null ? Math.round((steps / stepsGoal) * 100) : null;

  const sleepHours = latestPayload?.sleep_total_hours ?? null;
  const sleepEfficiency = latestPayload?.sleep_efficiency ?? null;
  const sleepLabel: string | null = sleepHours != null
    ? `${Math.floor(sleepHours)}h ${Math.round((sleepHours % 1) * 60)}m`
    : null;

  // Calorie sum from today's logs
  const caloriesToday = foodLogs
    ? foodLogs.reduce((sum: number, log: { calories?: number }) => sum + (log.calories ?? 0), 0)
    : null;

  // Emotion/readiness score: use valence remapped to 0-100 if available
  const emotionScore = hasData
    ? Math.round(((valence + 1) / 2) * 100)
    : null;
  const emotionLabel = hasData ? emotion : null;

  return (
    <main
      style={{
        background: "var(--background)",
        minHeight: "100vh",
        padding: 16,
        paddingBottom: 100,
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}
    >
      {/* ── Header ── */}
      <div style={{ marginBottom: 16 }}>
        <p style={{ fontSize: 18, fontWeight: 600, color: "var(--foreground)", margin: "0 0 3px 0" }}>
          Discover
        </p>
        <p style={{ fontSize: 12, color: "var(--muted-foreground)", margin: 0 }}>
          Your scores at a glance
        </p>
      </div>

      {/* ── Score Cards — Bevel style: scores first, then explore ── */}
      {hasData ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 16 }}>
          {/* Mood Score */}
          <button onClick={() => navigate("/emotions")} style={{
            background: "var(--card)", border: "1px solid var(--border)", borderRadius: 14,
            padding: "14px 12px", textAlign: "left" as const, cursor: "pointer",
          }}>
            <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginBottom: 4 }}>Mood</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: emoColor, textTransform: "capitalize" as const }}>{emotion}</div>
            <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>
              Valence {valence >= 0 ? "+" : ""}{valence.toFixed(1)}
            </div>
          </button>

          {/* Stress Score */}
          <button onClick={() => navigate("/emotions")} style={{
            background: "var(--card)", border: "1px solid var(--border)", borderRadius: 14,
            padding: "14px 12px", textAlign: "left" as const, cursor: "pointer",
          }}>
            <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginBottom: 4 }}>Stress</div>
            <div style={{
              fontSize: 22, fontWeight: 700,
              color: stress < 0.3 ? "#34d399" : stress < 0.6 ? "#fbbf24" : "#f87171",
            }}>{Math.round(stress * 100)}%</div>
            <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>
              {stress < 0.3 ? "Low" : stress < 0.6 ? "Moderate" : "High"}
            </div>
          </button>

          {/* Focus Score */}
          <button onClick={() => navigate("/emotions")} style={{
            background: "var(--card)", border: "1px solid var(--border)", borderRadius: 14,
            padding: "14px 12px", textAlign: "left" as const, cursor: "pointer",
          }}>
            <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginBottom: 4 }}>Focus</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: "#60a5fa" }}>{Math.round(focus * 100)}%</div>
            <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>
              {focus >= 0.7 ? "Sharp" : focus >= 0.4 ? "Moderate" : "Diffuse"}
            </div>
          </button>

          {/* Relaxation Score */}
          <button onClick={() => navigate("/inner-energy")} style={{
            background: "var(--card)", border: "1px solid var(--border)", borderRadius: 14,
            padding: "14px 12px", textAlign: "left" as const, cursor: "pointer",
          }}>
            <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginBottom: 4 }}>Relaxation</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: "#a78bfa" }}>{Math.round(relaxation * 100)}%</div>
            <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>
              {relaxation >= 0.6 ? "Calm" : relaxation >= 0.3 ? "Mixed" : "Tense"}
            </div>
          </button>
        </div>
      ) : (
        <div style={{
          background: "var(--card)", border: "1px solid var(--border)", borderRadius: 14,
          padding: 20, marginBottom: 16, textAlign: "center",
        }}>
          <div style={{ fontSize: 28, marginBottom: 6 }}>🎙️</div>
          <div style={{ fontSize: 13, color: "var(--muted-foreground)" }}>Do a voice check-in to see your scores</div>
        </div>
      )}

      {/* ── Health metrics row — 3 columns ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 10 }}>
        {/* Heart Rate */}
        <button
          onClick={() => navigate("/health-analytics")}
          style={{
            background: "var(--card)", border: "1px solid var(--border)", borderRadius: 14,
            padding: "14px 12px", textAlign: "left" as const, cursor: "pointer",
            WebkitTapHighlightColor: "transparent",
          }}
        >
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginBottom: 4 }}>Heart Rate</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#f87171" }}>
            {heartRate != null ? `${Math.round(heartRate)}` : "—"}
            {heartRate != null && (
              <span style={{ fontSize: 11, fontWeight: 400, color: "#f87171", marginLeft: 2 }}>bpm</span>
            )}
          </div>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>Resting</div>
        </button>

        {/* Steps */}
        <button
          onClick={() => navigate("/health-analytics")}
          style={{
            background: "var(--card)", border: "1px solid var(--border)", borderRadius: 14,
            padding: "14px 12px", textAlign: "left" as const, cursor: "pointer",
            WebkitTapHighlightColor: "transparent",
          }}
        >
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginBottom: 4 }}>Steps</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#34d399" }}>
            {steps != null ? steps.toLocaleString() : "—"}
          </div>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>
            {stepsPercent != null ? `${stepsPercent}% of 10K goal` : "No data"}
          </div>
        </button>

        {/* Sleep */}
        <button
          onClick={() => navigate("/sleep-session")}
          style={{
            background: "var(--card)", border: "1px solid var(--border)", borderRadius: 14,
            padding: "14px 12px", textAlign: "left" as const, cursor: "pointer",
            WebkitTapHighlightColor: "transparent",
          }}
        >
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginBottom: 4 }}>Sleep</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#a78bfa" }}>
            {sleepLabel ?? "—"}
          </div>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>
            {sleepEfficiency != null ? `${Math.round(sleepEfficiency)}% quality` : "No data"}
          </div>
        </button>
      </div>

      {/* ── Nutrition + Emotion row — 2 columns ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 16 }}>
        {/* Calories / Nutrition */}
        <button
          onClick={() => navigate("/nutrition")}
          style={{
            background: "var(--card)", border: "1px solid var(--border)", borderRadius: 14,
            padding: "14px 12px", textAlign: "left" as const, cursor: "pointer",
            WebkitTapHighlightColor: "transparent",
          }}
        >
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginBottom: 4 }}>Nutrition</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#f59e0b" }}>
            {caloriesToday != null && caloriesToday > 0
              ? caloriesToday.toLocaleString()
              : "—"}
            {caloriesToday != null && caloriesToday > 0 && (
              <span style={{ fontSize: 11, fontWeight: 400, color: "#f59e0b", marginLeft: 2 }}>kcal</span>
            )}
          </div>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>
            {caloriesToday != null && caloriesToday > 0
              ? `of 2,000 goal`
              : "No logs today"}
          </div>
        </button>

        {/* Emotion Score */}
        <button
          onClick={() => navigate("/emotions")}
          style={{
            background: "var(--card)", border: "1px solid var(--border)", borderRadius: 14,
            padding: "14px 12px", textAlign: "left" as const, cursor: "pointer",
            WebkitTapHighlightColor: "transparent",
          }}
        >
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginBottom: 4 }}>Emotion Score</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#2dd4bf" }}>
            {emotionScore != null ? `${emotionScore}` : "—"}
          </div>
          <div style={{
            fontSize: 10, color: "var(--muted-foreground)", marginTop: 2,
            textTransform: "capitalize" as const,
          }}>
            {emotionLabel ?? "No check-in"}
          </div>
        </button>
      </div>

      {/* ── Section label ── */}
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)", textTransform: "uppercase" as const,
        letterSpacing: "0.5px", marginBottom: 10,
      }}>
        Explore
      </div>

      {/* ── Featured Card — Emotion Trends ── */}
      <button
        onClick={() => navigate("/insights")}
        style={{
          width: "100%",
          background: "var(--card)",
          border: "1px solid #1f3a2e",
          borderRadius: 18,
          padding: 18,
          marginBottom: 14,
          cursor: "pointer",
          textAlign: "left",
          WebkitTapHighlightColor: "transparent",
          boxSizing: "border-box",
        }}
      >
        {/* Title row */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            marginBottom: 14,
          }}
        >
          <div>
            <p
              style={{
                fontSize: 14,
                fontWeight: 600,
                color: "#34d399",
                margin: "0 0 3px 0",
              }}
            >
              Emotion Trends
            </p>
            <p style={{ fontSize: 11, color: "var(--muted-foreground)", margin: 0 }}>
              7-day mood journey
            </p>
          </div>
          <span style={{ fontSize: 24 }}>📈</span>
        </div>

        {/* Sparkline SVG */}
        <svg
          viewBox="0 0 280 40"
          style={{ width: "100%", height: 40, display: "block" }}
          preserveAspectRatio="none"
        >
          <defs>
            <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#34d399" stopOpacity={0.35} />
              <stop offset="100%" stopColor="#34d399" stopOpacity={0} />
            </linearGradient>
          </defs>
          {/* Area fill */}
          <path
            d={pointsToArea(SPARKLINE_POINTS)}
            fill="url(#sparkGrad)"
          />
          {/* Line */}
          <polyline
            points={pointsToPolyline(SPARKLINE_POINTS)}
            fill="none"
            stroke="#34d399"
            strokeWidth={1.5}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        </svg>
      </button>

      {/* ── 2×2 Feature Grid ── */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 10,
          marginBottom: 10,
        }}
      >
        {GRID_CARDS.map((card) => (
          <GridCard
            key={card.route}
            card={card}
            onClick={() => navigate(card.route)}
          />
        ))}
      </div>

      {/* ── 2×1 Row ── */}
      <div
        style={{
          display: "flex",
          gap: 10,
          marginBottom: 20,
        }}
      >
        {ROW_CARDS.map((card) => (
          <RowCard
            key={card.route}
            card={card}
            onClick={() => navigate(card.route)}
          />
        ))}
      </div>

      {/* ── More (horizontal scroll chips) ── */}
      <div>
        <p
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: "var(--muted-foreground)",
            textTransform: "uppercase",
            letterSpacing: "0.5px",
            margin: "0 0 8px 0",
          }}
        >
          More
        </p>
        <div
          style={{
            display: "flex",
            overflowX: "auto",
            gap: 10,
            paddingBottom: 8,
            // Hide scrollbar on webkit
            scrollbarWidth: "none",
          }}
        >
          {CHIP_ITEMS.map((chip) => (
            <button
              key={chip.route}
              onClick={() => navigate(chip.route)}
              style={{
                background: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: 12,
                padding: "12px 16px",
                cursor: "pointer",
                whiteSpace: "nowrap",
                flexShrink: 0,
                display: "flex",
                alignItems: "center",
                gap: 6,
                WebkitTapHighlightColor: "transparent",
              }}
            >
              <span style={{ fontSize: 16 }}>{chip.emoji}</span>
              <span
                style={{
                  fontSize: 12,
                  fontWeight: 500,
                  color: "var(--foreground)",
                }}
              >
                {chip.label}
              </span>
            </button>
          ))}
        </div>
      </div>
    </main>
  );
}
