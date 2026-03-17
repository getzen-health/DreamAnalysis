import { useLocation } from "wouter";

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
        background: "#111827",
        border: "1px solid #1f2937",
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
          color: "#e8e0d4",
          margin: "0 0 3px 0",
        }}
      >
        {card.title}
      </p>
      <p style={{ fontSize: 10, color: "#8b8578", margin: 0 }}>{card.subtitle}</p>
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
        background: "#111827",
        border: "1px solid #1f2937",
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
          color: "#e8e0d4",
          margin: "0 0 3px 0",
        }}
      >
        {card.title}
      </p>
      <p style={{ fontSize: 10, color: "#8b8578", margin: 0 }}>{card.subtitle}</p>
    </button>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function Discover() {
  const [, navigate] = useLocation();

  return (
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
      <div style={{ marginBottom: 20 }}>
        <p
          style={{
            fontSize: 18,
            fontWeight: 600,
            color: "#e8e0d4",
            margin: "0 0 3px 0",
          }}
        >
          Discover
        </p>
        <p style={{ fontSize: 12, color: "#8b8578", margin: 0 }}>
          Explore your mind and body
        </p>
      </div>

      {/* ── Featured Card — Emotion Trends ── */}
      <button
        onClick={() => navigate("/insights")}
        style={{
          width: "100%",
          background: "linear-gradient(135deg, #0f1f1a, #0f1729)",
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
            <p style={{ fontSize: 11, color: "#8b8578", margin: 0 }}>
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
            color: "#8b8578",
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
                background: "#111827",
                border: "1px solid #1f2937",
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
                  color: "#e8e0d4",
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
