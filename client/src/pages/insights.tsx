import { useState, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Link } from "wouter";
import {
  Sparkles, Brain, Moon, Heart, Zap, UtensilsCrossed,
  TrendingUp, TrendingDown, Minus, ChevronRight, ArrowRight,
  Wind, Target, Activity, Lightbulb, Sun, AlertCircle,
  BarChart3, Flame, Droplets, Leaf,
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine,
} from "recharts";
import { pageTransition, cardVariants } from "@/lib/animations";
import { getParticipantId } from "@/lib/participant";
import { EEGPeakHours } from "@/components/eeg-peak-hours";

// ── Types ──────────────────────────────────────────────────────────────────

interface EmotionEntry {
  stress: number;
  happiness: number;
  focus: number;
  dominantEmotion: string;
  timestamp: string;
  valence?: number;
  arousal?: number;
}

interface FoodLogEntry {
  id: string;
  loggedAt: string;
  totalCalories: number | null;
  mealType: string | null;
  summary: string | null;
  dominantMacro: string | null;
}

interface PersonalBaseline {
  avgStress: number;
  avgFocus: number;
  avgHappiness: number;
  avgCalories: number;
  sampleCount: number;
}

interface InsightCard {
  id: string;
  icon: typeof Sparkles;
  category: "emotion" | "energy" | "nutrition" | "pattern" | "prediction";
  priority: "high" | "medium" | "low";
  headline: string;
  context: string;
  action: string;
  delta?: number; // positive = above baseline, negative = below
  color: string;
}

// ── Insight engine ─────────────────────────────────────────────────────────

function computeBaseline(entries: EmotionEntry[]): PersonalBaseline | null {
  if (entries.length < 3) return null;
  const n = entries.length;
  return {
    avgStress: entries.reduce((s, e) => s + e.stress, 0) / n,
    avgFocus: entries.reduce((s, e) => s + e.focus, 0) / n,
    avgHappiness: entries.reduce((s, e) => s + e.happiness, 0) / n,
    avgCalories: 0,
    sampleCount: n,
  };
}

function pct(v: number): string {
  return `${Math.round(v * 100)}%`;
}

function deltaLabel(d: number, unit = "pts"): string {
  if (Math.abs(d) < 0.02) return "on par with your average";
  const dir = d > 0 ? "above" : "below";
  return `${Math.round(Math.abs(d) * 100)} ${unit} ${dir} your average`;
}

function generateInsights(
  entries: EmotionEntry[],
  baseline: PersonalBaseline | null,
  foodLogs: FoodLogEntry[],
): InsightCard[] {
  const cards: InsightCard[] = [];
  if (!entries.length) return cards;

  const recent = entries.slice(-7); // last 7 readings
  const today = entries.filter(e => {
    const d = new Date(e.timestamp);
    const now = new Date();
    return d.toDateString() === now.toDateString();
  });
  const latest = entries[entries.length - 1];

  // ── Stress trend ──────────────────────────────────────────────────────
  if (recent.length >= 3) {
    const avgRecent = recent.reduce((s, e) => s + e.stress, 0) / recent.length;
    const delta = baseline ? avgRecent - baseline.avgStress : 0;
    if (avgRecent > 0.55) {
      cards.push({
        id: "stress-high",
        icon: AlertCircle,
        category: "emotion",
        priority: "high",
        headline: `Your stress has been elevated (${pct(avgRecent)})`,
        context: baseline
          ? `This is ${deltaLabel(delta)} across your last ${recent.length} check-ins. High-stress runs suppress focus and impair sleep quality — you may notice both effects today.`
          : `Stress has averaged ${pct(avgRecent)} recently — above the healthy zone. Sustained high-beta brain activity depletes prefrontal cortex resources over time.`,
        action: "Try 4-7-8 breathing (inhale 4s, hold 7s, exhale 8s) before your next task. Even 2 cycles lower cortisol measurably.",
        delta,
        color: "#f87171",
      });
    } else if (avgRecent < 0.35 && (baseline?.avgStress ?? 0.5) > 0.4) {
      cards.push({
        id: "stress-low",
        icon: Leaf,
        category: "emotion",
        priority: "medium",
        headline: `Stress is notably low today (${pct(avgRecent)})`,
        context: baseline
          ? `${deltaLabel(delta)} — one of your calmer stretches. Low-stress states enhance creative thinking and memory consolidation.`
          : `Stress at ${pct(avgRecent)} puts you in the recovery zone — ideal for creative or collaborative work.`,
        action: "Schedule your most creative or strategic tasks now. Your prefrontal cortex is operating at peak capacity.",
        delta,
        color: "#4ade80",
      });
    }
  }

  // ── Focus trend ───────────────────────────────────────────────────────
  if (recent.length >= 3) {
    const avgFocus = recent.reduce((s, e) => s + e.focus, 0) / recent.length;
    const delta = baseline ? avgFocus - baseline.avgFocus : 0;
    if (avgFocus > 0.65 && (delta > 0.05 || !baseline)) {
      cards.push({
        id: "focus-peak",
        icon: Target,
        category: "energy",
        priority: "medium",
        headline: `Focus is ${pct(avgFocus)} — above your usual`,
        context: baseline
          ? `You're running ${deltaLabel(delta)} for focus. Your beta/alpha ratio suggests sustained attention capacity. This window typically lasts 60–90 minutes.`
          : `High focus state detected. Elevated beta activity with stable alpha underneath — the neuroscience signature of deep work readiness.`,
        action: "Start your hardest task now. Close notifications, set a 45-minute block, and go deep.",
        delta,
        color: "#60a5fa",
      });
    }
    if (avgFocus < 0.35) {
      cards.push({
        id: "focus-low",
        icon: Activity,
        category: "energy",
        priority: "medium",
        headline: `Focus is diffuse right now (${pct(avgFocus)})`,
        context: "Low beta activity and elevated alpha suggest your brain is in an unfocused, mind-wandering state. This often follows a long work session or insufficient sleep.",
        action: "5 minutes of brisk walking increases prefrontal blood flow and restores attention. Don't push through — move first.",
        delta: baseline ? avgFocus - baseline.avgFocus : undefined,
        color: "#fbbf24",
      });
    }
  }

  // ── Mood pattern ──────────────────────────────────────────────────────
  if (today.length >= 2) {
    const earlyMood = today.slice(0, Math.ceil(today.length / 2)).reduce((s, e) => s + e.happiness, 0) / Math.ceil(today.length / 2);
    const lateMood = today.slice(-Math.ceil(today.length / 2)).reduce((s, e) => s + e.happiness, 0) / Math.ceil(today.length / 2);
    const shift = lateMood - earlyMood;
    if (Math.abs(shift) > 0.12) {
      const rising = shift > 0;
      cards.push({
        id: "mood-shift",
        icon: rising ? TrendingUp : TrendingDown,
        category: "pattern",
        priority: "medium",
        headline: `Your mood is ${rising ? "improving" : "declining"} through the day`,
        context: rising
          ? `Positivity has risen ${Math.round(shift * 100)} points since this morning — a pattern linked to completed tasks and social interaction. Keep the momentum.`
          : `Mood has dipped ${Math.round(Math.abs(shift) * 100)} points since morning. Afternoon energy crashes often accompany post-lunch glucose drops.`,
        action: rising
          ? "Capture what's working today — a brief note about what went well reinforces the pattern."
          : "A short outdoor break (even 5 min of natural light) resets the cortisol rhythm and lifts afternoon mood.",
        color: rising ? "#4ade80" : "#f87171",
      });
    }
  }

  // ── Dominant emotion pattern ──────────────────────────────────────────
  if (entries.length >= 5) {
    const emotionCounts: Record<string, number> = {};
    entries.slice(-14).forEach(e => {
      emotionCounts[e.dominantEmotion] = (emotionCounts[e.dominantEmotion] ?? 0) + 1;
    });
    const topEmotion = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0];
    if (topEmotion && topEmotion[1] >= 4) {
      const emotionMessages: Record<string, { context: string; action: string; color: string }> = {
        happy: {
          context: "Happiness has been your most frequent state recently — linked to approach motivation, social openness, and improved immune function.",
          action: "Share it: positive emotional contagion is real. A short message to someone you care about extends your own positive state.",
          color: "#4ade80",
        },
        sad: {
          context: "Sadness has dominated recent readings. This emotion signals a need for rest or reconnection — not a flaw to fix, but information to act on.",
          action: "Identify one small thing you can look forward to today. Future-positive thinking is the most effective short-term mood intervention.",
          color: "#818cf8",
        },
        angry: {
          context: "Anger is showing up frequently. It's an approach emotion — your brain is primed to act. Channel this toward something meaningful.",
          action: "Identify the source (people, situation, unmet need) and write it down. Naming it reduces amygdala reactivity within minutes.",
          color: "#f87171",
        },
        neutral: {
          context: "Neutral is your most common state — a sign of emotional stability and regulation. The baseline most people spend 60–70% of their waking hours in.",
          action: "Neutral is the foundation. Use it to plan deliberately — what emotion do you want to cultivate today?",
          color: "#94a3b8",
        },
        fear: {
          context: "Fear has been showing up frequently. This emotion activates the sympathetic nervous system and narrows attention — useful in bursts, draining when sustained.",
          action: "Ground yourself: 5 things you can see, 4 you can touch, 3 you hear. This activates the prefrontal cortex and reduces amygdala activation.",
          color: "#fbbf24",
        },
      };
      const msg = emotionMessages[topEmotion[0]];
      if (msg) {
        cards.push({
          id: "dominant-emotion",
          icon: Heart,
          category: "pattern",
          priority: "low",
          headline: `"${topEmotion[0]}" has been your most frequent emotion (${topEmotion[1]}× this week)`,
          context: msg.context,
          action: msg.action,
          color: msg.color,
        });
      }
    }
  }

  // ── Food-emotion correlation ───────────────────────────────────────────
  if (foodLogs.length >= 3 && entries.length >= 5) {
    const highCalDays = foodLogs
      .filter(f => (f.totalCalories ?? 0) > 2200)
      .map(f => f.loggedAt.split("T")[0]);

    const stressOnHighCalDays = entries.filter(e => {
      const day = new Date(e.timestamp).toISOString().split("T")[0];
      return highCalDays.includes(day);
    });
    const stressOtherDays = entries.filter(e => {
      const day = new Date(e.timestamp).toISOString().split("T")[0];
      return !highCalDays.includes(day);
    });

    if (stressOnHighCalDays.length >= 2 && stressOtherDays.length >= 2) {
      const avgStressHigh = stressOnHighCalDays.reduce((s, e) => s + e.stress, 0) / stressOnHighCalDays.length;
      const avgStressOther = stressOtherDays.reduce((s, e) => s + e.stress, 0) / stressOtherDays.length;
      const diff = avgStressHigh - avgStressOther;
      if (Math.abs(diff) > 0.1) {
        cards.push({
          id: "food-stress-correlation",
          icon: UtensilsCrossed,
          category: "nutrition",
          priority: "medium",
          headline: diff > 0
            ? `On high-calorie days, your stress runs ${Math.round(diff * 100)} points higher`
            : `High-calorie days correlate with ${Math.round(Math.abs(diff) * 100)} lower stress`,
          context: diff > 0
            ? `Across ${stressOnHighCalDays.length} high-calorie days in your history, stress was consistently elevated vs lighter days. Large meals trigger digestive effort that can increase cortisol.`
            : `Interestingly, higher-calorie days in your data correlate with lower stress — possibly because you eat more on comfortable, low-stress days.`,
          action: diff > 0
            ? "Try keeping lunch under 600 kcal and notice whether afternoon stress softens. The gut-brain axis responds within 2–3 hours."
            : "Your eating patterns look well-calibrated to your stress cycle. Keep your current timing.",
          delta: diff,
          color: diff > 0 ? "#f87171" : "#4ade80",
        });
      }
    }
  }

  // ── Weekly narrative prediction ───────────────────────────────────────
  if (entries.length >= 5 && latest) {
    const last3 = entries.slice(-3);
    const trend = last3[last3.length - 1].happiness - last3[0].happiness;
    if (Math.abs(trend) > 0.08) {
      cards.push({
        id: "prediction",
        icon: Sparkles,
        category: "prediction",
        priority: "low",
        headline: trend > 0
          ? "Positive momentum: your wellbeing trend is improving"
          : "Watch for fatigue: your wellbeing is trending down",
        context: trend > 0
          ? `Your last 3 readings show a ${Math.round(trend * 100)}-point upswing in happiness — an early signal of an improving cycle. Protect the habits driving this.`
          : `A ${Math.round(Math.abs(trend) * 100)}-point drop across recent readings suggests accumulating fatigue or stress. This often precedes a mood dip if not addressed.`,
        action: trend > 0
          ? "Identify what changed in the last 48h (sleep, exercise, social contact) — that's the variable to protect."
          : "Prioritize sleep tonight. Even 30 extra minutes of sleep reduces next-day stress by 18–23% on average.",
        color: trend > 0 ? "#a78bfa" : "#fbbf24",
      });
    }
  }

  // Sort: high priority first
  const pOrder = { high: 0, medium: 1, low: 2 };
  return cards.sort((a, b) => pOrder[a.priority] - pOrder[b.priority]);
}

// ── Weekly stats ──────────────────────────────────────────────────────────

function computeWeeklyStats(entries: EmotionEntry[]) {
  const week = entries.filter(e => {
    const d = new Date(e.timestamp);
    return Date.now() - d.getTime() < 7 * 86400000;
  });
  if (!week.length) return null;
  return {
    avgStress: week.reduce((s, e) => s + e.stress, 0) / week.length,
    avgFocus: week.reduce((s, e) => s + e.focus, 0) / week.length,
    avgHappiness: week.reduce((s, e) => s + e.happiness, 0) / week.length,
    checkIns: week.length,
    topEmotion: (() => {
      const c: Record<string, number> = {};
      week.forEach(e => { c[e.dominantEmotion] = (c[e.dominantEmotion] ?? 0) + 1; });
      return Object.entries(c).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "—";
    })(),
  };
}

// ── Category badge ────────────────────────────────────────────────────────

const CATEGORY_LABELS: Record<string, string> = {
  emotion: "Emotion",
  energy: "Energy",
  nutrition: "Food",
  pattern: "Pattern",
  prediction: "Prediction",
};

const CATEGORY_COLORS: Record<string, string> = {
  emotion: "rgba(168, 85, 247, 0.15)",
  energy: "rgba(96, 165, 250, 0.15)",
  nutrition: "rgba(74, 222, 128, 0.15)",
  pattern: "rgba(251, 191, 36, 0.15)",
  prediction: "rgba(167, 139, 250, 0.15)",
};

// ── Metric chip ───────────────────────────────────────────────────────────

function MetricChip({
  label,
  value,
  delta,
  color,
}: {
  label: string;
  value: string;
  delta?: number;
  color: string;
}) {
  const DeltaIcon = delta === undefined ? Minus : delta > 0.02 ? TrendingUp : delta < -0.02 ? TrendingDown : Minus;
  return (
    <div className="flex flex-col gap-1 p-3 rounded-2xl" style={{ background: "var(--glass-bg)", border: "1px solid var(--glass-border)" }}>
      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p>
      <p className="text-lg font-bold" style={{ color }}>{value}</p>
      {delta !== undefined && (
        <div className="flex items-center gap-0.5 text-[10px] text-muted-foreground">
          <DeltaIcon className="h-3 w-3" />
          <span>{Math.abs(delta) < 0.02 ? "on track" : `${delta > 0 ? "+" : ""}${Math.round(delta * 100)}pts`}</span>
        </div>
      )}
    </div>
  );
}

// ── Insight card component ────────────────────────────────────────────────

function InsightCardComponent({
  card,
  index,
}: {
  card: InsightCard;
  index: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const Icon = card.icon;

  return (
    <motion.div
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      custom={index}
      onClick={() => setExpanded(p => !p)}
      className="rounded-2xl cursor-pointer transition-all duration-200 active:scale-[0.98]"
      style={{
        background: "var(--glass-bg)",
        border: `1px solid ${card.color}22`,
        backdropFilter: "blur(20px)",
      }}
    >
      <div className="p-4">
        {/* Header */}
        <div className="flex items-start gap-3">
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0 mt-0.5"
            style={{ background: `${card.color}22` }}
          >
            <Icon className="h-4 w-4" style={{ color: card.color }} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span
                className="text-[10px] font-medium px-2 py-0.5 rounded-full"
                style={{ background: CATEGORY_COLORS[card.category], color: card.color }}
              >
                {CATEGORY_LABELS[card.category]}
              </span>
              {card.priority === "high" && (
                <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-red-500/15 text-red-400">
                  Priority
                </span>
              )}
            </div>
            <p className="text-sm font-semibold leading-snug text-foreground">{card.headline}</p>
          </div>
          <ChevronRight
            className="h-4 w-4 text-muted-foreground shrink-0 transition-transform duration-200"
            style={{ transform: expanded ? "rotate(90deg)" : "rotate(0deg)" }}
          />
        </div>

        {/* Context + Action — expanded */}
        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="mt-3 pt-3 space-y-3 border-t border-white/5">
                <p className="text-xs leading-relaxed text-muted-foreground">{card.context}</p>
                <div
                  className="flex items-start gap-2 p-3 rounded-xl"
                  style={{ background: `${card.color}11`, border: `1px solid ${card.color}22` }}
                >
                  <ArrowRight className="h-3.5 w-3.5 mt-0.5 shrink-0" style={{ color: card.color }} />
                  <p className="text-xs font-medium" style={{ color: card.color }}>{card.action}</p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {!expanded && (
          <p className="mt-2 ml-12 text-xs text-muted-foreground line-clamp-2">{card.context}</p>
        )}
      </div>
    </motion.div>
  );
}

// ── Main component ────────────────────────────────────────────────────────

const PARTICIPANT = getParticipantId();

export default function Insights() {
  const [emotionHistory, setEmotionHistory] = useState<EmotionEntry[]>([]);
  const [foodLogs, setFoodLogs] = useState<FoodLogEntry[]>([]);
  const [trendPeriod, setTrendPeriod] = useState<"7d" | "30d">("7d");

  // Load data from localStorage
  useEffect(() => {
    try {
      const raw = localStorage.getItem("ndw_emotion_history");
      if (raw) setEmotionHistory(JSON.parse(raw));
    } catch { /* ignore */ }

    try {
      const raw = localStorage.getItem("ndw_food_logs");
      if (raw) setFoodLogs(JSON.parse(raw));
    } catch { /* ignore */ }

    // Also try last emotion
    try {
      const lastRaw = localStorage.getItem("ndw_last_emotion");
      if (lastRaw) {
        const last = JSON.parse(lastRaw);
        if (last?.result) {
          const entry: EmotionEntry = {
            stress: last.result.stress_index ?? 0.5,
            happiness: Math.max(0, Math.min(1, (last.result.valence ?? 0 + 1) / 2)),
            focus: last.result.focus_index ?? 0.5,
            dominantEmotion: last.result.emotion ?? "neutral",
            timestamp: new Date(last.timestamp).toISOString(),
            valence: last.result.valence,
            arousal: last.result.arousal,
          };
          setEmotionHistory(prev => {
            if (prev.some(e => e.timestamp === entry.timestamp)) return prev;
            return [...prev, entry].slice(-200);
          });
        }
      }
    } catch { /* ignore */ }
  }, []);

  const baseline = useMemo(() => computeBaseline(emotionHistory), [emotionHistory]);
  const insights = useMemo(() => generateInsights(emotionHistory, baseline, foodLogs), [emotionHistory, baseline, foodLogs]);
  const weeklyStats = useMemo(() => computeWeeklyStats(emotionHistory), [emotionHistory]);
  const latest = emotionHistory[emotionHistory.length - 1] ?? null;

  // Daily-bucketed trend data for chart — average all readings within each calendar day
  const trendChartData = useMemo(() => {
    const daysBack = trendPeriod === "7d" ? 7 : 30;
    const now = new Date();
    const days: { date: string; label: string; ms: number }[] = [];
    for (let i = daysBack - 1; i >= 0; i--) {
      const d = new Date(now);
      d.setDate(d.getDate() - i);
      days.push({
        date: d.toDateString(),
        label: daysBack <= 7
          ? d.toLocaleDateString([], { weekday: "short" })
          : d.toLocaleDateString([], { month: "short", day: "numeric" }),
        ms: d.getTime(),
      });
    }
    const cutoff = now.getTime() - daysBack * 86_400_000;
    const inRange = emotionHistory.filter(e => new Date(e.timestamp).getTime() >= cutoff);

    return days.map(day => {
      const entries = inRange.filter(e => new Date(e.timestamp).toDateString() === day.date);
      if (entries.length === 0) return { label: day.label, stress: null, focus: null, happiness: null };
      const n = entries.length;
      return {
        label: day.label,
        stress: Math.round(entries.reduce((s, e) => s + e.stress, 0) / n * 100),
        focus: Math.round(entries.reduce((s, e) => s + e.focus, 0) / n * 100),
        happiness: Math.round(entries.reduce((s, e) => s + e.happiness, 0) / n * 100),
      };
    });
  }, [emotionHistory, trendPeriod]);

  const hasData = emotionHistory.length > 0;

  return (
    <motion.main
      {...pageTransition}
      className="min-h-screen px-4 py-5 pb-8 max-w-lg mx-auto"
    >
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-1">
          <div
            className="w-8 h-8 rounded-xl flex items-center justify-center"
            style={{ background: "linear-gradient(135deg, #a78bfa, #818cf8)" }}
          >
            <Sparkles className="h-4 w-4 text-white" />
          </div>
          <h1 className="text-xl font-bold gradient-text">Insights</h1>
        </div>
        <p className="text-xs text-muted-foreground ml-10">
          {hasData
            ? `${emotionHistory.length} readings · personal baseline active`
            : "Start a voice check-in to build your insight engine"}
        </p>
      </div>

      {/* No data state */}
      {!hasData && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center py-16 px-4"
        >
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4"
            style={{ background: "linear-gradient(135deg, #a78bfa22, #818cf822)" }}
          >
            <Brain className="h-8 w-8 text-primary" />
          </div>
          <p className="text-base font-semibold text-foreground mb-2">Your insight engine is waiting</p>
          <p className="text-sm text-muted-foreground mb-6 leading-relaxed max-w-xs mx-auto">
            Complete a few voice check-ins and the system will start detecting patterns, baselines, and cross-domain correlations — like Oura, but built for your emotional data.
          </p>
          <Link href="/">
            <button
              className="px-5 py-2.5 rounded-xl text-sm font-semibold text-white"
              style={{ background: "linear-gradient(135deg, #a78bfa, #818cf8)" }}
            >
              Go to Today's check-in
            </button>
          </Link>
        </motion.div>
      )}

      {hasData && (
        <div className="space-y-6">
          {/* Weekly stats bar */}
          {weeklyStats && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-2xl p-4"
              style={{
                background: "linear-gradient(135deg, rgba(167,139,250,0.12), rgba(129,140,248,0.08))",
                border: "1px solid rgba(167,139,250,0.2)",
              }}
            >
              <div className="flex items-center justify-between mb-3">
                <div>
                  <p className="text-xs font-medium text-purple-300 uppercase tracking-wider">This Week</p>
                  <p className="text-sm text-muted-foreground">{weeklyStats.checkIns} readings · top emotion: <span className="text-foreground font-medium capitalize">{weeklyStats.topEmotion}</span></p>
                </div>
                <div className="text-right">
                  <p className="text-[10px] text-muted-foreground">Personal baseline</p>
                  <p className="text-xs text-purple-300 font-medium">{baseline ? `${baseline.sampleCount} data points` : "Building..."}</p>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <MetricChip
                  label="Stress"
                  value={pct(weeklyStats.avgStress)}
                  delta={baseline ? weeklyStats.avgStress - baseline.avgStress : undefined}
                  color="#f87171"
                />
                <MetricChip
                  label="Focus"
                  value={pct(weeklyStats.avgFocus)}
                  delta={baseline ? weeklyStats.avgFocus - baseline.avgFocus : undefined}
                  color="#60a5fa"
                />
                <MetricChip
                  label="Happiness"
                  value={pct(weeklyStats.avgHappiness)}
                  delta={baseline ? weeklyStats.avgHappiness - baseline.avgHappiness : undefined}
                  color="#4ade80"
                />
              </div>
            </motion.div>
          )}

          {/* ── Oura-style Trends Chart ──────────────────────────────────── */}
          {trendChartData.some(d => d.stress !== null) && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.08 }}
              className="rounded-2xl p-4"
              style={{ background: "var(--glass-bg)", border: "1px solid var(--glass-border)" }}
            >
              {/* Header + period switcher */}
              <div className="flex items-center justify-between mb-3">
                <div>
                  <p className="text-xs font-semibold text-foreground">Trends</p>
                  <p className="text-[10px] text-muted-foreground">Daily averages · personal baseline</p>
                </div>
                <div
                  className="flex gap-1 rounded-xl p-1"
                  style={{ background: "rgba(255,255,255,0.06)" }}
                >
                  {(["7d", "30d"] as const).map(p => (
                    <button
                      key={p}
                      onClick={() => setTrendPeriod(p)}
                      className="text-[11px] font-semibold px-3 py-1 rounded-lg border-none cursor-pointer transition-all duration-150"
                      style={{
                        background: trendPeriod === p ? "rgba(167,139,250,0.25)" : "transparent",
                        color: trendPeriod === p ? "#a78bfa" : "var(--muted-foreground)",
                      }}
                    >
                      {p.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              {/* Legend */}
              <div className="flex gap-4 mb-3">
                {[
                  { label: "Stress", color: "#f87171" },
                  { label: "Focus", color: "#60a5fa" },
                  { label: "Happiness", color: "#4ade80" },
                ].map(({ label, color }) => (
                  <div key={label} className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full" style={{ background: color }} />
                    <span className="text-[10px] text-muted-foreground">{label}</span>
                  </div>
                ))}
              </div>

              {/* Chart */}
              <ResponsiveContainer width="100%" height={140}>
                <AreaChart
                  data={trendChartData}
                  margin={{ left: -28, right: 4, top: 4, bottom: 0 }}
                >
                  <defs>
                    <linearGradient id="insGradStress" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f87171" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#f87171" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="insGradFocus" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#60a5fa" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#60a5fa" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="insGradHappy" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4ade80" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#4ade80" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 2" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis
                    dataKey="label"
                    tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
                    axisLine={false}
                    tickLine={false}
                    interval={trendPeriod === "30d" ? 4 : 0}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
                    axisLine={false}
                    tickLine={false}
                    tickCount={3}
                    tickFormatter={v => `${v}%`}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "rgba(15,15,25,0.92)", border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: 10, fontSize: 11, padding: "6px 10px",
                    }}
                    formatter={(v: number, name: string) => [`${v}%`, name]}
                    labelStyle={{ color: "rgba(255,255,255,0.5)", fontSize: 10 }}
                  />
                  {/* Personal baseline lines */}
                  {baseline && (
                    <>
                      <ReferenceLine y={Math.round(baseline.avgStress * 100)} stroke="#f87171" strokeDasharray="4 3" strokeOpacity={0.4} strokeWidth={1.5} />
                      <ReferenceLine y={Math.round(baseline.avgFocus * 100)} stroke="#60a5fa" strokeDasharray="4 3" strokeOpacity={0.4} strokeWidth={1.5} />
                      <ReferenceLine y={Math.round(baseline.avgHappiness * 100)} stroke="#4ade80" strokeDasharray="4 3" strokeOpacity={0.3} strokeWidth={1.5} />
                    </>
                  )}
                  <Area type="monotone" dataKey="stress" name="Stress" stroke="#f87171" strokeWidth={1.5} fill="url(#insGradStress)" dot={false} connectNulls />
                  <Area type="monotone" dataKey="focus" name="Focus" stroke="#60a5fa" strokeWidth={1.5} fill="url(#insGradFocus)" dot={false} connectNulls />
                  <Area type="monotone" dataKey="happiness" name="Happiness" stroke="#4ade80" strokeWidth={1.5} fill="url(#insGradHappy)" dot={false} connectNulls />
                </AreaChart>
              </ResponsiveContainer>

              {baseline && (
                <p className="text-[10px] text-muted-foreground mt-2 text-center">
                  Dashed lines = your personal baseline · {baseline.sampleCount} total readings
                </p>
              )}
            </motion.div>
          )}

          {/* Today's snapshot (if latest reading is from today) */}
          {latest && new Date(latest.timestamp).toDateString() === new Date().toDateString() && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
              className="rounded-2xl p-4"
              style={{
                background: "var(--glass-bg)",
                border: "1px solid var(--glass-border)",
              }}
            >
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">Latest Reading</p>
              <div className="flex items-center gap-4">
                <div
                  className="w-12 h-12 rounded-2xl flex items-center justify-center text-lg"
                  style={{ background: "rgba(167,139,250,0.15)" }}
                >
                  {latest.dominantEmotion === "happy" ? "😊" :
                   latest.dominantEmotion === "sad" ? "😔" :
                   latest.dominantEmotion === "angry" ? "😤" :
                   latest.dominantEmotion === "fear" ? "😨" :
                   latest.dominantEmotion === "surprise" ? "😲" : "😐"}
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold capitalize text-foreground">{latest.dominantEmotion}</p>
                  <p className="text-xs text-muted-foreground">
                    Stress {pct(latest.stress)} · Focus {pct(latest.focus)}
                    {baseline && (
                      <span className="ml-1 text-purple-400">
                        · {latest.stress < baseline.avgStress ? "↓ below" : "↑ above"} your avg stress
                      </span>
                    )}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-[10px] text-muted-foreground">
                    {new Date(latest.timestamp).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          {/* Insight cards */}
          {insights.length > 0 ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-sm font-semibold text-foreground">Smart Insights</p>
                <span className="text-xs text-muted-foreground">{insights.length} signals detected</span>
              </div>
              {insights.map((card, i) => (
                <InsightCardComponent key={card.id} card={card} index={i} />
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-sm text-muted-foreground">Keep logging — insights appear after a few readings.</p>
            </div>
          )}

          {/* Cross-domain explorer */}
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl p-4"
            style={{
              background: "var(--glass-bg)",
              border: "1px solid var(--glass-border)",
            }}
          >
            {emotionHistory.length > 0 && (
              <div className="mb-4">
                <EEGPeakHours history={emotionHistory} />
              </div>
            )}
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">Explore Patterns</p>
            <div className="grid grid-cols-2 gap-2">
              {[
                { label: "Emotion Trends", icon: Heart, href: "/discover", color: "#e879a8" },
                { label: "Stress History", icon: Activity, href: "/discover", color: "#f87171" },
                { label: "Food & Mood", icon: UtensilsCrossed, href: "/food-emotion", color: "#4ade80" },
                { label: "Brain States", icon: Brain, href: "/brain-monitor", color: "#60a5fa" },
              ].map(({ label, icon: Icon, href, color }) => (
                <Link key={label} href={href}>
                  <div
                    className="flex items-center gap-2.5 p-3 rounded-xl cursor-pointer active:scale-95 transition-transform"
                    style={{ background: `${color}11`, border: `1px solid ${color}22` }}
                  >
                    <Icon className="h-4 w-4" style={{ color }} />
                    <span className="text-xs font-medium text-foreground">{label}</span>
                  </div>
                </Link>
              ))}
            </div>
          </motion.div>

          {/* Data quality note */}
          <div className="text-center py-2">
            <p className="text-[10px] text-muted-foreground">
              Insights improve with every check-in · {emotionHistory.length} readings recorded
            </p>
          </div>
        </div>
      )}
    </motion.main>
  );
}
