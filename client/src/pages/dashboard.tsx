import { useState, useEffect, useCallback, useMemo } from "react";
import { Link } from "wouter";
import { Card } from "@/components/ui/card";
import { ScoreCircle } from "@/components/score-circle";
import {
  AreaChart,
  Area,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";
import {
  Heart,
  Brain,
  Activity,
  AlertCircle,
  ArrowRight,
  Moon,
  Headphones,
  MessageSquare,
  Lightbulb,
  HeartPulse,
  Sparkles,
  Zap,
} from "lucide-react";

/* ---------- types ---------- */
interface EmotionState {
  emotion: string;
  confidence: number;
  valence: number;
  arousal: number;
  stress_index: number;
  focus_index: number;
  relaxation_index: number;
}

interface MoodPoint {
  time: string;
  mood: number;
  stress: number;
}

/* ---------- helpers ---------- */
const EMOTION_LABELS: Record<string, string> = {
  happy: "Happy",
  sad: "Sad",
  angry: "Angry",
  fearful: "Anxious",
  relaxed: "Relaxed",
  focused: "Focused",
};

function getInsightText(emotion: EmotionState, hour: number): string {
  const isNight = hour >= 21 || hour < 6;
  const isMorning = hour >= 6 && hour < 12;

  if (emotion.stress_index > 60) {
    return "Your stress levels are elevated. Consider a 5-minute breathing exercise or a neurofeedback session to recalibrate.";
  }
  if (emotion.relaxation_index > 70 && emotion.focus_index > 60) {
    return "You're in a flow state — high focus with low stress. This is optimal for creative work and deep thinking.";
  }
  if (isNight && emotion.relaxation_index > 50) {
    return "Your body is winding down naturally. This is a good time to journal your dreams and prepare for restorative sleep.";
  }
  if (isMorning) {
    return "Morning neural patterns suggest good baseline clarity. Set an intention now to maximize your cognitive potential today.";
  }
  if (emotion.focus_index > 65) {
    return "High focus detected — your prefrontal cortex is highly active. Great time for analytical tasks.";
  }
  return "Your neural patterns are balanced. Regular check-ins help build emotional awareness over time.";
}

const CHAKRA_COLORS = [
  "hsl(0, 72%, 55%)",    // Root
  "hsl(25, 85%, 55%)",   // Sacral
  "hsl(45, 90%, 55%)",   // Solar
  "hsl(152, 60%, 48%)",  // Heart
  "hsl(200, 70%, 55%)",  // Throat
  "hsl(240, 55%, 60%)",  // Third Eye
  "hsl(280, 50%, 60%)",  // Crown
];

const CHAKRA_LABELS = ["Root", "Sacral", "Solar", "Heart", "Throat", "Eye", "Crown"];

/* ---------- quick action cards ---------- */
const QUICK_ACTIONS = [
  { href: "/brain-monitor", icon: Activity, label: "Brain Monitor", color: "hsl(200, 70%, 55%)" },
  { href: "/dreams", icon: Moon, label: "Dream Journal", color: "hsl(262, 45%, 65%)" },
  { href: "/ai-companion", icon: MessageSquare, label: "AI Companion", color: "hsl(152, 60%, 48%)" },
  { href: "/neurofeedback", icon: Headphones, label: "Neurofeedback", color: "hsl(38, 85%, 58%)" },
  { href: "/insights", icon: Lightbulb, label: "Insights", color: "hsl(320, 55%, 60%)" },
  { href: "/health-analytics", icon: HeartPulse, label: "Health", color: "hsl(4, 72%, 55%)" },
];

/* ========== Component ========== */
export default function Dashboard() {
  const [emotion, setEmotion] = useState<EmotionState>({
    emotion: "relaxed",
    confidence: 0.72,
    valence: 0.35,
    arousal: 0.42,
    stress_index: 28,
    focus_index: 65,
    relaxation_index: 72,
  });

  const [shift, setShift] = useState<{
    detected: boolean;
    type: string;
    description: string;
  } | null>(null);

  // Mood timeline (last 12 points)
  const [moodHistory, setMoodHistory] = useState<MoodPoint[]>(() => {
    const now = new Date();
    return Array.from({ length: 12 }, (_, i) => {
      const t = new Date(now.getTime() - (11 - i) * 5 * 60000);
      return {
        time: t.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" }),
        mood: 50 + Math.random() * 35,
        stress: 15 + Math.random() * 30,
      };
    });
  });

  // Brain wave sparklines (alpha, theta, beta)
  const [brainWaves, setBrainWaves] = useState({
    alpha: Array.from({ length: 30 }, () => 20 + Math.random() * 60),
    theta: Array.from({ length: 30 }, () => 15 + Math.random() * 50),
    beta: Array.from({ length: 30 }, () => 25 + Math.random() * 55),
  });

  // Chakra energy levels
  const [chakras, setChakras] = useState(() =>
    Array.from({ length: 7 }, () => 30 + Math.random() * 60)
  );

  // Score trends (vs yesterday)
  const [trends] = useState({
    wellness: Math.round((Math.random() - 0.3) * 10),
    sleep: Math.round((Math.random() - 0.4) * 8),
    brain: Math.round((Math.random() - 0.3) * 12),
  });

  /* Real-time update loop */
  const update = useCallback(() => {
    setEmotion((prev) => {
      const stress = Math.max(0, Math.min(100, prev.stress_index + (Math.random() - 0.5) * 6));
      const focus = Math.max(0, Math.min(100, prev.focus_index + (Math.random() - 0.5) * 5));
      const relaxation = Math.max(0, Math.min(100, prev.relaxation_index + (Math.random() - 0.5) * 5));

      const emotions = ["happy", "sad", "relaxed", "focused", "fearful"];
      const weights = [
        relaxation * 0.3 + (100 - stress) * 0.2,
        stress * 0.15 + (100 - relaxation) * 0.1,
        relaxation * 0.4,
        focus * 0.35,
        stress * 0.2,
      ];
      const maxIdx = weights.indexOf(Math.max(...weights));
      const topEmotion = emotions[maxIdx];

      // Detect emotional shift
      if (topEmotion !== prev.emotion) {
        if (stress > 60 && prev.stress_index <= 50) {
          setShift({
            detected: true,
            type: "approaching_anxiety",
            description: "Your body is tensing before you feel it. Take a slow breath.",
          });
        } else if (relaxation > 70 && prev.relaxation_index <= 55) {
          setShift({
            detected: true,
            type: "approaching_calm",
            description: "A wave of calm is settling in. Let it happen.",
          });
        }
      }

      return {
        emotion: topEmotion,
        confidence: 0.5 + Math.random() * 0.4,
        valence: Math.tanh((relaxation - stress) * 0.02),
        arousal: Math.min(1, (stress + focus) * 0.008),
        stress_index: stress,
        focus_index: focus,
        relaxation_index: relaxation,
      };
    });

    // Update mood history
    setMoodHistory((prev) => {
      const now = new Date();
      const newPoint: MoodPoint = {
        time: now.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" }),
        mood: 50 + Math.random() * 35,
        stress: 15 + Math.random() * 30,
      };
      return [...prev.slice(1), newPoint];
    });

    // Update brain waves
    setBrainWaves((prev) => ({
      alpha: [...prev.alpha.slice(1), 20 + Math.random() * 60],
      theta: [...prev.theta.slice(1), 15 + Math.random() * 50],
      beta: [...prev.beta.slice(1), 25 + Math.random() * 55],
    }));

    // Update chakras
    setChakras((prev) =>
      prev.map((v) => Math.max(10, Math.min(95, v + (Math.random() - 0.5) * 8)))
    );
  }, []);

  useEffect(() => {
    const interval = setInterval(update, 3000);
    return () => clearInterval(interval);
  }, [update]);

  // Dismiss shift after 8s
  useEffect(() => {
    if (shift?.detected) {
      const timer = setTimeout(() => setShift(null), 8000);
      return () => clearTimeout(timer);
    }
  }, [shift]);

  /* Computed scores */
  const wellnessScore = Math.round(
    emotion.relaxation_index * 0.4 +
      (100 - emotion.stress_index) * 0.35 +
      emotion.focus_index * 0.25
  );
  const sleepScore = Math.round(70 + Math.random() * 20); // Would come from sleep model
  const brainScore = Math.round(
    emotion.focus_index * 0.4 +
      emotion.relaxation_index * 0.3 +
      (100 - emotion.arousal * 60) * 0.3
  );

  const hour = new Date().getHours();
  const insightText = useMemo(() => getInsightText(emotion, hour), [emotion, hour]);

  /* Sparkline renderer */
  const renderSparkline = (data: number[], color: string) => {
    const w = 200;
    const h = 40;
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;
    const points = data
      .map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * h}`)
      .join(" ");
    return (
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          strokeLinejoin="round"
        />
      </svg>
    );
  };

  return (
    <main className="p-6 space-y-6 max-w-6xl">
      {/* ---- Emotional Shift Alert ---- */}
      {shift?.detected && (
        <div
          className={
            shift.type === "approaching_calm" ? "shift-alert-calm" : "shift-alert"
          }
        >
          <div className="flex items-start gap-3">
            <AlertCircle
              className={`h-5 w-5 mt-0.5 shrink-0 ${
                shift.type === "approaching_calm"
                  ? "text-success"
                  : "text-warning"
              }`}
            />
            <div>
              <p className="text-sm font-medium text-foreground">
                Pre-Conscious Shift Detected
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                {shift.description}
              </p>
              {shift.type === "approaching_anxiety" && (
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

      {/* ---- Score Circles ---- */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        <div className="score-card p-6 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={wellnessScore}
            label="Wellness"
            gradientId="grad-wellness"
            colorFrom="hsl(152, 60%, 48%)"
            colorTo="hsl(180, 65%, 50%)"
            size="lg"
            trend={trends.wellness}
            sublabel="vs yesterday"
          />
          <p className="text-xs text-muted-foreground mt-2">
            {EMOTION_LABELS[emotion.emotion] || emotion.emotion} &middot;{" "}
            {Math.round(emotion.confidence * 100)}%
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
            trend={trends.sleep}
            sublabel="vs yesterday"
          />
          <p className="text-xs text-muted-foreground mt-2">
            7.4h &middot; 3 REM cycles
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
            trend={trends.brain}
            sublabel="vs yesterday"
          />
          <p className="text-xs text-muted-foreground mt-2">
            Focus {Math.round(emotion.focus_index)} &middot; Stress{" "}
            {Math.round(emotion.stress_index)}
          </p>
        </div>
      </div>

      {/* ---- AI Insight ---- */}
      <div className="ai-insight-card">
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
            <p className="text-sm text-muted-foreground leading-relaxed">
              {insightText}
            </p>
            <div className="flex gap-2 mt-3">
              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-primary/10 text-primary">
                Stress {Math.round(emotion.stress_index)}
              </span>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-secondary/10 text-secondary">
                Focus {Math.round(emotion.focus_index)}
              </span>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-accent/10 text-accent">
                Relax {Math.round(emotion.relaxation_index)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ---- Secondary Grid ---- */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Mood Timeline */}
        <Card className="glass-card p-5 md:col-span-2 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <Heart className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-medium">Mood Timeline</h3>
          </div>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={moodHistory}>
                <defs>
                  <linearGradient id="moodGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(152, 60%, 48%)" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="hsl(152, 60%, 48%)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="stressGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(38, 85%, 58%)" stopOpacity={0.2} />
                    <stop offset="100%" stopColor="hsl(38, 85%, 58%)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis hide domain={[0, 100]} />
                <Tooltip
                  contentStyle={{
                    background: "hsl(220, 22%, 9%)",
                    border: "1px solid hsl(220, 18%, 20%)",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  labelStyle={{ color: "hsl(38, 20%, 92%)" }}
                />
                <Area
                  type="monotone"
                  dataKey="mood"
                  stroke="hsl(152, 60%, 48%)"
                  fill="url(#moodGrad)"
                  strokeWidth={2}
                  dot={false}
                />
                <Area
                  type="monotone"
                  dataKey="stress"
                  stroke="hsl(38, 85%, 58%)"
                  fill="url(#stressGrad)"
                  strokeWidth={1.5}
                  dot={false}
                  strokeDasharray="4 4"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Brain Waves Sparklines */}
        <Card className="glass-card p-5 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="h-4 w-4 text-secondary" />
            <h3 className="text-sm font-medium">Brain Waves</h3>
          </div>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
                <span>Alpha</span>
                <span className="font-mono">{Math.round(brainWaves.alpha[brainWaves.alpha.length - 1])}%</span>
              </div>
              {renderSparkline(brainWaves.alpha, "hsl(152, 60%, 48%)")}
            </div>
            <div>
              <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
                <span>Theta</span>
                <span className="font-mono">{Math.round(brainWaves.theta[brainWaves.theta.length - 1])}%</span>
              </div>
              {renderSparkline(brainWaves.theta, "hsl(262, 45%, 65%)")}
            </div>
            <div>
              <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
                <span>Beta</span>
                <span className="font-mono">{Math.round(brainWaves.beta[brainWaves.beta.length - 1])}%</span>
              </div>
              {renderSparkline(brainWaves.beta, "hsl(200, 70%, 55%)")}
            </div>
          </div>
        </Card>
      </div>

      {/* ---- Energy Preview ---- */}
      <Card className="glass-card p-5 hover-glow">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-accent" />
            <h3 className="text-sm font-medium">Energy Centers</h3>
          </div>
          <Link
            href="/inner-energy"
            className="text-xs text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
          >
            Full map <ArrowRight className="h-3 w-3" />
          </Link>
        </div>
        <div className="flex items-end gap-3 justify-center h-24">
          {chakras.map((val, i) => (
            <div key={i} className="flex flex-col items-center gap-1">
              <div
                className="w-6 rounded-t-sm transition-all duration-700"
                style={{
                  height: `${val * 0.8}px`,
                  background: `linear-gradient(to top, ${CHAKRA_COLORS[i]}88, ${CHAKRA_COLORS[i]})`,
                  boxShadow: `0 0 8px ${CHAKRA_COLORS[i]}44`,
                }}
              />
              <span className="text-[9px] text-muted-foreground">{CHAKRA_LABELS[i]}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* ---- Quick Actions ---- */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
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
