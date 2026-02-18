import { useState, useEffect, useMemo } from "react";
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
  Radio,
} from "lucide-react";
import { useDevice } from "@/hooks/use-device";

/* ---------- types ---------- */
interface MoodPoint {
  time: string;
  mood: number;
  stress: number;
}

interface BandHistory {
  alpha: number[];
  theta: number[];
  beta: number[];
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

const CHAKRA_COLORS = [
  "hsl(0, 72%, 55%)",
  "hsl(25, 85%, 55%)",
  "hsl(45, 90%, 55%)",
  "hsl(152, 60%, 48%)",
  "hsl(200, 70%, 55%)",
  "hsl(240, 55%, 60%)",
  "hsl(280, 50%, 60%)",
];

const CHAKRA_LABELS = ["Root", "Sacral", "Solar", "Heart", "Throat", "Eye", "Crown"];

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
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;

  // Extract live data
  const emotions = analysis?.emotions;
  const bandPowers = analysis?.band_powers ?? {};
  const sleepStaging = analysis?.sleep_staging;
  const emotionShift = latestFrame?.emotion_shift;

  // Current metrics
  const stressIndex = (emotions?.stress_index ?? 0) * 100;
  const focusIndex = (emotions?.focus_index ?? 0) * 100;
  const relaxationIndex = (emotions?.relaxation_index ?? 0) * 100;
  const currentEmotion = emotions?.emotion ?? "—";
  const confidence = emotions?.confidence ?? 0;
  const valence = emotions?.valence ?? 0;
  const arousal = emotions?.arousal ?? 0;

  // Mood timeline — accumulate from live data
  const [moodHistory, setMoodHistory] = useState<MoodPoint[]>([]);

  useEffect(() => {
    if (!isStreaming || !emotions) return;
    const now = new Date().toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    setMoodHistory((prev) => [
      ...prev.slice(-30),
      {
        time: now,
        mood: Math.round(relaxationIndex * 0.5 + (100 - stressIndex) * 0.3 + focusIndex * 0.2),
        stress: Math.round(stressIndex),
      },
    ]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  // Brain wave sparklines — accumulate from band powers
  const [bandHistory, setBandHistory] = useState<BandHistory>({
    alpha: [],
    theta: [],
    beta: [],
  });

  useEffect(() => {
    if (!isStreaming || !bandPowers.alpha) return;
    setBandHistory((prev) => ({
      alpha: [...prev.alpha.slice(-30), (bandPowers.alpha ?? 0) * 100],
      theta: [...prev.theta.slice(-30), (bandPowers.theta ?? 0) * 100],
      beta: [...prev.beta.slice(-30), (bandPowers.beta ?? 0) * 100],
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  // Chakra energy derived from band powers
  const chakras = useMemo(() => {
    if (!bandPowers.delta) return [0, 0, 0, 0, 0, 0, 0];
    const d = (bandPowers.delta ?? 0) * 100;
    const t = (bandPowers.theta ?? 0) * 100;
    const a = (bandPowers.alpha ?? 0) * 100;
    const b = (bandPowers.beta ?? 0) * 100;
    const g = (bandPowers.gamma ?? 0) * 100;
    return [
      Math.min(95, d * 1.2),        // Root — delta
      Math.min(95, t * 1.1),        // Sacral — theta
      Math.min(95, (a + t) * 0.6),  // Solar — alpha/theta
      Math.min(95, a * 1.2),        // Heart — alpha
      Math.min(95, b * 1.0),        // Throat — beta
      Math.min(95, (b + g) * 0.7),  // Third Eye — high beta + gamma
      Math.min(95, g * 1.5),        // Crown — gamma
    ];
  }, [bandPowers.delta, bandPowers.theta, bandPowers.alpha, bandPowers.beta, bandPowers.gamma]);

  // Shift alert from emotion_shift
  const [shift, setShift] = useState<{
    detected: boolean;
    type: string;
    description: string;
  } | null>(null);

  useEffect(() => {
    if (!emotionShift?.shift_detected) return;
    const isCalm = emotionShift.to_state === "relaxed" || emotionShift.to_state === "calm";
    setShift({
      detected: true,
      type: isCalm ? "approaching_calm" : "approaching_anxiety",
      description: isCalm
        ? "A wave of calm is settling in. Let it happen."
        : `Shift from ${emotionShift.from_state} to ${emotionShift.to_state} detected. Take a slow breath.`,
    });
  }, [emotionShift?.shift_detected, emotionShift?.from_state, emotionShift?.to_state]);

  // Dismiss shift after 8s
  useEffect(() => {
    if (shift?.detected) {
      const timer = setTimeout(() => setShift(null), 8000);
      return () => clearTimeout(timer);
    }
  }, [shift]);

  // Computed scores from live data
  const wellnessScore = isStreaming
    ? Math.round(relaxationIndex * 0.4 + (100 - stressIndex) * 0.35 + focusIndex * 0.25)
    : 0;
  const sleepScore = isStreaming && sleepStaging
    ? Math.round(sleepStaging.confidence * 100)
    : 0;
  const brainScore = isStreaming
    ? Math.round(focusIndex * 0.4 + relaxationIndex * 0.3 + (100 - arousal * 60) * 0.3)
    : 0;

  const hour = new Date().getHours();
  const insightText = useMemo(
    () => isStreaming ? getInsightText(stressIndex, focusIndex, relaxationIndex, hour) : "",
    [isStreaming, stressIndex, focusIndex, relaxationIndex, hour],
  );

  /* Sparkline renderer */
  const renderSparkline = (data: number[], color: string) => {
    if (data.length < 2) return null;
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
      {/* Connection Banner */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          Connect your Muse 2 from the sidebar to see live dashboard data.
        </div>
      )}

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
                shift.type === "approaching_calm" ? "text-success" : "text-warning"
              }`}
            />
            <div>
              <p className="text-sm font-medium text-foreground">
                Pre-Conscious Shift Detected
              </p>
              <p className="text-sm text-muted-foreground mt-1">{shift.description}</p>
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
          />
          <p className="text-xs text-muted-foreground mt-2">
            {isStreaming
              ? `${EMOTION_LABELS[currentEmotion] || currentEmotion} · ${Math.round(confidence * 100)}%`
              : "—"}
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
          />
          <p className="text-xs text-muted-foreground mt-2">
            {isStreaming && sleepStaging ? `${sleepStaging.stage} stage` : "—"}
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
          />
          <p className="text-xs text-muted-foreground mt-2">
            {isStreaming
              ? `Focus ${Math.round(focusIndex)} · Stress ${Math.round(stressIndex)}`
              : "—"}
          </p>
        </div>
      </div>

      {/* ---- AI Insight ---- */}
      {isStreaming && (
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
              <p className="text-sm text-muted-foreground leading-relaxed">{insightText}</p>
              <div className="flex gap-2 mt-3">
                <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-primary/10 text-primary">
                  Stress {Math.round(stressIndex)}
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-secondary/10 text-secondary">
                  Focus {Math.round(focusIndex)}
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-accent/10 text-accent">
                  Relax {Math.round(relaxationIndex)}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ---- Secondary Grid ---- */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Mood Timeline */}
        <Card className="glass-card p-5 md:col-span-2 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <Heart className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-medium">Mood Timeline</h3>
            {isStreaming && (
              <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </div>
          <div className="h-40">
            {moodHistory.length < 2 ? (
              <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
                {isStreaming ? "Collecting data..." : "Connect device to see timeline"}
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={moodHistory.slice(-20)}>
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
                  <Area type="monotone" dataKey="mood" stroke="hsl(152, 60%, 48%)" fill="url(#moodGrad)" strokeWidth={2} dot={false} />
                  <Area type="monotone" dataKey="stress" stroke="hsl(38, 85%, 58%)" fill="url(#stressGrad)" strokeWidth={1.5} dot={false} strokeDasharray="4 4" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>

        {/* Brain Waves Sparklines */}
        <Card className="glass-card p-5 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="h-4 w-4 text-secondary" />
            <h3 className="text-sm font-medium">Brain Waves</h3>
            {isStreaming && (
              <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </div>
          {bandHistory.alpha.length < 2 ? (
            <div className="h-32 flex items-center justify-center text-sm text-muted-foreground">
              {isStreaming ? "Collecting..." : "No data"}
            </div>
          ) : (
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
                  <span>Alpha</span>
                  <span className="font-mono">{Math.round(bandHistory.alpha[bandHistory.alpha.length - 1])}%</span>
                </div>
                {renderSparkline(bandHistory.alpha, "hsl(152, 60%, 48%)")}
              </div>
              <div>
                <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
                  <span>Theta</span>
                  <span className="font-mono">{Math.round(bandHistory.theta[bandHistory.theta.length - 1])}%</span>
                </div>
                {renderSparkline(bandHistory.theta, "hsl(262, 45%, 65%)")}
              </div>
              <div>
                <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
                  <span>Beta</span>
                  <span className="font-mono">{Math.round(bandHistory.beta[bandHistory.beta.length - 1])}%</span>
                </div>
                {renderSparkline(bandHistory.beta, "hsl(200, 70%, 55%)")}
              </div>
            </div>
          )}
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
                  height: `${Math.max(4, val * 0.8)}px`,
                  background: `linear-gradient(to top, ${CHAKRA_COLORS[i]}88, ${CHAKRA_COLORS[i]})`,
                  boxShadow: `0 0 8px ${CHAKRA_COLORS[i]}44`,
                }}
              />
              <span className="text-[9px] text-muted-foreground">{CHAKRA_LABELS[i]}</span>
            </div>
          ))}
        </div>
        {!isStreaming && (
          <p className="text-center text-xs text-muted-foreground mt-2">Connect device to see energy levels</p>
        )}
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
