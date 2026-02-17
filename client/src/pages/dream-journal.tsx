import { useState, useEffect, useCallback } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScoreCircle } from "@/components/score-circle";
import {
  AreaChart,
  Area,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";
import { Moon, Brain, Eye, Sparkles, Waves, Activity } from "lucide-react";

/* ---------- types ---------- */
interface DreamState {
  isDreaming: boolean;
  probability: number;
  remLikelihood: number;
  dreamIntensity: number;
  lucidityEstimate: number;
  sleepStage: string;
  sleepStageConfidence: number;
}

interface DreamEpisode {
  startTime: string;
  duration: number; // minutes
  intensity: number;
  lucidity: number;
  remProbability: number;
  stage: string;
}

interface SleepPoint {
  time: string;
  rem: number;
  dreamProb: number;
}

/* ---------- helpers ---------- */
const STAGE_LABELS: Record<string, { label: string; color: string }> = {
  Wake: { label: "Awake", color: "hsl(38, 85%, 58%)" },
  N1: { label: "Light Sleep", color: "hsl(200, 70%, 55%)" },
  N2: { label: "Sleep", color: "hsl(220, 50%, 50%)" },
  N3: { label: "Deep Sleep", color: "hsl(262, 45%, 55%)" },
  REM: { label: "REM", color: "hsl(152, 60%, 48%)" },
};

function generateDreamEpisodes(): DreamEpisode[] {
  const episodes: DreamEpisode[] = [];
  const now = new Date();
  // Simulate detected dream episodes from last night
  const dreamCount = 2 + Math.floor(Math.random() * 3);
  for (let i = 0; i < dreamCount; i++) {
    const hoursAgo = 2 + i * 1.5 + Math.random();
    const t = new Date(now.getTime() - hoursAgo * 3600000);
    episodes.push({
      startTime: t.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" }),
      duration: Math.round(5 + Math.random() * 25),
      intensity: Math.round(30 + Math.random() * 60),
      lucidity: Math.round(5 + Math.random() * 40),
      remProbability: Math.round(60 + Math.random() * 35),
      stage: Math.random() > 0.3 ? "REM" : "N2",
    });
  }
  return episodes.reverse();
}

/* ========== Component ========== */
export default function DreamDetection() {
  const [dreamState, setDreamState] = useState<DreamState>({
    isDreaming: false,
    probability: 0.12,
    remLikelihood: 0.08,
    dreamIntensity: 15,
    lucidityEstimate: 5,
    sleepStage: "Wake",
    sleepStageConfidence: 0.85,
  });

  const [episodes] = useState<DreamEpisode[]>(generateDreamEpisodes);

  const [sleepTimeline, setSleepTimeline] = useState<SleepPoint[]>(() => {
    return Array.from({ length: 20 }, (_, i) => {
      const t = new Date(Date.now() - (19 - i) * 5 * 60000);
      return {
        time: t.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" }),
        rem: Math.random() * 40,
        dreamProb: Math.random() * 30,
      };
    });
  });

  const update = useCallback(() => {
    setDreamState((prev) => {
      const remLikelihood = Math.max(0, Math.min(1, prev.remLikelihood + (Math.random() - 0.5) * 0.08));
      const probability = Math.max(0, Math.min(1, remLikelihood * 0.7 + Math.random() * 0.15));
      const isDreaming = probability > 0.55;
      const dreamIntensity = isDreaming
        ? Math.round(40 + Math.random() * 50)
        : Math.round(Math.max(0, prev.dreamIntensity * 0.8 + Math.random() * 5));
      const lucidityEstimate = isDreaming
        ? Math.round(10 + Math.random() * 35)
        : Math.round(Math.max(0, prev.lucidityEstimate * 0.7));

      const stages = ["Wake", "N1", "N2", "N3", "REM"];
      const stageWeights = [
        1 - remLikelihood,
        remLikelihood * 0.2,
        0.3,
        0.15,
        remLikelihood * 0.8,
      ];
      const maxIdx = stageWeights.indexOf(Math.max(...stageWeights));

      return {
        isDreaming,
        probability,
        remLikelihood,
        dreamIntensity,
        lucidityEstimate,
        sleepStage: stages[maxIdx],
        sleepStageConfidence: 0.6 + Math.random() * 0.35,
      };
    });

    setSleepTimeline((prev) => {
      const now = new Date();
      return [
        ...prev.slice(1),
        {
          time: now.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" }),
          rem: Math.random() * 50,
          dreamProb: Math.random() * 40,
        },
      ];
    });
  }, []);

  useEffect(() => {
    const interval = setInterval(update, 4000);
    return () => clearInterval(interval);
  }, [update]);

  const stageInfo = STAGE_LABELS[dreamState.sleepStage] || STAGE_LABELS.Wake;

  return (
    <main className="p-6 space-y-6 max-w-5xl">
      {/* Live Detection Status */}
      {dreamState.isDreaming && (
        <div className="shift-alert-calm">
          <div className="flex items-start gap-3">
            <Moon className="h-5 w-5 text-success mt-0.5 shrink-0" />
            <div>
              <p className="text-sm font-medium text-foreground">
                Dream State Detected
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                REM activity detected with {Math.round(dreamState.probability * 100)}% confidence.
                Dream intensity: {dreamState.dreamIntensity}%. Lucidity: {dreamState.lucidityEstimate}%.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Score Gauges */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={Math.round(dreamState.probability * 100)}
            label="Dream Prob"
            gradientId="grad-dream-prob"
            colorFrom="hsl(262, 45%, 65%)"
            colorTo="hsl(320, 55%, 60%)"
            size="sm"
          />
        </div>
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={Math.round(dreamState.remLikelihood * 100)}
            label="REM"
            gradientId="grad-rem"
            colorFrom="hsl(152, 60%, 48%)"
            colorTo="hsl(200, 70%, 55%)"
            size="sm"
          />
        </div>
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={dreamState.dreamIntensity}
            label="Intensity"
            gradientId="grad-intensity"
            colorFrom="hsl(38, 85%, 58%)"
            colorTo="hsl(25, 85%, 55%)"
            size="sm"
          />
        </div>
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={dreamState.lucidityEstimate}
            label="Lucidity"
            gradientId="grad-lucidity"
            colorFrom="hsl(200, 70%, 55%)"
            colorTo="hsl(262, 45%, 65%)"
            size="sm"
          />
        </div>
      </div>

      {/* Sleep Stage + REM Timeline */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Current Sleep Stage */}
        <Card className="glass-card p-5 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="h-4 w-4 text-secondary" />
            <h3 className="text-sm font-medium">Sleep Stage</h3>
          </div>
          <div className="text-center py-4">
            <div
              className="w-16 h-16 rounded-full mx-auto flex items-center justify-center mb-3"
              style={{
                background: `${stageInfo.color}20`,
                border: `2px solid ${stageInfo.color}`,
                boxShadow: `0 0 12px ${stageInfo.color}33`,
              }}
            >
              <Waves className="h-6 w-6" style={{ color: stageInfo.color }} />
            </div>
            <p className="text-lg font-semibold" style={{ color: stageInfo.color }}>
              {stageInfo.label}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              {Math.round(dreamState.sleepStageConfidence * 100)}% confidence
            </p>
          </div>
          <div className="mt-3 pt-3 border-t border-border/30 space-y-2">
            {Object.entries(STAGE_LABELS).map(([key, info]) => (
              <div key={key} className="flex items-center gap-2 text-xs">
                <div
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{
                    backgroundColor: info.color,
                    opacity: dreamState.sleepStage === key ? 1 : 0.3,
                  }}
                />
                <span
                  className={
                    dreamState.sleepStage === key
                      ? "text-foreground font-medium"
                      : "text-muted-foreground"
                  }
                >
                  {info.label}
                </span>
              </div>
            ))}
          </div>
        </Card>

        {/* REM / Dream Timeline */}
        <Card className="glass-card p-5 md:col-span-2 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-medium">REM & Dream Activity</h3>
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={sleepTimeline}>
                <defs>
                  <linearGradient id="remGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(152, 60%, 48%)" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="hsl(152, 60%, 48%)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="dreamProbGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(262, 45%, 65%)" stopOpacity={0.25} />
                    <stop offset="100%" stopColor="hsl(262, 45%, 65%)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis hide domain={[0, 60]} />
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
                  dataKey="rem"
                  stroke="hsl(152, 60%, 48%)"
                  fill="url(#remGrad)"
                  strokeWidth={2}
                  dot={false}
                  name="REM Activity"
                />
                <Area
                  type="monotone"
                  dataKey="dreamProb"
                  stroke="hsl(262, 45%, 65%)"
                  fill="url(#dreamProbGrad)"
                  strokeWidth={1.5}
                  dot={false}
                  name="Dream Probability"
                  strokeDasharray="4 4"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Detected Dream Episodes */}
      <Card className="glass-card p-5 hover-glow">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Eye className="h-4 w-4 text-secondary" />
            <h3 className="text-sm font-medium">Detected Dream Episodes</h3>
          </div>
          <Badge variant="secondary" className="text-xs">
            {episodes.length} detected
          </Badge>
        </div>

        {episodes.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-6">
            No dream episodes detected yet. Connect your BCI device and sleep to begin detection.
          </p>
        ) : (
          <div className="space-y-3">
            {episodes.map((ep, i) => (
              <div
                key={i}
                className="flex items-center gap-4 p-3 rounded-xl transition-colors"
                style={{
                  background: "hsl(220, 22%, 8%)",
                  border: "1px solid hsl(220, 18%, 13%)",
                }}
              >
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
                  style={{
                    background:
                      ep.stage === "REM"
                        ? "hsl(152, 60%, 48%, 0.15)"
                        : "hsl(262, 45%, 65%, 0.15)",
                  }}
                >
                  <Moon
                    className="h-5 w-5"
                    style={{
                      color:
                        ep.stage === "REM"
                          ? "hsl(152, 60%, 48%)"
                          : "hsl(262, 45%, 65%)",
                    }}
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">{ep.startTime}</span>
                    <Badge
                      variant="outline"
                      className="text-[10px] px-1.5 py-0"
                      style={{
                        borderColor:
                          ep.stage === "REM"
                            ? "hsl(152, 60%, 48%, 0.4)"
                            : "hsl(262, 45%, 65%, 0.4)",
                        color:
                          ep.stage === "REM"
                            ? "hsl(152, 60%, 48%)"
                            : "hsl(262, 45%, 65%)",
                      }}
                    >
                      {ep.stage}
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {ep.duration}min &middot; Intensity {ep.intensity}% &middot; Lucidity {ep.lucidity}%
                  </p>
                </div>
                <div className="text-right shrink-0">
                  <p className="text-sm font-mono text-primary">{ep.remProbability}%</p>
                  <p className="text-[10px] text-muted-foreground">REM prob</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* AI Interpretation */}
      <div className="ai-insight-card">
        <div className="flex items-start gap-3">
          <Sparkles className="h-5 w-5 text-primary mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium text-foreground mb-1">Dream Analysis</p>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {episodes.length >= 3
                ? `${episodes.length} dream episodes detected overnight with an average REM probability of ${Math.round(episodes.reduce((s, e) => s + e.remProbability, 0) / episodes.length)}%. Your dream intensity pattern suggests active memory consolidation. Higher lucidity estimates in later cycles indicate healthy sleep architecture.`
                : episodes.length > 0
                  ? `${episodes.length} dream episode${episodes.length > 1 ? "s" : ""} detected. Dream patterns are still building — more overnight data will reveal your unique dream signature and REM cycling patterns.`
                  : "No dream episodes detected yet. Connect your BCI headband before sleep to enable automatic dream detection via REM and EEG spectral analysis."}
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
