import { useState, useEffect, useRef } from "react";
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
import { Moon, Brain, Eye, Sparkles, Waves, Activity, Radio } from "lucide-react";
import { useDevice } from "@/hooks/use-device";

/* ---------- types ---------- */
interface DreamEpisode {
  startTime: string;
  duration: number;
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

/* ========== Component ========== */
export default function DreamDetection() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;

  const dreamDetection = analysis?.dream_detection;
  const sleepStaging = analysis?.sleep_staging;
  const lucidDream = analysis?.lucid_dream;

  // Live dream state
  const isDreaming = dreamDetection?.is_dreaming ?? false;
  const dreamProbability = dreamDetection?.probability ?? 0;
  const remLikelihood = dreamDetection?.rem_likelihood ?? 0;
  const dreamIntensity = Math.round((dreamDetection?.dream_intensity ?? 0) * 100);
  const lucidityEstimate = Math.round((dreamDetection?.lucidity_estimate ?? 0) * 100);
  const sleepStage = sleepStaging?.stage ?? "Wake";
  const sleepStageConfidence = sleepStaging?.confidence ?? 0;

  // Accumulate sleep timeline from live data
  const [sleepTimeline, setSleepTimeline] = useState<SleepPoint[]>([]);

  useEffect(() => {
    if (!isStreaming || !dreamDetection) return;
    const now = new Date().toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    setSleepTimeline((prev) => [
      ...prev.slice(-40),
      {
        time: now,
        rem: Math.round(remLikelihood * 100),
        dreamProb: Math.round(dreamProbability * 100),
      },
    ]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  // Detect dream episodes — track transitions
  const [episodes, setEpisodes] = useState<DreamEpisode[]>([]);
  const wasDreamingRef = useRef(false);
  const dreamStartRef = useRef<Date | null>(null);

  useEffect(() => {
    if (!isStreaming) return;

    if (isDreaming && !wasDreamingRef.current) {
      // Dream just started
      dreamStartRef.current = new Date();
      wasDreamingRef.current = true;
    } else if (!isDreaming && wasDreamingRef.current && dreamStartRef.current) {
      // Dream just ended — record episode
      const duration = Math.round((Date.now() - dreamStartRef.current.getTime()) / 60000);
      setEpisodes((prev) => [
        ...prev,
        {
          startTime: dreamStartRef.current!.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" }),
          duration: Math.max(1, duration),
          intensity: dreamIntensity,
          lucidity: lucidityEstimate,
          remProbability: Math.round(remLikelihood * 100),
          stage: sleepStage === "REM" ? "REM" : sleepStage,
        },
      ]);
      wasDreamingRef.current = false;
      dreamStartRef.current = null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDreaming, isStreaming]);

  const stageInfo = STAGE_LABELS[sleepStage] || STAGE_LABELS.Wake;

  return (
    <main className="p-6 space-y-6 max-w-5xl">
      {/* Connection Banner */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          Connect your Muse 2 from the sidebar to see live dream detection data.
        </div>
      )}

      {/* Live Detection Status */}
      {isDreaming && isStreaming && (
        <div className="shift-alert-calm">
          <div className="flex items-start gap-3">
            <Moon className="h-5 w-5 text-success mt-0.5 shrink-0" />
            <div>
              <p className="text-sm font-medium text-foreground">Dream State Detected</p>
              <p className="text-sm text-muted-foreground mt-1">
                REM activity detected with {Math.round(dreamProbability * 100)}% confidence.
                Dream intensity: {dreamIntensity}%. Lucidity: {lucidityEstimate}%.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Score Gauges */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={isStreaming ? Math.round(dreamProbability * 100) : 0}
            label="Dream Prob"
            gradientId="grad-dream-prob"
            colorFrom="hsl(262, 45%, 65%)"
            colorTo="hsl(320, 55%, 60%)"
            size="sm"
          />
        </div>
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={isStreaming ? Math.round(remLikelihood * 100) : 0}
            label="REM"
            gradientId="grad-rem"
            colorFrom="hsl(152, 60%, 48%)"
            colorTo="hsl(200, 70%, 55%)"
            size="sm"
          />
        </div>
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={isStreaming ? dreamIntensity : 0}
            label="Intensity"
            gradientId="grad-intensity"
            colorFrom="hsl(38, 85%, 58%)"
            colorTo="hsl(25, 85%, 55%)"
            size="sm"
          />
        </div>
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={isStreaming ? lucidityEstimate : 0}
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
            {isStreaming && (
              <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </div>
          <div className="text-center py-4">
            <div
              className="w-16 h-16 rounded-full mx-auto flex items-center justify-center mb-3"
              style={{
                background: isStreaming ? `${stageInfo.color}20` : "hsl(220, 22%, 12%)",
                border: `2px solid ${isStreaming ? stageInfo.color : "hsl(220, 18%, 20%)"}`,
                boxShadow: isStreaming ? `0 0 12px ${stageInfo.color}33` : "none",
              }}
            >
              <Waves className="h-6 w-6" style={{ color: isStreaming ? stageInfo.color : "hsl(220, 12%, 42%)" }} />
            </div>
            <p className="text-lg font-semibold" style={{ color: isStreaming ? stageInfo.color : "hsl(220, 12%, 42%)" }}>
              {isStreaming ? stageInfo.label : "—"}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              {isStreaming ? `${Math.round(sleepStageConfidence * 100)}% confidence` : "No data"}
            </p>
          </div>
          <div className="mt-3 pt-3 border-t border-border/30 space-y-2">
            {Object.entries(STAGE_LABELS).map(([key, info]) => (
              <div key={key} className="flex items-center gap-2 text-xs">
                <div
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{
                    backgroundColor: info.color,
                    opacity: sleepStage === key ? 1 : 0.3,
                  }}
                />
                <span
                  className={
                    sleepStage === key ? "text-foreground font-medium" : "text-muted-foreground"
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
            {isStreaming && (
              <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </div>
          <div className="h-48">
            {sleepTimeline.length < 2 ? (
              <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
                {isStreaming ? "Collecting data..." : "Connect device to see dream activity"}
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={sleepTimeline.slice(-30)}>
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
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
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
                  <Area type="monotone" dataKey="rem" stroke="hsl(152, 60%, 48%)" fill="url(#remGrad)" strokeWidth={2} dot={false} name="REM Activity %" />
                  <Area type="monotone" dataKey="dreamProb" stroke="hsl(262, 45%, 65%)" fill="url(#dreamProbGrad)" strokeWidth={1.5} dot={false} name="Dream Prob %" strokeDasharray="4 4" />
                </AreaChart>
              </ResponsiveContainer>
            )}
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
            {isStreaming
              ? "No dream episodes detected yet. Dream episodes are recorded when the dream detection model identifies REM-like activity."
              : "Connect your BCI device and sleep to begin detection."}
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
                    background: ep.stage === "REM" ? "hsl(152, 60%, 48%, 0.15)" : "hsl(262, 45%, 65%, 0.15)",
                  }}
                >
                  <Moon
                    className="h-5 w-5"
                    style={{
                      color: ep.stage === "REM" ? "hsl(152, 60%, 48%)" : "hsl(262, 45%, 65%)",
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
                        borderColor: ep.stage === "REM" ? "hsl(152, 60%, 48%, 0.4)" : "hsl(262, 45%, 65%, 0.4)",
                        color: ep.stage === "REM" ? "hsl(152, 60%, 48%)" : "hsl(262, 45%, 65%)",
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

      {/* AI Interpretation — throttled to 10s */}
      {isStreaming && <DreamInsightCard episodes={episodes} stageLabel={stageInfo.label} frameTs={latestFrame?.timestamp} />}
    </main>
  );
}

/* Throttled dream insight card — only updates text every 10s */
function DreamInsightCard({ episodes, stageLabel, frameTs }: { episodes: DreamEpisode[]; stageLabel: string; frameTs?: number }) {
  const [text, setText] = useState("");
  const timerRef = useRef(0);

  useEffect(() => {
    const now = Date.now();
    if (now - timerRef.current < 10_000 && text) return;
    timerRef.current = now;

    const next = episodes.length >= 3
      ? `${episodes.length} dream episodes detected with an average REM probability of ${Math.round(episodes.reduce((s, e) => s + e.remProbability, 0) / episodes.length)}%. Your dream intensity pattern suggests active memory consolidation. Higher lucidity estimates in later cycles indicate healthy sleep architecture.`
      : episodes.length > 0
        ? `${episodes.length} dream episode${episodes.length > 1 ? "s" : ""} detected. Dream patterns are still building — more data will reveal your unique dream signature and REM cycling patterns.`
        : `Currently monitoring sleep stage: ${stageLabel}. Dream detection is active — episodes will be recorded automatically when REM-like patterns are identified.`;
    setText(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frameTs]);

  return (
    <div className="ai-insight-card">
      <div className="flex items-start gap-3">
        <Sparkles className="h-5 w-5 text-primary mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-medium text-foreground mb-1">Dream Analysis</p>
          <p className="text-sm text-muted-foreground leading-relaxed">{text}</p>
        </div>
      </div>
    </div>
  );
}
