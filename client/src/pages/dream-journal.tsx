import { useState, useEffect, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScoreCircle } from "@/components/score-circle";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  CartesianGrid,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";
import { Moon, Brain, Eye, Sparkles, Waves, Activity, Radio, TrendingUp } from "lucide-react";
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

interface NightlyPoint {
  time: string;
  remMinutes: number;
  dreamEpisodes: number;
  avgIntensity: number;
  avgLucidity: number;
  sleepStageIdx: number;
}

interface HypnogramPoint {
  time: string;
  stage: number;
  label: string;
}

/* ---------- helpers ---------- */
const STAGE_LABELS: Record<string, { label: string; color: string }> = {
  Wake: { label: "Awake", color: "hsl(38, 85%, 58%)" },
  N1: { label: "Light Sleep", color: "hsl(200, 70%, 55%)" },
  N2: { label: "Sleep", color: "hsl(220, 50%, 50%)" },
  N3: { label: "Deep Sleep", color: "hsl(262, 45%, 55%)" },
  REM: { label: "REM", color: "hsl(152, 60%, 48%)" },
};

const STAGE_NAMES = ["N3 (Deep)", "N2 (Light)", "N1", "REM", "Wake"];
const STAGE_VALUES: Record<string, number> = { Wake: 4, REM: 3, N1: 2, N2: 1, N3: 0 };

type Tab = "detection" | "patterns";

/* ========== Component ========== */
export default function DreamDetection() {
  const [activeTab, setActiveTab] = useState<Tab>("detection");
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;

  const dreamDetection = analysis?.dream_detection;
  const sleepStaging = analysis?.sleep_staging;
  const lucidDream = analysis?.lucid_dream;

  // ── Detection state ──
  const isDreaming = dreamDetection?.is_dreaming ?? false;
  const dreamProbability = dreamDetection?.probability ?? 0;
  const remLikelihood = dreamDetection?.rem_likelihood ?? 0;
  const dreamIntensity = Math.round((dreamDetection?.dream_intensity ?? 0) * 100);
  const lucidityEstimate = Math.round((dreamDetection?.lucidity_estimate ?? 0) * 100);
  const sleepStage = sleepStaging?.stage ?? "Wake";
  const sleepStageConfidence = sleepStaging?.confidence ?? 0;

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

  const [episodes, setEpisodes] = useState<DreamEpisode[]>([]);
  const wasDreamingRef = useRef(false);
  const dreamStartRef = useRef<Date | null>(null);

  useEffect(() => {
    if (!isStreaming) return;

    if (isDreaming && !wasDreamingRef.current) {
      dreamStartRef.current = new Date();
      wasDreamingRef.current = true;
    } else if (!isDreaming && wasDreamingRef.current && dreamStartRef.current) {
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

  // ── Patterns state ──
  const [sessionData, setSessionData] = useState<NightlyPoint[]>([]);
  const [hypnogram, setHypnogram] = useState<HypnogramPoint[]>([]);
  const [remCycles, setRemCycles] = useState<{ cycle: string; duration: number; intensity: number; lucidity: number }[]>([]);
  const [inRem, setInRem] = useState(false);
  const [remStart, setRemStart] = useState(0);

  useEffect(() => {
    if (!isStreaming || !sleepStaging) return;

    const now = new Date();
    const timeStr = now.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    const stage = sleepStaging.stage ?? "Wake";
    const stageIdx = STAGE_VALUES[stage] ?? 4;

    setHypnogram((prev) => [
      ...prev.slice(-60),
      { time: timeStr, stage: stageIdx, label: stage },
    ]);

    const rl = dreamDetection?.rem_likelihood ?? 0;
    const dreaming = dreamDetection?.is_dreaming ?? false;
    const intensity = Math.round((dreamDetection?.dream_intensity ?? 0) * 100);
    const lucidity = Math.round((dreamDetection?.lucidity_estimate ?? 0) * 100);

    setSessionData((prev) => [
      ...prev.slice(-60),
      {
        time: timeStr,
        remMinutes: Math.round(rl * 100),
        dreamEpisodes: dreaming ? 1 : 0,
        avgIntensity: intensity,
        avgLucidity: lucidity,
        sleepStageIdx: stageIdx,
      },
    ]);

    const isCurrentlyRem = stage === "REM";
    if (isCurrentlyRem && !inRem) {
      setInRem(true);
      setRemStart(Date.now());
    } else if (!isCurrentlyRem && inRem && remStart > 0) {
      setInRem(false);
      const durationMin = Math.round((Date.now() - remStart) / 60000);
      if (durationMin > 0) {
        setRemCycles((prev) => [
          ...prev,
          {
            cycle: `Cycle ${prev.length + 1}`,
            duration: Math.max(1, durationMin),
            intensity,
            lucidity,
          },
        ]);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  const stageInfo = STAGE_LABELS[sleepStage] || STAGE_LABELS.Wake;

  // Patterns summary stats
  const totalDreamFrames = sessionData.filter((d) => d.dreamEpisodes > 0).length;
  const avgRem = sessionData.length > 0
    ? Math.round(sessionData.reduce((s, d) => s + d.remMinutes, 0) / sessionData.length)
    : 0;
  const avgIntensityPatterns = sessionData.length > 0
    ? Math.round(sessionData.reduce((s, d) => s + d.avgIntensity, 0) / sessionData.length)
    : 0;

  return (
    <main className="p-6 space-y-6 max-w-5xl">
      {/* Tab Bar */}
      <div className="flex gap-1 p-1 rounded-xl" style={{ background: "hsl(220, 22%, 8%)", border: "1px solid hsl(220, 18%, 13%)" }}>
        {([["detection", "Detection"], ["patterns", "Patterns"]] as const).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              activeTab === key
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground/70"
            }`}
            style={
              activeTab === key
                ? {
                    background: "linear-gradient(135deg, hsl(152,60%,48%,0.15), hsl(262,45%,65%,0.10))",
                    borderBottom: "2px solid hsl(152,60%,48%)",
                  }
                : undefined
            }
          >
            {label}
          </button>
        ))}
      </div>

      {/* Connection Banner */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          Connect your Muse 2 from the sidebar to see live dream {activeTab === "detection" ? "detection" : "pattern"} data.
        </div>
      )}

      {/* ═══════ Detection Tab ═══════ */}
      {activeTab === "detection" && (
        <>
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
                  <ResponsiveContainer width="100%" height={192}>
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

          {/* AI Interpretation */}
          {isStreaming && <DreamInsightCard episodes={episodes} stageLabel={stageInfo.label} frameTs={latestFrame?.timestamp} />}
        </>
      )}

      {/* ═══════ Patterns Tab ═══════ */}
      {activeTab === "patterns" && (
        <>
          {/* Summary Stats */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "Dream Frames", value: totalDreamFrames, sub: "this session", color: "text-secondary" },
              { label: "Avg REM %", value: `${avgRem}`, sub: "session avg", color: "text-primary" },
              { label: "Avg Intensity", value: `${avgIntensityPatterns}%`, sub: "session avg", color: "text-accent" },
              { label: "REM Cycles", value: remCycles.length, sub: "this session", color: "text-foreground" },
            ].map((stat) => (
              <Card key={stat.label} className="glass-card p-4 hover-glow text-center">
                <p className={`text-2xl font-semibold ${stat.color}`}>{stat.value}</p>
                <p className="text-xs text-muted-foreground mt-1">{stat.label}</p>
                <p className="text-[10px] text-muted-foreground/60">{stat.sub}</p>
              </Card>
            ))}
          </div>

          {/* Sleep Architecture (Hypnogram) */}
          <Card className="glass-card p-5 hover-glow">
            <div className="flex items-center gap-2 mb-4">
              <Brain className="h-4 w-4 text-secondary" />
              <h3 className="text-sm font-medium">Sleep Architecture</h3>
              {isStreaming && (
                <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
              )}
            </div>
            {hypnogram.length < 2 ? (
              <div className="h-[200px] flex items-center justify-center text-sm text-muted-foreground">
                {isStreaming ? "Collecting sleep stage data..." : "Connect device to see hypnogram"}
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={hypnogram.slice(-40)}>
                  <defs>
                    <linearGradient id="hypnoGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="hsl(262, 45%, 65%)" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="hsl(262, 45%, 65%)" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
                  <YAxis
                    domain={[0, 4]}
                    ticks={[0, 1, 2, 3, 4]}
                    tickFormatter={(v: number) => STAGE_NAMES[v] || ""}
                    tick={{ fontSize: 9, fill: "hsl(220, 12%, 42%)" }}
                    width={70}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, fontSize: 12 }}
                    labelStyle={{ color: "hsl(38, 20%, 92%)" }}
                    formatter={(value: number) => [STAGE_NAMES[value] || "Unknown", "Stage"]}
                  />
                  <Area type="stepAfter" dataKey="stage" stroke="hsl(262, 45%, 65%)" fill="url(#hypnoGrad)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </Card>

          {/* Session Trends */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Dream Detection Timeline */}
            <Card className="glass-card p-5 hover-glow">
              <div className="flex items-center gap-2 mb-4">
                <Moon className="h-4 w-4 text-secondary" />
                <h3 className="text-sm font-medium">Dream Activity</h3>
              </div>
              {sessionData.length < 2 ? (
                <div className="h-[200px] flex items-center justify-center text-sm text-muted-foreground">
                  {isStreaming ? "Collecting..." : "No data"}
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={sessionData.slice(-30)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.4} />
                    <XAxis dataKey="time" tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
                    <YAxis hide />
                    <Tooltip
                      contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, fontSize: 12 }}
                      labelStyle={{ color: "hsl(38, 20%, 92%)" }}
                    />
                    <Bar dataKey="avgIntensity" fill="hsl(262, 45%, 65%)" radius={[4, 4, 0, 0]} name="Intensity %" />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </Card>

            {/* REM Likelihood Over Time */}
            <Card className="glass-card p-5 hover-glow">
              <div className="flex items-center gap-2 mb-4">
                <Activity className="h-4 w-4 text-primary" />
                <h3 className="text-sm font-medium">REM Likelihood</h3>
              </div>
              {sessionData.length < 2 ? (
                <div className="h-[200px] flex items-center justify-center text-sm text-muted-foreground">
                  {isStreaming ? "Collecting..." : "No data"}
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={sessionData.slice(-30)}>
                    <defs>
                      <linearGradient id="remPatternGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="hsl(152, 60%, 48%)" stopOpacity={0.3} />
                        <stop offset="100%" stopColor="hsl(152, 60%, 48%)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.4} />
                    <XAxis dataKey="time" tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0, 100]} hide />
                    <Tooltip
                      contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, fontSize: 12 }}
                      labelStyle={{ color: "hsl(38, 20%, 92%)" }}
                    />
                    <Area type="monotone" dataKey="remMinutes" stroke="hsl(152, 60%, 48%)" fill="url(#remPatternGrad)" strokeWidth={2} dot={false} name="REM %" />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </Card>
          </div>

          {/* REM Cycle Progression */}
          {remCycles.length > 0 && (
            <Card className="glass-card p-5 hover-glow">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="h-4 w-4 text-primary" />
                <h3 className="text-sm font-medium">REM Cycle Progression</h3>
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={remCycles}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.4} />
                  <XAxis dataKey="cycle" tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, fontSize: 12 }}
                    labelStyle={{ color: "hsl(38, 20%, 92%)" }}
                  />
                  <Line type="monotone" dataKey="intensity" stroke="hsl(38, 85%, 58%)" strokeWidth={2} dot={{ r: 4, fill: "hsl(38, 85%, 58%)" }} name="Intensity %" />
                  <Line type="monotone" dataKey="lucidity" stroke="hsl(200, 70%, 55%)" strokeWidth={2} dot={{ r: 4, fill: "hsl(200, 70%, 55%)" }} name="Lucidity %" />
                  <Line type="monotone" dataKey="duration" stroke="hsl(152, 60%, 48%)" strokeWidth={2} dot={{ r: 4, fill: "hsl(152, 60%, 48%)" }} name="Duration (min)" />
                </LineChart>
              </ResponsiveContainer>
              <div className="flex justify-center gap-4 mt-3">
                {[
                  { label: "Intensity", color: "hsl(38, 85%, 58%)" },
                  { label: "Lucidity", color: "hsl(200, 70%, 55%)" },
                  { label: "Duration", color: "hsl(152, 60%, 48%)" },
                ].map((item) => (
                  <div key={item.label} className="flex items-center gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ background: item.color }} />
                    <span className="text-[10px] text-muted-foreground">{item.label}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Pattern AI Analysis */}
          {isStreaming && sessionData.length > 5 && (
            <PatternInsightCard
              sessionLen={sessionData.length}
              dreamFrames={totalDreamFrames}
              avgRem={avgRem}
              remCycles={remCycles.length}
              frameTs={latestFrame?.timestamp}
            />
          )}
        </>
      )}
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

/* Throttled pattern insight card */
function PatternInsightCard({ sessionLen, dreamFrames, avgRem, remCycles, frameTs }: {
  sessionLen: number; dreamFrames: number; avgRem: number; remCycles: number; frameTs?: number;
}) {
  const [text, setText] = useState("");
  const timerRef = useRef(0);

  useEffect(() => {
    const now = Date.now();
    if (now - timerRef.current < 10_000 && text) return;
    timerRef.current = now;
    const cycleText = remCycles > 0
      ? ` ${remCycles} complete REM cycle${remCycles > 1 ? "s" : ""} detected. REM cycle duration naturally increases through the night — later cycles tend to show higher intensity and lucidity.`
      : " No complete REM cycles detected yet — continue monitoring for full cycle analysis.";
    setText(`This session captured ${sessionLen} data points with ${dreamFrames} dream-active frames and an average REM likelihood of ${avgRem}%.${cycleText}`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frameTs]);

  return (
    <div className="ai-insight-card">
      <div className="flex items-start gap-3">
        <Sparkles className="h-5 w-5 text-primary mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-medium text-foreground mb-1">Pattern Analysis</p>
          <p className="text-sm text-muted-foreground leading-relaxed">{text}</p>
        </div>
      </div>
    </div>
  );
}
