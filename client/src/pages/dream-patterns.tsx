import { useState, useEffect, useRef } from "react";
import { Card } from "@/components/ui/card";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Moon, TrendingUp, Brain, Activity, Sparkles, Radio } from "lucide-react";
import { useDevice } from "@/hooks/use-device";

/* ---------- types ---------- */
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

const STAGE_NAMES = ["N3 (Deep)", "N2 (Light)", "N1", "REM", "Wake"];
const STAGE_VALUES: Record<string, number> = { Wake: 4, REM: 3, N1: 2, N2: 1, N3: 0 };

/* ========== Component ========== */
export default function DreamPatterns() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;

  const dreamDetection = analysis?.dream_detection;
  const sleepStaging = analysis?.sleep_staging;

  // Accumulate session data for charts
  const [sessionData, setSessionData] = useState<NightlyPoint[]>([]);
  const [hypnogram, setHypnogram] = useState<HypnogramPoint[]>([]);

  // Track REM cycle metrics
  const [remCycles, setRemCycles] = useState<{ cycle: string; duration: number; intensity: number; lucidity: number }[]>([]);
  const [inRem, setInRem] = useState(false);
  const [remStart, setRemStart] = useState(0);

  useEffect(() => {
    if (!isStreaming || !sleepStaging) return;

    const now = new Date();
    const timeStr = now.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    const stage = sleepStaging.stage ?? "Wake";
    const stageIdx = STAGE_VALUES[stage] ?? 4;

    // Hypnogram
    setHypnogram((prev) => [
      ...prev.slice(-60),
      { time: timeStr, stage: stageIdx, label: stage },
    ]);

    // Session data accumulation
    const remLikelihood = dreamDetection?.rem_likelihood ?? 0;
    const isDreaming = dreamDetection?.is_dreaming ?? false;
    const intensity = Math.round((dreamDetection?.dream_intensity ?? 0) * 100);
    const lucidity = Math.round((dreamDetection?.lucidity_estimate ?? 0) * 100);

    setSessionData((prev) => [
      ...prev.slice(-60),
      {
        time: timeStr,
        remMinutes: Math.round(remLikelihood * 100),
        dreamEpisodes: isDreaming ? 1 : 0,
        avgIntensity: intensity,
        avgLucidity: lucidity,
        sleepStageIdx: stageIdx,
      },
    ]);

    // REM cycle tracking
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

  // Summary stats from accumulated session data
  const totalDreamFrames = sessionData.filter((d) => d.dreamEpisodes > 0).length;
  const avgRem = sessionData.length > 0
    ? Math.round(sessionData.reduce((s, d) => s + d.remMinutes, 0) / sessionData.length)
    : 0;
  const avgIntensity = sessionData.length > 0
    ? Math.round(sessionData.reduce((s, d) => s + d.avgIntensity, 0) / sessionData.length)
    : 0;

  return (
    <main className="p-6 space-y-6 max-w-5xl">
      {/* Connection Banner */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          Connect your Muse 2 from the sidebar to see live dream pattern data.
        </div>
      )}

      {/* Summary Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "Dream Frames", value: totalDreamFrames, sub: "this session", color: "text-secondary" },
          { label: "Avg REM %", value: `${avgRem}`, sub: "session avg", color: "text-primary" },
          { label: "Avg Intensity", value: `${avgIntensity}%`, sub: "session avg", color: "text-accent" },
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

      {/* AI Analysis — throttled to 10s */}
      {isStreaming && sessionData.length > 5 && (
        <PatternInsightCard
          sessionLen={sessionData.length}
          dreamFrames={totalDreamFrames}
          avgRem={avgRem}
          remCycles={remCycles.length}
          frameTs={latestFrame?.timestamp}
        />
      )}
    </main>
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
