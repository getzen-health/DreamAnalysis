import { useState, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { ContinuousBrainTimeline } from "@/components/charts/continuous-brain-timeline";
import { ChartTooltip } from "@/components/chart-tooltip";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis,
} from "recharts";
import { Brain, Heart, Activity, TrendingUp, Zap, Radio } from "lucide-react";
import { EmotionWheel } from "@/components/emotion-wheel";
import { BrainBands } from "@/components/brain-bands";
import { useDevice } from "@/hooks/use-device";
import { listSessions, type SessionSummary } from "@/lib/ml-api";

/* ---------- constants ---------- */
const PERIOD_TABS = [
  { label: "Today", days: 1 },
  { label: "Week", days: 7 },
  { label: "Month", days: 30 },
  { label: "3 Months", days: 90 },
  { label: "Year", days: 365 },
];

/* ---------- types ---------- */
interface EmotionState {
  emotion: string;
  confidence: number;
  valence: number;
  arousal: number;
  stress_index: number;
  focus_index: number;
  relaxation_index: number;
  band_powers: Record<string, number>;
  probabilities: Record<string, number>;
}

interface HistoryEntry extends EmotionState {
  time: string;
}

interface SessionPoint {
  date: string;
  stress_index: number;
  focus_index: number;
  relaxation_index: number;
}

interface VAPoint {
  valence: number;
  arousal: number;
  emotion: string;
  size: number;
}

/* ---------- helpers ---------- */
function avgNums(arr: number[]): number {
  return arr.length ? parseFloat((arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2)) : 0;
}

// 1. Smarter emotion label based on band powers + raw emotion
function getSmartEmotionLabel(emotion: string, bands: Record<string, number>): string {
  const alpha = bands.alpha ?? 0;
  const beta  = bands.beta  ?? 0;
  const theta = bands.theta ?? 0;
  if (theta > 0.35 && beta < 0.22)  return "Drifting / Meditative";
  if (alpha > 0.35 && beta < 0.12)  return "Calm & Resting";
  if (alpha > 0.25 && beta > 0.22)  return "Focused Calm";
  if (beta  > 0.35 && alpha < 0.15) return "Alert & Active";
  if (alpha > 0.20 && theta > 0.25) return "Relaxed & Dreamy";
  if (emotion === "relaxed")  return "Relaxed & At Ease";
  if (emotion === "focused")  return "Mentally Engaged";
  if (emotion === "happy")    return "Positive & Uplifted";
  if (emotion === "sad")      return "Low & Withdrawn";
  if (emotion === "angry")    return "Tense & Activated";
  if (emotion === "fearful")  return "Anxious & On-Edge";
  return "Neutral State";
}

// 2. One-line brain state description from band powers
function getBrainStateDesc(bands: Record<string, number>): string {
  const alpha = bands.alpha ?? 0;
  const beta  = bands.beta  ?? 0;
  const theta = bands.theta ?? 0;
  const gamma = bands.gamma ?? 0;
  const parts: string[] = [];
  if (alpha > 0.3)       parts.push("high alpha");
  else if (alpha > 0.15) parts.push("moderate alpha");
  else                   parts.push("low alpha");
  if (beta > 0.25)       parts.push("elevated beta");
  else if (beta > 0.12)  parts.push("moderate beta");
  if (theta > 0.35)      parts.push("high theta");
  else if (theta > 0.2)  parts.push("moderate theta");
  if (gamma > 0.05)      parts.push("some gamma");
  const interpretation =
    alpha > 0.35 && beta < 0.12  ? "calm resting state" :
    theta > 0.35 && beta < 0.15  ? "drowsy / meditative" :
    beta  > 0.3  && alpha < 0.15 ? "highly activated cortex" :
    alpha > 0.2  && beta > 0.2   ? "balanced active-calm" :
                                   "transitional brain state";
  return parts.join(", ") + " — " + interpretation;
}

// 3. Confidence → color + label
function getConfidenceStyle(confidence: number): { color: string; label: string } {
  if (confidence >= 0.40) return { color: "hsl(152, 60%, 45%)", label: "High confidence" };
  if (confidence >= 0.30) return { color: "hsl(38, 85%, 55%)",  label: "Moderate confidence" };
  return                         { color: "hsl(220, 12%, 55%)", label: "Low confidence — neutral" };
}

// 4. Plain-English valence
function getValenceLabel(valence: number): string {
  if (valence >  0.5) return "Very positive mood";
  if (valence >  0.2) return "Positive mood";
  if (valence > -0.2) return "Neutral mood";
  if (valence > -0.5) return "Slightly negative";
  return "Negative mood";
}

// 5. Plain-English arousal
function getArousalLabel(arousal: number): string {
  if (arousal > 0.70) return "High activation";
  if (arousal > 0.50) return "Moderate activation";
  if (arousal > 0.30) return "Low-moderate · calm";
  return "Very low · deeply relaxed";
}

// Emoji for each emotion
const EMOTION_EMOJI: Record<string, string> = {
  happy: "😊", sad: "😔", angry: "😠", fearful: "😨",
  relaxed: "😌", focused: "🎯", neutral: "😶",
};

function buildEmotionChartData(sessions: SessionSummary[], days: number): SessionPoint[] {
  const map: Record<
    string,
    { stress: number[]; focus: number[]; relaxation: number[]; ts: number }
  > = {};

  for (const s of sessions) {
    if (s.summary?.avg_focus == null) continue;
    const d = new Date((s.start_time ?? 0) * 1000);
    let key: string;
    let ts: number;

    if (days <= 1) {
      key = d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
      ts = d.getTime();
    } else if (days <= 7) {
      key = d.toLocaleDateString("en-US", { weekday: "short" });
      ts = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    } else if (days <= 30) {
      key = d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
      ts = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    } else if (days <= 90) {
      const ws = new Date(d);
      ws.setDate(d.getDate() - d.getDay());
      key = ws.toLocaleDateString("en-US", { month: "short", day: "numeric" });
      ts = ws.getTime();
    } else {
      key = d.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
      ts = new Date(d.getFullYear(), d.getMonth(), 1).getTime();
    }

    if (!map[key]) map[key] = { stress: [], focus: [], relaxation: [], ts };
    map[key].stress.push((s.summary.avg_stress ?? 0) * 100);
    map[key].focus.push((s.summary.avg_focus ?? 0) * 100);
    map[key].relaxation.push((s.summary.avg_relaxation ?? 0) * 100);
  }

  return Object.entries(map)
    .sort(([, a], [, b]) => a.ts - b.ts)
    .map(([date, data]) => ({
      date,
      stress_index: Math.round(avgNums(data.stress)),
      focus_index: Math.round(avgNums(data.focus)),
      relaxation_index: Math.round(avgNums(data.relaxation)),
    }));
}

/* ========== Component ========== */
export default function EmotionLab() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;

  const [periodDays, setPeriodDays] = useState(1);
  const isLiveToday = periodDays === 1;

  // Emotion window history — last 5 thirty-second results
  const [emotionWindowHistory, setEmotionWindowHistory] = useState<
    Array<{ emotion: string; label: string; confidence: number; time: string }>
  >([]);

  // Sessions query
  const { data: allSessions = [] } = useQuery<SessionSummary[]>({
    queryKey: ["sessions"],
    queryFn: () => listSessions(),
    retry: false,
    staleTime: 2 * 60 * 1000,
    refetchInterval: 60_000,
  });

  // Build current emotion from live Muse 2 data
  const emotions = analysis?.emotions;
  const bandPowers = analysis?.band_powers ?? {};

  // emotion is null while the 30s buffer is still filling — show calibrating state
  const emotionReady = !emotions || emotions.ready !== false || emotions.emotion != null;
  const bufferedSec = emotions?.buffered_sec ?? 0;
  const windowSec = emotions?.window_sec ?? 30;

  const currentEmotion: EmotionState = emotions && emotionReady
    ? {
        emotion: emotions.emotion ?? "unknown",
        confidence: emotions.confidence ?? 0,
        valence: emotions.valence ?? 0,
        arousal: emotions.arousal ?? 0,
        stress_index: (emotions.stress_index ?? 0) * 100,
        focus_index: (emotions.focus_index ?? 0) * 100,
        relaxation_index: (emotions.relaxation_index ?? 0) * 100,
        band_powers: bandPowers,
        probabilities: emotions.probabilities ?? {},
      }
    : {
        emotion: emotions ? `Buffering ${bufferedSec}s / ${windowSec}s` : "—",
        confidence: 0,
        valence: 0,
        arousal: 0,
        stress_index: 0,
        focus_index: 0,
        relaxation_index: 0,
        band_powers: {},
        probabilities: {},
      };

  // Accumulate live history (every frame)
  const [emotionHistory, setEmotionHistory] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    if (!isStreaming || !emotions) return;
    const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    setEmotionHistory((prev) => [
      ...prev.slice(-60),
      { ...currentEmotion, time: now },
    ]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  // Track 30s emotion window changes
  useEffect(() => {
    if (!emotions?.ready || !emotions?.emotion) return;
    const now = new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    const label = getSmartEmotionLabel(emotions.emotion, emotions.band_powers ?? {});
    setEmotionWindowHistory((prev) => {
      // Only add if emotion actually changed or first entry
      const last = prev[prev.length - 1];
      if (last?.label === label && last?.emotion === emotions.emotion) return prev;
      return [...prev.slice(-4), { emotion: emotions.emotion!, label, confidence: emotions.confidence, time: now }];
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [emotions?.emotion, emotions?.ready]);

  // Build period-filtered session data
  const cutoff = Date.now() / 1000 - periodDays * 86400;
  const periodSessions = allSessions.filter((s) => (s.start_time ?? 0) >= cutoff);
  const sessionChartData = buildEmotionChartData(periodSessions, periodDays);

  // Timeline chart data
  const liveNow: HistoryEntry | null = isLiveToday && isStreaming
    ? { ...currentEmotion, time: "Now" }
    : null;
  const todayTimeline = liveNow ? [...emotionHistory.slice(-30), liveNow] : emotionHistory.slice(-30);
  const timelineData = isLiveToday ? todayTimeline : sessionChartData;
  const timelineDataKey = isLiveToday ? "time" : "date";
  const hasTimelineData = timelineData.length >= 1;

  // V-A scatter data
  const liveVaData: VAPoint[] = emotionHistory.map((e, i) => ({
    valence: e.valence,
    arousal: e.arousal,
    emotion: e.emotion,
    size: 30 + i * 2,
  }));

  const historicalVaData: VAPoint[] = periodSessions.map((s, i) => ({
    valence: s.summary.avg_valence ?? 0,
    arousal: s.summary.avg_arousal ?? 0,
    emotion: s.summary.dominant_emotion ?? "",
    size: 80,
  }));

  const vaData = isLiveToday ? liveVaData : historicalVaData;
  const hasVaData = vaData.length >= 1;

  return (
    <main className="p-6 space-y-6 max-w-5xl">
      {/* Connection status */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          Connect your Muse 2 from the sidebar to see live emotion data.
        </div>
      )}

      {/* Top Row — always live */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Emotion Wheel */}
        <Card className="glass-card p-6">
          <h3 className="text-sm font-medium mb-4 flex items-center gap-2">
            <Heart className="h-4 w-4 text-primary" />
            Current Emotion
            {isStreaming && (
              <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </h3>

          {!emotionReady && emotions ? (
            <div className="flex flex-col items-center justify-center h-48 gap-3">
              <div className="text-4xl animate-pulse">🧠</div>
              <p className="text-sm text-muted-foreground">Buffering EEG data…</p>
              <div className="w-40 h-2 rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-primary transition-all duration-1000"
                  style={{ width: `${Math.min(100, (bufferedSec / windowSec) * 100)}%` }} />
              </div>
              <p className="text-xs text-muted-foreground font-mono">{bufferedSec}s / {windowSec}s</p>
            </div>
          ) : (
            <>
              <EmotionWheel
                probabilities={currentEmotion.probabilities}
                dominantEmotion={currentEmotion.emotion}
                confidence={currentEmotion.confidence}
              />

              {emotionReady && currentEmotion.emotion && currentEmotion.emotion !== "—" && (() => {
                const smartLabel = getSmartEmotionLabel(currentEmotion.emotion, currentEmotion.band_powers);
                const confStyle  = getConfidenceStyle(currentEmotion.confidence);
                const emoji      = EMOTION_EMOJI[currentEmotion.emotion] ?? "🧠";
                const stateDesc  = getBrainStateDesc(currentEmotion.band_powers);
                return (
                  <div className="mt-3 space-y-2">
                    {/* Smart label + confidence badge */}
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-semibold text-foreground">
                        {emoji} {smartLabel}
                      </span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded-full"
                        style={{ background: confStyle.color + "22", color: confStyle.color }}>
                        {confStyle.label}
                      </span>
                    </div>
                    {/* Brain state description */}
                    <p className="text-[11px] text-muted-foreground leading-relaxed">{stateDesc}</p>
                  </div>
                );
              })()}

              {/* Emotion window history — last 5 thirty-second windows */}
              {emotionWindowHistory.length > 0 && (
                <div className="mt-3 pt-3 border-t border-border">
                  <p className="text-[10px] text-muted-foreground mb-2">30s window history</p>
                  <div className="flex gap-1.5 flex-wrap">
                    {emotionWindowHistory.map((w, i) => {
                      const isLatest = i === emotionWindowHistory.length - 1;
                      const emoji = EMOTION_EMOJI[w.emotion] ?? "🧠";
                      return (
                        <div key={i}
                          className="flex flex-col items-center gap-0.5 px-2 py-1 rounded-lg text-center"
                          style={{ background: isLatest ? "hsl(152,60%,40%,0.15)" : "hsl(220,14%,15%)" }}>
                          <span className="text-sm">{emoji}</span>
                          <span className="text-[9px] text-muted-foreground leading-none">{w.time}</span>
                          <span className="text-[9px] font-medium capitalize leading-none"
                            style={{ color: isLatest ? "hsl(152,60%,55%)" : undefined }}>
                            {w.emotion}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </>
          )}
        </Card>

        {/* Brain Bands */}
        <Card className="glass-card p-6">
          <h3 className="text-sm font-medium mb-4 flex items-center gap-2">
            <Brain className="h-4 w-4 text-primary" />
            Brain Waves
            {isStreaming && (
              <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </h3>
          <BrainBands bandPowers={currentEmotion.band_powers} />
        </Card>

        {/* Mental State */}
        <Card className="glass-card p-6">
          <h3 className="text-sm font-medium mb-4 flex items-center gap-2">
            <Activity className="h-4 w-4 text-secondary" />
            Mental State
          </h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between mb-1 text-sm">
                <span className="text-muted-foreground">Stress</span>
                <span className="font-mono text-warning">{Math.round(currentEmotion.stress_index)}</span>
              </div>
              <Progress value={currentEmotion.stress_index} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between mb-1 text-sm">
                <span className="text-muted-foreground">Focus</span>
                <span className="font-mono text-primary">{Math.round(currentEmotion.focus_index)}</span>
              </div>
              <Progress value={currentEmotion.focus_index} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between mb-1 text-sm">
                <span className="text-muted-foreground">Relaxation</span>
                <span className="font-mono text-success">{Math.round(currentEmotion.relaxation_index)}</span>
              </div>
              <Progress value={currentEmotion.relaxation_index} className="h-2" />
            </div>
            <div className="pt-3 border-t border-border text-sm space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Mood</span>
                <span className={`text-xs font-medium ${currentEmotion.valence >= 0 ? "text-success" : "text-destructive"}`}>
                  {getValenceLabel(currentEmotion.valence)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Activation</span>
                <span className="text-xs font-medium text-secondary">
                  {getArousalLabel(currentEmotion.arousal)}
                </span>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Period Selector — shared for Timeline + V-A */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-primary" />
          <span className="text-sm font-medium">Emotion History</span>
          {isLiveToday && isStreaming && (
            <span className="text-[10px] font-mono text-primary animate-pulse">● LIVE</span>
          )}
        </div>
        <div className="flex gap-1 flex-wrap">
          {PERIOD_TABS.map((tab) => (
            <button
              key={tab.days}
              onClick={() => setPeriodDays(tab.days)}
              className={`px-3 py-1 text-xs rounded-full transition-colors ${
                periodDays === tab.days
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Continuous Timeline (TimescaleDB) */}
        <ContinuousBrainTimeline
          userId="default"
          defaultMetric="focus_index"
          title="Emotion History"
        />

        {/* Valence-Arousal Space */}
        <Card className="glass-card p-6">
          <h3 className="text-sm font-medium mb-4 flex items-center gap-2">
            <Zap className="h-4 w-4 text-warning" />
            Valence-Arousal Space
            {!isLiveToday && (
              <span className="ml-auto text-[10px] text-muted-foreground">{periodSessions.length} sessions</span>
            )}
          </h3>
          {!hasVaData ? (
            <div className="h-[220px] flex items-center justify-center text-sm text-muted-foreground">
              {isLiveToday
                ? isStreaming ? "Collecting data..." : "Connect device to see V-A plot"
                : "No sessions in this period"}
            </div>
          ) : (
            <>
            <ResponsiveContainer width="100%" height={220}>
              <ScatterChart>
                <defs>
                  <radialGradient id="vaGlow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="hsl(152, 60%, 55%)" stopOpacity={0.9} />
                    <stop offset="70%" stopColor="hsl(200, 70%, 55%)" stopOpacity={0.6} />
                    <stop offset="100%" stopColor="hsl(262, 45%, 65%)" stopOpacity={0.3} />
                  </radialGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 22%)" opacity={0.6} />
                <XAxis type="number" dataKey="valence" domain={[-1, 1]} name="Valence" tick={{ fontSize: 10, fill: "hsl(220, 12%, 55%)" }} axisLine={{ stroke: "hsl(220, 18%, 25%)" }} tickLine={false} />
                <YAxis type="number" dataKey="arousal" domain={[0, 1]} name="Arousal" tick={{ fontSize: 10, fill: "hsl(220, 12%, 55%)" }} axisLine={{ stroke: "hsl(220, 18%, 25%)" }} tickLine={false} />
                <ZAxis type="number" dataKey="size" range={[40, 200]} />
                <Tooltip
                  contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, fontSize: 12, color: "hsl(38, 20%, 92%)" }}
                  labelStyle={{ color: "hsl(38, 20%, 92%)" }}
                  itemStyle={{ color: "hsl(220, 12%, 75%)" }}
                  content={({ payload }) => {
                    if (!payload?.length) return null;
                    const d = payload[0]?.payload as { valence?: number; arousal?: number; emotion?: string };
                    return (
                      <div style={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, padding: "8px 12px", fontSize: 11 }}>
                        {d.emotion && <p style={{ color: "hsl(38, 85%, 65%)", fontWeight: 600, marginBottom: 4, textTransform: "capitalize" }}>{d.emotion}</p>}
                        <p style={{ color: "hsl(220, 12%, 75%)" }}>Valence: {d.valence?.toFixed(2)}</p>
                        <p style={{ color: "hsl(220, 12%, 75%)" }}>Arousal: {d.arousal?.toFixed(2)}</p>
                      </div>
                    );
                  }}
                />
                <Scatter data={vaData} fill="url(#vaGlow)" stroke="hsl(152, 60%, 55%)" strokeWidth={1} fillOpacity={0.85} />
              </ScatterChart>
            </ResponsiveContainer>
            <div className="grid grid-cols-3 text-[10px] text-muted-foreground mt-1 px-2">
              <span>← Negative mood</span>
              <span className="text-center font-medium">Valence (X)</span>
              <span className="text-right">Positive mood →</span>
            </div>
            <div className="text-[10px] text-muted-foreground text-center mt-0.5">Arousal (Y): low = calm · high = activated</div>
            </>
          )}
        </Card>
      </div>
    </main>
  );
}
