import { useState, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChartTooltip } from "@/components/chart-tooltip";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { Brain, Heart, Activity, TrendingUp, Zap, Radio, Smile, Clock } from "lucide-react";
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

      {/* Live Circumplex + AI History (shown while streaming) */}
      {isStreaming && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Beautiful Circumplex */}
          <Card className="glass-card p-6">
            <div className="flex items-center gap-2 mb-3">
              <Smile className="h-4 w-4 text-pink-400" />
              <h3 className="text-sm font-medium">Emotion Circumplex</h3>
              <span className="text-[10px] font-mono text-primary animate-pulse ml-auto">● LIVE</span>
            </div>
            <ValenceArousalPlot
              valence={currentEmotion.valence}
              arousal={currentEmotion.arousal}
              emotion={currentEmotion.emotion}
              history={emotionHistory.slice(0, 24).map(h => ({ valence: h.valence, arousal: h.arousal, emotion: h.emotion }))}
            />
            <div className="flex justify-between text-[10px] text-muted-foreground mt-1 font-mono">
              <span>Valence: {(currentEmotion.valence * 100).toFixed(0)}%</span>
              <span>Arousal: {(currentEmotion.arousal * 100).toFixed(0)}%</span>
              <span className="capitalize" style={{ color: EMOTION_COLORS[currentEmotion.emotion] ?? "inherit" }}>
                {currentEmotion.emotion}
              </span>
            </div>
          </Card>

          {/* AI Emotion History */}
          <Card className="glass-card p-6">
            <div className="flex items-center gap-2 mb-3">
              <Clock className="h-4 w-4 text-cyan-400" />
              <h3 className="text-sm font-medium">AI Emotion History</h3>
              <span className="text-xs text-muted-foreground ml-auto">{emotionHistory.length} readings</span>
            </div>
            {emotionHistory.length === 0 ? (
              <div className="h-48 flex items-center justify-center text-sm text-muted-foreground border border-dashed border-border/30 rounded-lg">
                Waiting for emotion data…
              </div>
            ) : (
              <div className="space-y-1 max-h-[260px] overflow-y-auto pr-1">
                {emotionHistory.slice().reverse().map((e, i) => (
                  <div key={i} className="flex items-center gap-3 text-xs py-1.5 border-b border-border/10 last:border-0">
                    <span className="font-mono text-muted-foreground w-16 shrink-0">{e.time}</span>
                    <EmotionDot emotion={e.emotion} />
                    <span className="capitalize font-medium w-16">{e.emotion}</span>
                    <div className="flex-1 flex gap-2 text-[10px] text-muted-foreground">
                      <span>V:{(e.valence * 100).toFixed(0)}%</span>
                      <span>A:{(e.arousal * 100).toFixed(0)}%</span>
                    </div>
                    <span className="text-[10px] text-muted-foreground shrink-0">{(e.confidence * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      )}

      {/* Emotion Timeline — always visible */}
      <Card className="glass-card p-6">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-medium">Emotion Timeline</h3>
          {isLiveToday && isStreaming && (
            <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">● LIVE</span>
          )}
        </div>
        {!hasTimelineData ? (
          <div className="h-[220px] flex flex-col items-center justify-center text-sm text-muted-foreground gap-2">
            <Activity className="h-8 w-8 opacity-30" />
            <p>{isLiveToday ? (isStreaming ? "Collecting data…" : "Connect device to see live data") : "No sessions in this period"}</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.5} />
              <XAxis dataKey={timelineDataKey} tick={{ fontSize: 9, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
              <YAxis domain={[0, 100]} tick={{ fontSize: 9, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} width={24} />
              <Tooltip
                contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, fontSize: 11 }}
                formatter={(v: number) => [`${v}%`]}
              />
              <Line type="monotone" dataKey="focus_index"      name="Focus"  stroke="hsl(200, 70%, 55%)" strokeWidth={2}   dot={false} isAnimationActive={false} activeDot={{ r: 4 }} />
              <Line type="monotone" dataKey="stress_index"     name="Stress" stroke="hsl(38, 85%, 58%)"  strokeWidth={1.5} strokeDasharray="4 3" dot={false} isAnimationActive={false} activeDot={{ r: 4 }} />
              <Line type="monotone" dataKey="relaxation_index" name="Relax"  stroke="hsl(152, 60%, 48%)" strokeWidth={1.5} dot={false} isAnimationActive={false} activeDot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        )}
        <div className="flex gap-4 mt-2">
          {[
            { label: "Focus",  color: "hsl(200,70%,55%)" },
            { label: "Stress", color: "hsl(38,85%,58%)", dashed: true },
            { label: "Relax",  color: "hsl(152,60%,48%)" },
          ].map((l) => (
            <div key={l.label} className="flex items-center gap-1.5">
              <svg width="16" height="8"><line x1="0" y1="4" x2="16" y2="4" stroke={l.color} strokeWidth="2" strokeDasharray={l.dashed ? "4 3" : "0"} /></svg>
              <span className="text-[10px] text-muted-foreground">{l.label}</span>
            </div>
          ))}
        </div>
      </Card>
    </main>
  );
}

/* ── Shared emotion colors ─────────────────────────────────────── */
const EMOTION_COLORS: Record<string, string> = {
  happy:   "hsl(48,90%,58%)",
  excited: "hsl(28,90%,58%)",
  focused: "hsl(200,80%,58%)",
  relaxed: "hsl(152,70%,50%)",
  calm:    "hsl(165,65%,48%)",
  angry:   "hsl(5,80%,55%)",
  fearful: "hsl(270,60%,60%)",
  sad:     "hsl(210,70%,55%)",
  neutral: "hsl(220,18%,55%)",
};

const EMOTION_ANCHORS = [
  { name: "Happy",   v:  0.72, a: 0.70, color: EMOTION_COLORS.happy   },
  { name: "Excited", v:  0.48, a: 0.90, color: EMOTION_COLORS.excited  },
  { name: "Focused", v:  0.20, a: 0.65, color: EMOTION_COLORS.focused  },
  { name: "Relaxed", v:  0.68, a: 0.20, color: EMOTION_COLORS.relaxed  },
  { name: "Calm",    v:  0.38, a: 0.08, color: EMOTION_COLORS.calm     },
  { name: "Angry",   v: -0.68, a: 0.80, color: EMOTION_COLORS.angry    },
  { name: "Fearful", v: -0.40, a: 0.72, color: EMOTION_COLORS.fearful  },
  { name: "Sad",     v: -0.65, a: 0.18, color: EMOTION_COLORS.sad      },
];

/* ── Valence/Arousal Circumplex ────────────────────────────────── */
function ValenceArousalPlot({
  valence, arousal, emotion = "neutral", history,
}: {
  valence: number; arousal: number; emotion?: string;
  history: Array<{ valence: number; arousal: number; emotion: string }>;
}) {
  const SIZE = 300, CX = SIZE / 2, CY = SIZE / 2, R = SIZE / 2 - 32;
  const toXY = (v: number, a: number) => ({ x: CX + v * R, y: CY - (a * 2 - 1) * R });
  const dot      = toXY(valence, arousal);
  const dotColor = EMOTION_COLORS[emotion] ?? EMOTION_COLORS.neutral;
  const trail    = [...history].reverse();

  return (
    <svg width="100%" viewBox={`0 0 ${SIZE} ${SIZE}`} style={{ display: "block" }}>
      <defs>
        <clipPath id="el-va-clip"><circle cx={CX} cy={CY} r={R} /></clipPath>
        <filter id="el-va-glow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="5" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <radialGradient id="el-va-vignette" cx="50%" cy="50%" r="50%">
          <stop offset="60%" stopColor="transparent" stopOpacity="0" />
          <stop offset="100%" stopColor="hsl(220,22%,4%)" stopOpacity="0.55" />
        </radialGradient>
      </defs>
      <circle cx={CX} cy={CY} r={R} fill="hsl(220,24%,6%)" />
      <g clipPath="url(#el-va-clip)">
        <rect x={CX}     y={CY-R} width={R} height={R} fill="hsl(42,90%,54%)"  opacity={0.13} />
        <rect x={CX-R}   y={CY-R} width={R} height={R} fill="hsl(0,80%,54%)"   opacity={0.12} />
        <rect x={CX}     y={CY}   width={R} height={R} fill="hsl(152,68%,44%)" opacity={0.12} />
        <rect x={CX-R}   y={CY}   width={R} height={R} fill="hsl(215,70%,54%)" opacity={0.10} />
      </g>
      <circle cx={CX} cy={CY} r={R*0.33} fill="none" stroke="hsl(220,20%,16%)" strokeWidth={0.7} strokeDasharray="3 5" />
      <circle cx={CX} cy={CY} r={R*0.67} fill="none" stroke="hsl(220,20%,16%)" strokeWidth={0.7} strokeDasharray="3 5" />
      <circle cx={CX} cy={CY} r={R}       fill="none" stroke="hsl(220,20%,20%)" strokeWidth={1.4} />
      <circle cx={CX} cy={CY} r={R} fill="url(#el-va-vignette)" />
      <line x1={CX-R+2} y1={CY} x2={CX+R-2} y2={CY} stroke="hsl(220,20%,24%)" strokeWidth={0.9} />
      <line x1={CX} y1={CY+R-2} x2={CX} y2={CY-R+2} stroke="hsl(220,20%,24%)" strokeWidth={0.9} />
      <polygon points={`${CX+R-1},${CY} ${CX+R-7},${CY-3} ${CX+R-7},${CY+3}`}  fill="hsl(220,20%,30%)" />
      <polygon points={`${CX},${CY-R+1} ${CX-3},${CY-R+7} ${CX+3},${CY-R+7}`} fill="hsl(220,20%,30%)" />
      <text x={CX+R-10} y={CY-7}  fontSize={7.5} fill="hsl(220,15%,36%)" textAnchor="end"   fontFamily="monospace">+VALENCE</text>
      <text x={CX-R+10} y={CY-7}  fontSize={7.5} fill="hsl(220,15%,36%)" textAnchor="start" fontFamily="monospace">−VALENCE</text>
      <text x={CX+5}    y={CY-R+14} fontSize={7.5} fill="hsl(220,15%,36%)" textAnchor="start" fontFamily="monospace">HIGH AROUSAL</text>
      <text x={CX+5}    y={CY+R-6}  fontSize={7.5} fill="hsl(220,15%,36%)" textAnchor="start" fontFamily="monospace">LOW AROUSAL</text>
      <text x={CX+R*0.60} y={CY-R*0.80} fontSize={9.5} fill="hsl(42,88%,60%)"  textAnchor="middle" fontWeight="700" letterSpacing="0.6">HAPPY</text>
      <text x={CX+R*0.60} y={CY-R*0.66} fontSize={7}   fill="hsl(42,70%,44%)"  textAnchor="middle">Excited · Joyful</text>
      <text x={CX-R*0.60} y={CY-R*0.80} fontSize={9.5} fill="hsl(0,80%,58%)"   textAnchor="middle" fontWeight="700" letterSpacing="0.6">STRESSED</text>
      <text x={CX-R*0.60} y={CY-R*0.66} fontSize={7}   fill="hsl(0,65%,44%)"   textAnchor="middle">Angry · Fearful</text>
      <text x={CX+R*0.60} y={CY+R*0.68} fontSize={9.5} fill="hsl(152,65%,50%)" textAnchor="middle" fontWeight="700" letterSpacing="0.6">RELAXED</text>
      <text x={CX+R*0.60} y={CY+R*0.82} fontSize={7}   fill="hsl(152,52%,38%)" textAnchor="middle">Calm · Serene</text>
      <text x={CX-R*0.60} y={CY+R*0.68} fontSize={9.5} fill="hsl(210,70%,58%)" textAnchor="middle" fontWeight="700" letterSpacing="0.6">SAD</text>
      <text x={CX-R*0.60} y={CY+R*0.82} fontSize={7}   fill="hsl(210,55%,44%)" textAnchor="middle">Depressed · Bored</text>
      <text x={CX+R*0.24} y={CY-R*0.28} fontSize={7.5} fill="hsl(200,70%,52%)" textAnchor="middle" opacity={0.75}>Focused</text>
      {EMOTION_ANCHORS.map(({ name, v, a, color }) => {
        const p = toXY(v, a);
        return <g key={name}><circle cx={p.x} cy={p.y} r={3.5} fill={color} opacity={0.28} /><circle cx={p.x} cy={p.y} r={2} fill={color} opacity={0.55} /></g>;
      })}
      {trail.length > 1 && (
        <polyline clipPath="url(#el-va-clip)"
          points={trail.map(h => { const p = toXY(h.valence, h.arousal); return `${p.x},${p.y}`; }).join(" ")}
          fill="none" stroke={dotColor} strokeWidth={1.5} strokeOpacity={0.28} strokeLinecap="round" strokeLinejoin="round" />
      )}
      {history.map((h, i) => {
        const p = toXY(h.valence, h.arousal);
        const ratio = 1 - i / Math.max(history.length - 1, 1);
        const col = EMOTION_COLORS[h.emotion] ?? EMOTION_COLORS.neutral;
        return <circle key={i} cx={p.x} cy={p.y} r={1.8 + ratio * 2.2} fill={col} opacity={0.12 + ratio * 0.50} clipPath="url(#el-va-clip)" />;
      })}
      <circle cx={dot.x} cy={dot.y} r={12} fill="none" stroke={dotColor} strokeWidth={1.5} opacity={0.0}>
        <animate attributeName="r"       values="10;20;10" dur="2.6s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.35;0;0.35" dur="2.6s" repeatCount="indefinite" />
      </circle>
      <circle cx={dot.x} cy={dot.y} r={9}  fill={dotColor} opacity={0.18} />
      <circle cx={dot.x} cy={dot.y} r={6}  fill={dotColor} opacity={0.9} filter="url(#el-va-glow)" />
      <circle cx={dot.x} cy={dot.y} r={6}  fill={dotColor} />
      <circle cx={dot.x} cy={dot.y} r={2}  fill="white" opacity={0.92} />
    </svg>
  );
}

/* ── Emotion Dot ───────────────────────────────────────────────── */
function EmotionDot({ emotion }: { emotion: string }) {
  return (
    <span className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
      style={{ background: EMOTION_COLORS[emotion] ?? EMOTION_COLORS.neutral }} />
  );
}
