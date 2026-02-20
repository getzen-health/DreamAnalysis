import { useMemo, useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { EEGWaveformCanvas } from "@/components/charts/eeg-waveform-canvas";
import { SpectrogramChart } from "@/components/charts/spectrogram-chart";
import { NeuralNetwork } from "@/components/neural-network";
import { SignalQualityBadge } from "@/components/signal-quality-badge";
import { AlertBanner, type AlertLevel } from "@/components/alert-banner";
import { SessionControls } from "@/components/session-controls";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Activity,
  Radio,
  Moon,
  Heart,
  Brain,
  Zap,
  Eye,
  Gauge,
  Smile,
  Lightbulb,
  Battery,
  Shield,
  Clock,
  TrendingUp,
  TrendingDown,
  Minus,
} from "lucide-react";
import { useInference } from "@/hooks/use-inference";
import { useDevice } from "@/hooks/use-device";
import {
  analyzeWavelet,
  getEmotionHistory,
  getTodayTotals,
  getAtThisTimeYesterday,
  type WaveletResult,
  type AnomalyResult,
  type StoredEmotionReading,
  type TodayTotals,
  type YesterdayComparison,
} from "@/lib/ml-api";
import { MoodMusicPlayer } from "@/components/mood-music-player";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

// ── History window options ─────────────────────────────────────────
type HistoryWindow = "today" | "yesterday" | "week";

const HISTORY_DAYS: Record<HistoryWindow, number> = {
  today: 1,
  yesterday: 2, // fetch 2 days, then filter to yesterday
  week: 7,
};

// ── User ID (placeholder — replace with real auth context) ─────────
const CURRENT_USER_ID = "default";

export default function BrainMonitor() {
  const { isLocal, latencyMs, isReady } = useInference();
  const device = useDevice();
  const { state: deviceState, latestFrame, deviceStatus } = device;
  const isStreaming = deviceState === "streaming";

  const [wavelet, setWavelet] = useState<WaveletResult | null>(null);
  const [anomaly] = useState<AnomalyResult | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [historyWindow, setHistoryWindow] = useState<HistoryWindow>("today");
  const waveletTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Live frames accumulated during streaming for chart merge
  const liveFramesRef = useRef<StoredEmotionReading[]>([]);

  // Emotion history for the recent-readings panel (last 30 entries)
  const [emotionHistory, setEmotionHistory] = useState<Array<{
    emotion: string;
    valence: number;
    arousal: number;
    confidence: number;
    ts: string;
  }>>([]);

  // ── DB history query ─────────────────────────────────────────────
  const { data: dbHistory = [] } = useQuery<StoredEmotionReading[]>({
    queryKey: ["emotion-history", CURRENT_USER_ID, historyWindow],
    queryFn: () => getEmotionHistory(CURRENT_USER_ID, HISTORY_DAYS[historyWindow]),
    refetchInterval: 60_000, // refresh every minute
    staleTime: 30_000,
  });

  const { data: todayTotals } = useQuery<TodayTotals>({
    queryKey: ["today-totals", CURRENT_USER_ID],
    queryFn: () => getTodayTotals(CURRENT_USER_ID),
    refetchInterval: 60_000,
    staleTime: 30_000,
  });

  const { data: yesterdayData } = useQuery<YesterdayComparison>({
    queryKey: ["yesterday-comparison", CURRENT_USER_ID],
    queryFn: () => getAtThisTimeYesterday(CURRENT_USER_ID),
    staleTime: 5 * 60_000,
  });

  // ── Accumulate live frames into liveFramesRef ────────────────────
  const analysis = latestFrame?.analysis;
  useEffect(() => {
    if (!isStreaming || !analysis?.emotions) return;
    const now = new Date().toISOString();
    liveFramesRef.current.push({
      id: `live-${Date.now()}`,
      userId: CURRENT_USER_ID,
      sessionId: null,
      stress: analysis.emotions.stress_index ?? 0,
      happiness: 1 - (analysis.emotions.stress_index ?? 0),
      focus: analysis.emotions.focus_index ?? 0,
      energy: analysis.emotions.arousal ?? 0,
      dominantEmotion: analysis.emotions.emotion ?? "unknown",
      valence: analysis.emotions.valence ?? null,
      arousal: analysis.emotions.arousal ?? null,
      timestamp: now,
    });
    // Cap live buffer at 500 frames (~8 min at 1fps)
    if (liveFramesRef.current.length > 500) {
      liveFramesRef.current = liveFramesRef.current.slice(-500);
    }

    // Also push to emotion history panel (keep last 30)
    if (analysis.emotions.emotion) {
      setEmotionHistory(prev => {
        const entry = {
          emotion: analysis.emotions!.emotion ?? "unknown",
          valence: analysis.emotions!.valence ?? 0,
          arousal: analysis.emotions!.arousal ?? 0,
          confidence: analysis.emotions!.confidence ?? 0,
          ts: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
        };
        return [entry, ...prev].slice(0, 30);
      });
    }
  }, [latestFrame?.timestamp]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Merged timeline: DB history + live frames (deduplicated by ts) ─
  const mergedTimeline = useMemo(() => {
    const liveFrames = liveFramesRef.current;
    const dbIds = new Set(dbHistory.map(r => r.id));
    const uniqueLive = liveFrames.filter(f => !dbIds.has(f.id));
    const combined = [...dbHistory, ...uniqueLive];
    combined.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
    return combined;
  }, [dbHistory, isStreaming]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Chart data — downsample to max 300 points for performance ────
  const chartData = useMemo(() => {
    if (mergedTimeline.length === 0) return [];
    const step = Math.max(1, Math.floor(mergedTimeline.length / 300));
    return mergedTimeline
      .filter((_, i) => i % step === 0)
      .map(r => ({
        time: new Date(r.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        stress: Math.round(r.stress * 100),
        focus: Math.round(r.focus * 100),
        happiness: Math.round(r.happiness * 100),
        energy: Math.round(r.energy * 100),
      }));
  }, [mergedTimeline]);

  // ── Wavelet analysis every 2s ───────────────────────────────────
  useEffect(() => {
    if (!isStreaming || !latestFrame?.signals) {
      if (waveletTimerRef.current) clearInterval(waveletTimerRef.current);
      return;
    }

    waveletTimerRef.current = setInterval(async () => {
      if (latestFrame?.signals) {
        try {
          const result = await analyzeWavelet(latestFrame.signals, latestFrame.sample_rate || 256);
          setWavelet(result);
        } catch {
          // Wavelet analysis not available
        }
      }
    }, 2000);

    return () => {
      if (waveletTimerRef.current) clearInterval(waveletTimerRef.current);
    };
  }, [isStreaming, latestFrame]);

  // Throttled model card snapshot (5s)
  const [stableAnalysis, setStableAnalysis] = useState(analysis);
  const modelCardTimerRef = useRef(0);
  const MODEL_CARD_THROTTLE = 5_000;

  useEffect(() => {
    if (!isStreaming || !analysis) return;
    const now = Date.now();
    if (now - modelCardTimerRef.current < MODEL_CARD_THROTTLE && stableAnalysis) return;
    modelCardTimerRef.current = now;
    setStableAnalysis(analysis);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  const rawQuality = latestFrame?.quality as Record<string, unknown> | undefined;
  const signalQuality = rawQuality
    ? {
        sqi: ((rawQuality.sqi as number) ?? (rawQuality.quality_score as number) ?? 0) * 100,
        artifacts_detected: (rawQuality.artifacts_detected as string[]) ?? (rawQuality.rejection_reasons as string[]) ?? [],
        clean_ratio: (rawQuality.clean_ratio as number) ?? 0,
        channel_quality: (rawQuality.channel_quality as number[]) ?? [],
      }
    : null;

  const alphaHz = analysis?.band_powers?.alpha
    ? `${(analysis.band_powers.alpha * 12).toFixed(1)} Hz`
    : "—";
  const betaHz = analysis?.band_powers?.beta
    ? `${(analysis.band_powers.beta * 30).toFixed(1)} Hz`
    : "—";

  const sourceLabel = isStreaming ? "LIVE" : "OFFLINE";
  const sourceColor = isStreaming ? "text-primary" : "text-muted-foreground";
  const alertLevel: AlertLevel = anomaly?.alert_level || "normal";

  // Electrode grid
  const electrodeStatuses = useMemo(() => {
    const channelQuality = signalQuality?.channel_quality;
    const nChannels = deviceStatus?.n_channels ?? (channelQuality?.length || 4);
    return Array.from({ length: nChannels }, (_, i) => {
      let statusClass: string;
      if (channelQuality && i < channelQuality.length) {
        const q = channelQuality[i];
        statusClass =
          q < 60
            ? "border-destructive/30 bg-destructive/20"
            : q < 80
              ? "border-warning/30 bg-warning/20"
              : "border-success/30 bg-success/20";
      } else {
        statusClass = "border-border/30 bg-muted/10";
      }
      const label = `${String.fromCharCode(65 + Math.floor(i / 8))}${(i % 8) + 1}`;
      return { statusClass, label };
    });
  }, [signalQuality?.channel_quality, deviceStatus?.n_channels]);

  const channelQuality = signalQuality?.channel_quality || [];
  const hasRealData = channelQuality.length > 0;
  const activeCount = hasRealData ? channelQuality.filter((q) => q >= 80).length : 0;
  const weakCount = hasRealData ? channelQuality.filter((q) => q >= 60 && q < 80).length : 0;
  const errorCount = hasRealData ? channelQuality.filter((q) => q < 60).length : 0;

  return (
    <main className="p-4 md:p-6 space-y-6">
      {/* Connection Banner */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          Connect your Muse 2 from the sidebar to see live brain data. Historical data is shown below.
        </div>
      )}

      {/* Alert Banner */}
      <AlertBanner
        level={alertLevel}
        anomalyScore={anomaly?.anomaly_score}
        seizureProbability={anomaly?.seizure_probability}
        spikesDetected={anomaly?.spikes_detected}
      />

      {/* Emotion Shift Alert */}
      {latestFrame?.emotion_shift?.shift_detected && (
        <div className="shift-alert p-4 rounded-xl flex items-start gap-3">
          <Zap className="h-5 w-5 text-accent shrink-0 mt-0.5" />
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-medium">Emotional Shift</span>
              {latestFrame.emotion_shift.previous_emotion && latestFrame.emotion_shift.previous_emotion !== "unknown" && (
                <>
                  <Badge variant="outline" className="text-[10px] capitalize">{latestFrame.emotion_shift.previous_emotion}</Badge>
                  <span className="text-xs text-muted-foreground">&rarr;</span>
                </>
              )}
              {latestFrame.emotion_shift.current_emotion && (
                <Badge className="text-[10px] capitalize">{latestFrame.emotion_shift.current_emotion}</Badge>
              )}
              <span className="text-[10px] text-muted-foreground ml-auto">
                {((latestFrame.emotion_shift.magnitude ?? 0) * 100).toFixed(0)}% magnitude
              </span>
            </div>
            {latestFrame.emotion_shift.reason && (
              <p className="text-xs text-foreground/70">{latestFrame.emotion_shift.reason}</p>
            )}
            <p className="text-xs text-muted-foreground">{latestFrame.emotion_shift.description}</p>
            {latestFrame.emotion_shift.body_feeling && (
              <p className="text-[10px] text-muted-foreground italic">{latestFrame.emotion_shift.body_feeling}</p>
            )}
            {latestFrame.emotion_shift.guidance && (
              <p className="text-[10px] text-accent/80">{latestFrame.emotion_shift.guidance}</p>
            )}
          </div>
        </div>
      )}

      {/* Mood Music Player */}
      <MoodMusicPlayer
        emotion={stableAnalysis?.emotions?.emotion ?? undefined}
        isStreaming={isStreaming}
      />

      {/* ── Today's Totals (from DB) ──────────────────────────────────── */}
      {todayTotals && todayTotals.count > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <TodayCard label="Avg Stress" value={todayTotals.avgStress} yesterday={yesterdayData?.avgStress} color="text-rose-400" />
          <TodayCard label="Avg Focus" value={todayTotals.avgFocus} yesterday={yesterdayData?.avgFocus} color="text-cyan-400" />
          <TodayCard label="Avg Happiness" value={todayTotals.avgHappiness} yesterday={yesterdayData?.avgHappiness} color="text-amber-400" />
          <TodayCard label="Avg Energy" value={todayTotals.avgEnergy} yesterday={yesterdayData?.avgEnergy} color="text-green-400" />
        </div>
      )}

      {/* ── Historical Emotion Timeline Chart ─────────────────────────── */}
      <Card className="glass-card p-6 rounded-xl hover-glow">
        <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-primary" />
            <h3 className="text-lg font-semibold">Emotion Timeline</h3>
            <span className="text-xs text-muted-foreground font-mono">
              {mergedTimeline.length} readings
            </span>
          </div>
          {/* Window toggle */}
          <div className="flex rounded-lg border border-border/30 overflow-hidden text-xs font-mono">
            {(["today", "yesterday", "week"] as HistoryWindow[]).map(w => (
              <button
                key={w}
                onClick={() => setHistoryWindow(w)}
                className={`px-3 py-1.5 capitalize transition-colors ${historyWindow === w ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"}`}
              >
                {w}
              </button>
            ))}
          </div>
        </div>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={chartData} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
              <defs>
                <linearGradient id="stressGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(340,70%,55%)" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="hsl(340,70%,55%)" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="focusGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(200,70%,55%)" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="hsl(200,70%,55%)" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="happinessGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(38,85%,58%)" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="hsl(38,85%,58%)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(220,22%,16%)" />
              <XAxis dataKey="time" tick={{ fontSize: 10, fill: "hsl(220,15%,50%)" }} interval="preserveStartEnd" />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "hsl(220,15%,50%)" }} />
              <Tooltip
                contentStyle={{ background: "hsl(220,22%,10%)", border: "1px solid hsl(220,22%,20%)", borderRadius: 8, fontSize: 11 }}
                formatter={(v: number) => `${v}%`}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Area type="monotone" dataKey="stress" stroke="hsl(340,70%,55%)" fill="url(#stressGrad)" strokeWidth={1.5} dot={false} name="Stress" />
              <Area type="monotone" dataKey="focus" stroke="hsl(200,70%,55%)" fill="url(#focusGrad)" strokeWidth={1.5} dot={false} name="Focus" />
              <Area type="monotone" dataKey="happiness" stroke="hsl(38,85%,58%)" fill="url(#happinessGrad)" strokeWidth={1.5} dot={false} name="Happiness" />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-48 flex flex-col items-center justify-center text-sm text-muted-foreground border border-dashed border-border/30 rounded-lg gap-2">
            <Brain className="h-8 w-8 opacity-30" />
            No historical data yet — start a recording session to see your emotion timeline
          </div>
        )}
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* EEG Brain Waves */}
        <div className="xl:col-span-2 glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold">
              EEG Brain Wave Activity
            </h3>
            <div className="flex items-center space-x-3">
              {isStreaming && <SessionControls onRecordingChange={setIsRecording} />}
              {signalQuality && (
                <SignalQualityBadge
                  sqi={signalQuality.sqi}
                  artifacts={signalQuality.artifacts_detected}
                  compact
                />
              )}
              {isStreaming && (
                <Radio className="h-4 w-4 text-primary animate-pulse" />
              )}
              <span className={`text-sm font-mono ${sourceColor}`}>
                {sourceLabel}
              </span>
              {isReady && isStreaming && (
                <span className="text-xs font-mono text-foreground/40" title={`Inference: ${isLocal ? "local" : "server"} (${latencyMs.toFixed(0)}ms)`}>
                  {isLocal ? "LOCAL" : "SERVER"}
                </span>
              )}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="text-center">
              <p className="text-sm text-primary font-mono">Alpha Waves</p>
              <p className="text-2xl font-bold text-primary" data-testid="alpha-waves">
                {alphaHz}
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-secondary font-mono">Beta Waves</p>
              <p className="text-2xl font-bold text-secondary" data-testid="beta-waves">
                {betaHz}
              </p>
            </div>
          </div>
          {isStreaming ? (
            <>
              <EEGWaveformCanvas
                signals={latestFrame?.signals as number[][] | undefined}
                windowSec={5}
                height={280}
              />
              {/* Channel legend */}
              <div className="flex items-center gap-4 mt-3">
                {[
                  { label: "TP9",  color: "hsl(200, 70%, 55%)" },
                  { label: "AF7",  color: "hsl(152, 60%, 48%)" },
                  { label: "AF8",  color: "hsl(38,  85%, 58%)" },
                  { label: "TP10", color: "hsl(262, 45%, 65%)" },
                ].map((ch) => (
                  <div key={ch.label} className="flex items-center gap-1.5">
                    <span
                      className="inline-block w-2.5 h-2.5 rounded-full"
                      style={{ background: ch.color }}
                    />
                    <span className="text-[10px] font-mono text-muted-foreground">{ch.label}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="h-64 flex items-center justify-center text-sm text-muted-foreground border border-dashed border-border/30 rounded-lg">
              Connect device to see live EEG waveforms
            </div>
          )}
        </div>

        {/* Neural Network Graph */}
        <div className="glass-card p-6 rounded-xl hover-glow neural-glow">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold">Brain Regions</h3>
            <Activity className="text-accent" />
          </div>
          {isStreaming ? (
            <NeuralNetwork />
          ) : (
            <div className="h-80 flex items-center justify-center text-sm text-muted-foreground border border-dashed border-border/30 rounded-lg">
              Connect device to see brain region activity
            </div>
          )}
        </div>
      </div>

      {/* ── Live 12-Model Analysis Panel (throttled to 5s) ────────────── */}
      {isStreaming && stableAnalysis && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          <ModelCard icon={<Moon className="h-4 w-4" />} title="Sleep Stage" value={stableAnalysis.sleep_staging?.stage ?? "—"} score={stableAnalysis.sleep_staging?.confidence} color="text-blue-400" />
          <ModelCard icon={<Smile className="h-4 w-4" />} title="Emotion" value={stableAnalysis.emotions?.emotion ?? "—"} score={stableAnalysis.emotions?.confidence} color="text-pink-400"
            extra={stableAnalysis.emotions ? (
              <div className="mt-2 space-y-1.5">
                <div className="grid grid-cols-3 gap-1 text-[10px] text-muted-foreground">
                  <span>V:{(stableAnalysis.emotions.valence * 100).toFixed(0)}%</span>
                  <span>A:{(stableAnalysis.emotions.arousal * 100).toFixed(0)}%</span>
                  <span>S:{(stableAnalysis.emotions.stress_index * 100).toFixed(0)}%</span>
                </div>
                {stableAnalysis.emotions.probabilities && (
                  <div className="space-y-0.5">
                    {Object.entries(stableAnalysis.emotions.probabilities as Record<string, number>)
                      .sort((a, b) => b[1] - a[1])
                      .map(([emo, prob]) => (
                        <div key={emo} className="flex items-center gap-1.5">
                          <span className="text-[9px] text-muted-foreground w-12 capitalize shrink-0">{emo}</span>
                          <div className="flex-1 h-1 rounded-full overflow-hidden" style={{ background: "hsl(220,22%,12%)" }}>
                            <div className="h-full rounded-full transition-all duration-700" style={{ width: `${Math.round(prob * 100)}%`, background: "hsl(340,70%,55%)" }} />
                          </div>
                          <span className="text-[9px] text-muted-foreground w-6 text-right">{Math.round(prob * 100)}%</span>
                        </div>
                      ))}
                  </div>
                )}
              </div>
            ) : null} />
          <ModelCard icon={<Moon className="h-4 w-4" />} title="Dream" value={stableAnalysis.dream_detection?.is_dreaming ? "Dreaming" : "Awake"} score={stableAnalysis.dream_detection?.probability} color="text-purple-400"
            extra={stableAnalysis.dream_detection ? (<div className="text-[10px] text-muted-foreground mt-1">REM: {(stableAnalysis.dream_detection.rem_likelihood * 100).toFixed(0)}% | Lucidity: {(stableAnalysis.dream_detection.lucidity_estimate * 100).toFixed(0)}%</div>) : null} />
          <ModelCard icon={<Zap className="h-4 w-4" />} title="Flow State" value={stableAnalysis.flow_state?.in_flow ? "In Flow" : "Normal"} score={stableAnalysis.flow_state?.flow_score} color="text-green-400" />
          <ModelCard icon={<Lightbulb className="h-4 w-4" />} title="Creativity" value={stableAnalysis.creativity?.state ?? "—"} score={stableAnalysis.creativity?.creativity_score} color="text-amber-400" />
          <ModelCard icon={<Brain className="h-4 w-4" />} title="Memory" value={stableAnalysis.memory_encoding?.state ?? "—"} score={stableAnalysis.memory_encoding?.encoding_score} color="text-cyan-400" />
          <ModelCard icon={<Eye className="h-4 w-4" />} title="Attention" value={stableAnalysis.attention?.state ?? "—"} score={stableAnalysis.attention?.attention_score} color="text-emerald-400" />
          <ModelCard icon={<Battery className="h-4 w-4" />} title="Drowsiness" value={stableAnalysis.drowsiness?.state ?? "—"} score={stableAnalysis.drowsiness?.drowsiness_index} color="text-orange-400" />
          <ModelCard icon={<Gauge className="h-4 w-4" />} title="Cognitive Load" value={stableAnalysis.cognitive_load?.level ?? "—"} score={stableAnalysis.cognitive_load?.load_index} color="text-red-400" />
          <ModelCard icon={<Shield className="h-4 w-4" />} title="Stress" value={stableAnalysis.stress?.level ?? "—"} score={stableAnalysis.stress?.stress_index} color="text-rose-400" />
          <ModelCard icon={<Heart className="h-4 w-4" />} title="Meditation" value={stableAnalysis.meditation?.depth ?? "—"} score={stableAnalysis.meditation?.meditation_score} color="text-violet-400" />
          {stableAnalysis.lucid_dream && (
            <ModelCard icon={<Moon className="h-4 w-4" />} title="Lucid Dream" value={stableAnalysis.lucid_dream.state} score={stableAnalysis.lucid_dream.lucidity_score} color="text-fuchsia-400" />
          )}
        </div>
      )}

      {/* ── Valence/Arousal Circumplex + Emotion History ──────────────── */}
      {isStreaming && stableAnalysis?.emotions && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Valence/Arousal 2D Space */}
          <Card className="glass-card p-6 rounded-xl hover-glow">
            <div className="flex items-center gap-2 mb-4">
              <Smile className="h-4 w-4 text-pink-400" />
              <h3 className="text-lg font-semibold">Emotion Circumplex</h3>
              <span className="text-xs text-muted-foreground ml-auto">Valence × Arousal</span>
            </div>
            <ValenceArousalPlot
              valence={stableAnalysis.emotions.valence ?? 0}
              arousal={stableAnalysis.emotions.arousal ?? 0}
              emotion={stableAnalysis.emotions.emotion ?? "neutral"}
              history={emotionHistory.slice(0, 24).map(h => ({ valence: h.valence, arousal: h.arousal, emotion: h.emotion }))}
            />
            <div className="flex justify-between text-[10px] text-muted-foreground mt-2 font-mono">
              <span>Valence: {((stableAnalysis.emotions.valence ?? 0) * 100).toFixed(0)}%</span>
              <span>Arousal: {((stableAnalysis.emotions.arousal ?? 0) * 100).toFixed(0)}%</span>
              <span className="capitalize text-pink-400">{stableAnalysis.emotions.emotion ?? "—"}</span>
            </div>
          </Card>

          {/* Emotion History List */}
          <Card className="glass-card p-6 rounded-xl hover-glow">
            <div className="flex items-center gap-2 mb-4">
              <Clock className="h-4 w-4 text-cyan-400" />
              <h3 className="text-lg font-semibold">AI Emotion History</h3>
              <span className="text-xs text-muted-foreground ml-auto">{emotionHistory.length} readings</span>
            </div>
            {emotionHistory.length === 0 ? (
              <div className="h-48 flex items-center justify-center text-sm text-muted-foreground border border-dashed border-border/30 rounded-lg">
                Waiting for emotion data…
              </div>
            ) : (
              <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
                {emotionHistory.map((e, i) => (
                  <div key={i} className="flex items-center gap-3 text-xs py-1.5 border-b border-border/10 last:border-0">
                    <span className="font-mono text-muted-foreground w-16 shrink-0">{e.ts}</span>
                    <EmotionDot emotion={e.emotion} />
                    <span className="capitalize font-medium w-16">{e.emotion}</span>
                    <div className="flex-1 flex gap-2 text-[10px] text-muted-foreground">
                      <span>V:{(e.valence * 100).toFixed(0)}%</span>
                      <span>A:{(e.arousal * 100).toFixed(0)}%</span>
                    </div>
                    <span className="text-[10px] text-muted-foreground shrink-0">{(e.confidence * 100).toFixed(0)}% conf</span>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      )}

      {/* Band Powers Bar */}
      {isStreaming && analysis?.band_powers && (
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-semibold mb-4">Band Powers</h3>
          <div className="space-y-3">
            {Object.entries(analysis.band_powers).map(([band, value]) => (
              <div key={band} className="flex items-center gap-3">
                <span className="text-xs font-mono w-16 text-muted-foreground capitalize">{band}</span>
                <div className="flex-1">
                  <div className="h-2 rounded-full overflow-hidden" style={{ background: "hsl(220,22%,12%)" }}>
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{
                        width: `${Math.min(100, (value as number) * 100)}%`,
                        background: bandColor(band),
                      }}
                    />
                  </div>
                </div>
                <span className="text-xs font-mono w-12 text-right">{((value as number) * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Spectrogram Panel */}
      {wavelet && (
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Wavelet Spectrogram</h3>
            <div className="flex items-center gap-4 text-xs text-foreground/50">
              {wavelet.events.sleep_spindles.length > 0 && (
                <span className="text-cyan-400">
                  {wavelet.events.sleep_spindles.length} spindle{wavelet.events.sleep_spindles.length !== 1 ? "s" : ""}
                </span>
              )}
              {wavelet.events.k_complexes.length > 0 && (
                <span className="text-fuchsia-400">
                  {wavelet.events.k_complexes.length} K-complex{wavelet.events.k_complexes.length !== 1 ? "es" : ""}
                </span>
              )}
              <span>
                DWT: δ {((wavelet.dwt_energies.delta || 0) * 100).toFixed(0)}% |
                θ {((wavelet.dwt_energies.theta || 0) * 100).toFixed(0)}% |
                α {((wavelet.dwt_energies.alpha || 0) * 100).toFixed(0)}%
              </span>
            </div>
          </div>
          <SpectrogramChart
            coefficients={wavelet.spectrogram.coefficients}
            frequencies={wavelet.spectrogram.frequencies}
            times={wavelet.spectrogram.times}
            events={wavelet.events}
          />
        </Card>
      )}

      {/* Electrode Status Grid */}
      <Card className="glass-card p-6 rounded-xl hover-glow">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold">Electrode Status Grid</h3>
          <span className="text-sm text-foreground/70 font-mono">
            {isStreaming
              ? deviceStatus?.n_channels
                ? `${deviceStatus.n_channels} Channels`
                : "4 Channels"
              : "No Device"}
          </span>
        </div>
        {isStreaming ? (
          <>
            <div className="grid grid-cols-8 gap-2 mb-4">
              {electrodeStatuses.map((electrode, i) => (
                <div
                  key={i}
                  className={`w-8 h-8 rounded border flex items-center justify-center text-xs font-mono ${electrode.statusClass}`}
                >
                  {electrode.label}
                </div>
              ))}
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-success">{activeCount} Active</span>
              <span className="text-warning">{weakCount} Weak</span>
              <span className="text-destructive">{errorCount} Error</span>
            </div>
          </>
        ) : (
          <div className="h-24 flex items-center justify-center text-sm text-muted-foreground border border-dashed border-border/30 rounded-lg">
            Connect device to see electrode status
          </div>
        )}
      </Card>
    </main>
  );
}

/* ── Today Card with yesterday comparison ──────────────────────────── */

function TodayCard({
  label,
  value,
  yesterday,
  color,
}: {
  label: string;
  value: number | null;
  yesterday?: number | null;
  color: string;
}) {
  const pct = value != null ? Math.round(value * 100) : null;
  const yPct = yesterday != null ? Math.round(yesterday * 100) : null;
  const diff = pct != null && yPct != null ? pct - yPct : null;

  return (
    <Card className="glass-card p-4 rounded-xl">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{pct != null ? `${pct}%` : "—"}</p>
      {diff != null && (
        <div className={`flex items-center gap-1 text-[10px] mt-1 ${diff > 0 ? "text-rose-400" : diff < 0 ? "text-green-400" : "text-muted-foreground"}`}>
          {diff > 0 ? <TrendingUp className="h-3 w-3" /> : diff < 0 ? <TrendingDown className="h-3 w-3" /> : <Minus className="h-3 w-3" />}
          <span>{diff > 0 ? "+" : ""}{diff}% vs yesterday</span>
        </div>
      )}
    </Card>
  );
}

/* ── Helper: Model Card Component ────────────────────────────────── */

function ModelCard({
  icon,
  title,
  value,
  score,
  color,
  extra,
}: {
  icon: React.ReactNode;
  title: string;
  value: string;
  score?: number;
  color: string;
  extra?: React.ReactNode;
}) {
  const pct = score != null ? Math.round(score * 100) : null;

  return (
    <Card className="glass-card p-4 rounded-xl hover-glow">
      <div className="flex items-center gap-2 mb-2">
        <span className={color}>{icon}</span>
        <span className="text-xs text-muted-foreground font-medium">{title}</span>
      </div>
      <p className={`text-lg font-semibold capitalize ${color}`}>{value}</p>
      {pct != null && (
        <div className="mt-2">
          <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-1">
            <span>Confidence</span>
            <span>{pct}%</span>
          </div>
          <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "hsl(220,22%,12%)" }}>
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${pct}%`,
                background: `linear-gradient(90deg, hsl(152,60%,48%), hsl(38,85%,58%))`,
              }}
            />
          </div>
        </div>
      )}
      {extra}
    </Card>
  );
}

/* ── Valence/Arousal 2D Circumplex ───────────────────────────────── */

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

// Standard positions of each emotion on the Russell circumplex
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

function ValenceArousalPlot({
  valence,
  arousal,
  emotion = "neutral",
  history,
}: {
  valence: number;   // -1 to +1
  arousal: number;   // 0 to +1
  emotion?: string;
  history: Array<{ valence: number; arousal: number; emotion: string }>;
}) {
  const SIZE = 300;
  const CX = SIZE / 2;
  const CY = SIZE / 2;
  const R  = SIZE / 2 - 32;

  // Map (valence -1..1, arousal 0..1) → SVG pixel coords
  const toXY = (v: number, a: number) => ({
    x: CX + v * R,
    y: CY - (a * 2 - 1) * R,
  });

  const dot      = toXY(valence, arousal);
  const dotColor = EMOTION_COLORS[emotion] ?? EMOTION_COLORS.neutral;
  // history[0] = most recent — reverse for polyline (oldest→newest)
  const trail    = [...history].reverse();

  return (
    <svg width="100%" viewBox={`0 0 ${SIZE} ${SIZE}`} style={{ display: "block" }}>
      <defs>
        {/* Clip everything to the circle */}
        <clipPath id="va-clip">
          <circle cx={CX} cy={CY} r={R} />
        </clipPath>

        {/* Glow for current dot */}
        <filter id="va-glow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        {/* Radial vignette for depth */}
        <radialGradient id="va-vignette" cx="50%" cy="50%" r="50%">
          <stop offset="60%" stopColor="transparent" stopOpacity="0" />
          <stop offset="100%" stopColor="hsl(220,22%,4%)" stopOpacity="0.55" />
        </radialGradient>
      </defs>

      {/* ── Base circle ── */}
      <circle cx={CX} cy={CY} r={R} fill="hsl(220,24%,6%)" />

      {/* ── Quadrant colour fills (clipped) ── */}
      <g clipPath="url(#va-clip)" opacity="1">
        {/* Top-right  → Happy / Excited  → warm yellow */}
        <rect x={CX}     y={CY - R} width={R} height={R} fill="hsl(42,90%,54%)"  opacity={0.13} />
        {/* Top-left   → Angry / Fearful  → red */}
        <rect x={CX - R} y={CY - R} width={R} height={R} fill="hsl(0,80%,54%)"   opacity={0.12} />
        {/* Bottom-right → Relaxed / Calm → green */}
        <rect x={CX}     y={CY}     width={R} height={R} fill="hsl(152,68%,44%)" opacity={0.12} />
        {/* Bottom-left  → Sad / Bored    → blue */}
        <rect x={CX - R} y={CY}     width={R} height={R} fill="hsl(215,70%,54%)" opacity={0.10} />
      </g>

      {/* ── Concentric rings ── */}
      <circle cx={CX} cy={CY} r={R * 0.33} fill="none" stroke="hsl(220,20%,16%)" strokeWidth={0.7} strokeDasharray="3 5" />
      <circle cx={CX} cy={CY} r={R * 0.67} fill="none" stroke="hsl(220,20%,16%)" strokeWidth={0.7} strokeDasharray="3 5" />
      <circle cx={CX} cy={CY} r={R}         fill="none" stroke="hsl(220,20%,20%)" strokeWidth={1.4} />

      {/* ── Vignette overlay ── */}
      <circle cx={CX} cy={CY} r={R} fill="url(#va-vignette)" />

      {/* ── Axis lines ── */}
      <line x1={CX - R + 2} y1={CY} x2={CX + R - 2} y2={CY} stroke="hsl(220,20%,24%)" strokeWidth={0.9} />
      <line x1={CX} y1={CY + R - 2} x2={CX} y2={CY - R + 2} stroke="hsl(220,20%,24%)" strokeWidth={0.9} />

      {/* ── Axis arrow tips ── */}
      <polygon points={`${CX+R-1},${CY} ${CX+R-7},${CY-3} ${CX+R-7},${CY+3}`}  fill="hsl(220,20%,30%)" />
      <polygon points={`${CX},${CY-R+1} ${CX-3},${CY-R+7} ${CX+3},${CY-R+7}`} fill="hsl(220,20%,30%)" />

      {/* ── Axis labels ── */}
      <text x={CX + R - 10} y={CY - 7}  fontSize={7.5} fill="hsl(220,15%,36%)" textAnchor="end"    fontFamily="monospace">+VALENCE</text>
      <text x={CX - R + 10} y={CY - 7}  fontSize={7.5} fill="hsl(220,15%,36%)" textAnchor="start"  fontFamily="monospace">−VALENCE</text>
      <text x={CX + 5}      y={CY-R+14} fontSize={7.5} fill="hsl(220,15%,36%)" textAnchor="start"  fontFamily="monospace">HIGH AROUSAL</text>
      <text x={CX + 5}      y={CY+R-6}  fontSize={7.5} fill="hsl(220,15%,36%)" textAnchor="start"  fontFamily="monospace">LOW AROUSAL</text>

      {/* ── Quadrant corner labels ── */}
      {/* Top-right */}
      <text x={CX+R*0.60} y={CY-R*0.80} fontSize={9.5} fill="hsl(42,88%,60%)"  textAnchor="middle" fontWeight="700" letterSpacing="0.6">HAPPY</text>
      <text x={CX+R*0.60} y={CY-R*0.66} fontSize={7}   fill="hsl(42,70%,44%)"  textAnchor="middle">Excited · Joyful</text>
      {/* Top-left */}
      <text x={CX-R*0.60} y={CY-R*0.80} fontSize={9.5} fill="hsl(0,80%,58%)"   textAnchor="middle" fontWeight="700" letterSpacing="0.6">STRESSED</text>
      <text x={CX-R*0.60} y={CY-R*0.66} fontSize={7}   fill="hsl(0,65%,44%)"   textAnchor="middle">Angry · Fearful</text>
      {/* Bottom-right */}
      <text x={CX+R*0.60} y={CY+R*0.68} fontSize={9.5} fill="hsl(152,65%,50%)" textAnchor="middle" fontWeight="700" letterSpacing="0.6">RELAXED</text>
      <text x={CX+R*0.60} y={CY+R*0.82} fontSize={7}   fill="hsl(152,52%,38%)" textAnchor="middle">Calm · Serene</text>
      {/* Bottom-left */}
      <text x={CX-R*0.60} y={CY+R*0.68} fontSize={9.5} fill="hsl(210,70%,58%)" textAnchor="middle" fontWeight="700" letterSpacing="0.6">SAD</text>
      <text x={CX-R*0.60} y={CY+R*0.82} fontSize={7}   fill="hsl(210,55%,44%)" textAnchor="middle">Depressed · Bored</text>
      {/* Center-right: Focused */}
      <text x={CX+R*0.24} y={CY-R*0.28} fontSize={7.5} fill="hsl(200,70%,52%)" textAnchor="middle" opacity={0.75}>Focused</text>

      {/* ── Emotion anchor dots (standard circumplex positions) ── */}
      {EMOTION_ANCHORS.map(({ name, v, a, color }) => {
        const p = toXY(v, a);
        return (
          <g key={name}>
            <circle cx={p.x} cy={p.y} r={3.5} fill={color} opacity={0.28} />
            <circle cx={p.x} cy={p.y} r={2}   fill={color} opacity={0.55} />
          </g>
        );
      })}

      {/* ── History trail polyline (oldest→newest) ── */}
      {trail.length > 1 && (
        <polyline
          clipPath="url(#va-clip)"
          points={trail.map(h => { const p = toXY(h.valence, h.arousal); return `${p.x},${p.y}`; }).join(" ")}
          fill="none"
          stroke={dotColor}
          strokeWidth={1.5}
          strokeOpacity={0.28}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      )}

      {/* ── History dots (newest = history[0] = most visible) ── */}
      {history.map((h, i) => {
        const p     = toXY(h.valence, h.arousal);
        const ratio = 1 - i / Math.max(history.length - 1, 1); // 1=newest, 0=oldest
        const col   = EMOTION_COLORS[h.emotion] ?? EMOTION_COLORS.neutral;
        return (
          <circle
            key={i}
            cx={p.x} cy={p.y}
            r={1.8 + ratio * 2.2}
            fill={col}
            opacity={0.12 + ratio * 0.50}
            clipPath="url(#va-clip)"
          />
        );
      })}

      {/* ── Current position ── */}
      {/* Outer pulsing ring */}
      <circle cx={dot.x} cy={dot.y} r={12} fill="none" stroke={dotColor} strokeWidth={1.5} opacity={0.0}>
        <animate attributeName="r"       values="10;20;10" dur="2.6s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.35;0;0.35" dur="2.6s" repeatCount="indefinite" />
      </circle>
      {/* Soft glow halo */}
      <circle cx={dot.x} cy={dot.y} r={9}  fill={dotColor} opacity={0.18} />
      {/* Main dot with glow filter */}
      <circle cx={dot.x} cy={dot.y} r={6}  fill={dotColor} opacity={0.9} filter="url(#va-glow)" />
      <circle cx={dot.x} cy={dot.y} r={6}  fill={dotColor} />
      {/* White center pinpoint */}
      <circle cx={dot.x} cy={dot.y} r={2}  fill="white" opacity={0.92} />
    </svg>
  );
}

/* ── Emotion Dot Color ───────────────────────────────────────────── */

function EmotionDot({ emotion }: { emotion: string }) {
  const colors: Record<string, string> = {
    happy:   "hsl(38,85%,58%)",
    sad:     "hsl(210,60%,55%)",
    angry:   "hsl(340,70%,55%)",
    fearful: "hsl(262,45%,65%)",
    relaxed: "hsl(152,60%,48%)",
    focused: "hsl(200,70%,55%)",
    neutral: "hsl(220,15%,50%)",
  };
  return (
    <span
      className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
      style={{ background: colors[emotion] ?? colors.neutral }}
    />
  );
}

/* ── Helper: band color ──────────────────────────────────────────── */

function bandColor(band: string): string {
  const colors: Record<string, string> = {
    delta: "hsl(262,45%,65%)",
    theta: "hsl(200,70%,55%)",
    alpha: "hsl(152,60%,48%)",
    beta: "hsl(38,85%,58%)",
    gamma: "hsl(340,70%,55%)",
  };
  return colors[band] || "hsl(152,60%,48%)";
}
