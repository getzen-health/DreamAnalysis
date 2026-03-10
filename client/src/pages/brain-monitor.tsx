import React, { useMemo, useState, useEffect, useRef } from "react";
import { EEGWaveformCanvas } from "@/components/charts/eeg-waveform-canvas";
import { SpectrogramChart } from "@/components/charts/spectrogram-chart";
import { SignalQualityBadge } from "@/components/signal-quality-badge";
import { AlertBanner, type AlertLevel } from "@/components/alert-banner";
import { SessionControls } from "@/components/session-controls";
import { SimulationModeBanner } from "@/components/simulation-mode-banner";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Radio,
  Brain,
  Zap,
} from "lucide-react";
import { useInference } from "@/hooks/use-inference";
import { useDevice } from "@/hooks/use-device";
import { useVoiceEmotion } from "@/hooks/use-voice-emotion";
import {
  analyzeWavelet,
  type WaveletResult,
  type AnomalyResult,
} from "@/lib/ml-api";
import { Link } from "wouter";
import { Music } from "lucide-react";

export default function BrainMonitor() {
  const { isLocal, latencyMs, isReady } = useInference();
  const device = useDevice();
  const { state: deviceState, latestFrame, deviceStatus, selectedDevice, reconnectCount, epochReady } = device;
  const isStreaming = deviceState === "streaming";
  const voiceEmotion = useVoiceEmotion();

  const [wavelet, setWavelet] = useState<WaveletResult | null>(null);
  const anomaly = (latestFrame?.analysis as { anomaly?: AnomalyResult } | undefined)?.anomaly ?? null;
  const [isRecording, setIsRecording] = useState(false);
  const waveletTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const analysis = latestFrame?.analysis;

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
        // Normalize to 0-100 regardless of source scale (backend uses 0-100, BLE used to use 0-1)
        channel_quality: ((rawQuality.channel_quality as number[]) ?? []).map((q) =>
          q <= 1.0 ? Math.round(q * 100) : q
        ),
      }
    : null;

  // Simple amplitude-threshold signal quality fields from /analyze-eeg
  const sqScore: number = (latestFrame as Record<string, unknown> | null)?.signal_quality_score as number ?? 100;
  const artifactDetected: boolean = (latestFrame as Record<string, unknown> | null)?.artifact_detected as boolean ?? false;
  const artifactType: "clean" | "blink" | "muscle" | "electrode_pop" =
    ((latestFrame as Record<string, unknown> | null)?.artifact_type as "clean" | "blink" | "muscle" | "electrode_pop") ?? "clean";

  // Badge label and color based on 0-100 quality score
  const sqLabel =
    sqScore > 70
      ? "Good signal"
      : sqScore >= 40
        ? "Fair signal — reduce movement"
        : "Poor signal — check headset & minimize jaw tension";
  const sqBadgeColor =
    sqScore > 70
      ? "text-success border-success/30 bg-success/10"
      : sqScore >= 40
        ? "text-warning border-warning/30 bg-warning/10"
        : "text-destructive border-destructive/30 bg-destructive/10";

  const alphaHz = analysis?.band_powers?.alpha
    ? `${(analysis.band_powers.alpha * 12).toFixed(1)} Hz`
    : "—";
  const betaHz = analysis?.band_powers?.beta
    ? `${(analysis.band_powers.beta * 30).toFixed(1)} Hz`
    : "—";

  const sourceLabel = isStreaming ? "LIVE" : "OFFLINE";
  const sourceColor = isStreaming ? "text-primary" : "text-muted-foreground";
  const alertLevel: AlertLevel = anomaly?.alert_level || "normal";

  // Electrode grid — suppress quality data for synthetic board (non-physiological signals)
  const isSynthetic = deviceStatus?.device_type === "synthetic" || selectedDevice === "synthetic";
  const channelQuality = isSynthetic ? [] : (signalQuality?.channel_quality || []);
  const hasRealData = channelQuality.length > 0;
  const activeCount = hasRealData ? channelQuality.filter((q) => q >= 80).length : 0;
  const weakCount = hasRealData ? channelQuality.filter((q) => q >= 60 && q < 80).length : 0;
  const errorCount = hasRealData ? channelQuality.filter((q) => q < 60).length : 0;

  const electrodeStatuses = useMemo(() => {
    const qualityData = isSynthetic ? undefined : signalQuality?.channel_quality;
    const nChannels = deviceStatus?.n_channels ?? (qualityData?.length || 4);
    return Array.from({ length: nChannels }, (_, i) => {
      let statusClass: string;
      if (qualityData && i < qualityData.length) {
        const q = qualityData[i];
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
  }, [isSynthetic, signalQuality?.channel_quality, deviceStatus?.n_channels]);

  const lf = latestFrame;
  const a = lf?.analysis as Record<string, unknown> | undefined;

  // Placeholder shown while the 4-second epoch buffer is filling
  const COLLECTING = isStreaming && !epochReady ? "Collecting…" : "—";

  const modelOutputs: { name: string; value: string }[] = [
    {
      name: "Emotion",
      value: epochReady
        ? ((a?.emotions as { emotion?: string } | undefined)?.emotion ?? "—")
        : COLLECTING,
    },
    {
      name: "Stress",
      value: epochReady && lf
        ? `${Math.round(((a?.emotions as { stress_index?: number } | undefined)?.stress_index ?? 0) * 100)}%`
        : COLLECTING,
    },
    {
      name: "Focus",
      value: epochReady && lf
        ? `${Math.round(((a?.emotions as { focus_index?: number } | undefined)?.focus_index ?? 0) * 100)}%`
        : COLLECTING,
    },
    {
      name: "Sleep",
      value: (a?.sleep_staging as { stage?: string } | undefined)?.stage ?? "—",
    },
    {
      name: "Creativity",
      value: lf
        ? `${Math.round(((a?.creativity as { creativity_score?: number } | undefined)?.creativity_score ?? 0) * 100)}%`
        : "—",
    },
    {
      name: "Flow",
      value: lf
        ? `${Math.round(((a?.flow_state as { flow_score?: number } | undefined)?.flow_score ?? 0) * 100)}%`
        : "—",
    },
    {
      name: "Drowsiness",
      value: lf
        ? `${Math.round(((a?.drowsiness as { drowsiness_index?: number } | undefined)?.drowsiness_index ?? 0) * 100)}%`
        : "—",
    },
    {
      name: "Cog. Load",
      value: (a?.cognitive_load as { level?: string } | undefined)?.level ?? "—",
    },
    {
      name: "Attention",
      value: lf
        ? `${Math.round(((a?.attention as { attention_score?: number } | undefined)?.attention_score ?? 0) * 100)}%`
        : "—",
    },
    {
      name: "Dream",
      value: lf
        ? ((a?.dream_detection as { is_dreaming?: boolean } | undefined)?.is_dreaming ? "Yes" : "No")
        : "—",
    },
    {
      name: "Lucid",
      value: lf
        ? `${Math.round(((a?.lucid_dream as { lucidity_score?: number } | undefined)?.lucidity_score ?? 0) * 100)}%`
        : "—",
    },
    {
      name: "Meditation",
      value: (a?.meditation as { depth?: string } | undefined)?.depth ?? "—",
    },
    {
      name: "Memory",
      value: lf
        ? `${Math.round(((a?.memory_encoding as { encoding_score?: number } | undefined)?.encoding_score ?? 0) * 100)}%`
        : "—",
    },
    {
      name: "Artifact",
      value: lf
        ? ((a?.artifact as { artifact_detected?: boolean } | undefined)?.artifact_detected ? "Yes" : "No")
        : "—",
    },
    {
      name: "Denoising",
      value: (() => {
        const snr = (a?.denoising as { snr_improvement?: number } | undefined)?.snr_improvement;
        return snr !== undefined ? `${snr.toFixed(1)}dB` : "—";
      })(),
    },
    {
      name: "Online Lrn",
      value: lf
        ? `${Math.round(((a?.online_learner as { adaptation_rate?: number } | undefined)?.adaptation_rate ?? 0) * 100)}%`
        : "—",
    },
  ];

  return (
    <main className="p-4 md:p-6 space-y-6">
      {isStreaming && reconnectCount > 0 && (
        <div className="rounded-md bg-amber-500/10 border border-amber-500/40 px-4 py-2 text-sm font-medium text-amber-400">
          Reconnecting to EEG stream… (attempt {reconnectCount})
        </div>
      )}
      <SimulationModeBanner />

      {/* Connection Banner */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          EEG is offline. You can still use voice + health features elsewhere; connect Muse only for live brain data here.
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

      {/* Music Session */}
      <Link href="/biofeedback?tab=music">
        <div className="glass-card rounded-xl p-4 hover-glow flex items-center gap-3 cursor-pointer group">
          <div className="h-9 w-9 rounded-full bg-violet-500/20 flex items-center justify-center shrink-0 group-hover:bg-violet-500/30 transition-colors">
            <Music className="h-4 w-4 text-violet-400" />
          </div>
          <div>
            <p className="text-sm font-medium">Music Session</p>
            <p className="text-xs text-muted-foreground">Open full binaural + focus music session</p>
          </div>
          <span className="ml-auto text-xs text-violet-400 group-hover:translate-x-0.5 transition-transform">→</span>
        </div>
      </Link>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* EEG Brain Waves */}
        <div className="xl:col-span-2 glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold">
              EEG Brain Wave Activity
            </h3>
            <div className="flex items-center space-x-3">
              {isStreaming && <SessionControls onRecordingChange={setIsRecording} />}
              {isStreaming && (
                <span
                  className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-xs font-medium ${sqBadgeColor}`}
                  title={artifactDetected ? artifactType : sqLabel}
                >
                  {sqLabel}
                </span>
              )}
              {signalQuality && (
                <SignalQualityBadge
                  sqi={signalQuality.sqi}
                  artifacts={signalQuality.artifacts_detected}
                  artifactDetected={artifactDetected}
                  artifactType={artifactType}
                  compact
                />
              )}
              {/* Signal source badge */}
              {(() => {
                const source: string =
                  (a?.emotions as { signal_source?: string } | undefined)?.signal_source ??
                  (isStreaming ? "eeg" : voiceEmotion.lastResult ? "voice" : "health");

                const config: Record<string, { label: string; className: string }> = {
                  "eeg":       { label: "EEG",          className: "bg-purple-500/20 text-purple-400 border-purple-500/30" },
                  "voice":     { label: "Voice",         className: "bg-blue-500/20 text-blue-400 border-blue-500/30" },
                  "health":    { label: "Health Est.",   className: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30" },
                  "eeg+voice": { label: "EEG + Voice",   className: "bg-green-500/20 text-green-400 border-green-500/30" },
                  "eeg+bio":   { label: "EEG + Bio",     className: "bg-green-500/20 text-green-400 border-green-500/30" },
                };
                const cfg = config[source] ?? config["health"];
                return (
                  <Badge className={`text-xs border ${cfg.className}`}>
                    {cfg.label}
                  </Badge>
                );
              })()}
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
            <div className="h-64 flex flex-col items-center justify-center gap-3 border border-dashed border-border/30 rounded-lg p-4">
              <Radio className="h-6 w-6 text-muted-foreground/50" />
              <p className="text-sm text-muted-foreground text-center">
                No EEG signal — showing voice + health estimates
              </p>
              {voiceEmotion.lastResult && (
                <div className="flex items-center gap-2 text-xs">
                  <span className="capitalize font-medium text-foreground/80">{voiceEmotion.lastResult.emotion}</span>
                  <span className="text-muted-foreground">
                    valence {voiceEmotion.lastResult.valence >= 0 ? "+" : ""}{voiceEmotion.lastResult.valence.toFixed(2)}
                  </span>
                  <span className="text-muted-foreground">·</span>
                  <span className="text-muted-foreground">{Math.round(voiceEmotion.lastResult.confidence * 100)}% conf</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Brain State Now */}
        <div className="glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Brain State Now</h3>
            <Brain className="text-primary" />
          </div>
          {isStreaming && stableAnalysis ? (() => {
            const narrative = getBrainStateNarrative(stableAnalysis);
            return (
              <div className="space-y-4">
                <div>
                  <p className={`text-xl font-bold ${narrative.color}`}>{narrative.headline}</p>
                  <p className="text-sm text-foreground/70 mt-2 leading-relaxed">{narrative.story}</p>
                </div>
                {/* 4 key stats */}
                <div className="grid grid-cols-2 gap-2 pt-3 border-t border-border/30">
                  {[
                    { label: "Sleep", value: stableAnalysis.sleep_staging?.stage ?? "Wake", sub: `${Math.round((stableAnalysis.sleep_staging?.confidence ?? 0) * 100)}% conf` },
                    { label: "Emotion", value: epochReady ? (stableAnalysis.emotions?.emotion ?? "—") : "Collecting signal…", sub: epochReady ? `V:${((stableAnalysis.emotions?.valence ?? 0) * 100).toFixed(0)}%` : "4s needed" },
                    { label: "Stress", value: stableAnalysis.stress?.level ?? "—", sub: `${Math.round((stableAnalysis.stress?.stress_index ?? 0) * 100)}%` },
                    { label: "Flow", value: stableAnalysis.flow_state?.in_flow ? "In Flow" : "Normal", sub: `${Math.round((stableAnalysis.flow_state?.flow_score ?? 0) * 100)}%` },
                  ].map(({ label, value, sub }) => (
                    <div key={label} className="bg-muted/20 rounded-lg p-3">
                      <p className="text-[10px] text-muted-foreground">{label}</p>
                      <p className="text-sm font-semibold capitalize">{value}</p>
                      <p className="text-[10px] text-muted-foreground font-mono">{sub}</p>
                    </div>
                  ))}
                </div>
                {/* Rest of models — compact 3-col grid */}
                <div className="grid grid-cols-3 gap-x-4 gap-y-2 text-[11px]">
                  {[
                    { label: "Attention", value: stableAnalysis.attention?.state },
                    { label: "Creativity", value: stableAnalysis.creativity?.state },
                    { label: "Memory", value: stableAnalysis.memory_encoding?.state },
                    { label: "Meditation", value: stableAnalysis.meditation?.depth },
                    { label: "Dream", value: stableAnalysis.dream_detection?.is_dreaming ? "Dreaming" : "Awake" },
                    { label: "Drowsy", value: stableAnalysis.drowsiness?.state },
                  ].map(({ label, value }) => (
                    <div key={label} className="flex flex-col gap-0.5">
                      <span className="text-muted-foreground text-[10px]">{label}</span>
                      <span className="font-medium capitalize text-foreground/80">{value ?? "—"}</span>
                    </div>
                  ))}
                </div>
                {/* Cognitive load + emotion probabilities */}
                {stableAnalysis.cognitive_load && (
                  <div className="text-[11px] text-muted-foreground pt-2 border-t border-border/20">
                    Cognitive load: <span className="font-medium text-foreground/80 capitalize">{stableAnalysis.cognitive_load.level}</span>
                    {epochReady && stableAnalysis.emotions?.probabilities ? (
                      <div className="mt-2 space-y-1">
                        {Object.entries(stableAnalysis.emotions.probabilities as Record<string, number>)
                          .sort((a, b) => b[1] - a[1])
                          .slice(0, 3)
                          .map(([emo, prob]) => (
                            <div key={emo} className="flex items-center gap-2">
                              <span className="w-14 capitalize">{emo}</span>
                              <div className="flex-1 h-1 rounded-full overflow-hidden bg-muted/40">
                                <div className="h-full rounded-full" style={{ width: `${Math.round(prob * 100)}%`, background: "hsl(340,70%,55%)" }} />
                              </div>
                              <span className="w-6 text-right font-mono">{Math.round(prob * 100)}%</span>
                            </div>
                          ))}
                      </div>
                    ) : !epochReady ? (
                      <p className="mt-2 text-muted-foreground/60 italic">Collecting signal…</p>
                    ) : null}
                  </div>
                )}
              </div>
            );
          })() : (
            <div className="h-80 flex flex-col items-center justify-center gap-3 border border-dashed border-border/30 rounded-lg p-4">
              <Brain className="h-6 w-6 text-muted-foreground/50" />
              <p className="text-sm text-muted-foreground text-center">
                No EEG signal — showing voice + health estimates
              </p>
              {voiceEmotion.lastResult ? (
                <div className="space-y-2 w-full max-w-[200px]">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Emotion</span>
                    <span className="capitalize font-medium">{voiceEmotion.lastResult.emotion}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Valence</span>
                    <span className="font-mono">
                      {voiceEmotion.lastResult.valence >= 0 ? "+" : ""}
                      {voiceEmotion.lastResult.valence.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Arousal</span>
                    <span className="font-mono">{voiceEmotion.lastResult.arousal.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Confidence</span>
                    <span className="font-mono">{Math.round(voiceEmotion.lastResult.confidence * 100)}%</span>
                  </div>
                  <p className="text-[10px] text-muted-foreground/60 text-center pt-1">
                    via {voiceEmotion.lastResult.model_type}
                  </p>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground/60 text-center">
                  Use voice analysis on the dashboard for estimates without EEG.
                </p>
              )}
            </div>
          )}
        </div>
      </div>


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

      {/* All 16 Models Grid */}
      <Card className="glass-card rounded-xl hover-glow">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">16 ML Models</CardTitle>
          <CardDescription className="text-xs">
            {lf ? "Live" : "Offline — connect device for live values"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-2">
            {modelOutputs.map(({ name, value }) => (
              <div key={name} className="rounded border border-border/30 p-2 text-center">
                <p className="text-xs text-muted-foreground truncate">{name}</p>
                <p className="text-sm font-mono font-semibold mt-0.5 truncate">{value}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

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
              {hasRealData ? (
                <>
                  <span className="text-success">{activeCount} Active</span>
                  <span className="text-warning">{weakCount} Weak</span>
                  <span className="text-destructive">{errorCount} Error</span>
                </>
              ) : (
                <span className="text-muted-foreground">
                  {isSynthetic ? "Simulated data — electrode quality n/a" : "Quality data not available for this device"}
                </span>
              )}
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

/* ── Helper: Brain State Narrative ───────────────────────────────── */

function getBrainStateNarrative(analysis: Record<string, unknown>): { headline: string; story: string; color: string } {
  const stress   = (analysis.stress   as { stress_index?: number })?.stress_index   ?? 0;
  const flow     = (analysis.flow_state as { flow_score?: number })?.flow_score     ?? 0;
  const sleep    = (analysis.sleep_staging as { stage?: string })?.stage             ?? "wake";
  const meditation = (analysis.meditation as { meditation_score?: number })?.meditation_score ?? 0;
  const creativity = (analysis.creativity  as { creativity_score?: number })?.creativity_score ?? 0;
  const attention  = (analysis.attention   as { attention_score?: number })?.attention_score   ?? 0;
  const drowsiness = (analysis.drowsiness  as { drowsiness_index?: number })?.drowsiness_index ?? 0;

  if (sleep && sleep !== "wake" && sleep !== "Wake" && sleep !== "—") {
    return { headline: `${sleep} Sleep`, story: `Your brain is in ${sleep} sleep. Delta waves dominate, signaling deep restorative rest. Models continue running passively.`, color: "text-blue-400" };
  }
  if (flow > 0.7) {
    return { headline: "Flow State", story: "Alpha and theta are balanced with engaged beta — you're in the zone. Focus is sharp without anxiety. Don't break the session.", color: "text-green-400" };
  }
  if (stress > 0.7) {
    return { headline: "Elevated Stress", story: "High-beta activity is dominant, signaling mental tension. Try 4-7-8 breathing or step away briefly. Your nervous system needs a reset.", color: "text-rose-400" };
  }
  if (meditation > 0.7) {
    return { headline: "Meditative State", story: "Alpha waves are dominant and theta is rising — your mind is calm and inward. Excellent state for creative insight or visualization.", color: "text-violet-400" };
  }
  if (creativity > 0.7) {
    return { headline: "Creative Flow", story: "Theta and alpha are elevated — the hallmark of divergent thinking and creative incubation. Great time to brainstorm or explore new ideas.", color: "text-amber-400" };
  }
  if (drowsiness > 0.7) {
    return { headline: "Getting Drowsy", story: "Theta is climbing and alpha is slowing down — your brain is shifting toward sleep onset. Take a break or try a short walk.", color: "text-orange-400" };
  }
  if (attention > 0.7) {
    return { headline: "Sharp & Focused", story: "Beta activity is high with suppressed alpha — classic active cognitive engagement. Your brain is fully online and problem-solving.", color: "text-emerald-400" };
  }
  return { headline: "Balanced Resting", story: "A balanced mix of alpha and beta signals a calm yet alert baseline state — the optimal foundation for learning or light work.", color: "text-primary" };
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
