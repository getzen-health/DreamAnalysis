import React, { useMemo, useState, useEffect, useRef, useCallback } from "react";
import { EEGWaveformCanvas } from "@/components/charts/eeg-waveform-canvas";
import { AlertBanner, type AlertLevel } from "@/components/alert-banner";
import { SessionControls } from "@/components/session-controls";
import { SimulationModeBanner } from "@/components/simulation-mode-banner";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Radio,
  Brain,
  Zap,
  Play,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { useInference } from "@/hooks/use-inference";
import { useDevice } from "@/hooks/use-device";
import { useVoiceCache } from "@/hooks/use-voice-cache";
import {
  type AnomalyResult,
} from "@/lib/ml-api";
import { assessSignalQuality, type SignalQualityResult as SQResult } from "@/lib/signal-quality";
import { Link } from "wouter";
import { museBle } from "@/lib/muse-ble";
import { BrainAgeCard } from "@/components/brain-age-card";
import { EEGCoherenceCard } from "@/components/eeg-coherence-card";
import { ConfidenceMeter } from "@/components/confidence-meter";
import { InterventionSuggestion } from "@/components/intervention-suggestion";
import { EnsembleExplanation } from "@/components/ensemble-explanation";
import { emgDetector, type EMGDetectionResult } from "@/lib/emg-detector";
import { calculateEmotionConfidence } from "@/lib/confidence-calculator";
import { computeBlinkStats, type BlinkStats } from "@/lib/blink-detector";
import { detectBreathingState, type BreathingAnalysis } from "@/lib/breathing-detector";
import { BreathingIndicator } from "@/components/breathing-indicator";
import { Eye } from "lucide-react";
import { useInterventionTriggers } from "@/hooks/use-intervention-triggers";
import { InterventionTriggerToast } from "@/components/intervention-trigger-toast";
import type { TriggerState } from "@/lib/eeg-intervention-trigger";

// Route targets for ML model cards — null means no linked page
const MODEL_ROUTES: Record<string, string | null> = {
  "Emotion": "/brain-monitor",
  "Stress": "/stress",
  "Focus": "/focus",
  "Sleep": "/sleep",
  "Creativity": "/biofeedback",
  "Flow": "/biofeedback",
  "Drowsiness": "/sleep",
  "Cog. Load": "/focus",
  "Attention": "/neurofeedback",
  "Dream": "/dreams",
  "Lucid": "/dreams",
  "Meditation": "/biofeedback",
  "Memory": "/insights",
  "Artifact": null,
  "Denoising": null,
  "Online Lrn": null,
};

// All 5 standard EEG frequency bands with labels and ranges
const ALL_BANDS = [
  { key: "delta", label: "Delta", range: "0.5-4 Hz" },
  { key: "theta", label: "Theta", range: "4-8 Hz" },
  { key: "alpha", label: "Alpha", range: "8-12 Hz" },
  { key: "beta",  label: "Beta",  range: "12-30 Hz" },
  { key: "gamma", label: "Gamma", range: "30-100 Hz" },
] as const;

export default function BrainMonitor() {
  const { isLocal, latencyMs, isReady } = useInference();
  const device = useDevice();
  const { state: deviceState, latestFrame, deviceStatus, selectedDevice, reconnectCount, epochReady, bleReconnect } = device;
  const isStreaming = deviceState === "streaming";
  const { cachedEmotion: voiceResult } = useVoiceCache();

  const anomaly = (latestFrame?.analysis as { anomaly?: AnomalyResult } | undefined)?.anomaly ?? null;
  const [_isRecording, setIsRecording] = useState(false);

  const analysis = latestFrame?.analysis;

  // ---- Per-electrode signal quality (spectral flatness + amplitude) ----
  const isSynthetic = deviceStatus?.device_type === "synthetic" || selectedDevice === "synthetic";
  const [sqResult, setSqResult] = useState<SQResult | null>(null);
  const sqTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ---- EMG artifact detection ----
  const [emgResult, setEmgResult] = useState<EMGDetectionResult | null>(null);

  useEffect(() => {
    if (!isStreaming || !latestFrame?.signals || isSynthetic) {
      setEmgResult(null);
      return;
    }
    const result = emgDetector.detect(
      latestFrame.signals as number[][],
      latestFrame.sample_rate || 256,
    );
    setEmgResult(result);
  }, [isStreaming, isSynthetic, latestFrame?.timestamp]);

  // ---- Blink-rate detection (AF7 channel) ----
  const [blinkStats, setBlinkStats] = useState<BlinkStats | null>(null);

  useEffect(() => {
    if (!isStreaming || !latestFrame?.signals || isSynthetic) {
      setBlinkStats(null);
      return;
    }
    const signals = latestFrame.signals as number[][];
    if (signals.length < 2) return;
    const af7 = new Float32Array(signals[1]);
    const fs = latestFrame.sample_rate || 256;
    const analysisAny = latestFrame.analysis as Record<string, unknown> | undefined;
    const bp = analysisAny?.band_powers as Record<string, number> | undefined;
    const alphaPower = bp?.alpha ?? 0.2;
    const stats = computeBlinkStats(af7, fs, alphaPower);
    setBlinkStats(stats);
  }, [isStreaming, isSynthetic, latestFrame?.timestamp]);

  // ---- Breathing state detection (AF7 channel, buffered) ----
  const [breathingAnalysis, setBreathingAnalysis] = useState<BreathingAnalysis | null>(null);
  const breathingBufferRef = useRef<Float32Array>(new Float32Array(0));

  useEffect(() => {
    if (!isStreaming || !latestFrame?.signals || isSynthetic) {
      setBreathingAnalysis(null);
      breathingBufferRef.current = new Float32Array(0);
      return;
    }
    const signals = latestFrame.signals as number[][];
    if (signals.length < 2) return;
    const af7 = new Float32Array(signals[1]);
    const fs = latestFrame.sample_rate || 256;

    const maxSamples = fs * 30;
    const prev = breathingBufferRef.current;
    const combined = new Float32Array(prev.length + af7.length);
    combined.set(prev);
    combined.set(af7, prev.length);
    if (combined.length > maxSamples) {
      breathingBufferRef.current = combined.slice(combined.length - maxSamples);
    } else {
      breathingBufferRef.current = combined;
    }

    if (breathingBufferRef.current.length >= fs * 10) {
      const result = detectBreathingState(breathingBufferRef.current, fs);
      setBreathingAnalysis(result);
    }
  }, [isStreaming, isSynthetic, latestFrame?.timestamp]);

  const CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"];

  // ---- EEG intervention trigger engine (#504) ----
  const sessionStartRef = useRef<number>(Date.now());
  const stressDurationRef = useRef<number>(0);
  const highBetaDurationRef = useRef<number>(0);
  const lastStressCheckRef = useRef<number>(Date.now());

  useEffect(() => {
    if (isStreaming) {
      sessionStartRef.current = Date.now();
      stressDurationRef.current = 0;
      highBetaDurationRef.current = 0;
    }
  }, [isStreaming]);

  useEffect(() => {
    if (!isStreaming || !analysis) return;
    const now = Date.now();
    const dt = (now - lastStressCheckRef.current) / 1000;
    lastStressCheckRef.current = now;
    if (dt > 10) return;

    const emotions = analysis.emotions as { stress_index?: number } | undefined;
    const stressIdx = emotions?.stress_index ?? 0;
    const bp = (analysis as Record<string, unknown>).band_powers as Record<string, number> | undefined;
    const alphaLvl = bp?.alpha ?? 0.2;
    const betaLvl = bp?.beta ?? 0.15;

    if (stressIdx > 0.7) {
      stressDurationRef.current += dt;
    } else {
      stressDurationRef.current = Math.max(0, stressDurationRef.current - dt * 0.5);
    }

    if (alphaLvl < 0.05 && betaLvl > 0.3) {
      highBetaDurationRef.current += dt;
    } else {
      highBetaDurationRef.current = Math.max(0, highBetaDurationRef.current - dt * 0.5);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isStreaming, latestFrame?.timestamp]);

  const getTriggerState = useCallback((): TriggerState | null => {
    if (!analysis) return null;
    const emotions = analysis.emotions as { stress_index?: number } | undefined;
    const bp = (analysis as Record<string, unknown>).band_powers as Record<string, number> | undefined;
    return {
      stressIndex: emotions?.stress_index ?? 0,
      stressDurationSeconds: stressDurationRef.current,
      blinksPerMinute: blinkStats?.blinksPerMinute ?? 15,
      sessionMinutes: (Date.now() - sessionStartRef.current) / 60000,
      alphaLevel: bp?.alpha ?? 0.2,
      betaLevel: bp?.beta ?? 0.15,
      highBetaDurationSeconds: highBetaDurationRef.current,
    };
  }, [analysis, blinkStats]);

  const { activeTrigger, dismiss: dismissTrigger } = useInterventionTriggers(
    isStreaming,
    getTriggerState,
  );

  useEffect(() => {
    if (!isStreaming || !latestFrame?.signals || isSynthetic) {
      if (sqTimerRef.current) clearInterval(sqTimerRef.current);
      setSqResult(null);
      return;
    }

    sqTimerRef.current = setInterval(() => {
      if (latestFrame?.signals && latestFrame.signals.length >= 4) {
        const channels = latestFrame.signals.map((ch: number[]) => new Float32Array(ch));
        const result = assessSignalQuality(channels, latestFrame.sample_rate || 256, CHANNEL_NAMES);
        setSqResult(result);
      }
    }, 2000);

    if (latestFrame.signals.length >= 4) {
      const channels = latestFrame.signals.map((ch: number[]) => new Float32Array(ch));
      const result = assessSignalQuality(channels, latestFrame.sample_rate || 256, CHANNEL_NAMES);
      setSqResult(result);
    }

    return () => {
      if (sqTimerRef.current) clearInterval(sqTimerRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isStreaming, isSynthetic, latestFrame?.timestamp]);

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
      ? "Good Signal"
      : sqScore >= 40
        ? "Fair Signal"
        : "Poor Signal";
  const sqDotColor =
    sqScore > 70
      ? "bg-success"
      : sqScore >= 40
        ? "bg-warning"
        : "bg-destructive";
  const sqBadgeColor =
    sqScore > 70
      ? "text-success border-success/30 bg-success/10"
      : sqScore >= 40
        ? "text-warning border-warning/30 bg-warning/10"
        : "text-destructive border-destructive/30 bg-destructive/10";

  const bp = analysis?.band_powers;
  const alphaVal = bp?.alpha != null ? bp.alpha : null;

  const sourceLabel = isStreaming ? "LIVE" : "OFFLINE";
  const sourceColor = isStreaming ? "text-primary" : "text-muted-foreground";
  const alertLevel: AlertLevel = anomaly?.alert_level || "normal";

  const lf = latestFrame;
  const a = lf?.analysis as Record<string, unknown> | undefined;

  // Placeholder shown while the 4-second epoch buffer is filling
  const COLLECTING = isStreaming && !epochReady ? "Collecting..." : "\u2014";

  // ---- ML Models collapsed/expanded state ----
  const [modelsExpanded, setModelsExpanded] = useState(false);

  const modelOutputs: { name: string; value: string }[] = [
    {
      name: "Emotion",
      value: epochReady
        ? ((a?.emotions as { emotion?: string } | undefined)?.emotion ?? "\u2014")
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
      value: (a?.sleep_staging as { stage?: string } | undefined)?.stage ?? "\u2014",
    },
    {
      name: "Creativity",
      value: lf
        ? `${Math.round(((a?.creativity as { creativity_score?: number } | undefined)?.creativity_score ?? 0) * 100)}%`
        : "\u2014",
    },
    {
      name: "Flow",
      value: lf
        ? `${Math.round(((a?.flow_state as { flow_score?: number } | undefined)?.flow_score ?? 0) * 100)}%`
        : "\u2014",
    },
    {
      name: "Drowsiness",
      value: lf
        ? `${Math.round(((a?.drowsiness as { drowsiness_index?: number } | undefined)?.drowsiness_index ?? 0) * 100)}%`
        : "\u2014",
    },
    {
      name: "Cog. Load",
      value: (a?.cognitive_load as { level?: string } | undefined)?.level ?? "\u2014",
    },
    {
      name: "Attention",
      value: lf
        ? `${Math.round(((a?.attention as { attention_score?: number } | undefined)?.attention_score ?? 0) * 100)}%`
        : "\u2014",
    },
    {
      name: "Dream",
      value: lf
        ? ((a?.dream_detection as { is_dreaming?: boolean } | undefined)?.is_dreaming ? "Yes" : "No")
        : "\u2014",
    },
    {
      name: "Lucid",
      value: lf
        ? `${Math.round(((a?.lucid_dream as { lucidity_score?: number } | undefined)?.lucidity_score ?? 0) * 100)}%`
        : "\u2014",
    },
    {
      name: "Meditation",
      value: (a?.meditation as { depth?: string } | undefined)?.depth ?? "\u2014",
    },
    {
      name: "Memory",
      value: lf
        ? `${Math.round(((a?.memory_encoding as { encoding_score?: number } | undefined)?.encoding_score ?? 0) * 100)}%`
        : "\u2014",
    },
    {
      name: "Artifact",
      value: lf
        ? ((a?.artifact as { artifact_detected?: boolean } | undefined)?.artifact_detected ? "Yes" : "No")
        : "\u2014",
    },
    {
      name: "Denoising",
      value: (() => {
        const snr = (a?.denoising as { snr_improvement?: number } | undefined)?.snr_improvement;
        return snr !== undefined ? `${snr.toFixed(1)}dB` : "\u2014";
      })(),
    },
    {
      name: "Online Lrn",
      value: lf
        ? `${Math.round(((a?.online_learner as { adaptation_rate?: number } | undefined)?.adaptation_rate ?? 0) * 100)}%`
        : "\u2014",
    },
  ];

  // Key models shown in collapsed view (first 4)
  const keyModels = modelOutputs.slice(0, 4);
  const remainingModels = modelOutputs.slice(4);

  return (
    <main className="p-4 md:p-6 pb-24 space-y-6">
      {/* ---- Reconnection banners ---- */}
      {isStreaming && reconnectCount > 0 && (
        <div className="rounded-md bg-amber-500/10 border border-amber-500/40 px-4 py-2 text-sm font-medium text-amber-400">
          Reconnecting to EEG stream... (attempt {reconnectCount})
        </div>
      )}
      {bleReconnect.isReconnecting && (
        <div
          data-testid="ble-reconnect-banner"
          className="rounded-md bg-amber-500/10 border border-amber-500/40 px-4 py-2 text-sm font-medium text-amber-400 flex items-center gap-2"
        >
          <span className="inline-block h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
          Reconnecting to Muse headband... (attempt {bleReconnect.attempt + 1} of 5)
        </div>
      )}
      {bleReconnect.gaveUp && (
        <div
          data-testid="ble-reconnect-failed-banner"
          className="rounded-md bg-red-500/10 border border-red-500/40 px-4 py-2 text-sm font-medium text-red-400"
        >
          Muse connection lost after 5 attempts. Make sure your headband is powered on and nearby, then tap Connect.
        </div>
      )}
      <SimulationModeBanner />

      {/* ---- 1. Header with signal quality badge + LIVE indicator ---- */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-bold">Brain Monitor</h2>
          <div className="flex items-center gap-2">
            {isStreaming && (
              <Radio className="h-4 w-4 text-primary animate-pulse" />
            )}
            <span className={`text-sm font-mono ${sourceColor}`} aria-live="polite">
              {sourceLabel}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {isStreaming && <SessionControls onRecordingChange={setIsRecording} />}
          {isStreaming && (
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-medium ${sqBadgeColor}`}
              data-testid="signal-quality-status"
              title={artifactDetected ? artifactType : sqLabel}
            >
              <span className={`h-2 w-2 rounded-full ${sqDotColor} shrink-0`} />
              {sqLabel}
            </span>
          )}
          {/* Signal source badge */}
          {(() => {
            const source: string =
              (a?.emotions as { signal_source?: string } | undefined)?.signal_source ??
              (isStreaming ? "eeg" : voiceResult ? "voice" : "health");

            const config: Record<string, { label: string; className: string }> = {
              "eeg":       { label: "EEG",          className: "bg-emerald-500/20 text-emerald-500 border-emerald-500/30" },
              "voice":     { label: "Voice",         className: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30" },
              "health":    { label: "Health Est.",   className: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30" },
              "eeg+voice": { label: "EEG + Voice",   className: "bg-cyan-600/20 text-cyan-400 border-cyan-500/30" },
              "eeg+bio":   { label: "EEG + Bio",     className: "bg-cyan-600/20 text-cyan-400 border-cyan-500/30" },
            };
            const cfg = config[source] ?? config["health"];
            return (
              <Badge className={`text-xs border ${cfg.className}`}>
                {cfg.label}
              </Badge>
            );
          })()}
          {isReady && isStreaming && (
            <span className="text-xs font-mono text-foreground/40" title={`Inference: ${isLocal ? "local" : "server"} (${latencyMs.toFixed(0)}ms)`}>
              {isLocal ? "LOCAL" : "SERVER"}
            </span>
          )}
        </div>
      </div>

      {/* ---- Connection Banner (offline) ---- */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning">
          <div className="flex items-center gap-3">
            <Radio className="h-4 w-4 shrink-0" />
            EEG is offline. Connect Muse for live brain data.
          </div>
          <div className="mt-3 flex justify-start">
            <Button
              size="sm"
              onClick={() => device.connect("synthetic")}
              className="bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30"
            >
              <Play className="h-3.5 w-3.5 mr-1.5" />
              Try Synthetic Device
            </Button>
          </div>
        </div>
      )}

      {/* Alert Banner */}
      <AlertBanner
        level={alertLevel}
        anomalyScore={anomaly?.anomaly_score}
        seizureProbability={anomaly?.seizure_probability}
        spikesDetected={anomaly?.spikes_detected}
      />

      {/* EMG Artifact Warning Banner */}
      {isStreaming && emgResult?.emgDetected && (
        <div
          className="rounded-xl border border-amber-500/30 bg-amber-500/5 px-4 py-3 flex items-start gap-3"
          data-testid="emg-warning-banner"
        >
          <Zap className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-amber-400">Signal too noisy -- try relaxing your forehead</p>
            <p className="text-xs text-muted-foreground mt-1">
              Muscle artifact detected. Jaw clenching or forehead tensing can contaminate readings.
            </p>
            {emgResult.artifactPercent > 0.3 && (
              <p className="text-[10px] text-amber-400/70 mt-1 font-mono">
                {Math.round(emgResult.artifactPercent * 100)}% of recent frames flagged
              </p>
            )}
          </div>
        </div>
      )}

      {/* Blink Rate & Alertness Indicator */}
      {isStreaming && blinkStats && (
        <div
          className={`rounded-xl border px-4 py-3 flex items-center gap-3 ${
            blinkStats.shouldSuggestBreak
              ? "border-amber-500/30 bg-amber-500/5"
              : "border-border/50 bg-card/50"
          }`}
          data-testid="blink-rate-indicator"
        >
          <Eye className={`h-5 w-5 shrink-0 ${
            blinkStats.alertnessState === "focused" ? "text-emerald-400" :
            blinkStats.alertnessState === "normal" ? "text-blue-400" :
            blinkStats.alertnessState === "fatigued" ? "text-amber-400" :
            "text-red-400"
          }`} />
          <div className="flex items-center gap-4 flex-wrap flex-1">
            <div>
              <span className="text-sm font-medium">{Math.round(blinkStats.blinksPerMinute)} blinks/min</span>
              <span className="text-xs text-muted-foreground ml-2">
                avg {Math.round(blinkStats.avgBlinkDuration)}ms
              </span>
            </div>
            <Badge
              variant="outline"
              className={`text-[10px] capitalize ${
                blinkStats.alertnessState === "focused" ? "text-emerald-400 border-emerald-400/30" :
                blinkStats.alertnessState === "normal" ? "text-blue-400 border-blue-400/30" :
                blinkStats.alertnessState === "fatigued" ? "text-amber-400 border-amber-400/30" :
                "text-red-400 border-red-400/30"
              }`}
            >
              {blinkStats.alertnessState}
            </Badge>
            {blinkStats.shouldSuggestBreak && (
              <span className="text-xs text-amber-400">Take a break</span>
            )}
          </div>
        </div>
      )}

      {/* Breathing State Indicator */}
      {isStreaming && breathingAnalysis && breathingAnalysis.state !== "unknown" && (
        <BreathingIndicator analysis={breathingAnalysis} />
      )}

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

      {/* ---- 2. Brain Age card (prominent) ---- */}
      {isStreaming && <BrainAgeCard />}

      {/* ---- 3. EEG Waveform (350px tall, clean) ---- */}
      <div className="rounded-[14px] bg-card border border-border p-4 sm:p-6 overflow-hidden">
        <h3 className="text-lg font-semibold mb-4">
          EEG Brain Wave Activity
        </h3>
        {isStreaming ? (
          <>
            <EEGWaveformCanvas
              signals={latestFrame?.signals as number[][] | undefined}
              windowSec={5}
              height={350}
            />
            {/* Channel legend + data diagnostic */}
            <div className="flex items-center gap-2 sm:gap-4 mt-3 flex-wrap">
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
              <span className="ml-auto text-[9px] font-mono text-muted-foreground/50">
                pkts:{museBle.packetCount}
                {latestFrame?.signals
                  ? ` | ${latestFrame.signals.length}ch\u00D7${latestFrame.signals[0]?.length ?? 0}`
                  : " | no signals"}
                {bp ? ` | a:${(alphaVal ?? 0).toFixed(3)}` : " | no bp"}
                {museBle._subscribeInfo ? ` | ${museBle._subscribeInfo}` : ""}
              </span>
            </div>
          </>
        ) : (
          <div className="h-64 flex flex-col items-center justify-center gap-3 border border-dashed border-border/30 rounded-lg p-4">
            <Radio className="h-6 w-6 text-muted-foreground/50" />
            <p className="text-sm text-muted-foreground text-center">
              No EEG signal -- showing voice + health estimates
            </p>
            {voiceResult && (
              <div className="flex items-center gap-2 text-xs">
                <span className="capitalize font-medium text-foreground/80">{voiceResult.emotion}</span>
                <span className="text-muted-foreground">
                  valence {voiceResult.valence >= 0 ? "+" : ""}{voiceResult.valence.toFixed(2)}
                </span>
                <span className="text-muted-foreground">-</span>
                <span className="text-muted-foreground">{Math.round(voiceResult.confidence * 100)}% conf</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ---- 4. Band Powers (horizontal bars) ---- */}
      {isStreaming && analysis?.band_powers && (
        <Card className="rounded-[14px] bg-card border border-border p-6">
          <h3 className="text-lg font-semibold mb-4">Band Powers</h3>
          <div className="space-y-3">
            {ALL_BANDS.map(({ key, label, range }) => {
              const value = (analysis.band_powers as Record<string, number>)[key] ?? 0;
              return (
                <div key={key} className="flex items-center gap-3" data-testid={`band-${key}`}>
                  <div className="w-20">
                    <span className="text-xs font-mono text-muted-foreground" style={{ color: bandColor(key) }}>{label}</span>
                    <span className="text-[9px] text-muted-foreground block">{range}</span>
                  </div>
                  <div className="flex-1">
                    <div className="h-2 rounded-full overflow-hidden bg-muted/40">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${Math.min(100, value * 100)}%`,
                          background: bandColor(key),
                        }}
                      />
                    </div>
                  </div>
                  <span className="text-xs font-mono w-12 text-right">{(value * 100).toFixed(0)}%</span>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* ---- 4b. EEG Brain Connectivity (PLV coherence arcs) ---- */}
      {(() => {
        const plv = (analysis as Record<string, unknown> | undefined)?.plv_connectivity as Record<string, number> | undefined;
        return (
          <EEGCoherenceCard
            frontalPlv={plv?.plv_frontal_alpha ?? null}
            temporalPlv={plv?.plv_mean_theta ?? null}
            leftFrontotemporalPlv={plv?.plv_fronto_temporal_alpha ?? null}
            isStreaming={isStreaming}
          />
        );
      })()}

      {/* ---- 5. Brain State summary (4 cards: Emotion, Stress, Focus, Flow) ---- */}
      <div className="rounded-[14px] bg-card border border-border p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Brain State</h3>
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
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { label: "Emotion", value: epochReady ? (stableAnalysis.emotions?.emotion ?? "\u2014") : "Collecting...", sub: epochReady ? `V:${((stableAnalysis.emotions?.valence ?? 0) * 100).toFixed(0)}%` : "4s needed" },
                  { label: "Stress", value: stableAnalysis.stress?.level ?? "\u2014", sub: `${Math.round((stableAnalysis.stress?.stress_index ?? 0) * 100)}%` },
                  { label: "Focus", value: stableAnalysis.attention?.state ?? "\u2014", sub: `${Math.round((stableAnalysis.attention?.attention_score ?? 0) * 100)}%` },
                  { label: "Flow", value: stableAnalysis.flow_state?.in_flow ? "In Flow" : "Normal", sub: `${Math.round((stableAnalysis.flow_state?.flow_score ?? 0) * 100)}%` },
                ].map(({ label, value, sub }) => (
                  <div key={label} className="bg-muted/20 rounded-lg p-3">
                    <p className="text-[10px] text-muted-foreground">{label}</p>
                    <p className="text-sm font-semibold capitalize">{value}</p>
                    <p className="text-[10px] text-muted-foreground font-mono">{sub}</p>
                  </div>
                ))}
              </div>
              {/* Emotion confidence meter */}
              {epochReady && stableAnalysis.emotions && (() => {
                const probs = stableAnalysis.emotions.probabilities as Record<string, number> | undefined;
                const topProb = probs ? Math.max(...Object.values(probs)) : 0.5;
                const sqNorm = sqScore != null ? sqScore / 100 : undefined;
                const conf = calculateEmotionConfidence({
                  modelConfidence: topProb,
                  signalQuality: sqNorm,
                });
                return conf.showEmotion ? (
                  <div className="pt-1">
                    <ConfidenceMeter confidence={conf.confidence} size="sm" showLabel />
                  </div>
                ) : (
                  <p className="text-[10px] text-muted-foreground/70 leading-relaxed pt-1">
                    Not enough data to determine your emotional state. Try adjusting the headband.
                  </p>
                );
              })()}
              {/* Ensemble model contribution breakdown */}
              {epochReady && stableAnalysis.emotions && (() => {
                const emo = stableAnalysis.emotions as Record<string, unknown>;
                return (
                  <EnsembleExplanation
                    eegnetContribution={emo.eegnet_contribution as number | undefined}
                    heuristicContribution={emo.heuristic_contribution as number | undefined}
                    epochQuality={emo.epoch_quality as number | undefined}
                  />
                );
              })()}
              {/* Top emotion probabilities */}
              {epochReady && stableAnalysis.emotions?.probabilities && (
                <div className="space-y-1 pt-2 border-t border-border/20">
                  {Object.entries(stableAnalysis.emotions.probabilities as Record<string, number>)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 3)
                    .map(([emo, prob]) => (
                      <div key={emo} className="flex items-center gap-2 text-[11px]">
                        <span className="w-14 capitalize text-muted-foreground">{emo}</span>
                        <div className="flex-1 h-1 rounded-full overflow-hidden bg-muted/40">
                          <div className="h-full rounded-full bg-emerald-500" style={{ width: `${Math.round(prob * 100)}%` }} />
                        </div>
                        <span className="w-6 text-right font-mono text-muted-foreground">{Math.round(prob * 100)}%</span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          );
        })() : (
          <div className="h-48 flex flex-col items-center justify-center gap-3 border border-dashed border-border/30 rounded-lg p-4">
            <Brain className="h-6 w-6 text-muted-foreground/50" />
            <p className="text-sm text-muted-foreground text-center">
              No EEG signal -- showing voice + health estimates
            </p>
            {voiceResult ? (
              <div className="space-y-2 w-full max-w-[200px]">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Emotion</span>
                  <span className="capitalize font-medium">{voiceResult.emotion}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Valence</span>
                  <span className="font-mono">
                    {voiceResult.valence >= 0 ? "+" : ""}
                    {voiceResult.valence.toFixed(2)}
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Arousal</span>
                  <span className="font-mono">{voiceResult.arousal.toFixed(2)}</span>
                </div>
                <p className="text-[10px] text-muted-foreground/60 text-center pt-1">
                  via {voiceResult.model_type}
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

      {/* Intervention Suggestion */}
      {(() => {
        const emo = isStreaming && epochReady
          ? (stableAnalysis?.emotions?.emotion ?? null)
          : (voiceResult?.emotion ?? null);
        const stressIdx = isStreaming && stableAnalysis?.stress
          ? stableAnalysis.stress.stress_index
          : undefined;
        const val = isStreaming && stableAnalysis?.emotions
          ? stableAnalysis.emotions.valence
          : (voiceResult?.valence ?? undefined);
        if (!emo) return null;
        if (emgResult?.emgDetected) return null;
        return (
          <InterventionSuggestion
            emotion={emo}
            stressIndex={stressIdx}
            valence={val}
            compact
          />
        );
      })()}

      {/* ---- 6. ML Models (collapsed/expandable) ---- */}
      <Card className="rounded-[14px] bg-card border border-border">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">ML Models</CardTitle>
            <span className="text-xs text-muted-foreground">
              {lf ? "Live" : "Offline"}
            </span>
          </div>
        </CardHeader>
        <CardContent>
          {/* Always show key 4 models */}
          <div className="grid grid-cols-4 gap-2">
            {keyModels.map(({ name, value }) => {
              const href = MODEL_ROUTES[name] ?? null;
              const inner = (
                <>
                  <p className="text-xs text-muted-foreground truncate">{name}</p>
                  <p className="text-sm font-mono font-semibold mt-0.5 truncate">{value}</p>
                </>
              );
              if (href) {
                return (
                  <Link key={name} href={href} className="block">
                    <div
                      className="rounded border border-border/30 p-2 text-center transition-colors hover:border-primary/50 hover:bg-primary/5 cursor-pointer"
                      data-testid={`model-link-${name}`}
                    >
                      {inner}
                    </div>
                  </Link>
                );
              }
              return (
                <div key={name} className="rounded border border-border/30 p-2 text-center" data-testid={`model-card-${name}`}>
                  {inner}
                </div>
              );
            })}
          </div>

          {/* Expandable remaining models */}
          {modelsExpanded && (
            <div className="grid grid-cols-4 gap-2 mt-2">
              {remainingModels.map(({ name, value }) => {
                const href = MODEL_ROUTES[name] ?? null;
                const inner = (
                  <>
                    <p className="text-xs text-muted-foreground truncate">{name}</p>
                    <p className="text-sm font-mono font-semibold mt-0.5 truncate">{value}</p>
                  </>
                );
                if (href) {
                  return (
                    <Link key={name} href={href} className="block">
                      <div
                        className="rounded border border-border/30 p-2 text-center transition-colors hover:border-primary/50 hover:bg-primary/5 cursor-pointer"
                        data-testid={`model-link-${name}`}
                      >
                        {inner}
                      </div>
                    </Link>
                  );
                }
                return (
                  <div key={name} className="rounded border border-border/30 p-2 text-center" data-testid={`model-card-${name}`}>
                    {inner}
                  </div>
                );
              })}
            </div>
          )}

          <button
            onClick={() => setModelsExpanded(!modelsExpanded)}
            className="w-full mt-3 flex items-center justify-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors py-1"
          >
            {modelsExpanded ? (
              <>
                <ChevronUp className="h-3.5 w-3.5" />
                Show less
              </>
            ) : (
              <>
                <ChevronDown className="h-3.5 w-3.5" />
                Show all 16 models
              </>
            )}
          </button>
        </CardContent>
      </Card>

      {/* ---- 7. Diagnostic log (at bottom, small text) ---- */}
      {museBle.diagLog.length > 0 && (
        <div className="p-3 rounded-[14px] bg-card border border-border max-h-40 overflow-y-auto">
          <p className="text-[10px] font-mono text-muted-foreground mb-1">Diagnostic Log</p>
          {museBle.diagLog.map((line, i) => (
            <p key={i} className="text-[9px] font-mono text-green-400/80 leading-tight">{line}</p>
          ))}
        </div>
      )}

      {/* EEG intervention trigger toast (#504) */}
      {activeTrigger && (
        <InterventionTriggerToast
          trigger={activeTrigger}
          onDismiss={dismissTrigger}
        />
      )}
    </main>
  );
}

/* -- Helper: Brain State Narrative -- */

function getBrainStateNarrative(analysis: Record<string, unknown>): { headline: string; story: string; color: string } {
  const stress   = (analysis.stress   as { stress_index?: number })?.stress_index   ?? 0;
  const flow     = (analysis.flow_state as { flow_score?: number })?.flow_score     ?? 0;
  const sleep    = (analysis.sleep_staging as { stage?: string })?.stage             ?? "wake";
  const meditation = (analysis.meditation as { meditation_score?: number })?.meditation_score ?? 0;
  const creativity = (analysis.creativity  as { creativity_score?: number })?.creativity_score ?? 0;
  const attention  = (analysis.attention   as { attention_score?: number })?.attention_score   ?? 0;
  const drowsiness = (analysis.drowsiness  as { drowsiness_index?: number })?.drowsiness_index ?? 0;

  if (sleep && sleep !== "wake" && sleep !== "Wake" && sleep !== "\u2014") {
    return { headline: `${sleep} Sleep`, story: `Your brain is in ${sleep} sleep. Delta waves dominate, signaling deep restorative rest. Models continue running passively.`, color: "text-emerald-400" };
  }
  if (flow > 0.7) {
    return { headline: "Flow State", story: "Alpha and theta are balanced with engaged beta -- you're in the zone. Focus is sharp without anxiety. Don't break the session.", color: "text-cyan-400" };
  }
  if (stress > 0.7) {
    return { headline: "Elevated Stress", story: "High-beta activity is dominant, signaling mental tension. Try 4-7-8 breathing or step away briefly. Your nervous system needs a reset.", color: "text-rose-400" };
  }
  if (meditation > 0.7) {
    return { headline: "Meditative State", story: "Alpha waves are dominant and theta is rising -- your mind is calm and inward. Excellent state for creative insight or visualization.", color: "text-emerald-500" };
  }
  if (creativity > 0.7) {
    return { headline: "Creative Flow", story: "Theta and alpha are elevated -- the hallmark of divergent thinking and creative incubation. Great time to brainstorm or explore new ideas.", color: "text-amber-400" };
  }
  if (drowsiness > 0.7) {
    return { headline: "Getting Drowsy", story: "Theta is climbing and alpha is slowing down -- your brain is shifting toward sleep onset. Take a break or try a short walk.", color: "text-orange-400" };
  }
  if (attention > 0.7) {
    return { headline: "Sharp & Focused", story: "Beta activity is high with suppressed alpha -- classic active cognitive engagement. Your brain is fully online and problem-solving.", color: "text-cyan-400" };
  }
  return { headline: "Balanced Resting", story: "A balanced mix of alpha and beta signals a calm yet alert baseline state -- the optimal foundation for learning or light work.", color: "text-primary" };
}

/* -- Helper: band color -- */

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
