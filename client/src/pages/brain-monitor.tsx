import React, { useMemo, useState, useEffect, useRef, useCallback } from "react";
import { EEGWaveformCanvas } from "@/components/charts/eeg-waveform-canvas";
import { SpectrogramChart } from "@/components/charts/spectrogram-chart";
import { SignalQualityIndicator } from "@/components/signal-quality-indicator";
import { AlphaReactivityTest } from "@/components/alpha-reactivity-test";
import { AlertBanner, type AlertLevel } from "@/components/alert-banner";
import { SessionControls } from "@/components/session-controls";
import { SimulationModeBanner } from "@/components/simulation-mode-banner";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Radio,
  Brain,
  Zap,
  Play,
} from "lucide-react";
import { useInference } from "@/hooks/use-inference";
import { useDevice } from "@/hooks/use-device";
import { useVoiceCache } from "@/hooks/use-voice-cache";
import {
  analyzeWavelet,
  type WaveletResult,
  type AnomalyResult,
} from "@/lib/ml-api";
import { assessSignalQuality, type SignalQualityResult as SQResult } from "@/lib/signal-quality";
import { Link } from "wouter";
import { Music } from "lucide-react";
import { museBle } from "@/lib/muse-ble";
import { ConfidenceMeter } from "@/components/confidence-meter";
import { InterventionSuggestion } from "@/components/intervention-suggestion";
import { emgDetector, type EMGDetectionResult } from "@/lib/emg-detector";
import { calculateEmotionConfidence } from "@/lib/confidence-calculator";
import { computeBlinkStats, type BlinkStats } from "@/lib/blink-detector";
import { detectBreathingState, type BreathingAnalysis } from "@/lib/breathing-detector";
import { BreathingIndicator } from "@/components/breathing-indicator";
import { Eye } from "lucide-react";
import { useInterventionTriggers } from "@/hooks/use-intervention-triggers";
import { InterventionTriggerToast } from "@/components/intervention-trigger-toast";
import type { TriggerState } from "@/lib/eeg-intervention-trigger";

// Route targets for each ML model card — null means no linked page
const MODEL_ROUTES: Record<string, string | null> = {
  "Emotion": "/emotions",
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

  const [wavelet, setWavelet] = useState<WaveletResult | null>(null);
  const anomaly = (latestFrame?.analysis as { anomaly?: AnomalyResult } | undefined)?.anomaly ?? null;
  const [_isRecording, setIsRecording] = useState(false);
  const waveletTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const analysis = latestFrame?.analysis;

  // ── Per-electrode signal quality (spectral flatness + amplitude) ──
  const isSynthetic = deviceStatus?.device_type === "synthetic" || selectedDevice === "synthetic";
  const [sqResult, setSqResult] = useState<SQResult | null>(null);
  const [showAlphaTest, setShowAlphaTest] = useState(false);
  const sqTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── EMG artifact detection ──────────────────────────────────────────
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

  // ── Blink-rate detection (AF7 channel) ────────────────────────────────
  const [blinkStats, setBlinkStats] = useState<BlinkStats | null>(null);

  useEffect(() => {
    if (!isStreaming || !latestFrame?.signals || isSynthetic) {
      setBlinkStats(null);
      return;
    }
    // AF7 is channel index 1 (BrainFlow Muse 2 order: TP9, AF7, AF8, TP10)
    const signals = latestFrame.signals as number[][];
    if (signals.length < 2) return;
    const af7 = new Float32Array(signals[1]);
    const fs = latestFrame.sample_rate || 256;
    // Use alpha power from analysis if available, otherwise default to 0.2
    const analysisAny = latestFrame.analysis as Record<string, unknown> | undefined;
    const bp = analysisAny?.band_powers as Record<string, number> | undefined;
    const alphaPower = bp?.alpha ?? 0.2;
    const stats = computeBlinkStats(af7, fs, alphaPower);
    setBlinkStats(stats);
  }, [isStreaming, isSynthetic, latestFrame?.timestamp]);

  // ── Breathing state detection (AF7 channel, buffered) ────────────────────
  const [breathingAnalysis, setBreathingAnalysis] = useState<BreathingAnalysis | null>(null);
  const breathingBufferRef = useRef<Float32Array>(new Float32Array(0));

  useEffect(() => {
    if (!isStreaming || !latestFrame?.signals || isSynthetic) {
      setBreathingAnalysis(null);
      breathingBufferRef.current = new Float32Array(0);
      return;
    }
    // AF7 is channel index 1 (BrainFlow Muse 2 order: TP9, AF7, AF8, TP10)
    const signals = latestFrame.signals as number[][];
    if (signals.length < 2) return;
    const af7 = new Float32Array(signals[1]);
    const fs = latestFrame.sample_rate || 256;

    // Accumulate into a sliding buffer (keep last 30 seconds for breathing analysis)
    const maxSamples = fs * 30;
    const prev = breathingBufferRef.current;
    const combined = new Float32Array(prev.length + af7.length);
    combined.set(prev);
    combined.set(af7, prev.length);
    // Trim to maxSamples
    if (combined.length > maxSamples) {
      breathingBufferRef.current = combined.slice(combined.length - maxSamples);
    } else {
      breathingBufferRef.current = combined;
    }

    // Only analyze once we have at least 10 seconds of data
    if (breathingBufferRef.current.length >= fs * 10) {
      const result = detectBreathingState(breathingBufferRef.current, fs);
      setBreathingAnalysis(result);
    }
  }, [isStreaming, isSynthetic, latestFrame?.timestamp]);

  const CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"];

  // ── EEG intervention trigger engine (#504) ────────────────────────
  const sessionStartRef = useRef<number>(Date.now());
  const stressDurationRef = useRef<number>(0);
  const highBetaDurationRef = useRef<number>(0);
  const lastStressCheckRef = useRef<number>(Date.now());

  // Reset session timer when streaming starts/stops
  useEffect(() => {
    if (isStreaming) {
      sessionStartRef.current = Date.now();
      stressDurationRef.current = 0;
      highBetaDurationRef.current = 0;
    }
  }, [isStreaming]);

  // Accumulate stress/beta duration from analysis
  useEffect(() => {
    if (!isStreaming || !analysis) return;
    const now = Date.now();
    const dt = (now - lastStressCheckRef.current) / 1000;
    lastStressCheckRef.current = now;
    if (dt > 10) return; // skip large gaps

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

    // Run quality assessment every 2 seconds
    sqTimerRef.current = setInterval(() => {
      if (latestFrame?.signals && latestFrame.signals.length >= 4) {
        const channels = latestFrame.signals.map((ch: number[]) => new Float32Array(ch));
        const result = assessSignalQuality(channels, latestFrame.sample_rate || 256, CHANNEL_NAMES);
        setSqResult(result);
      }
    }, 2000);

    // Run once immediately
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

  /** Provide current EEG data to the alpha reactivity test component. */
  const getCurrentEegData = useCallback((): Float32Array[] | null => {
    if (!latestFrame?.signals || latestFrame.signals.length < 4) return null;
    return latestFrame.signals.map((ch: number[]) => new Float32Array(ch));
  }, [latestFrame]);

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

  // Electrode grid — suppress quality data for synthetic board (non-physiological signals)
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
    <main className="p-4 md:p-6 pb-24 space-y-6">
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

      {/* Connection Banner */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning">
          <div className="flex items-center gap-3">
            <Radio className="h-4 w-4 shrink-0" />
            EEG is offline. You can still use voice + health features elsewhere; connect Muse only for live brain data here.
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
            <p className="text-sm font-medium text-amber-400">Signal too noisy — try relaxing your forehead</p>
            <p className="text-xs text-muted-foreground mt-1">
              Muscle artifact detected in the EEG. Jaw clenching, forehead tensing, or eye movement can contaminate readings.
              Emotion data is unreliable until the signal improves.
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
        <div className="xl:col-span-2 glass-card p-4 sm:p-6 rounded-xl hover-glow overflow-hidden">
          <div className="flex flex-col gap-3 mb-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">
                EEG Brain Wave Activity
              </h3>
              <div className="flex items-center gap-2">
                {isStreaming && (
                  <Radio className="h-4 w-4 text-primary animate-pulse" />
                )}
                <span className={`text-sm font-mono ${sourceColor}`} aria-live="polite">
                  {sourceLabel}
                </span>
              </div>
            </div>
            {/* Signal status bar — wraps on mobile */}
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
              {/* Per-electrode quality dots */}
              {sqResult && !isSynthetic && (
                <SignalQualityIndicator channels={sqResult.channels} />
              )}
              {/* Signal source badge */}
              {(() => {
                const source: string =
                  (a?.emotions as { signal_source?: string } | undefined)?.signal_source ??
                  (isStreaming ? "eeg" : voiceResult ? "voice" : "health");

                const config: Record<string, { label: string; className: string }> = {
                  "eeg":       { label: "EEG",          className: "bg-purple-500/20 text-purple-400 border-purple-500/30" },
                  "voice":     { label: "Voice",         className: "bg-indigo-500/20 text-indigo-400 border-indigo-500/30" },
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
          {isStreaming ? (
            <>
              <EEGWaveformCanvas
                signals={latestFrame?.signals as number[][] | undefined}
                windowSec={5}
                height={280}
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
                    ? ` | ${latestFrame.signals.length}ch×${latestFrame.signals[0]?.length ?? 0}`
                    : " | no signals"}
                  {bp ? ` | a:${(alphaVal ?? 0).toFixed(3)}` : " | no bp"}
                  {museBle._subscribeInfo ? ` | ${museBle._subscribeInfo}` : ""}
                </span>
              </div>
              {/* BLE diagnostic log — shows connection steps */}
              {museBle.diagLog.length > 0 && (
                <div className="mt-3 p-2 rounded bg-black/30 max-h-40 overflow-y-auto">
                  {museBle.diagLog.map((line, i) => (
                    <p key={i} className="text-[9px] font-mono text-green-400/80 leading-tight">{line}</p>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="h-64 flex flex-col items-center justify-center gap-3 border border-dashed border-border/30 rounded-lg p-4">
              <Radio className="h-6 w-6 text-muted-foreground/50" />
              <p className="text-sm text-muted-foreground text-center">
                No EEG signal — showing voice + health estimates
              </p>
              {voiceResult && (
                <div className="flex items-center gap-2 text-xs">
                  <span className="capitalize font-medium text-foreground/80">{voiceResult.emotion}</span>
                  <span className="text-muted-foreground">
                    valence {voiceResult.valence >= 0 ? "+" : ""}{voiceResult.valence.toFixed(2)}
                  </span>
                  <span className="text-muted-foreground">·</span>
                  <span className="text-muted-foreground">{Math.round(voiceResult.confidence * 100)}% conf</span>
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
              {voiceResult ? (
                <div className="space-y-2 w-full max-w-[160px] sm:max-w-[200px]">
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
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Confidence</span>
                    <span className="font-mono">{Math.round(voiceResult.confidence * 100)}%</span>
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
      </div>

      {/* Intervention Suggestion — pair emotion data with actionable intervention */}
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
        // When EMG is detected, override emotion display
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

      {/* Band Powers Bar — always show all 5 standard EEG bands */}
      {isStreaming && analysis?.band_powers && (
        <Card className="glass-card p-6 rounded-xl hover-glow">
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
                    <div className="h-2 rounded-full overflow-hidden" style={{ background: "hsl(220,22%,12%)" }}>
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
            {modelOutputs.map(({ name, value }) => {
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

      {/* Signal Quality Recommendation + Alpha Reactivity Test */}
      {isStreaming && !isSynthetic && (
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Signal Quality</h3>
            {sqResult && (
              <Badge
                className={
                  sqResult.overall === "good"
                    ? "bg-green-500/20 text-green-400 border-green-500/30"
                    : sqResult.overall === "fair"
                      ? "bg-amber-500/20 text-amber-400 border-amber-500/30"
                      : "bg-red-500/20 text-red-400 border-red-500/30"
                }
              >
                {sqResult.overall.charAt(0).toUpperCase() + sqResult.overall.slice(1)}
              </Badge>
            )}
          </div>

          {sqResult && (
            <div className="space-y-3">
              {/* Per-electrode quality indicator (larger version) */}
              <div className="flex justify-center">
                <SignalQualityIndicator channels={sqResult.channels} />
              </div>

              {/* Recommendation text */}
              <p className="text-sm text-muted-foreground text-center">
                {sqResult.recommendation}
              </p>

              {/* Alpha reactivity test button */}
              {!showAlphaTest && (
                <div className="flex justify-center pt-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowAlphaTest(true)}
                  >
                    Run Alpha Reactivity Test
                  </Button>
                </div>
              )}
            </div>
          )}

          {/* Alpha reactivity test panel */}
          {showAlphaTest && (
            <div className="mt-4 flex justify-center">
              <AlphaReactivityTest
                getCurrentData={getCurrentEegData}
                fs={latestFrame?.sample_rate || 256}
              />
            </div>
          )}
        </Card>
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
    return { headline: `${sleep} Sleep`, story: `Your brain is in ${sleep} sleep. Delta waves dominate, signaling deep restorative rest. Models continue running passively.`, color: "text-indigo-400" };
  }
  if (flow > 0.7) {
    return { headline: "Flow State", story: "Alpha and theta are balanced with engaged beta — you're in the zone. Focus is sharp without anxiety. Don't break the session.", color: "text-cyan-400" };
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
    return { headline: "Sharp & Focused", story: "Beta activity is high with suppressed alpha — classic active cognitive engagement. Your brain is fully online and problem-solving.", color: "text-cyan-400" };
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
