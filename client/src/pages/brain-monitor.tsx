import { useMemo, useState, useEffect, useRef } from "react";
import { EEGChart } from "@/components/charts/eeg-chart";
import { SpectrogramChart } from "@/components/charts/spectrogram-chart";
import { NeuralNetwork } from "@/components/neural-network";
import { SignalQualityBadge } from "@/components/signal-quality-badge";
import { AlertBanner, type AlertLevel } from "@/components/alert-banner";
import { SessionControls } from "@/components/session-controls";
import { Card } from "@/components/ui/card";
import { Activity, Radio } from "lucide-react";
import { useMetrics } from "@/hooks/use-metrics";
import { useInference } from "@/hooks/use-inference";
import {
  getWebSocketUrl,
  getDeviceStatus,
  analyzeWavelet,
  type DeviceStatusResponse,
  type WaveletResult,
  type SignalQuality,
  type AnomalyResult,
} from "@/lib/ml-api";

interface StreamFrame {
  signals: number[][];
  analysis: {
    band_powers: Record<string, number>;
    features: Record<string, number>;
  };
  timestamp: number;
  n_channels: number;
  sample_rate: number;
}

export default function BrainMonitor() {
  const { eegData, neuralActivity } = useMetrics();
  const { analyze, isLocal, latencyMs, isReady } = useInference();
  const [isStreaming, setIsStreaming] = useState(false);
  const [latestFrame, setLatestFrame] = useState<StreamFrame | null>(null);
  const [deviceStatus, setDeviceStatus] = useState<DeviceStatusResponse | null>(null);
  const [wavelet, setWavelet] = useState<WaveletResult | null>(null);
  const [signalQuality, setSignalQuality] = useState<SignalQuality | null>(null);
  const [anomaly, setAnomaly] = useState<AnomalyResult | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const waveletTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Poll device status to detect streaming
  useEffect(() => {
    let active = true;

    const checkStatus = async () => {
      try {
        const status = await getDeviceStatus();
        if (active) {
          setDeviceStatus(status);
          setIsStreaming(status.streaming);
        }
      } catch {
        // ML service not available
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 3000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  // Connect to WebSocket when streaming is detected
  useEffect(() => {
    if (!isStreaming) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      return;
    }

    const ws = new WebSocket(getWebSocketUrl());
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const frame: StreamFrame = JSON.parse(event.data);
        setLatestFrame(frame);
      } catch {
        // ignore
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [isStreaming]);

  // Request wavelet analysis every 2 seconds during streaming
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

  // Derive display values from stream or simulation
  const alphaHz = latestFrame?.analysis?.band_powers?.alpha
    ? `${(latestFrame.analysis.band_powers.alpha * 12).toFixed(1)} Hz`
    : "8.5 Hz";

  const betaHz = latestFrame?.analysis?.band_powers?.beta
    ? `${(latestFrame.analysis.band_powers.beta * 30).toFixed(1)} Hz`
    : "23.2 Hz";

  const sourceLabel = isStreaming ? "DEVICE" : "SIMULATION";
  const sourceColor = isStreaming ? "text-primary" : "text-success";

  const alertLevel: AlertLevel = anomaly?.alert_level || "normal";

  // Stabilize electrode grid — use real channel quality if available
  const electrodeStatuses = useMemo(() => {
    const channelQuality = signalQuality?.channel_quality;
    return Array.from({ length: 64 }, (_, i) => {
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
        const status = Math.random();
        statusClass =
          status > 0.9
            ? "border-destructive/30 bg-destructive/20"
            : status > 0.8
              ? "border-warning/30 bg-warning/20"
              : "border-success/30 bg-success/20";
      }
      const label = `${String.fromCharCode(65 + Math.floor(i / 8))}${(i % 8) + 1}`;
      return { statusClass, label };
    });
  }, [signalQuality?.channel_quality]);

  // Count electrode statuses
  const channelQuality = signalQuality?.channel_quality || [];
  const activeCount = channelQuality.filter((q) => q >= 80).length || 60;
  const weakCount = channelQuality.filter((q) => q >= 60 && q < 80).length || 3;
  const errorCount = channelQuality.filter((q) => q < 60).length || 1;

  return (
    <main className="p-4 md:p-6 space-y-6">
      {/* Alert Banner */}
      <AlertBanner
        level={alertLevel}
        anomalyScore={anomaly?.anomaly_score}
        seizureProbability={anomaly?.seizure_probability}
        spikesDetected={anomaly?.spikes_detected}
      />

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* EEG Brain Waves */}
        <div className="xl:col-span-2 glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-futuristic font-semibold">
              EEG Brain Wave Activity
            </h3>
            <div className="flex items-center space-x-3">
              <SessionControls onRecordingChange={setIsRecording} />
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
              <div className="status-indicator w-2 h-2"></div>
              <span className={`text-sm font-mono ${sourceColor}`}>
                {sourceLabel}
              </span>
              {isReady && (
                <span className="text-xs font-mono text-foreground/40" title={`Inference: ${isLocal ? "local" : "server"} (${latencyMs.toFixed(0)}ms)`}>
                  {isLocal ? "LOCAL" : "SERVER"}
                </span>
              )}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="text-center">
              <p className="text-sm text-primary font-mono">Alpha Waves</p>
              <p
                className="text-2xl font-bold text-primary"
                data-testid="alpha-waves"
              >
                {alphaHz}
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-secondary font-mono">Beta Waves</p>
              <p
                className="text-2xl font-bold text-secondary"
                data-testid="beta-waves"
              >
                {betaHz}
              </p>
            </div>
          </div>
          <EEGChart
            alphaWaves={eegData.alphaWaves}
            betaWaves={eegData.betaWaves}
          />
        </div>

        {/* Neural Network Graph */}
        <div className="glass-card p-6 rounded-xl hover-glow neural-glow">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-futuristic font-semibold">
              Brain Regions
            </h3>
            <Activity className="text-accent" />
          </div>
          <NeuralNetwork />
        </div>
      </div>

      {/* Spectrogram Panel */}
      {wavelet && (
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-futuristic font-semibold">
              Wavelet Spectrogram
            </h3>
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
          <h3 className="text-lg font-futuristic font-semibold">
            Electrode Status Grid
          </h3>
          <span className="text-sm text-foreground/70 font-mono">
            {deviceStatus?.n_channels
              ? `${deviceStatus.n_channels} Channels`
              : "64 Channels"}
          </span>
        </div>
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
      </Card>
    </main>
  );
}
