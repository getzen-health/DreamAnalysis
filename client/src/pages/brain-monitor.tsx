import { useMemo, useState, useEffect, useRef } from "react";
import { EEGChart } from "@/components/charts/eeg-chart";
import { SpectrogramChart } from "@/components/charts/spectrogram-chart";
import { NeuralNetwork } from "@/components/neural-network";
import { SignalQualityBadge } from "@/components/signal-quality-badge";
import { AlertBanner, type AlertLevel } from "@/components/alert-banner";
import { SessionControls } from "@/components/session-controls";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
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
} from "lucide-react";
import { useMetrics } from "@/hooks/use-metrics";
import { useInference } from "@/hooks/use-inference";
import { useDevice } from "@/hooks/use-device";
import {
  analyzeWavelet,
  type WaveletResult,
  type AnomalyResult,
} from "@/lib/ml-api";

export default function BrainMonitor() {
  const { eegData, neuralActivity } = useMetrics();
  const { isLocal, latencyMs, isReady } = useInference();
  const device = useDevice();
  const { state: deviceState, latestFrame, deviceStatus } = device;
  const isStreaming = deviceState === "streaming";

  const [wavelet, setWavelet] = useState<WaveletResult | null>(null);
  const [anomaly] = useState<AnomalyResult | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const waveletTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

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
  const analysis = latestFrame?.analysis;
  const signalQuality = latestFrame?.quality ?? null;

  const alphaHz = analysis?.band_powers?.alpha
    ? `${(analysis.band_powers.alpha * 12).toFixed(1)} Hz`
    : "8.5 Hz";

  const betaHz = analysis?.band_powers?.beta
    ? `${(analysis.band_powers.beta * 30).toFixed(1)} Hz`
    : "23.2 Hz";

  const sourceLabel = isStreaming ? "DEVICE" : "SIMULATION";
  const sourceColor = isStreaming ? "text-primary" : "text-success";
  const alertLevel: AlertLevel = anomaly?.alert_level || "normal";

  // Electrode grid
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

  const channelQuality = signalQuality?.channel_quality || [];
  const hasRealData = channelQuality.length > 0;
  const activeCount = hasRealData ? channelQuality.filter((q) => q >= 80).length : 64;
  const weakCount = hasRealData ? channelQuality.filter((q) => q >= 60 && q < 80).length : 0;
  const errorCount = hasRealData ? channelQuality.filter((q) => q < 60).length : 0;

  return (
    <main className="p-4 md:p-6 space-y-6">
      {/* Alert Banner */}
      <AlertBanner
        level={alertLevel}
        anomalyScore={anomaly?.anomaly_score}
        seizureProbability={anomaly?.seizure_probability}
        spikesDetected={anomaly?.spikes_detected}
      />

      {/* Emotion Shift Alert */}
      {latestFrame?.emotion_shift?.shift_detected && (
        <div className="shift-alert p-4 rounded-xl flex items-center gap-3">
          <Zap className="h-5 w-5 text-accent shrink-0" />
          <div>
            <p className="text-sm font-medium">Emotional Shift Detected</p>
            <p className="text-xs text-muted-foreground">
              {latestFrame.emotion_shift.from_state} → {latestFrame.emotion_shift.to_state}
              {" "}(magnitude: {(latestFrame.emotion_shift.magnitude * 100).toFixed(0)}%)
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* EEG Brain Waves */}
        <div className="xl:col-span-2 glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold">
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
          <EEGChart
            alphaWaves={eegData.alphaWaves}
            betaWaves={eegData.betaWaves}
          />
        </div>

        {/* Neural Network Graph */}
        <div className="glass-card p-6 rounded-xl hover-glow neural-glow">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold">Brain Regions</h3>
            <Activity className="text-accent" />
          </div>
          <NeuralNetwork />
        </div>
      </div>

      {/* ── Live 12-Model Analysis Panel ─────────────────────────────── */}
      {isStreaming && analysis && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {/* Sleep Staging */}
          <ModelCard
            icon={<Moon className="h-4 w-4" />}
            title="Sleep Stage"
            value={analysis.sleep_staging?.stage ?? "—"}
            score={analysis.sleep_staging?.confidence}
            color="text-blue-400"
          />

          {/* Emotion */}
          <ModelCard
            icon={<Smile className="h-4 w-4" />}
            title="Emotion"
            value={analysis.emotions?.emotion ?? "—"}
            score={analysis.emotions?.confidence}
            color="text-pink-400"
            extra={
              analysis.emotions ? (
                <div className="grid grid-cols-3 gap-2 mt-2 text-[10px] text-muted-foreground">
                  <span>Valence: {(analysis.emotions.valence * 100).toFixed(0)}%</span>
                  <span>Arousal: {(analysis.emotions.arousal * 100).toFixed(0)}%</span>
                  <span>Stress: {(analysis.emotions.stress_index * 100).toFixed(0)}%</span>
                </div>
              ) : null
            }
          />

          {/* Dream Detection */}
          <ModelCard
            icon={<Moon className="h-4 w-4" />}
            title="Dream"
            value={analysis.dream_detection?.is_dreaming ? "Dreaming" : "Awake"}
            score={analysis.dream_detection?.probability}
            color="text-purple-400"
            extra={
              analysis.dream_detection ? (
                <div className="text-[10px] text-muted-foreground mt-1">
                  REM: {(analysis.dream_detection.rem_likelihood * 100).toFixed(0)}% |
                  Lucidity: {(analysis.dream_detection.lucidity_estimate * 100).toFixed(0)}%
                </div>
              ) : null
            }
          />

          {/* Flow State */}
          <ModelCard
            icon={<Zap className="h-4 w-4" />}
            title="Flow State"
            value={analysis.flow_state?.in_flow ? "In Flow" : "Normal"}
            score={analysis.flow_state?.flow_score}
            color="text-green-400"
          />

          {/* Creativity */}
          <ModelCard
            icon={<Lightbulb className="h-4 w-4" />}
            title="Creativity"
            value={analysis.creativity?.state ?? "—"}
            score={analysis.creativity?.creativity_score}
            color="text-amber-400"
          />

          {/* Memory Encoding */}
          <ModelCard
            icon={<Brain className="h-4 w-4" />}
            title="Memory"
            value={analysis.memory_encoding?.state ?? "—"}
            score={analysis.memory_encoding?.encoding_score}
            color="text-cyan-400"
          />

          {/* Attention */}
          <ModelCard
            icon={<Eye className="h-4 w-4" />}
            title="Attention"
            value={analysis.attention?.state ?? "—"}
            score={analysis.attention?.attention_score}
            color="text-emerald-400"
          />

          {/* Drowsiness */}
          <ModelCard
            icon={<Battery className="h-4 w-4" />}
            title="Drowsiness"
            value={analysis.drowsiness?.state ?? "—"}
            score={analysis.drowsiness?.drowsiness_index}
            color="text-orange-400"
          />

          {/* Cognitive Load */}
          <ModelCard
            icon={<Gauge className="h-4 w-4" />}
            title="Cognitive Load"
            value={analysis.cognitive_load?.level ?? "—"}
            score={analysis.cognitive_load?.load_index}
            color="text-red-400"
          />

          {/* Stress */}
          <ModelCard
            icon={<Shield className="h-4 w-4" />}
            title="Stress"
            value={analysis.stress?.level ?? "—"}
            score={analysis.stress?.stress_index}
            color="text-rose-400"
          />

          {/* Meditation */}
          <ModelCard
            icon={<Heart className="h-4 w-4" />}
            title="Meditation"
            value={analysis.meditation?.depth ?? "—"}
            score={analysis.meditation?.meditation_score}
            color="text-violet-400"
          />

          {/* Lucid Dream (only visible during REM) */}
          {analysis.lucid_dream && (
            <ModelCard
              icon={<Moon className="h-4 w-4" />}
              title="Lucid Dream"
              value={analysis.lucid_dream.state}
              score={analysis.lucid_dream.lucidity_score}
              color="text-fuchsia-400"
            />
          )}
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
