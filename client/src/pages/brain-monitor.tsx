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

  // Throttled snapshot of analysis for model cards (updates every 5s so user can read)
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
  // Backend sends quality_score (0-1) or sqi; normalize to 0-100
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

  // Electrode grid — only show real data, no random simulation
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
        // No data — show inactive
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
          Connect your Muse 2 from the sidebar to see live brain data. All visualizations require a real device connection.
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
        <div className="shift-alert p-4 rounded-xl flex items-center gap-3">
          <Zap className="h-5 w-5 text-accent shrink-0" />
          <div>
            <p className="text-sm font-medium">Emotional Shift: {latestFrame.emotion_shift.shift_type?.replace(/_/g, " ")}</p>
            <p className="text-xs text-muted-foreground">
              {latestFrame.emotion_shift.description}
              {" "}(magnitude: {((latestFrame.emotion_shift.magnitude ?? 0) * 100).toFixed(0)}%)
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
            <EEGChart
              alphaWaves={eegData.alphaWaves}
              betaWaves={eegData.betaWaves}
            />
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
            extra={stableAnalysis.emotions ? (<div className="grid grid-cols-3 gap-2 mt-2 text-[10px] text-muted-foreground"><span>Valence: {(stableAnalysis.emotions.valence * 100).toFixed(0)}%</span><span>Arousal: {(stableAnalysis.emotions.arousal * 100).toFixed(0)}%</span><span>Stress: {(stableAnalysis.emotions.stress_index * 100).toFixed(0)}%</span></div>) : null} />
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
