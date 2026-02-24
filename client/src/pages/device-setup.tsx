import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import {
  Radio,
  Wifi,
  CheckCircle,
  AlertTriangle,
  ChevronRight,
  ArrowLeft,
  Activity,
  FlaskConical,
  Brain,
  Loader2,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useDevice, type DeviceState } from "@/hooks/use-device";

// ── Step indices ────────────────────────────────────────────────────
type Step = 0 | 1 | 2 | 3 | 4;
// 0 = Select device
// 1 = Placement guide  (real device only)
// 2 = Connecting spinner
// 3 = Signal quality check
// 4 = All done

// ── Channel names for Muse 2 (BrainFlow order) ──────────────────────
const CHANNEL_LABELS = ["TP9 (left ear)", "AF7 (left brow)", "AF8 (right brow)", "TP10 (right ear)"];
const CHANNEL_SHORT  = ["TP9", "AF7", "AF8", "TP10"];

// ── Quality helpers ──────────────────────────────────────────────────
function sqiLabel(sqi: number): { text: string; color: string } {
  if (sqi >= 0.70) return { text: "Excellent",  color: "text-emerald-400" };
  if (sqi >= 0.50) return { text: "Good",        color: "text-emerald-400" };
  if (sqi >= 0.35) return { text: "Fair",        color: "text-amber-400"   };
  return                   { text: "Poor",        color: "text-red-400"     };
}

function channelColor(q: number): string {
  if (q >= 0.70) return "bg-emerald-500";
  if (q >= 0.40) return "bg-amber-500";
  return "bg-red-500";
}

// ── Progress indicator ───────────────────────────────────────────────
function WizardProgress({ step, totalSteps }: { step: Step; totalSteps: number }) {
  return (
    <div className="w-full space-y-1.5">
      <Progress value={((step + 1) / totalSteps) * 100} className="h-1.5" />
      <p className="text-[11px] text-muted-foreground text-right">
        Step {step + 1} of {totalSteps}
      </p>
    </div>
  );
}

// ── Step 0: Select Device ────────────────────────────────────────────
function StepSelectDevice({
  devices,
  devicesLoaded,
  error,
  onSelect,
  onRefresh,
}: {
  devices: { type: string; name: string; channels: number; sample_rate: number }[];
  devicesLoaded: boolean;
  error: string | null;
  onSelect: (type: string) => void;
  onRefresh: () => void;
}) {
  const isSynthetic = (type: string) => type === "synthetic";

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold mb-1">Connect your EEG headset</h2>
        <p className="text-sm text-muted-foreground">
          Select a device below. We'll walk you through placement and check signal
          quality before your first session.
        </p>
      </div>

      {(error === "unreachable" || error === "Failed to fetch") && (
        <Card className="p-4 border-warning/30 bg-warning/5 space-y-2">
          <p className="text-sm font-medium text-warning flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            ML backend not reachable
          </p>
          <p className="text-xs text-muted-foreground">
            Start the backend first:
          </p>
          <code className="block text-[11px] bg-black/40 text-emerald-400 px-3 py-2 rounded font-mono">
            cd ~/NeuralDreamWorkshop/ml &amp;&amp; ./start.sh
          </code>
        </Card>
      )}

      <div className="space-y-2">
        {!devicesLoaded ? (
          <div className="flex items-center gap-2 py-6 justify-center text-muted-foreground text-sm">
            <Loader2 className="h-4 w-4 animate-spin" />
            Scanning for devices…
          </div>
        ) : devices.length === 0 ? (
          <p className="text-sm text-muted-foreground italic text-center py-6">
            No devices found.
          </p>
        ) : (
          devices.map((dev) => (
            <button
              key={dev.type}
              onClick={() => onSelect(dev.type)}
              className="w-full text-left"
            >
              <Card className="p-4 border-border/40 hover:border-primary/40 hover:bg-card/80 transition-all cursor-pointer group">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
                    {isSynthetic(dev.type)
                      ? <FlaskConical className="h-5 w-5 text-purple-400" />
                      : <Radio className="h-5 w-5 text-primary" />
                    }
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm">{dev.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {dev.channels} ch · {dev.sample_rate} Hz
                      {isSynthetic(dev.type) && " · no headset required"}
                    </p>
                  </div>
                  <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-foreground transition-colors" />
                </div>
              </Card>
            </button>
          ))
        )}
      </div>

      <Button variant="ghost" size="sm" onClick={onRefresh} className="w-full text-muted-foreground">
        Refresh device list
      </Button>
    </div>
  );
}

// ── Step 1: Placement Guide ──────────────────────────────────────────
function StepPlacement({ deviceName, onContinue }: { deviceName: string; onContinue: () => void }) {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold mb-1">Place your {deviceName}</h2>
        <p className="text-sm text-muted-foreground">
          Correct placement is the biggest factor in signal quality. Take 30 seconds to get it right.
        </p>
      </div>

      {/* ASCII electrode map */}
      <Card className="p-4 font-mono text-xs bg-black/20 border-border/30 leading-relaxed text-center">
        <pre className="text-muted-foreground select-none">{`
        Front
          AF7 ──── AF8
         (L brow) (R brow)
              ‖   ‖
         TP9 ──── TP10
        (L ear) (R ear)
        `}</pre>
        <div className="grid grid-cols-2 gap-2 mt-3 text-left not-italic">
          {[
            { ch: "TP9",  pos: "Left ear — conductive pad behind left ear" },
            { ch: "AF7",  pos: "Left brow — flat sensor on left forehead" },
            { ch: "AF8",  pos: "Right brow — flat sensor on right forehead" },
            { ch: "TP10", pos: "Right ear — conductive pad behind right ear" },
          ].map(({ ch, pos }) => (
            <div key={ch} className="flex items-start gap-1.5">
              <span className="text-primary font-bold shrink-0">{ch}</span>
              <span className="text-muted-foreground">{pos}</span>
            </div>
          ))}
        </div>
      </Card>

      <Card className="p-4 border-border/30 space-y-2">
        <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
          Tips for good signal
        </p>
        <ul className="space-y-1.5 text-xs text-muted-foreground">
          {[
            "Wet the ear contacts slightly if your ears are dry",
            "Adjust until forehead sensors sit flat — not angled",
            "Hold still for the first 10 seconds while signal stabilises",
            "Relax your jaw — clenching creates muscle noise",
            "Remove glasses if wearing them across the forehead",
          ].map((tip, i) => (
            <li key={i} className="flex items-start gap-1.5">
              <span className="text-emerald-400 font-bold shrink-0">·</span>
              {tip}
            </li>
          ))}
        </ul>
      </Card>

      <Button className="w-full" onClick={onContinue}>
        <Wifi className="h-4 w-4 mr-2" />
        Connect now
      </Button>
    </div>
  );
}

// ── Step 2: Connecting spinner ───────────────────────────────────────
function StepConnecting({
  deviceName,
  deviceState,
  error,
}: {
  deviceName: string;
  deviceState: DeviceState;
  error: string | null;
}) {
  const isError = !!error && error !== "unreachable" && error !== "Failed to fetch";
  return (
    <div className="space-y-6 flex flex-col items-center text-center py-4">
      <div>
        <h2 className="text-xl font-semibold mb-1">Connecting…</h2>
        <p className="text-sm text-muted-foreground">
          Pairing with <strong>{deviceName}</strong>. Keep it nearby.
        </p>
      </div>

      <div className="relative">
        <div className="w-24 h-24 rounded-full border-2 border-primary/20 flex items-center justify-center">
          {isError
            ? <AlertTriangle className="h-10 w-10 text-destructive" />
            : deviceState === "streaming"
              ? <CheckCircle className="h-10 w-10 text-emerald-400" />
              : <Loader2 className="h-10 w-10 text-primary animate-spin" />
          }
        </div>
        {!isError && deviceState !== "streaming" && (
          <div className="absolute inset-[-8px] rounded-full border border-primary/20 animate-ping opacity-30" />
        )}
      </div>

      <div className="space-y-1">
        {isError ? (
          <>
            <p className="text-sm font-medium text-destructive">Connection failed</p>
            <p className="text-xs text-muted-foreground">{error}</p>
          </>
        ) : deviceState === "streaming" ? (
          <p className="text-sm font-medium text-emerald-400">Connected — checking signal…</p>
        ) : (
          <>
            <p className="text-sm font-medium">
              {deviceState === "connecting" ? "Establishing Bluetooth link…" : "Starting stream…"}
            </p>
            <p className="text-xs text-muted-foreground">This can take up to 15 seconds</p>
          </>
        )}
      </div>
    </div>
  );
}

// ── Step 3: Signal Quality Check ─────────────────────────────────────
function StepSignalQuality({
  quality,
  onAccept,
  onSkip,
}: {
  quality: {
    sqi: number;
    channel_quality: number[];
    artifacts_detected: string[];
  } | null;
  onAccept: () => void;
  onSkip: () => void;
}) {
  const sqi = quality?.sqi ?? 0;
  const channelQ = quality?.channel_quality ?? [];
  const { text: sqiText, color: sqiColor } = sqiLabel(sqi);
  const isPassing = sqi >= 0.50;

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold mb-1">Signal quality check</h2>
        <p className="text-sm text-muted-foreground">
          Checking electrode contact. Adjust the headset until all channels are green.
        </p>
      </div>

      {/* Overall SQI gauge */}
      <Card className="p-5 border-border/30 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Overall signal quality</span>
          <span className={`text-sm font-semibold ${sqiColor}`}>{sqiText}</span>
        </div>
        <Progress
          value={sqi * 100}
          className="h-3"
        />
        <p className="text-xs text-muted-foreground text-center font-mono">
          {Math.round(sqi * 100)}%
          {!quality && " — waiting for signal…"}
        </p>
      </Card>

      {/* Per-channel indicators */}
      {channelQ.length > 0 && (
        <div className="grid grid-cols-2 gap-2">
          {CHANNEL_LABELS.map((label, i) => {
            const q = channelQ[i] ?? 0;
            const pct = Math.round(q * 100);
            const col = channelColor(q);
            return (
              <Card key={i} className="p-3 border-border/30 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono font-semibold">{CHANNEL_SHORT[i]}</span>
                  <span className="text-muted-foreground">{pct}%</span>
                </div>
                <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${col}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <p className="text-[10px] text-muted-foreground leading-tight">{label}</p>
              </Card>
            );
          })}
        </div>
      )}

      {/* Adjustment tips when signal is poor */}
      {!isPassing && quality && (
        <Card className="p-3 border-amber-500/20 bg-amber-500/5 space-y-1.5">
          <p className="text-xs font-semibold text-amber-400">Try these adjustments:</p>
          <ul className="text-xs text-muted-foreground space-y-1">
            <li>· Slide the headband up or down slightly</li>
            <li>· Press each electrode firmly for 3 seconds</li>
            <li>· Relax your jaw and forehead muscles</li>
            <li>· Wet ear contacts if skin is dry</li>
          </ul>
        </Card>
      )}

      {/* Pass banner */}
      {isPassing && (
        <div className="flex items-center gap-2 p-3 rounded-xl border border-emerald-500/30 bg-emerald-500/8 text-sm text-emerald-400">
          <CheckCircle className="h-4 w-4 shrink-0" />
          Signal looks good — you're ready to record.
        </div>
      )}

      <div className="flex flex-col gap-2">
        <Button
          className="w-full"
          disabled={!quality}
          onClick={onAccept}
        >
          <Activity className="h-4 w-4 mr-2" />
          {isPassing ? "Looks good →" : "Continue with current signal →"}
        </Button>
        <button
          onClick={onSkip}
          className="text-xs text-muted-foreground hover:text-foreground transition-colors text-center"
        >
          Skip quality check
        </button>
      </div>
    </div>
  );
}

// ── Step 4: Done ─────────────────────────────────────────────────────
function StepDone({
  deviceName,
  isSynthetic,
  onCalibrate,
  onEmotions,
}: {
  deviceName: string;
  isSynthetic: boolean;
  onCalibrate: () => void;
  onEmotions: () => void;
}) {
  return (
    <div className="space-y-6 flex flex-col items-center text-center">
      <div className="relative">
        <div className="w-24 h-24 rounded-full bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
          <CheckCircle className="h-12 w-12 text-emerald-400" />
        </div>
        <div className="absolute inset-0 rounded-full bg-emerald-500/5 animate-ping" />
      </div>

      <div>
        <h2 className="text-xl font-semibold mb-1">
          {isSynthetic ? "Synthetic stream active" : `${deviceName} connected`}
        </h2>
        <p className="text-sm text-muted-foreground max-w-xs mx-auto">
          {isSynthetic
            ? "Simulated EEG is streaming. All 16 models are running."
            : "Your headset is paired and streaming. Signal quality verified."
          }
        </p>
      </div>

      <div className="w-full space-y-3">
        {!isSynthetic && (
          <Card
            className="p-4 border-primary/20 bg-primary/5 cursor-pointer hover:bg-primary/10 transition-colors text-left"
            onClick={onCalibrate}
          >
            <div className="flex items-start gap-3">
              <Brain className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold">Calibrate baseline</p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  2-minute resting recording — improves emotion accuracy by up to 29%.
                  <span className="text-primary font-medium"> Recommended.</span>
                </p>
              </div>
              <ChevronRight className="h-4 w-4 text-muted-foreground ml-auto shrink-0 mt-0.5" />
            </div>
          </Card>
        )}

        <Button
          variant={isSynthetic ? "default" : "outline"}
          className="w-full"
          onClick={onEmotions}
        >
          <Activity className="h-4 w-4 mr-2" />
          Go to Emotions
        </Button>
      </div>
    </div>
  );
}

// ── Main Wizard Component ────────────────────────────────────────────
export default function DeviceSetup() {
  const [, navigate] = useLocation();
  const device = useDevice();
  const {
    state: deviceState,
    devices,
    devicesLoaded,
    error,
    latestFrame,
    refreshDevices,
    connect,
    disconnect,
  } = device;

  const [step, setStep]                 = useState<Step>(0);
  const [selectedType, setSelectedType] = useState<string>("");
  const [selectedName, setSelectedName] = useState<string>("");

  const isSynthetic = selectedType === "synthetic";
  const TOTAL_STEPS = isSynthetic ? 3 : 5; // synthetic skips placement + quality

  // Fetch device list on mount
  useEffect(() => {
    refreshDevices();
  }, [refreshDevices]);

  // Auto-advance from connecting (step 2) → signal quality (step 3) when stream starts
  useEffect(() => {
    if (step === 2 && deviceState === "streaming") {
      // Give 1 second to receive first WebSocket frame
      const t = setTimeout(() => setStep(3), 1000);
      return () => clearTimeout(t);
    }
  }, [step, deviceState]);

  // Auto-advance quality step for synthetic (no real signal to check)
  useEffect(() => {
    if (step === 3 && isSynthetic) {
      setStep(4);
    }
  }, [step, isSynthetic]);

  // ── Handlers ────────────────────────────────────────────────────
  const handleSelectDevice = async (type: string) => {
    const dev = devices.find((d) => d.type === type);
    setSelectedType(type);
    setSelectedName(dev?.name ?? type);

    if (type === "synthetic") {
      // Skip placement guide, connect immediately
      setStep(2);
      await connect(type);
    } else {
      setStep(1); // Go to placement guide first
    }
  };

  const handlePlacementContinue = async () => {
    setStep(2);
    await connect(selectedType);
  };

  const handleQualityAccept = () => setStep(4);
  const handleQualitySkip   = () => setStep(4);

  const handleBack = () => {
    if (step === 1) { disconnect(); setStep(0); }
    else if (step === 2) { disconnect(); setStep(isSynthetic ? 0 : 1); }
    else if (step > 2) setStep((step - 1) as Step);
  };

  // Quality from latest WebSocket frame
  const quality = latestFrame?.quality
    ? {
        sqi: (latestFrame.quality as { sqi: number; channel_quality: number[]; artifacts_detected: string[] }).sqi,
        channel_quality: (latestFrame.quality as { sqi: number; channel_quality: number[]; artifacts_detected: string[] }).channel_quality ?? [],
        artifacts_detected: (latestFrame.quality as { sqi: number; channel_quality: number[]; artifacts_detected: string[] }).artifacts_detected ?? [],
      }
    : null;

  // ── Render ───────────────────────────────────────────────────────
  return (
    <main className="min-h-[calc(100vh-4rem)] flex items-center justify-center p-6">
      <div className="w-full max-w-md space-y-6">

        {/* Top bar */}
        <div className="flex items-center gap-3">
          {step > 0 && step < 4 && (
            <button
              onClick={handleBack}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <ArrowLeft className="h-5 w-5" />
            </button>
          )}
          <div className="flex-1">
            <WizardProgress step={step} totalSteps={TOTAL_STEPS} />
          </div>
          <button
            onClick={() => navigate("/emotions")}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Skip
          </button>
        </div>

        {/* Step content */}
        {step === 0 && (
          <StepSelectDevice
            devices={devices}
            devicesLoaded={devicesLoaded}
            error={error}
            onSelect={handleSelectDevice}
            onRefresh={refreshDevices}
          />
        )}

        {step === 1 && (
          <StepPlacement
            deviceName={selectedName}
            onContinue={handlePlacementContinue}
          />
        )}

        {step === 2 && (
          <StepConnecting
            deviceName={selectedName}
            deviceState={deviceState}
            error={error && error !== "unreachable" ? error : null}
          />
        )}

        {step === 3 && (
          <StepSignalQuality
            quality={quality}
            onAccept={handleQualityAccept}
            onSkip={handleQualitySkip}
          />
        )}

        {step === 4 && (
          <StepDone
            deviceName={selectedName}
            isSynthetic={isSynthetic}
            onCalibrate={() => navigate("/calibration")}
            onEmotions={() => navigate("/emotions")}
          />
        )}
      </div>
    </main>
  );
}
