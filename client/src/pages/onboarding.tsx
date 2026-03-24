/**
 * Unified Onboarding — single state machine for all first-time user setup.
 *
 * Steps (persisted to localStorage key `ndw_onboarding_step`):
 *   1. Welcome        — brief app intro, what NDW does
 *   2. Choose path    — "Quick Start (Voice)" or "Full Setup (EEG)"
 *   3a. Voice analysis (Quick Start path) — 10-second voice emotion scan
 *   3b. EEG calibration (Full Setup path) — 2-min resting baseline
 *   4. Health sync    — connect Apple Health / Google Fit (skippable)
 *   5. Done           — marks onboarding complete, redirects to dashboard
 *
 * localStorage keys:
 *   ndw_onboarding_step     — last step reached (1–5), allows resume
 *   ndw_onboarding_path     — "voice" | "eeg" — chosen path
 *   ndw_onboarding_complete — set to "true" on step 5
 */

import { useEffect, useRef, useState } from "react";
import { useLocation } from "wouter";
import { motion, AnimatePresence } from "framer-motion";
import { FastOnboarding } from "@/components/fast-onboarding";
import {
  Brain,
  CheckCircle,
  CheckCircle2,
  ChevronRight,
  Heart,
  Loader2,
  Mic,
  Sparkles,
  Watch,
  Waves,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useDevice } from "@/hooks/use-device";
import { useVoiceEmotion } from "@/hooks/use-voice-emotion";
import { useAuth } from "@/hooks/use-auth";
import {
  addBaselineFrame,
  getBaselineStatus,
  simulateEEG,
} from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";
import { hapticLight, hapticSuccess } from "@/lib/haptics";
import { playSuccessChime } from "@/lib/sound-effects";
import { sbGetSetting, sbRemoveSetting, sbSaveSetting } from "../lib/supabase-store";

// ── Constants ────────────────────────────────────────────────────────────────

const USER_ID = getParticipantId();
const FS = 256;
const TARGET_FRAMES = 120; // 2 minutes
const MIN_FRAMES = 30;     // earliest early-exit allowed

const TOTAL_STEPS = 5;

type OnboardingStep = 1 | 2 | 3 | 4 | 5;
type PathChoice = "voice" | "eeg" | null;

// EEG recording sub-phases
type EegPhase = "intro" | "recording" | "done";

// ── localStorage helpers ─────────────────────────────────────────────────────

function loadStep(): OnboardingStep {
  try {
    const v = sbGetSetting("ndw_onboarding_step");
    const n = v ? parseInt(v, 10) : 1;
    if (n >= 1 && n <= TOTAL_STEPS) return n as OnboardingStep;
  } catch {}
  return 1;
}

function saveStep(step: OnboardingStep) {
  try {
    sbSaveSetting("ndw_onboarding_step", String(step));
  } catch {}
}

function loadPath(): PathChoice {
  try {
    const v = sbGetSetting("ndw_onboarding_path");
    if (v === "voice" || v === "eeg") return v;
  } catch {}
  return null;
}

function savePath(path: PathChoice) {
  try {
    if (path) sbSaveSetting("ndw_onboarding_path", path);
  } catch {}
}

function markComplete() {
  try {
    sbSaveSetting("ndw_onboarding_complete", "true");
    sbRemoveSetting("ndw_onboarding_step");
    sbRemoveSetting("ndw_onboarding_path");
  } catch {}
}

// ── Progress bar ──────────────────────────────────────────────────────────────

function ProgressBar({ step }: { step: OnboardingStep }) {
  const pct = Math.round((step / TOTAL_STEPS) * 100);
  return (
    <div className="w-full space-y-1.5">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span aria-live="polite">Step {step} of {TOTAL_STEPS}</span>
        <span>{pct}%</span>
      </div>
      <div
        role="progressbar"
        aria-label={`Onboarding progress: step ${step} of ${TOTAL_STEPS}`}
        aria-valuenow={step}
        aria-valuemin={1}
        aria-valuemax={TOTAL_STEPS}
        className="h-1.5 w-full rounded-full bg-muted overflow-hidden"
      >
        <div
          className="h-full bg-primary rounded-full transition-all duration-500 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ── EEG ring progress (for step 3b) ─────────────────────────────────────────

function RingProgress({ pct }: { pct: number }) {
  const R = 72;
  const circ = 2 * Math.PI * R;
  const offset = circ * (1 - Math.min(pct / 100, 1));
  return (
    <svg width="180" height="180" className="rotate-[-90deg]">
      <circle
        cx="90" cy="90" r={R} fill="none"
        stroke="hsl(var(--muted)/0.2)" strokeWidth="10"
      />
      <circle
        cx="90" cy="90" r={R} fill="none"
        stroke="hsl(var(--primary))"
        strokeWidth="10"
        strokeLinecap="round"
        strokeDasharray={circ}
        strokeDashoffset={offset}
        style={{ transition: "stroke-dashoffset 0.6s ease" }}
      />
    </svg>
  );
}

// ── Step 1: Welcome ──────────────────────────────────────────────────────────

function StepWelcome({ onNext }: { onNext: () => void }) {
  return (
    <div className="space-y-6">
      <div className="text-center space-y-4">
        <div className="inline-flex items-center justify-center h-20 w-20 rounded-full bg-primary/10 mx-auto">
          <img src="/logo-antarai.svg" alt="AntarAI" className="h-10 w-10" />
        </div>
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold">Welcome to AntarAI</h1>
          <p className="text-sm text-muted-foreground leading-relaxed max-w-sm mx-auto">
            Your personal brain-computer interface for real-time emotion reading,
            dream journaling, neurofeedback, and wellness insights.
          </p>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        {[
          { icon: Mic, label: "Voice emotions", desc: "10-second analysis" },
          { icon: Brain, label: "EEG analysis", desc: "Live neural signals" },
          { icon: Heart, label: "Health sync", desc: "Apple Health · Google Fit" },
        ].map(({ icon: Icon, label, desc }) => (
          <Card key={label} className="glass-card p-4 space-y-2">
            <Icon className="h-5 w-5 text-primary" />
            <p className="text-sm font-medium">{label}</p>
            <p className="text-xs text-muted-foreground">{desc}</p>
          </Card>
        ))}
      </div>

      <Button className="w-full" onClick={() => { hapticLight(); onNext(); }}>
        Get started
        <ChevronRight className="ml-1 h-4 w-4" />
      </Button>
    </div>
  );
}

// ── Step 2: Choose path ──────────────────────────────────────────────────────

function StepChoosePath({ onChoose }: { onChoose: (path: "voice" | "eeg") => void }) {
  return (
    <div className="space-y-6">
      <div className="text-center space-y-3">
        <h1 className="text-2xl font-semibold">How do you want to start?</h1>
        <p className="text-sm text-muted-foreground">
          Choose your setup path. You can always add more later.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Card className="glass-card p-5 space-y-4 border-primary/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Mic className="h-5 w-5 text-primary" />
              <h2 className="text-base font-semibold">Quick Start</h2>
            </div>
            <Badge>Recommended</Badge>
          </div>
          <p className="text-sm text-muted-foreground">
            Voice + watch. No headset required. Start measuring emotions right now.
          </p>
          <ul className="text-xs space-y-1.5 text-foreground/80">
            <li className="flex items-center gap-2">
              <CheckCircle2 className="h-3.5 w-3.5 text-cyan-400 shrink-0" />
              10-second voice analysis
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle2 className="h-3.5 w-3.5 text-cyan-400 shrink-0" />
              Works on any device
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle2 className="h-3.5 w-3.5 text-cyan-400 shrink-0" />
              EEG optional upgrade later
            </li>
          </ul>
          <Button className="w-full" onClick={() => { hapticLight(); onChoose("voice"); }}>
            Quick Start (Voice)
            <ChevronRight className="ml-1 h-4 w-4" />
          </Button>
        </Card>

        <Card className="glass-card p-5 space-y-4">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            <h2 className="text-base font-semibold">Full Setup</h2>
          </div>
          <p className="text-sm text-muted-foreground">
            Muse 2 EEG. Live neural signals from day one.
          </p>
          <ul className="text-xs space-y-1.5 text-foreground/80">
            <li className="flex items-center gap-2">
              <CheckCircle2 className="h-3.5 w-3.5 text-primary/70 shrink-0" />
              2-minute resting baseline
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle2 className="h-3.5 w-3.5 text-primary/70 shrink-0" />
              Live stress, focus, relaxation
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle2 className="h-3.5 w-3.5 text-primary/70 shrink-0" />
              Neurofeedback training
            </li>
          </ul>
          <Button variant="outline" className="w-full" onClick={() => { hapticLight(); onChoose("eeg"); }}>
            Full Setup (EEG)
          </Button>
        </Card>
      </div>
    </div>
  );
}

// ── Step 3a: Voice analysis ──────────────────────────────────────────────────

function StepVoice({ onNext }: { onNext: () => void }) {
  const voiceEmotion = useVoiceEmotion({
    durationMs: 10000,
    userId: USER_ID,
  });

  // Auto-advance when result is ready
  useEffect(() => {
    if (voiceEmotion.lastResult) {
      // Small delay so user can see the result card
      const t = setTimeout(onNext, 2000);
      return () => clearTimeout(t);
    }
  }, [voiceEmotion.lastResult, onNext]);

  if (voiceEmotion.lastResult) {
    const r = voiceEmotion.lastResult;
    return (
      <div className="space-y-6">
        <div className="text-center space-y-3">
          <Badge variant="outline" className="border-primary/40 text-primary">
            Analysis complete
          </Badge>
          <h1 className="text-2xl font-semibold">Your first state read is ready</h1>
        </div>

        <Card className="glass-card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Emotion</span>
            <span className="font-semibold capitalize">{r.emotion}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Confidence</span>
            <span className="font-mono">{Math.round(r.confidence * 100)}%</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Valence</span>
            <span className="font-mono">
              {r.valence >= 0 ? "+" : ""}{r.valence.toFixed(2)}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Model</span>
            <span className="font-mono text-xs">{r.model_type}</span>
          </div>
        </Card>

        <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" />
          Continuing…
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="text-center space-y-3">
        <div className="inline-flex items-center justify-center h-16 w-16 rounded-full bg-cyan-400/10 mx-auto">
          <Mic className="h-8 w-8 text-cyan-400" />
        </div>
        <h1 className="text-2xl font-semibold">Voice Analysis</h1>
        <p className="text-sm text-muted-foreground">
          A 10-second recording to read your current emotional state.
        </p>
      </div>

      <Card className="glass-card p-5 space-y-3">
        <div className="flex items-start gap-3">
          <Mic className="h-5 w-5 text-primary shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-sm">Voice + Health pipeline</p>
            <p className="text-xs text-muted-foreground">
              Uses the live voice-watch pipeline. Results are cached for
              downstream pages.
            </p>
          </div>
        </div>
        <div className="flex items-start gap-3">
          <Watch className="h-5 w-5 text-primary shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-sm">Health data when available</p>
            <p className="text-xs text-muted-foreground">
              Wearable biometrics refine stress, recovery, and readiness over
              time.
            </p>
          </div>
        </div>
      </Card>

      <div className="space-y-3">
        <Button
          className="w-full"
          onClick={() => { hapticLight(); voiceEmotion.startRecording(); }}
          disabled={voiceEmotion.isRecording || voiceEmotion.isAnalyzing}
        >
          {voiceEmotion.isRecording
            ? "Recording…"
            : voiceEmotion.isAnalyzing
            ? "Analyzing…"
            : "Start 10-second voice analysis"}
        </Button>
        <Button variant="outline" className="w-full" onClick={() => { hapticLight(); onNext(); }}>
          Skip for now
        </Button>
        {voiceEmotion.error && (
          <p className="text-sm text-destructive text-center">
            {voiceEmotion.error}
          </p>
        )}
      </div>
    </div>
  );
}

// ── Step 3b: EEG calibration ─────────────────────────────────────────────────

function StepEeg({ onNext }: { onNext: () => void }) {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";

  const [eegPhase, setEegPhase] = useState<EegPhase>("intro");
  const [nFrames, setNFrames] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Check if already calibrated
  useEffect(() => {
    getBaselineStatus(USER_ID)
      .then((s) => {
        if (s.ready) {
          setNFrames(s.n_frames);
          setEegPhase("done");
        }
      })
      .catch(() => {});
  }, []);

  // Recording loop
  useEffect(() => {
    if (eegPhase !== "recording") return;

    intervalRef.current = setInterval(async () => {
      try {
        let signals: number[][];
        if (isStreaming && latestFrame?.analysis) {
          const raw = (latestFrame as any)?.signals as number[][] | undefined;
          signals = raw ?? [[...Array(256)].map(() => Math.random() * 10 - 5)];
        } else {
          const sim = await simulateEEG("rest", 1, FS, 4);
          signals = sim.signals ?? [];
        }
        const result = await addBaselineFrame(signals, USER_ID, FS);
        setNFrames(result.n_frames);
        if (result.ready || result.n_frames >= TARGET_FRAMES) {
          clearInterval(intervalRef.current!);
          setEegPhase("done");
        }
      } catch {
        setError("ML backend offline — make sure it's running on port 8000.");
        clearInterval(intervalRef.current!);
      }
    }, 1000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [eegPhase, isStreaming]);

  const progress = Math.round((nFrames / TARGET_FRAMES) * 100);
  const remaining = Math.max(0, TARGET_FRAMES - nFrames);
  const canEarlyExit = nFrames >= MIN_FRAMES;

  if (eegPhase === "intro") {
    return (
      <div className="space-y-6">
        <div className="text-center space-y-3">
          <div className="inline-flex items-center justify-center h-16 w-16 rounded-full bg-primary/10 mx-auto">
            <Brain className="h-8 w-8 text-primary" />
          </div>
          <h1 className="text-2xl font-semibold">EEG Baseline Calibration</h1>
          <p className="text-sm text-muted-foreground">
            Muse 2 only · 2 minutes · Done once
          </p>
        </div>

        <Card className="glass-card p-5 space-y-3">
          <p className="text-sm leading-relaxed text-foreground/90">
            Records your personal resting-state EEG so neural features are
            measured relative to <em>you</em> instead of a population average.
            Improves accuracy by +15–29%.
          </p>
          <p className="text-sm font-medium text-primary">
            What you'll do: sit still, close your eyes, breathe naturally for 2 minutes.
          </p>
          {isStreaming && (
            <p className="text-xs text-cyan-400 font-medium">
              Muse 2 connected — will use live EEG
            </p>
          )}
          {!isStreaming && (
            <p className="text-xs text-muted-foreground">
              No headset detected — simulation mode will preview the flow
            </p>
          )}
        </Card>

        <div className="space-y-3">
          <Button className="w-full" onClick={() => { hapticLight(); setEegPhase("recording"); }}>
            Start calibration
            <ChevronRight className="ml-1 h-4 w-4" />
          </Button>
          <Button variant="outline" className="w-full" onClick={() => { hapticLight(); onNext(); }}>
            Skip for now
          </Button>
        </div>
      </div>
    );
  }

  if (eegPhase === "recording") {
    return (
      <div className="space-y-6 text-center">
        <h1 className="text-xl font-semibold">Recording your baseline…</h1>

        <div className="relative flex items-center justify-center mx-auto w-48 h-48">
          <div
            className="absolute inset-0 rounded-full bg-primary/10 animate-pulse"
            style={{ animationDuration: "4s" }}
          />
          <RingProgress pct={progress} />
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-2xl font-bold tabular-nums">
              {Math.floor(remaining / 60)}:
              {String(remaining % 60).padStart(2, "0")}
            </span>
            <span className="text-xs text-muted-foreground mt-1">remaining</span>
          </div>
        </div>

        <Card className="glass-card p-4 space-y-1">
          <p className="text-sm font-medium">Eyes closed · Breathe naturally</p>
          <p className="text-xs text-muted-foreground">
            {nFrames} / {TARGET_FRAMES} frames · {progress}% complete
          </p>
          {!isStreaming && (
            <p className="text-xs text-amber-400/80 pt-1">
              Simulation mode — no headset needed
            </p>
          )}
        </Card>

        {error && <p className="text-xs text-rose-400">{error}</p>}

        {canEarlyExit && (
          <button
            onClick={() => {
              clearInterval(intervalRef.current!);
              setEegPhase("done");
            }}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Done — use readings now ({nFrames} frames collected)
          </button>
        )}

        {nFrames === 0 && (
          <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" />
            Connecting…
          </div>
        )}
      </div>
    );
  }

  // done phase
  return (
    <div className="space-y-6 text-center">
      <div className="inline-flex items-center justify-center h-16 w-16 rounded-full bg-cyan-400/10 mx-auto">
        <CheckCircle className="h-8 w-8 text-cyan-400" />
      </div>
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold">Baseline saved</h1>
        <p className="text-sm text-muted-foreground">
          Your personal EEG profile has been created
          {nFrames > 0 ? ` (${nFrames} frames)` : ""}.
        </p>
      </div>
      <Card className="glass-card p-4">
        <p className="text-sm text-foreground/80">
          Emotion readings are now calibrated to your brain. Accuracy improves
          further with each session.
        </p>
      </Card>

      <div className="grid gap-3 sm:grid-cols-2">
        <Card className="p-4 border-primary/20">
          <div className="flex items-start gap-3">
            <Sparkles className="h-4 w-4 text-primary mt-0.5" />
            <div>
              <p className="text-sm font-medium">EEG calibrated</p>
              <p className="text-xs text-muted-foreground">
                Live neural features unlocked.
              </p>
            </div>
          </div>
        </Card>
        <Card className="p-4 border-primary/20">
          <div className="flex items-start gap-3">
            <Waves className="h-4 w-4 text-primary mt-0.5" />
            <div>
              <p className="text-sm font-medium">+15–29% accuracy</p>
              <p className="text-xs text-muted-foreground">
                Calibrated vs population average.
              </p>
            </div>
          </div>
        </Card>
      </div>

      <Button className="w-full" onClick={() => { hapticLight(); onNext(); }}>
        Continue
        <ChevronRight className="ml-1 h-4 w-4" />
      </Button>
    </div>
  );
}

// ── Step 4: Health sync ──────────────────────────────────────────────────────

function StepHealthSync({ onNext }: { onNext: () => void }) {
  const [status, setStatus] = useState<
    "idle" | "connecting" | "connected" | "error"
  >("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  async function connectHealth() {
    setStatus("connecting");
    setErrorMsg(null);
    try {
      const { Capacitor } = await import("@capacitor/core");
      const platform = Capacitor.getPlatform();

      if (platform === "ios") {
        const { CapacitorHealthkit } = await import(
          "@perfood/capacitor-healthkit"
        );
        await CapacitorHealthkit.requestAuthorization({
          all: [],
          read: [
            "heartRate",
            "restingHeartRate",
            "respiratoryRate",
            "oxygenSaturation",
            "sleepAnalysis",
            "stepCount",
            "activeEnergyBurned",
          ],
          write: [],
        });
        setStatus("connected");
      } else if (platform === "android") {
        const capgoModule = await import("@capgo/capacitor-health"); const Health = capgoModule.Health;
        const available = await Health.isAvailable();
        if (!available.available) {
          setErrorMsg(
            "Google Health Connect is not installed. Install it from the Play Store to continue."
          );
          setStatus("error");
          return;
        }
        await Health.requestAuthorization({
          read: ["steps", "heartRate", "calories", "mindfulness"],
        });
        setStatus("connected");
      } else {
        setErrorMsg(
          "Health data sync is available on mobile devices. You can connect later from Settings."
        );
        setStatus("error");
      }
    } catch (e) {
      setErrorMsg(`Permission denied or unavailable: ${String(e)}`);
      setStatus("error");
    }
  }

  return (
    <div className="space-y-6">
      <div className="text-center space-y-4">
        <div className="inline-flex items-center justify-center h-16 w-16 rounded-full bg-rose-500/10 mx-auto">
          <Heart className="h-8 w-8 text-rose-400" />
        </div>
        <div className="space-y-2">
          <h1 className="text-2xl font-semibold">Connect Health Data</h1>
          <p className="text-sm text-muted-foreground leading-relaxed max-w-sm mx-auto">
            Sync steps, heart rate, and sleep from Apple Health or Google Health
            Connect to enrich your wellness insights.
          </p>
        </div>
      </div>

      <Card className="glass-card p-5 space-y-3">
        <p className="text-xs text-muted-foreground uppercase tracking-wide">
          What we sync
        </p>
        <ul className="grid grid-cols-2 gap-2 text-sm">
          {["Steps", "Heart rate", "Sleep stages", "Active energy"].map(
            (item) => (
              <li key={item} className="flex items-center gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-cyan-400 shrink-0" />
                {item}
              </li>
            )
          )}
        </ul>
      </Card>

      {status === "connected" && (
        <div className="flex items-center justify-center gap-2 text-cyan-400 text-sm font-medium">
          <CheckCircle2 className="h-4 w-4" />
          Connected successfully
        </div>
      )}

      {errorMsg && (
        <p className="text-xs text-muted-foreground text-center">{errorMsg}</p>
      )}

      <div className="space-y-3">
        {status !== "connected" && (
          <Button
            className="w-full"
            onClick={connectHealth}
            disabled={status === "connecting"}
          >
            {status === "connecting" ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Connecting…
              </>
            ) : (
              "Connect Health Data"
            )}
          </Button>
        )}
        <Button
          variant={status === "connected" ? "default" : "outline"}
          className="w-full"
          onClick={onNext}
        >
          {status === "connected" ? (
            <>
              Continue
              <ChevronRight className="ml-1 h-4 w-4" />
            </>
          ) : (
            "Skip for now"
          )}
        </Button>
      </div>
    </div>
  );
}

// ── Step 5: Done ─────────────────────────────────────────────────────────────

function StepDone({ pathChoice, onFinish }: { pathChoice: PathChoice; onFinish: () => void }) {
  const [confettiPieces] = useState(() =>
    Array.from({ length: 25 }, (_, i) => ({
      id: i,
      color: ["#0891b2", "#0e7490", "#7c3aed", "#d4a017", "#d946ef"][i % 5],
      size: 6 + Math.random() * 6,
      angle: (i / 25) * 360 + Math.random() * 15,
      distance: 80 + Math.random() * 120,
      duration: 1.2 + Math.random() * 0.6,
      shape: i % 3 === 0 ? "50%" : "2px",
    }))
  );

  useEffect(() => {
    hapticSuccess();
    playSuccessChime();
  }, []);

  return (
    <div className="space-y-6 text-center">
      {/* Confetti burst */}
      <div
        className="pointer-events-none fixed inset-0 z-50 flex items-center justify-center"
        aria-hidden="true"
      >
        {confettiPieces.map((p) => (
          <span
            key={p.id}
            style={{
              position: "absolute",
              width: p.size,
              height: p.size,
              borderRadius: p.shape,
              backgroundColor: p.color,
              opacity: 0,
              animation: `confetti-burst ${p.duration}s ease-out forwards`,
              ["--confetti-x" as string]: `${Math.cos((p.angle * Math.PI) / 180) * p.distance}px`,
              ["--confetti-y" as string]: `${Math.sin((p.angle * Math.PI) / 180) * p.distance}px`,
            }}
          />
        ))}
        <style>{`
          @keyframes confetti-burst {
            0% {
              opacity: 1;
              transform: translate(0, 0) scale(1) rotate(0deg);
            }
            70% {
              opacity: 1;
            }
            100% {
              opacity: 0;
              transform: translate(var(--confetti-x), var(--confetti-y)) scale(0.3) rotate(360deg);
            }
          }
        `}</style>
      </div>

      <div className="inline-flex items-center justify-center h-20 w-20 rounded-full bg-cyan-400/10 mx-auto">
        <CheckCircle className="h-10 w-10 text-cyan-400" />
      </div>

      <div className="space-y-2">
        <h1 className="text-3xl font-semibold">You're all set</h1>
        <p className="text-sm text-muted-foreground">
          {pathChoice === "eeg"
            ? "Your EEG baseline is saved. Neural features are live."
            : "Voice analysis is ready. You can add EEG anytime from Settings."}
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        {[
          { icon: Mic, label: "Voice analysis", desc: "Daily mood tracking" },
          { icon: Brain, label: "Brain monitor", desc: "Live EEG waveforms" },
          { icon: Heart, label: "Health analytics", desc: "Trends and insights" },
        ].map(({ icon: Icon, label, desc }) => (
          <Card key={label} className="glass-card p-4 space-y-2">
            <Icon className="h-5 w-5 text-primary" />
            <p className="text-sm font-medium">{label}</p>
            <p className="text-xs text-muted-foreground">{desc}</p>
          </Card>
        ))}
      </div>

      <Button className="w-full" size="lg" onClick={() => { hapticSuccess(); onFinish(); }}>
        Go to dashboard
        <ChevronRight className="ml-1 h-4 w-4" />
      </Button>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function Onboarding() {
  const [, navigate] = useLocation();
  const { user } = useAuth();

  const [step, setStep] = useState<OnboardingStep>(loadStep);
  const [pathChoice, setPathChoice] = useState<PathChoice>(loadPath);

  // Persist step on change
  useEffect(() => {
    saveStep(step);
  }, [step]);

  function goNext() {
    setStep((s) => {
      const next = Math.min(s + 1, TOTAL_STEPS) as OnboardingStep;
      return next;
    });
  }

  function chooseAndAdvance(path: "voice" | "eeg") {
    savePath(path);
    setPathChoice(path);
    goNext();
  }

  function finish() {
    markComplete();
    navigate(user ? "/" : "/auth");
  }

  // Fast-track: show the 3-screen flow for brand-new users who haven't
  // chosen a path yet. If they're resuming mid-flow (step > 1 or path
  // already chosen), fall through to the full 5-step state machine.
  const isFreshStart = step === 1 && pathChoice === null;
  if (isFreshStart) {
    return (
      <FastOnboarding
        onComplete={() => {
          markComplete();
          navigate(user ? "/" : "/auth");
        }}
      />
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-background">
      <div className="max-w-lg w-full space-y-8">
        {/* Progress bar */}
        <ProgressBar step={step} />

        {/* Step content — slide transition between steps */}
        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.25 }}
          >
            {step === 1 && <StepWelcome onNext={goNext} />}

            {step === 2 && <StepChoosePath onChoose={chooseAndAdvance} />}

            {step === 3 && pathChoice === "voice" && (
              <StepVoice onNext={goNext} />
            )}

            {step === 3 && pathChoice === "eeg" && (
              <StepEeg onNext={goNext} />
            )}

            {step === 3 && pathChoice === null && (
              <StepVoice onNext={goNext} />
            )}

            {step === 4 && <StepHealthSync onNext={goNext} />}

            {step === 5 && (
              <StepDone pathChoice={pathChoice} onFinish={finish} />
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
