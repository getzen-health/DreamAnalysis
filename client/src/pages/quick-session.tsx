/**
 * Quick 5-Minute Session — accessible from Today page.
 * Works great even without EEG headband.
 *
 * Steps:
 *   1. Voice check-in (30 sec)
 *   2. Guided breathing (2 min)
 *   3. Brief meditation (2 min) — timer with calming UI
 *   4. Results summary
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation } from "wouter";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, Wind, Brain, BarChart3, ChevronRight, Pause, Play, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { hapticLight, hapticSuccess } from "@/lib/haptics";

type SessionStep = "voice" | "breathing" | "meditation" | "results";

// ── Step progress ─────────────────────────────────────────────────────────

const STEPS: Array<{ key: SessionStep; label: string; icon: typeof Mic; duration: string }> = [
  { key: "voice", label: "Voice check-in", icon: Mic, duration: "30 sec" },
  { key: "breathing", label: "Guided breathing", icon: Wind, duration: "2 min" },
  { key: "meditation", label: "Meditation", icon: Brain, duration: "2 min" },
  { key: "results", label: "Summary", icon: BarChart3, duration: "" },
];

function StepProgress({ current }: { current: SessionStep }) {
  const currentIdx = STEPS.findIndex((s) => s.key === current);
  return (
    <div className="flex items-center gap-2 justify-center mb-6">
      {STEPS.map((step, i) => (
        <div key={step.key} className="flex items-center gap-2">
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold transition-all ${
              i < currentIdx
                ? "bg-primary text-primary-foreground"
                : i === currentIdx
                ? "bg-primary/20 text-primary ring-2 ring-primary/40"
                : "bg-muted text-muted-foreground"
            }`}
          >
            {i < currentIdx ? (
              <Check className="h-4 w-4" />
            ) : (
              i + 1
            )}
          </div>
          {i < STEPS.length - 1 && (
            <div
              className={`w-8 h-0.5 ${
                i < currentIdx ? "bg-primary" : "bg-muted"
              }`}
            />
          )}
        </div>
      ))}
    </div>
  );
}

// ── Timer display ─────────────────────────────────────────────────────────

function TimerRing({
  secondsLeft,
  totalSeconds,
  label,
}: {
  secondsLeft: number;
  totalSeconds: number;
  label: string;
}) {
  const R = 60;
  const circ = 2 * Math.PI * R;
  const progress = 1 - secondsLeft / totalSeconds;
  const offset = circ * (1 - progress);

  const minutes = Math.floor(secondsLeft / 60);
  const seconds = secondsLeft % 60;

  return (
    <div className="flex flex-col items-center gap-3">
      <svg width="150" height="150" className="rotate-[-90deg]">
        <circle
          cx="75"
          cy="75"
          r={R}
          fill="none"
          stroke="hsl(var(--muted) / 0.3)"
          strokeWidth="6"
        />
        <circle
          cx="75"
          cy="75"
          r={R}
          fill="none"
          stroke="hsl(var(--primary))"
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 1s linear" }}
        />
      </svg>
      <div className="absolute flex flex-col items-center justify-center" style={{ marginTop: 45 }}>
        <span className="text-3xl font-bold tabular-nums">
          {minutes}:{String(seconds).padStart(2, "0")}
        </span>
        <span className="text-xs text-muted-foreground mt-1">{label}</span>
      </div>
    </div>
  );
}

// ── Step 1: Voice Check-in ────────────────────────────────────────────────

function VoiceStep({ onComplete }: { onComplete: () => void }) {
  const [secondsLeft, setSecondsLeft] = useState(30);
  const [recording, setRecording] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  function startRecording() {
    setRecording(true);
    hapticLight();
    intervalRef.current = setInterval(() => {
      setSecondsLeft((s) => {
        if (s <= 1) {
          clearInterval(intervalRef.current!);
          hapticSuccess();
          onComplete();
          return 0;
        }
        return s - 1;
      });
    }, 1000);
  }

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <div className="text-center space-y-6">
      <div className="space-y-2">
        <h2 className="text-xl font-semibold">Voice Check-in</h2>
        <p className="text-sm text-muted-foreground">
          Speak naturally about how you're feeling. The AI analyzes your voice patterns.
        </p>
      </div>

      <div className="relative flex items-center justify-center" style={{ height: 150 }}>
        <TimerRing secondsLeft={secondsLeft} totalSeconds={30} label={recording ? "Recording..." : "Ready"} />
      </div>

      {!recording && (
        <Button onClick={startRecording} className="w-full">
          <Mic className="h-4 w-4 mr-2" />
          Start recording
        </Button>
      )}

      {recording && (
        <div className="flex items-center justify-center gap-2 text-sm text-primary">
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          Listening...
        </div>
      )}
    </div>
  );
}

// ── Step 2: Guided Breathing ──────────────────────────────────────────────

function BreathingStep({ onComplete }: { onComplete: () => void }) {
  const TOTAL = 120; // 2 minutes
  const [secondsLeft, setSecondsLeft] = useState(TOTAL);
  const [started, setStarted] = useState(false);
  const [paused, setPaused] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const cycleRef = useRef(0);

  // Breathing phases: 5s inhale, 5s exhale (coherence breathing)
  const cycleSec = 10;
  const elapsed = TOTAL - secondsLeft;
  const phaseInCycle = elapsed % cycleSec;
  const isInhale = phaseInCycle < 5;
  const phaseLabel = isInhale ? "Breathe in" : "Breathe out";

  // Scale for breathing circle animation
  const phaseProgress = isInhale ? phaseInCycle / 5 : 1 - (phaseInCycle - 5) / 5;
  const scale = 0.6 + phaseProgress * 0.4;

  function start() {
    setStarted(true);
    hapticLight();
    intervalRef.current = setInterval(() => {
      setSecondsLeft((s) => {
        if (s <= 1) {
          clearInterval(intervalRef.current!);
          hapticSuccess();
          onComplete();
          return 0;
        }
        return s - 1;
      });
    }, 1000);
  }

  function togglePause() {
    if (paused) {
      intervalRef.current = setInterval(() => {
        setSecondsLeft((s) => {
          if (s <= 1) {
            clearInterval(intervalRef.current!);
            hapticSuccess();
            onComplete();
            return 0;
          }
          return s - 1;
        });
      }, 1000);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    setPaused(!paused);
  }

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <div className="text-center space-y-6">
      <div className="space-y-2">
        <h2 className="text-xl font-semibold">Guided Breathing</h2>
        <p className="text-sm text-muted-foreground">
          Coherence breathing: 5 seconds in, 5 seconds out. Syncs your heart and brain.
        </p>
      </div>

      {started && (
        <>
          <div className="flex flex-col items-center gap-4">
            {/* Breathing circle */}
            <div
              className="rounded-full bg-primary/10 flex items-center justify-center"
              style={{
                width: 140,
                height: 140,
                transform: `scale(${scale})`,
                transition: "transform 0.8s ease-in-out",
              }}
            >
              <div
                className="rounded-full bg-primary/20 flex items-center justify-center"
                style={{ width: 100, height: 100 }}
              >
                <span className="text-lg font-semibold text-primary">
                  {phaseLabel}
                </span>
              </div>
            </div>

            <div className="text-sm text-muted-foreground tabular-nums">
              {Math.floor(secondsLeft / 60)}:{String(secondsLeft % 60).padStart(2, "0")} remaining
            </div>
          </div>

          <div className="flex gap-3 justify-center">
            <Button variant="outline" size="sm" onClick={togglePause}>
              {paused ? <Play className="h-4 w-4 mr-1" /> : <Pause className="h-4 w-4 mr-1" />}
              {paused ? "Resume" : "Pause"}
            </Button>
            <Button variant="ghost" size="sm" onClick={() => { if (intervalRef.current) clearInterval(intervalRef.current); onComplete(); }}>
              Skip
            </Button>
          </div>
        </>
      )}

      {!started && (
        <Button onClick={start} className="w-full">
          <Wind className="h-4 w-4 mr-2" />
          Start breathing exercise
        </Button>
      )}
    </div>
  );
}

// ── Step 3: Meditation ────────────────────────────────────────────────────

function MeditationStep({ onComplete }: { onComplete: () => void }) {
  const TOTAL = 120; // 2 minutes
  const [secondsLeft, setSecondsLeft] = useState(TOTAL);
  const [started, setStarted] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  function start() {
    setStarted(true);
    hapticLight();
    intervalRef.current = setInterval(() => {
      setSecondsLeft((s) => {
        if (s <= 1) {
          clearInterval(intervalRef.current!);
          hapticSuccess();
          onComplete();
          return 0;
        }
        return s - 1;
      });
    }, 1000);
  }

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <div className="text-center space-y-6">
      <div className="space-y-2">
        <h2 className="text-xl font-semibold">Quiet Meditation</h2>
        <p className="text-sm text-muted-foreground">
          Close your eyes. Focus on your breath. Let thoughts come and go without judgment.
        </p>
      </div>

      {started && (
        <>
          <div className="flex flex-col items-center gap-4">
            {/* Calming pulse animation */}
            <div className="relative" style={{ width: 140, height: 140 }}>
              <div
                className="absolute inset-0 rounded-full bg-violet-500/10 animate-pulse"
                style={{ animationDuration: "4s" }}
              />
              <div
                className="absolute inset-4 rounded-full bg-violet-500/15 animate-pulse"
                style={{ animationDuration: "3s", animationDelay: "0.5s" }}
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <span className="text-2xl font-bold tabular-nums text-foreground">
                    {Math.floor(secondsLeft / 60)}:{String(secondsLeft % 60).padStart(2, "0")}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <Button variant="ghost" size="sm" onClick={() => { if (intervalRef.current) clearInterval(intervalRef.current); onComplete(); }}>
            End early
          </Button>
        </>
      )}

      {!started && (
        <Button onClick={start} className="w-full">
          <Brain className="h-4 w-4 mr-2" />
          Begin meditation
        </Button>
      )}
    </div>
  );
}

// ── Step 4: Results Summary ───────────────────────────────────────────────

function ResultsStep({ onFinish }: { onFinish: () => void }) {
  const [sessionTime] = useState(() => {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  });

  return (
    <div className="text-center space-y-6">
      <div className="inline-flex items-center justify-center h-16 w-16 rounded-full bg-primary/10 mx-auto">
        <Check className="h-8 w-8 text-primary" />
      </div>

      <div className="space-y-2">
        <h2 className="text-xl font-semibold">Session Complete</h2>
        <p className="text-sm text-muted-foreground">
          Great work. Here's your 5-minute session summary.
        </p>
      </div>

      <div className="grid gap-3">
        <Card className="p-4 text-left">
          <div className="flex items-center gap-3">
            <Mic className="h-4 w-4 text-cyan-400 shrink-0" />
            <div>
              <p className="text-sm font-medium">Voice check-in</p>
              <p className="text-xs text-muted-foreground">Emotional state captured</p>
            </div>
          </div>
        </Card>
        <Card className="p-4 text-left">
          <div className="flex items-center gap-3">
            <Wind className="h-4 w-4 text-green-400 shrink-0" />
            <div>
              <p className="text-sm font-medium">Coherence breathing</p>
              <p className="text-xs text-muted-foreground">2 minutes of heart-brain sync</p>
            </div>
          </div>
        </Card>
        <Card className="p-4 text-left">
          <div className="flex items-center gap-3">
            <Brain className="h-4 w-4 text-violet-400 shrink-0" />
            <div>
              <p className="text-sm font-medium">Meditation</p>
              <p className="text-xs text-muted-foreground">2 minutes of mindful presence</p>
            </div>
          </div>
        </Card>
      </div>

      <p className="text-xs text-muted-foreground">
        Completed at {sessionTime}
      </p>

      <Button className="w-full" onClick={onFinish}>
        Back to Today
        <ChevronRight className="h-4 w-4 ml-1" />
      </Button>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────

export default function QuickSession() {
  const [, navigate] = useLocation();
  const [step, setStep] = useState<SessionStep>("voice");

  function goTo(next: SessionStep) {
    hapticLight();
    setStep(next);
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-background">
      <div className="max-w-md w-full space-y-6">
        <StepProgress current={step} />

        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -16 }}
            transition={{ duration: 0.25 }}
          >
            {step === "voice" && (
              <VoiceStep onComplete={() => goTo("breathing")} />
            )}
            {step === "breathing" && (
              <BreathingStep onComplete={() => goTo("meditation")} />
            )}
            {step === "meditation" && (
              <MeditationStep onComplete={() => goTo("results")} />
            )}
            {step === "results" && (
              <ResultsStep onFinish={() => navigate("/")} />
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
