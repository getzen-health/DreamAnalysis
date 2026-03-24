/**
 * deep-work.tsx — Deep Work / Creative Flow timer for programmers (#536)
 *
 * A dedicated deep work timer with EEG-enhanced focus tracking.
 * Shows: pomodoro timer (25/50 min), focus score from EEG/voice,
 * flow state indicator, and distraction alerts.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Code, Play, Square, Timer, Zap, AlertTriangle, Brain, TrendingUp } from "lucide-react";
import { useDevice } from "@/hooks/use-device";
import { useFusedState } from "@/hooks/use-fused-state";

// ── Timer presets ──────────────────────────────────────────────────────────

interface TimerPreset {
  id: string;
  label: string;
  workMinutes: number;
  breakMinutes: number;
}

const TIMER_PRESETS: TimerPreset[] = [
  { id: "pomodoro-25", label: "Pomodoro 25 min", workMinutes: 25, breakMinutes: 5 },
  { id: "pomodoro-50", label: "Deep Work 50 min", workMinutes: 50, breakMinutes: 10 },
  { id: "flow-90", label: "Flow Block 90 min", workMinutes: 90, breakMinutes: 15 },
];

// ── Flow state classification ──────────────────────────────────────────────

type FlowLevel = "distracted" | "warming-up" | "focused" | "flow";

function classifyFlowLevel(focus: number, stress: number): FlowLevel {
  if (focus < 0.3) return "distracted";
  if (focus < 0.5) return "warming-up";
  // Flow = high focus + moderate-to-low stress
  if (focus >= 0.7 && stress < 0.5) return "flow";
  return "focused";
}

function flowLevelColor(level: FlowLevel): string {
  switch (level) {
    case "distracted": return "hsl(0, 70%, 60%)";
    case "warming-up": return "hsl(48, 96%, 53%)";
    case "focused": return "hsl(210, 85%, 60%)";
    case "flow": return "hsl(142, 65%, 48%)";
  }
}

function flowLevelLabel(level: FlowLevel): string {
  switch (level) {
    case "distracted": return "Distracted";
    case "warming-up": return "Warming Up";
    case "focused": return "Focused";
    case "flow": return "In Flow";
  }
}

// ── Component ──────────────────────────────────────────────────────────────

type SessionPhase = "idle" | "work" | "break" | "done";

export default function DeepWork() {
  const { state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const { fusedState } = useFusedState();

  const [phase, setPhase] = useState<SessionPhase>("idle");
  const [preset, setPreset] = useState<TimerPreset>(TIMER_PRESETS[0]);
  const [remainingSeconds, setRemainingSeconds] = useState(0);
  const [totalSeconds, setTotalSeconds] = useState(0);
  const [focusReadings, setFocusReadings] = useState<number[]>([]);
  const [distractionCount, setDistractionCount] = useState(0);
  const [showDistractionAlert, setShowDistractionAlert] = useState(false);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastFlowLevel = useRef<FlowLevel>("warming-up");

  // Current focus and stress from fused state
  const currentFocus = fusedState?.focus ?? 0.5;
  const currentStress = fusedState?.stress ?? 0.3;
  const flowLevel = classifyFlowLevel(currentFocus, currentStress);

  // Track distraction transitions
  useEffect(() => {
    if (phase !== "work") return;
    if (flowLevel === "distracted" && lastFlowLevel.current !== "distracted") {
      setDistractionCount((c) => c + 1);
      setShowDistractionAlert(true);
      setTimeout(() => setShowDistractionAlert(false), 5000);
    }
    lastFlowLevel.current = flowLevel;
  }, [flowLevel, phase]);

  // Collect focus readings during work phase
  useEffect(() => {
    if (phase !== "work") return;
    const id = setInterval(() => {
      setFocusReadings((prev) => [...prev.slice(-120), currentFocus]);
    }, 5000);
    return () => clearInterval(id);
  }, [phase, currentFocus]);

  const startWork = useCallback(() => {
    const seconds = preset.workMinutes * 60;
    setRemainingSeconds(seconds);
    setTotalSeconds(seconds);
    setFocusReadings([]);
    setDistractionCount(0);
    setShowDistractionAlert(false);
    setPhase("work");
  }, [preset]);

  const startBreak = useCallback(() => {
    const seconds = preset.breakMinutes * 60;
    setRemainingSeconds(seconds);
    setTotalSeconds(seconds);
    setPhase("break");
  }, [preset]);

  // Timer countdown
  useEffect(() => {
    if (phase !== "work" && phase !== "break") {
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }

    timerRef.current = setInterval(() => {
      setRemainingSeconds((prev) => {
        if (prev <= 1) {
          if (phase === "work") {
            setPhase("break");
            setRemainingSeconds(preset.breakMinutes * 60);
            setTotalSeconds(preset.breakMinutes * 60);
          } else {
            setPhase("done");
          }
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [phase, preset]);

  const handleStop = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setPhase("done");
  };

  const handleReset = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setPhase("idle");
    setFocusReadings([]);
    setDistractionCount(0);
  };

  // Computed values
  const progress = totalSeconds > 0 ? ((totalSeconds - remainingSeconds) / totalSeconds) * 100 : 0;
  const avgFocus = focusReadings.length > 0
    ? focusReadings.reduce((a, b) => a + b, 0) / focusReadings.length
    : 0;

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <main className="p-4 md:p-6 pb-24 space-y-6 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Code className="h-6 w-6 text-primary" />
        <div>
          <h2 className="text-xl font-semibold">Deep Work</h2>
          <p className="text-xs text-muted-foreground">
            EEG-enhanced focus timer for programmers
          </p>
        </div>
      </div>

      {/* IDLE — Timer Setup */}
      {phase === "idle" && (
        <Card className="glass-card p-6 rounded-xl space-y-5">
          <h3 className="text-lg font-semibold">Start a Focus Session</h3>

          <div className="space-y-3">
            <label className="text-sm text-muted-foreground">Timer Preset</label>
            <Select
              value={preset.id}
              onValueChange={(id) => {
                const found = TIMER_PRESETS.find((p) => p.id === id);
                if (found) setPreset(found);
              }}
            >
              <SelectTrigger className="w-full bg-card/50 border border-primary/30">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {TIMER_PRESETS.map((p) => (
                  <SelectItem key={p.id} value={p.id}>
                    {p.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              {preset.workMinutes} min work, {preset.breakMinutes} min break
            </p>
          </div>

          {!isStreaming && (
            <div className="flex items-center gap-2 text-xs text-warning border border-warning/30 bg-warning/5 rounded-lg p-3">
              <Brain className="h-4 w-4 shrink-0" />
              Connect your Muse for real-time focus tracking. Voice/health signals work too.
            </div>
          )}

          <Button
            onClick={startWork}
            className="w-full bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30"
          >
            <Play className="h-4 w-4 mr-2" />
            Start Deep Work
          </Button>
        </Card>
      )}

      {/* WORK phase — Timer + Flow State */}
      {phase === "work" && (
        <div className="space-y-4">
          {/* Timer card */}
          <Card className="glass-card p-8 rounded-xl text-center space-y-4">
            <Badge variant="secondary" className="mx-auto">
              <Timer className="h-3 w-3 mr-1" />
              Work Phase
            </Badge>

            <p className="text-5xl font-mono font-bold tracking-tight">
              {formatTime(remainingSeconds)}
            </p>

            <Progress value={progress} className="h-2" />

            <Button
              size="sm"
              variant="destructive"
              onClick={handleStop}
              className="bg-destructive/10 border border-destructive/30 text-destructive"
            >
              <Square className="h-3 w-3 mr-1" />
              Stop
            </Button>
          </Card>

          {/* Flow state indicator */}
          <Card className="glass-card p-5 rounded-xl">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium">Flow State</span>
              <Badge
                variant="outline"
                style={{ borderColor: flowLevelColor(flowLevel), color: flowLevelColor(flowLevel) }}
              >
                <Zap className="h-3 w-3 mr-1" />
                {flowLevelLabel(flowLevel)}
              </Badge>
            </div>

            <div className="grid grid-cols-2 gap-4 text-center">
              <div>
                <p className="text-xs text-muted-foreground mb-1">Focus</p>
                <p className="text-2xl font-mono font-bold">{Math.round(currentFocus * 100)}%</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">Stress</p>
                <p className="text-2xl font-mono font-bold">{Math.round(currentStress * 100)}%</p>
              </div>
            </div>
          </Card>

          {/* Distraction alert */}
          {showDistractionAlert && (
            <Card className="p-4 rounded-xl border border-warning/30 bg-warning/5">
              <div className="flex items-center gap-3">
                <AlertTriangle className="h-5 w-5 text-warning shrink-0" />
                <div>
                  <p className="text-sm font-medium text-warning">Focus dip detected</p>
                  <p className="text-xs text-muted-foreground">
                    Take a breath. Return your attention to the task.
                  </p>
                </div>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* BREAK phase */}
      {phase === "break" && (
        <Card className="glass-card p-8 rounded-xl text-center space-y-4">
          <Badge variant="outline" className="mx-auto border-green-500/30 text-green-500">
            Break Time
          </Badge>

          <p className="text-4xl font-mono font-bold">{formatTime(remainingSeconds)}</p>
          <p className="text-sm text-muted-foreground">
            Step away from the screen. Stretch. Hydrate.
          </p>

          <Button
            size="sm"
            variant="outline"
            onClick={handleReset}
          >
            End Session
          </Button>
        </Card>
      )}

      {/* DONE — Summary */}
      {phase === "done" && (
        <Card className="glass-card p-8 rounded-xl space-y-6">
          <div className="text-center space-y-2">
            <TrendingUp className="h-10 w-10 text-primary mx-auto" />
            <h3 className="text-xl font-semibold">Session Complete</h3>
          </div>

          <div className="grid grid-cols-3 gap-4 text-center py-4 border-y border-border/20">
            <div>
              <p className="text-xs text-muted-foreground mb-1">Avg Focus</p>
              <p className="text-2xl font-mono font-bold">
                {Math.round(avgFocus * 100)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">Duration</p>
              <p className="text-2xl font-mono font-bold">{preset.workMinutes}m</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">Distractions</p>
              <p className="text-2xl font-mono font-bold">{distractionCount}</p>
            </div>
          </div>

          <div className="flex gap-3 justify-center">
            <Button onClick={handleReset} variant="outline">
              New Session
            </Button>
            <Button
              onClick={startWork}
              className="bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30"
            >
              <Play className="h-4 w-4 mr-2" />
              Go Again
            </Button>
          </div>
        </Card>
      )}
    </main>
  );
}
