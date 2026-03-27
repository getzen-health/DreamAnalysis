import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation } from "wouter";
import { Brain, CheckCircle, Radio, RotateCcw, FlaskConical } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useDevice } from "@/hooks/use-device";
import {
  addBaselineFrame,
  getBaselineStatus,
  resetBaselineCalibration,
  simulateEEG,
} from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";

/* ── Constants ─────────────────────────────────────────────────────── */
const TARGET_FRAMES = 120; // 2 minutes at 1 frame/sec
const MIN_READY     = 30;  // backend marks ready after 30 frames
const RING_SIZE     = 220;
const RING_RADIUS   = 88;
const RING_CIRCUM   = 2 * Math.PI * RING_RADIUS;
const USER_ID       = getParticipantId();

type Phase = "idle" | "running" | "complete";

/* ── Helpers ───────────────────────────────────────────────────────── */
function phaseLabel(phase: Phase, frames: number, ready: boolean): string {
  if (phase === "complete") return "Calibration complete";
  if (phase === "idle")     return frames > 0 ? `${frames} frames collected` : "Ready to begin";
  if (frames <  10)         return "Getting ready…";
  if (frames <  MIN_READY)  return "Recording resting state…";
  if (frames >= 110)        return "Almost done…";
  if (ready)                return "Building your brain profile…";
  return "Recording…";
}

function elapsedLabel(frames: number): string {
  const m = Math.floor(frames / 60);
  const s = frames % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

/* ── Component ─────────────────────────────────────────────────────── */
export default function CalibrationPage() {
  const [, navigate]                   = useLocation();
  const { state: deviceState, latestFrame } = useDevice();
  const isStreaming                     = deviceState === "streaming";

  const [phase,        setPhase]       = useState<Phase>("idle");
  const [frameCount,   setFrameCount]  = useState(0);
  const [isReady,      setIsReady]     = useState(false);
  const [errMsg,       setErrMsg]      = useState<string | null>(null);
  const [simulateMode, setSimulateMode]= useState(false);

  const latestFrameRef   = useRef(latestFrame);
  const intervalRef      = useRef<ReturnType<typeof setInterval> | null>(null);
  const simulateModeRef  = useRef(simulateMode);

  // Keep refs in sync so interval callbacks always read freshest values
  useEffect(() => { latestFrameRef.current = latestFrame; }, [latestFrame]);
  useEffect(() => { simulateModeRef.current = simulateMode; }, [simulateMode]);

  // On mount: check existing baseline status
  useEffect(() => {
    getBaselineStatus(USER_ID)
      .then((s) => {
        setFrameCount(s.n_frames);
        setIsReady(s.ready);
        if (s.n_frames >= TARGET_FRAMES) setPhase("complete");
      })
      .catch(() => {});
  }, []);

  // Note: simulation mode is an explicit user choice — do not auto-enable it

  /* ── Stop interval helper ────────────────────────────────────────── */
  const stopInterval = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  /* ── Start / restart ─────────────────────────────────────────────── */
  const startCalibration = async () => {
    setErrMsg(null);
    try {
      await resetBaselineCalibration(USER_ID);
    } catch { /* ignore — backend may not have one yet */ }
    setFrameCount(0);
    setIsReady(false);
    setPhase("running");
  };

  /* ── Frame sender ────────────────────────────────────────────────── */
  useEffect(() => {
    if (phase !== "running") {
      stopInterval();
      return;
    }

    // Real device: pause if not streaming (interval cleared, resumes on reconnect)
    if (!simulateMode && !isStreaming) {
      stopInterval();
      return;
    }

    intervalRef.current = setInterval(async () => {
      let signals: number[][];

      if (simulateModeRef.current) {
        // Simulation mode: fetch 1s of synthetic resting-state EEG
        try {
          const sim = await simulateEEG("rest", 1, 256, 4);
          signals = sim.signals;
        } catch (e) {
          const msg = e instanceof Error ? e.message : "Simulation error";
          setErrMsg(`Simulation error: ${msg}`);
          return;
        }
      } else {
        // Real device: use latest WebSocket frame
        const frame = latestFrameRef.current;
        if (!frame?.signals?.length) return;
        signals = frame.signals;
      }

      try {
        const result = await addBaselineFrame(
          signals,
          USER_ID,
          256
        );
        setFrameCount(result.n_frames);
        setIsReady(result.ready);
        if (result.n_frames >= TARGET_FRAMES) {
          setPhase("complete");
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Unknown error";
        setErrMsg(`Frame error: ${msg}`);
      }
    }, 1000);

    return stopInterval;
  }, [phase, isStreaming, simulateMode, stopInterval]);

  /* ── Derived values ──────────────────────────────────────────────── */
  const progress         = Math.min(1, frameCount / TARGET_FRAMES);
  const strokeDashoffset = RING_CIRCUM * (1 - progress);
  const remainingSec     = Math.max(0, TARGET_FRAMES - frameCount);
  const canExitEarly     = isReady && phase === "running";
  const ringColor        = phase === "complete"
    ? "hsl(152,60%,48%)"
    : isReady
      ? "hsl(152,55%,45%)"
      : simulateMode
        ? "hsl(280,65%,60%)"   // purple for simulation
        : "hsl(200,70%,55%)";  // blue for real device

  const canStart = simulateMode || isStreaming;

  /* ── Render ──────────────────────────────────────────────────────── */
  return (
    <main className="min-h-[calc(100vh-4rem)] flex items-center justify-center p-6">
      <div className="flex flex-col items-center gap-7 max-w-sm w-full">

        {/* Title */}
        <div className="text-center">
          <div className="flex items-center justify-center gap-2 mb-1">
            <h1 className="text-2xl font-semibold">Baseline Calibration</h1>
            {simulateMode && phase !== "complete" && (
              <Badge variant="outline" className="text-xs border-purple-500/40 text-purple-400 bg-purple-500/10">
                Simulation
              </Badge>
            )}
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Record 2 minutes of resting brain activity so the app can learn
            your personal baseline — improves emotion accuracy by up to 29%.
          </p>
        </div>

        {/* Mode selector (only in idle phase) */}
        {phase === "idle" && (
          <div className="w-full flex rounded-lg overflow-hidden border border-border/50">
            <button
              onClick={() => setSimulateMode(false)}
              className={`flex-1 py-2 text-xs font-medium flex items-center justify-center gap-1.5 transition-colors ${
                !simulateMode
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Radio className="h-3.5 w-3.5" />
              EEG headband
            </button>
            <button
              onClick={() => setSimulateMode(true)}
              className={`flex-1 py-2 text-xs font-medium flex items-center justify-center gap-1.5 transition-colors ${
                simulateMode
                  ? "bg-purple-600 text-white"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <FlaskConical className="h-3.5 w-3.5" />
              Simulation (no device)
            </button>
          </div>
        )}

        {/* Device warning — only when real device mode and not streaming */}
        {!simulateMode && !isStreaming && phase !== "complete" && (
          <div className="w-full p-4 rounded-xl border border-warning/30 bg-warning/5 flex items-start gap-3">
            <Radio className="h-4 w-4 text-warning shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-warning">EEG headband not connected</p>
              <p className="text-xs text-muted-foreground mt-0.5">
                Connect your headset from the sidebar, or switch to Simulation mode.
              </p>
            </div>
          </div>
        )}

        {/* Simulation mode info */}
        {simulateMode && phase === "idle" && (
          <div className="w-full p-3 rounded-xl border border-purple-500/20 bg-purple-500/5 text-xs text-purple-300 leading-relaxed">
            Simulation generates synthetic resting-state EEG to demonstrate the calibration process. For real accuracy improvements, run with your EEG headband.
          </div>
        )}

        {/* Pause notice while running with real device */}
        {phase === "running" && !simulateMode && !isStreaming && (
          <div className="w-full p-3 rounded-xl text-center text-xs text-warning border border-warning/20 bg-warning/5">
            Calibration paused — reconnect headset to continue
          </div>
        )}

        {/* ── Progress ring ── */}
        <div className="relative select-none">
          {/* Breathing glow when running */}
          {phase === "running" && (
            <div
              className="absolute inset-[-16px] rounded-full"
              style={{
                background: `radial-gradient(circle, ${ringColor}28 0%, transparent 70%)`,
                animation: "breathe 4s ease-in-out infinite",
              }}
            />
          )}

          {/* Complete glow */}
          {phase === "complete" && (
            <div
              className="absolute inset-[-8px] rounded-full"
              style={{ background: "radial-gradient(circle, hsl(152,60%,48%,0.15) 0%, transparent 70%)" }}
            />
          )}

          <svg
            width={RING_SIZE}
            height={RING_SIZE}
            style={{ transform: "rotate(-90deg)", display: "block" }}
          >
            {/* Track */}
            <circle
              cx={RING_SIZE / 2} cy={RING_SIZE / 2} r={RING_RADIUS}
              fill="none"
              stroke="hsl(220,18%,12%)"
              strokeWidth={10}
            />
            {/* Progress */}
            <circle
              cx={RING_SIZE / 2} cy={RING_SIZE / 2} r={RING_RADIUS}
              fill="none"
              stroke={ringColor}
              strokeWidth={10}
              strokeLinecap="round"
              strokeDasharray={RING_CIRCUM}
              strokeDashoffset={strokeDashoffset}
              style={{ transition: "stroke-dashoffset 0.9s ease, stroke 0.5s ease" }}
            />
          </svg>

          {/* Center content */}
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-1">
            {phase === "complete" ? (
              <>
                <CheckCircle className="h-14 w-14 text-success" />
                <p className="text-[11px] text-muted-foreground mt-1">Complete</p>
              </>
            ) : phase === "idle" ? (
              <>
                <Brain className="h-10 w-10 text-primary opacity-40" />
                <p className="text-xs text-muted-foreground mt-1">Not started</p>
              </>
            ) : (
              <>
                <span className="text-4xl font-mono font-semibold tabular-nums">
                  {Math.round(progress * 100)}%
                </span>
                <span className="text-[11px] text-muted-foreground">
                  {remainingSec > 0 ? `~${remainingSec}s left` : "finishing…"}
                </span>
              </>
            )}
          </div>
        </div>

        {/* Phase label + status */}
        <div className="text-center space-y-1.5">
          <p className="text-sm font-medium">{phaseLabel(phase, frameCount, isReady)}</p>
          {isReady && phase === "running" && (
            <p className="text-xs text-success">
              Minimum reached — continue for better accuracy
            </p>
          )}
          {phase === "running" && (
            <p className="text-[11px] text-muted-foreground font-mono">
              {frameCount} / {TARGET_FRAMES} frames · {elapsedLabel(frameCount)}
            </p>
          )}
          {errMsg && (
            <p className="text-[11px] text-destructive">{errMsg}</p>
          )}
        </div>

        {/* Instructions */}
        {phase !== "complete" && (
          <Card className="glass-card p-4 w-full text-left">
            <p className="text-xs font-medium mb-2 text-muted-foreground uppercase tracking-wide">
              {simulateMode ? "How simulation works" : "How to calibrate"}
            </p>
            <ol className="text-xs text-muted-foreground space-y-2 list-none">
              {simulateMode
                ? [
                    "Press Start — synthetic resting-state EEG is generated automatically",
                    "The system builds a simulated personal baseline over 2 minutes",
                    "Demonstrates the calibration flow without hardware",
                    "For real +29% accuracy gains, run with your EEG headband",
                  ].map((step, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-purple-400 font-semibold shrink-0">{i + 1}.</span>
                      {step}
                    </li>
                  ))
                : [
                    "Connect EEG headband and press Start",
                    "Sit comfortably and close your eyes",
                    "Breathe naturally — avoid jaw clenching or blinking",
                    "Wait 2 minutes for best results (exit early at 30s minimum)",
                  ].map((step, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-primary font-semibold shrink-0">{i + 1}.</span>
                      {step}
                    </li>
                  ))
              }
            </ol>
            <p className="text-[11px] text-muted-foreground mt-3 pt-3 border-t border-border/30">
              Calibration normalises your readings against your personal resting-state, improving
              emotion accuracy by +15–29% across all 16 models.
            </p>
          </Card>
        )}

        {/* Action buttons */}
        <div className="flex flex-col items-center gap-3 w-full">
          {phase === "complete" && (
            <>
              <Button className="w-full" onClick={() => navigate("/brain-monitor")}>
                <CheckCircle className="h-4 w-4 mr-2" />
                Start measuring emotions
              </Button>
              <button
                className="text-xs text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
                onClick={startCalibration}
              >
                <RotateCcw className="h-3 w-3" /> Recalibrate
              </button>
            </>
          )}

          {phase === "idle" && (
            <Button
              className={`w-full ${simulateMode ? "bg-purple-600 hover:bg-purple-700 text-white border-purple-500" : ""}`}
              disabled={!canStart}
              onClick={startCalibration}
            >
              <Brain className="h-4 w-4 mr-2" />
              {simulateMode
                ? "Start simulation"
                : isStreaming
                  ? "Start calibration"
                  : "Connect device first"}
            </Button>
          )}

          {canExitEarly && (
            <Button
              variant="outline"
              className="w-full border-success/30 text-success hover:bg-success/10"
              onClick={() => navigate("/brain-monitor")}
            >
              Done — use readings now
            </Button>
          )}

          {phase === "running" && !canExitEarly && (
            <Button
              variant="ghost"
              size="sm"
              className="text-muted-foreground hover:text-foreground"
              onClick={() => { setPhase("idle"); stopInterval(); }}
            >
              Cancel
            </Button>
          )}

          <button
            onClick={() => navigate("/brain-monitor")}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Skip for now →
          </button>
        </div>
      </div>

      {/* Breathing keyframe */}
      <style>{`
        @keyframes breathe {
          0%, 100% { opacity: 0.4; transform: scale(1); }
          50%       { opacity: 0.9; transform: scale(1.06); }
        }
      `}</style>
    </main>
  );
}
