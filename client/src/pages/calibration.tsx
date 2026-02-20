import { useState, useEffect, useRef } from "react";
import { useLocation } from "wouter";
import { Brain, CheckCircle, Radio, RotateCcw } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useDevice } from "@/hooks/use-device";
import {
  addBaselineFrame,
  getBaselineStatus,
  resetBaselineCalibration,
} from "@/lib/ml-api";

/* ── Constants ─────────────────────────────────────────────────────── */
const TARGET_FRAMES = 120; // 2 minutes at 1 frame/sec
const MIN_READY    = 30;   // usable after 30 sec
const RING_SIZE    = 220;
const RING_RADIUS  = 88;
const RING_CIRCUM  = 2 * Math.PI * RING_RADIUS;

type Phase = "idle" | "running" | "complete";

/* ── Helpers ───────────────────────────────────────────────────────── */
function phaseLabel(phase: Phase, frames: number, ready: boolean): string {
  if (phase === "complete") return "Calibration complete";
  if (phase === "idle")     return frames > 0 ? `${frames} frames already collected` : "Ready to begin";
  if (frames <  10)         return "Getting ready…";
  if (frames <  30)         return "Recording resting state…";
  if (frames >= 110)        return "Almost done…";
  if (ready)                return "Building your brain profile…";
  return "Recording…";
}

/* ── Component ─────────────────────────────────────────────────────── */
export default function CalibrationPage() {
  const [, navigate]                = useLocation();
  const { state: deviceState, latestFrame } = useDevice();
  const isStreaming                  = deviceState === "streaming";

  const [phase,      setPhase]      = useState<Phase>("idle");
  const [frameCount, setFrameCount] = useState(0);
  const [isReady,    setIsReady]    = useState(false);
  const [errMsg,     setErrMsg]     = useState<string | null>(null);

  const latestFrameRef = useRef(latestFrame);
  const intervalRef    = useRef<ReturnType<typeof setInterval> | null>(null);

  // Keep ref in sync (so the 1s interval always reads the freshest frame)
  useEffect(() => { latestFrameRef.current = latestFrame; }, [latestFrame]);

  // On mount: check if baseline already partially / fully collected
  useEffect(() => {
    getBaselineStatus("default")
      .then((s) => {
        setFrameCount(s.n_frames);
        setIsReady(s.ready);
        if (s.n_frames >= TARGET_FRAMES) setPhase("complete");
      })
      .catch(() => {});
  }, []);

  /* ── Start / restart ─────────────────────────────────────────────── */
  const startCalibration = async () => {
    setErrMsg(null);
    try {
      await resetBaselineCalibration("default");
    } catch { /* ignore — backend may not have one yet */ }
    setFrameCount(0);
    setIsReady(false);
    setPhase("running");
  };

  /* ── Frame sender ────────────────────────────────────────────────── */
  useEffect(() => {
    if (phase !== "running" || !isStreaming) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    intervalRef.current = setInterval(async () => {
      const frame = latestFrameRef.current;
      if (!frame?.signals?.length) return;

      try {
        const result = await addBaselineFrame(
          frame.signals,
          "default",
          frame.sample_rate ?? 256
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

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [phase, isStreaming]);

  /* ── Derived values ──────────────────────────────────────────────── */
  const progress          = Math.min(1, frameCount / TARGET_FRAMES);
  const strokeDashoffset  = RING_CIRCUM * (1 - progress);
  const remainingSec      = Math.max(0, TARGET_FRAMES - frameCount);
  const canExitEarly      = isReady && phase === "running";
  const ringColor         = phase === "complete"
    ? "hsl(152,60%,48%)"
    : isReady
      ? "hsl(152,55%,45%)"
      : "hsl(200,70%,55%)";

  /* ── Render ──────────────────────────────────────────────────────── */
  return (
    <main className="min-h-[calc(100vh-4rem)] flex items-center justify-center p-6">
      <div className="flex flex-col items-center gap-7 max-w-sm w-full">

        {/* Title */}
        <div className="text-center">
          <h1 className="text-2xl font-semibold mb-1">Baseline Calibration</h1>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Sit still with eyes closed for 2 minutes so the app can learn
            your unique resting brain signature.
          </p>
        </div>

        {/* Device warning */}
        {!isStreaming && phase !== "complete" && (
          <div className="w-full p-4 rounded-xl border border-warning/30 bg-warning/5 flex items-start gap-3">
            <Radio className="h-4 w-4 text-warning shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-warning">Muse 2 not connected</p>
              <p className="text-xs text-muted-foreground mt-0.5">
                Connect your headset from the sidebar, then start calibration.
              </p>
            </div>
          </div>
        )}

        {/* Pause notice while running */}
        {phase === "running" && !isStreaming && (
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
                background: `radial-gradient(circle, ${ringColor}18 0%, transparent 70%)`,
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

        {/* Phase label + ready badge */}
        <div className="text-center space-y-1.5">
          <p className="text-sm font-medium">{phaseLabel(phase, frameCount, isReady)}</p>
          {isReady && phase === "running" && (
            <p className="text-xs text-success">
              Minimum reached — keep still for better accuracy
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
              How it works
            </p>
            <ol className="text-xs text-muted-foreground space-y-2 list-none">
              {[
                "Connect Muse 2 and press Start",
                "Sit comfortably and close your eyes",
                "Breathe naturally — avoid jaw clenching or blinking",
                "Wait 2 minutes for the best results (or exit at 30s)",
              ].map((step, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-primary font-semibold shrink-0">{i + 1}.</span>
                  {step}
                </li>
              ))}
            </ol>
            <p className="text-[11px] text-muted-foreground mt-3 pt-3 border-t border-border/30">
              Calibration improves emotion accuracy by +15–29% by normalising your readings
              against your personal resting-state baseline.
            </p>
          </Card>
        )}

        {/* Action buttons */}
        <div className="flex flex-col items-center gap-3 w-full">
          {phase === "complete" && (
            <>
              <Button className="w-full" onClick={() => navigate("/emotions")}>
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
              className="w-full"
              disabled={!isStreaming}
              onClick={startCalibration}
            >
              <Brain className="h-4 w-4 mr-2" />
              {isStreaming ? "Start calibration" : "Connect device first"}
            </Button>
          )}

          {canExitEarly && (
            <Button
              variant="outline"
              className="w-full border-success/30 text-success hover:bg-success/10"
              onClick={() => navigate("/emotions")}
            >
              Done — use readings now
            </Button>
          )}

          <button
            onClick={() => navigate("/emotions")}
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

function elapsedLabel(frames: number): string {
  const m = Math.floor(frames / 60);
  const s = frames % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}
