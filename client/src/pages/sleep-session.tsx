import { useState, useEffect, useRef } from "react";
import { Link } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { resolveUrl } from "@/lib/queryClient";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useDevice } from "@/hooks/use-device";
import { startSession, stopSession } from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";
import { hapticLight, hapticSuccess } from "@/lib/haptics";
import { backgroundEeg } from "@/lib/background-eeg";
import CGMSleepOverlay from "@/components/cgm-sleep-overlay";
import { SleepHypnogram, type StageEvent } from "@/components/sleep-hypnogram";
import {
  Moon,
  BrainCircuit,
  Sparkles,
  Clock,
  Play,
  Square,
  Eye,
} from "lucide-react";

const CURRENT_USER = getParticipantId();

// ─── Sleep Stage Definitions ──────────────────────────────────────────────────

type SleepStage = "Wake" | "N1" | "N2" | "N3" | "REM";

const STAGE_COLORS: Record<SleepStage, string> = {
  Wake:  "hsl(30, 90%, 55%)",
  N1:    "hsl(48, 90%, 55%)",
  N2:    "hsl(210, 80%, 55%)",
  N3:    "hsl(240, 65%, 55%)",
  REM:   "hsl(270, 75%, 60%)",
};

const STAGE_BG: Record<SleepStage, string> = {
  Wake:  "hsl(30, 90%, 55%, 0.12)",
  N1:    "hsl(48, 90%, 55%, 0.12)",
  N2:    "hsl(210, 80%, 55%, 0.12)",
  N3:    "hsl(240, 65%, 55%, 0.12)",
  REM:   "hsl(270, 75%, 60%, 0.12)",
};

const STAGE_DESCRIPTION: Record<SleepStage, string> = {
  Wake:  "Awake — monitoring brain activity",
  N1:    "Light Sleep — drifting off, easy to wake",
  N2:    "Core Sleep — spindles & K-complexes",
  N3:    "Deep Sleep — slow-wave, restorative",
  REM:   "Dream Sleep — rapid eye movement",
};

// Simulation cycle: N1 → N2 → N3 → REM → N2 → ... (each ~8 minutes simplified)
const SIM_CYCLE: SleepStage[] = ["N1", "N2", "N3", "REM", "N2", "N3", "REM", "N2"];
const SIM_STAGE_DURATION_SEC = 8 * 60; // 8 minutes per stage in simulation

// ─── Types ────────────────────────────────────────────────────────────────────

type SessionPhase = "idle" | "recording" | "summary";

interface StageTally {
  Wake: number;
  N1: number;
  N2: number;
  N3: number;
  REM: number;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function fmtDuration(sec: number): string {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.round(sec % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function fmtClock(sec: number): string {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.round(sec % 60);
  return h > 0
    ? `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`
    : `${m}:${s.toString().padStart(2, "0")}`;
}

function totalSec(tally: StageTally): number {
  return tally.Wake + tally.N1 + tally.N2 + tally.N3 + tally.REM;
}

function sleepScore(tally: StageTally): number {
  const total = totalSec(tally);
  if (total === 0) return 0;
  const remPct   = tally.REM  / total;
  const deepPct  = tally.N3   / total;
  const lightPct = (tally.N1 + tally.N2) / total;
  const wakePct  = tally.Wake / total;
  // Score out of 100: reward REM/deep, penalise wake
  const score = Math.round(
    100 * (0.40 * Math.min(remPct / 0.25, 1) +
           0.35 * Math.min(deepPct / 0.20, 1) +
           0.15 * Math.min(lightPct / 0.50, 1) -
           0.10 * Math.min(wakePct / 0.10, 1))
  );
  return Math.max(0, Math.min(100, score));
}

// ─── Component ────────────────────────────────────────────────────────────────

interface HealthMetric {
  sleepQuality?: number | null;
  sleepDuration?: number | null;
  timestamp?: string;
}

export default function SleepSession() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";

  const [phase, setPhase] = useState<SessionPhase>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [currentStage, setCurrentStage] = useState<SleepStage>("N1");
  const [stageTimeSec, setStageTimeSec] = useState(0); // seconds in current stage
  const [tally, setTally] = useState<StageTally>({ Wake: 0, N1: 0, N2: 0, N3: 0, REM: 0 });
  const [dreamCount, setDreamCount] = useState(0);
  const [dreamFlash, setDreamFlash] = useState(false);
  const [dreamsDetected, setDreamsDetected] = useState(0); // final summary count
  const [stageHistory, setStageHistory] = useState<StageEvent[]>([]);

  // Screen dim state — activates when recording starts, tap anywhere to peek
  const [dimmed, setDimmed] = useState(false);
  const [peekVisible, setPeekVisible] = useState(false);
  const peekTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const startTimeRef     = useRef<number>(0);
  const elapsedRef       = useRef<number>(0);
  const tallyRef         = useRef<StageTally>({ Wake: 0, N1: 0, N2: 0, N3: 0, REM: 0 });
  const stageTimeRef     = useRef<number>(0);
  const dreamCountRef    = useRef<number>(0);
  const lastDreamRef     = useRef<boolean>(false);
  const intervalRef      = useRef<ReturnType<typeof setInterval> | null>(null);
  const stageHistoryRef  = useRef<StageEvent[]>([]);
  const prevStageRef     = useRef<SleepStage | null>(null);

  // Real sleep stats from DB
  const { data: healthMetrics } = useQuery<HealthMetric[]>({
    queryKey: ["/api/health-metrics", CURRENT_USER],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/health-metrics/${CURRENT_USER}`));
      if (!res.ok) return [];
      return res.json();
    },
    staleTime: 5 * 60 * 1000,
  });

  // Compute recent sleep stats from DB records
  const recentStats = (() => {
    const rows = (healthMetrics ?? []).filter(m => m.sleepDuration != null || m.sleepQuality != null);
    if (rows.length === 0) return null;
    const latest = rows[0];
    const avgQuality = rows.slice(0, 7).reduce((s, r) => s + (r.sleepQuality ?? 0), 0) / Math.min(rows.length, 7);
    const lastDurH = latest.sleepDuration ? `${Math.floor(latest.sleepDuration)}h ${Math.round((latest.sleepDuration % 1) * 60)}m` : "—";
    return [
      { label: "Last session", value: lastDurH },
      { label: "Sleep quality", value: latest.sleepQuality ? `${latest.sleepQuality}/9` : "—" },
      { label: "7-day avg", value: rows.slice(0, 7).length > 0 ? `${(rows.slice(0, 7).reduce((s, r) => s + (r.sleepDuration ?? 0), 0) / rows.slice(0, 7).length).toFixed(1)}h` : "—" },
      { label: "Avg quality", value: avgQuality > 0 ? `${avgQuality.toFixed(1)}/9` : "—" },
    ];
  })() ?? [
    { label: "Last session", value: "—" },
    { label: "Sleep quality", value: "—" },
    { label: "7-day avg",    value: "—" },
    { label: "Avg quality",  value: "—" },
  ];

  // Screen peek — tap anywhere to reveal controls briefly then re-dim
  const handleScreenTap = () => {
    if (!dimmed) return;
    if (peekTimerRef.current) clearTimeout(peekTimerRef.current);
    setPeekVisible(true);
    peekTimerRef.current = setTimeout(() => setPeekVisible(false), 8000);
  };

  // Derive live stage from WebSocket frame (when device is streaming)
  const liveStage: SleepStage | null = (() => {
    const s = latestFrame?.analysis?.sleep_staging;
    if (!s) return null;
    const map: Record<string, SleepStage> = {
      Wake: "Wake", W: "Wake",
      N1: "N1", "Stage 1": "N1",
      N2: "N2", "Stage 2": "N2",
      N3: "N3", "Stage 3": "N3",
      REM: "REM",
    };
    return map[s.stage] ?? null;
  })();

  const liveDreaming: boolean =
    latestFrame?.analysis?.dream_detection?.is_dreaming ?? false;

  // ── Ticker ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (phase !== "recording") return;

    intervalRef.current = setInterval(() => {
      elapsedRef.current += 1;
      setElapsed(elapsedRef.current);

      // Determine current stage
      let stage: SleepStage;
      if (isStreaming && liveStage) {
        stage = liveStage;
      } else {
        // Simulated: cycle through stages
        const simIdx = Math.floor(elapsedRef.current / SIM_STAGE_DURATION_SEC) % SIM_CYCLE.length;
        stage = SIM_CYCLE[simIdx];
      }
      setCurrentStage(stage);

      // Record stage transitions for hypnogram
      if (stage !== prevStageRef.current) {
        stageHistoryRef.current.push({ stage, t: elapsedRef.current });
        prevStageRef.current = stage;
      }

      // Tally time in stage
      tallyRef.current = { ...tallyRef.current, [stage]: tallyRef.current[stage] + 1 };
      setTally({ ...tallyRef.current });

      // Track seconds in current stage (reset when stage changes)
      stageTimeRef.current += 1;
      setStageTimeSec(stageTimeRef.current);

      // Dream detection
      let isDreaming: boolean;
      if (isStreaming) {
        isDreaming = liveDreaming;
      } else {
        // Simulate: dreams happen during REM, roughly 30% of the time
        isDreaming = stage === "REM" && (elapsedRef.current % 60 < 18);
      }

      if (isDreaming && !lastDreamRef.current) {
        // Rising edge — new dream event
        dreamCountRef.current += 1;
        setDreamCount(dreamCountRef.current);
        setDreamFlash(true);
        hapticLight(); // subtle pulse: dream detected (doesn't fully wake user)
        setTimeout(() => setDreamFlash(false), 4000);
      }
      lastDreamRef.current = isDreaming;
    }, 1000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, isStreaming]);

  // Reset stage timer whenever stage changes
  useEffect(() => {
    stageTimeRef.current = 0;
    setStageTimeSec(0);
  }, [currentStage]);

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleStart = () => {
    startTimeRef.current = Date.now();
    elapsedRef.current = 0;
    tallyRef.current = { Wake: 0, N1: 0, N2: 0, N3: 0, REM: 0 };
    dreamCountRef.current = 0;
    lastDreamRef.current = false;
    stageTimeRef.current = 0;
    stageHistoryRef.current = [];
    prevStageRef.current = null;
    setStageHistory([]);
    setElapsed(0);
    setTally({ Wake: 0, N1: 0, N2: 0, N3: 0, REM: 0 });
    setDreamCount(0);
    setDreamFlash(false);
    setCurrentStage("N1");
    setStageTimeSec(0);
    setPhase("recording");
    // Dim screen after 15 seconds so the user can place their phone/laptop
    setTimeout(() => setDimmed(true), 15_000);
    // Best-effort API call — don't block UI if ML backend is offline
    startSession("sleep", CURRENT_USER).catch(() => {});
    // Start background processing (wake lock on web, BackgroundRunner on native)
    backgroundEeg.startSleepRecording().catch(() => {});
  };

  const handleWakeUp = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (peekTimerRef.current) clearTimeout(peekTimerRef.current);
    setDimmed(false);
    setPeekVisible(false);
    hapticSuccess(); // session complete
    setDreamsDetected(dreamCountRef.current);
    setStageHistory([...stageHistoryRef.current]);
    setPhase("summary");
    stopSession(CURRENT_USER).catch(() => {});
    backgroundEeg.stopSleepRecording().catch(() => {});
  };

  const handleReset = () => {
    setPhase("idle");
    setElapsed(0);
    setTally({ Wake: 0, N1: 0, N2: 0, N3: 0, REM: 0 });
    setDreamCount(0);
    setDreamFlash(false);
    setDimmed(false);
    setPeekVisible(false);
  };

  const stageColor = STAGE_COLORS[currentStage];
  const stageBg    = STAGE_BG[currentStage];
  const finalScore = sleepScore(tally);
  const totalSeconds = totalSec(tally);

  // ─── Idle ────────────────────────────────────────────────────────────────

  if (phase === "idle") {
    return (
      <main className="p-4 md:p-6 space-y-6 max-w-3xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-3">
          <Moon className="h-6 w-6 text-primary" />
          <div>
            <h2 className="text-xl font-semibold">Sleep Session</h2>
            <p className="text-xs text-muted-foreground">
              Overnight sleep tracking with health-based sleep summaries by default, plus optional EEG depth when available
            </p>
          </div>
        </div>

        {/* No-device notice */}
        {!isStreaming && (
          <div className="flex items-center gap-3 p-3 rounded-xl border border-yellow-500/30 bg-yellow-500/5 text-sm text-yellow-500">
            <Moon className="h-4 w-4 shrink-0 opacity-60" />
            No EEG connected — this page still works from recent sleep and health data. Live stages shown here use a preview cycle until EEG is added.
          </div>
        )}

        {/* Recent sleep stats */}
        <Card className="glass-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <h3 className="text-sm font-medium">Recent Sleep</h3>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {recentStats.map(({ label, value }) => (
              <div key={label} className="text-center">
                <p className="text-2xl font-mono font-bold text-primary">{value}</p>
                <p className="text-[10px] text-muted-foreground mt-1">{label}</p>
              </div>
            ))}
          </div>
        </Card>

        {/* Start button */}
        <div className="flex justify-center pt-2">
          <Button
            onClick={handleStart}
            className="bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30 px-12 h-12 text-base"
          >
            <Play className="h-5 w-5 mr-2" />
            Start Sleep Session
          </Button>
        </div>
      </main>
    );
  }

  // ─── Recording ────────────────────────────────────────────────────────────

  if (phase === "recording") {
    // Dim overlay: tap to peek for 8 seconds
    if (dimmed && !peekVisible) {
      return (
        <div
          className="fixed inset-0 z-50 flex flex-col items-center justify-center cursor-pointer select-none"
          style={{ background: "hsl(0,0%,2%)" }}
          onClick={handleScreenTap}
        >
          {/* Minimal sleep indicator */}
          <div className="flex flex-col items-center gap-6 text-center pointer-events-none">
            <Moon className="h-8 w-8 opacity-25" style={{ color: STAGE_COLORS[currentStage] }} />
            <p
              className="text-6xl font-mono font-bold opacity-30"
              style={{ color: STAGE_COLORS[currentStage] }}
            >
              {currentStage}
            </p>
            <p className="text-xs text-white/20 uppercase tracking-widest">
              {fmtClock(elapsed)}
            </p>
            {dreamCount > 0 && (
              <p className="text-xs text-white/15">
                {dreamCount} dream{dreamCount !== 1 ? "s" : ""} detected
              </p>
            )}
          </div>
          {/* Tap hint at bottom */}
          <p className="absolute bottom-8 text-[10px] text-white/15 uppercase tracking-widest">
            tap to reveal controls
          </p>
        </div>
      );
    }

    return (
      <div
        className="relative"
        onClick={dimmed ? handleScreenTap : undefined}
      >
        {/* Peek banner — shown for 8s when user taps during dim mode */}
        {dimmed && peekVisible && (
          <div className="fixed top-0 left-0 right-0 z-40 flex items-center justify-between px-4 py-2 bg-black/80 text-white/60 text-xs">
            <span className="flex items-center gap-1.5">
              <Eye className="h-3 w-3" />
              Peeking — auto-dims in 8s
            </span>
            <button
              className="underline text-white/50"
              onClick={(e) => { e.stopPropagation(); setDimmed(false); }}
            >
              Stay bright
            </button>
          </div>
        )}
      <main className={`p-4 md:p-6 space-y-5 max-w-3xl mx-auto ${dimmed ? "pt-10" : ""}`}>
        {/* Header with elapsed time */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Moon className="h-6 w-6 text-primary" />
            <div>
              <h2 className="text-xl font-semibold">Sleep Session</h2>
              <p className="text-xs text-muted-foreground">Recording in progress</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5 text-muted-foreground text-xs">
              <Clock className="h-3.5 w-3.5" />
              <span className="font-mono text-sm text-foreground">{fmtClock(elapsed)}</span>
            </div>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleWakeUp}
              className="border border-destructive/30 text-destructive hover:bg-destructive/10"
            >
              <Square className="h-3 w-3 mr-1" />
              Wake up
            </Button>
          </div>
        </div>

        {/* Current sleep stage — large display */}
        <Card
          className="glass-card p-8 rounded-2xl flex flex-col items-center gap-4"
          style={{ borderColor: stageColor + "40", background: stageBg }}
        >
          <BrainCircuit className="h-8 w-8" style={{ color: stageColor }} />

          <div className="text-center">
            <p className="text-[11px] text-muted-foreground uppercase tracking-widest mb-2">
              Current Stage
            </p>
            <p
              className="text-5xl font-bold font-mono"
              style={{ color: stageColor }}
            >
              {currentStage}
            </p>
            <p className="text-sm text-muted-foreground mt-2">
              {STAGE_DESCRIPTION[currentStage]}
            </p>
          </div>

          {/* Time in current stage */}
          <div className="text-center">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
              In this stage
            </p>
            <p className="text-lg font-mono text-foreground/70">
              {fmtDuration(stageTimeSec)}
            </p>
          </div>

          {/* Dream detection flash */}
          {dreamFlash && (
            <div
              className="flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium animate-pulse"
              style={{
                background: "hsl(270, 75%, 60%, 0.15)",
                border: "1px solid hsl(270, 75%, 60%, 0.4)",
                color: "hsl(270, 75%, 70%)",
              }}
            >
              <Sparkles className="h-4 w-4" />
              Dream detected!
            </div>
          )}
        </Card>

        {/* Dream count */}
        <Card className="glass-card p-5 flex items-center gap-4">
          <div
            className="w-12 h-12 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: "hsl(270, 75%, 60%, 0.12)" }}
          >
            <Sparkles className="h-6 w-6" style={{ color: "hsl(270, 75%, 60%)" }} />
          </div>
          <div>
            <p className="text-sm font-medium">Dreams detected this session</p>
            <p className="text-2xl font-mono font-bold" style={{ color: "hsl(270, 75%, 60%)" }}>
              {dreamCount}
            </p>
          </div>
        </Card>

        {/* Dim toggle */}
        {!dimmed && (
          <div className="flex justify-center pt-1">
            <Button
              variant="ghost"
              size="sm"
              className="text-muted-foreground text-xs border border-border/20"
              onClick={() => setDimmed(true)}
            >
              <Moon className="h-3.5 w-3.5 mr-1.5" />
              Dim screen for sleep
            </Button>
          </div>
        )}
      </main>
      </div>
    );
  }

  // ─── Summary ──────────────────────────────────────────────────────────────

  return (
    <main className="p-4 md:p-6 space-y-5 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Moon className="h-6 w-6 text-primary" />
        <div>
          <h2 className="text-xl font-semibold">Session Summary</h2>
          <p className="text-xs text-muted-foreground">
            Total duration: {fmtDuration(elapsed)}
          </p>
        </div>
      </div>

      {/* Score card */}
      <Card className="glass-card p-8 rounded-2xl text-center space-y-3">
        <div
          className="w-20 h-20 rounded-full mx-auto flex items-center justify-center"
          style={{
            background: "hsl(152, 60%, 48%, 0.12)",
            border: "1px solid hsl(152, 60%, 48%, 0.3)",
          }}
        >
          <Moon className="h-10 w-10 text-primary" />
        </div>
        <div>
          <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-1">
            Sleep Score
          </p>
          <p className="text-6xl font-mono font-bold text-primary">{finalScore}</p>
          <p className="text-sm text-muted-foreground mt-1">out of 100</p>
        </div>
        <p className="text-sm text-muted-foreground">
          {finalScore >= 80
            ? "Excellent sleep — great stage balance with solid REM and deep sleep."
            : finalScore >= 60
              ? "Good sleep — consider staying in bed a bit longer to boost deep sleep."
              : "Light sleep recorded — try reducing interruptions for deeper cycles."}
        </p>
      </Card>

      {/* Stage breakdown */}
      <Card className="glass-card p-5">
        <h3 className="text-sm font-medium mb-4">Stage Breakdown</h3>
        <div className="space-y-3">
          {(["Wake", "N1", "N2", "N3", "REM"] as SleepStage[]).map(stage => {
            const stageSec = tally[stage];
            const pct = totalSeconds > 0 ? (stageSec / totalSeconds) * 100 : 0;
            return (
              <div key={stage} className="flex items-center gap-3">
                <div
                  className="w-3 h-3 rounded-full shrink-0"
                  style={{ background: STAGE_COLORS[stage] }}
                />
                <span className="text-xs font-medium w-8 shrink-0">{stage}</span>
                <div className="flex-1 bg-muted/20 rounded-full h-2 overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${pct}%`, background: STAGE_COLORS[stage] }}
                  />
                </div>
                <span className="text-[10px] text-muted-foreground font-mono w-24 text-right shrink-0">
                  {fmtDuration(stageSec)} — {Math.round(pct)}%
                </span>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Hypnogram */}
      <Card className="glass-card p-5">
        <div className="flex items-center gap-2 mb-3">
          <BrainCircuit className="h-3.5 w-3.5 text-violet-400" />
          <h3 className="text-sm font-medium">Sleep Architecture</h3>
          <span className="text-[10px] font-mono text-violet-400/70 bg-violet-500/10 px-1.5 py-0.5 rounded-full ml-auto">EEG</span>
        </div>
        <SleepHypnogram stageHistory={stageHistory} totalSeconds={elapsed} height={72} />
      </Card>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3">
        <Card className="glass-card p-4 text-center">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Duration</p>
          <p className="text-xl font-mono font-bold">{fmtDuration(elapsed)}</p>
        </Card>
        <Card className="glass-card p-4 text-center">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Dreams</p>
          <p className="text-xl font-mono font-bold" style={{ color: "hsl(270, 75%, 60%)" }}>
            {dreamsDetected}
          </p>
        </Card>
        <Card className="glass-card p-4 text-center">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">REM</p>
          <p className="text-xl font-mono font-bold" style={{ color: STAGE_COLORS.REM }}>
            {totalSeconds > 0 ? Math.round((tally.REM / totalSeconds) * 100) : 0}%
          </p>
        </Card>
      </div>

      {/* Actions */}
      <div className="flex gap-3 justify-center pt-2">
        <Button
          onClick={handleReset}
          variant="ghost"
          className="border border-border/30"
        >
          New Session
        </Button>
        <Link href="/dreams">
          <Button className="bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30">
            <Sparkles className="h-4 w-4 mr-2" />
            View Dream Journal
          </Button>
        </Link>
      </div>

      {/* CGM + Sleep Overlay */}
      <div className="pt-4">
        <CGMSleepOverlay
          glucoseReadings={[]}
          sleepStages={[]}
          awakeningTimes={[]}
          metrics={undefined}
        />
      </div>
    </main>
  );
}
