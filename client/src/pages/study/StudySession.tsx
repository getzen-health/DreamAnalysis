/**
 * Unified study session page.
 *
 * Handles both stress and food blocks in one page.
 * Flow:
 *   block-pick  → (if not pre-selected via URL ?block=)
 *   muse-pair   → Bluetooth connect OR skip to simulation
 *   baseline    → 5 min resting EEG
 *   task        → 15 min work/food task; stress > 0.65 auto-advances
 *   intervention → 3 min box breathing
 *   recovery    → 5 min post-intervention EEG
 *   survey      → 3 short questions, submit → /study/complete
 *
 * EEG is checkpointed to DB every 30 seconds.
 * Muse disconnect mid-session saves partial data and shows reconnect overlay.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation, useSearch } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Loader2, Bluetooth, CheckCircle2, Wind, Utensils, Brain } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { pingBackend } from "@/lib/ml-api";
import { useToast } from "@/hooks/use-toast";
import { useDevice } from "@/hooks/use-device";

// ── Types ─────────────────────────────────────────────────────────────────────

type BlockType = "stress" | "food";
type Phase = "block-pick" | "muse-pair" | "baseline" | "task" | "intervention" | "recovery" | "survey";

interface EEGSnapshot {
  alpha: number;
  beta: number;
  theta: number;
  delta: number;
  gamma: number;
  stress_level: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function mockEEG(): EEGSnapshot {
  return {
    alpha: 0.3 + Math.random() * 0.2,
    beta: 0.2 + Math.random() * 0.3,
    theta: 0.1 + Math.random() * 0.15,
    delta: 0.05 + Math.random() * 0.1,
    gamma: 0.02 + Math.random() * 0.05,
    stress_level: 0.3 + Math.random() * 0.4,
  };
}

async function fetchSimEEG(): Promise<EEGSnapshot> {
  try {
    const res = await fetch("/api/simulate-eeg", { credentials: "include" });
    if (!res.ok) throw new Error("sim failed");
    return (await res.json()) as EEGSnapshot;
  } catch {
    return mockEEG();
  }
}

function avgSnapshots(snaps: EEGSnapshot[]): EEGSnapshot | null {
  if (!snaps.length) return null;
  const n = snaps.length;
  const s = snaps.reduce((a, b) => ({
    alpha: a.alpha + b.alpha,
    beta: a.beta + b.beta,
    theta: a.theta + b.theta,
    delta: a.delta + b.delta,
    gamma: a.gamma + b.gamma,
    stress_level: a.stress_level + b.stress_level,
  }), { alpha: 0, beta: 0, theta: 0, delta: 0, gamma: 0, stress_level: 0 });
  return { alpha: s.alpha/n, beta: s.beta/n, theta: s.theta/n, delta: s.delta/n, gamma: s.gamma/n, stress_level: s.stress_level/n };
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`;
}

function stressColor(level: number): string {
  if (level < 0.4) return "bg-green-500";
  if (level <= 0.65) return "bg-yellow-400";
  return "bg-red-500";
}

// ── Countdown hook ────────────────────────────────────────────────────────────

function useCountdown(totalSec: number, active: boolean, onDone: () => void) {
  const [remaining, setRemaining] = useState(totalSec);
  const doneRef = useRef(false);

  useEffect(() => {
    setRemaining(totalSec);
    doneRef.current = false;
  }, [totalSec]);

  useEffect(() => {
    if (!active) return;
    if (doneRef.current) return;
    const id = setInterval(() => {
      setRemaining((prev) => {
        if (prev <= 1) {
          clearInterval(id);
          if (!doneRef.current) { doneRef.current = true; onDone(); }
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  return remaining;
}

// ── Stepper ───────────────────────────────────────────────────────────────────

const PHASE_STEPS: Phase[] = ["baseline", "task", "intervention", "recovery", "survey"];
const PHASE_LABELS: Record<string, string> = {
  baseline: "Baseline", task: "Task", intervention: "Breathing", recovery: "Recovery", survey: "Survey",
};

function Stepper({ phase }: { phase: Phase }) {
  const idx = PHASE_STEPS.indexOf(phase);
  if (idx === -1) return null;
  return (
    <div className="w-full">
      <p className="text-xs text-muted-foreground mb-3 text-center">Step {idx + 1} of 5</p>
      <div className="flex items-center gap-0">
        {PHASE_STEPS.map((p, i) => (
          <div key={p} className="flex items-center flex-1">
            <div className="flex flex-col items-center w-full">
              <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold shrink-0 transition-colors ${
                i < idx ? "bg-primary text-primary-foreground" :
                i === idx ? "bg-primary text-primary-foreground ring-2 ring-primary/40" :
                "bg-muted text-muted-foreground"
              }`}>
                {i < idx ? <CheckCircle2 className="h-3.5 w-3.5" /> : i + 1}
              </div>
              <span className="text-[10px] text-muted-foreground mt-1 text-center">{PHASE_LABELS[p]}</span>
            </div>
            {i < PHASE_STEPS.length - 1 && (
              <div className={`h-0.5 flex-1 mb-4 mx-0.5 transition-colors ${i < idx ? "bg-primary" : "bg-muted"}`} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Box breathing animation ───────────────────────────────────────────────────

function BoxBreathing() {
  const labels = ["Inhale 4 counts", "Hold 4 counts", "Exhale 4 counts", "Hold 4 counts"];
  const [step, setStep] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setStep((p) => (p + 1) % 4), 4000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex flex-col items-center gap-6 py-4">
      <div className="relative flex items-center justify-center transition-all duration-[3800ms] ease-in-out"
        style={{ width: step === 0 || step === 2 ? 160 : 120, height: step === 0 || step === 2 ? 160 : 120 }}>
        <div className="rounded-2xl bg-primary/20 border-2 border-primary/60 absolute inset-0 transition-all duration-[3800ms] ease-in-out" />
        <span className="text-sm font-semibold text-primary z-10">{["Inhale","Hold","Exhale","Hold"][step]}</span>
      </div>
      <p className="text-base font-medium text-foreground">{labels[step]}</p>
      <div className="flex gap-2">
        {[0,1,2,3].map(i => (
          <div key={i} className={`h-1.5 w-8 rounded-full transition-colors ${i === step ? "bg-primary" : "bg-muted"}`} />
        ))}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function StudySession() {
  const [, navigate] = useLocation();
  const search = useSearch();
  const { toast } = useToast();

  const params = new URLSearchParams(search);
  const participantCode = params.get("code") ?? localStorage.getItem("ndw_study_code") ?? "";
  const preBlock = params.get("block") as BlockType | null;

  // Device (Muse BLE)
  const { state: deviceState, connect, latestFrame } = useDevice();

  // Phase state
  const [blockType, setBlockType] = useState<BlockType | null>(preBlock);
  const [phase, setPhase] = useState<Phase>(preBlock ? "muse-pair" : "block-pick");
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [useSimulation, setUseSimulation] = useState(false);
  const [isStarting, setIsStarting] = useState(false);

  // EEG accumulation
  const preReadings = useRef<EEGSnapshot[]>([]);
  const postReadings = useRef<EEGSnapshot[]>([]);
  const phaseLog = useRef<{ phase: string; at: number }[]>([]);

  // Derived EEG snapshots
  const [liveStress, setLiveStress] = useState(0);
  const [showStress, setShowStress] = useState(false);
  const [interventionTriggered, setInterventionTriggered] = useState(false);
  const [preEegJson, setPreEegJson] = useState<EEGSnapshot | null>(null);
  const [postEegJson, setPostEegJson] = useState<EEGSnapshot | null>(null);

  // Survey
  const [surveyQ1, setSurveyQ1] = useState<number | null>(null);
  const [surveyQ2, setSurveyQ2] = useState<number | null>(null);
  const [surveyQ3, setSurveyQ3] = useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [backendReady, setBackendReady] = useState<boolean | null>(null);

  // Phase timer flags
  const [baselineActive, setBaselineActive] = useState(false);
  const [taskActive, setTaskActive] = useState(false);
  const [recoveryActive, setRecoveryActive] = useState(false);

  // 30s checkpoint interval
  const checkpointTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Session start ─────────────────────────────────────────────────────────

  const startSession = useCallback(async (block: BlockType) => {
    setIsStarting(true);
    try {
      const res = await apiRequest("POST", "/api/study/session/start", {
        participant_code: participantCode,
        block_type: block,
      });
      const data = (await res.json()) as { session_id: number };
      setSessionId(data.session_id);
      phaseLog.current.push({ phase: "baseline", at: Date.now() });
      setPhase("baseline");
      setBaselineActive(true);
      startCheckpointLoop(data.session_id);
    } catch {
      // Offline fallback — run session locally with mock ID
      setSessionId(-1);
      phaseLog.current.push({ phase: "baseline", at: Date.now() });
      setPhase("baseline");
      setBaselineActive(true);
    } finally {
      setIsStarting(false);
    }
  }, [participantCode]);

  // ── Warm up ML backend when muse-pair screen appears ─────────────────────

  useEffect(() => {
    if (phase !== "muse-pair") return;
    setBackendReady(null);
    pingBackend().then((ok) => setBackendReady(ok));
  }, [phase]);

  // ── Auto-start on Muse connect ────────────────────────────────────────────

  useEffect(() => {
    if (phase === "muse-pair" && (deviceState === "connected" || deviceState === "streaming") && blockType && !sessionId) {
      startSession(blockType);
    }
  }, [deviceState, phase, blockType, sessionId, startSession]);

  // ── Muse disconnect mid-session ───────────────────────────────────────────

  useEffect(() => {
    if (!useSimulation && deviceState === "disconnected" && sessionId && !["block-pick","muse-pair","survey"].includes(phase)) {
      if (checkpointTimer.current) clearInterval(checkpointTimer.current);
      saveCheckpoint(sessionId, false);
      toast({ title: "Muse disconnected", description: "Reconnect to continue, or save and exit.", variant: "destructive" });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deviceState]);

  // ── 30s checkpoint loop ───────────────────────────────────────────────────

  function startCheckpointLoop(sid: number) {
    if (checkpointTimer.current) clearInterval(checkpointTimer.current);
    checkpointTimer.current = setInterval(() => saveCheckpoint(sid, false), 30_000);
  }

  async function saveCheckpoint(sid: number, isFinal: boolean) {
    if (sid < 0) return; // offline session
    const pre = avgSnapshots(preReadings.current);
    const post = avgSnapshots(postReadings.current);
    try {
      await apiRequest("PATCH", `/api/study/session/${sid}/checkpoint`, {
        ...(pre  && { pre_eeg_json: pre }),
        ...(post && { post_eeg_json: post }),
        eeg_features_json: { frame_count: preReadings.current.length + postReadings.current.length, last_stress: liveStress },
        intervention_triggered: interventionTriggered,
        phase_log: phaseLog.current,
        ...(isFinal && { partial: false }),
      });
    } catch {
      // non-fatal
    }
  }

  async function saveAndExit() {
    if (sessionId && sessionId > 0) {
      await apiRequest("PATCH", `/api/study/session/${sessionId}/checkpoint`, { partial: true }).catch(() => {});
    }
    navigate(`/study/complete?code=${participantCode}&partial=true`);
  }

  // ── EEG polling (simulation mode) ────────────────────────────────────────

  useEffect(() => {
    if (!useSimulation || !baselineActive) return;
    const id = setInterval(async () => {
      const r = await fetchSimEEG();
      preReadings.current.push(r);
    }, 4000);
    return () => clearInterval(id);
  }, [useSimulation, baselineActive]);

  useEffect(() => {
    if (!useSimulation || !taskActive) return;
    const id = setInterval(async () => {
      const r = await fetchSimEEG();
      preReadings.current.push(r);
      setLiveStress(r.stress_level);
      if (r.stress_level > 0.65 && !interventionTriggered) {
        setInterventionTriggered(true);
        setTaskActive(false);
        phaseLog.current.push({ phase: "intervention", at: Date.now() });
        setPhase("intervention");
      }
    }, 4000);
    return () => clearInterval(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [useSimulation, taskActive]);

  useEffect(() => {
    if (!useSimulation || !recoveryActive) return;
    const id = setInterval(async () => {
      const r = await fetchSimEEG();
      postReadings.current.push(r);
    }, 4000);
    return () => clearInterval(id);
  }, [useSimulation, recoveryActive]);

  // ── EEG from live Muse device ─────────────────────────────────────────────

  useEffect(() => {
    if (useSimulation || !latestFrame || !sessionId) return;
    const stress = latestFrame.analysis?.emotions?.stress_index ?? 0;
    setLiveStress(stress);

    const snap: EEGSnapshot = {
      alpha: latestFrame.analysis?.band_powers?.alpha ?? 0,
      beta: latestFrame.analysis?.band_powers?.beta ?? 0,
      theta: latestFrame.analysis?.band_powers?.theta ?? 0,
      delta: latestFrame.analysis?.band_powers?.delta ?? 0,
      gamma: latestFrame.analysis?.band_powers?.gamma ?? 0,
      stress_level: stress,
    };

    if (phase === "baseline") preReadings.current.push(snap);
    if (phase === "recovery") postReadings.current.push(snap);

    if (phase === "task" && stress > 0.65 && !interventionTriggered) {
      setInterventionTriggered(true);
      setTaskActive(false);
      phaseLog.current.push({ phase: "intervention", at: Date.now() });
      setPhase("intervention");
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame]);

  // ── Phase transitions ─────────────────────────────────────────────────────

  const onBaselineDone = useCallback(() => {
    setBaselineActive(false);
    setPreEegJson(avgSnapshots(preReadings.current));
    phaseLog.current.push({ phase: "task", at: Date.now() });
    setPhase("task");
    setTaskActive(true);
  }, []);

  const onTaskDone = useCallback(() => {
    setTaskActive(false);
    phaseLog.current.push({ phase: "intervention", at: Date.now() });
    setPhase("intervention");
  }, []);

  const onInterventionDone = useCallback(() => {
    phaseLog.current.push({ phase: "recovery", at: Date.now() });
    setPhase("recovery");
    setRecoveryActive(true);
  }, []);

  const onRecoveryDone = useCallback(() => {
    setRecoveryActive(false);
    setPostEegJson(avgSnapshots(postReadings.current));
    if (checkpointTimer.current) clearInterval(checkpointTimer.current);
    if (sessionId) saveCheckpoint(sessionId, true);
    phaseLog.current.push({ phase: "survey", at: Date.now() });
    setPhase("survey");
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // Timers
  const baselineRemaining   = useCountdown(5 * 60,  baselineActive,    onBaselineDone);
  const taskRemaining       = useCountdown(15 * 60, taskActive,        onTaskDone);
  const interventionRemaining = useCountdown(3 * 60, phase === "intervention", onInterventionDone);
  const recoveryRemaining   = useCountdown(5 * 60,  recoveryActive,    onRecoveryDone);

  // ── Survey submit ─────────────────────────────────────────────────────────

  // Track peak stress across the whole session
  const peakStressRef = useRef(0);
  useEffect(() => {
    if (liveStress > peakStressRef.current) peakStressRef.current = liveStress;
  }, [liveStress]);

  const canSubmitSurvey = surveyQ1 !== null && surveyQ2 !== null && surveyQ3 !== null;

  async function handleSurveySubmit() {
    if (!canSubmitSurvey || isSubmitting) return;
    setIsSubmitting(true);
    const finalPre  = preEegJson  ?? avgSnapshots(preReadings.current);
    const finalPost = postEegJson ?? avgSnapshots(postReadings.current);
    const surveyData = { q1: surveyQ1, q2: surveyQ2, q3: surveyQ3 };
    try {
      await apiRequest("POST", "/api/study/session/complete", {
        session_id: sessionId,
        pre_eeg_json:  finalPre,
        post_eeg_json: finalPost,
        eeg_features_json: { frame_count: preReadings.current.length + postReadings.current.length },
        survey_json: surveyData,
        intervention_triggered: interventionTriggered,
      });
      const preS  = (finalPre?.stress_level  ?? 0).toFixed(2);
      const postS = (finalPost?.stress_level ?? 0).toFixed(2);
      const peak  = peakStressRef.current.toFixed(2);
      navigate(
        `/study/complete?code=${encodeURIComponent(participantCode)}&done=${blockType}` +
        `&pre_stress=${preS}&peak_stress=${peak}&post_stress=${postS}`
      );
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Submit failed — try again";
      toast({ title: "Submit failed", description: msg, variant: "destructive" });
    } finally {
      setIsSubmitting(false);
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────

  // Block picker
  if (phase === "block-pick") {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center px-4">
        <div className="max-w-md w-full space-y-6">
          <div className="text-center space-y-2">
            <h1 className="text-2xl font-bold">Choose your session</h1>
            <p className="text-sm text-muted-foreground">Which block would you like to complete?</p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Card className="cursor-pointer border-2 hover:border-primary transition-colors"
              onClick={() => { setBlockType("stress"); setPhase("muse-pair"); }}>
              <CardContent className="pt-8 pb-8 text-center space-y-3">
                <Brain className="w-10 h-10 mx-auto text-primary" />
                <div>
                  <p className="font-semibold">Stress Block</p>
                  <p className="text-xs text-muted-foreground mt-1">15 min work task + breathing</p>
                </div>
              </CardContent>
            </Card>
            <Card className="cursor-pointer border-2 hover:border-primary transition-colors"
              onClick={() => { setBlockType("food"); setPhase("muse-pair"); }}>
              <CardContent className="pt-8 pb-8 text-center space-y-3">
                <Utensils className="w-10 h-10 mx-auto text-violet-400" />
                <div>
                  <p className="font-semibold">Food Block</p>
                  <p className="text-xs text-muted-foreground mt-1">Food cue task + breathing</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    );
  }

  // Muse pair screen
  if (phase === "muse-pair") {
    const isConnecting   = deviceState === "connecting";
    const isConnected    = deviceState === "connected" || deviceState === "streaming";
    const backendChecking = backendReady === null;

    return (
      <div className="min-h-screen bg-background flex items-center justify-center px-4">
        <div className="max-w-md w-full space-y-6">
          <div className="text-center space-y-2">
            <Bluetooth className="w-10 h-10 mx-auto text-primary" />
            <h1 className="text-2xl font-bold">Connect your Muse 2</h1>
            <p className="text-sm text-muted-foreground">
              Put on the headband, then tap Pair. The session starts automatically.
            </p>
          </div>

          <Card>
            <CardContent className="pt-6 space-y-4">
              {/* Backend warm-up status */}
              {backendChecking && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Waking up ML backend…
                </div>
              )}
              {backendReady === false && (
                <p className="text-xs text-amber-400">
                  ML backend unreachable. Check your backend URL in Settings or start it locally.
                </p>
              )}

              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Device status</span>
                <Badge variant="outline" className={isConnected ? "border-green-500/50 text-green-400" : "border-muted"}>
                  {isConnected ? "Connected" : isConnecting ? "Connecting…" : "Not connected"}
                </Badge>
              </div>

              {!isConnected && (
                <Button className="w-full" disabled={isConnecting || backendChecking}
                  onClick={() => connect("muse_2")}>
                  {isConnecting ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Connecting…</> : <><Bluetooth className="mr-2 h-4 w-4" />Pair Muse 2</>}
                </Button>
              )}

              {isConnected && isStarting && (
                <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Starting session…
                </div>
              )}
            </CardContent>
          </Card>

          <div className="text-center">
            <button className="text-xs text-muted-foreground hover:text-foreground underline"
              onClick={() => { setUseSimulation(true); if (blockType) startSession(blockType); }}>
              Continue without Muse (simulation mode)
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Starting overlay
  if (isStarting) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 text-muted-foreground">
          <Loader2 className="h-8 w-8 animate-spin" />
          <p className="text-sm">Setting up your session…</p>
        </div>
      </div>
    );
  }

  // Disconnected mid-session overlay
  const showDisconnectOverlay = !useSimulation && deviceState === "disconnected" &&
    sessionId !== null && !["block-pick","muse-pair","survey"].includes(phase);

  // ── Active session render ─────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-lg mx-auto px-4 py-8 space-y-6">

        <Stepper phase={phase} />

        {/* Disconnect overlay */}
        {showDisconnectOverlay && (
          <Card className="border-destructive/50 bg-destructive/5">
            <CardContent className="pt-5 space-y-3">
              <p className="text-sm font-medium text-destructive">Muse disconnected</p>
              <p className="text-xs text-muted-foreground">Your data has been saved. Reconnect to continue.</p>
              <div className="flex gap-2">
                <Button size="sm" onClick={() => connect("muse_2")}>Reconnect</Button>
                <Button size="sm" variant="outline" onClick={saveAndExit}>Save & Exit</Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* ── Baseline ── */}
        {phase === "baseline" && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Resting Baseline</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground leading-relaxed">
                Sit comfortably with eyes closed. Breathe naturally.
              </p>
              <div className="flex flex-col items-center gap-3">
                <span className="text-5xl font-mono font-bold tracking-tight">{formatTime(baselineRemaining)}</span>
                <Progress value={((5*60 - baselineRemaining) / (5*60)) * 100} className="h-2" />
              </div>
              <div className="flex items-center justify-center gap-2">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
                </span>
                <span className="text-sm text-muted-foreground">
                  Recording… <span className="font-medium text-foreground">{preReadings.current.length} samples</span>
                </span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* ── Task ── */}
        {phase === "task" && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">
                {blockType === "stress" ? "Work Session" : "Food Cue Task"}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground leading-relaxed">
                {blockType === "stress"
                  ? "Work on a challenging task. The breathing exercise will begin automatically when needed."
                  : "Think about your favourite food or look at the food images. Stay relaxed."}
              </p>
              <div className="flex flex-col items-center gap-3">
                <span className="text-5xl font-mono font-bold tracking-tight">{formatTime(taskRemaining)}</span>
                <Progress value={((15*60 - taskRemaining) / (15*60)) * 100} className="h-2" />
              </div>

              {/* Optional stress bar */}
              <div className="space-y-2">
                <button
                  className="text-xs text-muted-foreground hover:text-foreground underline"
                  onClick={() => setShowStress((v) => !v)}
                >
                  {showStress ? "Hide stress level" : "Show my stress level"}
                </button>
                {showStress && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Stress</span>
                      <span>{Math.round(liveStress * 100)}%</span>
                    </div>
                    <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                      <div className={`h-full rounded-full transition-all ${stressColor(liveStress)}`}
                        style={{ width: `${Math.round(liveStress * 100)}%` }} />
                    </div>
                  </div>
                )}
              </div>

              <Button variant="outline" size="sm" className="w-full"
                onClick={() => { setTaskActive(false); phaseLog.current.push({ phase: "intervention", at: Date.now() }); setPhase("intervention"); }}>
                End task early
              </Button>
            </CardContent>
          </Card>
        )}

        {/* ── Intervention ── */}
        {phase === "intervention" && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-lg">
                <Wind className="h-5 w-5 text-primary" />
                Breathing Exercise
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {interventionTriggered && (
                <p className="text-xs text-amber-400 font-medium">Stress threshold reached — intervention started automatically.</p>
              )}
              <BoxBreathing />
              <div className="flex flex-col items-center gap-2">
                <span className="text-2xl font-mono font-bold">{formatTime(interventionRemaining)}</span>
                <Progress value={((3*60 - interventionRemaining) / (3*60)) * 100} className="h-1.5 w-full" />
              </div>
            </CardContent>
          </Card>
        )}

        {/* ── Recovery ── */}
        {phase === "recovery" && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Recovery Baseline</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground leading-relaxed">
                Great work! Sit quietly with eyes closed while we record your post-session EEG.
              </p>
              <div className="flex flex-col items-center gap-3">
                <span className="text-5xl font-mono font-bold tracking-tight">{formatTime(recoveryRemaining)}</span>
                <Progress value={((5*60 - recoveryRemaining) / (5*60)) * 100} className="h-2" />
              </div>
              <div className="flex items-center justify-center gap-2">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500" />
                </span>
                <span className="text-sm text-muted-foreground">
                  Recording… <span className="font-medium text-foreground">{postReadings.current.length} samples</span>
                </span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* ── Survey ── */}
        {phase === "survey" && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Quick Survey</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground">Rate each on a scale from 1 (low) to 10 (high).</p>

              {[
                { label: blockType === "stress" ? "How stressed did you feel during the task?" : "How strong were your food cravings?", value: surveyQ1, set: setSurveyQ1 },
                { label: "How helpful was the breathing exercise?", value: surveyQ2, set: setSurveyQ2 },
                { label: "How calm do you feel right now?", value: surveyQ3, set: setSurveyQ3 },
              ].map(({ label, value, set }, i) => (
                <div key={i} className="space-y-3">
                  <p className="text-sm font-medium">{label}</p>
                  <Slider
                    min={1} max={10} step={1}
                    value={value !== null ? [value] : [5]}
                    onValueChange={([v]) => set(v)}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>1</span>
                    <span className="font-medium text-foreground">{value ?? "—"}</span>
                    <span>10</span>
                  </div>
                </div>
              ))}

              <Button className="w-full" disabled={!canSubmitSurvey || isSubmitting} onClick={handleSurveySubmit}>
                {isSubmitting ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Saving…</> : "Complete Session"}
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
