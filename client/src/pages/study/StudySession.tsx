/**
 * Unified study session page.
 *
 * Flow: block-pick → muse-pair → eeg (20 min recording) → survey → /study/complete
 * EEG is checkpointed to DB every 30 seconds.
 */

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useLocation, useSearch } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Loader2, Bluetooth, CheckCircle2, Utensils, Brain, AlertCircle, Watch, Mic } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { pingBackend, getMLApiUrl, type VoiceWatchEmotionResult } from "@/lib/ml-api";
import { VoiceWatchAnalyzer } from "@/components/voice-watch-analyzer";
import { healthSync } from "@/lib/health-sync";
import { getParticipantId } from "@/lib/participant";
import { useToast } from "@/hooks/use-toast";
import { useDevice } from "@/hooks/use-device";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

// ── Types ─────────────────────────────────────────────────────────────────────

type BlockType = "stress" | "food";
type Phase = "block-pick" | "muse-pair" | "eeg" | "survey";
const FALLBACK_USER_ID = getParticipantId();

interface EEGSnapshot {
  alpha: number;
  beta: number;
  theta: number;
  delta: number;
  gamma: number;
  stress_level: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function fetchSimEEG(): Promise<EEGSnapshot> {
  const res = await fetch(`${getMLApiUrl()}/api/simulate-eeg`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state: "rest", duration: 4 }),
  });
  if (!res.ok) throw new Error(`ML backend returned ${res.status}`);
  const data = await res.json();
  const emotions = data?.analysis?.emotions;
  const bands = emotions?.band_powers ?? {};
  return {
    alpha: bands.alpha ?? 0,
    beta: bands.beta ?? 0,
    theta: bands.theta ?? 0,
    delta: bands.delta ?? 0,
    gamma: bands.gamma ?? 0,
    stress_level: emotions?.stress_index ?? 0,
  };
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

const PHASE_STEPS: Phase[] = ["eeg", "survey"];
const PHASE_LABELS: Record<string, string> = { eeg: "EEG Recording", survey: "Survey" };

function Stepper({ phase }: { phase: Phase }) {
  const idx = PHASE_STEPS.indexOf(phase);
  if (idx === -1) return null;
  return (
    <div className="w-full">
      <p className="text-xs text-muted-foreground mb-3 text-center">Step {idx + 1} of 2</p>
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

// ── Main component ────────────────────────────────────────────────────────────

export default function StudySession() {
  const [, navigate] = useLocation();
  const search = useSearch();
  const { toast } = useToast();

  const params = new URLSearchParams(search);
  const participantCode = params.get("code") ?? localStorage.getItem("ndw_study_code") ?? "";
  const preBlock = params.get("block") as BlockType | null;

  const { state: deviceState, connect, latestFrame } = useDevice();

  const [blockType, setBlockType] = useState<BlockType | null>(preBlock);
  const [phase, setPhase] = useState<Phase>(preBlock ? "muse-pair" : "block-pick");
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [useSimulation, setUseSimulation] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [eegActive, setEegActive] = useState(false);
  const [backendReady, setBackendReady] = useState<boolean | null>(null);
  const [simError, setSimError] = useState<string | null>(null);

  const readings = useRef<EEGSnapshot[]>([]);
  const [eegJson, setEegJson] = useState<EEGSnapshot | null>(null);

  // Voice emotion results (accumulated across session)
  const voiceResults = useRef<VoiceWatchEmotionResult[]>([]);

  // Survey
  const [surveyQ1, setSurveyQ1] = useState<number | null>(null);
  const [surveyQ2, setSurveyQ2] = useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const checkpointTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  // Live chart data for emotion & stress during EEG recording
  interface ChartPoint { t: number; stress: number; alpha: number; beta: number; theta: number; }
  const [chartData, setChartData] = useState<ChartPoint[]>([]);

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
      setPhase("eeg");
      setEegActive(true);
      startCheckpointLoop(data.session_id);
    } catch {
      setSessionId(-1);
      setPhase("eeg");
      setEegActive(true);
    } finally {
      setIsStarting(false);
    }
  }, [participantCode]);

  // ── Warm up ML backend ────────────────────────────────────────────────────

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
    if (!useSimulation && deviceState === "disconnected" && sessionId && phase === "eeg") {
      if (checkpointTimer.current) clearInterval(checkpointTimer.current);
      saveCheckpoint(sessionId, false);
      toast({ title: "Muse disconnected", description: "Reconnect to continue, or save and exit.", variant: "destructive" });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deviceState]);

  // ── Checkpoint loop ───────────────────────────────────────────────────────

  function startCheckpointLoop(sid: number) {
    if (checkpointTimer.current) clearInterval(checkpointTimer.current);
    checkpointTimer.current = setInterval(() => saveCheckpoint(sid, false), 30_000);
  }

  function getWatchBiometrics() {
    const state = healthSync.getState();
    return state.latestPayload ?? null;
  }

  async function saveCheckpoint(sid: number, isFinal: boolean) {
    if (sid < 0) return;
    const snap = avgSnapshots(readings.current);
    const watchData = getWatchBiometrics();
    try {
      await apiRequest("PATCH", `/api/study/session/${sid}/checkpoint`, {
        ...(snap && { pre_eeg_json: snap }),
        eeg_features_json: { frame_count: readings.current.length },
        ...(isFinal && { partial: false }),
        ...(voiceResults.current.length > 0 && { voice_emotion_json: voiceResults.current }),
        ...(watchData && { watch_biometrics_json: watchData }),
      });
    } catch { /* non-fatal */ }
  }

  function onVoiceResult(result: VoiceWatchEmotionResult) {
    voiceResults.current.push(result);
    // Save immediately to DB
    if (sessionId && sessionId > 0) {
      saveCheckpoint(sessionId, false);
    }
  }

  async function saveAndExit() {
    if (sessionId && sessionId > 0) {
      await apiRequest("PATCH", `/api/study/session/${sessionId}/checkpoint`, { partial: true }).catch(() => {});
    }
    navigate(`/study/complete?code=${participantCode}&partial=true`);
  }

  // ── EEG polling (simulation) ──────────────────────────────────────────────

  useEffect(() => {
    if (!useSimulation || !eegActive) return;
    setSimError(null);
    const id = setInterval(async () => {
      try {
        const r = await fetchSimEEG();
        readings.current.push(r);
        setChartData((prev) => [...prev.slice(-149), {
          t: prev.length, stress: r.stress_level, alpha: r.alpha, beta: r.beta, theta: r.theta,
        }]);
        setSimError(null);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "ML backend unreachable";
        setSimError(msg);
      }
    }, 4000);
    return () => clearInterval(id);
  }, [useSimulation, eegActive]);

  // ── EEG from live Muse ────────────────────────────────────────────────────

  useEffect(() => {
    if (useSimulation || !latestFrame || !sessionId || phase !== "eeg") return;
    const snap: EEGSnapshot = {
      alpha: latestFrame.analysis?.band_powers?.alpha ?? 0,
      beta: latestFrame.analysis?.band_powers?.beta ?? 0,
      theta: latestFrame.analysis?.band_powers?.theta ?? 0,
      delta: latestFrame.analysis?.band_powers?.delta ?? 0,
      gamma: latestFrame.analysis?.band_powers?.gamma ?? 0,
      stress_level: latestFrame.analysis?.emotions?.stress_index ?? 0,
    };
    readings.current.push(snap);
    // Throttle chart updates to ~every 4th frame to avoid excessive re-renders
    if (readings.current.length % 4 === 0) {
      setChartData((prev) => [...prev.slice(-149), {
        t: prev.length, stress: snap.stress_level, alpha: snap.alpha, beta: snap.beta, theta: snap.theta,
      }]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame]);

  // ── EEG done → survey ─────────────────────────────────────────────────────

  const onEegDone = useCallback(() => {
    setEegActive(false);
    const snap = avgSnapshots(readings.current);
    setEegJson(snap);
    if (checkpointTimer.current) clearInterval(checkpointTimer.current);
    if (sessionId) saveCheckpoint(sessionId, true);
    setPhase("survey");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const eegRemaining = useCountdown(20 * 60, eegActive, onEegDone);

  // ── Survey submit ─────────────────────────────────────────────────────────

  const canSubmit = surveyQ1 !== null && surveyQ2 !== null;

  async function handleSurveySubmit() {
    if (!canSubmit || isSubmitting) return;
    setIsSubmitting(true);
    const snap = eegJson ?? avgSnapshots(readings.current);
    try {
      const watchData = getWatchBiometrics();
      await apiRequest("POST", "/api/study/session/complete", {
        session_id: sessionId,
        pre_eeg_json: snap,
        post_eeg_json: snap,
        eeg_features_json: { frame_count: readings.current.length },
        survey_json: { q1: surveyQ1, q2: surveyQ2 },
        intervention_triggered: false,
        voice_emotion_json: voiceResults.current.length > 0 ? voiceResults.current : null,
        watch_biometrics_json: watchData,
      });
      const avg = (snap?.stress_level ?? 0).toFixed(2);
      navigate(
        `/study/complete?code=${encodeURIComponent(participantCode)}&done=${blockType}&pre_stress=${avg}&peak_stress=${avg}&post_stress=${avg}`
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
                  <p className="text-xs text-muted-foreground mt-1">20 min EEG + survey</p>
                </div>
              </CardContent>
            </Card>
            <Card className="cursor-pointer border-2 hover:border-primary transition-colors"
              onClick={() => { setBlockType("food"); setPhase("muse-pair"); }}>
              <CardContent className="pt-8 pb-8 text-center space-y-3">
                <Utensils className="w-10 h-10 mx-auto text-violet-400" />
                <div>
                  <p className="font-semibold">Food Block</p>
                  <p className="text-xs text-muted-foreground mt-1">20 min EEG + survey</p>
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
    const isConnecting = deviceState === "connecting";
    const isConnected  = deviceState === "connected" || deviceState === "streaming";
    const isAndroid = typeof navigator !== "undefined" && /Android/i.test(navigator.userAgent);
    const hasWebBluetooth = typeof navigator !== "undefined" && "bluetooth" in navigator;

    return (
      <div className="min-h-screen bg-background flex items-center justify-center px-4">
        <div className="max-w-md w-full space-y-5">
          <div className="text-center space-y-2">
            <h1 className="text-2xl font-bold">Connect Devices</h1>
            <p className="text-sm text-muted-foreground">
              Connect your EEG headband and/or Apple Watch to start collecting data.
            </p>
          </div>

          {/* ── Option 1: Muse EEG ── */}
          <Card>
            <CardContent className="pt-5 space-y-4">
              <div className="flex items-center gap-2">
                <Bluetooth className="h-5 w-5 text-primary" />
                <p className="font-semibold text-sm">Muse 2 EEG Headband</p>
              </div>

              {backendReady === null && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Waking up ML backend…
                </div>
              )}
              {backendReady === false && (
                <p className="text-xs text-amber-400">
                  ML backend unreachable — EEG analysis will not work until it is online.
                </p>
              )}

              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Status</span>
                <Badge variant="outline" className={isConnected ? "border-green-500/50 text-green-400" : "border-muted"}>
                  {isConnected ? "Connected" : isConnecting ? "Connecting…" : "Not connected"}
                </Badge>
              </div>

              {!hasWebBluetooth && (
                <p className="text-xs text-amber-400">
                  Web Bluetooth is not supported in this browser. Use Chrome on desktop or Android.
                </p>
              )}

              {!isConnected && hasWebBluetooth && (
                <Button className="w-full" disabled={isConnecting || backendReady === null}
                  onClick={async () => {
                    try {
                      await connect("muse_2");
                    } catch (err) {
                      const msg = err instanceof Error ? err.message : "Connection failed";
                      toast({ title: "Muse connection failed", description: msg, variant: "destructive" });
                    }
                  }}>
                  {isConnecting ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Connecting…</> : <><Bluetooth className="mr-2 h-4 w-4" />Pair Muse 2</>}
                </Button>
              )}

              {isConnected && isStarting && (
                <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Starting session…
                </div>
              )}

              {/* Android tips */}
              {isAndroid && !isConnected && (
                <div className="rounded-md bg-muted/50 p-3 space-y-1">
                  <p className="text-xs font-medium text-muted-foreground">Android tips:</p>
                  <ul className="text-xs text-muted-foreground space-y-0.5 list-disc list-inside">
                    <li>Turn on Location Services (required for BLE)</li>
                    <li>Unpair Muse from Android Bluetooth settings first</li>
                    <li>Use Chrome browser</li>
                  </ul>
                </div>
              )}
            </CardContent>
          </Card>

          {/* ── Option 2: Apple Watch / Health Data ── */}
          <Card>
            <CardContent className="pt-5 space-y-3">
              <div className="flex items-center gap-2">
                <Watch className="h-5 w-5 text-green-400" />
                <p className="font-semibold text-sm">Apple Watch / Health Data</p>
              </div>
              <p className="text-xs text-muted-foreground">
                Apple Health data (heart rate, HRV, SpO2) is automatically read if you are
                using the native iOS app. On the web, the Voice + Watch analyzer below can
                read biometric data from your paired watch.
              </p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Status</span>
                <Badge variant="outline" className="border-muted">
                  Web — use Voice + Watch below
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* ── Option 3: Voice + Watch (headband-free) ── */}
          <Card>
            <CardContent className="pt-5 space-y-3">
              <div className="flex items-center gap-2">
                <Mic className="h-5 w-5 text-violet-400" />
                <p className="font-semibold text-sm">Voice + Watch Emotion (no headband)</p>
              </div>
              <p className="text-xs text-muted-foreground">
                Record 30 seconds of voice to detect emotion from speech patterns.
                If Apple Watch data is available, it blends heart rate and HRV for better accuracy.
              </p>
              <VoiceWatchAnalyzer userId={participantCode || FALLBACK_USER_ID} onResult={onVoiceResult} />
            </CardContent>
          </Card>

          {/* ── Start session ── */}
          <div className="space-y-2">
            {isConnected ? (
              <p className="text-xs text-center text-green-400">
                Muse connected — session will start automatically.
              </p>
            ) : (
              <Button className="w-full" variant="outline" size="lg"
                onClick={() => { if (blockType) startSession(blockType); }}>
                Start Session Without Muse
              </Button>
            )}
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

  const showDisconnectOverlay = !useSimulation && deviceState === "disconnected" &&
    sessionId !== null && phase === "eeg";

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

        {/* ── Simulation error banner ── */}
        {simError && phase === "eeg" && (
          <Card className="border-destructive/50 bg-destructive/5">
            <CardContent className="pt-5 flex items-start gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-destructive">ML backend unreachable</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Simulated EEG data cannot be fetched. Check that the ML backend is running or update the URL in Settings.
                </p>
                <p className="text-xs text-muted-foreground mt-1 font-mono">{simError}</p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* ── EEG Recording ── */}
        {phase === "eeg" && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">EEG Recording</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground leading-relaxed">
                {blockType === "stress"
                  ? "Sit comfortably. You may work, read, or simply rest while wearing the headband."
                  : "Sit comfortably. You may eat your meal or think about food while wearing the headband."}
              </p>
              <div className="flex flex-col items-center gap-3">
                <span className="text-5xl font-mono font-bold tracking-tight">{formatTime(eegRemaining)}</span>
                <Progress value={((20 * 60 - eegRemaining) / (20 * 60)) * 100} className="h-2" />
              </div>
              <div className="flex items-center justify-center gap-2">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
                </span>
                <span className="text-sm text-muted-foreground">
                  Recording… <span className="font-medium text-foreground">{readings.current.length} samples</span>
                </span>
              </div>
              {/* Live emotion & stress chart */}
              {chartData.length > 1 && (
                <div className="space-y-2">
                  <p className="text-xs font-medium text-muted-foreground">Emotion & Stress (live)</p>
                  <div className="h-40 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                        <XAxis dataKey="t" hide />
                        <YAxis domain={[0, 1]} tickCount={3} tick={{ fontSize: 10 }} />
                        <Tooltip
                          contentStyle={{ fontSize: 11, background: "hsl(var(--card))", border: "1px solid hsl(var(--border))" }}
                          formatter={(v: number, name: string) => [`${(v * 100).toFixed(0)}%`, name]}
                        />
                        <Area type="monotone" dataKey="stress" stroke="#ef4444" fill="#ef444433" name="Stress" strokeWidth={2} dot={false} />
                        <Area type="monotone" dataKey="alpha" stroke="#22c55e" fill="#22c55e22" name="Alpha" strokeWidth={1.5} dot={false} />
                        <Area type="monotone" dataKey="beta" stroke="#3b82f6" fill="#3b82f622" name="Beta" strokeWidth={1.5} dot={false} />
                        <Area type="monotone" dataKey="theta" stroke="#a855f7" fill="#a855f722" name="Theta" strokeWidth={1.5} dot={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              <Button variant="outline" size="sm" className="w-full" onClick={onEegDone}>
                End early & go to survey
              </Button>

              {/* Voice emotion during recording */}
              <div className="pt-2 border-t border-border">
                <p className="text-xs text-muted-foreground mb-2">Record voice emotion (optional — saves to database)</p>
                <VoiceWatchAnalyzer userId={participantCode || FALLBACK_USER_ID} onResult={onVoiceResult} />
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
                {
                  label: blockType === "stress"
                    ? "How stressed did you feel during this session?"
                    : "How strong were your food cravings during this session?",
                  value: surveyQ1,
                  set: setSurveyQ1,
                },
                {
                  label: "How calm do you feel right now?",
                  value: surveyQ2,
                  set: setSurveyQ2,
                },
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

              <Button className="w-full" disabled={!canSubmit || isSubmitting} onClick={handleSurveySubmit}>
                {isSubmitting ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Saving…</> : "Complete Session"}
              </Button>
            </CardContent>
          </Card>
        )}

      </div>
    </div>
  );
}
