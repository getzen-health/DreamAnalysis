import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation, useSearch } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Loader2, ChevronRight, Utensils, EyeOff, Eye, CheckCircle2, SkipForward, AlertTriangle } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { getMLApiUrl } from "@/lib/ml-api";
import { useToast } from "@/hooks/use-toast";

// ── Types ─────────────────────────────────────────────────────────────────────

interface EEGSample {
  alpha: number;
  beta: number;
  theta: number;
  delta: number;
  gamma: number;
  stress_level: number;
}

type Phase = "pre_survey" | "baseline" | "eating" | "post_eeg" | "post_survey";

interface PhaseLog {
  pre_survey_start?: string;
  baseline_start?: string;
  eating_start?: string;
  post_eeg_start?: string;
  post_survey_start?: string;
  completed?: string;
  skipped_to_survey?: boolean;
  skip_from_phase?: string;
}

interface SessionBackup {
  phase: Phase;
  sessionId: number | null;
  baselineReadings: EEGSample[];
  postReadings: EEGSample[];
  preSurvey: { hunger: number; mood: number };
  interventionTriggered: boolean;
  phaseLog: PhaseLog;
  timestamp: number;
}

const BACKUP_MAX_AGE_MS = 2 * 60 * 60 * 1000; // 2 hours

function getBackupKey(participantCode: string): string {
  return `study_session_backup_${participantCode}_food`;
}

// ── Durations (seconds) ──────────────────────────────────────────────────────

const FULL_DURATIONS = { baseline: 5 * 60, post_eeg: 10 * 60 };
const DEV_DURATIONS = { baseline: 15, post_eeg: 20 };

const EATING_OPTIONS = [
  { label: "15 min", seconds: 15 * 60 },
  { label: "20 min", seconds: 20 * 60 },
  { label: "25 min", seconds: 25 * 60 },
  { label: "30 min", seconds: 30 * 60 },
];
const DEV_EATING_SEC = 15;

const CHECKPOINT_INTERVAL_MS = 30_000;

// ── EEG helpers ───────────────────────────────────────────────────────────────

async function fetchEEG(): Promise<{ reading: EEGSample; ok: boolean }> {
  try {
    const mlUrl = getMLApiUrl();
    const res = await fetch(`${mlUrl}/api/simulate-eeg`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state: "rest", duration: 4, fs: 256 }),
    });
    if (!res.ok) throw new Error("fetch failed");
    const data = await res.json();
    const emotions = data.analysis?.emotions ?? {};
    const bp = emotions.band_powers ?? {};
    return {
      reading: {
        alpha: bp.alpha ?? 0.3,
        beta: bp.beta ?? 0.25,
        theta: bp.theta ?? 0.15,
        delta: bp.delta ?? 0.1,
        gamma: bp.gamma ?? 0.03,
        stress_level: emotions.stress_index ?? 0.3,
      },
      ok: true,
    };
  } catch {
    return {
      reading: {
        alpha: 0.3 + Math.random() * 0.2,
        beta: 0.2 + Math.random() * 0.3,
        theta: 0.1 + Math.random() * 0.15,
        delta: 0.05 + Math.random() * 0.1,
        gamma: 0.02 + Math.random() * 0.05,
        stress_level: 0.3 + Math.random() * 0.4,
      },
      ok: false,
    };
  }
}

function averageEEG(samples: EEGSample[]): EEGSample | null {
  if (samples.length === 0) return null;
  const sum = samples.reduce(
    (acc, s) => ({
      alpha: acc.alpha + s.alpha, beta: acc.beta + s.beta, theta: acc.theta + s.theta,
      delta: acc.delta + s.delta, gamma: acc.gamma + s.gamma, stress_level: acc.stress_level + s.stress_level,
    }),
    { alpha: 0, beta: 0, theta: 0, delta: 0, gamma: 0, stress_level: 0 }
  );
  const n = samples.length;
  return {
    alpha: sum.alpha / n, beta: sum.beta / n, theta: sum.theta / n,
    delta: sum.delta / n, gamma: sum.gamma / n, stress_level: sum.stress_level / n,
  };
}

/** Compute average band powers (alpha, beta, theta, delta, gamma) across readings */
function avgBands(readings: EEGSample[]): { alpha: number; beta: number; theta: number; delta: number; gamma: number } {
  if (readings.length === 0) {
    return { alpha: 0, beta: 0, theta: 0, delta: 0, gamma: 0 };
  }
  const sum = readings.reduce(
    (acc, r) => ({
      alpha: acc.alpha + r.alpha,
      beta: acc.beta + r.beta,
      theta: acc.theta + r.theta,
      delta: acc.delta + r.delta,
      gamma: acc.gamma + r.gamma,
    }),
    { alpha: 0, beta: 0, theta: 0, delta: 0, gamma: 0 }
  );
  const n = readings.length;
  return {
    alpha: sum.alpha / n,
    beta: sum.beta / n,
    theta: sum.theta / n,
    delta: sum.delta / n,
    gamma: sum.gamma / n,
  };
}

function computeQualityScore(readings: EEGSample[]): number {
  if (readings.length < 3) return 0;
  const alphas = readings.map((r) => r.alpha);
  const mean = alphas.reduce((a, b) => a + b, 0) / alphas.length;
  const variance = alphas.reduce((a, v) => a + (v - mean) ** 2, 0) / alphas.length;
  const hasVariance = variance > 0.0001 ? 30 : 0;
  const allInRange = readings.every((r) => r.alpha >= 0 && r.alpha <= 1 && r.beta >= 0 && r.beta <= 1);
  const rangeScore = allInRange ? 30 : 10;
  const sampleScore = Math.min(40, (readings.length / 30) * 40);
  return Math.round(hasVariance + rangeScore + sampleScore);
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

// ── Countdown timer hook ──────────────────────────────────────────────────────

function useCountdown(totalSeconds: number, active: boolean, onDone: () => void) {
  const [remaining, setRemaining] = useState(totalSeconds);
  const doneRef = useRef(false);

  useEffect(() => {
    setRemaining(totalSeconds);
    doneRef.current = false;
  }, [totalSeconds]);

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

const PHASE_ORDER: Phase[] = ["pre_survey", "baseline", "eating", "post_eeg", "post_survey"];
const PHASE_LABELS: Record<Phase, string> = {
  pre_survey: "Pre-survey",
  baseline: "Baseline",
  eating: "Eat",
  post_eeg: "Post EEG",
  post_survey: "Post-survey",
};

function SessionStepper({ phase, devMode }: { phase: Phase; devMode: boolean }) {
  const current = PHASE_ORDER.indexOf(phase);
  return (
    <div className="w-full">
      <div className="flex items-center justify-center gap-2 mb-3">
        <p className="text-xs text-muted-foreground text-center">Food-Emotion Session</p>
        {devMode && (
          <Badge variant="outline" className="border-yellow-500/50 text-yellow-400 text-[10px]">DEV</Badge>
        )}
      </div>
      <div className="flex items-center gap-0">
        {PHASE_ORDER.map((p, idx) => {
          const done = idx < current;
          const active = idx === current;
          return (
            <div key={p} className="flex items-center flex-1">
              <div className="flex flex-col items-center w-full">
                <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold shrink-0 transition-colors ${
                  done ? "bg-primary text-primary-foreground" : active ? "bg-primary text-primary-foreground ring-2 ring-primary/40" : "bg-muted text-muted-foreground"
                }`}>
                  {done ? <CheckCircle2 className="h-3.5 w-3.5" /> : idx + 1}
                </div>
                <span className="text-[10px] text-muted-foreground mt-1 text-center leading-tight">{PHASE_LABELS[p]}</span>
              </div>
              {idx < PHASE_ORDER.length - 1 && (
                <div className={`h-0.5 flex-1 mb-4 mx-0.5 transition-colors ${done ? "bg-primary" : "bg-muted"}`} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Slider row ────────────────────────────────────────────────────────────────

function SliderRow({ label, value, onChange, left, right }: {
  label: string; value: number; onChange: (v: number) => void; left?: string; right?: string;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-sm">{label}</Label>
        <Badge variant="outline" className="font-mono tabular-nums w-10 text-center">{value}</Badge>
      </div>
      <Slider min={1} max={10} step={1} value={[value]} onValueChange={([v]) => onChange(v)} className="w-full" />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{left ?? "1"}</span><span>{right ?? "10"}</span>
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function StudySessionFood() {
  const [, navigate] = useLocation();
  const search = useSearch();
  const { toast } = useToast();

  const params = new URLSearchParams(search);
  const participantCode = params.get("code") ?? "";
  const devMode = params.get("dev") === "1" || params.get("test") === "1";

  const durations = devMode ? DEV_DURATIONS : FULL_DURATIONS;

  const [sessionId, setSessionId] = useState<number | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [phase, setPhase] = useState<Phase>("pre_survey");

  // Phase timestamp log
  const phaseLogRef = useRef<PhaseLog>({ pre_survey_start: new Date().toISOString() });

  // Separate EEG buffers
  const [baselineReadings, setBaselineReadings] = useState<EEGSample[]>([]);
  const [postReadings, setPostReadings] = useState<EEGSample[]>([]);

  // Timer flags
  const [baselineActive, setBaselineActive] = useState(false);
  const [eatingActive, setEatingActive] = useState(false);
  const [postEegActive, setPostEegActive] = useState(false);

  // ML connection status
  const [lastSuccessTime, setLastSuccessTime] = useState<number>(0);
  const [connectionOk, setConnectionOk] = useState(false);

  // Eating duration (user-selectable)
  const [eatingDuration, setEatingDuration] = useState(devMode ? DEV_EATING_SEC : 20 * 60);

  // Pre-survey
  const [preHunger, setPreHunger] = useState(5);
  const [preMood, setPreMood] = useState(5);

  // Post-survey
  const [whatAte, setWhatAte] = useState("");
  const [foodHealthy, setFoodHealthy] = useState(5);
  const [postEnergy, setPostEnergy] = useState(5);
  const [postMood, setPostMood] = useState(5);
  const [postSatisfied, setPostSatisfied] = useState(5);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Recovery
  const [pendingBackup, setPendingBackup] = useState<SessionBackup | null>(null);
  const [showRecovery, setShowRecovery] = useState(false);

  // ── Check for existing backup on mount ─────────────────────────────────────

  useEffect(() => {
    if (!participantCode) return;
    try {
      const raw = localStorage.getItem(getBackupKey(participantCode));
      if (!raw) return;
      const backup: SessionBackup = JSON.parse(raw);
      const age = Date.now() - backup.timestamp;
      if (age < BACKUP_MAX_AGE_MS) {
        setPendingBackup(backup);
        setShowRecovery(true);
      } else {
        localStorage.removeItem(getBackupKey(participantCode));
      }
    } catch {
      localStorage.removeItem(getBackupKey(participantCode));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleResume() {
    if (!pendingBackup) return;
    setPhase(pendingBackup.phase);
    setSessionId(pendingBackup.sessionId);
    setBaselineReadings(pendingBackup.baselineReadings);
    setPostReadings(pendingBackup.postReadings);
    setPreHunger(pendingBackup.preSurvey.hunger);
    setPreMood(pendingBackup.preSurvey.mood);
    phaseLogRef.current = pendingBackup.phaseLog;
    setIsStarting(false);

    // Re-activate the correct timer for the restored phase
    switch (pendingBackup.phase) {
      case "baseline": setBaselineActive(true); break;
      case "eating":   setEatingActive(true); break;
      case "post_eeg": setPostEegActive(true); break;
      case "pre_survey":
      case "post_survey":
        break; // no timer needed
    }

    setShowRecovery(false);
    setPendingBackup(null);
  }

  function handleStartFresh() {
    if (participantCode) {
      localStorage.removeItem(getBackupKey(participantCode));
    }
    setShowRecovery(false);
    setPendingBackup(null);
  }

  // ── Save backup to localStorage every 30s ──────────────────────────────────

  useEffect(() => {
    if (!participantCode || showRecovery) return;
    if (isStarting) return;

    const id = setInterval(() => {
      const backup: SessionBackup = {
        phase,
        sessionId,
        baselineReadings,
        postReadings,
        preSurvey: { hunger: preHunger, mood: preMood },
        interventionTriggered: false,
        phaseLog: { ...phaseLogRef.current },
        timestamp: Date.now(),
      };
      try {
        localStorage.setItem(getBackupKey(participantCode), JSON.stringify(backup));
      } catch {
        // localStorage full or unavailable — ignore
      }
    }, CHECKPOINT_INTERVAL_MS);

    return () => clearInterval(id);
  }, [participantCode, showRecovery, isStarting, phase, sessionId, baselineReadings, postReadings, preHunger, preMood]);

  // ── Session start ─────────────────────────────────────────────────────────

  useEffect(() => {
    if (showRecovery) return; // wait for user to choose resume or start fresh
    if (!participantCode) return;
    setIsStarting(true);
    apiRequest("POST", "/api/study/session/start", { participant_code: participantCode, block_type: "food" })
      .then((res) => res.json())
      .then((data: { session_id: number }) => setSessionId(data.session_id))
      .catch(() => setSessionId(-1))
      .finally(() => setIsStarting(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showRecovery, participantCode]);

  // ── Auto-checkpoint every 30s ──────────────────────────────────────────────

  useEffect(() => {
    if (sessionId == null || sessionId < 0) return;
    if (phase === "pre_survey" || phase === "post_survey") return;

    const id = setInterval(() => {
      const preEeg = averageEEG(baselineReadings);
      const postEeg = averageEEG(postReadings);
      const features = averageEEG([...baselineReadings, ...postReadings]);
      apiRequest("PATCH", `/api/study/session/${sessionId}/checkpoint`, {
        pre_eeg_json: preEeg,
        post_eeg_json: postEeg,
        eeg_features_json: features,
        partial: true,
        phase_log: phaseLogRef.current,
      }).catch(() => {});
    }, CHECKPOINT_INTERVAL_MS);

    return () => clearInterval(id);
  }, [sessionId, phase, baselineReadings, postReadings]);

  // ── EEG polling ────────────────────────────────────────────────────────────

  const pollEEG = useCallback(async () => {
    const { reading, ok } = await fetchEEG();
    if (ok) {
      setLastSuccessTime(Date.now());
    }
    setConnectionOk(ok || Date.now() - lastSuccessTime < 10_000);
    if (baselineActive) {
      setBaselineReadings((prev) => [...prev, reading]);
    } else if (postEegActive) {
      setPostReadings((prev) => [...prev, reading]);
    }
  }, [baselineActive, postEegActive, lastSuccessTime]);

  useEffect(() => {
    if (!baselineActive && !postEegActive) return;
    const id = setInterval(pollEEG, 4000);
    pollEEG();
    return () => clearInterval(id);
  }, [baselineActive, postEegActive, pollEEG]);

  // ── Timer callbacks ───────────────────────────────────────────────────────

  const onBaselineDone = useCallback(() => {
    setBaselineActive(false);
    setPhase("eating");
    setEatingActive(true);
    phaseLogRef.current.eating_start = new Date().toISOString();
  }, []);

  const onEatingDone = useCallback(() => {
    setEatingActive(false);
    setPhase("post_eeg");
    setPostEegActive(true);
    phaseLogRef.current.post_eeg_start = new Date().toISOString();
  }, []);

  const onPostEegDone = useCallback(() => {
    setPostEegActive(false);
    setPhase("post_survey");
    phaseLogRef.current.post_survey_start = new Date().toISOString();
  }, []);

  // ── Skip to survey ─────────────────────────────────────────────────────────

  function handleSkipToSurvey() {
    setBaselineActive(false);
    setEatingActive(false);
    setPostEegActive(false);
    phaseLogRef.current.skipped_to_survey = true;
    phaseLogRef.current.skip_from_phase = phase;
    phaseLogRef.current.post_survey_start = new Date().toISOString();
    setPhase("post_survey");
  }

  // ── Timers ────────────────────────────────────────────────────────────────

  const remainingBaseline = useCountdown(durations.baseline, baselineActive, onBaselineDone);
  const remainingEating = useCountdown(eatingDuration, eatingActive, onEatingDone);
  const remainingPostEeg = useCountdown(durations.post_eeg, postEegActive, onPostEegDone);

  // ── Submit ────────────────────────────────────────────────────────────────

  async function handleComplete() {
    if (isSubmitting) return;
    setIsSubmitting(true);

    const preEeg = averageEEG(baselineReadings);
    const postEeg = averageEEG(postReadings);
    const preAvg = avgBands(baselineReadings);
    const postAvg = avgBands(postReadings);
    const allReadings = [...baselineReadings, ...postReadings];
    const features = averageEEG(allReadings);
    const quality = computeQualityScore(allReadings);

    phaseLogRef.current.completed = new Date().toISOString();

    const survey_json = {
      pre_hunger: preHunger,
      pre_mood: preMood,
      what_ate: whatAte.trim(),
      food_healthy: foodHealthy,
      post_energy: postEnergy,
      post_mood: postMood,
      post_satisfied: postSatisfied,
    };

    try {
      await apiRequest("POST", "/api/study/session/complete", {
        session_id: sessionId,
        pre_eeg_json: { ...preEeg, avg_bands: preAvg },
        post_eeg_json: { ...postEeg, avg_bands: postAvg },
        eeg_features_json: { ...features, quality_score: quality, sample_count: allReadings.length },
        survey_json,
        intervention_triggered: false,
        phase_log: phaseLogRef.current,
        data_quality_score: quality,
      });
      // Clear backup on successful completion
      localStorage.removeItem(getBackupKey(participantCode));
      navigate(`/study/complete?code=${encodeURIComponent(participantCode)}&done=food`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Submission failed";
      toast({ title: "Could not complete session", description: msg, variant: "destructive" });
    } finally {
      setIsSubmitting(false);
    }
  }

  // ── Loading ───────────────────────────────────────────────────────────────

  if (isStarting) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center space-y-3">
          <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto" />
          <p className="text-sm text-muted-foreground">Starting session...</p>
        </div>
      </div>
    );
  }

  // ── Recovery prompt ──────────────────────────────────────────────────────────

  if (showRecovery && pendingBackup) {
    const backupAge = Math.round((Date.now() - pendingBackup.timestamp) / 60_000);
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Card className="max-w-md mx-4">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-yellow-400" />
              Resume Previous Session?
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground leading-relaxed">
              A previous food session was found from{" "}
              <span className="font-medium text-foreground">
                {backupAge < 1 ? "less than a minute" : `${backupAge} minute${backupAge === 1 ? "" : "s"}`}
              </span>{" "}
              ago (phase: <Badge variant="outline" className="ml-1">{PHASE_LABELS[pendingBackup.phase]}</Badge>).
            </p>
            <p className="text-sm text-muted-foreground">
              You can resume where you left off or start a fresh session.
            </p>
            <div className="flex gap-3">
              <Button className="flex-1" onClick={handleResume}>
                Resume
              </Button>
              <Button variant="outline" className="flex-1" onClick={handleStartFresh}>
                Start Fresh
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // ── Recording dot (shared) ────────────────────────────────────────────────

  const sampleCount = baselineReadings.length + postReadings.length;
  const recordingDot = (
    <div className="flex items-center justify-center gap-2">
      <span className="relative flex h-2.5 w-2.5">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
        <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
      </span>
      <span className="text-sm text-muted-foreground">
        Recording... <span className="font-medium text-foreground">{sampleCount} samples</span>
      </span>
    </div>
  );

  const skipButton = phase !== "pre_survey" && phase !== "post_survey" && (
    <Button
      variant="ghost"
      size="sm"
      className="text-xs text-muted-foreground hover:text-yellow-400"
      onClick={handleSkipToSurvey}
    >
      <SkipForward className="h-3.5 w-3.5 mr-1" />
      Skip to Survey
    </Button>
  );

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background text-foreground relative">
      <div className="absolute top-3 right-3 flex items-center gap-1.5 z-10">
        <div
          className={`w-2 h-2 rounded-full ${connectionOk ? "bg-green-400" : "bg-red-400"}`}
        />
        <span
          className={`text-[10px] ${connectionOk ? "text-green-400" : "text-red-400"}`}
        >
          {connectionOk ? "Connected" : "EEG connection lost"}
        </span>
      </div>
      <div className="max-w-lg mx-auto px-4 py-10 space-y-6">

        {devMode && (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
            <AlertTriangle className="h-4 w-4 text-yellow-400 shrink-0" />
            <p className="text-xs text-yellow-400">
              Dev mode — timers shortened. Add <code>?dev=1</code> to URL.
            </p>
          </div>
        )}

        {/* Header */}
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Utensils className="h-5 w-5 text-primary" />
            <h1 className="text-xl font-bold">Food-Emotion Session</h1>
            {participantCode && (
              <Badge variant="outline" className="border-primary/40 text-primary font-mono ml-auto">
                {participantCode}
              </Badge>
            )}
          </div>
          <SessionStepper phase={phase} devMode={devMode} />
        </div>

        {/* ── Step 1: Pre-meal survey ── */}
        {phase === "pre_survey" && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Pre-Meal Survey</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6 pt-2">
              <SliderRow label="How hungry are you right now?" value={preHunger} onChange={setPreHunger} left="Not hungry" right="Starving" />
              <SliderRow label="What is your current mood?" value={preMood} onChange={setPreMood} left="Very low" right="Excellent" />

              {!devMode && (
                <div className="space-y-2">
                  <Label className="text-sm">How long will you need to eat?</Label>
                  <div className="grid grid-cols-4 gap-2">
                    {EATING_OPTIONS.map((opt) => (
                      <Button
                        key={opt.seconds}
                        variant={eatingDuration === opt.seconds ? "default" : "outline"}
                        size="sm"
                        onClick={() => setEatingDuration(opt.seconds)}
                      >
                        {opt.label}
                      </Button>
                    ))}
                  </div>
                </div>
              )}

              <Button className="w-full" size="lg" onClick={() => {
                setPhase("baseline");
                setBaselineActive(true);
                phaseLogRef.current.baseline_start = new Date().toISOString();
              }}>
                Start Baseline Recording
                <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            </CardContent>
          </Card>
        )}

        {/* ── Step 2: Pre-meal baseline ── */}
        {phase === "baseline" && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <EyeOff className="h-5 w-5 text-primary" />
                Pre-Meal Baseline
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground leading-relaxed">
                Sit still with your eyes closed. Breathe normally. We're capturing your resting brain state before eating.
              </p>

              <div className="flex flex-col items-center gap-3">
                <span className="text-5xl font-mono font-bold tracking-tight">
                  {formatTime(remainingBaseline)}
                </span>
                <Progress value={((durations.baseline - remainingBaseline) / durations.baseline) * 100} className="h-2" />
                <p className="text-xs text-muted-foreground">
                  {devMode ? "15 sec (dev)" : "5 minutes"} — eyes closed
                </p>
              </div>

              {recordingDot}
              <div className="flex justify-center">{skipButton}</div>
            </CardContent>
          </Card>
        )}

        {/* ── Step 3: Eat your meal ── */}
        {phase === "eating" && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <Utensils className="h-5 w-5 text-primary" />
                Eat Your Meal
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground leading-relaxed">
                Remove the headband and eat your normal meal. Come back when the timer ends or when you're done eating.
              </p>

              <div className="flex flex-col items-center gap-3">
                <span className="text-5xl font-mono font-bold tracking-tight">
                  {formatTime(remainingEating)}
                </span>
                <Progress value={((eatingDuration - remainingEating) / eatingDuration) * 100} className="h-2" />
                <p className="text-xs text-muted-foreground">
                  {devMode ? "15 sec (dev)" : `${eatingDuration / 60} minutes`}
                </p>
              </div>

              <Button
                className="w-full" size="lg" variant="outline"
                onClick={() => {
                  setEatingActive(false);
                  setPhase("post_eeg");
                  setPostEegActive(true);
                  phaseLogRef.current.post_eeg_start = new Date().toISOString();
                }}
              >
                I'm done eating — Continue
              </Button>
              <div className="flex justify-center">{skipButton}</div>
            </CardContent>
          </Card>
        )}

        {/* ── Step 4: Post-meal EEG ── */}
        {phase === "post_eeg" && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <Eye className="h-5 w-5 text-primary" />
                Post-Meal EEG Recording
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground leading-relaxed">
                Put the headband back on and sit comfortably. We're recording how your brain responds after eating.
              </p>

              <div className="flex flex-col items-center gap-3">
                <span className="text-5xl font-mono font-bold tracking-tight">
                  {formatTime(remainingPostEeg)}
                </span>
                <Progress value={((durations.post_eeg - remainingPostEeg) / durations.post_eeg) * 100} className="h-2" />
                <p className="text-xs text-muted-foreground">
                  {devMode ? "20 sec (dev)" : "10 minutes"}
                </p>
              </div>

              {recordingDot}
              <div className="flex justify-center">{skipButton}</div>
            </CardContent>
          </Card>
        )}

        {/* ── Step 5: Post-meal survey ── */}
        {phase === "post_survey" && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Post-Meal Survey</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6 pt-2">
              <div className="space-y-1.5">
                <Label className="text-sm">What did you eat?</Label>
                <Textarea
                  placeholder="e.g. rice and lentils, a sandwich..."
                  value={whatAte}
                  onChange={(e) => setWhatAte(e.target.value)}
                  rows={2}
                  className="resize-none"
                />
              </div>
              <SliderRow label="How healthy was this meal?" value={foodHealthy} onChange={setFoodHealthy} left="Very unhealthy" right="Very healthy" />
              <SliderRow label="Energy level now?" value={postEnergy} onChange={setPostEnergy} left="Exhausted" right="Energized" />
              <SliderRow label="Mood now?" value={postMood} onChange={setPostMood} left="Very low" right="Excellent" />
              <SliderRow label="Do you feel satisfied?" value={postSatisfied} onChange={setPostSatisfied} left="Still hungry" right="Very satisfied" />

              <Button
                className="w-full" size="lg"
                disabled={isSubmitting || whatAte.trim().length === 0}
                onClick={handleComplete}
              >
                {isSubmitting ? (
                  <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Saving...</>
                ) : (
                  <>Complete Session<ChevronRight className="ml-2 h-4 w-4" /></>
                )}
              </Button>
            </CardContent>
          </Card>
        )}

      </div>
    </div>
  );
}
