import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation, useSearch } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { Loader2, Wind, CheckCircle2 } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

// ── Types ─────────────────────────────────────────────────────────────────────

interface EEGReading {
  alpha: number;
  beta: number;
  theta: number;
  delta: number;
  gamma: number;
  stress_level: number;
}

type Step = 1 | 2 | 3 | 4 | 5;

// ── Helpers ───────────────────────────────────────────────────────────────────

const mockEEG = (): EEGReading => ({
  alpha: 0.3 + Math.random() * 0.2,
  beta: 0.2 + Math.random() * 0.3,
  theta: 0.1 + Math.random() * 0.15,
  delta: 0.05 + Math.random() * 0.1,
  gamma: 0.02 + Math.random() * 0.05,
  stress_level: 0.3 + Math.random() * 0.5,
});

async function fetchEEG(): Promise<EEGReading> {
  try {
    const res = await fetch("/api/simulate-eeg", { credentials: "include" });
    if (!res.ok) throw new Error("EEG fetch failed");
    return (await res.json()) as EEGReading;
  } catch {
    return mockEEG();
  }
}

function averageReadings(readings: EEGReading[]): EEGReading {
  if (readings.length === 0) return mockEEG();
  const sum = readings.reduce(
    (acc, r) => ({
      alpha: acc.alpha + r.alpha,
      beta: acc.beta + r.beta,
      theta: acc.theta + r.theta,
      delta: acc.delta + r.delta,
      gamma: acc.gamma + r.gamma,
      stress_level: acc.stress_level + r.stress_level,
    }),
    { alpha: 0, beta: 0, theta: 0, delta: 0, gamma: 0, stress_level: 0 }
  );
  const n = readings.length;
  return {
    alpha: sum.alpha / n,
    beta: sum.beta / n,
    theta: sum.theta / n,
    delta: sum.delta / n,
    gamma: sum.gamma / n,
    stress_level: sum.stress_level / n,
  };
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function stressColor(level: number): string {
  if (level < 0.4) return "bg-green-500";
  if (level <= 0.65) return "bg-yellow-400";
  return "bg-red-500";
}

function stressLabel(level: number): string {
  if (level < 0.4) return "Calm";
  if (level <= 0.65) return "Mild Stress";
  return "High Stress";
}

function stressTextColor(level: number): string {
  if (level < 0.4) return "text-green-400";
  if (level <= 0.65) return "text-yellow-400";
  return "text-red-400";
}

// ── Progress stepper ──────────────────────────────────────────────────────────

function StepStepper({ current }: { current: Step }) {
  const steps: { n: Step; label: string }[] = [
    { n: 1, label: "Baseline" },
    { n: 2, label: "Work" },
    { n: 3, label: "Breathing" },
    { n: 4, label: "Recovery" },
    { n: 5, label: "Survey" },
  ];

  return (
    <div className="w-full">
      <p className="text-xs text-muted-foreground mb-3 text-center">
        Step {current} of 5
      </p>
      <div className="flex items-center gap-0">
        {steps.map((s, idx) => (
          <div key={s.n} className="flex items-center flex-1">
            <div className="flex flex-col items-center w-full">
              <div
                className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold shrink-0 transition-colors ${
                  s.n < current
                    ? "bg-primary text-primary-foreground"
                    : s.n === current
                    ? "bg-primary text-primary-foreground ring-2 ring-primary/40"
                    : "bg-muted text-muted-foreground"
                }`}
              >
                {s.n < current ? <CheckCircle2 className="h-3.5 w-3.5" /> : s.n}
              </div>
              <span className="text-[10px] text-muted-foreground mt-1 text-center leading-tight">
                {s.label}
              </span>
            </div>
            {idx < steps.length - 1 && (
              <div
                className={`h-0.5 flex-1 mb-4 mx-0.5 transition-colors ${
                  s.n < current ? "bg-primary" : "bg-muted"
                }`}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Countdown timer hook ───────────────────────────────────────────────────────

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
          if (!doneRef.current) {
            doneRef.current = true;
            onDone();
          }
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

// ── Box Breathing animation ────────────────────────────────────────────────────

function BoxBreathing() {
  const phases = ["Inhale", "Hold", "Exhale", "Hold"];
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setPhase((p) => (p + 1) % 4);
    }, 4000);
    return () => clearInterval(id);
  }, []);

  const labels = ["Inhale 4 counts", "Hold 4 counts", "Exhale 4 counts", "Hold 4 counts"];

  return (
    <div className="flex flex-col items-center gap-6 py-4">
      <div
        className="relative flex items-center justify-center transition-all duration-[3800ms] ease-in-out"
        style={{
          width: phase === 0 || phase === 2 ? 160 : 120,
          height: phase === 0 || phase === 2 ? 160 : 120,
        }}
      >
        <div
          className="rounded-2xl bg-primary/20 border-2 border-primary/60 absolute inset-0 transition-all duration-[3800ms] ease-in-out"
        />
        <span className="text-sm font-semibold text-primary z-10">
          {phases[phase]}
        </span>
      </div>
      <p className="text-base font-medium text-foreground">{labels[phase]}</p>
      <div className="flex gap-2">
        {phases.map((_, i) => (
          <div
            key={i}
            className={`h-1.5 w-8 rounded-full transition-colors ${
              i === phase ? "bg-primary" : "bg-muted"
            }`}
          />
        ))}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function StudySessionStress() {
  const [, navigate] = useLocation();
  const search = useSearch();
  const { toast } = useToast();

  const params = new URLSearchParams(search);
  const participantCode = params.get("code") ?? "";

  // Session state
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [step, setStep] = useState<Step>(1);
  const [isStarting, setIsStarting] = useState(true);

  // EEG accumulation
  const [preEegReadings, setPreEegReadings] = useState<EEGReading[]>([]);
  const [postEegReadings, setPostEegReadings] = useState<EEGReading[]>([]);
  const [workEegReadings, setWorkEegReadings] = useState<EEGReading[]>([]);
  const [liveStress, setLiveStress] = useState(0);
  const [interventionTriggered, setInterventionTriggered] = useState(false);

  // Averaged EEG for submission
  const [preEegJson, setPreEegJson] = useState<EEGReading | null>(null);
  const [postEegJson, setPostEegJson] = useState<EEGReading | null>(null);

  // Survey sliders
  const [surveyStressed, setSurveyStressed] = useState<number | null>(null);
  const [surveyBreathing, setSurveyBreathing] = useState<number | null>(null);
  const [surveyNow, setSurveyNow] = useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Timer active flags per step
  const [step1Active, setStep1Active] = useState(false);
  const [step2Active, setStep2Active] = useState(false);
  const [step4Active, setStep4Active] = useState(false);

  // ── Session start ────────────────────────────────────────────────────────────

  useEffect(() => {
    let cancelled = false;

    async function startSession() {
      try {
        const res = await apiRequest("POST", "/api/study/session/start", {
          participant_code: participantCode,
          block_type: "stress",
        });
        const data = (await res.json()) as { session_id: number };
        if (!cancelled) {
          setSessionId(data.session_id);
          setIsStarting(false);
          setStep1Active(true);
        }
      } catch {
        if (!cancelled) {
          // Proceed with a local mock session id so the session can still run
          setSessionId(-1);
          setIsStarting(false);
          setStep1Active(true);
        }
      }
    }

    startSession();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── EEG polling ──────────────────────────────────────────────────────────────

  const pollEEG = useCallback(
    async (
      accumulator: React.Dispatch<React.SetStateAction<EEGReading[]>>,
      onReading?: (r: EEGReading) => void
    ) => {
      const reading = await fetchEEG();
      accumulator((prev) => [...prev, reading]);
      if (onReading) onReading(reading);
    },
    []
  );

  // Step 1 polling
  useEffect(() => {
    if (!step1Active) return;
    const id = setInterval(() => {
      pollEEG(setPreEegReadings);
    }, 4000);
    return () => clearInterval(id);
  }, [step1Active, pollEEG]);

  // Step 2 polling
  useEffect(() => {
    if (!step2Active) return;
    const id = setInterval(() => {
      pollEEG(setWorkEegReadings, (r) => {
        setLiveStress(r.stress_level);
        if (r.stress_level > 0.65) {
          setInterventionTriggered(true);
          setStep2Active(false);
          setStep(3);
        }
      });
    }, 4000);
    return () => clearInterval(id);
  }, [step2Active, pollEEG]);

  // Step 4 polling
  useEffect(() => {
    if (!step4Active) return;
    const id = setInterval(() => {
      pollEEG(setPostEegReadings);
    }, 4000);
    return () => clearInterval(id);
  }, [step4Active, pollEEG]);

  // ── Timer callbacks ──────────────────────────────────────────────────────────

  const onStep1Done = useCallback(() => {
    setStep1Active(false);
    setPreEegJson(averageReadings(preEegReadings));
    setStep(2);
    setStep2Active(true);
  }, [preEegReadings]);

  const onStep2Done = useCallback(() => {
    setStep2Active(false);
    setStep(3);
  }, []);

  const onStep4Done = useCallback(() => {
    setStep4Active(false);
    setPostEegJson(averageReadings(postEegReadings));
    setStep(5);
  }, [postEegReadings]);

  // ── Timers ───────────────────────────────────────────────────────────────────

  const remaining1 = useCountdown(5 * 60, step1Active, onStep1Done);
  const remaining2 = useCountdown(15 * 60, step2Active, onStep2Done);
  const remaining4 = useCountdown(5 * 60, step4Active, onStep4Done);

  // ── Survey submit ────────────────────────────────────────────────────────────

  const canComplete =
    surveyStressed !== null && surveyBreathing !== null && surveyNow !== null;

  async function handleComplete() {
    if (!canComplete || isSubmitting) return;
    setIsSubmitting(true);

    const finalPreEeg = preEegJson ?? averageReadings(preEegReadings);
    const finalPostEeg = postEegJson ?? averageReadings(postEegReadings);

    const surveyData = {
      stressed_during: surveyStressed ?? 5,
      breathing_helped: surveyBreathing ?? 5,
      feeling_now: surveyNow ?? 5,
    };

    try {
      await apiRequest("POST", "/api/study/session/complete", {
        session_id: sessionId,
        pre_eeg_json: finalPreEeg,
        post_eeg_json: finalPostEeg,
        eeg_features_json: finalPreEeg,
        survey_json: surveyData,
        intervention_triggered: interventionTriggered,
      });
      navigate(`/study/complete?code=${encodeURIComponent(participantCode)}&done=stress`);
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "Submission failed — please try again";
      toast({ title: "Session submit failed", description: msg, variant: "destructive" });
    } finally {
      setIsSubmitting(false);
    }
  }

  // ── Loading screen ────────────────────────────────────────────────────────────

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

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-lg mx-auto px-4 py-8 space-y-6">

        {/* Top stepper */}
        <StepStepper current={step} />

        {/* ── Step 1: Baseline ── */}
        {step === 1 && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Resting Baseline</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground leading-relaxed">
                Put on the Muse 2 headband and sit comfortably with your eyes closed.
              </p>

              {/* Countdown */}
              <div className="flex flex-col items-center gap-3">
                <span className="text-5xl font-mono font-bold tracking-tight">
                  {formatTime(remaining1)}
                </span>
                <Progress value={((5 * 60 - remaining1) / (5 * 60)) * 100} className="h-2" />
              </div>

              {/* Recording indicator */}
              <div className="flex items-center justify-center gap-2">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
                </span>
                <span className="text-sm text-muted-foreground">
                  Recording…{" "}
                  <span className="font-medium text-foreground">
                    {preEegReadings.length} samples
                  </span>
                </span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* ── Step 2: Work session ── */}
        {step === 2 && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Work Session</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground leading-relaxed">
                Continue your normal work with the headband on. We will monitor your stress level.
              </p>

              {/* Countdown */}
              <div className="flex flex-col items-center gap-3">
                <span className="text-5xl font-mono font-bold tracking-tight">
                  {formatTime(remaining2)}
                </span>
                <Progress value={((15 * 60 - remaining2) / (15 * 60)) * 100} className="h-2" />
              </div>

              {/* Live stress bar */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground uppercase tracking-wide">
                    Stress level
                  </span>
                  <span className={`text-sm font-semibold ${stressTextColor(liveStress)}`}>
                    {stressLabel(liveStress)}
                  </span>
                </div>
                <div className="h-3 w-full rounded-full bg-muted overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${stressColor(liveStress)}`}
                    style={{ width: `${Math.round(liveStress * 100)}%` }}
                  />
                </div>
                <p className="text-xs text-muted-foreground text-right">
                  {Math.round(liveStress * 100)}%
                </p>
              </div>

              {/* Recording indicator */}
              <div className="flex items-center justify-center gap-2">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
                </span>
                <span className="text-sm text-muted-foreground">
                  Recording…{" "}
                  <span className="font-medium text-foreground">
                    {workEegReadings.length} samples
                  </span>
                </span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* ── Step 3: Breathing intervention ── */}
        {step === 3 && (
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <Wind className="h-5 w-5 text-primary" />
                <CardTitle className="text-lg">Breathing Exercise</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              {interventionTriggered && (
                <div className="rounded-md bg-orange-500/10 border border-orange-500/30 px-3 py-2">
                  <p className="text-xs text-orange-400 font-medium">
                    High stress detected — let us help you reset.
                  </p>
                </div>
              )}

              <p className="text-sm text-muted-foreground text-center leading-relaxed">
                Inhale 4 counts &rarr; Hold 4 counts &rarr; Exhale 4 counts &rarr; Hold 4 counts
              </p>

              <BoxBreathing />

              <Button
                className="w-full"
                size="lg"
                onClick={() => {
                  setStep(4);
                  setStep4Active(true);
                }}
              >
                I feel calmer — Continue
              </Button>

              <p className="text-xs text-muted-foreground text-center">
                Take your time. Do at least 3 rounds (about 1 minute).
              </p>
            </CardContent>
          </Card>
        )}

        {/* ── Step 4: Post-intervention EEG ── */}
        {step === 4 && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Recovery Baseline</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground leading-relaxed">
                Continue sitting quietly with the headband on.
              </p>

              {/* Countdown */}
              <div className="flex flex-col items-center gap-3">
                <span className="text-5xl font-mono font-bold tracking-tight">
                  {formatTime(remaining4)}
                </span>
                <Progress value={((5 * 60 - remaining4) / (5 * 60)) * 100} className="h-2" />
              </div>

              {/* Recording indicator */}
              <div className="flex items-center justify-center gap-2">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
                </span>
                <span className="text-sm text-muted-foreground">
                  Recording…{" "}
                  <span className="font-medium text-foreground">
                    {postEegReadings.length} samples
                  </span>
                </span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* ── Step 5: Survey ── */}
        {step === 5 && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Quick Survey</CardTitle>
            </CardHeader>
            <CardContent className="space-y-8">
              <p className="text-sm text-muted-foreground">
                Three quick questions — answer honestly.
              </p>

              {/* Q1 */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium leading-snug">
                    How stressed did you feel during the work session?
                  </p>
                  <span className="text-lg font-bold text-primary ml-4 shrink-0">
                    {surveyStressed ?? "—"}
                  </span>
                </div>
                <Slider
                  min={1}
                  max={10}
                  step={1}
                  value={surveyStressed !== null ? [surveyStressed] : [5]}
                  onValueChange={([v]) => setSurveyStressed(v)}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Not at all</span>
                  <span>Extremely</span>
                </div>
              </div>

              {/* Q2 */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium leading-snug">
                    Did the breathing exercise help you feel calmer?
                  </p>
                  <span className="text-lg font-bold text-primary ml-4 shrink-0">
                    {surveyBreathing ?? "—"}
                  </span>
                </div>
                <Slider
                  min={1}
                  max={10}
                  step={1}
                  value={surveyBreathing !== null ? [surveyBreathing] : [5]}
                  onValueChange={([v]) => setSurveyBreathing(v)}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Not at all</span>
                  <span>Very much</span>
                </div>
              </div>

              {/* Q3 */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium leading-snug">
                    How do you feel right now?
                  </p>
                  <span className="text-lg font-bold text-primary ml-4 shrink-0">
                    {surveyNow ?? "—"}
                  </span>
                </div>
                <Slider
                  min={1}
                  max={10}
                  step={1}
                  value={surveyNow !== null ? [surveyNow] : [5]}
                  onValueChange={([v]) => setSurveyNow(v)}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Very stressed</span>
                  <span>Very calm</span>
                </div>
              </div>

              {!canComplete && (
                <p className="text-xs text-muted-foreground text-center">
                  Move all three sliders to enable completion.
                </p>
              )}

              <Button
                className="w-full"
                size="lg"
                disabled={!canComplete || isSubmitting}
                onClick={handleComplete}
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Submitting…
                  </>
                ) : (
                  "Complete Session"
                )}
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
