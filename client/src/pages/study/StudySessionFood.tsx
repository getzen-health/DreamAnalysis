import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation, useSearch } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Loader2, ChevronRight, Utensils, Clock, Brain, CheckCircle2 } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

// ── Types ────────────────────────────────────────────────────────────────────

interface EEGSample {
  alpha: number;
  beta: number;
  theta: number;
  delta: number;
  gamma: number;
  stress_level: number;
}

interface AveragedEEG {
  alpha: number;
  beta: number;
  theta: number;
  delta: number;
  gamma: number;
  stress_level: number;
}

// ── Mock EEG fallback ────────────────────────────────────────────────────────

function mockEEG(): EEGSample {
  return {
    alpha: 0.3 + Math.random() * 0.2,
    beta: 0.2 + Math.random() * 0.3,
    theta: 0.1 + Math.random() * 0.15,
    delta: 0.05 + Math.random() * 0.1,
    gamma: 0.02 + Math.random() * 0.05,
    stress_level: 0.3 + Math.random() * 0.4,
  };
}

function averageEEG(samples: EEGSample[]): AveragedEEG {
  if (samples.length === 0) return mockEEG();
  const sum = samples.reduce(
    (acc, s) => ({
      alpha: acc.alpha + s.alpha,
      beta: acc.beta + s.beta,
      theta: acc.theta + s.theta,
      delta: acc.delta + s.delta,
      gamma: acc.gamma + s.gamma,
      stress_level: acc.stress_level + s.stress_level,
    }),
    { alpha: 0, beta: 0, theta: 0, delta: 0, gamma: 0, stress_level: 0 }
  );
  const n = samples.length;
  return {
    alpha: sum.alpha / n,
    beta: sum.beta / n,
    theta: sum.theta / n,
    delta: sum.delta / n,
    gamma: sum.gamma / n,
    stress_level: sum.stress_level / n,
  };
}

// ── Countdown timer hook ─────────────────────────────────────────────────────

function useCountdown(initialSeconds: number, active: boolean, onDone: () => void) {
  const [remaining, setRemaining] = useState(initialSeconds);
  const doneRef = useRef(false);

  useEffect(() => {
    if (!active) return;
    doneRef.current = false;
    setRemaining(initialSeconds);
  }, [active, initialSeconds]);

  useEffect(() => {
    if (!active) return;
    if (remaining <= 0) {
      if (!doneRef.current) {
        doneRef.current = true;
        onDone();
      }
      return;
    }
    const id = setTimeout(() => setRemaining((r) => r - 1), 1000);
    return () => clearTimeout(id);
  }, [active, remaining, onDone]);

  const mm = String(Math.floor(remaining / 60)).padStart(2, "0");
  const ss = String(remaining % 60).padStart(2, "0");
  return { remaining, label: `${mm}:${ss}` };
}

// ── EEG recording hook ────────────────────────────────────────────────────────

function useEEGRecorder(active: boolean) {
  const [samples, setSamples] = useState<EEGSample[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!active) {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }
    setSamples([]);
    intervalRef.current = setInterval(async () => {
      try {
        const res = await fetch("/api/simulate-eeg");
        if (res.ok) {
          const data: EEGSample = await res.json();
          setSamples((prev) => [...prev, data]);
        } else {
          setSamples((prev) => [...prev, mockEEG()]);
        }
      } catch {
        setSamples((prev) => [...prev, mockEEG()]);
      }
    }, 4000);
    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [active]);

  return samples;
}

// ── Stepper ───────────────────────────────────────────────────────────────────

const STEP_LABELS = [
  "Pre-meal survey",
  "Pre-meal EEG",
  "Eat",
  "Wait",
  "Post-meal EEG",
  "Post-meal survey",
];

function Stepper({ current }: { current: number }) {
  return (
    <div className="w-full space-y-2">
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>Step {current} of 6</span>
        <span>{STEP_LABELS[current - 1]}</span>
      </div>
      <Progress value={((current - 1) / 5) * 100} className="h-1.5" />
      <div className="flex gap-1">
        {STEP_LABELS.map((_, i) => (
          <div
            key={i}
            className={`h-1 flex-1 rounded-full ${
              i + 1 < current
                ? "bg-primary"
                : i + 1 === current
                ? "bg-primary/60"
                : "bg-muted"
            }`}
          />
        ))}
      </div>
    </div>
  );
}

// ── Slider row ────────────────────────────────────────────────────────────────

function SliderRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-sm">{label}</Label>
        <Badge variant="outline" className="font-mono tabular-nums w-10 text-center">
          {value}
        </Badge>
      </div>
      <Slider
        min={1}
        max={10}
        step={1}
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        className="w-full"
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>1</span>
        <span>10</span>
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

  // Step tracking
  const [step, setStep] = useState(1);

  // Session state
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [isStarting, setIsStarting] = useState(false);

  // Step 1 — pre-meal survey
  const [preHunger, setPreHunger] = useState(5);
  const [preMood, setPreMood] = useState(5);

  // Step 2 — pre-meal EEG
  const [preEegActive, setPreEegActive] = useState(false);
  const [preEegDone, setPreEegDone] = useState(false);
  const [preEegJson, setPreEegJson] = useState<AveragedEEG | null>(null);
  const preEegSamples = useEEGRecorder(preEegActive);

  // Step 4 — wait timer
  const [waitActive, setWaitActive] = useState(false);

  // Step 5 — post-meal EEG
  const [postEegActive, setPostEegActive] = useState(false);
  const [postEegDone, setPostEegDone] = useState(false);
  const [postEegJson, setPostEegJson] = useState<AveragedEEG | null>(null);
  const postEegSamples = useEEGRecorder(postEegActive);

  // Step 6 — post-meal survey
  const [whatAte, setWhatAte] = useState("");
  const [foodHealthy, setFoodHealthy] = useState(5);
  const [postEnergy, setPostEnergy] = useState(5);
  const [postMood, setPostMood] = useState(5);
  const [postSatisfied, setPostSatisfied] = useState(5);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // ── Countdown callbacks ────────────────────────────────────────────────────

  const handlePreEegDone = useCallback(() => {
    setPreEegActive(false);
    setPreEegDone(true);
    setPreEegJson(averageEEG(preEegSamples));
    setStep(3);
  }, [preEegSamples]);

  const handleWaitDone = useCallback(() => {
    setWaitActive(false);
    setStep(5);
  }, []);

  const handlePostEegDone = useCallback(() => {
    setPostEegActive(false);
    setPostEegDone(true);
    setPostEegJson(averageEEG(postEegSamples));
    setStep(6);
  }, [postEegSamples]);

  const preCountdown = useCountdown(5 * 60, preEegActive, handlePreEegDone);
  const waitCountdown = useCountdown(5 * 60, waitActive, handleWaitDone);
  const postCountdown = useCountdown(10 * 60, postEegActive, handlePostEegDone);

  // ── Start session on mount ─────────────────────────────────────────────────

  useEffect(() => {
    if (!participantCode) return;
    setIsStarting(true);
    apiRequest("POST", "/api/study/session/start", {
      participant_code: participantCode,
      block_type: "food",
    })
      .then((res) => res.json())
      .then((data: { session_id: number }) => {
        setSessionId(data.session_id);
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : "Could not start session";
        toast({ title: "Session start failed", description: msg, variant: "destructive" });
      })
      .finally(() => setIsStarting(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [participantCode]);

  // ── Step 1 → 2 ────────────────────────────────────────────────────────────

  function handleStartBaseline() {
    setStep(2);
    setPreEegActive(true);
  }

  // ── Step 3 → 4 ────────────────────────────────────────────────────────────

  function handleFinishedEating() {
    setStep(4);
    setWaitActive(true);
  }

  // ── Submit ─────────────────────────────────────────────────────────────────

  async function handleComplete() {
    if (!sessionId) {
      toast({ title: "No session ID", description: "Session was not started properly.", variant: "destructive" });
      return;
    }

    setIsSubmitting(true);

    const survey_json = {
      what_ate: whatAte.trim(),
      pre_hunger: preHunger,
      pre_mood: preMood,
      food_healthy: foodHealthy,
      post_energy: postEnergy,
      post_mood: postMood,
      post_satisfied: postSatisfied,
    };

    try {
      await apiRequest("POST", "/api/study/session/complete", {
        session_id: sessionId,
        pre_eeg_json: preEegJson,
        post_eeg_json: postEegJson,
        survey_json,
      });
      navigate(`/study/complete?code=${encodeURIComponent(participantCode)}&done=food`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Submission failed";
      toast({ title: "Could not complete session", description: msg, variant: "destructive" });
    } finally {
      setIsSubmitting(false);
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  if (isStarting) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center space-y-3">
          <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto" />
          <p className="text-sm text-muted-foreground">Starting session…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-lg mx-auto px-4 py-10 space-y-6">

        {/* Header */}
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Utensils className="h-5 w-5 text-primary" />
            <h1 className="text-xl font-bold">Food Session</h1>
            {participantCode && (
              <Badge variant="outline" className="border-primary/40 text-primary font-mono ml-auto">
                {participantCode}
              </Badge>
            )}
          </div>
          <Stepper current={step} />
        </div>

        {/* ── Step 1: Pre-meal survey ─────────────────────────────────────── */}
        {step === 1 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-primary/10 text-primary text-xs flex items-center justify-center font-bold">1</span>
                Before Your Meal
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6 pt-2">
              <SliderRow label="How hungry are you right now?" value={preHunger} onChange={setPreHunger} />
              <SliderRow label="What is your current mood?" value={preMood} onChange={setPreMood} />
              <Button className="w-full" size="lg" onClick={handleStartBaseline}>
                Start Baseline Recording
                <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            </CardContent>
          </Card>
        )}

        {/* ── Step 2: Pre-meal EEG ────────────────────────────────────────── */}
        {step === 2 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-primary/10 text-primary text-xs flex items-center justify-center font-bold">2</span>
                Pre-Meal Baseline
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6 pt-2">
              <p className="text-sm text-muted-foreground">
                Keep the headband on and sit quietly for 5 minutes.
              </p>
              <div className="text-center py-8 space-y-3">
                <Brain className="h-10 w-10 text-primary mx-auto animate-pulse" />
                <p className="text-5xl font-mono font-bold tabular-nums tracking-widest">
                  {preCountdown.label}
                </p>
                <p className="text-xs text-muted-foreground">
                  {preEegSamples.length} EEG sample{preEegSamples.length !== 1 ? "s" : ""} collected
                </p>
              </div>
              <Progress value={((5 * 60 - preCountdown.remaining) / (5 * 60)) * 100} className="h-2" />
            </CardContent>
          </Card>
        )}

        {/* ── Step 3: Eat ─────────────────────────────────────────────────── */}
        {step === 3 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-primary/10 text-primary text-xs flex items-center justify-center font-bold">3</span>
                Time to Eat
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6 pt-2">
              {preEegDone && (
                <div className="flex items-center gap-2 text-xs text-green-400">
                  <CheckCircle2 className="h-4 w-4" />
                  Pre-meal baseline recorded successfully.
                </div>
              )}
              <div className="rounded-lg bg-muted/40 p-4 text-sm leading-relaxed text-muted-foreground space-y-2">
                <p className="font-medium text-foreground">Remove the headband and eat your normal meal.</p>
                <p>Come back when you are done eating.</p>
              </div>
              <Button className="w-full" size="lg" onClick={handleFinishedEating}>
                I finished eating
                <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            </CardContent>
          </Card>
        )}

        {/* ── Step 4: Wait 5 minutes ──────────────────────────────────────── */}
        {step === 4 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-primary/10 text-primary text-xs flex items-center justify-center font-bold">4</span>
                Please Wait
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6 pt-2">
              <p className="text-sm text-muted-foreground">
                Wait 5 minutes before putting the headband back on. This lets digestion begin.
              </p>
              <div className="text-center py-8 space-y-3">
                <Clock className="h-10 w-10 text-muted-foreground mx-auto" />
                <p className="text-5xl font-mono font-bold tabular-nums tracking-widest">
                  {waitCountdown.label}
                </p>
                <p className="text-xs text-muted-foreground">The next step will begin automatically.</p>
              </div>
              <Progress value={((5 * 60 - waitCountdown.remaining) / (5 * 60)) * 100} className="h-2" />
            </CardContent>
          </Card>
        )}

        {/* ── Step 5: Post-meal EEG ───────────────────────────────────────── */}
        {step === 5 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-primary/10 text-primary text-xs flex items-center justify-center font-bold">5</span>
                Post-Meal Recording
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6 pt-2">
              {!postEegActive && !postEegDone && (
                <>
                  <p className="text-sm text-muted-foreground">
                    Put the headband back on and sit quietly for 10 minutes.
                  </p>
                  <Button className="w-full" size="lg" onClick={() => setPostEegActive(true)}>
                    <Brain className="mr-2 h-4 w-4" />
                    Start Post-Meal Recording
                  </Button>
                </>
              )}
              {postEegActive && (
                <>
                  <p className="text-sm text-muted-foreground">
                    Put the headband back on and sit quietly.
                  </p>
                  <div className="text-center py-8 space-y-3">
                    <Brain className="h-10 w-10 text-primary mx-auto animate-pulse" />
                    <p className="text-5xl font-mono font-bold tabular-nums tracking-widest">
                      {postCountdown.label}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {postEegSamples.length} EEG sample{postEegSamples.length !== 1 ? "s" : ""} collected
                    </p>
                  </div>
                  <Progress value={((10 * 60 - postCountdown.remaining) / (10 * 60)) * 100} className="h-2" />
                </>
              )}
            </CardContent>
          </Card>
        )}

        {/* ── Step 6: Post-meal survey ────────────────────────────────────── */}
        {step === 6 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-primary/10 text-primary text-xs flex items-center justify-center font-bold">6</span>
                After Your Meal
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6 pt-2">
              {postEegDone && (
                <div className="flex items-center gap-2 text-xs text-green-400">
                  <CheckCircle2 className="h-4 w-4" />
                  Post-meal recording complete.
                </div>
              )}

              <div className="space-y-1.5">
                <Label className="text-sm">What did you eat? (brief description)</Label>
                <Textarea
                  placeholder="e.g. rice and lentils, a sandwich…"
                  value={whatAte}
                  onChange={(e) => setWhatAte(e.target.value)}
                  rows={2}
                  className="resize-none"
                />
              </div>

              <SliderRow label="How healthy was this meal?" value={foodHealthy} onChange={setFoodHealthy} />
              <SliderRow label="Energy level now?" value={postEnergy} onChange={setPostEnergy} />
              <SliderRow label="Mood now?" value={postMood} onChange={setPostMood} />
              <SliderRow label="Do you feel satisfied?" value={postSatisfied} onChange={setPostSatisfied} />

              <Button
                className="w-full"
                size="lg"
                disabled={isSubmitting || whatAte.trim().length === 0}
                onClick={handleComplete}
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Saving…
                  </>
                ) : (
                  <>
                    Complete Session
                    <ChevronRight className="ml-2 h-4 w-4" />
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
