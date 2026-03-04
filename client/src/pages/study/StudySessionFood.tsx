import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation, useSearch } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Loader2, ChevronRight, Utensils, Brain, Wind, CheckCircle2 } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
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

type Phase = "survey1" | "task1" | "checkin" | "breathing" | "task2" | "survey2";

// ── EEG helpers ───────────────────────────────────────────────────────────────

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

function averageEEG(samples: EEGSample[]): EEGSample {
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
    alpha: sum.alpha / n, beta: sum.beta / n, theta: sum.theta / n,
    delta: sum.delta / n, gamma: sum.gamma / n, stress_level: sum.stress_level / n,
  };
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

// ── Box Breathing ─────────────────────────────────────────────────────────────

function BoxBreathing() {
  const phases = ["Inhale", "Hold", "Exhale", "Hold"];
  const labels = ["Inhale 4 counts", "Hold 4 counts", "Exhale 4 counts", "Hold 4 counts"];
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setPhase((p) => (p + 1) % 4), 4000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex flex-col items-center gap-6 py-4">
      <div
        className="relative flex items-center justify-center transition-all duration-[3800ms] ease-in-out"
        style={{ width: phase === 0 || phase === 2 ? 160 : 120, height: phase === 0 || phase === 2 ? 160 : 120 }}
      >
        <div className="rounded-2xl bg-primary/20 border-2 border-primary/60 absolute inset-0 transition-all duration-[3800ms] ease-in-out" />
        <span className="text-sm font-semibold text-primary z-10">{phases[phase]}</span>
      </div>
      <p className="text-base font-medium text-foreground">{labels[phase]}</p>
      <div className="flex gap-2">
        {phases.map((_, i) => (
          <div key={i} className={`h-1.5 w-8 rounded-full transition-colors ${i === phase ? "bg-primary" : "bg-muted"}`} />
        ))}
      </div>
    </div>
  );
}

// ── Stepper ───────────────────────────────────────────────────────────────────

const PHASE_ORDER: Phase[] = ["survey1", "task1", "checkin", "breathing", "task2", "survey2"];
const PHASE_LABELS: Record<Phase, string> = {
  survey1: "Pre", task1: "Session", checkin: "Check", breathing: "Breathe", task2: "Session", survey2: "Post",
};

function SessionStepper({ phase }: { phase: Phase }) {
  const current = PHASE_ORDER.indexOf(phase);
  return (
    <div className="w-full">
      <p className="text-xs text-muted-foreground mb-3 text-center">Food session — 20 minutes</p>
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

function SliderRow({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-sm">{label}</Label>
        <Badge variant="outline" className="font-mono tabular-nums w-10 text-center">{value}</Badge>
      </div>
      <Slider min={1} max={10} step={1} value={[value]} onValueChange={([v]) => onChange(v)} className="w-full" />
      <div className="flex justify-between text-xs text-muted-foreground"><span>1</span><span>10</span></div>
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

  const [sessionId, setSessionId] = useState<number | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [phase, setPhase] = useState<Phase>("survey1");

  // EEG
  const [eegSamples, setEegSamples] = useState<EEGSample[]>([]);
  const [breathingUsed, setBreathingUsed] = useState(false);

  // Timer flags
  const [task1Active, setTask1Active] = useState(false);
  const [breathActive, setBreathActive] = useState(false);
  const [task2Active, setTask2Active] = useState(false);

  // Survey 1 state
  const [preHunger, setPreHunger] = useState(5);
  const [preMood, setPreMood] = useState(5);

  // Survey 2 state
  const [whatAte, setWhatAte] = useState("");
  const [foodHealthy, setFoodHealthy] = useState(5);
  const [postEnergy, setPostEnergy] = useState(5);
  const [postMood, setPostMood] = useState(5);
  const [postSatisfied, setPostSatisfied] = useState(5);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // ── Session start ─────────────────────────────────────────────────────────

  useEffect(() => {
    if (!participantCode) return;
    setIsStarting(true);
    apiRequest("POST", "/api/study/session/start", { participant_code: participantCode, block_type: "food" })
      .then((res) => res.json())
      .then((data: { session_id: number }) => setSessionId(data.session_id))
      .catch(() => setSessionId(-1))
      .finally(() => setIsStarting(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [participantCode]);

  // ── EEG polling ───────────────────────────────────────────────────────────

  const pollEEG = useCallback(async () => {
    try {
      const res = await fetch("/api/simulate-eeg", { credentials: "include" });
      const data: EEGSample = res.ok ? await res.json() : mockEEG();
      setEegSamples((prev) => [...prev, data]);
    } catch {
      setEegSamples((prev) => [...prev, mockEEG()]);
    }
  }, []);

  useEffect(() => {
    if (!task1Active && !task2Active) return;
    const id = setInterval(pollEEG, 4000);
    return () => clearInterval(id);
  }, [task1Active, task2Active, pollEEG]);

  // ── Timer callbacks ───────────────────────────────────────────────────────

  const onTask1Done = useCallback(() => { setTask1Active(false); setPhase("checkin"); }, []);
  const onBreathDone = useCallback(() => { setBreathActive(false); setPhase("task2"); setTask2Active(true); }, []);
  const onTask2Done = useCallback(() => { setTask2Active(false); setPhase("survey2"); }, []);

  // ── Timers ────────────────────────────────────────────────────────────────

  const remaining1 = useCountdown(10 * 60, task1Active, onTask1Done);
  const remainingBreath = useCountdown(3 * 60, breathActive, onBreathDone);
  const remaining2 = useCountdown(10 * 60, task2Active, onTask2Done);

  // Show single 20-min countdown
  const displayTime = task1Active ? remaining1 + 10 * 60 : remaining2;

  // ── Check-in ─────────────────────────────────────────────────────────────

  function onCheckinYes() { setBreathingUsed(true); setPhase("breathing"); setBreathActive(true); }
  function onCheckinNo() { setPhase("task2"); setTask2Active(true); }

  // ── Start session button ──────────────────────────────────────────────────

  function handleStart() { setPhase("task1"); setTask1Active(true); }

  // ── Submit ────────────────────────────────────────────────────────────────

  async function handleComplete() {
    if (isSubmitting) return;
    setIsSubmitting(true);
    const avgEeg = averageEEG(eegSamples);
    const survey_json = {
      what_ate: whatAte.trim(), pre_hunger: preHunger, pre_mood: preMood,
      food_healthy: foodHealthy, post_energy: postEnergy, post_mood: postMood,
      post_satisfied: postSatisfied, breathing_used: breathingUsed,
    };
    try {
      await apiRequest("POST", "/api/study/session/complete", {
        session_id: sessionId,
        pre_eeg_json: avgEeg,
        post_eeg_json: avgEeg,
        survey_json,
        intervention_triggered: breathingUsed,
      });
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
          <p className="text-sm text-muted-foreground">Starting session…</p>
        </div>
      </div>
    );
  }

  // ── Render ────────────────────────────────────────────────────────────────

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
          <SessionStepper phase={phase} />
        </div>

        {/* ── Pre-session survey ── */}
        {phase === "survey1" && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-primary/10 text-primary text-xs flex items-center justify-center font-bold">1</span>
                Before Your Session
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6 pt-2">
              <SliderRow label="How hungry are you right now?" value={preHunger} onChange={setPreHunger} />
              <SliderRow label="What is your current mood?" value={preMood} onChange={setPreMood} />
              <Button className="w-full" size="lg" onClick={handleStart}>
                Begin 20-Minute Session
                <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            </CardContent>
          </Card>
        )}

        {/* ── Task phases (task1 & task2) ── */}
        {(phase === "task1" || phase === "task2") && (
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-primary animate-pulse" />
                <CardTitle className="text-lg">
                  Food Session
                  {phase === "task2" && (
                    <span className="ml-2 text-sm font-normal text-muted-foreground">(second half)</span>
                  )}
                </CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground leading-relaxed">
                {phase === "task1"
                  ? "Eat your meal and keep the headband on. We are recording your EEG."
                  : "Continue with your normal activity. Recording post-meal EEG."}
              </p>

              {/* Timer */}
              <div className="flex flex-col items-center gap-3">
                <span className="text-5xl font-mono font-bold tracking-tight">{formatTime(displayTime)}</span>
                <Progress
                  value={phase === "task1" ? ((20 * 60 - displayTime) / (20 * 60)) * 100 : ((10 * 60 - remaining2) / (10 * 60)) * 100}
                  className="h-2"
                />
                <p className="text-xs text-muted-foreground">
                  {phase === "task1" ? "20:00 total" : "final 10 min"}
                </p>
              </div>

              {/* Recording dot */}
              <div className="flex items-center justify-center gap-2">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
                </span>
                <span className="text-sm text-muted-foreground">
                  Recording… <span className="font-medium text-foreground">{eegSamples.length} samples</span>
                </span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* ── Mid-session check-in ── */}
        {phase === "checkin" && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Quick check-in</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-base font-medium text-center">Are you feeling stressed right now?</p>
              <div className="grid grid-cols-2 gap-3">
                <Button size="lg" variant="destructive" className="h-16 text-base" onClick={onCheckinYes}>Yes</Button>
                <Button size="lg" className="h-16 text-base" onClick={onCheckinNo}>No</Button>
              </div>
              <p className="text-xs text-muted-foreground text-center">10 minutes remaining in your session.</p>
            </CardContent>
          </Card>
        )}

        {/* ── Breathing ── */}
        {phase === "breathing" && (
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <Wind className="h-5 w-5 text-primary" />
                <CardTitle className="text-lg">Breathing Exercise</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground text-center leading-relaxed">
                Inhale 4 counts → Hold 4 counts → Exhale 4 counts → Hold 4 counts
              </p>
              <div className="flex flex-col items-center gap-1">
                <span className="text-2xl font-mono font-bold text-primary">{formatTime(remainingBreath)}</span>
                <p className="text-xs text-muted-foreground">remaining</p>
              </div>
              <BoxBreathing />
              <Button className="w-full" size="lg" onClick={() => { setBreathActive(false); setPhase("task2"); setTask2Active(true); }}>
                I feel calmer — Continue
              </Button>
              <p className="text-xs text-muted-foreground text-center">Do at least 3 rounds (1 min). Continue whenever ready.</p>
            </CardContent>
          </Card>
        )}

        {/* ── Post-session survey ── */}
        {phase === "survey2" && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-primary/10 text-primary text-xs flex items-center justify-center font-bold">6</span>
                After Your Session
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6 pt-2">
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
                className="w-full" size="lg"
                disabled={isSubmitting || whatAte.trim().length === 0}
                onClick={handleComplete}
              >
                {isSubmitting ? (
                  <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Saving…</>
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
