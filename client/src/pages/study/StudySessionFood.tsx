import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation, useSearch } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Loader2, ChevronRight, Utensils, EyeOff, Eye, CheckCircle2 } from "lucide-react";
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

// ── Durations (seconds) ──────────────────────────────────────────────────────

const BASELINE_SEC = 5 * 60;   // 5 min pre-meal baseline
const POST_EEG_SEC = 10 * 60;  // 10 min post-meal EEG
const EATING_OPTIONS = [
  { label: "15 min", seconds: 15 * 60 },
  { label: "20 min", seconds: 20 * 60 },
  { label: "25 min", seconds: 25 * 60 },
  { label: "30 min", seconds: 30 * 60 },
];

// ── EEG helpers ───────────────────────────────────────────────────────────────

async function fetchEEG(): Promise<EEGSample> {
  try {
    const mlUrl = getMLApiUrl();
    const res = await fetch(`${mlUrl}/api/simulate-eeg`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state: "rest", duration: 4, fs: 256 }),
    });
    if (!res.ok) throw new Error("fetch failed");
    const data = await res.json();
    // ML backend returns: { analysis: { emotions: { band_powers, stress_index, ... } } }
    const emotions = data.analysis?.emotions ?? {};
    const bp = emotions.band_powers ?? {};
    return {
      alpha: bp.alpha ?? 0.3,
      beta: bp.beta ?? 0.25,
      theta: bp.theta ?? 0.15,
      delta: bp.delta ?? 0.1,
      gamma: bp.gamma ?? 0.03,
      stress_level: emotions.stress_index ?? 0.3,
    };
  } catch {
    return {
      alpha: 0.3 + Math.random() * 0.2,
      beta: 0.2 + Math.random() * 0.3,
      theta: 0.1 + Math.random() * 0.15,
      delta: 0.05 + Math.random() * 0.1,
      gamma: 0.02 + Math.random() * 0.05,
      stress_level: 0.3 + Math.random() * 0.4,
    };
  }
}

function averageEEG(samples: EEGSample[]): EEGSample | null {
  if (samples.length === 0) return null;
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

// ── Stepper ───────────────────────────────────────────────────────────────────

const PHASE_ORDER: Phase[] = ["pre_survey", "baseline", "eating", "post_eeg", "post_survey"];
const PHASE_LABELS: Record<Phase, string> = {
  pre_survey: "Pre-survey",
  baseline: "Baseline",
  eating: "Eat",
  post_eeg: "Post EEG",
  post_survey: "Post-survey",
};

function SessionStepper({ phase }: { phase: Phase }) {
  const current = PHASE_ORDER.indexOf(phase);
  return (
    <div className="w-full">
      <p className="text-xs text-muted-foreground mb-3 text-center">Food-Emotion Session</p>
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

  const [sessionId, setSessionId] = useState<number | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [phase, setPhase] = useState<Phase>("pre_survey");

  // Separate EEG buffers
  const [baselineReadings, setBaselineReadings] = useState<EEGSample[]>([]);
  const [postReadings, setPostReadings] = useState<EEGSample[]>([]);

  // Timer flags
  const [baselineActive, setBaselineActive] = useState(false);
  const [eatingActive, setEatingActive] = useState(false);
  const [postEegActive, setPostEegActive] = useState(false);

  // Eating duration (user-selectable)
  const [eatingDuration, setEatingDuration] = useState(20 * 60); // default 20 min

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

  // ── EEG polling (only during baseline and post-eeg phases) ────────────────

  const pollEEG = useCallback(async () => {
    const sample = await fetchEEG();
    if (baselineActive) {
      setBaselineReadings((prev) => [...prev, sample]);
    } else if (postEegActive) {
      setPostReadings((prev) => [...prev, sample]);
    }
  }, [baselineActive, postEegActive]);

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
  }, []);

  const onEatingDone = useCallback(() => {
    setEatingActive(false);
    setPhase("post_eeg");
    setPostEegActive(true);
  }, []);

  const onPostEegDone = useCallback(() => {
    setPostEegActive(false);
    setPhase("post_survey");
  }, []);

  // ── Timers ────────────────────────────────────────────────────────────────

  const remainingBaseline = useCountdown(BASELINE_SEC, baselineActive, onBaselineDone);
  const remainingEating = useCountdown(eatingDuration, eatingActive, onEatingDone);
  const remainingPostEeg = useCountdown(POST_EEG_SEC, postEegActive, onPostEegDone);

  // ── Submit ────────────────────────────────────────────────────────────────

  async function handleComplete() {
    if (isSubmitting) return;
    setIsSubmitting(true);

    const preEeg = averageEEG(baselineReadings);
    const postEeg = averageEEG(postReadings);
    const allReadings = [...baselineReadings, ...postReadings];
    const features = averageEEG(allReadings);

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
        pre_eeg_json: preEeg,
        post_eeg_json: postEeg,
        eeg_features_json: features,
        survey_json,
        intervention_triggered: false,
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
          <p className="text-sm text-muted-foreground">Starting session...</p>
        </div>
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

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-lg mx-auto px-4 py-10 space-y-6">

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
          <SessionStepper phase={phase} />
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

              <Button className="w-full" size="lg" onClick={() => { setPhase("baseline"); setBaselineActive(true); }}>
                Start Baseline Recording
                <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            </CardContent>
          </Card>
        )}

        {/* ── Step 2: Pre-meal baseline (5 min, eyes closed) ── */}
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
                <Progress value={((BASELINE_SEC - remainingBaseline) / BASELINE_SEC) * 100} className="h-2" />
                <p className="text-xs text-muted-foreground">5 minutes — eyes closed</p>
              </div>

              {recordingDot}
            </CardContent>
          </Card>
        )}

        {/* ── Step 3: Eat your meal (headband off, user-set timer) ── */}
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
                <p className="text-xs text-muted-foreground">{eatingDuration / 60} minutes</p>
              </div>

              <Button
                className="w-full"
                size="lg"
                variant="outline"
                onClick={() => {
                  setEatingActive(false);
                  setPhase("post_eeg");
                  setPostEegActive(true);
                }}
              >
                I'm done eating — Continue
              </Button>
            </CardContent>
          </Card>
        )}

        {/* ── Step 4: Post-meal EEG (10 min) ── */}
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
                <Progress value={((POST_EEG_SEC - remainingPostEeg) / POST_EEG_SEC) * 100} className="h-2" />
                <p className="text-xs text-muted-foreground">10 minutes</p>
              </div>

              {recordingDot}
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
