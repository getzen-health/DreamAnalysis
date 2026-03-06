import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation, useSearch } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { Loader2, Wind, CheckCircle2, Eye, EyeOff } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { getMLApiUrl } from "@/lib/ml-api";
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

type Phase = "baseline" | "work" | "breathing" | "post" | "survey";

// ── Helpers ───────────────────────────────────────────────────────────────────

async function fetchEEG(): Promise<EEGReading> {
  try {
    const mlUrl = getMLApiUrl();
    const res = await fetch(`${mlUrl}/api/simulate-eeg`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state: "stressed", duration: 4, fs: 256 }),
    });
    if (!res.ok) throw new Error("EEG fetch failed");
    const data = await res.json();
    // Map ML backend response to our EEGReading shape
    const bp = data.band_powers ?? data.features?.band_powers ?? {};
    return {
      alpha: bp.alpha ?? 0.3,
      beta: bp.beta ?? 0.25,
      theta: bp.theta ?? 0.15,
      delta: bp.delta ?? 0.1,
      gamma: bp.gamma ?? 0.03,
      stress_level: data.emotions?.stress_index ?? data.stress_index ?? 0.4,
    };
  } catch {
    // Fallback: generate plausible simulated EEG data
    return {
      alpha: 0.3 + Math.random() * 0.2,
      beta: 0.2 + Math.random() * 0.3,
      theta: 0.1 + Math.random() * 0.15,
      delta: 0.05 + Math.random() * 0.1,
      gamma: 0.02 + Math.random() * 0.05,
      stress_level: 0.3 + Math.random() * 0.5,
    };
  }
}

function averageReadings(readings: EEGReading[]): EEGReading | null {
  if (readings.length === 0) return null;
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

function stressTextColor(level: number): string {
  if (level < 0.4) return "text-green-400";
  if (level <= 0.65) return "text-yellow-400";
  return "text-red-400";
}

function stressLabel(level: number): string {
  if (level < 0.4) return "Calm";
  if (level <= 0.65) return "Mild Stress";
  return "High Stress";
}

// ── Durations (seconds) ──────────────────────────────────────────────────────

const BASELINE_SEC = 5 * 60;  // 5 min baseline
const WORK_SEC = 15 * 60;     // 15 min work
const BREATHING_SEC = 3 * 60; // 3 min breathing
const POST_SEC = 5 * 60;      // 5 min post-session
const STRESS_THRESHOLD = 0.65;

// ── Session stepper ───────────────────────────────────────────────────────────

const PHASE_LABELS: Record<Phase, string> = {
  baseline:  "Baseline",
  work:      "Work",
  breathing: "Breathing",
  post:      "Post-session",
  survey:    "Survey",
};

function SessionStepper({ phase }: { phase: Phase }) {
  const visible: Phase[] = ["baseline", "work", "breathing", "post", "survey"];
  const current = visible.indexOf(phase);

  return (
    <div className="w-full">
      <p className="text-xs text-muted-foreground mb-3 text-center">
        ~28 minute session
      </p>
      <div className="flex items-center gap-0">
        {visible.map((p, idx) => {
          const done = idx < current;
          const active = idx === current;
          return (
            <div key={p} className="flex items-center flex-1">
              <div className="flex flex-col items-center w-full">
                <div
                  className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold shrink-0 transition-colors ${
                    done
                      ? "bg-primary text-primary-foreground"
                      : active
                      ? "bg-primary text-primary-foreground ring-2 ring-primary/40"
                      : "bg-muted text-muted-foreground"
                  }`}
                >
                  {done ? <CheckCircle2 className="h-3.5 w-3.5" /> : idx + 1}
                </div>
                <span className="text-[10px] text-muted-foreground mt-1 text-center leading-tight">
                  {PHASE_LABELS[p]}
                </span>
              </div>
              {idx < visible.length - 1 && (
                <div
                  className={`h-0.5 flex-1 mb-4 mx-0.5 transition-colors ${
                    done ? "bg-primary" : "bg-muted"
                  }`}
                />
              )}
            </div>
          );
        })}
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

// ── Box Breathing ─────────────────────────────────────────────────────────────

function BoxBreathing() {
  const phases = ["Inhale", "Hold", "Exhale", "Hold"];
  const labels = ["Inhale 4 counts", "Hold 4 counts", "Exhale 4 counts", "Hold 4 counts"];
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setPhase((p) => (p + 1) % 4);
    }, 4000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex flex-col items-center gap-6 py-4">
      <div
        className="relative flex items-center justify-center transition-all duration-[3800ms] ease-in-out"
        style={{
          width: phase === 0 || phase === 2 ? 160 : 120,
          height: phase === 0 || phase === 2 ? 160 : 120,
        }}
      >
        <div className="rounded-2xl bg-primary/20 border-2 border-primary/60 absolute inset-0 transition-all duration-[3800ms] ease-in-out" />
        <span className="text-sm font-semibold text-primary z-10">{phases[phase]}</span>
      </div>
      <p className="text-base font-medium text-foreground">{labels[phase]}</p>
      <div className="flex gap-2">
        {phases.map((_, i) => (
          <div
            key={i}
            className={`h-1.5 w-8 rounded-full transition-colors ${i === phase ? "bg-primary" : "bg-muted"}`}
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

  const [sessionId, setSessionId] = useState<number | null>(null);
  const [phase, setPhase] = useState<Phase>("baseline");
  const [isStarting, setIsStarting] = useState(true);

  // Separate EEG buffers for pre (baseline) and post phases
  const [baselineReadings, setBaselineReadings] = useState<EEGReading[]>([]);
  const [workReadings, setWorkReadings] = useState<EEGReading[]>([]);
  const [postReadings, setPostReadings] = useState<EEGReading[]>([]);
  const [liveStress, setLiveStress] = useState(0);
  const [interventionTriggered, setInterventionTriggered] = useState(false);
  const interventionFiredRef = useRef(false);

  // Timer active flags
  const [baselineActive, setBaselineActive] = useState(false);
  const [workActive, setWorkActive] = useState(false);
  const [breathActive, setBreathActive] = useState(false);
  const [postActive, setPostActive] = useState(false);

  // Survey
  const [surveyStressed, setSurveyStressed] = useState<number | null>(null);
  const [surveyBreathing, setSurveyBreathing] = useState<number | null>(null);
  const [surveyNow, setSurveyNow] = useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // ── Session start ─────────────────────────────────────────────────────────────

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
          setBaselineActive(true);
        }
      } catch {
        if (!cancelled) {
          setSessionId(-1);
          setIsStarting(false);
          setBaselineActive(true);
        }
      }
    }
    startSession();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── EEG polling ───────────────────────────────────────────────────────────────

  const pollEEG = useCallback(async () => {
    const reading = await fetchEEG();
    setLiveStress(reading.stress_level);

    if (baselineActive) {
      setBaselineReadings((prev) => [...prev, reading]);
    } else if (workActive) {
      setWorkReadings((prev) => [...prev, reading]);
      // Auto-trigger intervention when stress exceeds threshold
      if (reading.stress_level > STRESS_THRESHOLD && !interventionFiredRef.current) {
        interventionFiredRef.current = true;
        setInterventionTriggered(true);
        setWorkActive(false);
        setPhase("breathing");
        setBreathActive(true);
      }
    } else if (postActive) {
      setPostReadings((prev) => [...prev, reading]);
    }
  }, [baselineActive, workActive, postActive]);

  useEffect(() => {
    if (!baselineActive && !workActive && !postActive) return;
    const id = setInterval(pollEEG, 4000);
    pollEEG(); // immediate first poll
    return () => clearInterval(id);
  }, [baselineActive, workActive, postActive, pollEEG]);

  // ── Timer callbacks ───────────────────────────────────────────────────────────

  const onBaselineDone = useCallback(() => {
    setBaselineActive(false);
    setPhase("work");
    setWorkActive(true);
  }, []);

  const onWorkDone = useCallback(() => {
    setWorkActive(false);
    // If stress never crossed threshold, go to breathing anyway (per PRD: OR at 15 min mark)
    if (!interventionFiredRef.current) {
      interventionFiredRef.current = true;
      setInterventionTriggered(true);
      setPhase("breathing");
      setBreathActive(true);
    }
  }, []);

  const onBreathDone = useCallback(() => {
    setBreathActive(false);
    setPhase("post");
    setPostActive(true);
  }, []);

  const onPostDone = useCallback(() => {
    setPostActive(false);
    setPhase("survey");
  }, []);

  // ── Timers ────────────────────────────────────────────────────────────────────

  const remainingBaseline = useCountdown(BASELINE_SEC, baselineActive, onBaselineDone);
  const remainingWork = useCountdown(WORK_SEC, workActive, onWorkDone);
  const remainingBreath = useCountdown(BREATHING_SEC, breathActive, onBreathDone);
  const remainingPost = useCountdown(POST_SEC, postActive, onPostDone);

  // ── Survey submit ─────────────────────────────────────────────────────────────

  const canComplete =
    surveyStressed !== null &&
    (interventionTriggered ? surveyBreathing !== null : true) &&
    surveyNow !== null;

  async function handleComplete() {
    if (!canComplete || isSubmitting) return;
    setIsSubmitting(true);

    const preEeg = averageReadings(baselineReadings);
    const postEeg = averageReadings(postReadings);
    const allReadings = [...baselineReadings, ...workReadings, ...postReadings];
    const features = averageReadings(allReadings);

    const surveyData = {
      stressed_during: surveyStressed ?? 5,
      breathing_helped: surveyBreathing ?? null,
      feeling_now: surveyNow ?? 5,
    };

    try {
      await apiRequest("POST", "/api/study/session/complete", {
        session_id: sessionId,
        pre_eeg_json: preEeg,
        post_eeg_json: postEeg,
        eeg_features_json: features,
        survey_json: surveyData,
        intervention_triggered: interventionTriggered,
      });
      navigate(`/study/complete?code=${encodeURIComponent(participantCode)}&done=stress`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Submission failed";
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
          <p className="text-sm">Setting up your session...</p>
        </div>
      </div>
    );
  }

  // ── Recording indicator (shared) ──────────────────────────────────────────────

  const sampleCount = baselineReadings.length + workReadings.length + postReadings.length;
  const recordingDot = (
    <div className="flex items-center justify-center gap-2">
      <span className="relative flex h-2.5 w-2.5">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
        <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
      </span>
      <span className="text-sm text-muted-foreground">
        Recording...{" "}
        <span className="font-medium text-foreground">{sampleCount} samples</span>
      </span>
    </div>
  );

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-lg mx-auto px-4 py-8 space-y-6">

        <SessionStepper phase={phase} />

        {/* ── Baseline phase ── */}
        {phase === "baseline" && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <EyeOff className="h-5 w-5 text-primary" />
                Baseline Recording
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground leading-relaxed">
                Sit still and close your eyes. Breathe normally. This captures your resting brain state.
              </p>

              <div className="flex flex-col items-center gap-3">
                <span className="text-5xl font-mono font-bold tracking-tight">
                  {formatTime(remainingBaseline)}
                </span>
                <Progress
                  value={((BASELINE_SEC - remainingBaseline) / BASELINE_SEC) * 100}
                  className="h-2"
                />
                <p className="text-xs text-muted-foreground">5 minutes — eyes closed</p>
              </div>

              {recordingDot}
            </CardContent>
          </Card>
        )}

        {/* ── Work phase ── */}
        {phase === "work" && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Work Session</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground leading-relaxed">
                Continue your normal work with the headband on. A breathing exercise will
                start automatically if your stress rises.
              </p>

              <div className="flex flex-col items-center gap-3">
                <span className="text-5xl font-mono font-bold tracking-tight">
                  {formatTime(remainingWork)}
                </span>
                <Progress
                  value={((WORK_SEC - remainingWork) / WORK_SEC) * 100}
                  className="h-2"
                />
                <p className="text-xs text-muted-foreground">15 minutes</p>
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
                {liveStress > 0.5 && (
                  <p className="text-xs text-yellow-400 text-center">
                    Breathing exercise will start automatically at high stress
                  </p>
                )}
              </div>

              {recordingDot}
            </CardContent>
          </Card>
        )}

        {/* ── Breathing exercise ── */}
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
                Inhale 4 counts, Hold 4 counts, Exhale 4 counts, Hold 4 counts
              </p>

              <div className="flex flex-col items-center gap-1">
                <span className="text-2xl font-mono font-bold text-primary">
                  {formatTime(remainingBreath)}
                </span>
                <p className="text-xs text-muted-foreground">remaining</p>
              </div>

              <BoxBreathing />

              <Button
                className="w-full"
                size="lg"
                onClick={() => {
                  setBreathActive(false);
                  setPhase("post");
                  setPostActive(true);
                }}
              >
                I feel calmer — Continue
              </Button>

              <p className="text-xs text-muted-foreground text-center">
                Do at least 3 rounds (1 min). You can continue whenever ready.
              </p>
            </CardContent>
          </Card>
        )}

        {/* ── Post-session recording ── */}
        {phase === "post" && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <Eye className="h-5 w-5 text-primary" />
                Post-Session Recording
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm text-muted-foreground leading-relaxed">
                Sit still and relax. We're recording your brain state after the session to compare with your baseline.
              </p>

              <div className="flex flex-col items-center gap-3">
                <span className="text-5xl font-mono font-bold tracking-tight">
                  {formatTime(remainingPost)}
                </span>
                <Progress
                  value={((POST_SEC - remainingPost) / POST_SEC) * 100}
                  className="h-2"
                />
                <p className="text-xs text-muted-foreground">5 minutes — sit still</p>
              </div>

              {/* Live stress (should be lower after breathing) */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground uppercase tracking-wide">
                    Current stress
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
              </div>

              {recordingDot}
            </CardContent>
          </Card>
        )}

        {/* ── Survey ── */}
        {phase === "survey" && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Quick Survey</CardTitle>
            </CardHeader>
            <CardContent className="space-y-8">
              <p className="text-sm text-muted-foreground">
                A few quick questions — answer honestly.
              </p>

              {/* Q1 */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium leading-snug">
                    How stressed did you feel during the session?
                  </p>
                  <span className="text-2xl font-bold text-primary">{surveyStressed ?? "—"}</span>
                </div>
                <Slider
                  min={1} max={10} step={1}
                  value={surveyStressed !== null ? [surveyStressed] : [5]}
                  onValueChange={([v]) => setSurveyStressed(v)}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Very calm</span><span>Extremely stressed</span>
                </div>
              </div>

              {/* Q2 — breathing helpfulness */}
              {interventionTriggered && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium leading-snug">
                      How helpful was the breathing exercise?
                    </p>
                    <span className="text-2xl font-bold text-primary">{surveyBreathing ?? "—"}</span>
                  </div>
                  <Slider
                    min={1} max={10} step={1}
                    value={surveyBreathing !== null ? [surveyBreathing] : [5]}
                    onValueChange={([v]) => setSurveyBreathing(v)}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Not helpful</span><span>Very helpful</span>
                  </div>
                </div>
              )}

              {/* Q3 */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium leading-snug">How do you feel right now?</p>
                  <span className="text-2xl font-bold text-primary">{surveyNow ?? "—"}</span>
                </div>
                <Slider
                  min={1} max={10} step={1}
                  value={surveyNow !== null ? [surveyNow] : [5]}
                  onValueChange={([v]) => setSurveyNow(v)}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Very stressed</span><span>Very calm</span>
                </div>
              </div>

              <Button
                className="w-full"
                size="lg"
                disabled={!canComplete || isSubmitting}
                onClick={handleComplete}
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Saving...
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
