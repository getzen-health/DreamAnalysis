import { getParticipantId } from "@/lib/participant";
import { useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import {
  Brain, CheckCircle2, Loader2, Wind, Utensils,
  ExternalLink, ChevronRight, Coffee, Timer,
} from "lucide-react";
import { apiRequest, resolveUrl } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

const USER_ID = getParticipantId();

// ── Types ─────────────────────────────────────────────────────────────────────

type Step = "intake" | "breathing" | "stress-session" | "food-session" | "done";

// ── SAM scales ────────────────────────────────────────────────────────────────

const VALENCE_LABELS: Record<number, string> = { 1:"😞", 3:"😟", 5:"😐", 7:"🙂", 9:"😊" };
const AROUSAL_LABELS: Record<number, string> = { 1:"😴", 3:"😌", 5:"😐", 7:"⚡", 9:"🔥" };
const STRESS_LABELS:  Record<number, string> = { 1:"😌", 3:"🙂", 5:"😐", 7:"😰", 9:"😱" };

function closestLabel(val: number, labels: Record<number, string>) {
  const keys = Object.keys(labels).map(Number);
  const nearest = keys.reduce((a, b) => Math.abs(b - val) < Math.abs(a - val) ? b : a);
  return labels[nearest];
}

function RatingSlider({
  label, value, onChange, labels,
}: {
  label: string; value: number; onChange: (v: number) => void;
  labels: Record<number, string>;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium">{label}</p>
        <span className="text-xl">{closestLabel(value, labels)}</span>
      </div>
      <Slider min={1} max={9} step={1} value={[value]} onValueChange={([v]) => onChange(v)} />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>1</span><span>{value}</span><span>9</span>
      </div>
    </div>
  );
}

// ── Breathing exercise (box breathing 4-4-4-4) ────────────────────────────────

const BREATHING_PHASES = [
  { label: "Inhale", duration: 4000 },
  { label: "Hold",   duration: 4000 },
  { label: "Exhale", duration: 4000 },
  { label: "Hold",   duration: 4000 },
];
const TOTAL_CYCLES = 5;

function BreathingExercise({ onDone }: { onDone: () => void }) {
  const [phaseIdx, setPhaseIdx] = useState(0);
  const [cycle,    setCycle]    = useState(1);
  const [elapsed,  setElapsed]  = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const phase    = BREATHING_PHASES[phaseIdx];
  const progress = (elapsed / phase.duration) * 100;
  const expand   = phaseIdx === 0 || phaseIdx === 1; // inhale/hold phases = expand

  useEffect(() => {
    intervalRef.current = setInterval(() => {
      setElapsed(e => {
        const next = e + 50;
        if (next >= phase.duration) {
          const nextPhase = (phaseIdx + 1) % BREATHING_PHASES.length;
          const nextCycle = phaseIdx === BREATHING_PHASES.length - 1 ? cycle + 1 : cycle;
          if (nextCycle > TOTAL_CYCLES) {
            clearInterval(intervalRef.current!);
            onDone();
            return 0;
          }
          setCycle(nextCycle);
          setPhaseIdx(nextPhase);
          return 0;
        }
        return next;
      });
    }, 50);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phaseIdx, cycle]);

  return (
    <div className="max-w-lg mx-auto py-8 px-4 space-y-8 text-center">
      <div>
        <Wind className="w-7 h-7 text-sky-400 mx-auto mb-2" />
        <h2 className="text-xl font-bold">Box Breathing</h2>
        <p className="text-sm text-muted-foreground mt-1">
          We detected elevated stress. A short breathing exercise will help you relax before recording.
        </p>
      </div>

      {/* Animated circle */}
      <div className="flex justify-center items-center" style={{ height: 200 }}>
        <div
          className="rounded-full bg-sky-500/20 border border-sky-500/40 flex items-center justify-center transition-all"
          style={{
            width:  expand ? 160 : 100,
            height: expand ? 160 : 100,
            transition: `width ${phase.duration}ms ease-in-out, height ${phase.duration}ms ease-in-out`,
          }}
        >
          <span className="text-lg font-semibold text-sky-300">{phase.label}</span>
        </div>
      </div>

      <Progress value={progress} className="h-1.5" />

      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span>Cycle {cycle} / {TOTAL_CYCLES}</span>
        <span>{Math.ceil((phase.duration - elapsed) / 1000)}s</span>
      </div>

      <Button variant="ghost" size="sm" className="text-muted-foreground" onClick={onDone}>
        Skip breathing
      </Button>
    </div>
  );
}

// ── Session timer (20 min countdown) ─────────────────────────────────────────

function SessionTimer({
  title, icon: Icon, iconColor, hint, linkLabel, linkPath, onDone,
}: {
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  iconColor: string;
  hint: string;
  linkLabel: string;
  linkPath: string;
  onDone: () => void;
}) {
  const [, navigate] = useLocation();
  const SESSION_SECS = 20 * 60;
  const [remaining, setRemaining] = useState(SESSION_SECS);
  const [running,   setRunning]   = useState(false);
  const ref = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!running) return;
    ref.current = setInterval(() => {
      setRemaining(r => {
        if (r <= 1) {
          clearInterval(ref.current!);
          return 0;
        }
        return r - 1;
      });
    }, 1000);
    return () => { if (ref.current) clearInterval(ref.current); };
  }, [running]);

  const mins = String(Math.floor(remaining / 60)).padStart(2, "0");
  const secs = String(remaining % 60).padStart(2, "0");
  const pct  = ((SESSION_SECS - remaining) / SESSION_SECS) * 100;

  return (
    <div className="max-w-lg mx-auto py-6 px-4 space-y-6">
      <div className="flex items-center gap-3">
        <Icon className={`w-6 h-6 ${iconColor}`} />
        <div>
          <h2 className="text-xl font-bold">{title}</h2>
          <p className="text-xs text-muted-foreground">{hint}</p>
        </div>
      </div>

      {/* Timer display */}
      <Card>
        <CardContent className="pt-6 pb-6 text-center space-y-4">
          <p className="text-6xl font-mono font-bold tracking-wider">
            {mins}:{secs}
          </p>
          <Progress value={pct} className="h-2" />
          <div className="flex justify-center gap-3 pt-1">
            {!running && remaining === SESSION_SECS && (
              <Button className="bg-violet-600 hover:bg-violet-700" onClick={() => setRunning(true)}>
                Start 20-min session
              </Button>
            )}
            {running && (
              <Button variant="outline" onClick={() => setRunning(false)}>
                Pause
              </Button>
            )}
            {!running && remaining < SESSION_SECS && (
              <Button className="bg-violet-600 hover:bg-violet-700" onClick={() => setRunning(true)}>
                Resume
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Open EEG / tool */}
      <button
        onClick={() => navigate(linkPath)}
        className="w-full flex items-center gap-3 rounded-xl border border-violet-500/30 bg-violet-500/5 hover:bg-violet-500/10 p-4 transition-colors text-left"
      >
        <ExternalLink className="w-5 h-5 text-violet-400 shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium">{linkLabel}</p>
          <p className="text-xs text-muted-foreground">Opens in the same app — come back here when done</p>
        </div>
        <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />
      </button>

      <Button
        className="w-full"
        variant={remaining === 0 ? "default" : "outline"}
        onClick={onDone}
      >
        {remaining === 0 ? "Session complete — next" : "I'm done recording — next"}
      </Button>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function ResearchDaytime() {
  const { toast } = useToast();
  const [, navigate] = useLocation();

  // Intake state
  const [samValence,  setSamValence]  = useState(5);
  const [samArousal,  setSamArousal]  = useState(5);
  const [samStress,   setSamStress]   = useState(5);
  const [caffeine,    setCaffeine]    = useState(0);
  const [significant, setSignificant] = useState<boolean | null>(null);

  const [step,        setStep]        = useState<Step>("intake");
  const [submitting,  setSubmitting]  = useState(false);

  const { data: status } = useQuery({
    queryKey: ["/api/study/status", USER_ID],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/study/status/${USER_ID}`), { credentials: "include" });
      if (!res.ok) throw new Error("Failed");
      return res.json() as Promise<{ enrolled: boolean; todaySession: { daytimeCompleted: boolean } | null }>;
    },
  });

  const alreadyDone = status?.todaySession?.daytimeCompleted === true;

  // ── Submit ─────────────────────────────────────────────────────────────────
  async function handleSubmit() {
    setSubmitting(true);
    try {
      await apiRequest("POST", "/api/study/daytime", {
        userId: USER_ID,
        samValence, samArousal, samStress,
        caffeineServings: caffeine,
        significantEventYN: significant ?? false,
      });
      setStep("done");
    } catch {
      toast({ title: "Submit failed", description: "Try again in a moment.", variant: "destructive" });
    } finally {
      setSubmitting(false);
    }
  }

  // ── Intake → next step ─────────────────────────────────────────────────────
  function handleIntakeContinue() {
    if (significant === null) {
      toast({ title: "One more thing", description: "Did anything significant happen today?", variant: "destructive" });
      return;
    }
    // Breathing only if stress > 5
    if (samStress > 5) {
      setStep("breathing");
    } else {
      setStep("stress-session");
    }
  }

  // ── Already done ───────────────────────────────────────────────────────────
  if (alreadyDone || step === "done") {
    return (
      <div className="max-w-lg mx-auto py-16 px-4 text-center space-y-4">
        <CheckCircle2 className="w-12 h-12 text-green-400 mx-auto" />
        <h2 className="text-xl font-bold">Daytime sessions complete</h2>
        <p className="text-sm text-muted-foreground">
          Both your stress and food EEG sessions are logged for today.
        </p>
        <Button variant="outline" onClick={() => navigate("/research")}>Back to Study Hub</Button>
      </div>
    );
  }

  // ── Not enrolled ───────────────────────────────────────────────────────────
  if (status && !status.enrolled) {
    return (
      <div className="max-w-lg mx-auto py-16 px-4 text-center space-y-4">
        <p className="text-muted-foreground">You're not enrolled in the study.</p>
        <Button onClick={() => navigate("/research/enroll")}>Enroll now</Button>
      </div>
    );
  }

  // ── Step: breathing ────────────────────────────────────────────────────────
  if (step === "breathing") {
    return <BreathingExercise onDone={() => setStep("stress-session")} />;
  }

  // ── Step: stress session ───────────────────────────────────────────────────
  if (step === "stress-session") {
    return (
      <SessionTimer
        title="Stress EEG Session"
        icon={Brain}
        iconColor="text-violet-400"
        hint="20 minutes · sit comfortably, stay still"
        linkLabel="Open Brain Monitor (EEG recording)"
        linkPath="/brain-monitor"
        onDone={() => setStep("food-session")}
      />
    );
  }

  // ── Step: food session ─────────────────────────────────────────────────────
  if (step === "food-session") {
    return (
      <SessionTimer
        title="Food & Appetite Session"
        icon={Utensils}
        iconColor="text-amber-400"
        hint="20 minutes · think about food, cravings, hunger"
        linkLabel="Open Food & Cravings page"
        linkPath="/food-emotion"
        onDone={handleSubmit}
      />
    );
  }

  // ── Step: intake ───────────────────────────────────────────────────────────
  return (
    <div className="max-w-lg mx-auto py-6 px-4 space-y-5">

      {/* Header */}
      <div className="flex items-center gap-3">
        <Timer className="w-6 h-6 text-violet-400" />
        <div>
          <h1 className="text-xl font-bold">Daytime Research Session</h1>
          <p className="text-xs text-muted-foreground">
            20-min stress session + 20-min food session · starts after this check-in
          </p>
        </div>
      </div>

      {/* Session overview */}
      <div className="flex gap-2 text-xs">
        <div className="flex-1 rounded-lg border border-violet-500/30 bg-violet-500/5 p-3 text-center">
          <Brain className="w-4 h-4 text-violet-400 mx-auto mb-1" />
          <p className="font-semibold text-violet-300">20 min</p>
          <p className="text-muted-foreground">Stress EEG</p>
        </div>
        <div className="flex items-center text-muted-foreground/40">→</div>
        <div className="flex-1 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-center">
          <Utensils className="w-4 h-4 text-amber-400 mx-auto mb-1" />
          <p className="font-semibold text-amber-300">20 min</p>
          <p className="text-muted-foreground">Food EEG</p>
        </div>
      </div>

      {/* SAM scales */}
      <Card>
        <CardContent className="pt-5 space-y-6">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            How are you feeling right now?
          </p>
          <RatingSlider label="Mood"   value={samValence} onChange={setSamValence} labels={VALENCE_LABELS} />
          <RatingSlider label="Energy" value={samArousal} onChange={setSamArousal} labels={AROUSAL_LABELS} />
          <RatingSlider label="Stress" value={samStress}  onChange={setSamStress}  labels={STRESS_LABELS} />
        </CardContent>
      </Card>

      {/* Breathing notice when stress > 5 */}
      {samStress > 5 && (
        <div className="flex items-start gap-3 rounded-xl border border-sky-500/30 bg-sky-500/5 p-4 text-sm">
          <Wind className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
          <p className="text-muted-foreground">
            Your stress is elevated. We'll start with a short <span className="text-sky-300 font-medium">box breathing</span> exercise before your EEG session.
          </p>
        </div>
      )}

      {/* Caffeine */}
      <Card>
        <CardContent className="pt-5 space-y-3">
          <div className="flex items-center gap-2">
            <Coffee className="w-4 h-4 text-amber-400" />
            <p className="text-sm font-medium">Caffeine servings so far today</p>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setCaffeine(c => Math.max(0, c - 1))}
              className="w-9 h-9 rounded-full border border-border flex items-center justify-center text-lg hover:bg-muted/50 transition-colors"
            >−</button>
            <span className="text-2xl font-bold w-6 text-center">{caffeine}</span>
            <button
              onClick={() => setCaffeine(c => Math.min(8, c + 1))}
              className="w-9 h-9 rounded-full border border-border flex items-center justify-center text-lg hover:bg-muted/50 transition-colors"
            >+</button>
            <span className="text-xs text-muted-foreground ml-1">(coffee, tea, energy drinks…)</span>
          </div>
        </CardContent>
      </Card>

      {/* Significant event */}
      <Card>
        <CardContent className="pt-5 space-y-3">
          <p className="text-sm font-medium">Did anything emotionally significant happen today?</p>
          <div className="flex gap-3">
            {([["Yes", true], ["No", false]] as [string, boolean][]).map(([label, val]) => (
              <button
                key={label}
                onClick={() => setSignificant(val)}
                className={`flex-1 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                  significant === val
                    ? "border-violet-500 bg-violet-500/15 text-violet-400"
                    : "border-border hover:border-muted-foreground/40"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Button
        className="w-full bg-violet-600 hover:bg-violet-700 gap-2"
        onClick={handleIntakeContinue}
        disabled={submitting}
      >
        {submitting ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
        {samStress > 5 ? "Continue to breathing exercise" : "Start 20-min sessions"}
        <ChevronRight className="w-4 h-4" />
      </Button>
    </div>
  );
}
