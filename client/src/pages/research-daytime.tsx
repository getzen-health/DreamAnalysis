import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Brain, CheckCircle2, Loader2, Zap, Coffee, ExternalLink } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

const USER_ID = "default";

// SAM emoji labels
const VALENCE_LABELS: Record<number, string> = { 1:"😞", 3:"😟", 5:"😐", 7:"🙂", 9:"😊" };
const AROUSAL_LABELS: Record<number, string> = { 1:"😴", 3:"😌", 5:"😐", 7:"⚡", 9:"🔥" };
const STRESS_LABELS:  Record<number, string> = { 1:"😌", 3:"🙂", 5:"😐", 7:"😰", 9:"😱" };

function closestLabel(val: number, labels: Record<number, string>) {
  const keys = Object.keys(labels).map(Number);
  const nearest = keys.reduce((a, b) => Math.abs(b - val) < Math.abs(a - val) ? b : a);
  return labels[nearest];
}

function RatingSlider({
  label, value, onChange, labels, color,
}: {
  label: string; value: number; onChange: (v: number) => void;
  labels: Record<number, string>; color: string;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium">{label}</p>
        <span className="text-xl">{closestLabel(value, labels)}</span>
      </div>
      <Slider
        min={1} max={9} step={1}
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        className={color}
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>1</span><span>{value}</span><span>9</span>
      </div>
    </div>
  );
}

export default function ResearchDaytime() {
  const { toast } = useToast();
  const [, navigate] = useLocation();

  const [samValence,  setSamValence]  = useState(5);
  const [samArousal,  setSamArousal]  = useState(5);
  const [samStress,   setSamStress]   = useState(5);
  const [caffeine,    setCaffeine]    = useState(0);
  const [significant, setSignificant] = useState<boolean | null>(null);
  const [submitting,  setSubmitting]  = useState(false);
  const [done,        setDone]        = useState(false);

  const { data: status } = useQuery({
    queryKey: ["/api/study/status", USER_ID],
    queryFn: async () => {
      const res = await fetch(`/api/study/status/${USER_ID}`, { credentials: "include" });
      if (!res.ok) throw new Error("Failed");
      return res.json() as Promise<{ enrolled: boolean; todaySession: { daytimeCompleted: boolean } | null }>;
    },
  });

  const alreadyDone = status?.todaySession?.daytimeCompleted === true;

  const handleSubmit = async () => {
    if (significant === null) {
      toast({ title: "One more thing", description: "Did anything significant happen today?", variant: "destructive" });
      return;
    }
    setSubmitting(true);
    try {
      await apiRequest("POST", "/api/study/daytime", {
        userId: USER_ID,
        samValence, samArousal, samStress,
        caffeineServings: caffeine,
        significantEventYN: significant,
      });
      setDone(true);
    } catch {
      toast({ title: "Submit failed", description: "Try again in a moment.", variant: "destructive" });
    } finally {
      setSubmitting(false);
    }
  };

  // ── Already done ────────────────────────────────────────────────────────────
  if (alreadyDone || done) {
    return (
      <div className="max-w-lg mx-auto py-16 px-4 text-center space-y-4">
        <CheckCircle2 className="w-12 h-12 text-green-400 mx-auto" />
        <h2 className="text-xl font-bold">Daytime check-in done</h2>
        <p className="text-sm text-muted-foreground">See you this evening.</p>
        <Button variant="outline" onClick={() => navigate("/research")}>Back to Study Hub</Button>
      </div>
    );
  }

  // ── Not enrolled ────────────────────────────────────────────────────────────
  if (status && !status.enrolled) {
    return (
      <div className="max-w-lg mx-auto py-16 px-4 text-center space-y-4">
        <p className="text-muted-foreground">You're not enrolled in the study.</p>
        <Button onClick={() => navigate("/research/enroll")}>Enroll now</Button>
      </div>
    );
  }

  // ── Form ────────────────────────────────────────────────────────────────────
  return (
    <div className="max-w-lg mx-auto py-6 px-4 space-y-5">

      {/* Header */}
      <div className="flex items-center gap-3">
        <Brain className="w-6 h-6 text-violet-400" />
        <div>
          <h1 className="text-xl font-bold">Daytime Check-in</h1>
          <p className="text-xs text-muted-foreground">9 AM – 1 PM · takes ~3 min</p>
        </div>
      </div>

      {/* EEG nudge */}
      <button
        onClick={() => navigate("/brain-monitor")}
        className="w-full flex items-center gap-3 rounded-xl border border-violet-500/30 bg-violet-500/5 hover:bg-violet-500/10 p-4 transition-colors text-left"
      >
        <Zap className="w-5 h-5 text-violet-400 shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium">Run a 10-min EEG session first</p>
          <p className="text-xs text-muted-foreground">Put on your Muse 2, record, then come back here</p>
        </div>
        <ExternalLink className="w-4 h-4 text-muted-foreground shrink-0" />
      </button>

      {/* SAM scales */}
      <Card>
        <CardContent className="pt-5 space-y-6">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            How are you feeling right now?
          </p>
          <RatingSlider label="Mood"    value={samValence} onChange={setSamValence} labels={VALENCE_LABELS} color="" />
          <RatingSlider label="Energy"  value={samArousal} onChange={setSamArousal} labels={AROUSAL_LABELS} color="" />
          <RatingSlider label="Stress"  value={samStress}  onChange={setSamStress}  labels={STRESS_LABELS}  color="" />
        </CardContent>
      </Card>

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
            {([["yes", true], ["no", false]] as [string, boolean][]).map(([label, val]) => (
              <button
                key={label}
                onClick={() => setSignificant(val)}
                className={`flex-1 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                  significant === val
                    ? "border-violet-500 bg-violet-500/15 text-violet-400"
                    : "border-border hover:border-muted-foreground/40"
                }`}
              >
                {label.charAt(0).toUpperCase() + label.slice(1)}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Button
        className="w-full bg-violet-600 hover:bg-violet-700"
        onClick={handleSubmit}
        disabled={submitting}
      >
        {submitting ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : null}
        Submit daytime check-in
      </Button>
    </div>
  );
}
