import { getParticipantId } from "@/lib/participant";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Separator } from "@/components/ui/separator";
import {
  Moon,
  CheckCircle2,
  AlertCircle,
  Phone,
  MessageSquare,
  ChevronRight,
  Loader2,
} from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

const USER_ID = getParticipantId();

// ── Helpers ───────────────────────────────────────────────────────────────────

function valenceLabel(v: number) {
  if (v <= 2) return "Very negative";
  if (v <= 3) return "Negative";
  if (v <= 4) return "Slightly negative";
  if (v === 5) return "Neutral";
  if (v <= 6) return "Slightly positive";
  if (v <= 7) return "Positive";
  if (v <= 8) return "Very positive";
  return "Extremely positive";
}

function sleepLabel(v: number) {
  if (v <= 2) return "Very poor";
  if (v <= 3) return "Poor";
  if (v <= 4) return "Below average";
  if (v === 5) return "Average";
  if (v <= 6) return "Above average";
  if (v <= 7) return "Good";
  if (v <= 8) return "Very good";
  return "Excellent";
}

function moodLabel(v: number) {
  if (v <= 2) return "Very low";
  if (v <= 3) return "Low";
  if (v <= 4) return "Below average";
  if (v === 5) return "Okay";
  if (v <= 6) return "Good";
  if (v <= 7) return "Pretty good";
  if (v <= 8) return "Great";
  return "Excellent";
}

// ── Scale component ───────────────────────────────────────────────────────────

function RatingScale({
  value,
  onChange,
  leftLabel,
  rightLabel,
  leftEmoji,
  rightEmoji,
  color = "violet",
}: {
  value: number;
  onChange: (v: number) => void;
  leftLabel: string;
  rightLabel: string;
  leftEmoji: string;
  rightEmoji: string;
  color?: "violet" | "blue" | "green" | "amber";
}) {
  const colorMap = {
    violet: "text-violet-400",
    blue: "text-blue-400",
    green: "text-green-400",
    amber: "text-amber-400",
  };

  return (
    <div className="space-y-3">
      <Slider
        min={1}
        max={9}
        step={1}
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        className="w-full"
      />
      <div className="flex justify-between items-center text-xs">
        <span className="flex items-center gap-1 text-muted-foreground">
          <span>{leftEmoji}</span>
          <span>{leftLabel}</span>
        </span>
        <span className={`font-semibold text-sm ${colorMap[color]}`}>
          {value} — {
            color === "violet" ? valenceLabel(value) :
            color === "blue" ? sleepLabel(value) :
            moodLabel(value)
          }
        </span>
        <span className="flex items-center gap-1 text-muted-foreground">
          <span>{rightLabel}</span>
          <span>{rightEmoji}</span>
        </span>
      </div>
      {/* tick marks */}
      <div className="flex justify-between px-2.5">
        {Array.from({ length: 9 }, (_, i) => (
          <button
            key={i}
            onClick={() => onChange(i + 1)}
            className={`w-5 h-5 rounded-full text-[10px] flex items-center justify-center transition-colors ${
              value === i + 1
                ? "bg-primary text-primary-foreground font-bold"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Mental health resources ───────────────────────────────────────────────────

function WelfareResourcesCard({ onContinue }: { onContinue: () => void }) {
  return (
    <Card className="border-amber-500/40 bg-amber-500/10">
      <CardContent className="pt-5 pb-5 space-y-4">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-amber-400 mt-0.5 shrink-0" />
          <div className="space-y-1">
            <p className="text-sm font-medium">We noticed you're feeling low today</p>
            <p className="text-xs text-muted-foreground leading-relaxed">
              It's okay to have hard days. Your morning entry has been saved.
              Here are some resources if you'd like support:
            </p>
          </div>
        </div>

        <div className="space-y-2">
          {[
            { icon: Phone, label: "988 Suicide & Crisis Lifeline", sub: "Call or text 988 (free, 24/7)" },
            { icon: MessageSquare, label: "Crisis Text Line", sub: "Text HOME to 741741" },
            { icon: Phone, label: "NAMI Helpline", sub: "1-800-950-6264 (Mon–Fri 10 AM–10 PM ET)" },
          ].map(({ icon: Icon, label, sub }) => (
            <div key={label} className="flex items-center gap-3 p-2.5 rounded-lg bg-background/40">
              <Icon className="w-4 h-4 text-amber-400 shrink-0" />
              <div>
                <p className="text-xs font-medium">{label}</p>
                <p className="text-xs text-muted-foreground">{sub}</p>
              </div>
            </div>
          ))}
        </div>

        <p className="text-xs text-muted-foreground">
          You can skip any study day at any time — just come back when you feel ready.
          Your compensation is unaffected.
        </p>

        <Button variant="outline" size="sm" onClick={onContinue} className="w-full gap-1">
          I'm okay — go to my dashboard
          <ChevronRight className="w-3.5 h-3.5" />
        </Button>
      </CardContent>
    </Card>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ResearchMorning() {
  const [, navigate] = useLocation();
  const { toast } = useToast();

  // Study status
  const { data: status, isLoading: statusLoading } = useQuery<{
    enrolled: boolean;
    dayNumber: number;
    todaySession?: { morningCompleted: boolean } | null;
  }>({
    queryKey: ["/api/study/status", USER_ID],
    queryFn: async () => {
      const res = await fetch(`/api/study/status/${USER_ID}`, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to load study status");
      return res.json();
    },
  });

  // Form state
  const [hasRecall, setHasRecall] = useState<"yes" | "no" | "">("");
  const [dreamText, setDreamText] = useState("");
  const [dreamValence, setDreamValence] = useState(5);
  const [nightmareFlag, setNightmareFlag] = useState<"yes" | "no" | "unsure" | "">("");
  const [sleepQuality, setSleepQuality] = useState(5);
  const [sleepHours, setSleepHours] = useState("");
  const [currentMoodRating, setCurrentMoodRating] = useState(5);

  // Submit state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [showWelfareResources, setShowWelfareResources] = useState(false);

  // ── Validation ──────────────────────────────────────────────────────────────
  const canSubmit =
    hasRecall !== "" &&
    (hasRecall === "no" || dreamText.trim().length > 0) &&
    (hasRecall === "no" || nightmareFlag !== "");

  // ── Submit ──────────────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      await apiRequest("POST", "/api/study/morning", {
        userId: USER_ID,
        noRecall: hasRecall === "no",
        dreamText: hasRecall === "yes" ? dreamText.trim() : null,
        dreamValence: hasRecall === "yes" ? dreamValence : null,
        dreamArousal: null,
        nightmareFlag: hasRecall === "yes" ? nightmareFlag : null,
        sleepQuality,
        sleepHours: sleepHours ? parseFloat(sleepHours) : null,
        minutesFromWaking: null,
        currentMoodRating,
      });

      setSubmitted(true);
      if (currentMoodRating <= 2) {
        setShowWelfareResources(true);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Submission failed";
      toast({ title: "Could not save entry", description: msg, variant: "destructive" });
    } finally {
      setIsSubmitting(false);
    }
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Render: loading
  // ─────────────────────────────────────────────────────────────────────────
  if (statusLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!status?.enrolled) {
    return (
      <div className="max-w-md mx-auto py-12 px-4 text-center space-y-4">
        <AlertCircle className="w-10 h-10 text-muted-foreground mx-auto" />
        <p className="text-muted-foreground">You're not enrolled in the study yet.</p>
        <Button onClick={() => navigate("/research/enroll")}>Join the Study</Button>
      </div>
    );
  }

  const dayNumber = status.dayNumber ?? 1;
  const alreadyDone = status.todaySession?.morningCompleted === true;

  // ─────────────────────────────────────────────────────────────────────────
  // Render: already completed today
  // ─────────────────────────────────────────────────────────────────────────
  if (alreadyDone && !submitted) {
    return (
      <div className="max-w-md mx-auto py-12 px-4 space-y-6 text-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-14 h-14 rounded-full bg-green-500/20 flex items-center justify-center">
            <CheckCircle2 className="w-7 h-7 text-green-400" />
          </div>
          <h2 className="text-xl font-semibold">Morning entry done ✓</h2>
          <p className="text-sm text-muted-foreground">
            Day {dayNumber} — you already recorded this morning's entry.
          </p>
        </div>
        <Button variant="outline" onClick={() => navigate("/research")} className="gap-1">
          Back to Dashboard
          <ChevronRight className="w-4 h-4" />
        </Button>
      </div>
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Render: success state
  // ─────────────────────────────────────────────────────────────────────────
  if (submitted) {
    return (
      <div className="max-w-md mx-auto py-12 px-4 space-y-5">
        {showWelfareResources ? (
          <WelfareResourcesCard onContinue={() => navigate("/research")} />
        ) : (
          <div className="text-center space-y-5">
            <div className="flex flex-col items-center gap-3">
              <div className="w-14 h-14 rounded-full bg-green-500/20 flex items-center justify-center">
                <CheckCircle2 className="w-7 h-7 text-green-400" />
              </div>
              <h2 className="text-xl font-semibold">Morning entry saved ✓</h2>
              <p className="text-sm text-muted-foreground">
                Day {dayNumber} — great start. See you this afternoon for your EEG session.
              </p>
            </div>

            <div className="bg-muted/40 rounded-lg p-4 text-sm text-left space-y-1.5">
              <p className="font-medium text-xs uppercase tracking-wide text-muted-foreground mb-2">What's next today</p>
              <div className="flex items-center gap-2 text-muted-foreground">
                <CheckCircle2 className="w-3.5 h-3.5 text-green-400 shrink-0" />
                <span>Morning entry — done</span>
              </div>
              <div className="flex items-center gap-2 text-foreground">
                <div className="w-3.5 h-3.5 rounded-full border-2 border-violet-400 shrink-0" />
                <span>Daytime EEG session — 9 AM–1 PM</span>
              </div>
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="w-3.5 h-3.5 rounded-full border-2 border-border shrink-0" />
                <span>Evening check-in — tonight before bed</span>
              </div>
            </div>

            <Button onClick={() => navigate("/research")} className="w-full gap-1">
              Go to Dashboard
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>
        )}
      </div>
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Render: form
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="max-w-lg mx-auto py-6 px-4 space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Moon className="w-6 h-6 text-blue-400 shrink-0" />
        <div>
          <h1 className="text-xl font-bold">Day {dayNumber} — Morning Entry</h1>
          <p className="text-xs text-muted-foreground">Record this before doing anything else</p>
        </div>
      </div>

      {/* ── Dream recall ─────────────────────────────────────────────────────── */}
      <Card>
        <CardContent className="pt-5 space-y-4">
          <Label className="text-sm font-semibold">Did you remember a dream?</Label>
          <RadioGroup
            value={hasRecall}
            onValueChange={(v) => setHasRecall(v as "yes" | "no")}
            className="flex gap-4"
          >
            <div className="flex items-center gap-2 flex-1 p-3 rounded-lg border border-border hover:bg-muted/30 cursor-pointer transition-colors"
              style={hasRecall === "yes" ? { borderColor: "hsl(var(--primary))", background: "hsl(var(--primary)/0.05)" } : {}}>
              <RadioGroupItem value="yes" id="recall-yes" />
              <Label htmlFor="recall-yes" className="cursor-pointer text-sm font-medium">Yes — I remember</Label>
            </div>
            <div className="flex items-center gap-2 flex-1 p-3 rounded-lg border border-border hover:bg-muted/30 cursor-pointer transition-colors"
              style={hasRecall === "no" ? { borderColor: "hsl(var(--primary))", background: "hsl(var(--primary)/0.05)" } : {}}>
              <RadioGroupItem value="no" id="recall-no" />
              <Label htmlFor="recall-no" className="cursor-pointer text-sm font-medium">No recall</Label>
            </div>
          </RadioGroup>
          {hasRecall === "no" && (
            <p className="text-xs text-muted-foreground bg-muted/30 rounded p-2">
              That's fine — "no recall" entries are valid data. Complete the rest below.
            </p>
          )}
        </CardContent>
      </Card>

      {/* ── Dream content (shown only if recall) ─────────────────────────────── */}
      {hasRecall === "yes" && (
        <Card>
          <CardContent className="pt-5 space-y-5">
            {/* Dream text */}
            <div className="space-y-2">
              <Label htmlFor="dream-text" className="text-sm font-semibold">
                What do you remember?
              </Label>
              <p className="text-xs text-muted-foreground">
                Even a word, a feeling, or a single image. Whatever came first.
              </p>
              <Textarea
                id="dream-text"
                placeholder="I was in a house I didn't recognise, and there was a strange light coming through the window…"
                className={`min-h-[100px] resize-none text-sm leading-relaxed ${
                  dreamText.length > 0 && !dreamText.trim()
                    ? "border-red-500/50 focus-visible:ring-red-500/30"
                    : ""
                }`}
                value={dreamText}
                onChange={(e) => setDreamText(e.target.value)}
              />
              <div className="flex items-center justify-between">
                {!dreamText.trim() ? (
                  <p className="text-xs text-muted-foreground">
                    Write something to continue -- even a single word counts.
                  </p>
                ) : (
                  <span />
                )}
                <p className="text-xs text-muted-foreground">
                  {dreamText.length} characters
                </p>
              </div>
            </div>

            <Separator />

            {/* Dream valence */}
            <div className="space-y-3">
              <div>
                <Label className="text-sm font-semibold">Dream emotional tone</Label>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Overall, how positive or negative did the dream feel?
                </p>
              </div>
              <RatingScale
                value={dreamValence}
                onChange={setDreamValence}
                leftLabel="Very negative"
                rightLabel="Very positive"
                leftEmoji="😞"
                rightEmoji="😊"
                color="violet"
              />
            </div>

            <Separator />

            {/* Nightmare flag */}
            <div className="space-y-3">
              <Label className="text-sm font-semibold">Was it a nightmare?</Label>
              <RadioGroup
                value={nightmareFlag}
                onValueChange={(v) => setNightmareFlag(v as "yes" | "no" | "unsure")}
                className="flex gap-3"
              >
                {[
                  { value: "yes", label: "Yes" },
                  { value: "no", label: "No" },
                  { value: "unsure", label: "Not sure" },
                ].map(({ value, label }) => (
                  <div
                    key={value}
                    className="flex items-center gap-2 flex-1 p-2.5 rounded-lg border border-border hover:bg-muted/30 cursor-pointer transition-colors text-sm"
                    style={nightmareFlag === value ? { borderColor: "hsl(var(--primary))", background: "hsl(var(--primary)/0.05)" } : {}}
                  >
                    <RadioGroupItem value={value} id={`nightmare-${value}`} />
                    <Label htmlFor={`nightmare-${value}`} className="cursor-pointer text-sm">{label}</Label>
                  </div>
                ))}
              </RadioGroup>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Sleep quality (shown once recall is answered) ─────────────────────── */}
      {hasRecall !== "" && (
        <Card>
          <CardContent className="pt-5 space-y-5">
            {/* Sleep quality */}
            <div className="space-y-3">
              <div>
                <Label className="text-sm font-semibold">Sleep quality last night</Label>
                <p className="text-xs text-muted-foreground mt-0.5">
                  How well did you sleep overall?
                </p>
              </div>
              <RatingScale
                value={sleepQuality}
                onChange={setSleepQuality}
                leftLabel="Very poor"
                rightLabel="Excellent"
                leftEmoji="😴"
                rightEmoji="✨"
                color="blue"
              />
            </div>

            <Separator />

            {/* Hours slept */}
            <div className="space-y-2">
              <Label htmlFor="sleep-hours" className="text-sm font-semibold">
                Hours slept (approx)
              </Label>
              <div className="flex items-center gap-3">
                <input
                  id="sleep-hours"
                  type="number"
                  min={1}
                  max={14}
                  step={0.5}
                  placeholder="7.5"
                  className="w-24 h-9 rounded-md border border-input bg-background px-3 text-sm text-center focus:outline-none focus:ring-2 focus:ring-ring"
                  value={sleepHours}
                  onChange={(e) => setSleepHours(e.target.value)}
                />
                <span className="text-sm text-muted-foreground">hours</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Welfare check (subtle, shown last) ───────────────────────────────── */}
      {hasRecall !== "" && (
        <Card className="border-border/50">
          <CardContent className="pt-4 space-y-3">
            <div>
              <Label className="text-sm font-medium text-muted-foreground">
                How are you feeling right now?
              </Label>
              <p className="text-xs text-muted-foreground/70 mt-0.5">
                A quick check — takes 2 seconds
              </p>
            </div>
            <RatingScale
              value={currentMoodRating}
              onChange={setCurrentMoodRating}
              leftLabel="Very low"
              rightLabel="Excellent"
              leftEmoji="😔"
              rightEmoji="😊"
              color="green"
            />
            {currentMoodRating <= 2 && (
              <p className="text-xs text-amber-400 bg-amber-500/10 rounded p-2">
                If you're going through a tough time, support resources will be shown after you submit.
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {/* ── Submit ───────────────────────────────────────────────────────────── */}
      {hasRecall !== "" && (
        <Button
          className="w-full h-11 text-sm font-semibold gap-2 bg-blue-600 hover:bg-blue-700"
          disabled={!canSubmit || isSubmitting}
          onClick={handleSubmit}
        >
          {isSubmitting ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Saving…
            </>
          ) : (
            <>
              <CheckCircle2 className="w-4 h-4" />
              Submit Morning Entry
            </>
          )}
        </Button>
      )}

      {hasRecall === "yes" && !canSubmit && !isSubmitting && (
        <p className="text-xs text-muted-foreground text-center -mt-2">
          {dreamText.trim().length === 0
            ? "Describe what you remember, then submit"
            : "Select nightmare flag above to continue"}
        </p>
      )}
    </div>
  );
}
