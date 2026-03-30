import { getParticipantId } from "@/lib/participant";
import { useState, useRef, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
  Mic,
  MicOff,
} from "lucide-react";
import { apiRequest, resolveUrl } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

// ── Speech Recognition types ──────────────────────────────────────────────────

declare global {
  interface Window {
    SpeechRecognition: typeof SpeechRecognition;
    webkitSpeechRecognition: typeof SpeechRecognition;
  }
}

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

// ── Voice recorder hook ───────────────────────────────────────────────────────

const SILENCE_TIMEOUT_MS = 5000;

function useVoiceRecorder(onTranscript: (text: string, isFinal: boolean) => void) {
  const [isListening, setIsListening] = useState(false);
  const [isSupported, setIsSupported] = useState(true);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const interimRef = useRef("");

  useEffect(() => {
    const SpeechRecognitionImpl =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognitionImpl) {
      setIsSupported(false);
    }
    return () => {
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    };
  }, []);

  const stopRecording = useCallback(() => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    setIsListening(false);
    interimRef.current = "";
  }, []);

  const resetSilenceTimer = useCallback(() => {
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    silenceTimerRef.current = setTimeout(() => {
      stopRecording();
    }, SILENCE_TIMEOUT_MS);
  }, [stopRecording]);

  const startRecording = useCallback(() => {
    const SpeechRecognitionImpl =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognitionImpl) return;

    const recognition = new SpeechRecognitionImpl();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      setIsListening(true);
      resetSilenceTimer();
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      resetSilenceTimer();
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          onTranscript(result[0].transcript, true);
        } else {
          interim += result[0].transcript;
        }
      }
      if (interim !== interimRef.current) {
        interimRef.current = interim;
        onTranscript(interim, false);
      }
    };

    recognition.onerror = () => {
      stopRecording();
    };

    recognition.onend = () => {
      setIsListening(false);
      interimRef.current = "";
    };

    recognitionRef.current = recognition;
    recognition.start();
  }, [onTranscript, resetSilenceTimer, stopRecording]);

  const toggle = useCallback(() => {
    if (isListening) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isListening, startRecording, stopRecording]);

  return { isListening, isSupported, toggle, stopRecording };
}

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
    blue: "text-indigo-400",
    green: "text-cyan-400",
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
      const res = await fetch(resolveUrl(`/api/study/status/${USER_ID}`), { credentials: "include" });
      if (!res.ok) throw new Error("Failed to load study status");
      return res.json();
    },
  });

  // Form state
  const [hasRecall, setHasRecall] = useState<"yes" | "no" | "">("");
  const [dreamText, setDreamText] = useState("");
  const [interimText, setInterimText] = useState("");
  const finalTextRef = useRef("");

  const handleVoiceTranscript = useCallback(
    (text: string, isFinal: boolean) => {
      if (isFinal) {
        const updated = finalTextRef.current
          ? finalTextRef.current + " " + text.trim()
          : text.trim();
        finalTextRef.current = updated;
        setDreamText(updated);
        setInterimText("");
      } else {
        setInterimText(text);
      }
    },
    [],
  );

  const { isListening, isSupported, toggle: toggleVoice } = useVoiceRecorder(
    handleVoiceTranscript,
  );

  // Keep finalTextRef in sync when user edits the textarea manually
  useEffect(() => {
    finalTextRef.current = dreamText;
  }, [dreamText]);
  const [dreamValence, setDreamValence] = useState(5);
  const [nightmareFlag, setNightmareFlag] = useState<"yes" | "no" | "unsure" | "">("");
  const [sleepQuality, setSleepQuality] = useState(5);
  const [sleepHours, setSleepHours] = useState("");
  const [currentMoodRating, setCurrentMoodRating] = useState(5);

  // Submit state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [showWelfareResources, setShowWelfareResources] = useState(false);
  const [dreamAnalysis, setDreamAnalysis] = useState<{
    symbols: Array<{ symbol: string; meaning: string }>;
    emotions: string[];
    themes: string[];
    insights: string;
    morningMoodPrediction: string;
  } | null>(null);

  // ── Validation ──────────────────────────────────────────────────────────────
  const canSubmit =
    hasRecall !== "" &&
    (hasRecall === "no" || dreamText.trim().length > 0) &&
    (hasRecall === "no" || nightmareFlag !== "");

  // ── Submit ──────────────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      const res = await apiRequest("POST", "/api/study/morning", {
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

      const data: Record<string, unknown> = await res.json();
      if (data.dreamAnalysis) {
        setDreamAnalysis(data.dreamAnalysis as typeof dreamAnalysis);
      }

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
          <div className="w-14 h-14 rounded-full bg-cyan-600/20 flex items-center justify-center">
            <CheckCircle2 className="w-7 h-7 text-cyan-400" />
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
          <div className="space-y-5">
            <div className="text-center space-y-3">
              <div className="flex flex-col items-center gap-3">
                <div className="w-14 h-14 rounded-full bg-cyan-600/20 flex items-center justify-center">
                  <CheckCircle2 className="w-7 h-7 text-cyan-400" />
                </div>
                <h2 className="text-xl font-semibold">Morning entry saved ✓</h2>
                <p className="text-sm text-muted-foreground">
                  Day {dayNumber} — great start. See you this afternoon for your EEG session.
                </p>
              </div>
            </div>

            {dreamAnalysis && (
              <Card className="border-indigo-500/30 bg-indigo-500/5">
                <CardHeader className="pb-2 pt-4">
                  <CardTitle className="text-sm font-semibold text-indigo-300">Dream Analysis</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 pb-4">
                  {dreamAnalysis.themes.length > 0 && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1">Primary theme</p>
                      <p className="text-sm">{dreamAnalysis.themes[0]}</p>
                    </div>
                  )}

                  {dreamAnalysis.symbols.slice(0, 3).length > 0 && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1.5">Key symbols</p>
                      <div className="space-y-1">
                        {dreamAnalysis.symbols.slice(0, 3).map((s) => (
                          <div key={s.symbol} className="flex gap-2 text-sm">
                            <span className="font-medium text-indigo-300 shrink-0">{s.symbol}</span>
                            <span className="text-muted-foreground">{s.meaning}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {dreamAnalysis.insights && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1">Insight</p>
                      <p className="text-sm text-muted-foreground leading-relaxed">{dreamAnalysis.insights}</p>
                    </div>
                  )}

                  {dreamAnalysis.morningMoodPrediction && (
                    <div className="bg-background/40 rounded-md p-2.5">
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1">Morning mood prediction</p>
                      <p className="text-sm">{dreamAnalysis.morningMoodPrediction}</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            <div className="bg-muted/40 rounded-lg p-4 text-sm text-left space-y-1.5">
              <p className="font-medium text-xs uppercase tracking-wide text-muted-foreground mb-2">What's next today</p>
              <div className="flex items-center gap-2 text-muted-foreground">
                <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                <span>Morning entry — done</span>
              </div>
              <div className="flex items-center gap-2 text-foreground">
                <div className="w-3.5 h-3.5 rounded-full border-2 border-violet-400 shrink-0" />
                <span>Daytime EEG session — 9 AM–1 PM</span>
              </div>
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="w-3.5 h-3.5 rounded-full border-2 border-border shrink-0" />
                <span>Evening analysis — tonight before bed</span>
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
        <Moon className="w-6 h-6 text-indigo-400 shrink-0" />
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

              {/* Voice recording prompt + button */}
              <div className="space-y-2">
                <p className="text-xs text-indigo-400 font-medium">
                  Don't move yet — speak what just happened
                </p>
                {isSupported ? (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={toggleVoice}
                    className={`gap-2 transition-colors ${
                      isListening
                        ? "border-rose-500 text-rose-400 hover:border-rose-400 hover:text-rose-300"
                        : "border-indigo-500/50 text-indigo-400 hover:border-indigo-400 hover:text-indigo-300"
                    }`}
                    aria-label={isListening ? "Stop recording" : "Start voice recording"}
                  >
                    {isListening ? (
                      <>
                        <span className="relative flex h-2.5 w-2.5">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75" />
                          <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-rose-500" />
                        </span>
                        <Mic className="w-3.5 h-3.5" />
                        Listening… speak your dream
                      </>
                    ) : (
                      <>
                        <MicOff className="w-3.5 h-3.5" />
                        Record with voice
                      </>
                    )}
                  </Button>
                ) : (
                  <p className="text-xs text-muted-foreground bg-muted/30 rounded p-2">
                    Voice recording isn't supported in this browser. Type your dream below.
                  </p>
                )}
              </div>

              <Textarea
                id="dream-text"
                aria-label="Dream description — describe what you remember, even a single word counts"
                aria-describedby="dream-text-count"
                placeholder="I was in a house I didn't recognise, and there was a strange light coming through the window…"
                className={`min-h-[100px] resize-none text-sm leading-relaxed ${
                  dreamText.length > 0 && !dreamText.trim()
                    ? "border-rose-500/50 focus-visible:ring-rose-500/30"
                    : ""
                }`}
                value={isListening && interimText ? dreamText + (dreamText ? " " : "") + interimText : dreamText}
                onChange={(e) => {
                  if (!isListening) setDreamText(e.target.value);
                }}
                readOnly={isListening}
              />

              <div className="flex items-center justify-between">
                {!dreamText.trim() ? (
                  <p className="text-xs text-muted-foreground">
                    Write something to continue -- even a single word counts.
                  </p>
                ) : (
                  <span />
                )}
                <p id="dream-text-count" className="text-xs text-muted-foreground" aria-live="polite">
                  {dreamText.length} characters
                </p>
              </div>

              {/* Sequential recall prompts — shown after recording or if text exists */}
              {(dreamText.trim().length > 0 || !isListening) && dreamText.trim().length > 0 && (
                <div className="space-y-1.5">
                  <p className="text-xs text-muted-foreground">Add more detail:</p>
                  <div className="flex flex-wrap gap-2">
                    {[
                      { label: "Who was there?", prefix: "\nWho was there: " },
                      { label: "Where were you?", prefix: "\nWhere: " },
                      { label: "How did you feel?", prefix: "\nHow I felt: " },
                    ].map(({ label, prefix }) => (
                      <button
                        key={label}
                        type="button"
                        onClick={() =>
                          setDreamText((prev) => prev + prefix)
                        }
                        className="text-xs px-3 py-1.5 rounded-full border border-indigo-500/40 text-indigo-300 hover:bg-indigo-500/10 hover:border-indigo-400 transition-colors"
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>
              )}
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
          className="w-full h-11 text-sm font-semibold gap-2 bg-indigo-600 hover:bg-indigo-700"
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
