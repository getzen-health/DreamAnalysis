/**
 * Lucid Dreaming Protocol — WBTB + MILD
 *
 * Guides users through the highest-evidence lucid dreaming induction:
 *   Wake Back To Bed (5.5h after sleep onset) + Mnemonic Induction of Lucid Dreams.
 * RCT evidence: Stumbrys et al. 2012, Appel et al. 2020.
 */

import { useState, useEffect, useRef } from "react";
import { useLocation } from "wouter";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Moon,
  AlarmClock,
  Eye,
  BookOpen,
  Sparkles,
  CheckCircle2,
  ChevronRight,
  RotateCcw,
} from "lucide-react";
import { getParticipantId } from "@/lib/participant";
import { RealityTestCard } from "@/components/reality-test-card";
import { RealityTestSettings } from "@/components/reality-test-settings";

const USER_ID = getParticipantId();

// ── Protocol phases ───────────────────────────────────────────────────────────

type Phase =
  | "setup"       // choose sleep onset time, view WBTB alarm
  | "sleeping"    // countdown to WBTB alarm — show minimal UI
  | "wbtb-wake"   // alarm fired — 30-min MILD session
  | "mild"        // step-by-step MILD technique
  | "return"      // return-to-sleep guidance
  | "log";        // log whether lucid dream occurred

const MILD_STEPS = [
  {
    title: "Recall the dream",
    body: "Close your eyes and replay the most recent dream you can remember. Walk through every scene, character, and sensation. Don't evaluate — just remember.",
    duration: 300, // 5 min
  },
  {
    title: "Find the missed cue",
    body: "Scan the dream you just recalled for a moment that was strange, impossible, or inconsistent. That was your missed cue. In a lucid dream you would have noticed it.",
    duration: 180, // 3 min
  },
  {
    title: "Set the intention",
    body: "Repeat slowly in your mind: \"Next time I'm dreaming, I will notice I'm dreaming.\" Feel the words. Imagine your future self recognising the dream state.",
    duration: 300, // 5 min
  },
  {
    title: "Visualise lucidity",
    body: "Return to the dream you recalled. This time, see yourself noticing the strange cue, feeling that electric shift of awareness — you know you're dreaming. Hold that feeling.",
    duration: 300, // 5 min
  },
  {
    title: "Carry the intention into sleep",
    body: "Let your eyes close. Keep the intention gently present as you drift. Don't force it — just let the thought float with you into sleep.",
    duration: 120, // 2 min
  },
];

const TOTAL_MILD_SEC = MILD_STEPS.reduce((s, x) => s + x.duration, 0); // ~20 min

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtMin(sec: number) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function addHours(date: Date, h: number) {
  return new Date(date.getTime() + h * 3600 * 1000);
}

function fmtTime(d: Date) {
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function LucidProtocol() {
  const [, navigate] = useLocation();

  // Setup
  const [sleepOnsetStr, setSleepOnsetStr] = useState("23:00");
  const [phase, setPhase] = useState<Phase>("setup");

  // WBTB countdown
  const [wbtbSecondsLeft, setWbtbSecondsLeft] = useState(0);
  const wbtbIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // MILD session
  const [mildStep, setMildStep] = useState(0);
  const [stepSecondsLeft, setStepSecondsLeft] = useState(MILD_STEPS[0].duration);
  const mildIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Log
  const [hadLucidDream, setHadLucidDream] = useState<boolean | null>(null);
  const [lucidDreamDetails, setLucidDreamDetails] = useState("");

  // Compute WBTB alarm time (sleep onset + 5.5h)
  const wbtbTime = (() => {
    const [hh, mm] = sleepOnsetStr.split(":").map(Number);
    const onset = new Date();
    onset.setHours(hh, mm, 0, 0);
    return addHours(onset, 5.5);
  })();

  // ── WBTB countdown ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (phase !== "sleeping") return;

    const tick = () => {
      const diff = Math.max(0, Math.round((wbtbTime.getTime() - Date.now()) / 1000));
      setWbtbSecondsLeft(diff);
      if (diff === 0) {
        clearInterval(wbtbIntervalRef.current!);
        setPhase("wbtb-wake");
      }
    };

    tick();
    wbtbIntervalRef.current = setInterval(tick, 1000);
    return () => clearInterval(wbtbIntervalRef.current!);
  }, [phase]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── MILD step timer ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (phase !== "mild") return;

    setStepSecondsLeft(MILD_STEPS[mildStep].duration);
    mildIntervalRef.current = setInterval(() => {
      setStepSecondsLeft((prev) => {
        if (prev <= 1) {
          clearInterval(mildIntervalRef.current!);
          const next = mildStep + 1;
          if (next < MILD_STEPS.length) {
            setMildStep(next);
          } else {
            setPhase("return");
          }
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(mildIntervalRef.current!);
  }, [phase, mildStep]);

  // ── Cleanup ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      clearInterval(wbtbIntervalRef.current!);
      clearInterval(mildIntervalRef.current!);
    };
  }, []);

  // ── Render: Setup ───────────────────────────────────────────────────────────

  if (phase === "setup") {
    return (
      <main className="p-4 md:p-6 space-y-6 max-w-2xl mx-auto">
        <div className="flex items-center gap-3">
          <Eye className="h-6 w-6 text-primary" />
          <div>
            <h2 className="text-xl font-semibold">Lucid Dream Protocol</h2>
            <p className="text-xs text-muted-foreground">WBTB + MILD — highest RCT evidence</p>
          </div>
        </div>

        {/* Science card */}
        <Card className="rounded-[14px] bg-card border border-border p-5 space-y-3">
          <div className="flex items-center gap-2">
            <BookOpen className="h-4 w-4 text-muted-foreground" />
            <h3 className="text-sm font-medium">How it works</h3>
          </div>
          <ul className="text-xs text-muted-foreground space-y-2 leading-relaxed">
            <li><span className="text-foreground font-medium">Wake Back To Bed (WBTB)</span> — wake after 5.5h, when REM periods are longest and most vivid.</li>
            <li><span className="text-foreground font-medium">MILD</span> — 20 min of dream recall + intention setting before returning to sleep.</li>
            <li>Combined success rate: ~46% in controlled studies (Stumbrys et al. 2012).</li>
          </ul>
        </Card>

        {/* Sleep onset input */}
        <Card className="rounded-[14px] bg-card border border-border p-5 space-y-4">
          <div className="flex items-center gap-2">
            <AlarmClock className="h-4 w-4 text-muted-foreground" />
            <h3 className="text-sm font-medium">Set your sleep onset time</h3>
          </div>
          <p className="text-xs text-muted-foreground">
            When do you plan to fall asleep? The WBTB alarm fires 5.5 hours later.
          </p>
          <div className="flex items-center gap-3">
            <label className="text-xs text-muted-foreground shrink-0">Sleep onset</label>
            <input
              type="time"
              value={sleepOnsetStr}
              onChange={(e) => setSleepOnsetStr(e.target.value)}
              className="rounded-md border border-border bg-background px-3 py-1.5 text-sm font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
          <div className="rounded-lg bg-primary/5 border border-primary/20 p-3 text-sm">
            <span className="text-muted-foreground">WBTB alarm: </span>
            <span className="font-mono font-bold text-primary">{fmtTime(wbtbTime)}</span>
            <span className="text-muted-foreground text-xs ml-2">5.5 hours after sleep onset</span>
          </div>
        </Card>

        {/* Reality check reminder */}
        <Card className="rounded-[14px] bg-card border border-border p-5 space-y-3">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-muted-foreground" />
            <h3 className="text-sm font-medium">Before you sleep</h3>
          </div>
          <ul className="text-xs text-muted-foreground space-y-1.5 leading-relaxed">
            <li>• Do 5 reality checks right now: count your fingers, read text twice, check if hands look normal.</li>
            <li>• Set the intention: "Tonight I will realise I'm dreaming."</li>
            <li>• Keep a dream journal within reach of your bed.</li>
          </ul>
        </Card>

        {/* Daytime reality testing */}
        <RealityTestCard />

        {/* Reality test notification settings */}
        <RealityTestSettings />

        <div className="flex justify-center pt-2">
          <Button
            onClick={() => setPhase("sleeping")}
            className="bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30 px-10 h-11"
          >
            <Moon className="h-4 w-4 mr-2" />
            Start protocol — go to sleep
          </Button>
        </div>
      </main>
    );
  }

  // ── Render: Sleeping countdown ──────────────────────────────────────────────

  if (phase === "sleeping") {
    const hours = Math.floor(wbtbSecondsLeft / 3600);
    const mins  = Math.floor((wbtbSecondsLeft % 3600) / 60);
    const secs  = wbtbSecondsLeft % 60;
    return (
      <main className="flex flex-col items-center justify-center min-h-[70vh] gap-8 text-center p-6">
        <Moon className="h-10 w-10 text-primary opacity-60" />
        <div>
          <p className="text-xs text-muted-foreground uppercase tracking-widest mb-2">WBTB alarm in</p>
          <p className="text-5xl font-mono font-bold text-primary">
            {String(hours).padStart(2, "0")}:{String(mins).padStart(2, "0")}:{String(secs).padStart(2, "0")}
          </p>
          <p className="text-xs text-muted-foreground mt-2">fires at {fmtTime(wbtbTime)}</p>
        </div>
        <p className="text-xs text-muted-foreground max-w-xs">
          Sleep now. Keep this tab open — the app will alert you at {fmtTime(wbtbTime)}.
        </p>
        <button
          onClick={() => setPhase("setup")}
          className="text-xs text-muted-foreground hover:text-foreground underline"
        >
          Cancel
        </button>
      </main>
    );
  }

  // ── Render: WBTB wake ───────────────────────────────────────────────────────

  if (phase === "wbtb-wake") {
    return (
      <main className="flex flex-col items-center justify-center min-h-[70vh] gap-6 text-center p-6 max-w-md mx-auto">
        <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center">
          <AlarmClock className="h-8 w-8 text-primary" />
        </div>
        <div>
          <h2 className="text-2xl font-semibold">WBTB — time to wake up</h2>
          <p className="text-sm text-muted-foreground mt-2">
            You're in the most REM-rich phase of the night. Stay awake for 20 minutes, then return to sleep with your intention.
          </p>
        </div>
        <ul className="text-left text-sm text-muted-foreground space-y-2 w-full">
          <li className="flex gap-2"><CheckCircle2 className="h-4 w-4 text-primary shrink-0 mt-0.5" />Get up, drink water, use the bathroom</li>
          <li className="flex gap-2"><CheckCircle2 className="h-4 w-4 text-primary shrink-0 mt-0.5" />Keep lights dim — no screens beyond this one</li>
          <li className="flex gap-2"><CheckCircle2 className="h-4 w-4 text-primary shrink-0 mt-0.5" />Do one reality check — count your fingers</li>
        </ul>
        <Button
          onClick={() => { setMildStep(0); setPhase("mild"); }}
          className="bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30 px-10 h-11 w-full"
        >
          Start 20-min MILD session
          <ChevronRight className="h-4 w-4 ml-1" />
        </Button>
      </main>
    );
  }

  // ── Render: MILD steps ──────────────────────────────────────────────────────

  if (phase === "mild") {
    const step = MILD_STEPS[mildStep];
    const progress = (mildStep / MILD_STEPS.length) * 100 + (1 - stepSecondsLeft / step.duration) * (100 / MILD_STEPS.length);
    return (
      <main className="flex flex-col items-center gap-6 p-6 max-w-md mx-auto min-h-[70vh]">
        {/* Progress bar */}
        <div className="w-full h-1 rounded-full bg-muted">
          <div
            className="h-1 rounded-full bg-primary transition-all duration-1000"
            style={{ width: `${Math.min(100, progress)}%` }}
          />
        </div>

        {/* Step indicator */}
        <div className="flex gap-2 items-center">
          {MILD_STEPS.map((_, i) => (
            <div
              key={i}
              className={`h-2 w-2 rounded-full transition-colors ${i === mildStep ? "bg-primary" : i < mildStep ? "bg-primary/40" : "bg-muted"}`}
            />
          ))}
        </div>

        {/* Step content */}
        <Card className="rounded-[14px] bg-card border border-border p-6 space-y-4 w-full flex-1 flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Eye className="h-4 w-4 text-primary" />
              <span className="text-xs text-muted-foreground uppercase tracking-wider">
                Step {mildStep + 1} of {MILD_STEPS.length}
              </span>
            </div>
            <h3 className="text-lg font-semibold">{step.title}</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">{step.body}</p>
          </div>

          <div className="space-y-3">
            <div className="text-center">
              <p className="text-3xl font-mono font-bold text-primary">{fmtMin(stepSecondsLeft)}</p>
              <p className="text-[10px] text-muted-foreground mt-1">remaining in this step</p>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="w-full"
              onClick={() => {
                clearInterval(mildIntervalRef.current!);
                const next = mildStep + 1;
                if (next < MILD_STEPS.length) {
                  setMildStep(next);
                } else {
                  setPhase("return");
                }
              }}
            >
              Next step
              <ChevronRight className="h-3.5 w-3.5 ml-1" />
            </Button>
          </div>
        </Card>
      </main>
    );
  }

  // ── Render: Return to sleep ─────────────────────────────────────────────────

  if (phase === "return") {
    return (
      <main className="flex flex-col items-center justify-center min-h-[70vh] gap-6 text-center p-6 max-w-md mx-auto">
        <Moon className="h-10 w-10 text-primary opacity-70" />
        <div>
          <h2 className="text-xl font-semibold">Return to sleep</h2>
          <p className="text-sm text-muted-foreground mt-2 leading-relaxed">
            MILD session complete. Carry your intention into sleep: <em>"I will notice I'm dreaming."</em>
          </p>
        </div>
        <ul className="text-left text-sm text-muted-foreground space-y-2 w-full">
          <li className="flex gap-2"><Moon className="h-4 w-4 text-primary shrink-0 mt-0.5" />Lie in your usual sleep position</li>
          <li className="flex gap-2"><Moon className="h-4 w-4 text-primary shrink-0 mt-0.5" />Keep the intention gently present — don't force it</li>
          <li className="flex gap-2"><Moon className="h-4 w-4 text-primary shrink-0 mt-0.5" />If you wake from a dream — immediately write it down</li>
        </ul>
        <Button
          onClick={() => setPhase("log")}
          className="bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30 px-10 h-11 w-full mt-2"
        >
          I just woke up — log my dream
        </Button>
        <button
          onClick={() => navigate("/dreams")}
          className="text-xs text-muted-foreground hover:text-foreground underline"
        >
          Go to dream journal instead
        </button>
      </main>
    );
  }

  // ── Render: Log ─────────────────────────────────────────────────────────────

  if (phase === "log") {
    const done = hadLucidDream !== null;
    return (
      <main className="flex flex-col items-center gap-6 p-6 max-w-md mx-auto">
        <div className="flex items-center gap-3 self-start">
          <BookOpen className="h-5 w-5 text-primary" />
          <h2 className="text-lg font-semibold">Log this session</h2>
        </div>

        <Card className="rounded-[14px] bg-card border border-border p-5 space-y-4 w-full">
          <p className="text-sm font-medium">Did you have a lucid dream?</p>
          <div className="flex gap-3">
            {[true, false].map((v) => (
              <button
                key={String(v)}
                onClick={() => setHadLucidDream(v)}
                className={`flex-1 py-2.5 rounded-lg border text-sm font-medium transition-colors ${
                  hadLucidDream === v
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-border text-muted-foreground hover:border-muted-foreground"
                }`}
              >
                {v ? "Yes — I was lucid!" : "Not this time"}
              </button>
            ))}
          </div>

          {hadLucidDream === true && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">Describe your lucid dream (optional)</p>
              <textarea
                value={lucidDreamDetails}
                onChange={(e) => setLucidDreamDetails(e.target.value)}
                placeholder="What did you do when you realised you were dreaming?"
                className="w-full min-h-[80px] resize-none rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
          )}

          {hadLucidDream === false && (
            <p className="text-xs text-muted-foreground bg-muted/30 rounded p-2 leading-relaxed">
              It typically takes 3–7 attempts. Each practice session strengthens the mental pathway. Keep going.
            </p>
          )}
        </Card>

        {done && (
          <div className="flex flex-col gap-3 w-full">
            <Button
              onClick={() => navigate("/dreams")}
              className="bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30 h-11 w-full"
            >
              <BookOpen className="h-4 w-4 mr-2" />
              Write full dream journal entry
            </Button>
            <Button
              variant="outline"
              onClick={() => { setPhase("setup"); setHadLucidDream(null); setLucidDreamDetails(""); }}
              className="h-11 w-full"
            >
              <RotateCcw className="h-4 w-4 mr-2" />
              Start another session
            </Button>
          </div>
        )}
      </main>
    );
  }

  return null;
}
