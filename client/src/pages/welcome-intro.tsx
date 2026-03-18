/**
 * Welcome Intro — 3-step setup wizard shown once after first registration.
 *
 * Step 1: Connect Health Data (Apple Health / Google Health Connect)
 * Step 2: Enable Voice Analyses (mic permission + test recording)
 * Step 3: Do you have any health devices? (Muse 2 / fitness tracker)
 *
 * Sets localStorage flag and redirects to "/" (dashboard) when done.
 */

import { useState, useRef, useEffect } from "react";
import { useLocation } from "wouter";
import {
  Heart,
  Mic,
  Brain,
  Watch,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  XCircle,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

// ── Step 1: Health Connect ──────────────────────────────────────────────────

function HealthStep({
  onComplete,
}: {
  onComplete: (connected: boolean) => void;
}) {
  const [status, setStatus] = useState<
    "idle" | "connecting" | "connected" | "skipped" | "error"
  >("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  async function connectHealth() {
    setStatus("connecting");
    setErrorMsg(null);
    try {
      const { Capacitor } = await import("@capacitor/core");
      const platform = Capacitor.getPlatform();

      if (platform === "ios") {
        const { CapacitorHealthkit } = await import(
          "@perfood/capacitor-healthkit"
        );
        await CapacitorHealthkit.requestAuthorization({
          all: [],
          read: [
            "heartRate",
            "restingHeartRate",
            "respiratoryRate",
            "oxygenSaturation",
            "sleepAnalysis",
            "stepCount",
            "activeEnergyBurned",
          ],
          write: [],
        });
        setStatus("connected");
        onComplete(true);
      } else if (platform === "android") {
        const { Health } = await import("capacitor-health");
        const available = await Health.isHealthAvailable();
        if (!available.available) {
          setErrorMsg(
            "Google Health Connect is not installed. Install it from the Play Store to continue."
          );
          setStatus("error");
          return;
        }
        await Health.requestHealthPermissions({
          permissions: [
            "READ_STEPS",
            "READ_HEART_RATE",
            "READ_ACTIVE_CALORIES",
            "READ_WORKOUTS",
            "READ_MINDFULNESS",
          ],
        });
        setStatus("connected");
        onComplete(true);
      } else {
        // Web — no native health API
        setErrorMsg(
          "Health data sync is available on mobile devices. You can connect later from Settings."
        );
        setStatus("error");
      }
    } catch (e) {
      setErrorMsg(`Permission denied or unavailable: ${String(e)}`);
      setStatus("error");
    }
  }

  return (
    <div className="space-y-6">
      <div className="text-center space-y-6">
        <div className="inline-flex items-center justify-center h-20 w-20 rounded-full bg-rose-500/10 mx-auto">
          <Heart className="h-10 w-10 text-rose-400" />
        </div>
        <div className="space-y-3">
          <h1 className="text-3xl font-semibold">Connect Your Health Data</h1>
          <p className="text-sm text-muted-foreground leading-relaxed max-w-sm mx-auto">
            Sync steps, heart rate, and sleep from Apple Health or Google Health
            Connect to enrich your wellness insights.
          </p>
        </div>
      </div>

      <Card className="p-5 space-y-3">
        <p className="text-xs text-muted-foreground uppercase tracking-wide">
          What we sync
        </p>
        <ul className="grid grid-cols-2 gap-2 text-sm">
          {["Steps", "Heart rate", "Sleep stages", "Active energy"].map(
            (item) => (
              <li key={item} className="flex items-center gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-cyan-400 shrink-0" />
                {item}
              </li>
            )
          )}
        </ul>
      </Card>

      {status === "connected" && (
        <div className="flex items-center justify-center gap-2 text-cyan-400 text-sm font-medium">
          <CheckCircle2 className="h-4 w-4" />
          Connected successfully
        </div>
      )}

      {errorMsg && (
        <p className="text-xs text-muted-foreground text-center">{errorMsg}</p>
      )}

      {status !== "connected" && (
        <Button
          className="w-full"
          onClick={connectHealth}
          disabled={status === "connecting"}
        >
          {status === "connecting" ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Connecting...
            </>
          ) : (
            "Connect Health Data"
          )}
        </Button>
      )}
    </div>
  );
}

// ── Step 2: Voice Analysis ──────────────────────────────────────────────────

function VoiceStep({
  onComplete,
}: {
  onComplete: (tested: boolean) => void;
}) {
  const [status, setStatus] = useState<
    "idle" | "requesting" | "recording" | "done" | "error"
  >("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [seconds, setSeconds] = useState(3);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  async function startTest() {
    setStatus("requesting");
    setErrorMsg(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      setStatus("recording");
      setSeconds(3);

      // 3-second countdown
      let remaining = 3;
      timerRef.current = setInterval(() => {
        remaining -= 1;
        setSeconds(remaining);
        if (remaining <= 0) {
          clearInterval(timerRef.current!);
          // Stop the stream
          stream.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
          setStatus("done");
          onComplete(true);
        }
      }, 1000);
    } catch (e) {
      setErrorMsg(
        "Microphone access denied. You can enable it later in your browser or device settings."
      );
      setStatus("error");
    }
  }

  return (
    <div className="space-y-6">
      <div className="text-center space-y-6">
        <div className="inline-flex items-center justify-center h-20 w-20 rounded-full bg-cyan-400/10 mx-auto">
          <Mic className="h-10 w-10 text-cyan-400" />
        </div>
        <div className="space-y-3">
          <h1 className="text-3xl font-semibold">Enable Voice Analyses</h1>
          <p className="text-sm text-muted-foreground leading-relaxed max-w-sm mx-auto">
            A quick 3-second test to make sure your microphone works. Voice
            analyses detect your mood and stress from how you speak.
          </p>
        </div>
      </div>

      {status === "recording" && (
        <Card className="p-5 text-center space-y-3">
          <div className="relative inline-flex items-center justify-center h-16 w-16 mx-auto">
            <div
              className="absolute inset-0 rounded-full bg-cyan-400/20 animate-ping"
              style={{ animationDuration: "1.5s" }}
            />
            <Mic className="h-7 w-7 text-cyan-400 relative z-10" />
          </div>
          <p className="text-sm font-medium">Listening... {seconds}s</p>
          <p className="text-xs text-muted-foreground">
            Say anything — this is just a microphone test
          </p>
        </Card>
      )}

      {status === "done" && (
        <div className="flex items-center justify-center gap-2 text-cyan-400 text-sm font-medium">
          <CheckCircle2 className="h-4 w-4" />
          Microphone works perfectly
        </div>
      )}

      {errorMsg && (
        <p className="text-xs text-muted-foreground text-center">{errorMsg}</p>
      )}

      {status !== "recording" && status !== "done" && (
        <Button
          className="w-full"
          onClick={startTest}
          disabled={status === "requesting"}
        >
          {status === "requesting" ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Requesting permission...
            </>
          ) : (
            "Test Microphone"
          )}
        </Button>
      )}
    </div>
  );
}

// ── Step 3: Health Devices ──────────────────────────────────────────────────

function DeviceStep() {
  const [selected, setSelected] = useState<Set<string>>(new Set());

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const devices = [
    {
      id: "muse",
      icon: Brain,
      name: "Muse 2 / Muse S",
      desc: "EEG headband for brain monitoring",
    },
    {
      id: "tracker",
      icon: Watch,
      name: "Fitness Tracker / Smartwatch",
      desc: "Steps, heart rate, sleep tracking",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="text-center space-y-6">
        <div className="inline-flex items-center justify-center h-20 w-20 rounded-full bg-violet-400/10 mx-auto">
          <Watch className="h-10 w-10 text-violet-400" />
        </div>
        <div className="space-y-3">
          <h1 className="text-3xl font-semibold">
            Do You Have Any Health Devices?
          </h1>
          <p className="text-sm text-muted-foreground leading-relaxed max-w-sm mx-auto">
            Select any devices you own. This helps us show the right features
            for you.
          </p>
        </div>
      </div>

      <div className="space-y-3">
        {devices.map((device) => {
          const Icon = device.icon;
          const isSelected = selected.has(device.id);
          return (
            <button
              key={device.id}
              onClick={() => toggle(device.id)}
              className={`w-full flex items-center gap-4 p-4 rounded-xl border transition-all text-left ${
                isSelected
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-muted-foreground/30"
              }`}
            >
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${
                  isSelected ? "bg-primary/15" : "bg-muted"
                }`}
              >
                <Icon
                  className={`h-5 w-5 ${
                    isSelected ? "text-primary" : "text-muted-foreground"
                  }`}
                />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium">{device.name}</p>
                <p className="text-xs text-muted-foreground">{device.desc}</p>
              </div>
              {isSelected && (
                <CheckCircle2 className="h-5 w-5 text-primary shrink-0" />
              )}
            </button>
          );
        })}
      </div>

      {selected.size === 0 && (
        <Card className="p-4 border-cyan-400/20 bg-cyan-400/5">
          <p className="text-sm text-center text-cyan-400/90">
            No problem! Voice analyses and manual logging work great on their
            own.
          </p>
        </Card>
      )}

      {selected.has("muse") && (
        <Card className="p-4 border-primary/20">
          <p className="text-xs text-muted-foreground">
            You can connect your Muse headband from the Settings page anytime.
            It unlocks live brain monitoring, emotion detection, and
            neurofeedback.
          </p>
        </Card>
      )}
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────────

const STEPS = [
  { key: "health", label: "Health" },
  { key: "voice", label: "Voice" },
  { key: "devices", label: "Devices" },
] as const;

export default function WelcomeIntro() {
  const [, navigate] = useLocation();
  const [current, setCurrent] = useState(0);
  const [healthDone, setHealthDone] = useState(false);
  const [voiceDone, setVoiceDone] = useState(false);

  function finish() {
    try {
      localStorage.setItem("onboarding_complete", "true");
    } catch {
      /* ignore */
    }
    navigate("/");
  }

  function next() {
    if (current < STEPS.length - 1) setCurrent(current + 1);
    else finish();
  }

  function prev() {
    if (current > 0) setCurrent(current - 1);
  }

  const isLast = current === STEPS.length - 1;

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col items-center justify-center p-6">
      <div className="max-w-md w-full space-y-8">
        {/* Step content */}
        {current === 0 && (
          <HealthStep
            onComplete={(connected) => {
              setHealthDone(connected);
            }}
          />
        )}
        {current === 1 && (
          <VoiceStep
            onComplete={(tested) => {
              setVoiceDone(tested);
            }}
          />
        )}
        {current === 2 && <DeviceStep />}

        {/* Dot indicators */}
        <div className="flex justify-center gap-2">
          {STEPS.map((step, i) => (
            <button
              key={step.key}
              onClick={() => setCurrent(i)}
              className={`h-2 rounded-full transition-all duration-300 ${
                i === current
                  ? "w-6 bg-primary"
                  : "w-2 bg-muted-foreground/30 hover:bg-muted-foreground/50"
              }`}
              aria-label={`Go to ${step.label}`}
            />
          ))}
        </div>

        {/* Navigation buttons */}
        <div className="flex items-center justify-between gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={prev}
            disabled={current === 0}
            className="text-muted-foreground"
          >
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back
          </Button>

          <Button onClick={next} className="px-8">
            {isLast ? "Continue to Dashboard" : "Next"}
            {!isLast && <ChevronRight className="h-4 w-4 ml-1" />}
          </Button>
        </div>

        {/* Skip */}
        <div className="text-center">
          <button
            onClick={finish}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Skip for now
          </button>
        </div>
      </div>
    </div>
  );
}
