/**
 * New onboarding flow — voice + health first, EEG optional.
 *
 * Phases:
 *   welcome → (Voice path) → voice-permission → voice-checkin → voice-result → /
 *           → (EEG path)   → /onboarding
 */

import { useEffect, useRef, useState } from "react";
import { useLocation } from "wouter";
import {
  Brain,
  Mic,
  Heart,
  ChevronRight,
  CheckCircle,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { NeuralBackground } from "@/components/neural-background";
import { submitVoiceCheckin } from "@/lib/ml-api";

const USER_ID = "default";
const RECORD_SECONDS = 10;

type Phase =
  | "welcome"
  | "voice-permission"
  | "voice-checkin"
  | "voice-result"
  | "eeg-redirect";

/* ── PCM Float32 → WAV base64 ──────────────────────────────────────────── */
function pcmToWavB64(pcm: Float32Array, sampleRate: number): string {
  const numSamples = pcm.length;
  const buf = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buf);
  const write = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i));
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, numSamples * 2, true);
  for (let i = 0; i < numSamples; i++) {
    view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 0x7fff, true);
  }
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

/* ── Ring timer ─────────────────────────────────────────────────────────── */
function RingTimer({ secondsLeft, total }: { secondsLeft: number; total: number }) {
  const R = 56;
  const circ = 2 * Math.PI * R;
  const pct = secondsLeft / total;
  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width="140" height="140" className="rotate-[-90deg]">
        <circle cx="70" cy="70" r={R} fill="none" stroke="hsl(var(--muted)/0.2)" strokeWidth="8" />
        <circle
          cx="70" cy="70" r={R} fill="none"
          stroke="hsl(var(--secondary))"
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={circ * (1 - pct)}
          style={{ transition: "stroke-dashoffset 0.5s ease" }}
        />
      </svg>
      <span className="absolute text-3xl font-mono font-bold text-secondary">
        {secondsLeft}
      </span>
    </div>
  );
}

/* ── Emotion emoji helper ───────────────────────────────────────────────── */
function emotionEmoji(emotion: string): string {
  const map: Record<string, string> = {
    happy: "😊",
    sad: "😔",
    angry: "😤",
    fear: "😰",
    surprise: "😲",
    neutral: "😌",
  };
  return map[emotion] ?? "🧠";
}

export default function OnboardingNew() {
  const [, setLocation] = useLocation();
  const [phase, setPhase] = useState<Phase>("welcome");
  const [path, setPath] = useState<"voice" | "eeg" | null>(null);
  const [micError, setMicError] = useState<string | null>(null);
  const [secondsLeft, setSecondsLeft] = useState(RECORD_SECONDS);
  const [recording, setRecording] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<{ emotion: string; valence: number; stress_index: number } | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /* Redirect to /onboarding (EEG path) */
  useEffect(() => {
    if (phase === "eeg-redirect") {
      setLocation("/onboarding");
    }
  }, [phase, setLocation]);

  function choosePath(chosen: "voice" | "eeg") {
    setPath(chosen);
    if (chosen === "eeg") {
      setPhase("eeg-redirect");
    } else {
      setPhase("voice-permission");
    }
  }

  async function requestMic() {
    setMicError(null);
    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
      setPhase("voice-checkin");
    } catch {
      setMicError("Microphone access denied. Please allow access and try again.");
    }
  }

  function startRecording() {
    setMicError(null);
    setSubmitError(null);
    chunksRef.current = [];
    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      const mr = new MediaRecorder(stream);
      mediaRef.current = mr;
      mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        processAudio();
      };
      mr.start();
      setRecording(true);
      setSecondsLeft(RECORD_SECONDS);

      let s = RECORD_SECONDS;
      timerRef.current = setInterval(() => {
        s--;
        setSecondsLeft(s);
        if (s <= 0) {
          clearInterval(timerRef.current!);
          mr.stop();
          setRecording(false);
        }
      }, 1000);
    }).catch(() => {
      setMicError("Could not access microphone.");
    });
  }

  function stopEarly() {
    if (timerRef.current) clearInterval(timerRef.current);
    mediaRef.current?.stop();
    setRecording(false);
  }

  async function processAudio() {
    setSubmitting(true);
    try {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      const arrayBuf = await blob.arrayBuffer();
      const ctx = new AudioContext({ sampleRate: 16000 });
      const decoded = await ctx.decodeAudioData(arrayBuf);
      const pcm = decoded.getChannelData(0);
      const wavB64 = pcmToWavB64(pcm, decoded.sampleRate);
      const res = await submitVoiceCheckin({
        user_id: USER_ID,
        audio_b64: wavB64,
        sample_rate: decoded.sampleRate,
        note: "onboarding check-in",
      });
      setResult({ emotion: res.emotion, valence: res.valence, stress_index: res.stress_index });
      setPhase("voice-result");
    } catch {
      setSubmitError("Could not analyze audio — skipping to dashboard.");
      setPhase("voice-result");
    } finally {
      setSubmitting(false);
    }
  }

  /* ── Screens ─────────────────────────────────────────────────────────── */

  if (phase === "welcome") {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center px-4">
        <NeuralBackground />
        <div className="max-w-lg w-full z-10 text-center space-y-8">
          <div>
            <Badge variant="outline" className="mb-4 border-primary/40 text-primary px-3 py-1">
              Mental Readiness Tracker
            </Badge>
            <h1 className="text-4xl md:text-5xl font-futuristic font-bold text-gradient mb-3">
              Svapnastra
            </h1>
            <p className="text-foreground/60 text-lg">
              Understand your mind. No special hardware needed to start.
            </p>
          </div>

          <div className="grid gap-4">
            <Card
              className="p-6 border-secondary/40 bg-secondary/5 cursor-pointer hover:bg-secondary/10 transition-colors text-left"
              onClick={() => choosePath("voice")}
            >
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-full bg-secondary/20 flex items-center justify-center shrink-0">
                  <Mic className="h-5 w-5 text-secondary" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold">Voice + Health</span>
                    <Badge className="bg-secondary/20 text-secondary border-0 text-xs">Recommended</Badge>
                  </div>
                  <p className="text-sm text-foreground/60">
                    Works right now. 10-second voice check-ins detect your emotional state 3× daily.
                    No hardware required.
                  </p>
                </div>
                <ChevronRight className="h-5 w-5 text-foreground/40 shrink-0 mt-1" />
              </div>
            </Card>

            <Card
              className="p-6 border-muted bg-muted/5 cursor-pointer hover:bg-muted/10 transition-colors text-left"
              onClick={() => choosePath("eeg")}
            >
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                  <Brain className="h-5 w-5 text-primary" />
                </div>
                <div className="flex-1">
                  <span className="font-semibold">I have a Muse 2 headband</span>
                  <p className="text-sm text-foreground/60 mt-1">
                    Full real-time EEG brain monitoring, 16 ML models, and deep emotion analysis.
                    Requires a Muse 2 device.
                  </p>
                </div>
                <ChevronRight className="h-5 w-5 text-foreground/40 shrink-0 mt-1" />
              </div>
            </Card>
          </div>

          <p className="text-xs text-foreground/40">
            You can always add an EEG headband later from Settings.
          </p>
        </div>
      </div>
    );
  }

  if (phase === "voice-permission") {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center px-4">
        <NeuralBackground />
        <div className="max-w-md w-full z-10 text-center space-y-6">
          <div className="w-20 h-20 rounded-full bg-secondary/20 flex items-center justify-center mx-auto">
            <Mic className="h-10 w-10 text-secondary" />
          </div>
          <div>
            <h2 className="text-2xl font-bold mb-2">Allow Microphone</h2>
            <p className="text-foreground/60">
              Voice check-ins use your microphone to detect stress and emotional state from
              vocal patterns. Audio is processed locally — never stored.
            </p>
          </div>
          {micError && (
            <div className="flex items-center gap-2 text-destructive text-sm bg-destructive/10 rounded-lg px-4 py-3">
              <AlertCircle className="h-4 w-4 shrink-0" />
              {micError}
            </div>
          )}
          <div className="flex flex-col gap-3">
            <Button
              size="lg"
              className="bg-gradient-to-r from-primary to-secondary text-primary-foreground"
              onClick={requestMic}
            >
              Allow Microphone Access
              <ChevronRight className="ml-2 h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" className="text-foreground/40" onClick={() => setLocation("/")}>
              Skip for now
            </Button>
          </div>
        </div>
      </div>
    );
  }

  if (phase === "voice-checkin") {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center px-4">
        <NeuralBackground />
        <div className="max-w-md w-full z-10 text-center space-y-6">
          <div>
            <h2 className="text-2xl font-bold mb-2">Your First Check-In</h2>
            <p className="text-foreground/60">
              Say anything — how you feel, what's on your mind, or just read aloud.
              10 seconds is all it takes.
            </p>
          </div>

          <div className="flex items-center justify-center">
            {recording ? (
              <RingTimer secondsLeft={secondsLeft} total={RECORD_SECONDS} />
            ) : submitting ? (
              <div className="w-[140px] h-[140px] flex items-center justify-center">
                <Loader2 className="h-12 w-12 text-secondary animate-spin" />
              </div>
            ) : (
              <div className="w-[140px] h-[140px] flex items-center justify-center">
                <div className="w-24 h-24 rounded-full bg-secondary/20 flex items-center justify-center">
                  <Mic className="h-12 w-12 text-secondary" />
                </div>
              </div>
            )}
          </div>

          {micError && (
            <div className="flex items-center gap-2 text-destructive text-sm bg-destructive/10 rounded-lg px-4 py-3">
              <AlertCircle className="h-4 w-4 shrink-0" />
              {micError}
            </div>
          )}

          {!recording && !submitting && (
            <Button
              size="lg"
              className="bg-gradient-to-r from-primary to-secondary text-primary-foreground w-full"
              onClick={startRecording}
            >
              <Mic className="mr-2 h-5 w-5" />
              Start Recording
            </Button>
          )}

          {recording && (
            <Button variant="outline" size="lg" className="w-full" onClick={stopEarly}>
              Done (stop early)
            </Button>
          )}

          {submitting && (
            <p className="text-sm text-foreground/50">Analyzing your voice...</p>
          )}

          <Button variant="ghost" size="sm" className="text-foreground/40" onClick={() => setLocation("/")}>
            Skip for now
          </Button>
        </div>
      </div>
    );
  }

  if (phase === "voice-result") {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center px-4">
        <NeuralBackground />
        <div className="max-w-md w-full z-10 text-center space-y-6">
          <CheckCircle className="h-16 w-16 text-success mx-auto" />
          <div>
            <h2 className="text-2xl font-bold mb-2">Your First Reading</h2>
            {result ? (
              <div className="space-y-4">
                <div className="text-6xl">{emotionEmoji(result.emotion)}</div>
                <p className="text-xl font-semibold capitalize">{result.emotion}</p>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <Card className="p-3 bg-muted/20">
                    <div className="text-foreground/50 mb-1">Mood</div>
                    <div className="font-semibold">
                      {result.valence > 0.2 ? "Positive" : result.valence < -0.2 ? "Negative" : "Neutral"}
                    </div>
                  </Card>
                  <Card className="p-3 bg-muted/20">
                    <div className="text-foreground/50 mb-1">Stress</div>
                    <div className="font-semibold">
                      {result.stress_index < 0.33 ? "Low" : result.stress_index < 0.66 ? "Moderate" : "High"}
                    </div>
                  </Card>
                </div>
              </div>
            ) : (
              <p className="text-foreground/60">
                {submitError ?? "Analysis unavailable — check back after more check-ins."}
              </p>
            )}
          </div>
          <Card className="p-4 bg-secondary/5 border-secondary/30 text-left">
            <div className="flex items-start gap-3">
              <Heart className="h-5 w-5 text-secondary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground/70">
                Check in 3× daily (morning, afternoon, evening) and you'll see mood patterns
                within a week. No EEG headband required.
              </p>
            </div>
          </Card>
          <Button
            size="lg"
            className="bg-gradient-to-r from-primary to-secondary text-primary-foreground w-full"
            onClick={() => setLocation("/")}
          >
            Go to Dashboard
            <ChevronRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </div>
    );
  }

  /* eeg-redirect — handled by useEffect above */
  return null;
}
