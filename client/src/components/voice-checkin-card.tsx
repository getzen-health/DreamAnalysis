/**
 * VoiceCheckinCard — 10-second voice micro check-in.
 *
 * Shows when the current time falls within a check-in window and no check-in
 * has been recorded for that period today.  Windows:
 *   morning  05:00 – 12:00
 *   noon     12:00 – 17:00
 *   evening  17:00 – 23:00
 *
 * Completed check-ins are persisted in localStorage under the key
 * `voice-checkin-{YYYY-MM-DD}-{period}` so the card won't re-appear until
 * the next window.
 */

import { useState, useRef, useEffect, useCallback } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Mic, MicOff, X } from "lucide-react";
import { getMLApiUrl, submitVoiceWatch } from "@/lib/ml-api";
import type { VoiceWatchCheckinResult } from "@/lib/ml-api";

// ─── period helpers ──────────────────────────────────────────────────────────

type Period = "morning" | "noon" | "evening";

function getCurrentPeriod(): Period | null {
  const h = new Date().getHours();
  if (h >= 5 && h < 12) return "morning";
  if (h >= 12 && h < 17) return "noon";
  if (h >= 17 && h < 23) return "evening";
  return null; // outside all windows
}

function nextWindowLabel(period: Period | null): string {
  if (!period) return "5:00 AM";
  if (period === "morning") return "12:00 PM";
  if (period === "noon") return "5:00 PM";
  return "tomorrow morning";
}

function todayKey(period: Period): string {
  const d = new Date();
  const ymd = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  return `voice-checkin-${ymd}-${period}`;
}

function isCheckinDone(period: Period): boolean {
  try {
    return !!localStorage.getItem(todayKey(period));
  } catch {
    return false;
  }
}

function markCheckinDone(period: Period, result: VoiceWatchCheckinResult): void {
  try {
    localStorage.setItem(todayKey(period), JSON.stringify(result));
  } catch {
    // ignore quota errors
  }
}

// ─── WAV encoding (same approach as use-voice-emotion.ts) ────────────────────

function audioBufferToWav(buffer: AudioBuffer): ArrayBuffer {
  const numChannels = 1;
  const sampleRate = buffer.sampleRate;
  const numSamples = buffer.length;
  const bytesPerSample = 2;

  const monoData = new Float32Array(numSamples);
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const channelData = buffer.getChannelData(ch);
    for (let i = 0; i < numSamples; i++) {
      monoData[i] += channelData[i] / buffer.numberOfChannels;
    }
  }

  const dataLength = numSamples * bytesPerSample;
  const wavBuffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(wavBuffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
  view.setUint16(32, numChannels * bytesPerSample, true);
  view.setUint16(34, 8 * bytesPerSample, true);
  writeString(36, "data");
  view.setUint32(40, dataLength, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, monoData[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return wavBuffer;
}

// ─── emotion helpers ─────────────────────────────────────────────────────────

const EMOTION_EMOJI: Record<string, string> = {
  happy: "😊",
  sad: "😢",
  angry: "😠",
  fear: "😨",
  surprise: "😲",
  neutral: "😐",
};

function valenceLabel(v: number): { text: string; className: string } {
  if (v >= 0.4) return { text: "Positive", className: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" };
  if (v <= -0.4) return { text: "Negative", className: "bg-rose-500/20 text-rose-400 border-rose-500/30" };
  return { text: "Neutral", className: "bg-muted/50 text-muted-foreground border-border/40" };
}

// ─── component ───────────────────────────────────────────────────────────────

interface VoiceCheckinCardProps {
  userId?: string;
  onComplete?: (result: VoiceWatchCheckinResult) => void;
}

type CardState = "idle" | "recording" | "analyzing" | "done" | "dismissed";

const RECORD_SEC = 10;

export function VoiceCheckinCard({
  userId = "default",
  onComplete,
}: VoiceCheckinCardProps) {
  const period = getCurrentPeriod();

  const [cardState, setCardState] = useState<CardState>(() => {
    if (!period) return "dismissed";
    if (isCheckinDone(period)) return "dismissed";
    return "idle";
  });

  const [countdown, setCountdown] = useState(RECORD_SEC);
  const [amplitude, setAmplitude] = useState<number[]>(Array(12).fill(0.15));
  const [result, setResult] = useState<VoiceWatchCheckinResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const amplitudeRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stopTimerRef.current) clearTimeout(stopTimerRef.current);
      if (countdownRef.current) clearInterval(countdownRef.current);
      if (amplitudeRef.current) clearInterval(amplitudeRef.current);
      if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    };
  }, []);

  const dismiss = useCallback(() => setCardState("dismissed"), []);

  const startRecording = useCallback(async () => {
    if (cardState !== "idle") return;
    setError(null);
    setCountdown(RECORD_SEC);

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setError("Microphone access denied");
      return;
    }

    // Set up Web Audio analyser for amplitude bars
    try {
      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 32;
      source.connect(analyser);
      analyserRef.current = analyser;

      amplitudeRef.current = setInterval(() => {
        const data = new Uint8Array(analyser.fftSize);
        analyser.getByteTimeDomainData(data);
        const bars = Array.from({ length: 12 }, (_, i) => {
          const idx = Math.floor((i / 12) * data.length);
          return Math.abs((data[idx] - 128) / 128);
        });
        setAmplitude(bars);
      }, 80);
    } catch {
      // Amplitude visualization is best-effort; ignore errors
    }

    chunksRef.current = [];
    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/webm")
      ? "audio/webm"
      : "";

    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
    recorderRef.current = recorder;

    recorder.ondataavailable = (e: BlobEvent) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      if (amplitudeRef.current) clearInterval(amplitudeRef.current);
      if (countdownRef.current) clearInterval(countdownRef.current);
      setAmplitude(Array(12).fill(0.15));
      setCardState("analyzing");

      try {
        const blob = new Blob(chunksRef.current, { type: mimeType || "audio/webm" });
        const arrayBuffer = await blob.arrayBuffer();
        const audioCtx = new AudioContext();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        audioCtx.close();
        const wavBuffer = audioBufferToWav(audioBuffer);

        const bytes = new Uint8Array(wavBuffer);
        let binary = "";
        for (let i = 0; i < bytes.byteLength; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        const audio_b64 = btoa(binary);

        const raw = await submitVoiceWatch(audio_b64, userId);
        // Map voice-watch/analyze response to VoiceWatchCheckinResult shape
        // stress_from_watch and stress_index are both 0-1 scale
        const stressIndex = raw.stress_index ?? raw.stress_from_watch ?? 0.5;
        const focusIndex = raw.focus_index ?? Math.max(0.2, Math.min(0.85, raw.confidence ?? 0.5));
        const checkinResult: VoiceWatchCheckinResult = {
          checkin_id:     `${Date.now()}`,
          checkin_type:   period ?? "morning",
          emotion:        raw.emotion ?? "neutral",
          valence:        raw.valence ?? 0,
          arousal:        raw.arousal ?? 0.5,
          confidence:     raw.confidence ?? 0.5,
          stress_index:   stressIndex,
          focus_index:    focusIndex,
          model_type:     raw.model_type ?? "voice",
          timestamp:      Date.now() / 1000,
          biomarkers:     raw.biomarkers,
        };
        setResult(checkinResult);
        if (period) markCheckinDone(period, checkinResult);
        setCardState("done");
        onComplete?.(checkinResult);

        // Fire-and-forget: record a streak check-in so the StreakCard updates
        fetch(`${getMLApiUrl()}/api/streaks/checkin`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: userId, checkin_type: "voice" }),
        }).catch(() => {}); // ignore errors — streak is best-effort
      } catch (err) {
        setError(err instanceof Error ? err.message : "Check-in failed");
        setCardState("idle");
      }
    };

    recorder.start();
    setCardState("recording");

    // Countdown tick
    countdownRef.current = setInterval(() => {
      setCountdown((c) => {
        if (c <= 1) {
          if (countdownRef.current) clearInterval(countdownRef.current);
          return 0;
        }
        return c - 1;
      });
    }, 1000);

    // Auto-stop after RECORD_SEC seconds
    stopTimerRef.current = setTimeout(() => {
      if (recorder.state === "recording") recorder.stop();
    }, RECORD_SEC * 1000);
  }, [cardState, userId, period, onComplete]);

  // Don't render if dismissed or outside all windows
  if (cardState === "dismissed") return null;
  if (!period) return null;

  const periodLabel = period.charAt(0).toUpperCase() + period.slice(1);

  return (
    <Card className="bg-card rounded-xl border border-border/50 shadow-sm">
      <CardContent className="p-4">
        {/* Header row */}
        <div className="flex items-start justify-between mb-3">
          <div>
            <p className="text-sm font-semibold">{periodLabel} Check-In</p>
            <p className="text-xs text-muted-foreground">How are you feeling?</p>
          </div>
          {cardState !== "recording" && cardState !== "analyzing" && (
            <button
              onClick={dismiss}
              className="text-muted-foreground hover:text-foreground transition-colors p-1 -mt-1 -mr-1"
              aria-label="Skip check-in"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Idle state — big mic button */}
        {cardState === "idle" && (
          <div className="flex flex-col items-center gap-3 py-2">
            <Button
              size="lg"
              variant="outline"
              className="h-16 w-16 rounded-full border-2 border-primary/40 hover:border-primary hover:bg-primary/10 transition-all"
              onClick={startRecording}
            >
              <Mic className="h-7 w-7 text-primary" />
            </Button>
            <p className="text-xs text-muted-foreground text-center">
              Tap to record 10 seconds
            </p>
            {error && (
              <p className="text-xs text-destructive text-center">{error}</p>
            )}
          </div>
        )}

        {/* Recording state — countdown + amplitude bars */}
        {cardState === "recording" && (
          <div className="flex flex-col items-center gap-3 py-2">
            <div className="relative flex items-center justify-center h-16 w-16">
              {/* Pulsing ring */}
              <span className="absolute inset-0 rounded-full border-2 border-primary animate-ping opacity-30" />
              <div className="h-16 w-16 rounded-full bg-primary/10 border-2 border-primary flex items-center justify-center">
                <MicOff className="h-7 w-7 text-primary" />
              </div>
            </div>
            {/* Amplitude bars */}
            <div className="flex items-end gap-0.5 h-8">
              {amplitude.map((v, i) => (
                <div
                  key={i}
                  className="w-1.5 rounded-full bg-primary transition-all duration-75"
                  style={{ height: `${Math.max(4, Math.round(v * 32))}px` }}
                />
              ))}
            </div>
            <p className="text-sm font-mono text-primary tabular-nums">
              {countdown}s
            </p>
          </div>
        )}

        {/* Analyzing state */}
        {cardState === "analyzing" && (
          <div className="flex flex-col items-center gap-2 py-4">
            <div className="h-8 w-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
            <p className="text-xs text-muted-foreground">Analyzing voice…</p>
          </div>
        )}

        {/* Done state — show result */}
        {cardState === "done" && result && (() => {
          const vl = valenceLabel(result.valence);
          const emoji = EMOTION_EMOJI[result.emotion] ?? "🧠";
          return (
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <span className="text-3xl">{emoji}</span>
                <div>
                  <p className="text-sm font-semibold capitalize">{result.emotion}</p>
                  <p className="text-xs text-muted-foreground font-mono">
                    {Math.round(result.confidence * 100)}% confidence
                  </p>
                </div>
                <Badge className={`ml-auto text-xs border ${vl.className}`}>
                  {vl.text}
                </Badge>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                <span>Stress: <span className="font-mono text-foreground/80">{Math.round(result.stress_index * 100)}%</span></span>
                <span>Focus: <span className="font-mono text-foreground/80">{Math.round(result.focus_index * 100)}%</span></span>
              </div>
              <p className="text-xs text-muted-foreground">
                Next check-in at {nextWindowLabel(period)}
              </p>
            </div>
          );
        })()}
      </CardContent>
    </Card>
  );
}
