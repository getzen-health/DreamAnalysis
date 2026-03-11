/**
 * VoiceWatchAnalyzer
 *
 * Headband-free emotion detection using microphone + Apple Watch biometrics.
 * Records 5–10 seconds of audio, encodes to WAV base64, reads latest watch
 * biometrics from health-sync, and calls POST /voice-watch/analyze.
 *
 * Shows: emotion label + confidence bar + audio/watch probability breakdown.
 */

import { useEffect, useRef, useState } from "react";
import {
  analyzeVoiceWatch,
  type VoiceWatchEmotionResult,
  type WatchBiometrics,
} from "@/lib/ml-api";
import { healthSync } from "@/lib/health-sync";

// ── Constants ─────────────────────────────────────────────────────────────────
const RECORD_SECONDS = 30;
const SAMPLE_RATE    = 22050;
const EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral"] as const;

const EMOTION_COLOR: Record<string, string> = {
  happy:    "text-green-400",
  sad:      "text-blue-400",
  angry:    "text-rose-400",
  fear:     "text-amber-400",
  surprise: "text-purple-400",
  neutral:  "text-slate-400",
};

const EMOTION_BG: Record<string, string> = {
  happy:    "bg-green-500",
  sad:      "bg-blue-500",
  angry:    "bg-rose-500",
  fear:     "bg-amber-500",
  surprise: "bg-purple-500",
  neutral:  "bg-slate-500",
};

// ── WAV encoding ──────────────────────────────────────────────────────────────
/** Convert raw Float32 PCM samples to a 16-bit WAV ArrayBuffer. */
function pcmToWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const numSamples  = samples.length;
  const bytesPerSample = 2;
  const blockAlign  = bytesPerSample;
  const byteRate    = sampleRate * blockAlign;
  const dataSize    = numSamples * bytesPerSample;
  const buffer      = new ArrayBuffer(44 + dataSize);
  const view        = new DataView(buffer);

  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);        // PCM chunk size
  view.setUint16(20, 1, true);         // PCM format
  view.setUint16(22, 1, true);         // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);        // bits per sample
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return buffer;
}

/** ArrayBuffer → base64 string. */
function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let binary  = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

// ── Component ─────────────────────────────────────────────────────────────────
interface Props {
  /** Optional user ID forwarded to future /sessions integration. */
  userId?: string;
  /** Called when a result arrives (for parent state updates). */
  onResult?: (result: VoiceWatchEmotionResult) => void;
}

type Phase = "idle" | "recording" | "analyzing" | "done" | "error";

export function VoiceWatchAnalyzer({ userId, onResult }: Props) {
  const [phase,   setPhase]   = useState<Phase>("idle");
  const [result,  setResult]  = useState<VoiceWatchEmotionResult | null>(null);
  const [error,   setError]   = useState<string | null>(null);
  const [seconds, setSeconds] = useState(0);
  const [level,   setLevel]   = useState(0);  // 0–1 audio level for visualizer

  const audioCtxRef  = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const samplesRef   = useRef<Float32Array[]>([]);
  const timerRef     = useRef<ReturnType<typeof setInterval> | null>(null);
  const levelAnimRef = useRef<number>(0);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, []);

  // ── Audio level animation ─────────────────────────────────────────────────
  function animateLevel(analyser: AnalyserNode) {
    const buf = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
      analyser.getByteFrequencyData(buf);
      const avg = buf.reduce((s, v) => s + v, 0) / buf.length;
      setLevel(avg / 128);
      levelAnimRef.current = requestAnimationFrame(tick);
    };
    levelAnimRef.current = requestAnimationFrame(tick);
  }

  // ── Start recording ───────────────────────────────────────────────────────
  async function startRecording() {
    setPhase("recording");
    setError(null);
    setResult(null);
    setSeconds(0);
    samplesRef.current = [];

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setError("Microphone access denied. Allow mic permission and try again.");
      setPhase("error");
      return;
    }

    const ctx       = new AudioContext({ sampleRate: SAMPLE_RATE });
    audioCtxRef.current = ctx;
    const source    = ctx.createMediaStreamSource(stream);
    const analyser  = ctx.createAnalyser();
    analyser.fftSize = 256;
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (e) => {
      const data = e.inputBuffer.getChannelData(0);
      samplesRef.current.push(new Float32Array(data));
    };

    source.connect(analyser);
    source.connect(processor);
    processor.connect(ctx.destination);
    animateLevel(analyser);

    // Count up + auto-stop at RECORD_SECONDS
    timerRef.current = setInterval(() => {
      setSeconds((s) => {
        const next = s + 1;
        if (next >= RECORD_SECONDS) stopAndAnalyze(stream);
        return next;
      });
    }, 1000);
  }

  // ── Stop recording ────────────────────────────────────────────────────────
  function stopRecording() {
    if (timerRef.current)       clearInterval(timerRef.current);
    if (levelAnimRef.current)   cancelAnimationFrame(levelAnimRef.current);
    if (processorRef.current)   processorRef.current.disconnect();
    if (audioCtxRef.current)    audioCtxRef.current.close();
    timerRef.current       = null;
    processorRef.current   = null;
    audioCtxRef.current    = null;
  }

  // ── Stop + send to backend ────────────────────────────────────────────────
  async function stopAndAnalyze(stream?: MediaStream) {
    stopRecording();
    stream?.getTracks().forEach((t) => t.stop());
    setLevel(0);
    setPhase("analyzing");

    // Combine all captured PCM chunks
    const chunks = samplesRef.current;
    if (chunks.length === 0) {
      setError("No audio captured. Try again.");
      setPhase("error");
      return;
    }

    const totalLen  = chunks.reduce((s, c) => s + c.length, 0);
    const combined  = new Float32Array(totalLen);
    let offset      = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    // Encode to WAV base64
    const wavBuf    = pcmToWav(combined, SAMPLE_RATE);
    const audioB64  = arrayBufferToBase64(wavBuf);

    // Read latest watch biometrics from health-sync
    const healthState = healthSync.getState();
    const payload     = healthState.latestPayload;
    const watch: WatchBiometrics = {
      hr:   payload?.current_heart_rate ?? undefined,
      hrv:  payload?.hrv_sdnn           ?? undefined,
      spo2: payload?.spo2               ?? undefined,
    };

    try {
      const res = await analyzeVoiceWatch(audioB64, watch, userId);
      setResult(res);
      setPhase("done");
      onResult?.(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
      setPhase("error");
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────
  const isRecording = phase === "recording";
  const isAnalyzing = phase === "analyzing";

  return (
    <div className="rounded-xl border border-border bg-card p-5 space-y-4 w-full max-w-sm">
      {/* Header */}
      <div className="space-y-0.5">
        <h3 className="font-semibold text-sm">Voice + Watch Emotion</h3>
        <p className="text-xs text-muted-foreground">
          Headband-free · mic + Apple Watch biometrics
        </p>
      </div>

      {/* Level bars (recording only) */}
      {isRecording && (
        <div className="flex items-end gap-0.5 h-8">
          {Array.from({ length: 12 }).map((_, i) => {
            const threshold = (i + 1) / 12;
            const active    = level >= threshold;
            return (
              <div
                key={i}
                className={`flex-1 rounded-sm transition-all duration-75 ${
                  active ? "bg-primary" : "bg-muted"
                }`}
                style={{ height: `${20 + i * 4}%` }}
              />
            );
          })}
        </div>
      )}

      {/* Timer / status */}
      {isRecording && (
        <p className="text-xs text-center text-muted-foreground">
          Recording… {seconds}s / {RECORD_SECONDS}s
        </p>
      )}
      {isAnalyzing && (
        <p className="text-xs text-center text-muted-foreground animate-pulse">
          Analyzing emotion…
        </p>
      )}

      {/* Result */}
      {result && phase === "done" && (
        <div className="space-y-3">
          {/* Dominant emotion */}
          <div className="flex items-center justify-between">
            <span className={`text-lg font-bold capitalize ${EMOTION_COLOR[result.emotion] ?? "text-foreground"}`}>
              {result.emotion}
            </span>
            <span className="text-xs text-muted-foreground">
              {Math.round(result.confidence * 100)}% confident
            </span>
          </div>

          {/* Confidence bar */}
          <div className="h-2 rounded-full bg-muted overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${EMOTION_BG[result.emotion] ?? "bg-primary"}`}
              style={{ width: `${result.confidence * 100}%` }}
            />
          </div>

          {/* Valence / Arousal */}
          <div className="flex gap-4 text-xs text-muted-foreground">
            <span>Valence: <span className="text-foreground font-medium">{result.valence > 0 ? "+" : ""}{result.valence.toFixed(2)}</span></span>
            <span>Arousal: <span className="text-foreground font-medium">{result.arousal.toFixed(2)}</span></span>
          </div>

          {/* Probability breakdown */}
          <div className="space-y-1">
            {EMOTIONS.map((em) => {
              const prob = result.probabilities[em] ?? 0;
              return (
                <div key={em} className="flex items-center gap-1.5 text-xs">
                  <div
                    className={`h-1.5 rounded-full flex-shrink-0 ${EMOTION_BG[em]}`}
                    style={{ width: `${prob * 80}px` }}
                  />
                  <span className="capitalize text-muted-foreground w-16">{em}</span>
                  <span className="ml-auto tabular-nums">
                    {Math.round(prob * 100)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <p className="text-xs text-rose-400">{error}</p>
      )}

      {/* Action buttons */}
      <div className="flex gap-2">
        {!isRecording && !isAnalyzing && (
          <button
            onClick={startRecording}
            className="flex-1 rounded-lg bg-primary text-primary-foreground text-sm py-2 font-medium hover:opacity-90 transition-opacity"
          >
            {phase === "done" ? "Record Again" : "Record Voice"}
          </button>
        )}
        {isRecording && (
          <button
            onClick={() => stopAndAnalyze()}
            className="flex-1 rounded-lg bg-destructive text-destructive-foreground text-sm py-2 font-medium hover:opacity-90 transition-opacity"
          >
            Stop & Analyze
          </button>
        )}
      </div>
    </div>
  );
}
