/**
 * useVoiceSession — continuous voice emotion monitoring.
 *
 * Uses MediaRecorder in timeslice mode: every `chunkMs` milliseconds (default
 * 15 000 ms) a `dataavailable` event fires with the buffered audio. Each chunk
 * is decoded to WAV and sent to /api/voice-watch/analyze.  Results accumulate
 * in `results[]` and EMA smoothing is applied to valence and arousal.
 *
 * Exposes:
 *   startSession()   — request mic, start recording
 *   stopSession()    — stop recording, mic released
 *   isActive         — true while recording
 *   isAnalyzing      — true while a chunk is being sent
 *   results[]        — array of VoiceSessionEntry (newest last)
 *   currentEmotion   — latest emotion string or null
 *   smoothedValence  — EMA-smoothed valence (-1..1) or null
 *   smoothedArousal  — EMA-smoothed arousal (0..1) or null
 *   duration         — elapsed seconds since session start
 *   error            — last error string or null
 */

import { useState, useRef, useCallback, useEffect } from "react";
import { getMLApiUrl } from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";
import { resolveUrl } from "@/lib/queryClient";

export interface VoiceSessionResult {
  emotion: string;
  valence: number;
  arousal: number;
  confidence: number;
  probabilities: Record<string, number>;
  model_type: string;
  stress_index?: number;
  focus_index?: number;
}

export interface VoiceSessionEntry {
  timestamp: number;   // ms since epoch
  elapsed: number;     // seconds since session start
  result: VoiceSessionResult;
}

export interface UseVoiceSessionOptions {
  /** Interval in ms between audio chunks sent to the backend. Default: 15000. */
  chunkMs?: number;
  /** EMA alpha for smoothing valence/arousal. Range 0-1. Default: 0.3. */
  emaAlpha?: number;
  userId?: string;
}

export interface UseVoiceSessionReturn {
  startSession: () => Promise<void>;
  stopSession: () => void;
  isActive: boolean;
  isAnalyzing: boolean;
  results: VoiceSessionEntry[];
  currentEmotion: string | null;
  smoothedValence: number | null;
  smoothedArousal: number | null;
  duration: number;
  error: string | null;
}

// ── WAV encoding (mirrors use-voice-emotion.ts) ─────────────────────────────

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

  const ws = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };

  ws(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  ws(8, "WAVE");
  ws(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
  view.setUint16(32, numChannels * bytesPerSample, true);
  view.setUint16(34, 8 * bytesPerSample, true);
  ws(36, "data");
  view.setUint32(40, dataLength, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, monoData[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return wavBuffer;
}

async function blobToBase64Wav(blob: Blob, _mimeType: string): Promise<{ b64: string; sampleRate: number }> {
  const arrayBuffer = await blob.arrayBuffer();

  // On some Android WebViews, decodeAudioData fails on WebM chunks
  // (especially timeslice fragments missing the container header).
  // Try decoding; if it fails, create a minimal silent WAV so the
  // backend can at least detect the failure cleanly rather than crash.
  let audioBuffer: AudioBuffer;
  const audioCtx = new AudioContext();
  try {
    // decodeAudioData needs a COPY — some engines detach the original buffer
    const copy = arrayBuffer.slice(0);
    audioBuffer = await audioCtx.decodeAudioData(copy);
  } catch {
    // Retry with OfflineAudioContext for broader codec support
    try {
      const copy2 = arrayBuffer.slice(0);
      const offCtx = new OfflineAudioContext(1, 1, 22050);
      audioBuffer = await offCtx.decodeAudioData(copy2);
    } catch {
      audioCtx.close();
      throw new Error("Your browser cannot decode this audio format. Try Chrome or a different device.");
    }
  }
  audioCtx.close();

  const wavBuffer = audioBufferToWav(audioBuffer);
  const bytes = new Uint8Array(wavBuffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return { b64: btoa(binary), sampleRate: audioBuffer.sampleRate };
}

// ── hook ────────────────────────────────────────────────────────────────────

export function useVoiceSession(options: UseVoiceSessionOptions = {}): UseVoiceSessionReturn {
  const { chunkMs = 15000, emaAlpha = 0.3, userId } = options;
  const resolvedUserId = userId ?? getParticipantId();

  const [isActive, setIsActive] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<VoiceSessionEntry[]>([]);
  const [currentEmotion, setCurrentEmotion] = useState<string | null>(null);
  const [smoothedValence, setSmoothedValence] = useState<number | null>(null);
  const [smoothedArousal, setSmoothedArousal] = useState<number | null>(null);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const startTimeRef = useRef<number>(0);
  const durationTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const chunkTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mimeTypeRef = useRef<string>("");

  // EMA state stored in refs to avoid stale closures inside ondataavailable
  const emaValenceRef = useRef<number | null>(null);
  const emaArousalRef = useRef<number | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (durationTimerRef.current) clearInterval(durationTimerRef.current);
      if (chunkTimerRef.current) clearInterval(chunkTimerRef.current);
      if (recorderRef.current?.state === "recording") recorderRef.current.stop();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const sendChunk = useCallback(async (blob: Blob) => {
    if (blob.size === 0) return;
    setIsAnalyzing(true);
    try {
      const { b64, sampleRate } = await blobToBase64Wav(blob, mimeTypeRef.current);
      const baseUrl = getMLApiUrl();
      const res = await fetch(`${baseUrl}/api/voice-watch/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio_b64: b64,
          sample_rate: sampleRate,
          user_id: resolvedUserId,
        }),
      });

      if (!res.ok) {
        setError(`Analysis failed (HTTP ${res.status})`);
        return;
      }

      const result: VoiceSessionResult = await res.json();
      const elapsed = Math.round((Date.now() - startTimeRef.current) / 1000);

      // EMA smoothing
      const prevV = emaValenceRef.current;
      const prevA = emaArousalRef.current;
      const newV = prevV === null ? result.valence : prevV + emaAlpha * (result.valence - prevV);
      const newA = prevA === null ? result.arousal : prevA + emaAlpha * (result.arousal - prevA);
      emaValenceRef.current = newV;
      emaArousalRef.current = newA;

      setCurrentEmotion(result.emotion);
      setSmoothedValence(newV);
      setSmoothedArousal(newA);
      setResults((prev) => [...prev, { timestamp: Date.now(), elapsed, result }]);
      setError(null);

      // Persist to user_readings for model retraining (fire-and-forget)
      fetch(resolveUrl("/api/readings"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: resolvedUserId,
          source: "voice",
          emotion: result.emotion,
          valence: result.valence,
          arousal: result.arousal,
          stress: result.stress_index ?? null,
          confidence: result.confidence,
          modelType: result.model_type,
          features: {
            probabilities: result.probabilities,
            focus_index: result.focus_index,
            stress_index: result.stress_index,
            elapsed_seconds: elapsed,
          },
        }),
      }).catch(() => {
        // Silent — storage failure is not user-facing
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      if (/fetch|network|ECONNREFUSED/i.test(msg)) {
        setError("Cannot reach ML backend — check your connection");
      } else if (/decode|audio|WAV/i.test(msg)) {
        setError("Audio encoding failed — try again");
      } else {
        setError(`Voice analysis failed: ${msg}`);
      }
    } finally {
      setIsAnalyzing(false);
    }
  }, [resolvedUserId, emaAlpha]);

  const startSession = useCallback(async (): Promise<void> => {
    if (isActive) return;
    setError(null);
    setResults([]);
    setCurrentEmotion(null);
    setSmoothedValence(null);
    setSmoothedArousal(null);
    setDuration(0);
    emaValenceRef.current = null;
    emaArousalRef.current = null;

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setError("Microphone access denied");
      return;
    }

    streamRef.current = stream;

    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/webm")
      ? "audio/webm"
      : "";
    mimeTypeRef.current = mimeType;

    // Helper: create a fresh recorder on the same stream, record for chunkMs,
    // then stop to produce a COMPLETE audio blob (not a timeslice fragment).
    const chunksCollected: Blob[] = [];

    function startOneRecording() {
      if (!streamRef.current || streamRef.current.getTracks().every(t => t.readyState === "ended")) return;
      const rec = new MediaRecorder(streamRef.current!, mimeType ? { mimeType } : {});
      recorderRef.current = rec;
      chunksCollected.length = 0;

      rec.ondataavailable = (e: BlobEvent) => {
        if (e.data.size > 0) chunksCollected.push(e.data);
      };

      rec.onstop = () => {
        if (chunksCollected.length > 0) {
          const completeBlob = new Blob(chunksCollected, { type: mimeType || "audio/webm" });
          sendChunk(completeBlob);
        }
      };

      rec.start();
    }

    startTimeRef.current = Date.now();
    startOneRecording();
    setIsActive(true);

    // Every chunkMs: stop current recording (triggers onstop → sendChunk),
    // then immediately start a new one. Each blob is a complete audio file.
    chunkTimerRef.current = setInterval(() => {
      if (recorderRef.current?.state === "recording") {
        recorderRef.current.stop();
      }
      startOneRecording();
    }, chunkMs);

    durationTimerRef.current = setInterval(() => {
      setDuration(Math.round((Date.now() - startTimeRef.current) / 1000));
    }, 1000);
  }, [isActive, chunkMs, sendChunk]);

  const stopSession = useCallback(() => {
    if (chunkTimerRef.current) {
      clearInterval(chunkTimerRef.current);
      chunkTimerRef.current = null;
    }
    if (durationTimerRef.current) {
      clearInterval(durationTimerRef.current);
      durationTimerRef.current = null;
    }
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setIsActive(false);
  }, []);

  return {
    startSession,
    stopSession,
    isActive,
    isAnalyzing,
    results,
    currentEmotion,
    smoothedValence,
    smoothedArousal,
    duration,
    error,
  };
}
