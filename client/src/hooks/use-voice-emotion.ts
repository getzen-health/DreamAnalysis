/**
 * useVoiceEmotion — 30-second microphone recording + voice emotion detection.
 *
 * Records audio via MediaRecorder, sends to /voice-watch/analyze,
 * caches the result via /voice-watch/cache for WebSocket EEG fusion.
 *
 * Never throws — errors are surfaced via the `error` return value.
 */
import { useState, useRef, useCallback, useEffect, useContext } from "react";
import { getMLApiUrl } from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";
import { VoiceCacheContext } from "./use-voice-cache";
import { resolveUrl } from "@/lib/queryClient";

export interface VoiceEmotionResult {
  emotion: string;
  valence: number;
  arousal: number;
  confidence: number;
  probabilities: Record<string, number>;
  model_type: string;
  stress_from_watch?: number;
}

export interface UseVoiceEmotionOptions {
  /** Recording duration in milliseconds. Default: 30000 */
  durationMs?: number;
  /** User ID for cache keying. Default: "default" */
  userId?: string;
  /** Apple Watch heart rate (bpm) */
  hr?: number | null;
  /** Apple Watch HRV SDNN (ms) */
  hrv?: number | null;
  /** Apple Watch SpO2 (%) */
  spo2?: number | null;
}

export interface UseVoiceEmotionReturn {
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  isRecording: boolean;
  isAnalyzing: boolean;
  lastResult: VoiceEmotionResult | null;
  error: string | null;
}

export function useVoiceEmotion(
  options: UseVoiceEmotionOptions = {}
): UseVoiceEmotionReturn {
  const {
    durationMs = 30000,
    userId,
    hr = null,
    hrv = null,
    spo2 = null,
  } = options;
  const resolvedUserId = userId ?? getParticipantId();

  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [lastResult, setLastResult] = useState<VoiceEmotionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Update shared cross-page cache when a recording completes.
  // useContext is safe here: falls back to the default (no-op) when
  // VoiceCacheProvider is not mounted (e.g. in tests or story previews).
  const { setVoiceCacheResult } = useContext(VoiceCacheContext);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cleanup on unmount — cancel pending timer, stop any active recording
  useEffect(() => {
    return () => {
      if (stopTimerRef.current !== null) {
        clearTimeout(stopTimerRef.current);
        stopTimerRef.current = null;
      }
      if (recorderRef.current && recorderRef.current.state === "recording") {
        recorderRef.current.stop();
      }
    };
  }, []);

  const startRecording = useCallback(async (): Promise<void> => {
    if (isRecording || isAnalyzing) return;
    setError(null);

    // Request microphone access
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setError("Microphone access denied");
      return;
    }

    chunksRef.current = [];

    // Pick a supported MIME type
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
      // Stop all mic tracks
      stream.getTracks().forEach((t) => t.stop());
      setIsRecording(false);
      setIsAnalyzing(true);

      try {
        const blob = new Blob(chunksRef.current, { type: mimeType || "audio/webm" });
        const arrayBuffer = await blob.arrayBuffer();

        // Decode WebM/Opus → PCM, then re-encode as WAV (soundfile/librosa require WAV)
        const audioCtx = new AudioContext();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        audioCtx.close();
        const wavBuffer = audioBufferToWav(audioBuffer);
        const sampleRate = audioBuffer.sampleRate;

        // Encode WAV bytes to base64
        const bytes = new Uint8Array(wavBuffer);
        let binary = "";
        for (let i = 0; i < bytes.byteLength; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        const audio_b64 = btoa(binary);

        const baseUrl = getMLApiUrl();

        const res = await fetch(`${baseUrl}/api/voice-watch/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            audio_b64,
            sample_rate: sampleRate,
            user_id: resolvedUserId,
            ...(hr != null && { hr }),
            ...(hrv != null && { hrv }),
            ...(spo2 != null && { spo2 }),
          }),
        });

        if (!res.ok) {
          setError(`Voice analysis failed (HTTP ${res.status})`);
          return;
        }

        const result: VoiceEmotionResult = await res.json();
        setLastResult(result);
        // Propagate to shared cross-page cache immediately (no 30-s poll wait)
        setVoiceCacheResult(result);

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
            stress: result.stress_from_watch ?? null,
            confidence: result.confidence,
            modelType: result.model_type,
            features: { probabilities: result.probabilities },
          }),
        }).catch(() => {
          // Silent — storage failure is not user-facing
        });

        // Cache result for WebSocket EEG fusion (fire-and-forget)
        fetch(`${baseUrl}/api/voice-watch/cache`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: resolvedUserId, emotion_result: result }),
        }).catch(() => {
          // Silent — cache failure is not user-facing
        });
      } catch {
        setError("Voice analysis request failed");
      } finally {
        setIsAnalyzing(false);
      }
    };

    recorder.start();
    setIsRecording(true);

    // Auto-stop after durationMs
    stopTimerRef.current = setTimeout(() => {
      if (recorder.state === "recording") {
        recorder.stop();
      }
    }, durationMs);
  }, [isRecording, isAnalyzing, durationMs, resolvedUserId, hr, hrv, spo2]);

  const stopRecording = useCallback(() => {
    if (stopTimerRef.current !== null) {
      clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop();
    }
  }, []);

  return { startRecording, stopRecording, isRecording, isAnalyzing, lastResult, error };
}

/**
 * Encodes an AudioBuffer as a WAV ArrayBuffer.
 * Mixes down to mono by averaging all channels.
 */
function audioBufferToWav(buffer: AudioBuffer): ArrayBuffer {
  const numChannels = 1; // mono
  const sampleRate = buffer.sampleRate;
  const numSamples = buffer.length;
  const bytesPerSample = 2; // 16-bit PCM

  // Mix all channels down to mono
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

  // WAV header
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);           // PCM chunk size
  view.setUint16(20, 1, true);            // PCM format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
  view.setUint16(32, numChannels * bytesPerSample, true);
  view.setUint16(34, 8 * bytesPerSample, true);
  writeString(view, 36, "data");
  view.setUint32(40, dataLength, true);

  // PCM samples — clamp and convert float32 → int16
  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, monoData[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return wavBuffer;
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}
