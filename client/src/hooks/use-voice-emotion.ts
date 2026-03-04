/**
 * useVoiceEmotion — 7-second microphone recording + voice emotion detection.
 *
 * Records audio via MediaRecorder, sends to /voice-watch/analyze,
 * caches the result via /voice-watch/cache for WebSocket EEG fusion.
 *
 * Never throws — errors are surfaced via the `error` return value.
 */
import { useState, useRef, useCallback } from "react";
import { getMLApiUrl } from "@/lib/ml-api";

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
  /** Recording duration in milliseconds. Default: 7000 */
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
  isRecording: boolean;
  isAnalyzing: boolean;
  lastResult: VoiceEmotionResult | null;
  error: string | null;
}

export function useVoiceEmotion(
  options: UseVoiceEmotionOptions = {}
): UseVoiceEmotionReturn {
  const {
    durationMs = 7000,
    userId = "default",
    hr = null,
    hrv = null,
    spo2 = null,
  } = options;

  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [lastResult, setLastResult] = useState<VoiceEmotionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

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
        // Encode to base64
        const bytes = new Uint8Array(arrayBuffer);
        let binary = "";
        for (let i = 0; i < bytes.byteLength; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        const audio_b64 = btoa(binary);

        const baseUrl = getMLApiUrl();

        const res = await fetch(`${baseUrl}/voice-watch/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            audio_b64,
            sample_rate: 48000,
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

        // Cache result for WebSocket EEG fusion (fire-and-forget)
        fetch(`${baseUrl}/voice-watch/cache`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: userId, emotion_result: result }),
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
  }, [isRecording, isAnalyzing, durationMs, userId, hr, hrv, spo2]);

  return { startRecording, isRecording, isAnalyzing, lastResult, error };
}
