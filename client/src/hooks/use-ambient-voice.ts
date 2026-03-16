/**
 * useAmbientVoice — passive VAD-driven voice emotion detection.
 *
 * How it works:
 *   1. Opens mic via getUserMedia (audio-only).
 *   2. Creates AudioContext + AnalyserNode — monitors RMS energy every 200 ms.
 *      The AnalyserNode does NOT record; it only reads live audio levels.
 *   3. When RMS > ENERGY_THRESHOLD for >= SPEECH_MIN_DURATION ms, starts a
 *      MediaRecorder capture.
 *   4. When RMS drops below threshold for >= SILENCE_TIMEOUT ms, stops the
 *      recorder and sends the accumulated blob to /api/voice-watch/analyze.
 *   5. A COOLDOWN timer prevents hammering the backend between analyses.
 *   6. Recording is also capped at MAX_CHUNK_DURATION ms.
 *
 * Exposed:
 *   start()            — request mic, begin monitoring
 *   stop()             — release mic, stop everything
 *   isListening        — true while in monitoring mode
 *   isSpeechDetected   — true while above the energy threshold
 *   isAnalyzing        — true while a chunk is in-flight to the backend
 *   currentEmotion     — latest emotion string, or null
 *   energyLevel        — 0–1 normalised RMS for UI meter
 *   speechCount        — number of speech segments analysed so far
 *   results            — array of AmbientVoiceResult (newest last)
 *   error              — last error string, or null
 */

import { useState, useRef, useCallback, useEffect } from "react";
import { getMLApiUrl } from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";
import { resolveUrl } from "@/lib/queryClient";

// ── tuning parameters ────────────────────────────────────────────────────────

const ENERGY_THRESHOLD = 0.02;        // RMS below this = silence
const SPEECH_MIN_DURATION = 1000;     // ms above threshold before recording starts
const SILENCE_TIMEOUT = 2000;         // ms below threshold before recording stops
const COOLDOWN = 10000;               // ms between analyses (battery saving)
const MAX_CHUNK_DURATION = 30000;     // ms — hard cap on a single recording
const POLL_INTERVAL = 200;            // ms between RMS samples

// ── types ────────────────────────────────────────────────────────────────────

export interface AmbientVoiceResult {
  timestamp: number;
  emotion: string;
  valence: number;
  arousal: number;
  confidence: number;
  probabilities: Record<string, number>;
  model_type: string;
  stress_index?: number;
  focus_index?: number;
}

export interface UseAmbientVoiceReturn {
  start: () => Promise<void>;
  stop: () => void;
  isListening: boolean;
  isSpeechDetected: boolean;
  isAnalyzing: boolean;
  currentEmotion: string | null;
  energyLevel: number;
  speechCount: number;
  results: AmbientVoiceResult[];
  error: string | null;
}

// ── WAV encoding (reuses pattern from use-voice-session.ts) ──────────────────

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

async function blobToBase64Wav(blob: Blob): Promise<{ b64: string; sampleRate: number }> {
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
  return { b64: btoa(binary), sampleRate: audioBuffer.sampleRate };
}

// ── hook ─────────────────────────────────────────────────────────────────────

export function useAmbientVoice(): UseAmbientVoiceReturn {
  const resolvedUserId = getParticipantId();

  // ── React state (drives UI) ───────────────────────────────────────────────
  const [isListening, setIsListening] = useState(false);
  const [isSpeechDetected, setIsSpeechDetected] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState<string | null>(null);
  const [energyLevel, setEnergyLevel] = useState(0);
  const [speechCount, setSpeechCount] = useState(0);
  const [results, setResults] = useState<AmbientVoiceResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  // ── Internal refs (not React-reactive) ───────────────────────────────────
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Recording state
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const mimeTypeRef = useRef<string>("");
  const isRecordingRef = useRef(false);

  // VAD timing refs
  const speechStartRef = useRef<number | null>(null);   // when continuous speech began
  const silenceStartRef = useRef<number | null>(null);  // when silence began
  const recordStartRef = useRef<number | null>(null);   // when MediaRecorder started
  const maxChunkTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cooldown ref — after an analysis, block new recordings for COOLDOWN ms
  const cooldownUntilRef = useRef<number>(0);

  // Keep isAnalyzing in a ref to avoid stale closure in the poll callback
  const isAnalyzingRef = useRef(false);

  // ── Cleanup helper ────────────────────────────────────────────────────────
  const teardown = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    if (maxChunkTimerRef.current) {
      clearTimeout(maxChunkTimerRef.current);
      maxChunkTimerRef.current = null;
    }
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
    recorderRef.current = null;
    isRecordingRef.current = false;

    sourceRef.current?.disconnect();
    sourceRef.current = null;
    analyserRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;

    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    speechStartRef.current = null;
    silenceStartRef.current = null;
    recordStartRef.current = null;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      teardown();
    };
  }, [teardown]);

  // ── Send a completed speech chunk to the backend ─────────────────────────
  const sendChunk = useCallback(
    async (blob: Blob) => {
      if (blob.size === 0) return;
      isAnalyzingRef.current = true;
      setIsAnalyzing(true);

      try {
        const { b64, sampleRate } = await blobToBase64Wav(blob);
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

        const result = await res.json();

        setCurrentEmotion(result.emotion);
        setSpeechCount((prev) => prev + 1);
        setResults((prev) => [
          ...prev,
          {
            timestamp: Date.now(),
            emotion: result.emotion,
            valence: result.valence,
            arousal: result.arousal,
            confidence: result.confidence,
            probabilities: result.probabilities,
            model_type: result.model_type,
            stress_index: result.stress_index,
            focus_index: result.focus_index,
          },
        ]);
        setError(null);

        // Persist to user_readings for model retraining (fire-and-forget)
        fetch(resolveUrl("/api/readings"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            userId: resolvedUserId,
            source: "voice_ambient",
            emotion: result.emotion,
            valence: result.valence,
            arousal: result.arousal,
            stress: result.stress_index ?? null,
            confidence: result.confidence,
            modelType: result.model_type,
            features: {
              probabilities: result.probabilities,
              focus_index: result.focus_index,
            },
          }),
        }).catch(() => {});
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        if (/fetch|network|ECONNREFUSED/i.test(msg)) {
          setError("Cannot reach ML backend — check your connection");
        } else {
          setError(`Voice analysis failed: ${msg}`);
        }
      } finally {
        isAnalyzingRef.current = false;
        setIsAnalyzing(false);
        // Start cooldown after analysis completes
        cooldownUntilRef.current = Date.now() + COOLDOWN;
      }
    },
    [resolvedUserId],
  );

  // ── Start a MediaRecorder capture ─────────────────────────────────────────
  const startRecording = useCallback(() => {
    if (isRecordingRef.current || !streamRef.current) return;

    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/webm")
      ? "audio/webm"
      : "";
    mimeTypeRef.current = mimeType;

    const recorder = new MediaRecorder(
      streamRef.current,
      mimeType ? { mimeType } : {},
    );
    recorderRef.current = recorder;
    recordingChunksRef.current = [];
    recordStartRef.current = Date.now();
    isRecordingRef.current = true;

    recorder.ondataavailable = (e: BlobEvent) => {
      if (e.data.size > 0) recordingChunksRef.current.push(e.data);
    };

    recorder.onstop = () => {
      isRecordingRef.current = false;
      const blob = new Blob(recordingChunksRef.current, {
        type: mimeTypeRef.current || "audio/webm",
      });
      recordingChunksRef.current = [];
      sendChunk(blob);
    };

    recorder.start();

    // Hard cap — stop recording after MAX_CHUNK_DURATION
    maxChunkTimerRef.current = setTimeout(() => {
      if (recorderRef.current?.state === "recording") {
        recorderRef.current.stop();
      }
      maxChunkTimerRef.current = null;
    }, MAX_CHUNK_DURATION);
  }, [sendChunk]);

  // ── Stop a MediaRecorder capture ──────────────────────────────────────────
  const stopRecording = useCallback(() => {
    if (maxChunkTimerRef.current) {
      clearTimeout(maxChunkTimerRef.current);
      maxChunkTimerRef.current = null;
    }
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
  }, []);

  // ── Poll the AnalyserNode for RMS energy ──────────────────────────────────
  const startPolling = useCallback(() => {
    if (pollTimerRef.current) return;

    pollTimerRef.current = setInterval(() => {
      const analyser = analyserRef.current;
      if (!analyser) return;

      // Read time-domain data and compute RMS
      const bufLen = analyser.frequencyBinCount;
      const data = new Float32Array(bufLen);
      analyser.getFloatTimeDomainData(data);

      let sumSq = 0;
      for (let i = 0; i < data.length; i++) {
        sumSq += data[i] * data[i];
      }
      const rms = Math.sqrt(sumSq / data.length);

      // Clamp to 0–1 for the UI meter (RMS rarely exceeds 0.5 in normal speech)
      const level = Math.min(1, rms / 0.5);
      setEnergyLevel(level);

      const now = Date.now();
      const inCooldown = now < cooldownUntilRef.current;
      const analyzing = isAnalyzingRef.current;

      if (rms >= ENERGY_THRESHOLD) {
        // ── Speech energy detected ──────────────────────────────────────────
        silenceStartRef.current = null;

        if (!speechStartRef.current) {
          speechStartRef.current = now;
        }

        const speechDuration = now - speechStartRef.current;

        setIsSpeechDetected(true);

        // Start recording once speech has been sustained long enough
        if (
          speechDuration >= SPEECH_MIN_DURATION &&
          !isRecordingRef.current &&
          !inCooldown &&
          !analyzing
        ) {
          startRecording();
        }
      } else {
        // ── Silence ──────────────────────────────────────────────────────────
        speechStartRef.current = null;
        setIsSpeechDetected(false);

        if (isRecordingRef.current) {
          if (!silenceStartRef.current) {
            silenceStartRef.current = now;
          }
          const silenceDuration = now - silenceStartRef.current;
          if (silenceDuration >= SILENCE_TIMEOUT) {
            silenceStartRef.current = null;
            stopRecording();
          }
        } else {
          silenceStartRef.current = null;
        }
      }
    }, POLL_INTERVAL);
  }, [startRecording, stopRecording]);

  // ── Public: start ambient monitoring ─────────────────────────────────────
  const start = useCallback(async () => {
    if (isListening) return;
    setError(null);
    setResults([]);
    setCurrentEmotion(null);
    setEnergyLevel(0);
    setSpeechCount(0);
    setIsSpeechDetected(false);
    cooldownUntilRef.current = 0;

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setError("Microphone access denied");
      return;
    }

    streamRef.current = stream;

    // AudioContext + AnalyserNode for level monitoring (NOT for recording)
    const audioCtx = new AudioContext();
    audioCtxRef.current = audioCtx;

    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.3;
    analyserRef.current = analyser;

    const source = audioCtx.createMediaStreamSource(stream);
    source.connect(analyser);
    // Deliberately do NOT connect analyser to audioCtx.destination — no playback
    sourceRef.current = source;

    speechStartRef.current = null;
    silenceStartRef.current = null;
    isRecordingRef.current = false;
    isAnalyzingRef.current = false;

    setIsListening(true);
    startPolling();
  }, [isListening, startPolling]);

  // ── Public: stop ambient monitoring ───────────────────────────────────────
  const stop = useCallback(() => {
    teardown();
    setIsListening(false);
    setIsSpeechDetected(false);
    setEnergyLevel(0);
  }, [teardown]);

  return {
    start,
    stop,
    isListening,
    isSpeechDetected,
    isAnalyzing,
    currentEmotion,
    energyLevel,
    speechCount,
    results,
    error,
  };
}
