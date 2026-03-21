/**
 * VoiceCheckinCard — 15-second voice micro check-in.
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
import { useQueryClient } from "@tanstack/react-query";
import { resolveUrl } from "@/lib/queryClient";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Mic, MicOff, X, Check, Pencil, ChevronDown, ChevronUp } from "lucide-react";
import { getMLApiUrl, submitVoiceWatch } from "@/lib/ml-api";
import type { VoiceWatchCheckinResult } from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";
import { runVoiceEmotionONNX } from "@/lib/voice-onnx";
import { writeEmotionToHealth } from "@/lib/health-connect";
import { extractVoiceBiomarkers } from "@/lib/voice-biomarkers";
import type { VoiceBiomarkers } from "@/lib/voice-biomarkers";

// ─── positive affirmations shown during recording ───────────────────────────

const RECORDING_AFFIRMATIONS = [
  "Your feelings matter. Speak freely.",
  "Every emotion is valid. Let it out.",
  "You're doing something brave right now.",
  "Taking time for yourself is strength.",
  "Your voice tells a beautiful story.",
  "Breathe easy. You're safe here.",
  "This moment of self-awareness changes everything.",
  "You're more resilient than you know.",
  "Feelings are messengers, not enemies.",
  "Right now, you're choosing to grow.",
  "Your inner world deserves attention.",
  "Small moments of awareness lead to big clarity.",
  "Be gentle with yourself today.",
  "You're building emotional intelligence, one moment at a time.",
  "The fact that you're here shows you care about yourself.",
];

const ANALYZING_AFFIRMATIONS = [
  "Understanding yourself is a superpower.",
  "Your emotions are being heard.",
  "Every reading makes you stronger.",
  "Self-knowledge is the beginning of wisdom.",
  "You're investing in your wellbeing.",
  "Awareness is the first step to change.",
];

function getRandomAffirmation(list: string[]): string {
  return list[Math.floor(Math.random() * list.length)];
}

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
  fearful: "😨",
  surprise: "😲",
  neutral: "😐",
  anxious: "😰",
  nervous: "😬",
  grateful: "🙏",
  proud: "😤",
  lonely: "🥺",
  hopeful: "🌱",
  overwhelmed: "😵",
  peaceful: "😌",
  excited: "🤩",
  frustrated: "😣",
};

// ─── emotion context ────────────────────────────────────────────────────────
// Evidence-informed descriptions of what each detected emotion means in the
// context of voice biomarkers. Voice emotion detection uses acoustic features
// like pitch variability, speaking rate, energy, and spectral tilt — these
// descriptions help users understand what was detected and why, rather than
// showing a bare label. Phrased as observations, not diagnoses.
//
// Sources: Scherer (2003) vocal affect expression, Juslin & Laukka (2003)
// acoustic profiles of emotion, Schuller et al. (2018) computational
// paralinguistics.

interface EmotionContext {
  /** What voice patterns are associated with this emotion */
  voicePattern: string;
  /** A brief, gentle insight — not a diagnosis */
  insight: string;
}

const EMOTION_CONTEXT: Record<string, EmotionContext> = {
  happy: {
    voicePattern: "Higher pitch, varied intonation, energetic pace",
    insight: "Your voice sounds upbeat. Positive energy tends to show up as wider pitch range and lively rhythm.",
  },
  sad: {
    voicePattern: "Lower pitch, slower pace, less variation",
    insight: "Your voice sounds quieter and more subdued. This can reflect tiredness or low mood — both are normal parts of the day.",
  },
  angry: {
    voicePattern: "Higher energy, faster rate, tense quality",
    insight: "Your voice carries more tension and intensity. This could reflect frustration, urgency, or just a stressful moment.",
  },
  fear: {
    voicePattern: "Higher pitch, uneven rhythm, breathier quality",
    insight: "Your voice shows some tension and irregularity. This can come from anxiety, uncertainty, or simply feeling unsettled.",
  },
  surprise: {
    voicePattern: "Sudden pitch jumps, varied energy, quick shifts",
    insight: "Your voice has noticeable shifts in pitch and energy, which can reflect alertness or reacting to something unexpected.",
  },
  neutral: {
    voicePattern: "Steady pitch, moderate pace, even energy",
    insight: "Your voice sounds calm and balanced. A steady vocal pattern often reflects a settled, present state of mind.",
  },
  anxious: {
    voicePattern: "Slightly higher pitch, faster pace, tighter quality",
    insight: "Anxiety often surfaces as subtle vocal tension and a faster speaking rhythm. Acknowledging it is already a step forward.",
  },
  nervous: {
    voicePattern: "Uneven pace, slight tremor, breathier tone",
    insight: "Nervousness can show up as hesitation or a slight quiver. It often means something matters to you.",
  },
  grateful: {
    voicePattern: "Warm tone, moderate pace, relaxed resonance",
    insight: "Gratitude often brings warmth and steadiness to the voice. This is a grounding emotion.",
  },
  proud: {
    voicePattern: "Strong projection, measured pace, confident tone",
    insight: "Pride carries a sense of accomplishment. Your voice reflects confidence and self-assurance.",
  },
  lonely: {
    voicePattern: "Softer volume, slower pace, lower energy",
    insight: "Loneliness can make the voice quieter and more withdrawn. Connection is a basic human need.",
  },
  hopeful: {
    voicePattern: "Slightly rising intonation, open quality, gentle energy",
    insight: "Hope brings a lightness and openness to vocal expression. It signals looking forward.",
  },
  overwhelmed: {
    voicePattern: "Variable pace, higher tension, uneven energy",
    insight: "Feeling overwhelmed can scatter vocal patterns. It usually means too many things are competing for attention.",
  },
  peaceful: {
    voicePattern: "Steady, slow pace, soft resonance, even breathing",
    insight: "Peace shows up as vocal ease and regularity. Your nervous system sounds settled.",
  },
  excited: {
    voicePattern: "Higher pitch, faster pace, wide range, more energy",
    insight: "Excitement lifts the voice and quickens the pace. Positive anticipation is a wonderful state.",
  },
  frustrated: {
    voicePattern: "Tense quality, clipped rhythm, moderate-high energy",
    insight: "Frustration often sounds like contained tension. It usually signals unmet expectations or blocked goals.",
  },
};

// ─── explainability helpers (XAI) ────────────────────────────────────────────
// Plain-language explanations of why a particular emotion was detected, based on
// which acoustic features drove the prediction. Uses hedging language per XAI
// best practices ("suggests", "may indicate").

export const EMOTION_EXPLANATIONS: Record<string, string> = {
  happy: "Your voice had higher energy and faster pace, which typically indicates positive mood.",
  sad: "Your voice had lower energy and slower pace, often associated with lower mood.",
  angry: "Your voice had high energy with rapid pace, which can indicate frustration or intensity.",
  neutral: "Your voice showed balanced energy and pace, suggesting a calm state.",
  calm: "Your voice had gentle energy with steady rhythm, suggesting relaxation.",
  peaceful: "Your voice had gentle energy with steady rhythm, suggesting relaxation.",
  fear: "Your voice showed tension patterns with higher pitch variation, which can suggest unease or alertness.",
  fearful: "Your voice showed tension patterns with higher pitch variation, which can suggest unease or alertness.",
  surprise: "Your voice had sudden energy shifts and pitch variation, which may suggest alertness.",
  anxious: "Your voice showed subtle tension and a slightly faster pace, which may reflect unease.",
  nervous: "Your voice had uneven pacing with slight breathiness, which can suggest apprehension.",
  grateful: "Your voice had warm, steady tones with relaxed pacing, often reflecting contentment.",
  proud: "Your voice carried strong projection and measured pace, suggesting confidence.",
  lonely: "Your voice was softer with lower energy, which can reflect a withdrawn state.",
  hopeful: "Your voice had a gentle lift in energy and openness, suggesting forward-looking mood.",
  overwhelmed: "Your voice showed variable pacing and higher tension, which may reflect mental load.",
  excited: "Your voice had higher energy and faster pace with wide variation, suggesting enthusiasm.",
  frustrated: "Your voice carried contained tension with clipped rhythm, which can indicate blocked goals.",
};

/** Compute percentage values for the four feature-contribution bars. */
export function computeFeatureBars(result: {
  arousal: number;
  stress_index: number;
  valence: number;
  confidence: number;
}): { label: string; value: number; color: string }[] {
  // Energy level: derived from arousal (0-1 scale)
  const energy = Math.round(Math.max(0, Math.min(1, result.arousal)) * 100);
  // Voice pace: inverse of stress — lower stress suggests more natural pace
  const pace = Math.round(Math.max(0, Math.min(1, 1 - result.stress_index)) * 100);
  // Positivity: valence mapped from [-1,1] to [0,100]
  const positivity = Math.round(Math.max(0, Math.min(1, (result.valence + 1) / 2)) * 100);
  // Confidence: model confidence (0-1 scale)
  const confidence = Math.round(Math.max(0, Math.min(1, result.confidence)) * 100);

  return [
    { label: "Energy level", value: energy, color: "bg-violet-500" },
    { label: "Voice pace", value: pace, color: "bg-cyan-500" },
    { label: "Positivity", value: positivity, color: positivity >= 50 ? "bg-violet-500" : "bg-rose-500" },
    { label: "Confidence", value: confidence, color: "bg-cyan-500" },
  ];
}

function valenceLabel(v: number): { text: string; className: string } {
  if (v >= 0.4) return { text: "Positive", className: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30" };
  if (v <= -0.4) return { text: "Negative", className: "bg-rose-500/20 text-rose-400 border-rose-500/30" };
  return { text: "Neutral", className: "bg-muted/50 text-muted-foreground border-border/40" };
}

// ─── component ───────────────────────────────────────────────────────────────

interface VoiceCheckinCardProps {
  userId?: string;
  onComplete?: (result: VoiceWatchCheckinResult) => void;
  /** When true, always show the card regardless of period/completion status. Used by bottom tab mic. */
  forceShow?: boolean;
}

type CardState = "idle" | "recording" | "analyzing" | "done" | "dismissed";

const RECORD_SEC = 15;

export function VoiceCheckinCard({
  userId,
  onComplete,
  forceShow = false,
}: VoiceCheckinCardProps) {
  const resolvedUserId = userId ?? getParticipantId();
  const period = getCurrentPeriod();
  const queryClient = useQueryClient();

  const [cardState, setCardState] = useState<CardState>(() => {
    if (forceShow) return "idle";
    if (!period) return "dismissed";
    if (isCheckinDone(period)) return "dismissed";
    return "idle";
  });

  const [countdown, setCountdown] = useState(RECORD_SEC);
  const [amplitude, setAmplitude] = useState<number[]>(Array(12).fill(0.15));
  const [result, setResult] = useState<VoiceWatchCheckinResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [affirmation, setAffirmation] = useState("");
  const [feedbackState, setFeedbackState] = useState<"ask" | "correcting" | "confirmed" | "corrected">("ask");
  const [correctedEmotion, setCorrectedEmotion] = useState<string | null>(null);
  const [showNuancedEmotions, setShowNuancedEmotions] = useState(false);
  const [showExplainability, setShowExplainability] = useState(false);
  const [biomarkers, setBiomarkers] = useState<VoiceBiomarkers | null>(null);
  const [showVoiceHealth, setShowVoiceHealth] = useState(false);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const amplitudeRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const affirmationTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  // PCM capture: ScriptProcessorNode grabs raw samples during recording so we
  // don't depend on AudioContext.decodeAudioData() (fails on some Android WebViews).
  const pcmChunksRef = useRef<Float32Array[]>([]);
  const recordingCtxRef = useRef<AudioContext | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stopTimerRef.current) clearTimeout(stopTimerRef.current);
      if (countdownRef.current) clearInterval(countdownRef.current);
      if (amplitudeRef.current) clearInterval(amplitudeRef.current);
      if (affirmationTimerRef.current) clearInterval(affirmationTimerRef.current);
      if (recorderRef.current?.state === "recording") recorderRef.current.stop();
      if (scriptProcessorRef.current) { scriptProcessorRef.current.disconnect(); scriptProcessorRef.current = null; }
      if (recordingCtxRef.current) { recordingCtxRef.current.close().catch(() => {}); recordingCtxRef.current = null; }
    };
  }, []);

  const dismiss = useCallback(() => setCardState("dismissed"), []);

  const submitCorrection = useCallback(async (emotion: string) => {
    setCorrectedEmotion(emotion);
    setFeedbackState("corrected");
    // Fire-and-forget: save correction to backend
    fetch(resolveUrl(`/api/readings/${resolvedUserId}/correct-latest`), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ correctedEmotion: emotion }),
    }).catch((err) => console.error("Failed to save emotion correction:", err));
  }, [resolvedUserId]);

  const confirmEmotion = useCallback(async () => {
    setFeedbackState("confirmed");
    // Confirm = correct with the same emotion that was predicted
    if (result?.emotion) {
      fetch(resolveUrl(`/api/readings/${resolvedUserId}/correct-latest`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ correctedEmotion: result.emotion }),
      }).catch(() => {});
    }
  }, [resolvedUserId, result]);

  const startRecording = useCallback(async () => {
    if (cardState !== "idle") return;
    setError(null);
    setCountdown(RECORD_SEC);

    let stream: MediaStream;
    try {
      // On native Capacitor platforms (Android/iOS), the WebView's
      // onPermissionRequest in MainActivity handles bridging the
      // getUserMedia call to Android runtime permissions. We just
      // need to call getUserMedia and the native layer takes care of
      // prompting the user if app-level permissions aren't granted yet.
      //
      // On web, this goes through the standard browser permission flow.
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      // Provide a more helpful error on native platforms
      let message = "Microphone access denied";
      try {
        const { Capacitor } = await import("@capacitor/core");
        if (Capacitor.isNativePlatform()) {
          message = "Microphone access denied. Open your device Settings and enable microphone permission for this app.";
        }
      } catch {
        // Not on Capacitor — use default message
      }
      setError(message);
      return;
    }

    // Set up Web Audio analyser for amplitude bars + PCM capture
    pcmChunksRef.current = [];
    try {
      const audioCtx = new AudioContext();
      recordingCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 32;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Capture raw PCM via ScriptProcessorNode so we don't depend on
      // decodeAudioData (which fails on some Android WebViews for webm/opus).
      try {
        const sp = audioCtx.createScriptProcessor(4096, 1, 1);
        sp.onaudioprocess = (e: AudioProcessingEvent) => {
          pcmChunksRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        };
        source.connect(sp);
        sp.connect(audioCtx.destination);
        scriptProcessorRef.current = sp;
      } catch {
        // ScriptProcessor not available — will rely on decodeAudioData path
      }

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
      // Audio processing is best-effort; recording still works via MediaRecorder
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
      if (scriptProcessorRef.current) { scriptProcessorRef.current.disconnect(); scriptProcessorRef.current = null; }
      setAmplitude(Array(12).fill(0.15));
      setCardState("analyzing");

      // Grab captured PCM + sample rate before closing the recording context
      const capturedPcm = pcmChunksRef.current;
      const capturedSr = recordingCtxRef.current?.sampleRate ?? 44100;
      if (recordingCtxRef.current) { recordingCtxRef.current.close().catch(() => {}); recordingCtxRef.current = null; }

      // ── Helpers ──────────────────────────────────────────────────────────

      /** Concatenate captured PCM chunks into a single buffer. */
      function concatPcm(chunks: Float32Array[]): Float32Array {
        const total = chunks.reduce((s, c) => s + c.length, 0);
        const out = new Float32Array(total);
        let off = 0;
        for (const c of chunks) { out.set(c, off); off += c.length; }
        return out;
      }

      /** Encode mono Float32Array as 16-bit PCM WAV. */
      function pcmToWav(samples: Float32Array, sr: number): ArrayBuffer {
        const n = samples.length;
        const bps = 2;
        const dataLen = n * bps;
        const buf = new ArrayBuffer(44 + dataLen);
        const v = new DataView(buf);
        const ws = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
        ws(0, "RIFF"); v.setUint32(4, 36 + dataLen, true);
        ws(8, "WAVE"); ws(12, "fmt ");
        v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
        v.setUint32(24, sr, true); v.setUint32(28, sr * bps, true);
        v.setUint16(32, bps, true); v.setUint16(34, 16, true);
        ws(36, "data"); v.setUint32(40, dataLen, true);
        let o = 44;
        for (let i = 0; i < n; i++) {
          const s = Math.max(-1, Math.min(1, samples[i]));
          v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true); o += 2;
        }
        return buf;
      }

      /** On-device emotion heuristics from raw PCM. */
      function analyzeFromPcm(samples: Float32Array, sr: number): VoiceWatchCheckinResult {
        const n = samples.length;
        let sumSq = 0;
        for (let i = 0; i < n; i++) sumSq += samples[i] * samples[i];
        const rms = Math.sqrt(sumSq / n);
        let zcr = 0;
        for (let i = 1; i < n; i++) { if ((samples[i] >= 0) !== (samples[i - 1] >= 0)) zcr++; }
        const zcrRate = zcr / n;
        const frameSize = Math.floor(sr * 0.025);
        let syllables = 0; let wasSilent = true;
        for (let i = 0; i < n; i += frameSize) {
          let fe = 0; const end = Math.min(i + frameSize, n);
          for (let j = i; j < end; j++) fe += samples[j] * samples[j];
          fe /= (end - i);
          if (fe > 0.001 && wasSilent) syllables++;
          wasSilent = fe <= 0.001;
        }
        const speakingRate = syllables / (n / sr);
        const energy = Math.min(1, rms * 15);
        const stress = Math.min(1, Math.max(0, zcrRate * 8 + energy * 0.3));
        const arousal = Math.min(1, energy * 0.6 + speakingRate * 0.08);
        const valence = Math.max(-1, Math.min(1,
          energy > 0.5 && stress < 0.4 ? 0.3 + energy * 0.3 :
          stress > 0.6 ? -0.2 - stress * 0.3 :
          0.1 - stress * 0.2));
        let emotion = "neutral";
        if (valence > 0.3 && arousal > 0.4) emotion = "happy";
        else if (valence < -0.3 && arousal < 0.4) emotion = "sad";
        else if (valence < -0.2 && arousal > 0.6) emotion = "angry";
        else if (stress > 0.6) emotion = "fear";
        else if (energy < 0.15) emotion = "neutral";
        else if (valence > 0.1) emotion = "happy";
        return {
          checkin_id: `${Date.now()}`, checkin_type: period ?? "morning",
          emotion, valence, arousal, confidence: 0.45,
          stress_index: stress, focus_index: Math.max(0.2, Math.min(0.8, 1 - stress * 0.5)),
          model_type: "on-device" as any, timestamp: Date.now() / 1000, biomarkers: undefined,
        };
      }

      // ── Main analysis pipeline ──────────────────────────────────────────

      try {
        // Step 1: Obtain WAV + PCM.
        // Try decodeAudioData first (works on desktop browsers).
        // If it fails (Android WebView can't decode webm/opus), fall back to
        // PCM captured in real-time via ScriptProcessorNode.
        let wavBuffer: ArrayBuffer;
        let pcmSamples: Float32Array;
        let pcmSr: number;

        try {
          const blob = new Blob(chunksRef.current, { type: mimeType || "audio/webm" });
          const ab = await blob.arrayBuffer();
          const ctx = new AudioContext();
          const decoded = await ctx.decodeAudioData(ab);
          ctx.close();
          wavBuffer = audioBufferToWav(decoded);
          pcmSamples = decoded.getChannelData(0);
          pcmSr = decoded.sampleRate;
        } catch (decodeErr) {
          // decodeAudioData failed — use PCM captured during recording
          console.warn("decodeAudioData failed — using captured PCM:", decodeErr);
          if (capturedPcm.length === 0) throw new Error("No audio data: decodeAudioData failed and no PCM was captured");
          pcmSamples = concatPcm(capturedPcm);
          pcmSr = capturedSr;
          wavBuffer = pcmToWav(pcmSamples, pcmSr);
        }

        // Step 1.5: Extract voice biomarkers from PCM (runs synchronously, no network)
        try {
          const vb = extractVoiceBiomarkers(pcmSamples, pcmSr);
          setBiomarkers(vb);
          // Persist to localStorage as an array with timestamp
          try {
            const stored = JSON.parse(localStorage.getItem("ndw_voice_biomarkers") || "[]") as VoiceBiomarkers[];
            stored.push(vb);
            // Keep last 100 entries to avoid unbounded growth
            if (stored.length > 100) stored.splice(0, stored.length - 100);
            localStorage.setItem("ndw_voice_biomarkers", JSON.stringify(stored));
          } catch { /* storage quota */ }
        } catch (bioErr) {
          console.warn("Voice biomarker extraction failed:", bioErr);
        }

        // Step 2: Try on-device ONNX first, then ML backend, then heuristics
        let checkinResult: VoiceWatchCheckinResult;
        try {
          // Priority 1: On-device ONNX voice emotion (no network needed)
          const onnxResult = await runVoiceEmotionONNX(pcmSamples, pcmSr);
          checkinResult = {
            checkin_id:     `${Date.now()}`,
            checkin_type:   period ?? "morning",
            emotion:        onnxResult.emotion,
            valence:        onnxResult.valence,
            arousal:        onnxResult.arousal,
            confidence:     onnxResult.confidence,
            stress_index:   onnxResult.stress_index,
            focus_index:    onnxResult.focus_index,
            model_type:     onnxResult.model_type,
            timestamp:      Date.now() / 1000,
            biomarkers:     onnxResult.probabilities,
          };
        } catch (onnxErr) {
          console.warn("On-device ONNX failed — trying ML backend:", onnxErr);
          try {
            // Priority 2: ML backend (requires network)
            const bytes = new Uint8Array(wavBuffer);
            let binary = "";
            for (let i = 0; i < bytes.byteLength; i++) {
              binary += String.fromCharCode(bytes[i]);
            }
            const audio_b64 = btoa(binary);

            const raw = await submitVoiceWatch(audio_b64, resolvedUserId);
            const stressIndex = raw.stress_index ?? raw.stress_from_watch ?? 0.5;
            const focusIndex = raw.focus_index ?? Math.max(0.2, Math.min(0.85, raw.confidence ?? 0.5));
            checkinResult = {
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
          } catch (mlErr) {
            // Priority 3: On-device heuristics (always available)
            console.warn("ML backend also unavailable — using on-device heuristics:", mlErr);
            checkinResult = analyzeFromPcm(pcmSamples, pcmSr);
          }
        }

        // ── Persist result ─────────────────────────────────────────────────
        setResult(checkinResult);
        if (period) markCheckinDone(period, checkinResult);
        setCardState("done");

        // Persist to ndw_last_emotion so useCurrentEmotion picks it up everywhere
        try {
          localStorage.setItem("ndw_last_emotion", JSON.stringify({
            result: checkinResult,
            timestamp: Date.now(),
          }));
        } catch { /* storage quota */ }
        // Append to voice history for "Recent Voice Analyses" display
        try {
          const historyKey = "ndw_voice_history";
          const existing = JSON.parse(localStorage.getItem(historyKey) || "[]") as any[];
          existing.unshift({
            ...checkinResult,
            timestamp: Date.now(),
          });
          if (existing.length > 50) existing.length = 50;
          localStorage.setItem(historyKey, JSON.stringify(existing));
        } catch { /* storage quota */ }
        // Track unique emotions for Emotion Explorer badge
        try {
          const emotion = checkinResult.emotion;
          if (emotion) {
            const seen = JSON.parse(localStorage.getItem("ndw_emotions_seen") || "[]") as string[];
            if (!seen.includes(emotion)) {
              seen.push(emotion);
              localStorage.setItem("ndw_emotions_seen", JSON.stringify(seen));
            }
          }
        } catch { /* storage quota */ }
        // Notify all mounted useCurrentEmotion hooks instantly
        window.dispatchEvent(new CustomEvent("ndw-emotion-update"));

        onComplete?.(checkinResult);

        // Fire-and-forget: write emotion to HealthKit (iOS only, Android/web no-op)
        writeEmotionToHealth(checkinResult.emotion, checkinResult.valence).catch(() => {});

        // Fire-and-forget: record a streak check-in so the StreakCard updates
        fetch(`${getMLApiUrl()}/api/streaks/checkin`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: resolvedUserId, checkin_type: "voice" }),
        }).catch((err) => console.error("Failed to save streak check-in:", err));

        // Save emotion reading to Express DB so Daily Report + Session History see it
        fetch(resolveUrl("/api/emotion-readings/batch"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            readings: [{
              userId: resolvedUserId,
              sessionId: `voice-${checkinResult.checkin_id}`,
              stress: checkinResult.stress_index,
              happiness: checkinResult.valence > 0 ? checkinResult.valence : 0,
              focus: checkinResult.focus_index,
              energy: checkinResult.arousal,
              dominantEmotion: checkinResult.emotion,
              valence: checkinResult.valence,
              arousal: checkinResult.arousal,
            }],
          }),
        }).catch((err) => console.error("Failed to save voice reading:", err));

        // Also save to userReadings for the brain history merge
        fetch(resolveUrl("/api/user-readings"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            userId: resolvedUserId,
            source: "voice",
            emotion: checkinResult.emotion,
            valence: checkinResult.valence,
            arousal: checkinResult.arousal,
            stress: checkinResult.stress_index,
            confidence: checkinResult.confidence,
            modelType: checkinResult.model_type || "voice",
          }),
        }).catch((err) => console.error("Failed to save user reading:", err));

        // Invalidate cached queries so Daily Report, Sessions, Streak, and Emotion Lab update
        queryClient.invalidateQueries({ queryKey: ["streak-status"] });
        queryClient.invalidateQueries({ queryKey: ["sessions-brain-report"] });
        queryClient.invalidateQueries({ queryKey: ["sessions"] });
        queryClient.invalidateQueries({ queryKey: ["emotions"] });
        queryClient.invalidateQueries({ queryKey: ["health-brain-report"] });
        queryClient.invalidateQueries({ queryKey: ["voice-latest-brain-report"] });
        queryClient.invalidateQueries({ queryKey: ["yesterday-insights"] });
        queryClient.invalidateQueries({ queryKey: ["brain-patterns"] });
        // Emotion Lab mood trends + recent readings
        queryClient.invalidateQueries({ queryKey: [`/api/brain/history/${resolvedUserId}?days=1`] });
        queryClient.invalidateQueries({ queryKey: [`/api/brain/history/${resolvedUserId}?days=7`] });
        queryClient.invalidateQueries({ queryKey: [`/api/brain/history/${resolvedUserId}?days=30`] });
        // Inner Energy voice fallback
        queryClient.invalidateQueries({ queryKey: ["voice-inner-energy", resolvedUserId] });
        // Food-emotion correlation
        queryClient.invalidateQueries({ queryKey: ["/api/food/logs", resolvedUserId] });
      } catch (err) {
        console.error("Voice analysis pipeline failed:", err);
        // Even if everything fails, save a basic neutral result so the app updates
        const fallbackResult: VoiceWatchCheckinResult = {
          checkin_id:     `${Date.now()}`,
          checkin_type:   period ?? "morning",
          emotion:        "neutral",
          valence:        0,
          arousal:        0.5,
          confidence:     0.2,
          stress_index:   0.3,
          focus_index:    0.5,
          model_type:     "fallback" as any,
          timestamp:      Date.now() / 1000,
          biomarkers:     undefined,
        };
        setResult(fallbackResult);
        if (period) markCheckinDone(period, fallbackResult);
        setCardState("done");
        try {
          localStorage.setItem("ndw_last_emotion", JSON.stringify({
            result: fallbackResult,
            timestamp: Date.now(),
          }));
        } catch { /* ok */ }
        // Append to voice history
        try {
          const historyKey = "ndw_voice_history";
          const existing = JSON.parse(localStorage.getItem(historyKey) || "[]") as any[];
          existing.unshift({ ...fallbackResult, timestamp: Date.now() });
          if (existing.length > 50) existing.length = 50;
          localStorage.setItem(historyKey, JSON.stringify(existing));
        } catch { /* ok */ }
        window.dispatchEvent(new CustomEvent("ndw-emotion-update"));
        onComplete?.(fallbackResult);
      }
    };

    recorder.start();
    setCardState("recording");

    // Show a positive affirmation and rotate every 3 seconds
    setAffirmation(getRandomAffirmation(RECORDING_AFFIRMATIONS));
    affirmationTimerRef.current = setInterval(() => {
      setAffirmation(getRandomAffirmation(RECORDING_AFFIRMATIONS));
    }, 3000);

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
  }, [cardState, resolvedUserId, period, onComplete]);

  // Don't render if dismissed or outside all windows (unless forceShow)
  if (!forceShow && cardState === "dismissed") return null;
  if (!forceShow && !period) return null;

  const periodLabel = period ? period.charAt(0).toUpperCase() + period.slice(1) : "Check-in";

  return (
    <Card className="bg-card rounded-xl border border-border/50 shadow-sm">
      <CardContent className="p-4">
        {/* Header row */}
        <div className="flex items-start justify-between mb-3">
          <div>
            <p className="text-sm font-semibold">{periodLabel} Voice Analysis</p>
            <p className="text-xs text-muted-foreground">Speak naturally for 15 seconds</p>
          </div>
          {cardState !== "recording" && cardState !== "analyzing" && (
            <button
              onClick={dismiss}
              className="text-muted-foreground hover:text-foreground transition-colors p-1 -mt-1 -mr-1"
              aria-label="Skip voice analysis"
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
              Tap to record 15 seconds
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
            {affirmation && (
              <p className="text-xs text-center text-muted-foreground/80 italic max-w-[220px] leading-relaxed mt-1">
                "{affirmation}"
              </p>
            )}
          </div>
        )}

        {/* Analyzing state */}
        {cardState === "analyzing" && (
          <div className="flex flex-col items-center gap-2 py-4">
            <div className="h-8 w-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
            <p className="text-xs text-muted-foreground">Analyzing voice…</p>
            <p className="text-xs text-center text-muted-foreground/80 italic max-w-[220px] leading-relaxed">
              "{getRandomAffirmation(ANALYZING_AFFIRMATIONS)}"
            </p>
          </div>
        )}

        {/* Done state — show result with emotion context */}
        {cardState === "done" && result && (() => {
          const vl = valenceLabel(result.valence);
          const emoji = EMOTION_EMOJI[result.emotion] ?? "🧠";
          const ctx = EMOTION_CONTEXT[result.emotion];
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
              {/* Emotion context — explain what was detected and why */}
              {ctx && (
                <div className="rounded-lg bg-muted/30 border border-border/30 p-2.5 space-y-1">
                  <p className="text-xs text-foreground/80 leading-relaxed">
                    {ctx.insight}
                  </p>
                  <p className="text-[10px] text-muted-foreground/70 leading-relaxed">
                    Voice pattern: {ctx.voicePattern}
                  </p>
                </div>
              )}

              {/* "Why this emotion?" explainability card */}
              {(() => {
                const explanation = EMOTION_EXPLANATIONS[result.emotion];
                const bars = computeFeatureBars(result);
                return (
                  <div className="rounded-lg border border-border/30 overflow-hidden">
                    <button
                      onClick={() => setShowExplainability((s) => !s)}
                      className="flex items-center justify-between w-full px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-muted/20 transition-colors"
                      aria-expanded={showExplainability}
                    >
                      <span>Why this emotion?</span>
                      {showExplainability
                        ? <ChevronUp className="h-3.5 w-3.5" />
                        : <ChevronDown className="h-3.5 w-3.5" />}
                    </button>
                    {showExplainability && (
                      <div className="px-3 pb-3 space-y-3 border-t border-border/20">
                        {/* Plain-language explanation */}
                        {explanation && (
                          <p className="text-xs text-foreground/80 leading-relaxed pt-2">
                            {explanation}
                          </p>
                        )}
                        {/* Feature contribution bars */}
                        <div className="space-y-2">
                          {bars.map((bar) => (
                            <div key={bar.label} className="space-y-0.5">
                              <div className="flex items-center justify-between">
                                <span className="text-[10px] text-muted-foreground">{bar.label}</span>
                                <span className="text-[10px] font-mono text-foreground/70">{bar.value}%</span>
                              </div>
                              <div className="h-1.5 rounded-full bg-muted/40 overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${bar.color} transition-all duration-500`}
                                  style={{ width: `${bar.value}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                        {/* Wellness disclaimer */}
                        <p className="text-[10px] text-muted-foreground/50 italic leading-relaxed" data-testid="explainability-disclaimer">
                          Voice analysis is for wellness awareness only. It reflects acoustic patterns, not clinical assessment.
                        </p>
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* ── Collapsible Voice Health section ── */}
              {biomarkers && (
                <div className="rounded-lg border border-border/30 overflow-hidden">
                  <button
                    onClick={() => setShowVoiceHealth((s) => !s)}
                    className="flex items-center justify-between w-full px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-muted/20 transition-colors"
                    aria-expanded={showVoiceHealth}
                  >
                    <span>Voice Health</span>
                    {showVoiceHealth
                      ? <ChevronUp className="h-3.5 w-3.5" />
                      : <ChevronDown className="h-3.5 w-3.5" />}
                  </button>
                  {showVoiceHealth && (
                    <div className="px-3 pb-3 space-y-2.5 border-t border-border/20 pt-2">
                      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Vocal Energy</span>
                          <span className="font-mono text-foreground/80 capitalize">{biomarkers.vocalEnergyLevel}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Variability</span>
                          <span className="font-mono text-foreground/80 capitalize">{biomarkers.vocalVariability}</span>
                        </div>
                      </div>
                      {/* Wellness score bar */}
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">Wellness Score</span>
                          <span className="font-mono text-foreground/80">{biomarkers.wellnessScore}/100</span>
                        </div>
                        <div className="h-2 rounded-full bg-muted/40 overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              biomarkers.wellnessScore >= 70
                                ? "bg-emerald-500"
                                : biomarkers.wellnessScore >= 40
                                  ? "bg-amber-500"
                                  : "bg-rose-500"
                            }`}
                            style={{ width: `${biomarkers.wellnessScore}%` }}
                          />
                        </div>
                      </div>
                      <p className="text-[10px] text-muted-foreground/50 italic leading-relaxed">
                        {biomarkers.disclaimer}
                      </p>
                    </div>
                  )}
                </div>
              )}

              <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                <span>Stress: <span className="font-mono text-foreground/80">{Math.round(result.stress_index * 100)}%</span></span>
                <span>Focus: <span className="font-mono text-foreground/80">{Math.round(result.focus_index * 100)}%</span></span>
              </div>
              <p className="text-[10px] text-muted-foreground/60 italic">
                Voice analysis reflects acoustic patterns, not a clinical assessment.
              </p>

              {/* ── "Was this right?" feedback row ── */}
              {feedbackState === "ask" && (
                <div className="flex items-center justify-between pt-1 border-t border-border/30">
                  <span className="text-xs text-muted-foreground">Was this right?</span>
                  <div className="flex gap-2">
                    <button
                      onClick={confirmEmotion}
                      className="flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-medium bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
                    >
                      <Check className="h-3 w-3" />
                      Yes
                    </button>
                    <button
                      onClick={() => setFeedbackState("correcting")}
                      className="flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-medium bg-muted/50 text-muted-foreground hover:bg-muted transition-colors"
                    >
                      <Pencil className="h-3 w-3" />
                      Correct it
                    </button>
                  </div>
                </div>
              )}

              {feedbackState === "correcting" && (
                <div className="space-y-2 pt-1 border-t border-border/30">
                  <span className="text-xs text-muted-foreground">What were you actually feeling?</span>
                  <div className="flex flex-wrap gap-1.5">
                    {(["happy", "sad", "angry", "fearful", "surprised", "neutral"] as const).map((em) => (
                      <button
                        key={em}
                        onClick={() => submitCorrection(em)}
                        className={`px-3 py-1.5 rounded-full text-xs font-medium capitalize transition-colors ${
                          em === result.emotion
                            ? "bg-muted/30 text-muted-foreground/60"
                            : "bg-primary/10 text-primary hover:bg-primary/20"
                        }`}
                      >
                        {em}
                      </button>
                    ))}
                  </div>
                  {!showNuancedEmotions ? (
                    <button
                      onClick={() => setShowNuancedEmotions(true)}
                      className="text-xs text-muted-foreground hover:text-primary transition-colors underline underline-offset-2"
                    >
                      More feelings...
                    </button>
                  ) : (
                    <div className="flex flex-wrap gap-1.5 pt-1">
                      {(["anxious", "nervous", "grateful", "proud", "lonely", "hopeful", "overwhelmed", "peaceful", "excited", "frustrated"] as const).map((em) => (
                        <button
                          key={em}
                          onClick={() => submitCorrection(em)}
                          className="px-3 py-1.5 rounded-full text-xs font-medium capitalize transition-colors bg-secondary/50 text-secondary-foreground hover:bg-secondary/80"
                        >
                          {em}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {feedbackState === "confirmed" && (
                <div className="flex items-center gap-2 pt-1 border-t border-border/30">
                  <Check className="h-3.5 w-3.5 text-primary" />
                  <span className="text-xs text-primary font-medium">Thanks!</span>
                </div>
              )}

              {feedbackState === "corrected" && correctedEmotion && (
                <div className="flex items-center gap-2 pt-1 border-t border-border/30">
                  <Check className="h-3.5 w-3.5 text-primary" />
                  <span className="text-xs text-primary font-medium">
                    Saved as {correctedEmotion}. Thanks for helping improve accuracy!
                  </span>
                </div>
              )}

              <p className="text-xs text-muted-foreground">
                Next analysis at {nextWindowLabel(period)}
              </p>
            </div>
          );
        })()}
      </CardContent>
    </Card>
  );
}
