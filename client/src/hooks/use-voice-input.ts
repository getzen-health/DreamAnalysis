/**
 * useVoiceInput — Web Speech API hook for dream recording.
 *
 * Design choices:
 * - continuous: true  — keeps listening until the user explicitly stops,
 *   which is better for narrating a dream (not just one utterance)
 * - interimResults: false — only final results; no jumpy interim text
 * - Accumulated transcript is delivered via `appendedText` state so the
 *   consumer can watch it with useEffect without stale-closure risk
 *
 * Usage:
 *   const { isSupported, isListening, appendedText, clearAppended, start, stop }
 *     = useVoiceInput();
 *
 *   useEffect(() => {
 *     if (appendedText) {
 *       setMyText(prev => prev ? `${prev} ${appendedText}` : appendedText);
 *       clearAppended();
 *     }
 *   }, [appendedText, clearAppended]);
 */

import { useState, useRef, useCallback } from "react";

// ── Pure helpers (exported for testing) ──────────────────────────────────────

/** Join final transcript segments from a SpeechRecognitionResultList-like array. */
export function joinTranscriptParts(parts: Array<{ transcript: string }>): string {
  return parts
    .map((p) => p.transcript.trim())
    .filter(Boolean)
    .join(" ")
    .trim();
}

/** Detect if SpeechRecognition is available in the current environment. */
export function detectSpeechSupport(): boolean {
  if (typeof window === "undefined") return false;
  return Boolean(
    (window as unknown as Record<string, unknown>).SpeechRecognition ??
    (window as unknown as Record<string, unknown>).webkitSpeechRecognition,
  );
}

// ── Hook ─────────────────────────────────────────────────────────────────────

export interface UseVoiceInputResult {
  /** Whether the browser supports Web Speech API. */
  isSupported: boolean;
  /** True while SpeechRecognition is actively listening. */
  isListening: boolean;
  /**
   * The latest chunk of text produced when recognition ends.
   * Non-null only while unread — call clearAppended() to consume it.
   */
  appendedText: string | null;
  /** Reset appendedText to null after the consumer has applied it. */
  clearAppended: () => void;
  /** Start listening. No-op if unsupported or already listening. */
  start: () => void;
  /** Stop listening and trigger onend (which sets appendedText). */
  stop: () => void;
}

export function useVoiceInput(): UseVoiceInputResult {
  const [isListening, setIsListening] = useState(false);
  const [appendedText, setAppendedText] = useState<string | null>(null);
  const recRef = useRef<{
    stop: () => void;
    continuous: boolean;
    interimResults: boolean;
    lang: string;
    onresult: ((e: unknown) => void) | null;
    onerror: ((e: unknown) => void) | null;
    onend: (() => void) | null;
    start: () => void;
  } | null>(null);
  const accumulatedRef = useRef<string[]>([]);

  const isSupported = detectSpeechSupport();

  const stop = useCallback(() => {
    recRef.current?.stop();
  }, []);

  const clearAppended = useCallback(() => setAppendedText(null), []);

  const start = useCallback(() => {
    const SR =
      typeof window !== "undefined"
        ? ((window as unknown as Record<string, unknown>).SpeechRecognition ??
            (window as unknown as Record<string, unknown>).webkitSpeechRecognition)
        : null;
    if (!SR || recRef.current) return;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const rec = new (SR as any)();
    rec.continuous = true;
    rec.interimResults = false;
    rec.lang = "en-US";
    accumulatedRef.current = [];

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    rec.onresult = (event: any) => {
      for (let i = (event.resultIndex as number); i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          const segment: string = (event.results[i][0].transcript as string).trim();
          if (segment) accumulatedRef.current.push(segment);
        }
      }
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    rec.onerror = (event: any) => {
      // "no-speech" is expected (silence) — don't kill the session
      if ((event.error as string) !== "no-speech") {
        setIsListening(false);
        recRef.current = null;
      }
    };

    rec.onend = () => {
      const text = joinTranscriptParts(
        accumulatedRef.current.map((t) => ({ transcript: t })),
      );
      if (text) setAppendedText(text);
      accumulatedRef.current = [];
      setIsListening(false);
      recRef.current = null;
    };

    recRef.current = rec;
    rec.start();
    setIsListening(true);
  }, []);

  return { isSupported, isListening, appendedText, clearAppended, start, stop };
}
