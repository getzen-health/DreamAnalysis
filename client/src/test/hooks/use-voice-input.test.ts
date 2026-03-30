import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import {
  joinTranscriptParts,
  detectSpeechSupport,
  useVoiceInput,
} from "@/hooks/use-voice-input";

// ── Pure helper tests ─────────────────────────────────────────────────────────

describe("joinTranscriptParts", () => {
  it("joins multiple segments with a single space", () => {
    expect(joinTranscriptParts([
      { transcript: "I was flying" },
      { transcript: "over a green city" },
    ])).toBe("I was flying over a green city");
  });

  it("trims leading/trailing whitespace from each segment", () => {
    expect(joinTranscriptParts([
      { transcript: "  hello  " },
      { transcript: " world  " },
    ])).toBe("hello world");
  });

  it("filters empty/whitespace-only segments", () => {
    expect(joinTranscriptParts([
      { transcript: "" },
      { transcript: "  " },
      { transcript: "dream" },
    ])).toBe("dream");
  });

  it("returns empty string for empty array", () => {
    expect(joinTranscriptParts([])).toBe("");
  });

  it("returns empty string for all-whitespace array", () => {
    expect(joinTranscriptParts([{ transcript: "   " }, { transcript: "" }])).toBe("");
  });

  it("returns single segment correctly", () => {
    expect(joinTranscriptParts([{ transcript: "I was in a house" }])).toBe("I was in a house");
  });
});

// ── detectSpeechSupport ───────────────────────────────────────────────────────

describe("detectSpeechSupport", () => {
  it("returns false in jsdom (SpeechRecognition not defined)", () => {
    // jsdom doesn't include Web Speech API
    expect(detectSpeechSupport()).toBe(false);
  });

  it("returns true when SpeechRecognition is present on window", () => {
    const mockSR = vi.fn();
    Object.defineProperty(window, "SpeechRecognition", { value: mockSR, configurable: true });
    expect(detectSpeechSupport()).toBe(true);
    // cleanup
    Object.defineProperty(window, "SpeechRecognition", { value: undefined, configurable: true });
  });

  it("returns true when webkitSpeechRecognition is present on window", () => {
    const mockSR = vi.fn();
    Object.defineProperty(window, "webkitSpeechRecognition", { value: mockSR, configurable: true });
    expect(detectSpeechSupport()).toBe(true);
    // cleanup
    Object.defineProperty(window, "webkitSpeechRecognition", { value: undefined, configurable: true });
  });
});

// ── useVoiceInput hook ────────────────────────────────────────────────────────

describe("useVoiceInput", () => {
  it("isSupported is false in jsdom", () => {
    const { result } = renderHook(() => useVoiceInput());
    expect(result.current.isSupported).toBe(false);
  });

  it("isListening starts false", () => {
    const { result } = renderHook(() => useVoiceInput());
    expect(result.current.isListening).toBe(false);
  });

  it("appendedText starts null", () => {
    const { result } = renderHook(() => useVoiceInput());
    expect(result.current.appendedText).toBeNull();
  });

  it("start() is a no-op when SpeechRecognition is absent", () => {
    const { result } = renderHook(() => useVoiceInput());
    act(() => { result.current.start(); });
    expect(result.current.isListening).toBe(false);
    expect(result.current.appendedText).toBeNull();
  });

  it("stop() does not throw when not listening", () => {
    const { result } = renderHook(() => useVoiceInput());
    expect(() => act(() => { result.current.stop(); })).not.toThrow();
  });

  it("clearAppended resets appendedText to null", () => {
    const { result } = renderHook(() => useVoiceInput());
    // Manually simulate appendedText being set (via mocked SpeechRecognition)
    const mockRec = {
      continuous: false,
      interimResults: false,
      lang: "",
      onresult: null as ((e: unknown) => void) | null,
      onerror: null as ((e: unknown) => void) | null,
      onend: null as (() => void) | null,
      start: vi.fn(),
      stop: vi.fn(),
    };
    // Must use a regular function so `new MockSR()` works as a constructor
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    function MockSR(this: any) { return mockRec; }
    Object.defineProperty(window, "SpeechRecognition", { value: MockSR, configurable: true, writable: true });

    act(() => { result.current.start(); });
    expect(result.current.isListening).toBe(true);

    // Simulate onresult + onend
    act(() => {
      const fakeEvent = {
        resultIndex: 0,
        results: [{ isFinal: true, 0: { transcript: "flying over ocean" } }],
      };
      mockRec.onresult?.(fakeEvent);
      mockRec.onend?.();
    });

    expect(result.current.appendedText).toBe("flying over ocean");
    expect(result.current.isListening).toBe(false);

    act(() => { result.current.clearAppended(); });
    expect(result.current.appendedText).toBeNull();

    // cleanup
    Object.defineProperty(window, "SpeechRecognition", { value: undefined, configurable: true, writable: true });
  });

  it("accumulates multiple result segments before onend", () => {
    const mockRec = {
      continuous: true,
      interimResults: false,
      lang: "",
      onresult: null as ((e: unknown) => void) | null,
      onerror: null as ((e: unknown) => void) | null,
      onend: null as (() => void) | null,
      start: vi.fn(),
      stop: vi.fn(),
    };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    function MockSR(this: any) { return mockRec; }
    Object.defineProperty(window, "SpeechRecognition", { value: MockSR, configurable: true, writable: true });

    const { result } = renderHook(() => useVoiceInput());
    act(() => { result.current.start(); });

    act(() => {
      // First phrase
      mockRec.onresult?.({
        resultIndex: 0,
        results: [{ isFinal: true, 0: { transcript: "I was in a forest" } }],
      });
      // Second phrase
      mockRec.onresult?.({
        resultIndex: 1,
        results: [
          { isFinal: true, 0: { transcript: "I was in a forest" } },
          { isFinal: true, 0: { transcript: "and it was dark" } },
        ],
      });
      // End
      mockRec.onend?.();
    });

    expect(result.current.appendedText).toBe("I was in a forest and it was dark");

    // cleanup
    Object.defineProperty(window, "SpeechRecognition", { value: undefined, configurable: true, writable: true });
  });

  it("onerror with 'no-speech' does not stop listening", () => {
    const mockRec = {
      continuous: true,
      interimResults: false,
      lang: "",
      onresult: null as ((e: unknown) => void) | null,
      onerror: null as ((e: unknown) => void) | null,
      onend: null as (() => void) | null,
      start: vi.fn(),
      stop: vi.fn(),
    };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    function MockSR(this: any) { return mockRec; }
    Object.defineProperty(window, "SpeechRecognition", { value: MockSR, configurable: true, writable: true });

    const { result } = renderHook(() => useVoiceInput());
    act(() => { result.current.start(); });
    expect(result.current.isListening).toBe(true);

    // "no-speech" should NOT kill the session
    act(() => { mockRec.onerror?.({ error: "no-speech" }); });
    expect(result.current.isListening).toBe(true);

    // Cleanup
    act(() => { mockRec.onend?.(); });
    Object.defineProperty(window, "SpeechRecognition", { value: undefined, configurable: true, writable: true });
  });

  it("onerror with 'network' stops listening", () => {
    const mockRec = {
      continuous: true,
      interimResults: false,
      lang: "",
      onresult: null as ((e: unknown) => void) | null,
      onerror: null as ((e: unknown) => void) | null,
      onend: null as (() => void) | null,
      start: vi.fn(),
      stop: vi.fn(),
    };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    function MockSR(this: any) { return mockRec; }
    Object.defineProperty(window, "SpeechRecognition", { value: MockSR, configurable: true, writable: true });

    const { result } = renderHook(() => useVoiceInput());
    act(() => { result.current.start(); });

    act(() => { mockRec.onerror?.({ error: "network" }); });
    expect(result.current.isListening).toBe(false);

    Object.defineProperty(window, "SpeechRecognition", { value: undefined, configurable: true, writable: true });
  });
});
