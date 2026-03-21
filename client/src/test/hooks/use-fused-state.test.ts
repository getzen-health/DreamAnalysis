import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useFusedState } from "@/hooks/use-fused-state";
import { dataFusionBus } from "@/lib/data-fusion";

beforeEach(() => {
  localStorage.clear();
  dataFusionBus.destroy();
});

afterEach(() => {
  localStorage.clear();
  dataFusionBus.destroy();
});

describe("useFusedState", () => {
  it("returns null state when no data sources are available", () => {
    const { result } = renderHook(() => useFusedState());
    expect(result.current.fusedState).toBeNull();
    expect(result.current.source).toBeNull();
    expect(result.current.isReady).toBe(false);
  });

  it("returns EEG state when EEG data is present", () => {
    localStorage.setItem("ndw_last_eeg_emotion", JSON.stringify({
      emotion: "happy",
      valence: 0.6,
      arousal: 0.7,
      stress_index: 0.2,
      focus_index: 0.8,
      confidence: 0.9,
      timestamp: Date.now(),
    }));

    const { result } = renderHook(() => useFusedState());

    // The bus initializes and reads localStorage on first render
    expect(result.current.isReady).toBe(true);
    expect(result.current.fusedState).not.toBeNull();
    expect(result.current.fusedState!.emotion).toBe("happy");
    expect(result.current.source).toBe("eeg");
  });

  it("returns voice state when only voice data is present", () => {
    localStorage.setItem("ndw_last_emotion", JSON.stringify({
      result: {
        emotion: "sad",
        valence: -0.3,
        arousal: 0.4,
        stress_index: 0.6,
        focus_index: 0.3,
        confidence: 0.7,
      },
      timestamp: Date.now(),
    }));

    const { result } = renderHook(() => useFusedState());

    expect(result.current.isReady).toBe(true);
    expect(result.current.fusedState!.emotion).toBe("sad");
    expect(result.current.source).toBe("voice");
  });

  it("updates when a source event fires", () => {
    // Start with no data
    localStorage.clear();
    dataFusionBus.destroy();

    const { result } = renderHook(() => useFusedState());

    // No data sources — should not be ready
    expect(result.current.isReady).toBe(false);

    // Simulate EEG data arriving
    act(() => {
      localStorage.setItem("ndw_last_eeg_emotion", JSON.stringify({
        emotion: "neutral",
        valence: 0.0,
        arousal: 0.5,
        confidence: 0.6,
        timestamp: Date.now(),
      }));
      window.dispatchEvent(new Event("ndw-eeg-updated"));
    });

    expect(result.current.isReady).toBe(true);
    expect(result.current.fusedState!.emotion).toBe("neutral");
  });

  it("returns fused source when multiple sources are present", () => {
    localStorage.setItem("ndw_last_eeg_emotion", JSON.stringify({
      emotion: "happy",
      valence: 0.5,
      arousal: 0.6,
      stress_index: 0.2,
      focus_index: 0.8,
      confidence: 0.9,
      timestamp: Date.now(),
    }));
    localStorage.setItem("ndw_last_emotion", JSON.stringify({
      result: {
        emotion: "neutral",
        valence: 0.1,
        arousal: 0.5,
        stress_index: 0.3,
        focus_index: 0.5,
        confidence: 0.7,
      },
      timestamp: Date.now(),
    }));

    const { result } = renderHook(() => useFusedState());

    expect(result.current.isReady).toBe(true);
    expect(result.current.source).toBe("fused");
    // EEG has highest weight, so emotion should be "happy"
    expect(result.current.fusedState!.emotion).toBe("happy");
  });
});
