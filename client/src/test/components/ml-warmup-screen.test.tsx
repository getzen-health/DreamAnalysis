import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, fireEvent, act } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import * as connectionHook from "@/hooks/use-ml-connection";
import type { MLConnectionState } from "@/hooks/use-ml-connection";
import { MLWarmupScreen } from "@/components/ml-warmup-screen";

// ── helpers ───────────────────────────────────────────────────────────────────

function makeState(overrides: Partial<MLConnectionState> = {}): MLConnectionState {
  return {
    status: "connecting",
    latencyMs: null,
    warmupProgress: 0,
    retryCount: 0,
    reconnect: vi.fn(),
    ...overrides,
  };
}

function mockHook(state: MLConnectionState) {
  vi.spyOn(connectionHook, "useMLConnection").mockReturnValue(state);
}

// ── tests ─────────────────────────────────────────────────────────────────────

describe("MLWarmupScreen", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("renders when status is 'connecting'", () => {
    mockHook(makeState({ status: "connecting" }));
    renderWithProviders(<MLWarmupScreen />);
    expect(screen.getByText("Neural Dream Workshop")).toBeInTheDocument();
  });

  it("renders when status is 'warming'", () => {
    mockHook(makeState({ status: "warming", warmupProgress: 45 }));
    renderWithProviders(<MLWarmupScreen />);
    expect(screen.getByText("Neural Dream Workshop")).toBeInTheDocument();
  });

  it("returns null when status is 'ready'", () => {
    mockHook(makeState({ status: "ready" }));
    const { container } = renderWithProviders(<MLWarmupScreen />);
    expect(container.firstChild).toBeNull();
  });

  it("returns null when status is 'idle'", () => {
    mockHook(makeState({ status: "idle" }));
    const { container } = renderWithProviders(<MLWarmupScreen />);
    expect(container.firstChild).toBeNull();
  });

  it("returns null when status is 'error'", () => {
    mockHook(makeState({ status: "error" }));
    const { container } = renderWithProviders(<MLWarmupScreen />);
    expect(container.firstChild).toBeNull();
  });

  it("shows simulation mode button after threshold seconds when prop is provided", () => {
    mockHook(makeState({ status: "connecting" }));
    const onSimulationMode = vi.fn();
    renderWithProviders(<MLWarmupScreen onSimulationMode={onSimulationMode} />);

    // Button must not be visible yet
    expect(
      screen.queryByRole("button", { name: /browse app while loading/i })
    ).not.toBeInTheDocument();

    // Advance past SIMULATION_MODE_THRESHOLD_S (10s) — wrap in act so React flushes the state update.
    act(() => {
      vi.advanceTimersByTime(10_000);
    });

    // Button should now appear (synchronous query — no findBy needed with fake timers).
    expect(
      screen.getByRole("button", { name: /browse app while loading/i })
    ).toBeInTheDocument();
  });

  it("does not show simulation mode button after threshold when prop is absent", () => {
    mockHook(makeState({ status: "connecting" }));
    renderWithProviders(<MLWarmupScreen />);

    act(() => {
      vi.advanceTimersByTime(10_000);
    });

    // Still no button — prop was not provided
    expect(
      screen.queryByRole("button", { name: /browse app while loading/i })
    ).not.toBeInTheDocument();
  });

  it("calls onSimulationMode when the button is clicked", () => {
    mockHook(makeState({ status: "connecting" }));
    const onSimulationMode = vi.fn();
    renderWithProviders(<MLWarmupScreen onSimulationMode={onSimulationMode} />);

    act(() => {
      vi.advanceTimersByTime(10_000);
    });

    const button = screen.getByRole("button", { name: /browse app while loading/i });
    fireEvent.click(button);

    expect(onSimulationMode).toHaveBeenCalledOnce();
  });
});
