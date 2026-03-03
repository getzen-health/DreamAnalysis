import { describe, it, expect, vi, afterEach } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import * as connectionHook from "@/hooks/use-ml-connection";
import type { MLConnectionState } from "@/hooks/use-ml-connection";
import { SimulationModeBanner } from "@/components/simulation-mode-banner";

// ── helpers ───────────────────────────────────────────────────────────────────

function makeState(overrides: Partial<MLConnectionState> = {}): MLConnectionState {
  return {
    status: "ready",
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

describe("SimulationModeBanner", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders banner when status is 'error'", () => {
    mockHook(makeState({ status: "error" }));
    renderWithProviders(<SimulationModeBanner />);
    expect(
      screen.getByText(/ML backend unreachable — running in simulation mode/i)
    ).toBeInTheDocument();
  });

  it("returns null when status is 'ready'", () => {
    mockHook(makeState({ status: "ready" }));
    const { container } = renderWithProviders(<SimulationModeBanner />);
    expect(container.firstChild).toBeNull();
  });

  it("clicking Reconnect calls reconnect()", () => {
    const reconnect = vi.fn();
    mockHook(makeState({ status: "error", reconnect }));
    renderWithProviders(<SimulationModeBanner />);

    const button = screen.getByRole("button", { name: /reconnect/i });
    fireEvent.click(button);

    expect(reconnect).toHaveBeenCalledOnce();
  });
});
