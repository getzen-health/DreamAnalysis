import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { MLConnectionProvider, useMLConnection } from "@/hooks/use-ml-connection";
import type { ReactNode } from "react";

// ── mock pingBackend ──────────────────────────────────────────────────────────
// vi.hoisted() runs before module imports, making mockPingBackend available
// when vi.mock()'s hoisted factory executes.
const { mockPingBackend } = vi.hoisted(() => ({
  mockPingBackend: vi.fn<() => Promise<boolean>>(),
}));

vi.mock("@/lib/ml-api", () => ({ pingBackend: mockPingBackend }));

// ── helpers ───────────────────────────────────────────────────────────────────

function wrapper({ children }: { children: ReactNode }) {
  return <MLConnectionProvider>{children}</MLConnectionProvider>;
}

// ── tests ─────────────────────────────────────────────────────────────────────

describe("useMLConnection", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    mockPingBackend.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts in 'connecting' state", () => {
    // Ping never resolves — keep it pending for the duration of this test
    mockPingBackend.mockReturnValue(new Promise(() => {}));

    const { result } = renderHook(() => useMLConnection(), { wrapper });

    expect(result.current.status).toBe("connecting");
    expect(result.current.latencyMs).toBeNull();
    expect(result.current.warmupProgress).toBe(0);
    expect(result.current.retryCount).toBe(0);
  });

  it("transitions to 'ready' when ping returns true", async () => {
    mockPingBackend.mockResolvedValue(true);

    const { result } = renderHook(() => useMLConnection(), { wrapper });

    // Advance just enough for the initial ping to be invoked and resolved.
    // Using advanceTimersByTimeAsync(0) flushes microtasks (the resolved Promise)
    // without draining the repeating progress-bar interval to infinity.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.status).toBe("ready");
    expect(result.current.latencyMs).not.toBeNull();
    expect(result.current.warmupProgress).toBe(100);
  });

  it("transitions to 'error' after 3 consecutive false pings", async () => {
    mockPingBackend.mockResolvedValue(false);

    const { result } = renderHook(() => useMLConnection(), { wrapper });

    // First ping fires immediately on mount → failure #1
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // After first failure the status must be 'warming' (not yet 'error')
    expect(result.current.status).toBe("warming");

    // Second ping fires after 5 s
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });

    // Third ping fires after another 5 s
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });

    expect(result.current.status).toBe("error");
    expect(result.current.retryCount).toBeGreaterThanOrEqual(3);
  });

  it("reconnect() resets to 'connecting' and then 'ready'", async () => {
    // First ping fails; every subsequent ping succeeds
    mockPingBackend
      .mockResolvedValueOnce(false)
      .mockResolvedValue(true);

    const { result } = renderHook(() => useMLConnection(), { wrapper });

    // Let the first ping fail
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Trigger reconnect — must be synchronous state reset then 'connecting'
    act(() => {
      result.current.reconnect();
    });

    expect(result.current.status).toBe("connecting");

    // Allow reconnect's immediate ping to resolve (it returns true)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.status).toBe("ready");
  });
});
