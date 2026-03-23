import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";

import { useSyncStatus } from "@/hooks/use-sync-status";

beforeEach(() => {
  localStorage.clear();
  vi.clearAllMocks();
  Object.defineProperty(navigator, "onLine", { value: true, writable: true, configurable: true });
});

afterEach(() => {
  localStorage.clear();
});

describe("useSyncStatus", () => {
  it("returns synced status when queue is empty and online", () => {
    const { result } = renderHook(() => useSyncStatus());
    expect(result.current.state).toBe("synced");
    expect(result.current.pendingCount).toBe(0);
    expect(result.current.displayText).toBe("Synced");
  });

  it("returns pending status when queue has items", () => {
    localStorage.setItem(
      "ndw_sync_queue",
      JSON.stringify([
        { id: "w1", table: "mood_logs", operation: "insert", data: {}, timestamp: Date.now() },
        { id: "w2", table: "mood_logs", operation: "insert", data: {}, timestamp: Date.now() },
      ])
    );

    const { result } = renderHook(() => useSyncStatus());
    expect(result.current.state).toBe("pending");
    expect(result.current.pendingCount).toBe(2);
    expect(result.current.displayText).toBe("Pending 2 items");
  });

  it("returns offline status when navigator is offline", () => {
    Object.defineProperty(navigator, "onLine", { value: false, writable: true, configurable: true });

    const { result } = renderHook(() => useSyncStatus());
    expect(result.current.state).toBe("offline");
    expect(result.current.displayText).toBe("Offline — data saved locally");
  });

  it("exposes enqueue method to add items to the queue", () => {
    const { result } = renderHook(() => useSyncStatus());

    act(() => {
      result.current.enqueue({
        id: "w1",
        table: "mood_logs",
        operation: "insert",
        data: { mood: 7 },
        timestamp: Date.now(),
      });
    });

    expect(result.current.pendingCount).toBe(1);
    expect(result.current.state).toBe("pending");
  });
});
