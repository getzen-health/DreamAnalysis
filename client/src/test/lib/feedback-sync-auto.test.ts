import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock supabase-browser
vi.mock("@/lib/supabase-browser", () => ({
  getSupabase: vi.fn(() => Promise.resolve(null)),
}));

vi.mock("@/lib/ml-api", () => ({
  getMLApiUrl: vi.fn(() => "http://localhost:8080"),
}));

// Mock fetch globally
const originalFetch = globalThis.fetch;

import {
  triggerRetrainCheck,
  syncOnStartup,
  getRetrainingStatus,
} from "@/lib/feedback-sync";

const CORRECTION_COUNT_KEY = "ndw_correction_count";
const LAST_SYNC_KEY = "ndw_last_training_sync";
const LAST_RETRAINED_KEY = "ndw_last_retrained";

describe("feedback-sync auto-retrain triggers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true, json: () => ({}) });
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  describe("triggerRetrainCheck", () => {
    it("does NOT fire on corrections 1-9", async () => {
      for (let i = 1; i <= 9; i++) {
        await triggerRetrainCheck("user1");
      }
      // fetch should not have been called for training/sync
      const calls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls;
      const syncCalls = calls.filter(
        (c: unknown[]) => typeof c[0] === "string" && c[0].includes("/training/sync/"),
      );
      expect(syncCalls).toHaveLength(0);
    });

    it("fires on the 10th correction", async () => {
      for (let i = 1; i <= 10; i++) {
        await triggerRetrainCheck("user1");
      }
      const calls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls;
      const syncCalls = calls.filter(
        (c: unknown[]) => typeof c[0] === "string" && c[0].includes("/training/sync/"),
      );
      expect(syncCalls).toHaveLength(1);
      expect(syncCalls[0][0]).toBe("http://localhost:8080/training/sync/user1");
    });

    it("fires again on the 20th correction", async () => {
      for (let i = 1; i <= 20; i++) {
        await triggerRetrainCheck("user1");
      }
      const calls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls;
      const syncCalls = calls.filter(
        (c: unknown[]) => typeof c[0] === "string" && c[0].includes("/training/sync/"),
      );
      expect(syncCalls).toHaveLength(2);
    });

    it("includes userId in the endpoint", async () => {
      localStorage.setItem(CORRECTION_COUNT_KEY, "9");
      await triggerRetrainCheck("abc-123");

      const calls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls;
      const syncCalls = calls.filter(
        (c: unknown[]) => typeof c[0] === "string" && c[0].includes("/training/sync/"),
      );
      expect(syncCalls).toHaveLength(1);
      expect(syncCalls[0][0]).toContain("abc-123");
    });
  });

  describe("syncOnStartup", () => {
    it("fires on first call of the day", async () => {
      await syncOnStartup("user1");

      const calls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls;
      const syncCalls = calls.filter(
        (c: unknown[]) => typeof c[0] === "string" && c[0].includes("/training/sync/"),
      );
      expect(syncCalls).toHaveLength(1);
    });

    it("does NOT fire on second call same day", async () => {
      await syncOnStartup("user1");
      await syncOnStartup("user1");

      const calls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls;
      const syncCalls = calls.filter(
        (c: unknown[]) => typeof c[0] === "string" && c[0].includes("/training/sync/"),
      );
      expect(syncCalls).toHaveLength(1);
    });

    it("sets lastSync date in localStorage", async () => {
      await syncOnStartup("user1");

      const stored = localStorage.getItem(LAST_SYNC_KEY);
      const today = new Date().toISOString().split("T")[0];
      expect(stored).toBe(today);
    });

    it("fires again on a new day", async () => {
      // Simulate yesterday
      localStorage.setItem(LAST_SYNC_KEY, "2020-01-01");
      await syncOnStartup("user1");

      const calls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls;
      const syncCalls = calls.filter(
        (c: unknown[]) => typeof c[0] === "string" && c[0].includes("/training/sync/"),
      );
      expect(syncCalls).toHaveLength(1);
    });
  });

  describe("getRetrainingStatus", () => {
    it("returns zero count and null lastRetrained when fresh", () => {
      const status = getRetrainingStatus();
      expect(status.correctionsCount).toBe(0);
      expect(status.lastRetrained).toBeNull();
      expect(status.nextRetrainAt).toBe(10);
    });

    it("returns correct count after corrections", () => {
      localStorage.setItem(CORRECTION_COUNT_KEY, "7");
      const status = getRetrainingStatus();
      expect(status.correctionsCount).toBe(7);
      expect(status.nextRetrainAt).toBe(10);
    });

    it("returns correct nextRetrainAt at boundary", () => {
      localStorage.setItem(CORRECTION_COUNT_KEY, "10");
      const status = getRetrainingStatus();
      expect(status.correctionsCount).toBe(10);
      expect(status.nextRetrainAt).toBe(20);
    });

    it("returns lastRetrained when set", () => {
      localStorage.setItem(LAST_RETRAINED_KEY, "2026-03-23T12:00:00Z");
      const status = getRetrainingStatus();
      expect(status.lastRetrained).toBe("2026-03-23T12:00:00Z");
    });
  });

  describe("offline resilience", () => {
    it("triggerRetrainCheck fails silently when fetch throws", async () => {
      localStorage.setItem(CORRECTION_COUNT_KEY, "9");
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("offline"));

      // Should not throw
      await triggerRetrainCheck("user1");
      expect(true).toBe(true);
    });

    it("syncOnStartup fails silently when fetch throws", async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("offline"));

      // Should not throw
      await syncOnStartup("user1");
      // Should NOT set the last-sync date on failure, so it retries next time
      const stored = localStorage.getItem(LAST_SYNC_KEY);
      expect(stored).toBeNull();
    });
  });
});
