import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock supabase-browser before importing feedback-sync
const mockInsert = vi.fn();
const mockSelect = vi.fn();
const mockEq = vi.fn();

const mockFrom = vi.fn((table: string) => ({
  insert: mockInsert,
  select: (...args: unknown[]) => {
    mockSelect(...args);
    return {
      eq: (...eqArgs: unknown[]) => {
        mockEq(...eqArgs);
        return Promise.resolve({ count: 42 });
      },
    };
  },
}));

const mockSupabase = { from: mockFrom };

vi.mock("@/lib/supabase-browser", () => ({
  getSupabase: vi.fn(() => Promise.resolve(mockSupabase)),
}));

vi.mock("@/lib/ml-api", () => ({
  getMLApiUrl: vi.fn(() => "http://localhost:8080"),
}));

// Mock fetch globally
const originalFetch = globalThis.fetch;

import {
  recordCorrection,
  flushPendingCorrections,
  getCorrectionCount,
  type CorrectionRecord,
} from "@/lib/feedback-sync";

const QUEUE_KEY = "ndw_pending_corrections";

function makeCorrectionRecord(overrides?: Partial<CorrectionRecord>): CorrectionRecord {
  return {
    userId: "test-user",
    predictedEmotion: "happy",
    correctedEmotion: "sad",
    source: "manual",
    ...overrides,
  };
}

describe("feedback-sync", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    mockInsert.mockResolvedValue({ error: null });
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true, json: () => ({}) });
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  describe("recordCorrection", () => {
    it("calls supabase.from('user_feedback').insert with correct data", async () => {
      const record = makeCorrectionRecord();
      await recordCorrection(record);

      expect(mockFrom).toHaveBeenCalledWith("user_feedback");
      expect(mockInsert).toHaveBeenCalledWith({
        user_id: "test-user",
        predicted_emotion: "happy",
        corrected_emotion: "sad",
        source: "manual",
        confidence: null,
        features: null,
        session_id: null,
      });
    });

    it("calls ML backend /api/feedback", async () => {
      const record = makeCorrectionRecord();
      await recordCorrection(record);

      expect(globalThis.fetch).toHaveBeenCalledWith(
        "http://localhost:8080/api/feedback",
        expect.objectContaining({
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: "test-user",
            predicted_label: "happy",
            correct_label: "sad",
          }),
        }),
      );
    });

    it("queues correction in localStorage if Supabase fails", async () => {
      mockInsert.mockResolvedValue({ error: { message: "network error" } });
      const record = makeCorrectionRecord();
      await recordCorrection(record);

      const queued = JSON.parse(localStorage.getItem(QUEUE_KEY) || "[]");
      expect(queued).toHaveLength(1);
      expect(queued[0].userId).toBe("test-user");
      expect(queued[0].correctedEmotion).toBe("sad");
      expect(queued[0].queuedAt).toBeDefined();
    });
  });

  describe("flushPendingCorrections", () => {
    it("sends queued items to Supabase and clears them", async () => {
      // Pre-populate queue
      const queued = [
        { ...makeCorrectionRecord(), queuedAt: "2026-01-01T00:00:00Z" },
        { ...makeCorrectionRecord({ correctedEmotion: "angry" }), queuedAt: "2026-01-01T01:00:00Z" },
      ];
      localStorage.setItem(QUEUE_KEY, JSON.stringify(queued));
      mockInsert.mockResolvedValue({ error: null });

      const flushed = await flushPendingCorrections();

      expect(flushed).toBe(2);
      expect(mockInsert).toHaveBeenCalledTimes(2);
      const remaining = JSON.parse(localStorage.getItem(QUEUE_KEY) || "[]");
      expect(remaining).toHaveLength(0);
    });

    it("handles partial failures — keeps failed items in queue", async () => {
      const queued = [
        { ...makeCorrectionRecord(), queuedAt: "2026-01-01T00:00:00Z" },
        { ...makeCorrectionRecord({ correctedEmotion: "angry" }), queuedAt: "2026-01-01T01:00:00Z" },
      ];
      localStorage.setItem(QUEUE_KEY, JSON.stringify(queued));

      // First insert succeeds, second fails
      mockInsert
        .mockResolvedValueOnce({ error: null })
        .mockResolvedValueOnce({ error: { message: "fail" } });

      const flushed = await flushPendingCorrections();

      expect(flushed).toBe(1);
      const remaining = JSON.parse(localStorage.getItem(QUEUE_KEY) || "[]");
      expect(remaining).toHaveLength(1);
      expect(remaining[0].correctedEmotion).toBe("angry");
    });

    it("returns 0 when queue is empty", async () => {
      const flushed = await flushPendingCorrections();
      expect(flushed).toBe(0);
      expect(mockInsert).not.toHaveBeenCalled();
    });
  });

  describe("getCorrectionCount", () => {
    it("returns count from Supabase", async () => {
      const count = await getCorrectionCount("test-user");
      expect(count).toBe(42);
      expect(mockFrom).toHaveBeenCalledWith("user_feedback");
    });
  });
});
