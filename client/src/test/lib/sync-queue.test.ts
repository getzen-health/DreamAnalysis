import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

import {
  SyncQueue,
  SyncStatus,
  type QueuedWrite,
} from "@/lib/sync-queue";

// ── Setup / Teardown ────────────────────────────────────────────────────────

beforeEach(() => {
  localStorage.clear();
  vi.clearAllMocks();
  // Reset navigator.onLine to true
  Object.defineProperty(navigator, "onLine", { value: true, writable: true, configurable: true });
});

afterEach(() => {
  localStorage.clear();
});

// ── SyncQueue core ──────────────────────────────────────────────────────────

describe("SyncQueue", () => {
  it("starts with synced status and zero pending items", () => {
    const queue = new SyncQueue();
    const status = queue.getStatus();
    expect(status.state).toBe("synced");
    expect(status.pendingCount).toBe(0);
  });

  it("enqueues a failed write and updates status to pending", () => {
    const queue = new SyncQueue();
    const write: QueuedWrite = {
      id: "w1",
      table: "mood_logs",
      operation: "insert",
      data: { user_id: "u1", mood: 7 },
      timestamp: Date.now(),
    };

    queue.enqueue(write);
    const status = queue.getStatus();
    expect(status.state).toBe("pending");
    expect(status.pendingCount).toBe(1);
  });

  it("returns offline status when navigator.onLine is false", () => {
    Object.defineProperty(navigator, "onLine", { value: false, writable: true, configurable: true });
    const queue = new SyncQueue();
    queue.enqueue({
      id: "w1",
      table: "mood_logs",
      operation: "insert",
      data: { mood: 5 },
      timestamp: Date.now(),
    });
    const status = queue.getStatus();
    expect(status.state).toBe("offline");
    expect(status.pendingCount).toBe(1);
  });

  it("persists queue to localStorage", () => {
    const queue = new SyncQueue();
    queue.enqueue({
      id: "w1",
      table: "mood_logs",
      operation: "insert",
      data: { mood: 5 },
      timestamp: Date.now(),
    });

    const raw = localStorage.getItem("ndw_sync_queue");
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw!);
    expect(parsed.length).toBe(1);
    expect(parsed[0].id).toBe("w1");
  });

  it("loads existing queue from localStorage on construction", () => {
    localStorage.setItem(
      "ndw_sync_queue",
      JSON.stringify([
        { id: "w1", table: "mood_logs", operation: "insert", data: {}, timestamp: Date.now() },
        { id: "w2", table: "voice_history", operation: "insert", data: {}, timestamp: Date.now() },
      ])
    );

    const queue = new SyncQueue();
    expect(queue.getStatus().pendingCount).toBe(2);
  });

  it("groups pending writes by table for batching", () => {
    const queue = new SyncQueue();
    queue.enqueue({ id: "w1", table: "mood_logs", operation: "insert", data: { mood: 5 }, timestamp: 1 });
    queue.enqueue({ id: "w2", table: "mood_logs", operation: "insert", data: { mood: 7 }, timestamp: 2 });
    queue.enqueue({ id: "w3", table: "voice_history", operation: "insert", data: { emotion: "happy" }, timestamp: 3 });

    const batches = queue.getBatches();
    expect(Object.keys(batches)).toContain("mood_logs");
    expect(Object.keys(batches)).toContain("voice_history");
    expect(batches["mood_logs"].length).toBe(2);
    expect(batches["voice_history"].length).toBe(1);
  });

  it("removes items after successful flush", () => {
    const queue = new SyncQueue();
    queue.enqueue({ id: "w1", table: "mood_logs", operation: "insert", data: { mood: 5 }, timestamp: 1 });
    queue.enqueue({ id: "w2", table: "mood_logs", operation: "insert", data: { mood: 7 }, timestamp: 2 });

    queue.markFlushed(["w1", "w2"]);

    expect(queue.getStatus().state).toBe("synced");
    expect(queue.getStatus().pendingCount).toBe(0);
    // localStorage should also be cleared
    const raw = JSON.parse(localStorage.getItem("ndw_sync_queue") || "[]");
    expect(raw.length).toBe(0);
  });

  it("only removes flushed items, keeps others", () => {
    const queue = new SyncQueue();
    queue.enqueue({ id: "w1", table: "mood_logs", operation: "insert", data: {}, timestamp: 1 });
    queue.enqueue({ id: "w2", table: "voice_history", operation: "insert", data: {}, timestamp: 2 });

    queue.markFlushed(["w1"]);

    expect(queue.getStatus().pendingCount).toBe(1);
  });

  it("clear removes all pending items", () => {
    const queue = new SyncQueue();
    queue.enqueue({ id: "w1", table: "mood_logs", operation: "insert", data: {}, timestamp: 1 });
    queue.enqueue({ id: "w2", table: "mood_logs", operation: "insert", data: {}, timestamp: 2 });

    queue.clear();
    expect(queue.getStatus().pendingCount).toBe(0);
    expect(queue.getStatus().state).toBe("synced");
  });
});

// ── SyncStatus display text ─────────────────────────────────────────────────

describe("SyncStatus display", () => {
  it("shows 'Synced' when no pending items and online", () => {
    const queue = new SyncQueue();
    expect(queue.getDisplayText()).toBe("Synced");
  });

  it("shows pending count when items are queued and online", () => {
    const queue = new SyncQueue();
    queue.enqueue({ id: "w1", table: "mood_logs", operation: "insert", data: {}, timestamp: 1 });
    queue.enqueue({ id: "w2", table: "mood_logs", operation: "insert", data: {}, timestamp: 2 });
    queue.enqueue({ id: "w3", table: "voice_history", operation: "insert", data: {}, timestamp: 3 });
    expect(queue.getDisplayText()).toBe("Pending 3 items");
  });

  it("shows offline message when offline with pending items", () => {
    Object.defineProperty(navigator, "onLine", { value: false, writable: true, configurable: true });
    const queue = new SyncQueue();
    queue.enqueue({ id: "w1", table: "mood_logs", operation: "insert", data: {}, timestamp: 1 });
    expect(queue.getDisplayText()).toBe("Offline — data saved locally");
  });

  it("shows offline message when offline with no pending items", () => {
    Object.defineProperty(navigator, "onLine", { value: false, writable: true, configurable: true });
    const queue = new SyncQueue();
    expect(queue.getDisplayText()).toBe("Offline — data saved locally");
  });
});
