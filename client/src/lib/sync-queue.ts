/**
 * sync-queue.ts — Queue failed Supabase writes for retry on reconnect.
 *
 * When a Supabase write fails (network down, server error), add it to the queue.
 * On reconnect (navigator.onLine event), flush the queue in batches grouped by table.
 *
 * Usage:
 *   import { SyncQueue } from "@/lib/sync-queue";
 *   const queue = new SyncQueue();
 *   queue.enqueue({ id, table, operation, data, timestamp });
 *   // On reconnect:
 *   const batches = queue.getBatches();
 *   // After successful flush:
 *   queue.markFlushed(["id1", "id2"]);
 */

// ── Types ───────────────────────────────────────────────────────────────────

export interface QueuedWrite {
  id: string;
  table: string;
  operation: "insert" | "update" | "delete";
  data: Record<string, unknown>;
  timestamp: number;
}

export type SyncState = "synced" | "pending" | "offline";

export interface SyncStatus {
  state: SyncState;
  pendingCount: number;
}

// ── Constants ───────────────────────────────────────────────────────────────

const STORAGE_KEY = "ndw_sync_queue";

// ── Helpers ─────────────────────────────────────────────────────────────────

function loadQueue(): QueuedWrite[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as QueuedWrite[];
  } catch {
    return [];
  }
}

function saveQueue(items: QueuedWrite[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  } catch {
    // localStorage full or unavailable
  }
}

// ── SyncQueue ───────────────────────────────────────────────────────────────

export class SyncQueue {
  private _items: QueuedWrite[];

  constructor() {
    this._items = loadQueue();
  }

  /** Add a failed write to the queue. */
  enqueue(write: QueuedWrite): void {
    this._items.push(write);
    saveQueue(this._items);
  }

  /** Get current sync status. */
  getStatus(): SyncStatus {
    const isOnline = typeof navigator !== "undefined" ? navigator.onLine : true;

    if (!isOnline) {
      return { state: "offline", pendingCount: this._items.length };
    }

    if (this._items.length > 0) {
      return { state: "pending", pendingCount: this._items.length };
    }

    return { state: "synced", pendingCount: 0 };
  }

  /** Get human-readable display text for the current status. */
  getDisplayText(): string {
    const status = this.getStatus();
    switch (status.state) {
      case "synced":
        return "Synced";
      case "pending":
        return `Pending ${status.pendingCount} items`;
      case "offline":
        return "Offline \u2014 data saved locally";
    }
  }

  /** Group pending writes by table for batch upload. */
  getBatches(): Record<string, QueuedWrite[]> {
    const batches: Record<string, QueuedWrite[]> = {};
    for (const item of this._items) {
      if (!batches[item.table]) {
        batches[item.table] = [];
      }
      batches[item.table].push(item);
    }
    return batches;
  }

  /** Remove successfully flushed items by ID. */
  markFlushed(ids: string[]): void {
    const idSet = new Set(ids);
    this._items = this._items.filter((item) => !idSet.has(item.id));
    saveQueue(this._items);
  }

  /** Remove all pending items. */
  clear(): void {
    this._items = [];
    saveQueue(this._items);
  }

  /** Get the raw list of pending items (read-only copy). */
  getPendingItems(): readonly QueuedWrite[] {
    return [...this._items];
  }
}
