/**
 * offline-store.ts — IndexedDB queue for offline data capture.
 *
 * Two stores:
 *   dream_drafts   — dream journal entries written offline
 *   eeg_queue      — EEG session results + health metrics queued when server unreachable
 *
 * Call `syncAll(userId)` when the device comes back online.
 */

const DB_NAME = "neural-dream-offline";
const DB_VERSION = 2; // bumped to add eeg_queue store

// ─── Types ────────────────────────────────────────────────────────────────────

interface DreamDraft {
  id: string;
  dreamText: string;
  tags: string[];
  sleepQuality: number;
  sleepDuration: number;
  timestamp: number;
  synced: boolean;
}

export interface QueuedEEGSession {
  id: string;
  userId: string;
  sessionType: string;
  durationSeconds: number;
  emotions: Record<string, number>;  // emotion → probability
  stress: number;
  focus: number;
  sleepScore?: number;
  dreamCount?: number;
  timestamp: number;
  synced: boolean;
}

export interface QueuedHealthMetric {
  id: string;
  userId: string;
  metricType: string;  // "stress" | "focus" | "sleep" | "hrv" etc.
  value: number;
  timestamp: number;
  synced: boolean;
}

// ─── DB Init ──────────────────────────────────────────────────────────────────

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = request.result;
      const oldVersion = event.oldVersion;

      // v1 — dream_drafts
      if (!db.objectStoreNames.contains("dream_drafts")) {
        const store = db.createObjectStore("dream_drafts", { keyPath: "id" });
        store.createIndex("synced", "synced", { unique: false });
        store.createIndex("timestamp", "timestamp", { unique: false });
      }

      // v2 — eeg_queue
      if (oldVersion < 2) {
        if (!db.objectStoreNames.contains("eeg_queue")) {
          const eegStore = db.createObjectStore("eeg_queue", { keyPath: "id" });
          eegStore.createIndex("synced", "synced", { unique: false });
          eegStore.createIndex("userId", "userId", { unique: false });
          eegStore.createIndex("timestamp", "timestamp", { unique: false });
        }
        if (!db.objectStoreNames.contains("health_queue")) {
          const healthStore = db.createObjectStore("health_queue", { keyPath: "id" });
          healthStore.createIndex("synced", "synced", { unique: false });
          healthStore.createIndex("userId", "userId", { unique: false });
        }
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

// ─── Dream Drafts (unchanged) ─────────────────────────────────────────────────

export async function saveDreamDraft(draft: DreamDraft): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("dream_drafts", "readwrite");
    tx.objectStore("dream_drafts").put(draft);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function getDreamDrafts(): Promise<DreamDraft[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("dream_drafts", "readonly");
    const request = tx.objectStore("dream_drafts").index("timestamp").getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function getUnsyncedDrafts(): Promise<DreamDraft[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("dream_drafts", "readonly");
    const request = tx.objectStore("dream_drafts").index("synced").getAll(0 as unknown as IDBValidKey);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function markDraftSynced(id: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("dream_drafts", "readwrite");
    const store = tx.objectStore("dream_drafts");
    const getReq = store.get(id);
    getReq.onsuccess = () => {
      if (getReq.result) store.put({ ...getReq.result, synced: true });
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function deleteDreamDraft(id: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("dream_drafts", "readwrite");
    tx.objectStore("dream_drafts").delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// ─── EEG Session Queue ────────────────────────────────────────────────────────

export async function queueEEGSession(session: Omit<QueuedEEGSession, "synced">): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("eeg_queue", "readwrite");
    tx.objectStore("eeg_queue").put({ ...session, synced: false });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function getUnsyncedEEGSessions(): Promise<QueuedEEGSession[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("eeg_queue", "readonly");
    const request = tx.objectStore("eeg_queue").index("synced").getAll(0 as unknown as IDBValidKey);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function markEEGSessionSynced(id: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("eeg_queue", "readwrite");
    const store = tx.objectStore("eeg_queue");
    const req = store.get(id);
    req.onsuccess = () => { if (req.result) store.put({ ...req.result, synced: true }); };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// ─── Health Metric Queue ──────────────────────────────────────────────────────

export async function queueHealthMetric(metric: Omit<QueuedHealthMetric, "synced">): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("health_queue", "readwrite");
    tx.objectStore("health_queue").put({ ...metric, synced: false });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function getUnsyncedHealthMetrics(): Promise<QueuedHealthMetric[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("health_queue", "readonly");
    const request = tx.objectStore("health_queue").index("synced").getAll(0 as unknown as IDBValidKey);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function markHealthMetricSynced(id: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("health_queue", "readwrite");
    const store = tx.objectStore("health_queue");
    const req = store.get(id);
    req.onsuccess = () => { if (req.result) store.put({ ...req.result, synced: true }); };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// ─── Sync All ─────────────────────────────────────────────────────────────────

export interface SyncResult {
  dreams: number;
  sessions: number;
  metrics: number;
}

export async function syncAll(userId: string): Promise<SyncResult> {
  const result: SyncResult = { dreams: 0, sessions: 0, metrics: 0 };

  // Sync dream drafts
  const unsyncedDreams = await getUnsyncedDrafts();
  for (const draft of unsyncedDreams) {
    try {
      const res = await fetch("/api/dream-analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          dreamText: draft.dreamText,
          userId,
          tags: draft.tags,
          sleepQuality: draft.sleepQuality,
          sleepDuration: draft.sleepDuration,
        }),
      });
      if (res.ok) { await markDraftSynced(draft.id); result.dreams++; }
    } catch { /* still offline */ }
  }

  // Sync EEG sessions
  const unsyncedSessions = await getUnsyncedEEGSessions();
  for (const session of unsyncedSessions) {
    try {
      const res = await fetch("/api/health-metrics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          userId,
          sessionType: session.sessionType,
          durationSeconds: session.durationSeconds,
          stressIndex: session.stress,
          focusIndex: session.focus,
          sleepScore: session.sleepScore,
          dreamCount: session.dreamCount,
          timestamp: new Date(session.timestamp).toISOString(),
        }),
      });
      if (res.ok) { await markEEGSessionSynced(session.id); result.sessions++; }
    } catch { /* still offline */ }
  }

  // Sync health metrics
  const unsyncedMetrics = await getUnsyncedHealthMetrics();
  for (const metric of unsyncedMetrics) {
    try {
      const res = await fetch("/api/health-metrics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          userId,
          metricType: metric.metricType,
          value: metric.value,
          timestamp: new Date(metric.timestamp).toISOString(),
        }),
      });
      if (res.ok) { await markHealthMetricSynced(metric.id); result.metrics++; }
    } catch { /* still offline */ }
  }

  return result;
}

// Legacy export (backward compat)
export async function syncDrafts(userId: string): Promise<number> {
  const r = await syncAll(userId);
  return r.dreams;
}
