/**
 * offline-store.ts — IndexedDB queue for offline data capture.
 *
 * Stores:
 *   dream_drafts      — dream journal entries written offline
 *   eeg_queue         — EEG session results + health metrics queued when server unreachable
 *   voice_emotion_queue — voice analysis emotion results pending sync
 *   food_log_queue    — food log entries pending sync
 *
 * Call `syncAll(userId)` when the device comes back online.
 * Call `registerBackgroundSync()` to request a Background Sync event from the SW.
 */

import { apiRequest } from "@/lib/queryClient";

const DB_NAME = "neural-dream-offline";
const DB_VERSION = 3; // v3 adds voice_emotion_queue + food_log_queue

// ─── Types ────────────────────────────────────────────────────────────────────

export interface QueuedVoiceEmotion {
  id: string;
  userId: string;
  emotion: string;
  valence: number;
  arousal: number;
  confidence: number;
  probabilities: Record<string, number>;
  timestamp: number;
  synced: boolean;
}

export interface QueuedFoodLog {
  id: string;
  userId: string;
  foodName: string;
  calories?: number;
  mealType: string;  // "breakfast" | "lunch" | "dinner" | "snack"
  timestamp: number;
  synced: boolean;
}

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

      // v2 — eeg_queue + health_queue
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

      // v3 — voice_emotion_queue + food_log_queue
      if (oldVersion < 3) {
        if (!db.objectStoreNames.contains("voice_emotion_queue")) {
          const voiceStore = db.createObjectStore("voice_emotion_queue", { keyPath: "id" });
          voiceStore.createIndex("synced", "synced", { unique: false });
          voiceStore.createIndex("userId", "userId", { unique: false });
          voiceStore.createIndex("timestamp", "timestamp", { unique: false });
        }
        if (!db.objectStoreNames.contains("food_log_queue")) {
          const foodStore = db.createObjectStore("food_log_queue", { keyPath: "id" });
          foodStore.createIndex("synced", "synced", { unique: false });
          foodStore.createIndex("userId", "userId", { unique: false });
          foodStore.createIndex("timestamp", "timestamp", { unique: false });
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

// ─── Voice Emotion Queue ──────────────────────────────────────────────────────

export async function queueVoiceEmotion(
  entry: Omit<QueuedVoiceEmotion, "synced">
): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("voice_emotion_queue", "readwrite");
    tx.objectStore("voice_emotion_queue").put({ ...entry, synced: false });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function getUnsyncedVoiceEmotions(): Promise<QueuedVoiceEmotion[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("voice_emotion_queue", "readonly");
    const request = tx
      .objectStore("voice_emotion_queue")
      .index("synced")
      .getAll(0 as unknown as IDBValidKey);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function markVoiceEmotionSynced(id: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("voice_emotion_queue", "readwrite");
    const store = tx.objectStore("voice_emotion_queue");
    const req = store.get(id);
    req.onsuccess = () => {
      if (req.result) store.put({ ...req.result, synced: true });
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// ─── Food Log Queue ───────────────────────────────────────────────────────────

export async function queueFoodLog(
  entry: Omit<QueuedFoodLog, "synced">
): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("food_log_queue", "readwrite");
    tx.objectStore("food_log_queue").put({ ...entry, synced: false });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function getUnsyncedFoodLogs(): Promise<QueuedFoodLog[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("food_log_queue", "readonly");
    const request = tx
      .objectStore("food_log_queue")
      .index("synced")
      .getAll(0 as unknown as IDBValidKey);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function markFoodLogSynced(id: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("food_log_queue", "readwrite");
    const store = tx.objectStore("food_log_queue");
    const req = store.get(id);
    req.onsuccess = () => {
      if (req.result) store.put({ ...req.result, synced: true });
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// ─── Queue Size Helper ────────────────────────────────────────────────────────

/** Returns the total number of unsynced items across all queues. */
export async function getOfflineQueueSize(): Promise<number> {
  const [dreams, sessions, metrics, voice, food] = await Promise.all([
    getUnsyncedDrafts(),
    getUnsyncedEEGSessions(),
    getUnsyncedHealthMetrics(),
    getUnsyncedVoiceEmotions(),
    getUnsyncedFoodLogs(),
  ]);
  return (
    dreams.length +
    sessions.length +
    metrics.length +
    voice.length +
    food.length
  );
}

// ─── Background Sync Registration ─────────────────────────────────────────────

/**
 * Registers a Background Sync event with the service worker.
 * When the device comes back online the SW fires "sync-offline-data",
 * which posts SW_BACKGROUND_SYNC to all clients, triggering syncAll().
 *
 * Call this whenever data is written to any offline queue.
 */
export async function registerBackgroundSync(): Promise<void> {
  if (!("serviceWorker" in navigator) || !("SyncManager" in window)) return;
  try {
    const reg = await navigator.serviceWorker.ready;
    // @ts-expect-error — SyncManager types are not in all lib.dom versions
    await reg.sync.register("sync-offline-data");
  } catch {
    // Silently fail — background sync is a progressive enhancement
  }
}

// ─── Sync All ─────────────────────────────────────────────────────────────────

export interface SyncResult {
  dreams: number;
  sessions: number;
  metrics: number;
  voice: number;
  food: number;
}

export async function syncAll(userId: string): Promise<SyncResult> {
  const result: SyncResult = { dreams: 0, sessions: 0, metrics: 0, voice: 0, food: 0 };

  // Sync dream drafts
  const unsyncedDreams = await getUnsyncedDrafts();
  for (const draft of unsyncedDreams) {
    try {
      const res = await apiRequest("POST", "/api/dream-analysis", {
        dreamText: draft.dreamText,
        tags: draft.tags,
        sleepQuality: draft.sleepQuality,
        sleepDuration: draft.sleepDuration,
      });
      if (res.ok) { await markDraftSynced(draft.id); result.dreams++; }
    } catch { /* still offline */ }
  }

  // Sync EEG sessions
  const unsyncedSessions = await getUnsyncedEEGSessions();
  for (const session of unsyncedSessions) {
    try {
      const res = await apiRequest("POST", "/api/health-metrics", {
        userId,
        sessionType: session.sessionType,
        durationSeconds: session.durationSeconds,
        stressIndex: session.stress,
        focusIndex: session.focus,
        sleepScore: session.sleepScore,
        dreamCount: session.dreamCount,
        timestamp: new Date(session.timestamp).toISOString(),
      });
      if (res.ok) { await markEEGSessionSynced(session.id); result.sessions++; }
    } catch { /* still offline */ }
  }

  // Sync health metrics
  const unsyncedMetrics = await getUnsyncedHealthMetrics();
  for (const metric of unsyncedMetrics) {
    try {
      const res = await apiRequest("POST", "/api/health-metrics", {
        userId,
        metricType: metric.metricType,
        value: metric.value,
        timestamp: new Date(metric.timestamp).toISOString(),
      });
      if (res.ok) { await markHealthMetricSynced(metric.id); result.metrics++; }
    } catch { /* still offline */ }
  }

  // Sync voice emotion results
  const unsyncedVoice = await getUnsyncedVoiceEmotions();
  for (const entry of unsyncedVoice) {
    try {
      const res = await apiRequest("POST", "/api/voice-emotion", {
        emotion: entry.emotion,
        valence: entry.valence,
        arousal: entry.arousal,
        confidence: entry.confidence,
        probabilities: entry.probabilities,
        timestamp: new Date(entry.timestamp).toISOString(),
      });
      if (res.ok) { await markVoiceEmotionSynced(entry.id); result.voice++; }
    } catch { /* still offline */ }
  }

  // Sync food log entries
  const unsyncedFood = await getUnsyncedFoodLogs();
  for (const entry of unsyncedFood) {
    try {
      const res = await apiRequest("POST", "/api/food-log", {
        foodName: entry.foodName,
        calories: entry.calories,
        mealType: entry.mealType,
        timestamp: new Date(entry.timestamp).toISOString(),
      });
      if (res.ok) { await markFoodLogSynced(entry.id); result.food++; }
    } catch { /* still offline */ }
  }

  return result;
}

// Legacy export (backward compat)
export async function syncDrafts(userId: string): Promise<number> {
  const r = await syncAll(userId);
  return r.dreams;
}
