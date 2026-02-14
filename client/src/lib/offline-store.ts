const DB_NAME = "neural-dream-offline";
const DB_VERSION = 1;

interface DreamDraft {
  id: string;
  dreamText: string;
  tags: string[];
  sleepQuality: number;
  sleepDuration: number;
  timestamp: number;
  synced: boolean;
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains("dream_drafts")) {
        const store = db.createObjectStore("dream_drafts", { keyPath: "id" });
        store.createIndex("synced", "synced", { unique: false });
        store.createIndex("timestamp", "timestamp", { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

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
      if (getReq.result) {
        store.put({ ...getReq.result, synced: true });
      }
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

export async function syncDrafts(userId: string): Promise<number> {
  const unsynced = await getUnsyncedDrafts();
  let synced = 0;

  for (const draft of unsynced) {
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
      if (res.ok) {
        await markDraftSynced(draft.id);
        synced++;
      }
    } catch {
      // Offline - will retry later
    }
  }

  return synced;
}
