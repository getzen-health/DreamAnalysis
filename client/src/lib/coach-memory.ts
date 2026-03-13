/**
 * Persistent coach memory via IndexedDB.
 *
 * Stores user-confirmed context strings that the AI coach will receive on
 * every request so it can give personalised, longitudinal advice without
 * requiring the user to repeat themselves.
 */

const DB_NAME = "ndw-coach-memories";
const STORE_NAME = "memories";
const DB_VERSION = 1;

export interface CoachMemory {
  id: string;
  text: string;
  createdAt: Date;
}

// ── Raw record as stored in IndexedDB ─────────────────────────────────────

interface RawMemory {
  id: string;
  text: string;
  createdAt: number; // stored as epoch ms
}

// ── Open / initialise DB ───────────────────────────────────────────────────

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);

    req.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "id" });
        store.createIndex("createdAt", "createdAt", { unique: false });
      }
    };

    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Persist a new memory string.  A random UUID is generated for the id.
 */
export async function addMemory(text: string): Promise<CoachMemory> {
  const memory: RawMemory = {
    id: crypto.randomUUID(),
    text: text.trim(),
    createdAt: Date.now(),
  };

  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const req = tx.objectStore(STORE_NAME).add(memory);
    req.onsuccess = () =>
      resolve({ ...memory, createdAt: new Date(memory.createdAt) });
    req.onerror = () => reject(req.error);
    tx.oncomplete = () => db.close();
  });
}

/**
 * Return all stored memories, sorted oldest-first.
 */
export async function getMemories(): Promise<CoachMemory[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const req = tx.objectStore(STORE_NAME).index("createdAt").getAll();
    req.onsuccess = () => {
      const raw: RawMemory[] = req.result ?? [];
      resolve(
        raw.map((r) => ({ id: r.id, text: r.text, createdAt: new Date(r.createdAt) }))
      );
    };
    req.onerror = () => reject(req.error);
    tx.oncomplete = () => db.close();
  });
}

/**
 * Delete a single memory by id.
 */
export async function deleteMemory(id: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const req = tx.objectStore(STORE_NAME).delete(id);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
    tx.oncomplete = () => db.close();
  });
}

/**
 * Wipe the entire memory store.
 */
export async function clearAll(): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const req = tx.objectStore(STORE_NAME).clear();
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
    tx.oncomplete = () => db.close();
  });
}
