/**
 * model-updater — Downloads per-user fine-tuned ONNX models from the ML backend.
 *
 * After the ML backend retrains a user's emotion model (from accumulated
 * corrections), this module:
 *   1. Checks if a newer model version is available on the server
 *   2. Downloads the ONNX binary
 *   3. Stores it in IndexedDB (persists across app restarts, unlike localStorage)
 *   4. Makes it available for on-device inference via loadModelFromStorage()
 *
 * Called on app startup (daily) and after retraining triggers.
 */

import { getMLApiUrl } from "./ml-api";

const MODEL_VERSION_KEY = "ndw_model_versions";
const DB_NAME = "ndw_models";
const DB_VERSION = 1;
const STORE_NAME = "models";

interface ModelVersions {
  eeg_updated: number | null;
  voice_updated: number | null;
  lastChecked: string;
}

interface ServerVersionResponse {
  eeg_available: boolean;
  voice_available: boolean;
  eeg_updated: number | null;
  voice_updated: number | null;
}

interface ModelRecord {
  name: string;
  data: ArrayBuffer;
  updatedAt: number;
}

/**
 * Check if the ML backend has a newer personalized model.
 * If yes, download it to IndexedDB for on-device use.
 *
 * Returns which models were updated (if any).
 */
export async function checkAndUpdateModels(
  userId: string,
): Promise<{ updated: boolean; models: string[] }> {
  const mlUrl = getMLApiUrl();
  const updated: string[] = [];

  // Check version — what the server has
  const resp = await fetch(`${mlUrl}/api/training/model/${userId}/version`);
  if (!resp.ok) {
    return { updated: false, models: [] };
  }

  const version: ServerVersionResponse = await resp.json();

  // Compare with what we have locally
  const local = getLocalVersions();

  // Download EEG model if server has a newer one
  if (
    version.eeg_available &&
    (!local.eeg_updated || version.eeg_updated! > local.eeg_updated)
  ) {
    const onnxResp = await fetch(
      `${mlUrl}/api/training/model/${userId}/eeg.onnx`,
    );
    if (onnxResp.ok) {
      const contentType = onnxResp.headers.get("content-type") || "";
      // Server returns JSON error when no model exists; only save binary
      if (contentType.includes("octet-stream")) {
        const blob = await onnxResp.blob();
        await saveModelToStorage("eeg_emotion_user.onnx", blob);
        updated.push("eeg");
      }
    }
  }

  // Download voice model if server has a newer one
  if (
    version.voice_available &&
    (!local.voice_updated || version.voice_updated! > local.voice_updated)
  ) {
    const onnxResp = await fetch(
      `${mlUrl}/api/training/model/${userId}/voice.onnx`,
    );
    if (onnxResp.ok) {
      const contentType = onnxResp.headers.get("content-type") || "";
      if (contentType.includes("octet-stream")) {
        const blob = await onnxResp.blob();
        await saveModelToStorage("voice_emotion_user.onnx", blob);
        updated.push("voice");
      }
    }
  }

  // Persist version info so we know what we have next time
  saveLocalVersions({
    eeg_updated: version.eeg_updated,
    voice_updated: version.voice_updated,
    lastChecked: new Date().toISOString(),
  });

  return { updated: updated.length > 0, models: updated };
}

// ---- IndexedDB model storage ------------------------------------------------

/**
 * Save an ONNX model binary to IndexedDB.
 * IndexedDB persists across app restarts (unlike localStorage which has size limits).
 */
async function saveModelToStorage(name: string, blob: Blob): Promise<void> {
  const db = await openModelDB();
  const tx = db.transaction(STORE_NAME, "readwrite");
  const store = tx.objectStore(STORE_NAME);

  const record: ModelRecord = {
    name,
    data: await blob.arrayBuffer(),
    updatedAt: Date.now(),
  };

  await new Promise<void>((resolve, reject) => {
    const req = store.put(record);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });

  db.close();
}

/**
 * Load an ONNX model binary from IndexedDB.
 * Returns the ArrayBuffer ready for onnxruntime-web, or null if not cached.
 */
export async function loadModelFromStorage(
  name: string,
): Promise<ArrayBuffer | null> {
  try {
    const db = await openModelDB();
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);

    const result = await new Promise<ModelRecord | undefined>(
      (resolve, reject) => {
        const req = store.get(name);
        req.onsuccess = () => resolve(req.result as ModelRecord | undefined);
        req.onerror = () => reject(req.error);
      },
    );

    db.close();
    return result?.data ?? null;
  } catch {
    return null;
  }
}

function openModelDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "name" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// ---- Local version tracking (localStorage) ----------------------------------

function getLocalVersions(): ModelVersions {
  try {
    const raw = localStorage.getItem(MODEL_VERSION_KEY);
    if (!raw) return { eeg_updated: null, voice_updated: null, lastChecked: "" };
    return JSON.parse(raw);
  } catch {
    return { eeg_updated: null, voice_updated: null, lastChecked: "" };
  }
}

function saveLocalVersions(v: ModelVersions): void {
  try {
    localStorage.setItem(MODEL_VERSION_KEY, JSON.stringify(v));
  } catch {
    // localStorage unavailable (private browsing, quota exceeded)
  }
}

// Re-export for testing
export { getLocalVersions as _getLocalVersions, saveModelToStorage as _saveModelToStorage };
