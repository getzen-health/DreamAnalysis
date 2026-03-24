import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock ml-api before importing model-updater
vi.mock("@/lib/ml-api", () => ({
  getMLApiUrl: vi.fn(() => "http://localhost:8080"),
}));

// Mock IndexedDB using a simple in-memory store
const mockStore = new Map<string, { name: string; data: ArrayBuffer; updatedAt: number }>();

const mockIDBObjectStore = {
  put: vi.fn((record: { name: string; data: ArrayBuffer; updatedAt: number }) => {
    mockStore.set(record.name, record);
    const req = { onsuccess: null as (() => void) | null, onerror: null as (() => void) | null, result: undefined, error: null };
    setTimeout(() => req.onsuccess?.(), 0);
    return req;
  }),
  get: vi.fn((name: string) => {
    const result = mockStore.get(name);
    const req = { onsuccess: null as (() => void) | null, onerror: null as (() => void) | null, result, error: null };
    setTimeout(() => req.onsuccess?.(), 0);
    return req;
  }),
};

const mockTransaction = {
  objectStore: vi.fn(() => mockIDBObjectStore),
};

const mockDB = {
  transaction: vi.fn(() => mockTransaction),
  close: vi.fn(),
  objectStoreNames: { contains: vi.fn(() => true) },
  createObjectStore: vi.fn(),
};

// Mock indexedDB globally
const originalIndexedDB = globalThis.indexedDB;
Object.defineProperty(globalThis, "indexedDB", {
  value: {
    open: vi.fn(() => {
      const req = {
        onupgradeneeded: null as (() => void) | null,
        onsuccess: null as (() => void) | null,
        onerror: null as (() => void) | null,
        result: mockDB,
        error: null,
      };
      setTimeout(() => req.onsuccess?.(), 0);
      return req;
    }),
  },
  writable: true,
});

// Mock fetch
const originalFetch = globalThis.fetch;

import {
  checkAndUpdateModels,
  loadModelFromStorage,
  _getLocalVersions,
} from "@/lib/model-updater";

const MODEL_VERSION_KEY = "ndw_model_versions";

describe("model-updater", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockStore.clear();
    localStorage.clear();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  describe("checkAndUpdateModels", () => {
    it("fetches version endpoint from ML backend", async () => {
      const fetchMock = vi.fn()
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            eeg_available: false,
            voice_available: false,
            eeg_updated: null,
            voice_updated: null,
          }),
        });
      globalThis.fetch = fetchMock;

      await checkAndUpdateModels("user-123");

      expect(fetchMock).toHaveBeenCalledWith(
        "http://localhost:8080/api/training/model/user-123/version"
      );
    });

    it("downloads EEG ONNX when server version is newer", async () => {
      const fakeOnnxBlob = new Blob([new Uint8Array([1, 2, 3, 4])]);

      const fetchMock = vi.fn()
        // First call: version check
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            eeg_available: true,
            voice_available: false,
            eeg_updated: Date.now() / 1000,
            voice_updated: null,
          }),
        })
        // Second call: download EEG ONNX
        .mockResolvedValueOnce({
          ok: true,
          headers: { get: () => "application/octet-stream" },
          blob: () => Promise.resolve(fakeOnnxBlob),
        });
      globalThis.fetch = fetchMock;

      const result = await checkAndUpdateModels("user-123");

      expect(result.updated).toBe(true);
      expect(result.models).toContain("eeg");
      expect(fetchMock).toHaveBeenCalledTimes(2);
    });

    it("skips download when local version matches server", async () => {
      const serverTime = 1234567890;
      // Pre-set local versions to match server
      localStorage.setItem(
        MODEL_VERSION_KEY,
        JSON.stringify({
          eeg_updated: serverTime,
          voice_updated: null,
          lastChecked: new Date().toISOString(),
        })
      );

      const fetchMock = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          eeg_available: true,
          voice_available: false,
          eeg_updated: serverTime, // same as local
          voice_updated: null,
        }),
      });
      globalThis.fetch = fetchMock;

      const result = await checkAndUpdateModels("user-123");

      expect(result.updated).toBe(false);
      expect(result.models).toEqual([]);
      // Only 1 fetch call (version check), no download
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });

    it("handles offline gracefully (fetch fails)", async () => {
      const fetchMock = vi.fn().mockRejectedValueOnce(new Error("Network error"));
      globalThis.fetch = fetchMock;

      await expect(checkAndUpdateModels("user-123")).rejects.toThrow("Network error");
    });

    it("handles server returning not-ok response", async () => {
      const fetchMock = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 500,
      });
      globalThis.fetch = fetchMock;

      const result = await checkAndUpdateModels("user-123");

      expect(result.updated).toBe(false);
      expect(result.models).toEqual([]);
    });
  });

  describe("getLocalVersions", () => {
    it("returns empty object on first run", () => {
      const versions = _getLocalVersions();
      expect(versions.eeg_updated).toBeNull();
      expect(versions.voice_updated).toBeNull();
    });

    it("returns saved versions after update check", async () => {
      const serverTime = 9999999;
      const fetchMock = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          eeg_available: false,
          voice_available: false,
          eeg_updated: serverTime,
          voice_updated: null,
        }),
      });
      globalThis.fetch = fetchMock;

      await checkAndUpdateModels("user-123");

      const versions = _getLocalVersions();
      expect(versions.eeg_updated).toBe(serverTime);
      expect(versions.lastChecked).toBeTruthy();
    });
  });

  describe("loadModelFromStorage", () => {
    it("returns null when no model is stored", async () => {
      const result = await loadModelFromStorage("nonexistent.onnx");
      expect(result).toBeNull();
    });
  });
});
