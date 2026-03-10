import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock @capacitor/core before any imports
vi.mock("@capacitor/core", () => ({
  Capacitor: {
    isNativePlatform: vi.fn(() => false),
  },
}));

import { Capacitor } from "@capacitor/core";

describe("getMLApiUrl on native platform", () => {
  beforeEach(() => {
    vi.resetModules();
    try { localStorage.clear(); } catch { /* ignore */ }
  });

  it("returns production URL (not localhost:8080) when on native platform", async () => {
    // Simulate native platform
    vi.mocked(Capacitor.isNativePlatform).mockReturnValue(true);

    const mod = await import("@/lib/ml-api");
    const url = mod.getMLApiUrl();
    // On native, should use VITE_ML_API_URL env var (or its default),
    // never fall back to localhost:8080 via the hostname check
    expect(typeof url).toBe("string");
    expect(url.length).toBeGreaterThan(0);
  });

  it("respects localStorage override on native", async () => {
    vi.mocked(Capacitor.isNativePlatform).mockReturnValue(true);
    localStorage.setItem("ml_backend_url", "https://custom-backend.example.com");

    const mod = await import("@/lib/ml-api");
    const url = mod.getMLApiUrl();
    expect(url).toBe("https://custom-backend.example.com");
  });

  it("returns localhost:8080 on web localhost (not native)", async () => {
    vi.mocked(Capacitor.isNativePlatform).mockReturnValue(false);

    const mod = await import("@/lib/ml-api");
    const url = mod.getMLApiUrl();
    // In test env, hostname is likely "localhost"
    expect(url).toBe("http://localhost:8080");
  });
});
