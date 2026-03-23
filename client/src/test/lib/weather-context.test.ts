import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";

// We'll test the pure functions from weather-context.ts
import {
  mapWeatherCodeToContext,
  computeDaylightHours,
  buildMoodContext,
  getCachedWeather,
  setCachedWeather,
  type WeatherData,
  type WeatherMoodContext,
} from "@/lib/weather-context";

// ── Mock localStorage ─────────────────────────────────────────────────────

const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => { store[key] = value; }),
    removeItem: vi.fn((key: string) => { delete store[key]; }),
    clear: vi.fn(() => { store = {}; }),
  };
})();

Object.defineProperty(globalThis, "localStorage", { value: localStorageMock });

beforeEach(() => {
  localStorageMock.clear();
  vi.clearAllMocks();
});

afterEach(() => {
  localStorageMock.clear();
});

// ── Weather code mapping ──────────────────────────────────────────────────

describe("mapWeatherCodeToContext", () => {
  it("maps clear sky (0) to sunny", () => {
    const ctx = mapWeatherCodeToContext(0);
    expect(ctx.condition).toBe("sunny");
    expect(ctx.emoji).toBe("sun");
    expect(ctx.moodImpact).toBe("positive");
  });

  it("maps partly cloudy (2) to partly_cloudy", () => {
    const ctx = mapWeatherCodeToContext(2);
    expect(ctx.condition).toBe("partly_cloudy");
  });

  it("maps overcast (3) to cloudy", () => {
    const ctx = mapWeatherCodeToContext(3);
    expect(ctx.condition).toBe("cloudy");
    expect(ctx.moodImpact).toBe("neutral");
  });

  it("maps rain codes (61, 63, 65) to rainy", () => {
    for (const code of [61, 63, 65]) {
      const ctx = mapWeatherCodeToContext(code);
      expect(ctx.condition).toBe("rainy");
      expect(ctx.moodImpact).toBe("negative");
    }
  });

  it("maps snow codes (71, 73, 75) to snowy", () => {
    for (const code of [71, 73, 75]) {
      const ctx = mapWeatherCodeToContext(code);
      expect(ctx.condition).toBe("snowy");
    }
  });

  it("maps thunderstorm (95, 96, 99) to stormy", () => {
    for (const code of [95, 96, 99]) {
      const ctx = mapWeatherCodeToContext(code);
      expect(ctx.condition).toBe("stormy");
      expect(ctx.moodImpact).toBe("negative");
    }
  });

  it("maps fog (45, 48) to foggy", () => {
    const ctx = mapWeatherCodeToContext(45);
    expect(ctx.condition).toBe("foggy");
  });

  it("returns unknown for unrecognized code", () => {
    const ctx = mapWeatherCodeToContext(999);
    expect(ctx.condition).toBe("unknown");
    expect(ctx.moodImpact).toBe("neutral");
  });
});

// ── Daylight hours ────────────────────────────────────────────────────────

describe("computeDaylightHours", () => {
  it("computes daylight hours from sunrise/sunset times", () => {
    // sunrise 06:00, sunset 18:00 = 12 hours
    const hours = computeDaylightHours("06:00", "18:00");
    expect(hours).toBeCloseTo(12, 0);
  });

  it("handles early sunrise, late sunset (summer)", () => {
    const hours = computeDaylightHours("05:30", "20:30");
    expect(hours).toBeCloseTo(15, 0);
  });

  it("handles late sunrise, early sunset (winter)", () => {
    const hours = computeDaylightHours("07:45", "16:30");
    expect(hours).toBeCloseTo(8.75, 0);
  });

  it("returns 0 for invalid times", () => {
    expect(computeDaylightHours("", "")).toBe(0);
    expect(computeDaylightHours("invalid", "18:00")).toBe(0);
  });
});

// ── Mood context ──────────────────────────────────────────────────────────

describe("buildMoodContext", () => {
  it("builds positive context for sunny warm weather", () => {
    const data: WeatherData = {
      temperature: 24,
      weatherCode: 0,
      sunrise: "06:00",
      sunset: "19:00",
    };
    const ctx = buildMoodContext(data);
    expect(ctx.moodImpact).toBe("positive");
    expect(ctx.message).toBeTruthy();
    expect(ctx.message.length).toBeGreaterThan(0);
  });

  it("builds negative context for rainy weather", () => {
    const data: WeatherData = {
      temperature: 10,
      weatherCode: 63,
      sunrise: "07:30",
      sunset: "16:30",
    };
    const ctx = buildMoodContext(data);
    expect(ctx.moodImpact).toBe("negative");
    expect(ctx.message).toContain("may affect mood");
  });

  it("notes short daylight in winter-like conditions", () => {
    const data: WeatherData = {
      temperature: 2,
      weatherCode: 3,
      sunrise: "08:00",
      sunset: "16:00",
    };
    const ctx = buildMoodContext(data);
    expect(ctx.daylightHours).toBeCloseTo(8, 0);
    // Short daylight should be mentioned
    expect(ctx.message).toContain("daylight");
  });

  it("includes temperature label", () => {
    const data: WeatherData = {
      temperature: 30,
      weatherCode: 0,
      sunrise: "06:00",
      sunset: "19:00",
    };
    const ctx = buildMoodContext(data);
    expect(ctx.temperatureLabel).toBeTruthy();
  });
});

// ── Cache ─────────────────────────────────────────────────────────────────

describe("getCachedWeather / setCachedWeather", () => {
  it("returns null when no cache exists", () => {
    expect(getCachedWeather()).toBeNull();
  });

  it("roundtrips data correctly", () => {
    const data: WeatherData = {
      temperature: 22,
      weatherCode: 1,
      sunrise: "06:15",
      sunset: "18:45",
    };
    setCachedWeather(data);
    const cached = getCachedWeather();
    expect(cached).not.toBeNull();
    expect(cached!.temperature).toBe(22);
    expect(cached!.weatherCode).toBe(1);
  });

  it("returns null if cache is older than 1 hour", () => {
    const data: WeatherData = {
      temperature: 20,
      weatherCode: 0,
      sunrise: "06:00",
      sunset: "18:00",
    };
    setCachedWeather(data);

    // Manually age the cache
    const raw = JSON.parse(localStorage.getItem("ndw_weather_cache")!);
    raw.timestamp = Date.now() - 61 * 60 * 1000; // 61 minutes ago
    localStorage.setItem("ndw_weather_cache", JSON.stringify(raw));

    expect(getCachedWeather()).toBeNull();
  });

  it("handles corrupted cache gracefully", () => {
    localStorage.setItem("ndw_weather_cache", "not-json{{");
    expect(getCachedWeather()).toBeNull();
  });
});
