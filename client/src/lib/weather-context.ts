/**
 * weather-context.ts — Weather + seasonal mood context.
 *
 * Uses free Open-Meteo API (no API key needed):
 *   https://api.open-meteo.com/v1/forecast
 *
 * Features:
 *   - Get location from browser geolocation API (with consent check)
 *   - Fetch temperature, weather code, daylight hours
 *   - Map weather to mood context (cloudy/rainy -> note, sunny -> positive)
 *   - Cache results for 1 hour in localStorage
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface WeatherData {
  temperature: number;    // Celsius
  weatherCode: number;    // WMO weather code
  sunrise: string;        // "HH:MM" format
  sunset: string;         // "HH:MM" format
}

export interface WeatherConditionContext {
  condition: string;
  emoji: string;
  moodImpact: "positive" | "neutral" | "negative";
  label: string;
}

export interface WeatherMoodContext {
  condition: string;
  emoji: string;
  moodImpact: "positive" | "neutral" | "negative";
  message: string;
  daylightHours: number;
  temperatureLabel: string;
  temperature: number;
}

// ── Weather code mapping (WMO standard) ────────────────────────────────────

const WEATHER_CODE_MAP: Record<number, WeatherConditionContext> = {
  0: { condition: "sunny", emoji: "sun", moodImpact: "positive", label: "Clear sky" },
  1: { condition: "mostly_clear", emoji: "sun", moodImpact: "positive", label: "Mostly clear" },
  2: { condition: "partly_cloudy", emoji: "cloud-sun", moodImpact: "neutral", label: "Partly cloudy" },
  3: { condition: "cloudy", emoji: "cloud", moodImpact: "neutral", label: "Overcast" },
  45: { condition: "foggy", emoji: "cloud-fog", moodImpact: "neutral", label: "Fog" },
  48: { condition: "foggy", emoji: "cloud-fog", moodImpact: "neutral", label: "Rime fog" },
  51: { condition: "drizzle", emoji: "cloud-drizzle", moodImpact: "negative", label: "Light drizzle" },
  53: { condition: "drizzle", emoji: "cloud-drizzle", moodImpact: "negative", label: "Drizzle" },
  55: { condition: "drizzle", emoji: "cloud-drizzle", moodImpact: "negative", label: "Heavy drizzle" },
  61: { condition: "rainy", emoji: "cloud-rain", moodImpact: "negative", label: "Light rain" },
  63: { condition: "rainy", emoji: "cloud-rain", moodImpact: "negative", label: "Rain" },
  65: { condition: "rainy", emoji: "cloud-rain", moodImpact: "negative", label: "Heavy rain" },
  66: { condition: "rainy", emoji: "cloud-rain", moodImpact: "negative", label: "Freezing rain" },
  67: { condition: "rainy", emoji: "cloud-rain", moodImpact: "negative", label: "Heavy freezing rain" },
  71: { condition: "snowy", emoji: "snowflake", moodImpact: "neutral", label: "Light snow" },
  73: { condition: "snowy", emoji: "snowflake", moodImpact: "neutral", label: "Snow" },
  75: { condition: "snowy", emoji: "snowflake", moodImpact: "neutral", label: "Heavy snow" },
  77: { condition: "snowy", emoji: "snowflake", moodImpact: "neutral", label: "Snow grains" },
  80: { condition: "rainy", emoji: "cloud-rain", moodImpact: "negative", label: "Rain showers" },
  81: { condition: "rainy", emoji: "cloud-rain", moodImpact: "negative", label: "Moderate rain showers" },
  82: { condition: "rainy", emoji: "cloud-rain", moodImpact: "negative", label: "Heavy rain showers" },
  85: { condition: "snowy", emoji: "snowflake", moodImpact: "neutral", label: "Snow showers" },
  86: { condition: "snowy", emoji: "snowflake", moodImpact: "neutral", label: "Heavy snow showers" },
  95: { condition: "stormy", emoji: "cloud-lightning", moodImpact: "negative", label: "Thunderstorm" },
  96: { condition: "stormy", emoji: "cloud-lightning", moodImpact: "negative", label: "Thunderstorm with hail" },
  99: { condition: "stormy", emoji: "cloud-lightning", moodImpact: "negative", label: "Heavy thunderstorm" },
};

const UNKNOWN_CONDITION: WeatherConditionContext = {
  condition: "unknown",
  emoji: "help-circle",
  moodImpact: "neutral",
  label: "Unknown",
};

export function mapWeatherCodeToContext(code: number): WeatherConditionContext {
  return WEATHER_CODE_MAP[code] ?? UNKNOWN_CONDITION;
}

// ── Daylight computation ───────────────────────────────────────────────────

/**
 * Compute daylight hours from sunrise/sunset strings ("HH:MM").
 * Returns 0 for invalid inputs.
 */
export function computeDaylightHours(sunrise: string, sunset: string): number {
  if (!sunrise || !sunset) return 0;

  const parseTime = (t: string): number | null => {
    const parts = t.split(":");
    if (parts.length < 2) return null;
    const h = parseInt(parts[0], 10);
    const m = parseInt(parts[1], 10);
    if (isNaN(h) || isNaN(m)) return null;
    return h + m / 60;
  };

  const riseHours = parseTime(sunrise);
  const setHours = parseTime(sunset);

  if (riseHours === null || setHours === null) return 0;
  return Math.max(0, setHours - riseHours);
}

// ── Temperature labels ─────────────────────────────────────────────────────

function getTemperatureLabel(tempC: number): string {
  if (tempC <= 0) return "Freezing";
  if (tempC <= 10) return "Cold";
  if (tempC <= 18) return "Cool";
  if (tempC <= 25) return "Comfortable";
  if (tempC <= 32) return "Warm";
  return "Hot";
}

// ── Build mood context ─────────────────────────────────────────────────────

export function buildMoodContext(data: WeatherData): WeatherMoodContext {
  const weatherCtx = mapWeatherCodeToContext(data.weatherCode);
  const daylightHours = computeDaylightHours(data.sunrise, data.sunset);
  const temperatureLabel = getTemperatureLabel(data.temperature);

  const messages: string[] = [];

  // Weather impact
  if (weatherCtx.moodImpact === "negative") {
    messages.push("Weather may affect mood");
  } else if (weatherCtx.moodImpact === "positive") {
    messages.push("Good weather for outdoor activity");
  }

  // Short daylight (< 10 hours)
  if (daylightHours > 0 && daylightHours < 10) {
    messages.push(`Short daylight (${daylightHours.toFixed(1)}h) — consider light exposure`);
  }

  // Temperature extremes
  if (data.temperature <= 0) {
    messages.push("Freezing temperatures — stay warm");
  } else if (data.temperature >= 35) {
    messages.push("Extreme heat — stay hydrated");
  }

  const message = messages.length > 0
    ? messages.join(". ")
    : `${weatherCtx.label}, ${temperatureLabel.toLowerCase()}`;

  return {
    condition: weatherCtx.condition,
    emoji: weatherCtx.emoji,
    moodImpact: weatherCtx.moodImpact,
    message,
    daylightHours,
    temperatureLabel,
    temperature: data.temperature,
  };
}

// ── Cache (1 hour TTL) ─────────────────────────────────────────────────────

const CACHE_KEY = "ndw_weather_cache";
const CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour

interface CachedEntry {
  data: WeatherData;
  timestamp: number;
}

export function getCachedWeather(): WeatherData | null {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const entry: CachedEntry = JSON.parse(raw);
    if (!entry || typeof entry.timestamp !== "number") return null;
    if (Date.now() - entry.timestamp > CACHE_TTL_MS) return null;
    return entry.data;
  } catch {
    return null;
  }
}

export function setCachedWeather(data: WeatherData): void {
  try {
    const entry: CachedEntry = { data, timestamp: Date.now() };
    localStorage.setItem(CACHE_KEY, JSON.stringify(entry));
  } catch {
    // localStorage full or unavailable
  }
}

// ── Fetch weather from Open-Meteo ──────────────────────────────────────────

/**
 * Fetch current weather from Open-Meteo API.
 * Uses browser geolocation for coordinates.
 * Returns cached data if available and fresh (< 1 hour).
 */
export async function fetchWeather(): Promise<WeatherData | null> {
  // Check cache first
  const cached = getCachedWeather();
  if (cached) return cached;

  // Get location
  let lat: number;
  let lon: number;
  try {
    const pos = await new Promise<GeolocationPosition>((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(resolve, reject, {
        enableHighAccuracy: false,
        timeout: 10000,
        maximumAge: 30 * 60 * 1000, // 30 min cache
      });
    });
    lat = pos.coords.latitude;
    lon = pos.coords.longitude;
  } catch {
    return null; // User denied location or API unavailable
  }

  // Fetch from Open-Meteo
  try {
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,weather_code&daily=sunrise,sunset&timezone=auto&forecast_days=1`;
    const resp = await fetch(url);
    if (!resp.ok) return null;
    const json = await resp.json();

    const data: WeatherData = {
      temperature: json.current?.temperature_2m ?? 20,
      weatherCode: json.current?.weather_code ?? 0,
      sunrise: json.daily?.sunrise?.[0]?.split("T")[1]?.slice(0, 5) ?? "",
      sunset: json.daily?.sunset?.[0]?.split("T")[1]?.slice(0, 5) ?? "",
    };

    setCachedWeather(data);
    return data;
  } catch {
    return null;
  }
}
