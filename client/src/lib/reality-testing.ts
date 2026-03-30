/**
 * Reality Testing Notification System
 *
 * Sends periodic "reality check" prompts during waking hours to build
 * lucid dreaming habits. Divides the active window into N equal slots,
 * picks a random minute within each slot, and assigns a random technique.
 *
 * Persistence: config stored via sbGetSetting/sbSaveSetting (localStorage + Supabase).
 * Notification delivery: browser Notification API (PWA / Capacitor WebView).
 */

import { sbGetSetting, sbSaveSetting } from "./supabase-store";

// ── Types ────────────────────────────────────────────────────────────────────

export interface RealityTestConfig {
  enabled: boolean;
  frequency: number;   // tests per day (3–10)
  startHour: number;   // e.g. 9 (9 am)
  endHour: number;     // e.g. 22 (10 pm)
  randomize: boolean;  // randomize timing within windows
}

export interface RealityTest {
  prompt: string;
  technique: string;
}

export interface ScheduledTest {
  hour: number;
  minute: number;
  test: RealityTest;
}

// ── Constants ────────────────────────────────────────────────────────────────

export const REALITY_TESTS: RealityTest[] = [
  { prompt: "Look at your hands — do they look normal?", technique: "hand-check" },
  { prompt: "Try to push your finger through your palm", technique: "finger-push" },
  { prompt: "Read some text, look away, read it again — did it change?", technique: "text-check" },
  { prompt: "Check the time, look away, check again — is it consistent?", technique: "clock-check" },
  { prompt: "Try to remember how you got here", technique: "memory-trace" },
  { prompt: "Is anything unusual about your surroundings?", technique: "environment-scan" },
];

export const DEFAULT_CONFIG: RealityTestConfig = {
  enabled: false,
  frequency: 5,
  startHour: 9,
  endHour: 22,
  randomize: true,
};

const CONFIG_KEY = "ndw_reality_test_config";
const SCHEDULE_KEY = "ndw_reality_test_schedule";
const FIRED_PREFIX = "ndw_rt_fired_";

// ── Seeded random (deterministic for a given day) ────────────────────────────

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function todaySeed(): number {
  const d = new Date();
  return d.getFullYear() * 10000 + (d.getMonth() + 1) * 100 + d.getDate();
}

// ── Scheduling ───────────────────────────────────────────────────────────────

/**
 * Divide waking hours into `frequency` equal windows.
 * Pick a random minute within each window (or the midpoint if randomize=false).
 * Assign a random (non-repeating where possible) test to each slot.
 *
 * Uses a day-based seed so the schedule is stable across calls within the same day
 * but different each day.
 */
export function scheduleRealityTests(config: RealityTestConfig): ScheduledTest[] {
  const { frequency, startHour, endHour, randomize } = config;

  if (frequency <= 0 || startHour >= endHour) return [];

  const totalMinutes = (endHour - startHour) * 60;
  const windowSize = totalMinutes / frequency;

  const rand = seededRandom(todaySeed());

  // Build a shuffled pool of tests so we avoid repeats until the pool runs out
  const pool = shuffleArray([...REALITY_TESTS], rand);

  const schedule: ScheduledTest[] = [];

  for (let i = 0; i < frequency; i++) {
    const windowStartMin = startHour * 60 + Math.floor(i * windowSize);

    let offsetMin: number;
    if (randomize) {
      offsetMin = Math.floor(rand() * Math.floor(windowSize));
    } else {
      offsetMin = Math.floor(windowSize / 2);
    }

    const absoluteMin = windowStartMin + offsetMin;
    const hour = Math.floor(absoluteMin / 60);
    const minute = absoluteMin % 60;

    // Cycle through pool
    const test = pool[i % pool.length];

    schedule.push({ hour, minute, test });
  }

  // Sort by time
  schedule.sort((a, b) => a.hour * 60 + a.minute - (b.hour * 60 + b.minute));

  return schedule;
}

function shuffleArray<T>(arr: T[], rand: () => number): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ── Config persistence ───────────────────────────────────────────────────────

export function loadRealityTestConfig(): RealityTestConfig {
  const raw = sbGetSetting(CONFIG_KEY);
  if (!raw) return { ...DEFAULT_CONFIG };
  try {
    const parsed = JSON.parse(raw);
    return {
      enabled: typeof parsed.enabled === "boolean" ? parsed.enabled : DEFAULT_CONFIG.enabled,
      frequency: clamp(parsed.frequency ?? DEFAULT_CONFIG.frequency, 3, 10),
      startHour: clamp(parsed.startHour ?? DEFAULT_CONFIG.startHour, 0, 23),
      endHour: clamp(parsed.endHour ?? DEFAULT_CONFIG.endHour, 1, 24),
      randomize: typeof parsed.randomize === "boolean" ? parsed.randomize : DEFAULT_CONFIG.randomize,
    };
  } catch {
    return { ...DEFAULT_CONFIG };
  }
}

export function saveRealityTestConfig(config: RealityTestConfig): void {
  sbSaveSetting(CONFIG_KEY, JSON.stringify(config));
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// ── Runtime notification scheduling ──────────────────────────────────────────

/**
 * Initialize the reality test notification system.
 * Call once on app mount. Returns a cleanup function.
 *
 * Checks every 60 seconds whether a scheduled test is due.
 * Fires a browser Notification and marks it as fired for the day.
 */
export function initRealityTestNotifications(): () => void {
  const check = () => {
    const config = loadRealityTestConfig();
    if (!config.enabled) return;

    if (!("Notification" in window) || Notification.permission !== "granted") return;

    const schedule = scheduleRealityTests(config);
    const now = new Date();
    const currentMin = now.getHours() * 60 + now.getMinutes();
    const today = now.toISOString().slice(0, 10);

    for (const slot of schedule) {
      const slotMin = slot.hour * 60 + slot.minute;
      // Fire if we're within 1 minute of the scheduled time
      if (Math.abs(currentMin - slotMin) <= 1) {
        const firedKey = `${FIRED_PREFIX}${today}_${slot.hour}_${slot.minute}`;
        if (sbGetSetting(firedKey)) continue;

        new Notification("Reality Check", {
          body: slot.test.prompt,
          icon: "/icon-192.png",
          tag: `reality-test-${slot.hour}-${slot.minute}`,
        });

        sbSaveSetting(firedKey, "true");
      }
    }
  };

  // Request permission if needed
  if ("Notification" in window && Notification.permission === "default") {
    Notification.requestPermission();
  }

  // Check immediately and then every 60 seconds
  check();
  const intervalId = setInterval(check, 60 * 1000);

  return () => clearInterval(intervalId);
}
