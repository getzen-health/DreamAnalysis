/**
 * EEG-Triggered Intervention Engine — closed-loop auto-suggest system.
 *
 * Monitors real-time EEG-derived metrics (stress, blink rate, alpha/beta levels,
 * session duration) and triggers non-blocking intervention suggestions when
 * thresholds are exceeded.
 *
 * Trigger priority (highest first):
 *   1. Sustained stress → breathing exercise (high)
 *   2. Low alpha + high beta → music change (medium)
 *   3. High blink rate → break suggestion (medium)
 *   4. Long session → break suggestion (low)
 *
 * Cooldown: 5 minutes per trigger type to avoid spamming.
 *
 * @see Issue #504
 */

// ── Types ────────────────────────────────────────────────────────────────────

export interface InterventionTrigger {
  type: "breathing" | "music_change" | "notification" | "break_suggestion";
  reason: string;
  priority: "low" | "medium" | "high";
  action: () => void;
}

export interface TriggerConfig {
  enabled: boolean;
  stressThreshold: number;        // default 0.7
  stressDuration: number;         // seconds threshold must be exceeded (default 30)
  fatigueThreshold: number;       // blinks/min from blink detector (default 25)
  autoBreathing: boolean;         // auto-suggest breathing exercise
  autoMusicChange: boolean;       // auto-switch to calm playlist
  autoBreakReminder: boolean;     // suggest break after fatigue detection
}

// ── Constants ────────────────────────────────────────────────────────────────

const STORAGE_KEY = "ndw_intervention_trigger_config";
const COOLDOWN_MS = 5 * 60 * 1000; // 5 minutes
const SESSION_LENGTH_THRESHOLD_MIN = 25;
const ALPHA_LOW_THRESHOLD = 0.05;
const HIGH_BETA_DURATION_THRESHOLD_SEC = 60;

// ── Cooldown tracking (module-level) ─────────────────────────────────────────

const lastTriggerTime: Map<string, number> = new Map();

/** Reset all cooldowns. Exported for testing only. */
export function _resetCooldowns(): void {
  lastTriggerTime.clear();
}

function isOnCooldown(type: string): boolean {
  const last = lastTriggerTime.get(type);
  if (last == null) return false;
  return Date.now() - last < COOLDOWN_MS;
}

function recordTrigger(type: string): void {
  lastTriggerTime.set(type, Date.now());
}

// ── Priority ordering ────────────────────────────────────────────────────────

const PRIORITY_ORDER: Record<string, number> = {
  high: 3,
  medium: 2,
  low: 1,
};

// ── Config management ────────────────────────────────────────────────────────

export function getDefaultTriggerConfig(): TriggerConfig {
  return {
    enabled: true,
    stressThreshold: 0.7,
    stressDuration: 30,
    fatigueThreshold: 25,
    autoBreathing: true,
    autoMusicChange: true,
    autoBreakReminder: true,
  };
}

export function saveTriggerConfig(config: TriggerConfig): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
  } catch {
    // localStorage unavailable — silent fail
  }
}

export function loadTriggerConfig(): TriggerConfig {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as TriggerConfig;
      return parsed;
    }
  } catch {
    // corrupt or unavailable — return defaults
  }
  return getDefaultTriggerConfig();
}

// ── Trigger evaluation ───────────────────────────────────────────────────────

export interface TriggerState {
  stressIndex: number;
  stressDurationSeconds: number;
  blinksPerMinute: number;
  sessionMinutes: number;
  alphaLevel: number;
  betaLevel: number;
  highBetaDurationSeconds: number;
}

/**
 * Evaluate all trigger conditions against current EEG state.
 * Returns the highest-priority trigger that is not on cooldown, or null.
 */
export function checkTriggers(
  state: TriggerState,
  config: TriggerConfig,
): InterventionTrigger | null {
  if (!config.enabled) return null;

  const candidates: InterventionTrigger[] = [];

  // 1. Sustained stress → breathing exercise (high priority)
  if (
    config.autoBreathing &&
    state.stressIndex > config.stressThreshold &&
    state.stressDurationSeconds > config.stressDuration
  ) {
    candidates.push({
      type: "breathing",
      reason: `Elevated stress detected for ${Math.round(state.stressDurationSeconds)}+ seconds`,
      priority: "high",
      action: () => {
        // Navigation handled by the component consuming this trigger
      },
    });
  }

  // 2. Alpha very low + high beta for > 60s → music change (medium priority)
  if (
    config.autoMusicChange &&
    state.alphaLevel < ALPHA_LOW_THRESHOLD &&
    state.betaLevel > 0.3 &&
    state.highBetaDurationSeconds > HIGH_BETA_DURATION_THRESHOLD_SEC
  ) {
    candidates.push({
      type: "music_change",
      reason: "Low alpha and elevated beta activity — calming music may help",
      priority: "medium",
      action: () => {},
    });
  }

  // 3. High blink rate → fatigue break (medium priority)
  if (
    config.autoBreakReminder &&
    state.blinksPerMinute > config.fatigueThreshold
  ) {
    candidates.push({
      type: "break_suggestion",
      reason: `High blink rate (${Math.round(state.blinksPerMinute)} blinks/min) suggests eye fatigue`,
      priority: "medium",
      action: () => {},
    });
  }

  // 4. Long session → break (low priority, neurofeedback safety)
  if (
    config.autoBreakReminder &&
    state.sessionMinutes > SESSION_LENGTH_THRESHOLD_MIN
  ) {
    candidates.push({
      type: "break_suggestion",
      reason: `Session running ${Math.round(state.sessionMinutes)} minutes — consider a short break`,
      priority: "low",
      action: () => {},
    });
  }

  // Sort by priority descending
  candidates.sort(
    (a, b) => (PRIORITY_ORDER[b.priority] ?? 0) - (PRIORITY_ORDER[a.priority] ?? 0),
  );

  // Return highest-priority candidate that is not on cooldown
  for (const candidate of candidates) {
    if (!isOnCooldown(candidate.type)) {
      recordTrigger(candidate.type);
      return candidate;
    }
  }

  return null;
}
