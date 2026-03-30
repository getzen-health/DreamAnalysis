/**
 * smart-alarm.ts — Smart sleep alarm that targets light-sleep wake windows.
 *
 * Wakes the user during N1/N2 or at the end of a REM cycle (REM→N1 transition)
 * within a configurable window before their target wake time, instead of during
 * deep sleep (N3) which causes sleep inertia.
 */

export interface SmartAlarmConfig {
  /** User's desired wake time */
  targetWakeTime: Date;
  /** How many minutes before target time the alarm may trigger early (e.g., 30) */
  windowMinutes: number;
}

export interface AlarmDecision {
  /** Whether the alarm should fire now */
  shouldWake: boolean;
  /** Human-readable reason for the decision */
  reason: string;
  /** How many minutes before target time this alarm would fire (0 if at/past target) */
  minutesEarly: number;
}

/**
 * Evaluate whether the alarm should trigger right now.
 *
 * @param config       - Target wake time and window size
 * @param currentStage - Current sleep stage: "Wake" | "N1" | "N2" | "N3" | "REM"
 * @param previousStage - Previous sleep stage (for detecting REM→N1 transitions), or null
 * @param now          - Current time
 * @returns AlarmDecision with shouldWake, reason, and minutesEarly
 */
export function shouldTriggerAlarm(
  config: SmartAlarmConfig,
  currentStage: string,
  previousStage: string | null,
  now: Date,
): AlarmDecision {
  const msToTarget = config.targetWakeTime.getTime() - now.getTime();
  const minutesToTarget = msToTarget / 60_000;

  // Past target time — wake immediately regardless of stage
  if (minutesToTarget <= 0) {
    return {
      shouldWake: true,
      reason: "Target time reached",
      minutesEarly: 0,
    };
  }

  // Not yet within the early-wake window — don't wake
  if (minutesToTarget > config.windowMinutes) {
    return {
      shouldWake: false,
      reason: "Outside wake window",
      minutesEarly: Math.round(minutesToTarget),
    };
  }

  // Within window — check for optimal stages

  // REM cycle just completed (transitioned from REM to N1)
  if (previousStage === "REM" && currentStage === "N1") {
    return {
      shouldWake: true,
      reason: "REM cycle complete",
      minutesEarly: Math.round(minutesToTarget),
    };
  }

  // Light sleep — optimal wake window
  if (currentStage === "N1" || currentStage === "N2") {
    return {
      shouldWake: true,
      reason: "Light sleep — optimal wake window",
      minutesEarly: Math.round(minutesToTarget),
    };
  }

  // Within window but in deep sleep or REM — wait for lighter stage
  return {
    shouldWake: false,
    reason:
      currentStage === "N3"
        ? "Deep sleep — waiting for lighter stage"
        : currentStage === "REM"
          ? "REM sleep — waiting for cycle to end"
          : "Waiting for optimal wake stage",
    minutesEarly: Math.round(minutesToTarget),
  };
}
