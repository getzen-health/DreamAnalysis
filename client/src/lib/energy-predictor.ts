/**
 * energy-predictor.ts — Predict hourly energy levels for remaining day.
 *
 * Uses circadian baseline + sleep data + recovery score to forecast
 * which hours will have peak energy for deep work.
 */

import { getCircadianAdjustment } from "./circadian-adjustment";

export interface HourlyEnergy {
  hour: number;         // 0-23
  energy: number;       // 0-1
  label: string;        // "Peak Focus", "Post-lunch Dip", etc.
  isPeak: boolean;      // true for top 2 hours
  isCurrent: boolean;   // true for current hour
}

export interface EnergyForecast {
  hours: HourlyEnergy[];
  peakWindow: string;   // e.g., "9 AM - 11 AM"
  currentEnergy: number;
}

/**
 * Generate hourly energy forecast from current hour to end of day.
 *
 * @param sleepHours - Hours slept last night (0-12)
 * @param recovery - Recovery score 0-100
 * @param wakeHour - Hour the user woke up (0-23), default 7
 */
export function predictEnergy(
  sleepHours: number = 7,
  recovery: number = 50,
  wakeHour: number = 7,
): EnergyForecast {
  const now = new Date();
  const currentHour = now.getHours();
  const hours: HourlyEnergy[] = [];

  // Sleep debt factor: 7h = baseline, each hour less reduces energy ~10%
  const sleepFactor = Math.min(1.0, Math.max(0.3, sleepHours / 8));

  // Recovery factor: maps 0-100 to 0.5-1.0
  const recoveryFactor = 0.5 + (recovery / 100) * 0.5;

  // Hours awake factor: energy decreases as the day goes on
  const hoursAwake = (h: number) => {
    const awake = ((h - wakeHour + 24) % 24);
    // Energy follows an inverted U: peaks 2-5h after wake, declines after 10h
    if (awake < 2) return 0.7;   // still waking up
    if (awake < 5) return 1.0;   // peak window
    if (awake < 8) return 0.85;  // solid
    if (awake < 10) return 0.7;  // declining
    if (awake < 13) return 0.55; // tired
    return 0.4;                  // very tired
  };

  // Generate forecast for remaining hours
  for (let h = currentHour; h <= 23; h++) {
    const circadian = getCircadianAdjustment(h);
    // Base energy: combine circadian focus offset + awake curve
    const focusBoost = circadian.focusBaselineOffset;
    const baseEnergy = hoursAwake(h) + focusBoost;
    // Apply sleep and recovery modifiers
    const energy = Math.min(1.0, Math.max(0, baseEnergy * sleepFactor * recoveryFactor));
    hours.push({
      hour: h,
      energy,
      label: getEnergyLabel(h, energy, circadian.label),
      isPeak: false,
      isCurrent: h === currentHour,
    });
  }

  // Mark top 2 hours as peak
  const sorted = [...hours].sort((a, b) => b.energy - a.energy);
  for (let i = 0; i < Math.min(2, sorted.length); i++) {
    const peak = hours.find(h => h.hour === sorted[i].hour);
    if (peak) peak.isPeak = true;
  }

  // Format peak window
  const peaks = hours.filter(h => h.isPeak).sort((a, b) => a.hour - b.hour);
  const peakWindow = peaks.length >= 2
    ? `${formatHour(peaks[0].hour)} - ${formatHour(peaks[peaks.length - 1].hour + 1)}`
    : peaks.length === 1
    ? `${formatHour(peaks[0].hour)} - ${formatHour(peaks[0].hour + 1)}`
    : "No peak predicted";

  const current = hours.find(h => h.isCurrent);

  return {
    hours,
    peakWindow,
    currentEnergy: current?.energy ?? 0.5,
  };
}

function getEnergyLabel(hour: number, energy: number, circadianLabel: string): string {
  if (energy >= 0.8) return "Peak Focus";
  if (energy >= 0.6) return "Good Energy";
  if (circadianLabel.includes("post-lunch")) return "Post-lunch Dip";
  if (circadianLabel.includes("wind-down")) return "Wind Down";
  if (energy >= 0.4) return "Moderate";
  return "Low Energy";
}

function formatHour(h: number): string {
  const hour = ((h % 24) + 24) % 24;
  if (hour === 0) return "12 AM";
  if (hour === 12) return "12 PM";
  if (hour < 12) return `${hour} AM`;
  return `${hour - 12} PM`;
}
