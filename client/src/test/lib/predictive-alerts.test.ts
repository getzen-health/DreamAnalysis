import { describe, it, expect } from "vitest";
import { predictNextDay, type PredictionInput } from "@/lib/predictive-alerts";

/** Baseline input where no rule triggers */
function noAlertInput(): PredictionInput {
  return {
    todayStress: 0.4,
    todayValence: 0.1,
    todayFocus: 0.5,
    todaySleepHours: 7.5,
    todaySleepQuality: 65,
    todaySteps: 5000,
    todayInnerScore: 60,
    lateEating: false,
    recentScores: [60, 62, 61, 63, 62, 64, 60],
    recentStress: [0.4, 0.35, 0.45, 0.38, 0.42, 0.4, 0.39],
    recentSleep: [7.5, 7, 7.2, 7.8, 7.1, 7.3, 7.5],
  };
}

describe("predictNextDay", () => {
  // ── Rule 1: Sleep debt warning ────────────────────────────────────────

  it("triggers sleep debt warning when todaySleepHours < 6 and avg recentSleep < 6.5", () => {
    const input: PredictionInput = {
      ...noAlertInput(),
      todaySleepHours: 5,
      recentSleep: [5.5, 6, 5.8, 6.2, 5.5, 6.1, 5.9],
    };
    const alert = predictNextDay(input);
    expect(alert).not.toBeNull();
    expect(alert!.id).toBe("sleep-debt-warning");
    expect(alert!.type).toBe("warning");
    expect(alert!.headline).toContain("Sleep debt");
    expect(alert!.action).toBe("Try to get 8+ hours tonight");
  });

  it("does NOT trigger sleep debt when sleep is fine", () => {
    const input: PredictionInput = {
      ...noAlertInput(),
      todaySleepHours: 7.5,
      recentSleep: [7, 7.5, 8, 7.2, 7.8, 7.5, 7.3],
    };
    const alert = predictNextDay(input);
    // Should not be a sleep debt alert
    if (alert) {
      expect(alert.id).not.toBe("sleep-debt-warning");
    }
  });

  it("does NOT trigger sleep debt when today is low but history is fine", () => {
    const input: PredictionInput = {
      ...noAlertInput(),
      todaySleepHours: 4,
      recentSleep: [7, 7.5, 8, 7.2, 7.8, 7.5, 7.3],
    };
    const alert = predictNextDay(input);
    if (alert) {
      expect(alert.id).not.toBe("sleep-debt-warning");
    }
  });

  // ── Rule 2: Stress spiral ────────────────────────────────────────────

  it("triggers stress spiral when todayStress > 0.7 and 2+ of last 3 recentStress > 0.6", () => {
    const input: PredictionInput = {
      ...noAlertInput(),
      todayStress: 0.8,
      recentStress: [0.4, 0.5, 0.55, 0.5, 0.65, 0.72, 0.68],
    };
    const alert = predictNextDay(input);
    expect(alert).not.toBeNull();
    expect(alert!.id).toBe("stress-spiral");
    expect(alert!.type).toBe("warning");
    expect(alert!.headline).toContain("Stress");
    expect(alert!.action).toBe("Try a breathing exercise before bed");
  });

  it("does NOT trigger stress spiral when today's stress is low", () => {
    const input: PredictionInput = {
      ...noAlertInput(),
      todayStress: 0.5,
      recentStress: [0.7, 0.8, 0.75],
    };
    const alert = predictNextDay(input);
    if (alert) {
      expect(alert.id).not.toBe("stress-spiral");
    }
  });

  // ── Rule 3: Score declining ──────────────────────────────────────────

  it("triggers score declining when 3+ consecutive drops in recentScores", () => {
    const input: PredictionInput = {
      ...noAlertInput(),
      recentScores: [80, 75, 70, 65, 60, 55, 50],
    };
    const alert = predictNextDay(input);
    expect(alert).not.toBeNull();
    expect(alert!.id).toBe("score-declining");
    expect(alert!.type).toBe("warning");
    expect(alert!.headline).toContain("declining");
    expect(alert!.action).toBe("Prioritize rest tonight");
  });

  it("does NOT trigger score declining when scores are stable or rising", () => {
    const input: PredictionInput = {
      ...noAlertInput(),
      recentScores: [60, 62, 65, 63, 67, 70, 72],
    };
    const alert = predictNextDay(input);
    if (alert) {
      expect(alert.id).not.toBe("score-declining");
    }
  });

  // ── Rule 4: Late eating ──────────────────────────────────────────────

  it("triggers late eating warning when lateEating=true and avg recentSleep < 7", () => {
    const input: PredictionInput = {
      ...noAlertInput(),
      lateEating: true,
      recentSleep: [6.5, 6.2, 6.8, 6.0, 6.5, 6.3, 6.7],
    };
    const alert = predictNextDay(input);
    expect(alert).not.toBeNull();
    expect(alert!.id).toBe("late-eating-sleep");
    expect(alert!.type).toBe("warning");
    expect(alert!.headline).toContain("Late meal");
    expect(alert!.action).toBe("No food 2 hours before bed");
  });

  it("does NOT trigger late eating when lateEating is false", () => {
    const input: PredictionInput = {
      ...noAlertInput(),
      lateEating: false,
      recentSleep: [6, 5.5, 6, 5.8, 6.2, 5.5, 6.1],
    };
    const alert = predictNextDay(input);
    if (alert) {
      expect(alert.id).not.toBe("late-eating-sleep");
    }
  });

  // ── Rule 5: Great day ahead ──────────────────────────────────────────

  it("triggers great day ahead when sleepQuality > 75, stress < 0.3, innerScore > 75", () => {
    const input: PredictionInput = {
      ...noAlertInput(),
      todaySleepQuality: 85,
      todayStress: 0.15,
      todayInnerScore: 82,
    };
    const alert = predictNextDay(input);
    expect(alert).not.toBeNull();
    expect(alert!.id).toBe("great-day-ahead");
    expect(alert!.type).toBe("positive");
    expect(alert!.headline).toContain("Tomorrow looks strong");
  });

  it("does NOT trigger great day ahead when stress is elevated", () => {
    const input: PredictionInput = {
      ...noAlertInput(),
      todaySleepQuality: 85,
      todayStress: 0.5,
      todayInnerScore: 82,
    };
    const alert = predictNextDay(input);
    if (alert) {
      expect(alert.id).not.toBe("great-day-ahead");
    }
  });

  // ── Rule 6: Active day boost ─────────────────────────────────────────

  it("triggers active day boost when steps > 8000 and sleep >= 7h", () => {
    const input: PredictionInput = {
      ...noAlertInput(),
      todaySteps: 10000,
      todaySleepHours: 7.5,
    };
    const alert = predictNextDay(input);
    expect(alert).not.toBeNull();
    expect(alert!.id).toBe("active-day-boost");
    expect(alert!.type).toBe("positive");
    expect(alert!.headline).toContain("Active day");
  });

  it("does NOT trigger active day boost when sleep is short", () => {
    const input: PredictionInput = {
      ...noAlertInput(),
      todaySteps: 12000,
      todaySleepHours: 5,
      recentSleep: [5, 5.5, 5, 5.8, 5.2, 5.5, 5.9],
    };
    const alert = predictNextDay(input);
    if (alert) {
      // Sleep debt should fire first due to priority, but active-day-boost should not appear
      expect(alert.id).not.toBe("active-day-boost");
    }
  });

  // ── Returns null when no rules match ─────────────────────────────────

  it("returns null when no rules match", () => {
    const alert = predictNextDay(noAlertInput());
    expect(alert).toBeNull();
  });

  // ── Confidence ranges ────────────────────────────────────────────────

  it("confidence is between 50 and 85 for all alerts", () => {
    // Sleep debt
    const sleepDebt = predictNextDay({
      ...noAlertInput(),
      todaySleepHours: 4,
      recentSleep: [5, 5.5, 5, 5.8, 5.2, 5.5, 5.9],
    });
    expect(sleepDebt).not.toBeNull();
    expect(sleepDebt!.confidence).toBeGreaterThanOrEqual(50);
    expect(sleepDebt!.confidence).toBeLessThanOrEqual(85);

    // Stress spiral
    const stressSpiral = predictNextDay({
      ...noAlertInput(),
      todayStress: 0.8,
      recentStress: [0.4, 0.5, 0.55, 0.5, 0.65, 0.72, 0.68],
    });
    expect(stressSpiral).not.toBeNull();
    expect(stressSpiral!.confidence).toBeGreaterThanOrEqual(50);
    expect(stressSpiral!.confidence).toBeLessThanOrEqual(85);

    // Great day
    const greatDay = predictNextDay({
      ...noAlertInput(),
      todaySleepQuality: 85,
      todayStress: 0.15,
      todayInnerScore: 82,
    });
    expect(greatDay).not.toBeNull();
    expect(greatDay!.confidence).toBeGreaterThanOrEqual(50);
    expect(greatDay!.confidence).toBeLessThanOrEqual(85);
  });

  // ── Factors array populated correctly ────────────────────────────────

  it("populates factors array with supporting data points", () => {
    const alert = predictNextDay({
      ...noAlertInput(),
      todaySleepHours: 4,
      recentSleep: [5, 5.5, 5, 5.8, 5.2, 5.5, 5.9],
    });
    expect(alert).not.toBeNull();
    expect(alert!.factors.length).toBeGreaterThan(0);
    expect(alert!.factors.some((f) => f.includes("6 hours"))).toBe(true);
  });

  it("populates factors for positive alerts", () => {
    const alert = predictNextDay({
      ...noAlertInput(),
      todaySleepQuality: 85,
      todayStress: 0.15,
      todayInnerScore: 82,
    });
    expect(alert).not.toBeNull();
    expect(alert!.factors.length).toBeGreaterThanOrEqual(2);
    expect(alert!.factors.some((f) => f.toLowerCase().includes("sleep"))).toBe(true);
  });

  // ── Priority: first matching rule wins ───────────────────────────────

  it("returns sleep debt (rule 1) over stress spiral (rule 2) when both match", () => {
    const input: PredictionInput = {
      ...noAlertInput(),
      todaySleepHours: 4,
      todayStress: 0.85,
      recentSleep: [5, 5.5, 5, 5.8, 5.2, 5.5, 5.9],
      recentStress: [0.65, 0.72, 0.68, 0.65, 0.72, 0.68, 0.7],
    };
    const alert = predictNextDay(input);
    expect(alert).not.toBeNull();
    expect(alert!.id).toBe("sleep-debt-warning");
  });

  // ── Null inputs handled gracefully ───────────────────────────────────

  it("handles all-null inputs gracefully", () => {
    const input: PredictionInput = {
      todayStress: null,
      todayValence: null,
      todayFocus: null,
      todaySleepHours: null,
      todaySleepQuality: null,
      todaySteps: null,
      todayInnerScore: null,
      lateEating: false,
      recentScores: [],
      recentStress: [],
      recentSleep: [],
    };
    const alert = predictNextDay(input);
    expect(alert).toBeNull();
  });
});
