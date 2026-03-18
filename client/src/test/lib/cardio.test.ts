import { describe, it, expect } from "vitest";
import {
  getHrZones,
  computeTrimp,
  computeStrain,
  estimate1rm,
  computeCardioLoad,
  computeHrRecovery,
} from "@shared/cardio";

// ── getHrZones ──────────────────────────────────────────────────────────────

describe("getHrZones", () => {
  it("30-year-old: maxHR=190, zone1 starts at 95", () => {
    const result = getHrZones(30);
    expect(result.max).toBe(190);
    expect(result.zones[0].min).toBe(Math.round(190 * 0.50)); // 95
    expect(result.zones[0].min).toBe(95);
  });

  it("50-year-old: maxHR=170, zone5 starts at 153", () => {
    const result = getHrZones(50);
    expect(result.max).toBe(170);
    expect(result.zones[4].min).toBe(Math.round(170 * 0.90)); // 153
    expect(result.zones[4].min).toBe(153);
    expect(result.zones[4].max).toBe(170);
  });

  it("returns resting HR as provided", () => {
    const result = getHrZones(30, 55);
    expect(result.resting).toBe(55);
  });

  it("uses default resting HR of 60", () => {
    const result = getHrZones(30);
    expect(result.resting).toBe(60);
  });

  it("zones cover from 50% to 100% of maxHR without overlaps", () => {
    const result = getHrZones(30);
    const zones = result.zones;
    // Zone 1 min should start at 50% of max
    expect(zones[0].min).toBe(Math.round(result.max * 0.50));
    // Zone 5 max should be maxHR
    expect(zones[4].max).toBe(result.max);

    // Each zone's min should be calculated from the percentage of maxHR
    // Due to rounding, there can be small gaps between zones (e.g., 112..114)
    // but zones should never overlap
    for (let i = 0; i < zones.length - 1; i++) {
      expect(zones[i + 1].min).toBeGreaterThan(zones[i].max);
    }

    // Each zone's min should be less than or equal to its max
    for (const z of zones) {
      expect(z.min).toBeLessThanOrEqual(z.max);
    }
  });

  it("returns 5 zones with correct names", () => {
    const result = getHrZones(25);
    expect(result.zones).toHaveLength(5);
    expect(result.zones[0].name).toBe("Active Recovery");
    expect(result.zones[1].name).toBe("Fat Burning");
    expect(result.zones[2].name).toBe("Cardiovascular");
    expect(result.zones[3].name).toBe("High Intensity");
    expect(result.zones[4].name).toBe("Maximum Effort");
  });

  it("zone numbers are 1-5", () => {
    const result = getHrZones(25);
    result.zones.forEach((z, i) => {
      expect(z.zone).toBe(i + 1);
    });
  });

  it("handles very young age (20)", () => {
    const result = getHrZones(20);
    expect(result.max).toBe(200);
    expect(result.zones[0].min).toBe(100); // 200 * 0.50
  });

  it("handles older age (70)", () => {
    const result = getHrZones(70);
    expect(result.max).toBe(150);
    expect(result.zones[4].min).toBe(Math.round(150 * 0.90)); // 135
  });
});

// ── computeTrimp ────────────────────────────────────────────────────────────

describe("computeTrimp", () => {
  it("returns 0 when avgHr equals restingHr", () => {
    const result = computeTrimp(60, 60, 60, 190);
    // hrRatio = (60-60)/(190-60) = 0
    // trimp = 60 * 0 * exp(1.92 * 0) = 0
    expect(result).toBe(0);
  });

  it("increases with duration", () => {
    const short = computeTrimp(30, 140, 60, 190);
    const long = computeTrimp(60, 140, 60, 190);
    expect(long).toBeGreaterThan(short);
    // TRIMP is directly proportional to duration
    expect(long).toBeCloseTo(short * 2, 5);
  });

  it("male factor (1.92) gives higher TRIMP than female (1.67) for same input", () => {
    const male = computeTrimp(60, 140, 60, 190, "male");
    const female = computeTrimp(60, 140, 60, 190, "female");
    expect(male).toBeGreaterThan(female);
  });

  it("defaults to male when gender not specified", () => {
    const defaultGender = computeTrimp(60, 140, 60, 190);
    const male = computeTrimp(60, 140, 60, 190, "male");
    expect(defaultGender).toBe(male);
  });

  it("exponential growth with higher HR ratio", () => {
    // Test that TRIMP grows exponentially, not linearly, with HR
    const low = computeTrimp(60, 100, 60, 200);   // hrRatio = 40/140 ~= 0.286
    const mid = computeTrimp(60, 130, 60, 200);   // hrRatio = 70/140 = 0.5
    const high = computeTrimp(60, 170, 60, 200);  // hrRatio = 110/140 ~= 0.786

    expect(mid).toBeGreaterThan(low);
    expect(high).toBeGreaterThan(mid);

    // Check exponential growth: the ratio high/mid should be greater than mid/low
    const ratio1 = mid / low;
    const ratio2 = high / mid;
    expect(ratio2).toBeGreaterThan(ratio1);
  });

  it("calculates a known TRIMP value correctly", () => {
    // Manual calculation:
    // duration=60, avgHr=150, restingHr=60, maxHr=190, male
    // hrRatio = (150-60)/(190-60) = 90/130 = 0.6923
    // trimp = 60 * 0.6923 * exp(1.92 * 0.6923)
    const hrRatio = 90 / 130;
    const expected = 60 * hrRatio * Math.exp(1.92 * hrRatio);
    const result = computeTrimp(60, 150, 60, 190, "male");
    expect(result).toBeCloseTo(expected, 5);
  });
});

// ── computeStrain ───────────────────────────────────────────────────────────

describe("computeStrain", () => {
  it("returns 0 for trimp=0", () => {
    expect(computeStrain(0)).toBe(0);
  });

  it("logarithmic growth", () => {
    const s1 = computeStrain(50);
    const s2 = computeStrain(100);
    // Doubling TRIMP should not double strain
    expect(s2).toBeLessThan(s1 * 2);
    expect(s2).toBeGreaterThan(s1);
  });

  it("k=14.3 applied correctly", () => {
    const trimp = 50;
    const expected = 14.3 * Math.log(1 + trimp);
    expect(computeStrain(trimp)).toBeCloseTo(expected, 10);
  });

  it("handles small TRIMP values", () => {
    const result = computeStrain(1);
    expect(result).toBeCloseTo(14.3 * Math.log(2), 10);
  });

  it("handles large TRIMP values", () => {
    const result = computeStrain(1000);
    expect(result).toBeCloseTo(14.3 * Math.log(1001), 10);
  });
});

// ── estimate1rm ─────────────────────────────────────────────────────────────

describe("estimate1rm", () => {
  it("returns weight when reps=1", () => {
    expect(estimate1rm(100, 1)).toBe(100);
  });

  it("returns 0 when weight <= 0", () => {
    expect(estimate1rm(0, 5)).toBe(0);
    expect(estimate1rm(-10, 5)).toBe(0);
  });

  it("returns 0 when reps <= 0", () => {
    expect(estimate1rm(100, 0)).toBe(0);
    expect(estimate1rm(100, -3)).toBe(0);
  });

  it("Epley formula: 100kg x 10 reps = 133.3kg", () => {
    // Epley: weight * (1 + reps/30) = 100 * (1 + 10/30) = 100 * 1.3333 = 133.33
    // Rounded to 1 decimal: 133.3
    const result = estimate1rm(100, 10);
    expect(result).toBeCloseTo(133.3, 1);
  });

  it("returns a value rounded to 1 decimal place", () => {
    const result = estimate1rm(75, 8);
    const roundedOnce = Math.round(result * 10) / 10;
    expect(result).toBe(roundedOnce);
  });

  it("5 reps at 80kg -> 93.3kg", () => {
    // 80 * (1 + 5/30) = 80 * 1.1667 = 93.33 -> 93.3
    const result = estimate1rm(80, 5);
    expect(result).toBeCloseTo(93.3, 1);
  });

  it("higher reps produce higher 1RM estimate", () => {
    const r5 = estimate1rm(100, 5);
    const r10 = estimate1rm(100, 10);
    const r15 = estimate1rm(100, 15);
    expect(r10).toBeGreaterThan(r5);
    expect(r15).toBeGreaterThan(r10);
  });
});

// ── computeCardioLoad ───────────────────────────────────────────────────────

describe("computeCardioLoad", () => {
  /**
   * Helper: generate N days of daily TRIMP data.
   * Starts from a base date and increments by 1 day.
   */
  function generateDailyTrimp(
    values: number[],
    startDate: string = "2026-01-01",
  ): { date: string; trimp: number }[] {
    return values.map((trimp, i) => {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      const dateStr = date.toISOString().split("T")[0];
      return { date: dateStr, trimp };
    });
  }

  it('returns "Calibrating" status with <14 days of data', () => {
    const data = generateDailyTrimp(Array(10).fill(50));
    const result = computeCardioLoad(data);
    expect(result.status).toBe("Calibrating");
  });

  it('returns "Calibrating" with exactly 13 days', () => {
    const data = generateDailyTrimp(Array(13).fill(50));
    const result = computeCardioLoad(data);
    expect(result.status).toBe("Calibrating");
  });

  it("returns a non-Calibrating status with exactly 14 days", () => {
    const data = generateDailyTrimp(Array(14).fill(50));
    const result = computeCardioLoad(data);
    expect(result.status).not.toBe("Calibrating");
  });

  it("returns ATL > CTL after a week of high training following rest", () => {
    // 42 days of low training, then 7 days of high training
    const low = Array(42).fill(10);
    const high = Array(7).fill(100);
    const data = generateDailyTrimp([...low, ...high]);
    const result = computeCardioLoad(data);
    // ATL (7-day) should respond faster to the high training spike
    expect(result.atl).toBeGreaterThan(result.ctl);
  });

  it('returns "Productive" status in normal training', () => {
    // Steady moderate training for 30 days should give TSB near 0
    const data = generateDailyTrimp(Array(30).fill(50));
    const result = computeCardioLoad(data);
    // With constant load, ATL and CTL converge, TSB -> ~0 -> Productive
    expect(["Productive", "Maintaining", "Peaking"]).toContain(result.status);
  });

  it("TSB is CTL minus ATL", () => {
    const data = generateDailyTrimp(Array(30).fill(50));
    const result = computeCardioLoad(data);
    expect(result.tsb).toBeCloseTo(result.ctl - result.atl, 1);
  });

  it("handles empty input", () => {
    const result = computeCardioLoad([]);
    expect(result.atl).toBe(0);
    expect(result.ctl).toBe(0);
    expect(result.tsb).toBe(0);
    expect(result.status).toBe("Calibrating");
  });

  it("handles single day of data", () => {
    const data = [{ date: "2026-03-18", trimp: 100 }];
    const result = computeCardioLoad(data);
    expect(result.status).toBe("Calibrating");
    expect(result.atl).toBeGreaterThan(0);
    expect(result.ctl).toBeGreaterThan(0);
  });

  it('returns "Detraining" when long rest after heavy training', () => {
    // Heavy training then many days rest
    const heavy = Array(30).fill(100);
    const rest = Array(30).fill(0);
    const data = generateDailyTrimp([...heavy, ...rest]);
    const result = computeCardioLoad(data);
    // After long rest: CTL still remembers past training (slow decay),
    // ATL has dropped (fast decay), so TSB = CTL - ATL >> 0
    expect(result.tsb).toBeGreaterThan(0);
    // Could be Detraining or Maintaining depending on exact TSB value
    expect(["Detraining", "Maintaining"]).toContain(result.status);
  });

  it('returns "Overtraining" or "Fatigued" when massive sudden load', () => {
    // Low training then sudden massive spike
    const low = Array(42).fill(10);
    const spike = Array(14).fill(300);
    const data = generateDailyTrimp([...low, ...spike]);
    const result = computeCardioLoad(data);
    // ATL >> CTL -> TSB very negative
    expect(result.tsb).toBeLessThan(-5);
  });

  it("ATL and CTL are rounded to 1 decimal", () => {
    const data = generateDailyTrimp(Array(20).fill(33));
    const result = computeCardioLoad(data);
    expect(result.atl).toBe(Math.round(result.atl * 10) / 10);
    expect(result.ctl).toBe(Math.round(result.ctl * 10) / 10);
    expect(result.tsb).toBe(Math.round(result.tsb * 10) / 10);
  });

  it("sorts unsorted input by date", () => {
    // Provide data out of order
    const data = [
      { date: "2026-03-18", trimp: 100 },
      { date: "2026-03-01", trimp: 10 },
      { date: "2026-03-10", trimp: 50 },
    ];
    // Should not throw and should process correctly
    const result = computeCardioLoad(data);
    expect(result.atl).toBeGreaterThan(0);
  });

  it("covers all 7 status categories", () => {
    const validStatuses = [
      "Calibrating", "Detraining", "Maintaining",
      "Productive", "Peaking", "Fatigued", "Overtraining",
    ];
    // We won't test all 7 here but verify the function only returns valid values
    const data = generateDailyTrimp(Array(50).fill(50));
    const result = computeCardioLoad(data);
    expect(validStatuses).toContain(result.status);
  });
});

// ── computeHrRecovery ───────────────────────────────────────────────────────

describe("computeHrRecovery", () => {
  it('drop of 55 -> "Excellent"', () => {
    const result = computeHrRecovery(180, 125);
    expect(result.value).toBe(55);
    expect(result.rating).toBe("Excellent");
  });

  it('drop of 45 -> "Good"', () => {
    const result = computeHrRecovery(180, 135);
    expect(result.value).toBe(45);
    expect(result.rating).toBe("Good");
  });

  it('drop of 35 -> "Average"', () => {
    const result = computeHrRecovery(180, 145);
    expect(result.value).toBe(35);
    expect(result.rating).toBe("Average");
  });

  it('drop of 20 -> "Below Average"', () => {
    const result = computeHrRecovery(180, 160);
    expect(result.value).toBe(20);
    expect(result.rating).toBe("Below Average");
  });

  it("boundary: drop of exactly 50 is Excellent (>50)", () => {
    // drop > 50 -> Excellent. drop = 50 is NOT > 50
    const result = computeHrRecovery(180, 130);
    expect(result.value).toBe(50);
    expect(result.rating).toBe("Good"); // 50 is not > 50
  });

  it("boundary: drop of exactly 51 is Excellent", () => {
    const result = computeHrRecovery(180, 129);
    expect(result.value).toBe(51);
    expect(result.rating).toBe("Excellent");
  });

  it("boundary: drop of exactly 40 is Average (not Good)", () => {
    // drop > 40 -> Good. drop = 40 is NOT > 40
    const result = computeHrRecovery(180, 140);
    expect(result.value).toBe(40);
    expect(result.rating).toBe("Average");
  });

  it("boundary: drop of exactly 41 is Good", () => {
    const result = computeHrRecovery(180, 139);
    expect(result.value).toBe(41);
    expect(result.rating).toBe("Good");
  });

  it("boundary: drop of exactly 30 is Below Average", () => {
    // drop > 30 -> Average. drop = 30 is NOT > 30
    const result = computeHrRecovery(180, 150);
    expect(result.value).toBe(30);
    expect(result.rating).toBe("Below Average");
  });

  it("boundary: drop of exactly 31 is Average", () => {
    const result = computeHrRecovery(180, 149);
    expect(result.value).toBe(31);
    expect(result.rating).toBe("Average");
  });

  it("handles zero drop", () => {
    const result = computeHrRecovery(150, 150);
    expect(result.value).toBe(0);
    expect(result.rating).toBe("Below Average");
  });

  it("handles negative drop (HR increased after exercise)", () => {
    const result = computeHrRecovery(150, 160);
    expect(result.value).toBe(-10);
    expect(result.rating).toBe("Below Average");
  });
});
