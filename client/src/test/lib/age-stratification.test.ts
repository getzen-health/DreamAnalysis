import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  classifyAgeGroup,
  getAgeGroupInfo,
  getAgeAdjustments,
  applyAgeAdjustments,
  getUserAge,
  setUserAge,
  AGE_GROUPS,
} from "@/lib/age-stratification";

describe("age-stratification", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  describe("classifyAgeGroup", () => {
    it("classifies children (6-12)", () => {
      expect(classifyAgeGroup(6)).toBe("child");
      expect(classifyAgeGroup(9)).toBe("child");
      expect(classifyAgeGroup(12)).toBe("child");
    });

    it("classifies teens (13-17)", () => {
      expect(classifyAgeGroup(13)).toBe("teen");
      expect(classifyAgeGroup(15)).toBe("teen");
      expect(classifyAgeGroup(17)).toBe("teen");
    });

    it("classifies adults (18-64)", () => {
      expect(classifyAgeGroup(18)).toBe("adult");
      expect(classifyAgeGroup(30)).toBe("adult");
      expect(classifyAgeGroup(64)).toBe("adult");
    });

    it("classifies seniors (65+)", () => {
      expect(classifyAgeGroup(65)).toBe("senior");
      expect(classifyAgeGroup(80)).toBe("senior");
      expect(classifyAgeGroup(100)).toBe("senior");
    });

    it("treats age below 6 as child", () => {
      expect(classifyAgeGroup(3)).toBe("child");
      expect(classifyAgeGroup(1)).toBe("child");
    });
  });

  describe("getAgeGroupInfo", () => {
    it("returns correct info for each age group", () => {
      const childInfo = getAgeGroupInfo(8);
      expect(childInfo.id).toBe("child");
      expect(childInfo.ageRange).toBe("6-12");

      const adultInfo = getAgeGroupInfo(30);
      expect(adultInfo.id).toBe("adult");
    });
  });

  describe("getAgeAdjustments", () => {
    it("returns no-op adjustments for adults", () => {
      const adj = getAgeAdjustments(30);
      expect(adj.arousalMultiplier).toBe(1.0);
      expect(adj.stressMultiplier).toBe(1.0);
      expect(adj.volatilityDampening).toBe(1.0);
      expect(adj.focusBaselineOffset).toBe(0.0);
    });

    it("returns no-op adjustments for null age", () => {
      const adj = getAgeAdjustments(null);
      expect(adj.arousalMultiplier).toBe(1.0);
    });

    it("returns dampened adjustments for children", () => {
      const adj = getAgeAdjustments(8);
      expect(adj.arousalMultiplier).toBeLessThan(1.0);
      expect(adj.stressMultiplier).toBeLessThan(1.0);
      expect(adj.focusBaselineOffset).toBeLessThan(0);
    });

    it("returns boosted adjustments for seniors", () => {
      const adj = getAgeAdjustments(70);
      expect(adj.arousalMultiplier).toBeGreaterThan(1.0);
      expect(adj.stressMultiplier).toBeGreaterThan(1.0);
      expect(adj.focusBaselineOffset).toBeGreaterThan(0);
    });
  });

  describe("applyAgeAdjustments", () => {
    const raw = { stress: 0.5, focus: 0.5, arousal: 0.5 };

    it("returns raw values for adults (no adjustment)", () => {
      const result = applyAgeAdjustments(raw, 30);
      expect(result.stress).toBe(0.5);
      expect(result.focus).toBe(0.5);
      expect(result.arousal).toBe(0.5);
    });

    it("lowers stress for children", () => {
      const result = applyAgeAdjustments(raw, 8);
      expect(result.stress).toBeLessThan(0.5);
    });

    it("boosts arousal for seniors", () => {
      const result = applyAgeAdjustments(raw, 70);
      expect(result.arousal).toBeGreaterThan(0.5);
    });

    it("clamps values to 0-1 range", () => {
      const extreme = { stress: 0.95, focus: 0.02, arousal: 0.99 };
      const result = applyAgeAdjustments(extreme, 70);
      expect(result.stress).toBeLessThanOrEqual(1);
      expect(result.focus).toBeGreaterThanOrEqual(0);
      expect(result.arousal).toBeLessThanOrEqual(1);
    });

    it("returns raw values when age is null", () => {
      const result = applyAgeAdjustments(raw, null);
      expect(result.stress).toBe(0.5);
      expect(result.focus).toBe(0.5);
      expect(result.arousal).toBe(0.5);
    });
  });

  describe("localStorage persistence", () => {
    it("returns null when no age is stored", () => {
      expect(getUserAge()).toBeNull();
    });

    it("stores and retrieves age", () => {
      setUserAge(25);
      expect(getUserAge()).toBe(25);
    });

    it("rounds fractional age", () => {
      setUserAge(25.7);
      expect(getUserAge()).toBe(26);
    });

    it("rejects invalid stored values", () => {
      localStorage.setItem("ndw_user_age", "abc");
      expect(getUserAge()).toBeNull();

      localStorage.setItem("ndw_user_age", "0");
      expect(getUserAge()).toBeNull();

      localStorage.setItem("ndw_user_age", "150");
      expect(getUserAge()).toBeNull();
    });
  });

  describe("AGE_GROUPS constant", () => {
    it("has 4 groups", () => {
      expect(AGE_GROUPS).toHaveLength(4);
    });

    it("covers all group IDs", () => {
      const ids = AGE_GROUPS.map((g) => g.id);
      expect(ids).toEqual(["child", "teen", "adult", "senior"]);
    });
  });
});
