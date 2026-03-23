import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock posthog-js since it's not installed (placeholder dependency)
vi.mock("posthog-js", () => ({
  default: {
    init: vi.fn(),
    capture: vi.fn(),
    identify: vi.fn(),
    reset: vi.fn(),
  },
}));

import {
  hasAnalyticsConsent,
  setAnalyticsConsent,
  trackPageView,
  trackVoiceAnalysisCompleted,
  trackEegSessionStarted,
  trackMoodLogged,
  trackAchievementUnlocked,
} from "@/lib/posthog";

describe("posthog analytics", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  describe("consent management", () => {
    it("defaults to no consent", () => {
      expect(hasAnalyticsConsent()).toBe(false);
    });

    it("grants consent", () => {
      setAnalyticsConsent(true);
      expect(hasAnalyticsConsent()).toBe(true);
      expect(localStorage.getItem("ndw_analytics_consent")).toBe("true");
    });

    it("revokes consent", () => {
      setAnalyticsConsent(true);
      expect(hasAnalyticsConsent()).toBe(true);

      setAnalyticsConsent(false);
      expect(hasAnalyticsConsent()).toBe(false);
      expect(localStorage.getItem("ndw_analytics_consent")).toBe("false");
    });
  });

  describe("tracking functions", () => {
    it("does not throw when called without consent", () => {
      // PostHog is not initialized. These should all be silent no-ops.
      expect(() => trackPageView("/")).not.toThrow();
      expect(() => trackVoiceAnalysisCompleted({ emotion: "happy" })).not.toThrow();
      expect(() => trackEegSessionStarted({ device: "Muse 2" })).not.toThrow();
      expect(() => trackMoodLogged({ mood: 7 })).not.toThrow();
      expect(() => trackAchievementUnlocked({ achievement: "first_session" })).not.toThrow();
    });

    it("does not throw when called with consent but no PostHog loaded", () => {
      setAnalyticsConsent(true);
      expect(() => trackPageView("/discover")).not.toThrow();
      expect(() => trackMoodLogged({ mood: 5, energy: 6 })).not.toThrow();
    });
  });
});
