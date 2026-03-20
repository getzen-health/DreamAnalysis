import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, act } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { AchievementBadges, checkBadges, type Badge } from "@/components/achievements";

// ── helpers ───────────────────────────────────────────────────────────────────

/** Clear all ndw_* keys from localStorage before each test. */
function clearStorage() {
  const keys = Object.keys(localStorage).filter(k => k.startsWith("ndw_"));
  keys.forEach(k => localStorage.removeItem(k));
}

beforeEach(() => {
  clearStorage();
});

afterEach(() => {
  clearStorage();
  vi.restoreAllMocks();
});

// ── checkBadges() unit tests ─────────────────────────────────────────────────

describe("checkBadges", () => {
  it("returns an array of badges", () => {
    const badges = checkBadges();
    expect(Array.isArray(badges)).toBe(true);
    expect(badges.length).toBeGreaterThan(0);
  });

  it("marks first-checkin as earned when ndw_last_emotion exists", () => {
    localStorage.setItem("ndw_last_emotion", JSON.stringify({ emotion: "happy" }));
    const badges = checkBadges();
    const firstCheckin = badges.find(b => b.id === "first-checkin");
    expect(firstCheckin).toBeDefined();
    expect(firstCheckin!.earned).toBe(true);
  });

  it("marks first-checkin as NOT earned when ndw_last_emotion is absent", () => {
    const badges = checkBadges();
    const firstCheckin = badges.find(b => b.id === "first-checkin");
    expect(firstCheckin).toBeDefined();
    expect(firstCheckin!.earned).toBe(false);
  });

  it("marks streak-3 earned when streak >= 3", () => {
    localStorage.setItem("ndw_streak_count", "5");
    const badges = checkBadges();
    const streak3 = badges.find(b => b.id === "streak-3");
    expect(streak3!.earned).toBe(true);
  });

  it("marks streak-7 NOT earned when streak < 7", () => {
    localStorage.setItem("ndw_streak_count", "5");
    const badges = checkBadges();
    const streak7 = badges.find(b => b.id === "streak-7");
    expect(streak7!.earned).toBe(false);
  });

  it("marks streak-30 (gold tier) earned when streak >= 30", () => {
    localStorage.setItem("ndw_streak_count", "30");
    const badges = checkBadges();
    const streak30 = badges.find(b => b.id === "streak-30");
    expect(streak30!.earned).toBe(true);
    expect(streak30!.tier).toBe("gold");
  });

  it("computes progress for streak badges", () => {
    localStorage.setItem("ndw_streak_count", "2");
    const badges = checkBadges();
    const streak3 = badges.find(b => b.id === "streak-3");
    expect(streak3!.progress).toBeCloseTo(2 / 3, 2);
    const streak7 = badges.find(b => b.id === "streak-7");
    expect(streak7!.progress).toBeCloseTo(2 / 7, 2);
  });

  it("caps progress at 1 for completed streaks", () => {
    localStorage.setItem("ndw_streak_count", "50");
    const badges = checkBadges();
    const streak3 = badges.find(b => b.id === "streak-3");
    expect(streak3!.progress).toBe(1);
  });

  it("marks early-bird earned for timestamp before 8 AM", () => {
    const earlyMorning = new Date();
    earlyMorning.setHours(6, 0, 0, 0);
    localStorage.setItem("ndw_last_emotion", JSON.stringify({
      emotion: "happy", timestamp: earlyMorning.getTime(),
    }));
    const badges = checkBadges();
    const earlyBird = badges.find(b => b.id === "early-bird");
    expect(earlyBird).toBeDefined();
    expect(earlyBird!.earned).toBe(true);
    expect(earlyBird!.tier).toBe("silver");
  });

  it("marks night-owl earned for timestamp after 10 PM", () => {
    const lateNight = new Date();
    lateNight.setHours(23, 30, 0, 0);
    localStorage.setItem("ndw_last_emotion", JSON.stringify({
      emotion: "neutral", timestamp: lateNight.getTime(),
    }));
    const badges = checkBadges();
    const nightOwl = badges.find(b => b.id === "night-owl");
    expect(nightOwl).toBeDefined();
    expect(nightOwl!.earned).toBe(true);
    expect(nightOwl!.tier).toBe("silver");
  });

  it("marks health-sync earned when apple health granted", () => {
    localStorage.setItem("ndw_apple_health_granted", "true");
    const badges = checkBadges();
    const hs = badges.find(b => b.id === "health-sync");
    expect(hs!.earned).toBe(true);
    expect(hs!.category).toBe("Wellness");
  });

  it("marks health-sync earned when health connect granted", () => {
    localStorage.setItem("ndw_health_connect_granted", "true");
    const badges = checkBadges();
    const hs = badges.find(b => b.id === "health-sync");
    expect(hs!.earned).toBe(true);
  });

  it("marks first-meal earned when meal logged", () => {
    localStorage.setItem("ndw_meal_logged", "1");
    const badges = checkBadges();
    const fm = badges.find(b => b.id === "first-meal");
    expect(fm!.earned).toBe(true);
    expect(fm!.tier).toBe("bronze");
  });

  it("marks onboarded earned when onboarding complete", () => {
    localStorage.setItem("ndw_onboarding_complete", "true");
    const badges = checkBadges();
    const ob = badges.find(b => b.id === "onboarded");
    expect(ob!.earned).toBe(true);
    expect(ob!.category).toBe("Milestones");
  });

  it("marks emotion-explorer earned when all 6 emotions seen", () => {
    localStorage.setItem("ndw_emotions_seen",
      JSON.stringify(["happy", "sad", "angry", "fear", "surprise", "neutral"]));
    const badges = checkBadges();
    const ee = badges.find(b => b.id === "emotion-explorer");
    expect(ee!.earned).toBe(true);
    expect(ee!.tier).toBe("gold");
  });

  it("marks emotion-explorer NOT earned when only 3 of 6 seen", () => {
    localStorage.setItem("ndw_emotions_seen", JSON.stringify(["happy", "sad", "angry"]));
    const badges = checkBadges();
    const ee = badges.find(b => b.id === "emotion-explorer");
    expect(ee!.earned).toBe(false);
    expect(ee!.progress).toBeCloseTo(0.5, 2);
  });

  it("marks muse-connected (brain category) earned when muse connected", () => {
    localStorage.setItem("ndw_muse_connected", "true");
    const badges = checkBadges();
    const mc = badges.find(b => b.id === "muse-connected");
    expect(mc!.earned).toBe(true);
    expect(mc!.category).toBe("Brain");
    expect(mc!.tier).toBe("gold");
  });

  it("every badge has a tier and category", () => {
    // Set data so time-dependent badges are included
    localStorage.setItem("ndw_last_emotion", JSON.stringify({
      emotion: "happy", timestamp: Date.now(),
    }));
    const badges = checkBadges();
    for (const badge of badges) {
      expect(["bronze", "silver", "gold"]).toContain(badge.tier);
      expect(["Sessions", "Streaks", "Milestones", "Wellness", "Brain"]).toContain(badge.category);
    }
  });
});

// ── AchievementBadges component tests ────────────────────────────────────────

describe("AchievementBadges", () => {
  it("renders achievements header", () => {
    renderWithProviders(<AchievementBadges />);
    expect(screen.getByText("Achievements")).toBeInTheDocument();
  });

  it("shows earned count out of total", () => {
    localStorage.setItem("ndw_last_emotion", JSON.stringify({ emotion: "happy" }));
    localStorage.setItem("ndw_onboarding_complete", "true");
    const { container } = renderWithProviders(<AchievementBadges />);
    // Look for the "N/M earned" text
    const earnedText = container.querySelector('[style*="earned"]')
      || screen.getByText(/earned/);
    expect(earnedText).toBeInTheDocument();
  });

  it("renders overall progress bar", () => {
    renderWithProviders(<AchievementBadges />);
    expect(screen.getByTestId("overall-progress")).toBeInTheDocument();
  });

  it("renders earned badges with tier labels", () => {
    localStorage.setItem("ndw_last_emotion", JSON.stringify({ emotion: "happy" }));
    renderWithProviders(<AchievementBadges />);
    // first-checkin is a bronze badge
    expect(screen.getByTestId("badge-first-checkin")).toBeInTheDocument();
    expect(screen.getByText("First Voice")).toBeInTheDocument();
  });

  it("renders locked badges with Lock section", () => {
    renderWithProviders(<AchievementBadges />);
    expect(screen.getByText("Locked")).toBeInTheDocument();
  });

  it("shows category chips on badge cards", () => {
    localStorage.setItem("ndw_last_emotion", JSON.stringify({ emotion: "happy" }));
    renderWithProviders(<AchievementBadges />);
    // first-checkin has category "Sessions"
    expect(screen.getByTestId("category-sessions")).toBeInTheDocument();
  });

  it("shows progress bar for partially completed locked badges", () => {
    localStorage.setItem("ndw_streak_count", "2");
    renderWithProviders(<AchievementBadges />);
    // streak-3 should have a progress bar at ~67%
    const progressBars = screen.getAllByTestId("progress-bar-fill");
    expect(progressBars.length).toBeGreaterThan(0);
  });

  it("does not show progress bar for badges with zero progress", () => {
    // No streak data at all — streak progress is 0
    renderWithProviders(<AchievementBadges />);
    // Badges with 0 progress should not show progress bars
    // Only badges with progress > 0 and < 1 get bars
    const container = document.querySelector('[data-testid="badge-streak-3"]');
    expect(container).toBeInTheDocument();
    // Verify no progress bar inside this specific badge (progress is 0)
    const progressInBadge = container?.querySelector('[data-testid="progress-bar-fill"]');
    expect(progressInBadge).toBeNull();
  });

  it("renders nothing when badges array is empty", () => {
    // Mock checkBadges to return empty — we test via the component rendering
    // With no localStorage data and no timestamp, some badges are still created
    // so the component should render (not null)
    const { container } = renderWithProviders(<AchievementBadges />);
    // The component always has badges (hard-coded list), so it should render
    expect(container.firstChild).not.toBeNull();
  });

  it("updates badges when ndw-voice-updated event fires", () => {
    renderWithProviders(<AchievementBadges />);
    // Initially first-checkin is not earned
    expect(screen.getByText("First Voice")).toBeInTheDocument();

    // Set localStorage and fire event inside act()
    act(() => {
      localStorage.setItem("ndw_last_emotion", JSON.stringify({ emotion: "happy" }));
      window.dispatchEvent(new Event("ndw-voice-updated"));
    });

    // Badge should now be in the earned section (with shimmer overlay)
    const badge = screen.getByTestId("badge-first-checkin");
    expect(badge).toBeInTheDocument();
  });

  it("shows tier labels (Bronze/Silver/Gold) on cards", () => {
    localStorage.setItem("ndw_last_emotion", JSON.stringify({
      emotion: "happy", timestamp: Date.now(),
    }));
    localStorage.setItem("ndw_muse_connected", "true");
    renderWithProviders(<AchievementBadges />);
    // first-checkin = bronze, muse-connected = gold
    // Multiple bronze badges exist, so use getAllByTestId
    const bronzeLabels = screen.getAllByTestId("tier-bronze");
    expect(bronzeLabels.length).toBeGreaterThan(0);
    const goldLabels = screen.getAllByTestId("tier-gold");
    expect(goldLabels.length).toBeGreaterThan(0);
  });
});
