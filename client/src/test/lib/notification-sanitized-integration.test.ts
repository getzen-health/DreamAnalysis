/**
 * Integration tests: addNotification must always sanitize PHI before saving.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";

// Mock Supabase before importing the module under test
vi.mock("@/lib/supabase-browser", () => ({
  getSupabase: vi.fn().mockResolvedValue(null),
}));

import {
  addNotification,
  loadNotifications,
  clearNotifications,
} from "@/pages/notifications";

describe("addNotification PHI sanitization", () => {
  beforeEach(() => {
    localStorage.clear();
    clearNotifications();
  });

  it("sanitizes stress level data from notification body", () => {
    addNotification({
      type: "voice_result",
      title: "Voice Analysis",
      body: "Your stress level is 78%",
    });
    const [saved] = loadNotifications();
    expect(saved.body).not.toMatch(/stress level/i);
    expect(saved.body).not.toMatch(/78%/);
  });

  it("sanitizes EEG content from notification body", () => {
    addNotification({
      type: "eeg_summary",
      title: "Brain Session",
      body: "EEG shows elevated anxiety",
    });
    const [saved] = loadNotifications();
    expect(saved.body).not.toMatch(/EEG/i);
    expect(saved.body).not.toMatch(/anxiety/i);
  });

  it("sanitizes mood detection from notification title", () => {
    addNotification({
      type: "voice_result",
      title: "Your mood was detected as Sad",
      body: "Check your daily update",
    });
    const [saved] = loadNotifications();
    expect(saved.title).not.toMatch(/mood.*detected/i);
    expect(saved.title).not.toMatch(/\bsad\b/i);
  });

  it("does not alter safe notification text", () => {
    addNotification({
      type: "streak",
      title: "Check-in streak",
      body: "5-day streak and counting",
    });
    const [saved] = loadNotifications();
    expect(saved.title).toBe("Check-in streak");
    expect(saved.body).toBe("5-day streak and counting");
  });

  it("sanitizes both title and body", () => {
    addNotification({
      type: "voice_result",
      title: "Your stress level is 92%",
      body: "EEG shows elevated anxiety",
    });
    const [saved] = loadNotifications();
    expect(saved.title).not.toMatch(/stress level/i);
    expect(saved.body).not.toMatch(/EEG/i);
  });
});
