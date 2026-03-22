import { describe, it, expect } from "vitest";
import {
  sanitizeNotificationText,
  PHI_PATTERNS,
} from "@/lib/notification-sanitizer";

describe("sanitizeNotificationText", () => {
  // ── Core sanitization: must strip PHI from notification text ──────────

  it("replaces stress level percentage with generic message", () => {
    const result = sanitizeNotificationText("Your stress level is 78%");
    expect(result).not.toMatch(/stress level/i);
    expect(result).not.toMatch(/78%/);
    expect(result.length).toBeGreaterThan(0);
  });

  it("replaces EEG-specific content with generic message", () => {
    const result = sanitizeNotificationText("EEG shows elevated anxiety");
    expect(result).not.toMatch(/EEG/i);
    expect(result).not.toMatch(/anxiety/i);
    expect(result.length).toBeGreaterThan(0);
  });

  it("replaces detected mood with generic message", () => {
    const result = sanitizeNotificationText("Your mood was detected as Sad");
    expect(result).not.toMatch(/mood.*detected/i);
    expect(result).not.toMatch(/\bsad\b/i);
    expect(result.length).toBeGreaterThan(0);
  });

  it("replaces focus percentage with generic message", () => {
    const result = sanitizeNotificationText("Focus level: 92%");
    expect(result).not.toMatch(/focus level/i);
    expect(result).not.toMatch(/92%/);
  });

  it("replaces heart rate values with generic message", () => {
    const result = sanitizeNotificationText("Your heart rate is 120 bpm");
    expect(result).not.toMatch(/heart rate/i);
    expect(result).not.toMatch(/120/);
  });

  it("replaces sleep quality specifics with generic message", () => {
    const result = sanitizeNotificationText("Sleep quality score: 45/100");
    expect(result).not.toMatch(/sleep quality/i);
    expect(result).not.toMatch(/45/);
  });

  it("replaces HRV values with generic message", () => {
    const result = sanitizeNotificationText("HRV dropped to 22ms");
    expect(result).not.toMatch(/HRV/i);
    expect(result).not.toMatch(/22ms/);
  });

  it("replaces emotion classification details", () => {
    const result = sanitizeNotificationText("Emotion detected: angry (85% confidence)");
    expect(result).not.toMatch(/emotion detected/i);
    expect(result).not.toMatch(/angry/i);
    expect(result).not.toMatch(/85%/);
  });

  it("replaces relaxation index values", () => {
    const result = sanitizeNotificationText("Relaxation index is 0.3 — below normal");
    expect(result).not.toMatch(/relaxation index/i);
    expect(result).not.toMatch(/0\.3/);
  });

  it("replaces brain session specific data", () => {
    const result = sanitizeNotificationText("Brain session: alpha waves at 12Hz, theta elevated");
    expect(result).not.toMatch(/alpha waves/i);
    expect(result).not.toMatch(/theta/i);
    expect(result).not.toMatch(/12Hz/);
  });

  // ── Safe text passes through unchanged ────────────────────────────────

  it("does not alter safe generic messages", () => {
    const safe = "You have a new wellness insight";
    expect(sanitizeNotificationText(safe)).toBe(safe);
  });

  it("does not alter session reminder text", () => {
    const safe = "Time for your daily check-in";
    expect(sanitizeNotificationText(safe)).toBe(safe);
  });

  it("does not alter streak messages", () => {
    const safe = "5-day streak and counting";
    expect(sanitizeNotificationText(safe)).toBe(safe);
  });

  it("does not alter achievement messages", () => {
    const safe = "Achievement unlocked: First week complete";
    expect(sanitizeNotificationText(safe)).toBe(safe);
  });

  // ── Edge cases ────────────────────────────────────────────────────────

  it("returns empty string for empty input", () => {
    expect(sanitizeNotificationText("")).toBe("");
  });

  it("handles multiple PHI patterns in one string", () => {
    const result = sanitizeNotificationText(
      "Your stress level is 78% and mood was detected as Sad"
    );
    expect(result).not.toMatch(/stress level/i);
    expect(result).not.toMatch(/mood.*detected/i);
    expect(result).not.toMatch(/78%/);
    expect(result).not.toMatch(/\bsad\b/i);
  });

  it("is case-insensitive for PHI detection", () => {
    const result = sanitizeNotificationText("YOUR STRESS LEVEL IS 78%");
    expect(result).not.toMatch(/stress level/i);
  });
});

describe("PHI_PATTERNS", () => {
  it("exports a non-empty array of pattern objects", () => {
    expect(Array.isArray(PHI_PATTERNS)).toBe(true);
    expect(PHI_PATTERNS.length).toBeGreaterThan(0);
  });

  it("each pattern has a regex and a replacement string", () => {
    for (const p of PHI_PATTERNS) {
      expect(p.pattern).toBeInstanceOf(RegExp);
      expect(typeof p.replacement).toBe("string");
      expect(p.replacement.length).toBeGreaterThan(0);
    }
  });
});
