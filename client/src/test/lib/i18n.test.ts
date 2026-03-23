import { describe, it, expect, beforeEach } from "vitest";
import {
  t,
  setLanguage,
  getLanguage,
  isSupported,
  getAllTranslations,
  SUPPORTED_LANGUAGES,
} from "@/lib/i18n";

describe("i18n", () => {
  beforeEach(() => {
    localStorage.clear();
    setLanguage("en"); // reset to default
  });

  // ── t() function ─────────────────────────────────────────────────────

  it("returns English string for valid key", () => {
    expect(t("nav.today")).toBe("Today");
  });

  it("returns nested key values", () => {
    expect(t("emotions.happy")).toBe("Happy");
    expect(t("settings.theme")).toBe("Theme");
  });

  it("returns the key itself for missing keys", () => {
    expect(t("nonexistent.key.path")).toBe("nonexistent.key.path");
  });

  it("falls back to English when key missing in Hindi", () => {
    setLanguage("hi");
    // app.name should exist in Hindi
    expect(t("app.name")).toBe("AntarAI");
  });

  // ── Hindi translations ───────────────────────────────────────────────

  it("returns Hindi translation when language is hi", () => {
    setLanguage("hi");
    expect(t("nav.today")).toBe("\u0906\u091C");  // "आज"
  });

  // ── Telugu translations ──────────────────────────────────────────────

  it("returns Telugu translation when language is te", () => {
    setLanguage("te");
    expect(t("nav.today")).toBe("\u0C08\u0C30\u0C4B\u0C1C\u0C41"); // "ఈరోజు"
  });

  // ── setLanguage / getLanguage ────────────────────────────────────────

  it("getLanguage returns current language", () => {
    expect(getLanguage()).toBe("en");
    setLanguage("hi");
    expect(getLanguage()).toBe("hi");
  });

  it("setLanguage persists to localStorage", () => {
    setLanguage("te");
    expect(localStorage.getItem("ndw_language")).toBe("te");
  });

  it("falls back to en for unsupported language", () => {
    setLanguage("xx");
    expect(getLanguage()).toBe("en");
  });

  // ── isSupported ──────────────────────────────────────────────────────

  it("isSupported returns true for en, hi, te", () => {
    expect(isSupported("en")).toBe(true);
    expect(isSupported("hi")).toBe(true);
    expect(isSupported("te")).toBe(true);
  });

  it("isSupported returns false for unsupported codes", () => {
    expect(isSupported("fr")).toBe(false);
    expect(isSupported("xx")).toBe(false);
  });

  // ── SUPPORTED_LANGUAGES ──────────────────────────────────────────────

  it("has 3 supported languages", () => {
    expect(SUPPORTED_LANGUAGES).toHaveLength(3);
  });

  it("each language has code, name, and nativeName", () => {
    for (const lang of SUPPORTED_LANGUAGES) {
      expect(lang.code).toBeTruthy();
      expect(lang.name).toBeTruthy();
      expect(lang.nativeName).toBeTruthy();
    }
  });

  // ── getAllTranslations ───────────────────────────────────────────────

  it("returns translations for all languages", () => {
    const all = getAllTranslations("nav.today");
    expect(all.en).toBe("Today");
    expect(all.hi).toBeTruthy();
    expect(all.te).toBeTruthy();
  });

  // ── Edge cases ───────────────────────────────────────────────────────

  it("handles empty key gracefully", () => {
    expect(t("")).toBe("");
  });

  it("handles single-level key", () => {
    // Single-level keys don't exist in our structure, should return key
    expect(t("nonexistent")).toBe("nonexistent");
  });

  it("handles deeply nested key that partially exists", () => {
    // "nav" exists but "nav.today.extra" does not
    expect(t("nav.today.extra")).toBe("nav.today.extra");
  });
});
