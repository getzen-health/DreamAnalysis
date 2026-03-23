/**
 * i18n — Internationalization framework for AntarAI.
 *
 * Provides a simple translation function `t("key.path")` that resolves
 * nested keys from locale JSON files. Supports English (default),
 * Hindi (hi), and Telugu (te).
 *
 * Usage:
 *   import { t, setLanguage, getLanguage, SUPPORTED_LANGUAGES } from "@/lib/i18n";
 *
 *   t("nav.today")           // "Today" (en) or "आज" (hi) or "ఈరోజు" (te)
 *   setLanguage("hi")        // Switch to Hindi
 *   getLanguage()            // "hi"
 */

import en from "@/locales/en.json";
import hi from "@/locales/hi.json";
import te from "@/locales/te.json";

// ── Types ──────────────────────────────────────────────────────────────────

export interface Language {
  code: string;
  name: string;
  nativeName: string;
}

type NestedRecord = { [key: string]: string | NestedRecord };

// ── Constants ──────────────────────────────────────────────────────────────

const STORAGE_KEY = "ndw_language";
const DEFAULT_LANGUAGE = "en";

export const SUPPORTED_LANGUAGES: Language[] = [
  { code: "en", name: "English", nativeName: "English" },
  { code: "hi", name: "Hindi", nativeName: "हिन्दी" },
  { code: "te", name: "Telugu", nativeName: "తెలుగు" },
];

const LOCALE_MAP: Record<string, NestedRecord> = {
  en: en as NestedRecord,
  hi: hi as NestedRecord,
  te: te as NestedRecord,
};

// ── State ──────────────────────────────────────────────────────────────────

let _currentLanguage: string = DEFAULT_LANGUAGE;

// Initialize from localStorage if available
if (typeof window !== "undefined") {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored && LOCALE_MAP[stored]) {
    _currentLanguage = stored;
  }
}

// ── Core Functions ─────────────────────────────────────────────────────────

/**
 * Resolve a dotted key path from a nested object.
 * Example: resolve("nav.today", enLocale) -> "Today"
 */
function resolve(keyPath: string, obj: NestedRecord): string | undefined {
  const keys = keyPath.split(".");
  let current: string | NestedRecord = obj;

  for (const key of keys) {
    if (current === undefined || current === null || typeof current === "string") {
      return undefined;
    }
    current = (current as NestedRecord)[key];
  }

  return typeof current === "string" ? current : undefined;
}

/**
 * Translate a key path to the current language.
 * Falls back to English if key not found in current locale.
 * Falls back to the key itself if not found in English either.
 *
 * @param keyPath - Dotted path like "nav.today" or "emotions.happy"
 * @returns Translated string
 */
export function t(keyPath: string): string {
  const locale = LOCALE_MAP[_currentLanguage];

  // Try current language first
  if (locale) {
    const value = resolve(keyPath, locale);
    if (value !== undefined) return value;
  }

  // Fall back to English
  if (_currentLanguage !== DEFAULT_LANGUAGE) {
    const enValue = resolve(keyPath, LOCALE_MAP[DEFAULT_LANGUAGE]);
    if (enValue !== undefined) return enValue;
  }

  // Last resort: return the key itself
  return keyPath;
}

/**
 * Set the active language. Persists to localStorage.
 *
 * @param code - ISO 639-1 language code ("en", "hi", "te")
 */
export function setLanguage(code: string): void {
  if (!LOCALE_MAP[code]) {
    console.warn(`[i18n] Unsupported language: "${code}". Falling back to "${DEFAULT_LANGUAGE}".`);
    code = DEFAULT_LANGUAGE;
  }
  _currentLanguage = code;
  if (typeof window !== "undefined") {
    localStorage.setItem(STORAGE_KEY, code);
  }
}

/**
 * Get the current active language code.
 */
export function getLanguage(): string {
  return _currentLanguage;
}

/**
 * Check if a language code is supported.
 */
export function isSupported(code: string): boolean {
  return code in LOCALE_MAP;
}

/**
 * Get all available translations for a key (useful for debugging).
 */
export function getAllTranslations(keyPath: string): Record<string, string | undefined> {
  const result: Record<string, string | undefined> = {};
  for (const [code, locale] of Object.entries(LOCALE_MAP)) {
    result[code] = resolve(keyPath, locale);
  }
  return result;
}
