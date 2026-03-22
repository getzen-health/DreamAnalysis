/**
 * HIPAA-safe notification text sanitizer.
 *
 * Push notifications are visible on lock screens, notification centers,
 * and potentially to anyone with physical access to the device.
 * Protected Health Information (PHI) must NEVER appear in notification text.
 *
 * This module strips health-specific data from notification strings and
 * replaces them with generic, safe alternatives that link users to
 * in-app content instead.
 */

// ── PHI pattern definitions ──────────────────────────────────────────────

export interface PHIPattern {
  /** Regex that matches PHI-containing text (case-insensitive) */
  pattern: RegExp;
  /** Safe replacement text */
  replacement: string;
}

/**
 * Ordered list of PHI patterns to detect and replace.
 * Patterns are applied in order; more specific patterns come first.
 */
export const PHI_PATTERNS: PHIPattern[] = [
  // Stress level with percentage
  {
    pattern: /\b(?:your\s+)?stress\s+level\s+(?:is\s+)?[\d.]+%?/gi,
    replacement: "You have a new wellness insight",
  },
  // Focus level with percentage
  {
    pattern: /\b(?:your\s+)?focus\s+level[:\s]+[\d.]+%?/gi,
    replacement: "You have a new wellness insight",
  },
  // EEG-specific content (elevated anxiety, brain patterns, etc.)
  {
    pattern: /\bEEG\s+shows?\b[^.!?\n]*/gi,
    replacement: "New brain session summary available",
  },
  // Mood detection results
  {
    pattern: /\b(?:your\s+)?mood\s+(?:was\s+)?detected\s+(?:as\s+)?\w+/gi,
    replacement: "Check in with your daily update",
  },
  // Emotion detection with confidence
  {
    pattern: /\b(?:emotion\s+detected)[:\s]+\w+(?:\s*\([\d.]+%\s*confidence\))?/gi,
    replacement: "You have a new wellness insight",
  },
  // Heart rate values
  {
    pattern: /\b(?:your\s+)?heart\s+rate\s+(?:is\s+)?[\d.]+\s*(?:bpm)?/gi,
    replacement: "You have a new health update",
  },
  // HRV values
  {
    pattern: /\bHRV\s+(?:dropped\s+to\s+|is\s+|at\s+)?[\d.]+\s*(?:ms)?/gi,
    replacement: "You have a new health update",
  },
  // Sleep quality scores
  {
    pattern: /\b(?:your\s+)?sleep\s+quality\s+(?:score)?[:\s]+[\d.]+(?:\/\d+)?/gi,
    replacement: "Your sleep summary is ready",
  },
  // Relaxation index
  {
    pattern: /\b(?:your\s+)?relaxation\s+index\s+(?:is\s+)?[\d.]+[^.!?\n]*/gi,
    replacement: "You have a new wellness insight",
  },
  // Brain session with wave data (alpha, theta, delta, beta, gamma)
  {
    pattern: /\bbrain\s+session[:\s]+[^.!?\n]*(?:alpha|theta|delta|beta|gamma)\b[^.!?\n]*/gi,
    replacement: "New brain session summary available",
  },
  // Generic: any remaining "alpha waves", "theta elevated" etc. in notifications
  {
    pattern: /\b(?:alpha|theta|delta|beta|gamma)\s+(?:waves?|power|elevated|suppressed|activity)\b[^.!?\n]*/gi,
    replacement: "New brain session summary available",
  },
  // Catch-all for specific emotion words paired with percentages
  {
    pattern: /\b(?:angry|sad|fearful?|anxious|depressed|happy|excited|stressed)\b[^.!?\n]*\d+%/gi,
    replacement: "You have a new wellness insight",
  },
];

// ── Sanitizer function ───────────────────────────────────────────────────

/**
 * Strips Protected Health Information from notification text.
 *
 * Call this on EVERY notification title and body before saving or displaying
 * outside the app (push notifications, lock screen, notification center).
 *
 * @param text - Raw notification text that may contain PHI
 * @returns Sanitized text safe for external display
 */
export function sanitizeNotificationText(text: string): string {
  if (!text) return "";

  let result = text;
  for (const { pattern, replacement } of PHI_PATTERNS) {
    // Reset lastIndex for global regexes
    pattern.lastIndex = 0;
    result = result.replace(pattern, replacement);
  }

  return result;
}
