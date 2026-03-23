/**
 * posthog.ts — Privacy-respecting analytics via PostHog.
 *
 * Initializes PostHog with VITE_POSTHOG_KEY env var.
 * ALL tracking calls check analytics consent BEFORE sending any event.
 * If consent is not granted, calls are silently no-ops.
 *
 * Events tracked:
 *   - page_view
 *   - voice_analysis_completed
 *   - eeg_session_started
 *   - mood_logged
 *   - achievement_unlocked
 */

// ── Types ──────────────────────────────────────────────────────────────────

type PostHogInstance = {
  capture: (event: string, properties?: Record<string, unknown>) => void;
  identify: (distinctId: string, properties?: Record<string, unknown>) => void;
  reset: () => void;
};

// ── State ──────────────────────────────────────────────────────────────────

let _posthog: PostHogInstance | null = null;
let _initAttempted = false;

// ── Consent check ──────────────────────────────────────────────────────────

const CONSENT_KEY = "ndw_analytics_consent";

/**
 * Check if the user has granted analytics consent.
 * Returns false if not explicitly opted in.
 */
export function hasAnalyticsConsent(): boolean {
  try {
    return localStorage.getItem(CONSENT_KEY) === "true";
  } catch {
    return false;
  }
}

/**
 * Set analytics consent. When set to false, no events will be tracked.
 */
export function setAnalyticsConsent(granted: boolean): void {
  try {
    localStorage.setItem(CONSENT_KEY, granted ? "true" : "false");
  } catch {
    // localStorage unavailable
  }
}

// ── Init ───────────────────────────────────────────────────────────────────

/**
 * Initialize PostHog. Safe to call multiple times; only initializes once.
 * No-op if VITE_POSTHOG_KEY is not set or consent is not granted.
 */
export async function initPostHog(): Promise<void> {
  if (_initAttempted) return;
  _initAttempted = true;

  if (!hasAnalyticsConsent()) return;

  const key = import.meta.env.VITE_POSTHOG_KEY;
  if (!key) return;

  try {
    const posthog = await import("posthog-js");
    posthog.default.init(key, {
      api_host: "https://app.posthog.com",
      autocapture: false,
      capture_pageview: false, // we track manually for consent
      persistence: "localStorage",
      disable_session_recording: true,
      loaded: (ph) => {
        _posthog = ph as unknown as PostHogInstance;
      },
    });
    _posthog = posthog.default as unknown as PostHogInstance;
  } catch {
    // PostHog not installed or failed to load — degrade gracefully
    _posthog = null;
  }
}

// ── Track helpers ──────────────────────────────────────────────────────────

/**
 * Track an event. No-op if consent is not granted or PostHog is not loaded.
 */
function track(event: string, properties?: Record<string, unknown>): void {
  if (!hasAnalyticsConsent()) return;
  if (!_posthog) return;
  try {
    _posthog.capture(event, properties);
  } catch {
    // Swallow errors — analytics should never break the app
  }
}

// ── Public event helpers ───────────────────────────────────────────────────

export function trackPageView(path: string): void {
  track("page_view", { path });
}

export function trackVoiceAnalysisCompleted(properties?: {
  emotion?: string;
  confidence?: number;
}): void {
  track("voice_analysis_completed", properties);
}

export function trackEegSessionStarted(properties?: {
  device?: string;
  mode?: string;
}): void {
  track("eeg_session_started", properties);
}

export function trackMoodLogged(properties?: {
  mood?: number;
  energy?: number;
}): void {
  track("mood_logged", properties);
}

export function trackAchievementUnlocked(properties?: {
  achievement?: string;
}): void {
  track("achievement_unlocked", properties);
}

/**
 * Identify a user (call after login). No-op without consent.
 */
export function identifyUser(userId: string): void {
  if (!hasAnalyticsConsent()) return;
  if (!_posthog) return;
  try {
    _posthog.identify(userId);
  } catch {
    // Swallow
  }
}

/**
 * Reset user identity (call on logout). No-op without consent.
 */
export function resetPostHog(): void {
  if (!_posthog) return;
  try {
    _posthog.reset();
  } catch {
    // Swallow
  }
}
