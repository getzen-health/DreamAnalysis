/**
 * Smart check-in reminders — schedules daily notifications for voice analysis.
 * Context-aware: skips periods the user already checked in for.
 * Uses Notification API (works in Capacitor WebView + PWA).
 */

const REMINDER_PERIODS = [
  { id: "morning", hour: 9,  label: "morning",   greeting: "Good morning" },
  { id: "noon",    hour: 13, label: "afternoon",  greeting: "Hey" },
  { id: "evening", hour: 20, label: "evening",    greeting: "Good evening" },
] as const;

const MESSAGES = [
  "How are you feeling right now?",
  "Quick check-in — what's your mood?",
  "Take a moment to reflect on your state.",
  "Your emotional awareness matters.",
  "A 10-second voice check-in can reveal a lot.",
];

function getRandomMessage(): string {
  return MESSAGES[Math.floor(Math.random() * MESSAGES.length)];
}

function isCheckedIn(period: string): boolean {
  const today = new Date().toISOString().slice(0, 10);
  return !!localStorage.getItem(`voice-checkin-${today}-${period}`);
}

function isQuietHours(): boolean {
  try {
    const prefs = localStorage.getItem("ndw_notification_prefs");
    if (!prefs) return false;
    const { quiet_hours_start = 22, quiet_hours_end = 6, enabled = true } = JSON.parse(prefs);
    if (!enabled) return true; // All notifications disabled
    const hour = new Date().getHours();
    if (quiet_hours_start > quiet_hours_end) {
      return hour >= quiet_hours_start || hour < quiet_hours_end;
    }
    return hour >= quiet_hours_start && hour < quiet_hours_end;
  } catch {
    return false;
  }
}

function getLastEmotion(): string | null {
  try {
    const raw = localStorage.getItem("ndw_last_emotion");
    if (!raw) return null;
    const data = JSON.parse(raw);
    return data?.result?.emotion || null;
  } catch {
    return null;
  }
}

/**
 * Check if a reminder should fire for the current period.
 * Returns the reminder text or null if no reminder needed.
 */
export function checkReminder(): { title: string; body: string; period: string } | null {
  if (isQuietHours()) return null;

  const hour = new Date().getHours();
  const currentPeriod = REMINDER_PERIODS.find(p => {
    if (p.id === "morning") return hour >= 8 && hour <= 10;
    if (p.id === "noon") return hour >= 12 && hour <= 14;
    if (p.id === "evening") return hour >= 19 && hour <= 21;
    return false;
  });

  if (!currentPeriod) return null;
  if (isCheckedIn(currentPeriod.id)) return null;

  const lastEmotion = getLastEmotion();
  const personalized = lastEmotion
    ? `You felt ${lastEmotion} last time — let's see how you're doing now.`
    : getRandomMessage();

  return {
    title: `${currentPeriod.greeting}!`,
    body: personalized,
    period: currentPeriod.id,
  };
}

/**
 * Request notification permission and schedule periodic checks.
 * Call once on app mount.
 */
export function initCheckinReminders(): () => void {
  // Request permission
  if ("Notification" in window && Notification.permission === "default") {
    Notification.requestPermission();
  }

  // Check every 30 minutes if a reminder should fire
  const intervalId = setInterval(() => {
    if (!("Notification" in window) || Notification.permission !== "granted") return;

    const reminder = checkReminder();
    if (!reminder) return;

    // Don't fire more than once per period per day
    const today = new Date().toISOString().slice(0, 10);
    const firedKey = `ndw_reminder_fired_${today}_${reminder.period}`;
    if (localStorage.getItem(firedKey)) return;

    new Notification(reminder.title, {
      body: reminder.body,
      icon: "/icon-192.png",
      tag: `checkin-${reminder.period}`,
    });

    localStorage.setItem(firedKey, "true");
  }, 30 * 60 * 1000); // 30 minutes

  return () => clearInterval(intervalId);
}
