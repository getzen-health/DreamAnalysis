/**
 * TTL-based localStorage cleanup for sensitive neurodata.
 * Purges emotion/health data older than 7 days on app load.
 */

const SENSITIVE_KEYS_WITH_TTL = [
  "ndw_last_emotion",
  "ndw_today_emotion",
  "ndw_yesterday_emotion",
  "ndw_emotions_seen",
] as const;

const TTL_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

interface TimestampedData {
  timestamp?: string;
  stored_at?: number;
}

export function cleanExpiredLocalStorage(): void {
  const now = Date.now();

  for (const key of SENSITIVE_KEYS_WITH_TTL) {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) continue;

      const data: TimestampedData = JSON.parse(raw);
      const storedAt = data.stored_at
        ? data.stored_at
        : data.timestamp
          ? new Date(data.timestamp).getTime()
          : 0;

      if (storedAt > 0 && now - storedAt > TTL_MS) {
        localStorage.removeItem(key);
      }
    } catch {
      // If we can't parse it, remove it (stale data)
      localStorage.removeItem(key);
    }
  }

  // Clean food logs older than 30 days
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key?.startsWith("ndw_food_logs_")) {
      try {
        const logs = JSON.parse(localStorage.getItem(key) || "[]");
        const filtered = logs.filter((log: { timestamp?: string }) => {
          if (!log.timestamp) return true;
          return now - new Date(log.timestamp).getTime() < 30 * 24 * 60 * 60 * 1000;
        });
        if (filtered.length === 0) {
          localStorage.removeItem(key);
        } else if (filtered.length < logs.length) {
          localStorage.setItem(key, JSON.stringify(filtered));
        }
      } catch {
        // skip malformed entries
      }
    }
  }
}
