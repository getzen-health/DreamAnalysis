/**
 * RecentReadings — reusable component that reads the last N entries from a
 * localStorage key and displays them as a compact list with timestamps.
 *
 * Designed for the cross-cutting "last 5 readings" requirement on every page.
 * Each page passes its own localStorage key + a formatter function that knows
 * how to render a single entry.
 *
 * Data contract: the localStorage value must be a JSON array of objects.
 * Each object should have at least a `loggedAt` or `timestamp` field.
 */

import { useState, useEffect, useCallback } from "react";
import { Clock } from "lucide-react";

// ── Types ────────────────────────────────────────────────────────────────────

export interface RecentReadingsProps {
  /** localStorage key to read from */
  storageKey: string;
  /** Section title shown above the list */
  title: string;
  /** Maximum number of entries to display (default 5) */
  maxEntries?: number;
  /** Render a single entry row. Receives the entry object and index. */
  renderEntry: (entry: any, index: number) => React.ReactNode;
  /** Optional: custom event name(s) to listen for re-reads (e.g. "ndw-voice-updated") */
  listenEvents?: string[];
  /** If true, reads the value as a single object (not an array) and wraps it */
  singleObject?: boolean;
  /** Optional: empty state message */
  emptyMessage?: string;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function readFromStorage(key: string, singleObject: boolean): any[] {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (singleObject) {
      return parsed ? [parsed] : [];
    }
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function formatTimeAgo(dateOrTimestamp: string | number | undefined | null): string {
  if (!dateOrTimestamp) return "";
  const ts = typeof dateOrTimestamp === "number"
    ? (dateOrTimestamp > 1e12 ? dateOrTimestamp : dateOrTimestamp * 1000)
    : new Date(dateOrTimestamp).getTime();
  if (isNaN(ts)) return "";
  const diffMs = Date.now() - ts;
  const diffMin = Math.floor(diffMs / 60_000);
  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay === 1) return "yesterday";
  if (diffDay < 7) return `${diffDay}d ago`;
  try {
    return new Date(ts).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  } catch {
    return "";
  }
}

// ── Component ────────────────────────────────────────────────────────────────

export function RecentReadings({
  storageKey,
  title,
  maxEntries = 5,
  renderEntry,
  listenEvents = [],
  singleObject = false,
  emptyMessage,
}: RecentReadingsProps) {
  const [entries, setEntries] = useState<any[]>(() =>
    readFromStorage(storageKey, singleObject).slice(0, maxEntries),
  );

  const refresh = useCallback(() => {
    setEntries(readFromStorage(storageKey, singleObject).slice(0, maxEntries));
  }, [storageKey, singleObject, maxEntries]);

  // Stabilize listenEvents reference to avoid re-subscribe loops
  const eventsKey = listenEvents.join(",");

  useEffect(() => {
    refresh();
    const events = eventsKey ? eventsKey.split(",") : [];
    // Listen for custom events that signal new data
    for (const evt of events) {
      window.addEventListener(evt, refresh);
    }
    return () => {
      for (const evt of events) {
        window.removeEventListener(evt, refresh);
      }
    };
  }, [refresh, eventsKey]);

  if (entries.length === 0) {
    if (emptyMessage) {
      return (
        <div
          data-testid="recent-readings-empty"
          style={{
            borderTop: "1px solid var(--border)",
            paddingTop: 12,
            marginTop: 12,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 6 }}>
            <Clock style={{ width: 11, height: 11, color: "var(--muted-foreground)" }} />
            <span
              style={{
                fontSize: 10,
                fontWeight: 600,
                color: "var(--muted-foreground)",
                textTransform: "uppercase" as const,
                letterSpacing: "0.4px",
              }}
            >
              {title}
            </span>
          </div>
          <p style={{ fontSize: 12, color: "var(--muted-foreground)", margin: 0 }}>
            {emptyMessage}
          </p>
        </div>
      );
    }
    return null;
  }

  return (
    <div
      data-testid="recent-readings"
      style={{
        borderTop: "1px solid var(--border)",
        paddingTop: 12,
        marginTop: 12,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 8 }}>
        <Clock style={{ width: 11, height: 11, color: "var(--muted-foreground)" }} />
        <span
          style={{
            fontSize: 10,
            fontWeight: 600,
            color: "var(--muted-foreground)",
            textTransform: "uppercase" as const,
            letterSpacing: "0.4px",
          }}
        >
          {title}
        </span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {entries.map((entry, i) => (
          <div key={entry.id ?? entry.timestamp ?? i}>
            {renderEntry(entry, i)}
          </div>
        ))}
      </div>
    </div>
  );
}
