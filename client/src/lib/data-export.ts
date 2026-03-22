/**
 * Self-service data export utilities.
 *
 * Exports session summaries, voice check-ins, and full account data
 * in standard formats (CSV, JSON) for GDPR data portability compliance.
 */

import type { SessionSummary } from "@/lib/ml-api";

// ── Types ────────────────────────────────────────────────────────────────────

export interface CheckinData {
  id: string;
  timestamp: string;          // ISO 8601
  emotion: string;
  intensity: number;          // 0-1
  notes: string;
  voiceBiomarkers?: {
    energy?: number;
    stress?: number;
    valence?: number;
  };
}

// ── CSV helpers ──────────────────────────────────────────────────────────────

/** Escape a value for CSV: wrap in quotes if it contains commas, quotes, or newlines. */
function csvEscape(value: string | number | null | undefined): string {
  if (value == null) return "";
  const str = String(value);
  if (str.includes(",") || str.includes('"') || str.includes("\n")) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

/** Convert an array of rows (each row is an array of values) to CSV string. */
function rowsToCsv(headers: string[], rows: (string | number | null | undefined)[][]): string {
  const headerLine = headers.map(csvEscape).join(",");
  const dataLines = rows.map((row) => row.map(csvEscape).join(","));
  return [headerLine, ...dataLines].join("\n");
}

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Export session summaries as a CSV string.
 */
export function exportSessionsCSV(sessions: SessionSummary[]): string {
  const headers = [
    "session_id",
    "user_id",
    "session_type",
    "start_time",
    "status",
    "duration_sec",
    "n_frames",
    "avg_stress",
    "avg_focus",
    "avg_relaxation",
    "avg_flow",
    "avg_creativity",
    "avg_valence",
    "avg_arousal",
    "dominant_emotion",
  ];

  const rows = sessions.map((s) => [
    s.session_id,
    s.user_id,
    s.session_type,
    s.start_time != null ? new Date(s.start_time * 1000).toISOString() : null,
    s.status,
    s.summary?.duration_sec ?? null,
    s.summary?.n_frames ?? null,
    s.summary?.avg_stress ?? null,
    s.summary?.avg_focus ?? null,
    s.summary?.avg_relaxation ?? null,
    s.summary?.avg_flow ?? null,
    s.summary?.avg_creativity ?? null,
    s.summary?.avg_valence ?? null,
    s.summary?.avg_arousal ?? null,
    s.summary?.dominant_emotion ?? null,
  ]);

  return rowsToCsv(headers, rows);
}

/**
 * Export voice check-in history as a CSV string.
 */
export function exportVoiceCheckinsCSV(checkins: CheckinData[]): string {
  const headers = [
    "id",
    "timestamp",
    "emotion",
    "intensity",
    "notes",
    "voice_energy",
    "voice_stress",
    "voice_valence",
  ];

  const rows = checkins.map((c) => [
    c.id,
    c.timestamp,
    c.emotion,
    c.intensity,
    c.notes,
    c.voiceBiomarkers?.energy ?? null,
    c.voiceBiomarkers?.stress ?? null,
    c.voiceBiomarkers?.valence ?? null,
  ]);

  return rowsToCsv(headers, rows);
}

/**
 * Export full account data as a JSON string (GDPR Article 20 compliance).
 *
 * Gathers all user data from localStorage and provided session/checkin data.
 */
export function exportAccountJSON(
  userId: string,
  sessions: SessionSummary[] = [],
  checkins: CheckinData[] = [],
): string {
  // Gather localStorage keys that belong to this app
  const localStorageData: Record<string, string> = {};
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith("ndw_")) {
        localStorageData[key] = localStorage.getItem(key) ?? "";
      }
    }
  } catch {
    // localStorage may not be available
  }

  const exportData = {
    export_version: "1.0",
    export_date: new Date().toISOString(),
    user_id: userId,
    sessions,
    voice_checkins: checkins,
    preferences: localStorageData,
  };

  return JSON.stringify(exportData, null, 2);
}

/**
 * Trigger a browser download for the given content.
 */
export function downloadFile(
  content: ArrayBuffer | string,
  filename: string,
  mimeType: string,
): void {
  const blob =
    content instanceof ArrayBuffer
      ? new Blob([content], { type: mimeType })
      : new Blob([content], { type: mimeType });

  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

/**
 * Delete all user data from localStorage (GDPR Article 17 — right to erasure).
 *
 * Returns the number of keys removed.
 */
export function deleteAllLocalData(): number {
  let removed = 0;
  try {
    const keysToRemove: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith("ndw_")) {
        keysToRemove.push(key);
      }
    }
    // Also remove auth-related keys if they exist
    if (localStorage.getItem("auth_token") !== null) {
      keysToRemove.push("auth_token");
    }

    for (const key of keysToRemove) {
      localStorage.removeItem(key);
      removed++;
    }
  } catch {
    // localStorage may not be available
  }
  return removed;
}
