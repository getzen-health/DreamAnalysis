/**
 * EEGPeakHours — "When is your brain sharpest?"
 *
 * Groups historical EEG readings by hour of day and shows a 24-bar chart.
 * Highlights the top 3 peak focus hours. Answers the North Star question:
 * "Peak focus 9:30am – 12:00pm ← protect this time"
 *
 * Unique EEG differentiator: data comes from real EEG session readings,
 * not from generic circadian rhythm assumptions.
 */

import { Clock, Zap } from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────

/** Minimal shape required — compatible with StoredEmotionReading and local EmotionEntry */
export interface FocusReading {
  focus: number;
  timestamp: string;
}

export interface EEGPeakHoursProps {
  history: FocusReading[];
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Returns hourly average focus (index 0–23) and count per hour */
export function computeHourlyFocus(
  history: FocusReading[],
): { avg: number | null; count: number }[] {
  const buckets: { sum: number; count: number }[] = Array.from({ length: 24 }, () => ({
    sum: 0,
    count: 0,
  }));

  for (const r of history) {
    const hour = new Date(r.timestamp).getHours();
    if (hour >= 0 && hour < 24) {
      buckets[hour].sum += r.focus;
      buckets[hour].count += 1;
    }
  }

  return buckets.map((b) => ({
    avg: b.count > 0 ? b.sum / b.count : null,
    count: b.count,
  }));
}

/** Returns the indices of the top N hours by focus average */
export function topPeakHours(hourly: { avg: number | null }[], n = 3): number[] {
  return hourly
    .map((h, i) => ({ i, avg: h.avg ?? -1 }))
    .filter((h) => h.avg > 0)
    .sort((a, b) => b.avg - a.avg)
    .slice(0, n)
    .map((h) => h.i);
}

function fmtHour(h: number): string {
  if (h === 0) return "12am";
  if (h < 12) return `${h}am`;
  if (h === 12) return "12pm";
  return `${h - 12}pm`;
}

// ─── Component ────────────────────────────────────────────────────────────────

export function EEGPeakHours({ history }: EEGPeakHoursProps) {
  const hourly = computeHourlyFocus(history);
  const peaks = topPeakHours(hourly, 3);
  const maxAvg = Math.max(...hourly.map((h) => h.avg ?? 0), 0.01);
  const hasData = hourly.some((h) => h.count > 0);

  // Only show waking hours (6am–11pm) for cleaner display
  const wakerHours = Array.from({ length: 18 }, (_, i) => i + 6); // 6..23

  return (
    <div
      data-testid="eeg-peak-hours"
      className="rounded-2xl border border-border/40 bg-card/60 p-4"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <Clock className="h-3.5 w-3.5 text-amber-400" />
        <h2 className="text-sm font-semibold">Peak Focus Hours</h2>
        <span className="text-[10px] font-mono text-amber-400/70 bg-amber-500/10 px-1.5 py-0.5 rounded-full ml-auto">
          EEG
        </span>
      </div>

      {!hasData ? (
        <div
          data-testid="eeg-peak-hours-empty"
          className="rounded-xl border border-dashed border-border/60 p-4 text-center bg-card/40"
        >
          <Clock className="h-5 w-5 text-muted-foreground/30 mx-auto mb-1.5" />
          <p className="text-xs text-muted-foreground/50">
            EEG session data will reveal your peak focus hours
          </p>
        </div>
      ) : (
        <>
          {/* Peak hours summary */}
          {peaks.length > 0 && (
            <div className="flex items-center gap-1.5 mb-3">
              <Zap className="h-3 w-3 text-amber-400 shrink-0" />
              <p className="text-[11px] text-muted-foreground">
                <span className="text-foreground font-medium">Best hours: </span>
                {peaks.map(fmtHour).join(", ")}
              </p>
            </div>
          )}

          {/* 18-bar chart (6am–11pm) */}
          <div className="flex items-end gap-[2px] h-10">
            {wakerHours.map((h) => {
              const slot = hourly[h];
              const pct = slot.avg !== null ? (slot.avg / maxAvg) * 100 : 0;
              const isPeak = peaks.includes(h);
              const isActive = slot.count > 0;
              return (
                <div
                  key={h}
                  data-testid={`hour-bar-${h}`}
                  title={`${fmtHour(h)}: ${slot.avg !== null ? `${Math.round(slot.avg * 100)}% focus` : "no data"}`}
                  className="flex-1 rounded-sm relative"
                  style={{
                    height: "40px",
                    display: "flex",
                    alignItems: "flex-end",
                  }}
                >
                  <div
                    className="w-full rounded-sm transition-all duration-500"
                    style={{
                      height: isActive ? `${Math.max(4, pct)}%` : "4%",
                      background: isPeak
                        ? "hsl(38,85%,58%)"
                        : isActive
                        ? "hsl(210,70%,55%)"
                        : "hsl(210,15%,25%)",
                      opacity: isActive ? 1 : 0.3,
                    }}
                  />
                </div>
              );
            })}
          </div>

          {/* X-axis labels: every 3 hours */}
          <div className="flex justify-between mt-1">
            {[6, 9, 12, 15, 18, 21].map((h) => (
              <span key={h} className="text-[8px] text-muted-foreground/40 font-mono">
                {fmtHour(h)}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
