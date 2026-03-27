/**
 * EEGWeekCompareCard — "This week vs last week" EEG summary.
 *
 * Compares the past 7 days of EEG readings (focus, stress, mood) to the
 * 7 days before that and shows directional trend arrows.
 *
 * Unique EEG differentiator: Oura shows HRV/sleep trends; we show brain trends.
 */

import { TrendingUp, TrendingDown, Minus, Brain } from "lucide-react";
import type { StoredEmotionReading } from "@/lib/ml-api";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface EEGWeekCompareProps {
  /** Up to 14 days of EEG readings, sorted newest-first or any order */
  history: StoredEmotionReading[];
}

interface MetricSummary {
  label: string;
  thisWeek: number | null;
  lastWeek: number | null;
  unit: string;
  higherIsBetter: boolean;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const MS_PER_DAY = 86_400_000;

function average(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/** Returns {thisWeek, lastWeek} averages for a given field */
export function computeWeekComparison(
  history: StoredEmotionReading[],
  field: keyof Pick<StoredEmotionReading, "focus" | "stress" | "happiness" | "energy">,
): { thisWeek: number | null; lastWeek: number | null } {
  const now = Date.now();
  const oneWeekAgo = now - 7 * MS_PER_DAY;
  const twoWeeksAgo = now - 14 * MS_PER_DAY;

  const thisWeekVals: number[] = [];
  const lastWeekVals: number[] = [];

  for (const r of history) {
    const t = new Date(r.timestamp).getTime();
    const val = r[field] as number | null;
    if (typeof val !== "number") continue;
    if (t >= oneWeekAgo && t <= now) {
      thisWeekVals.push(val);
    } else if (t >= twoWeeksAgo && t < oneWeekAgo) {
      lastWeekVals.push(val);
    }
  }

  return {
    thisWeek: average(thisWeekVals),
    lastWeek: average(lastWeekVals),
  };
}

/** Build the 7-day daily averages for a sparkline */
export function buildSparkline(
  history: StoredEmotionReading[],
  field: keyof Pick<StoredEmotionReading, "focus" | "stress" | "happiness" | "energy">,
): (number | null)[] {
  const now = Date.now();
  const sums: number[] = Array(7).fill(0);
  const counts: number[] = Array(7).fill(0);

  for (const r of history) {
    const t = new Date(r.timestamp).getTime();
    const dayIndex = Math.floor((now - t) / MS_PER_DAY); // 0=today, 6=6 days ago
    if (dayIndex < 0 || dayIndex >= 7) continue;
    const val = r[field] as number | null;
    if (typeof val !== "number") continue;
    const slot = 6 - dayIndex; // reverse so [0]=oldest, [6]=today
    sums[slot] += val;
    counts[slot]++;
  }

  return sums.map((sum, i) => counts[i] > 0 ? sum / counts[i] : null);
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function TrendArrow({
  thisWeek,
  lastWeek,
  higherIsBetter,
}: {
  thisWeek: number | null;
  lastWeek: number | null;
  higherIsBetter: boolean;
}) {
  if (thisWeek === null || lastWeek === null) {
    return <Minus className="h-3.5 w-3.5 text-muted-foreground/40" />;
  }
  const delta = thisWeek - lastWeek;
  const isPositive = higherIsBetter ? delta > 0.02 : delta < -0.02;
  const isNegative = higherIsBetter ? delta < -0.02 : delta > 0.02;

  if (isPositive) return <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />;
  if (isNegative) return <TrendingDown className="h-3.5 w-3.5 text-rose-400" />;
  return <Minus className="h-3.5 w-3.5 text-muted-foreground/50" />;
}

function MiniSparkline({ data }: { data: (number | null)[] }) {
  const valid = data.filter((v): v is number => v !== null);
  if (valid.length < 2) return null;

  const min = Math.min(...valid);
  const max = Math.max(...valid);
  const range = max - min || 0.01;
  const H = 24;
  const W = 56;
  const step = W / (data.length - 1);

  const pts = data
    .map((v, i) => {
      const x = i * step;
      const y = v !== null ? H - ((v - min) / range) * H : null;
      return { x, y };
    })
    .filter((p): p is { x: number; y: number } => p.y !== null);

  if (pts.length < 2) return null;

  const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="overflow-visible">
      <path d={d} fill="none" stroke="hsl(210,80%,60%)" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function MetricRow({ metric }: { metric: MetricSummary }) {
  const { thisWeek, lastWeek } = metric;
  const fmt = (v: number | null) =>
    v !== null ? `${Math.round(v * 100)}${metric.unit}` : "—";

  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-muted-foreground w-14 shrink-0">{metric.label}</span>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold font-mono">{fmt(thisWeek)}</span>
          <TrendArrow
            thisWeek={thisWeek}
            lastWeek={lastWeek}
            higherIsBetter={metric.higherIsBetter}
          />
          {lastWeek !== null && (
            <span className="text-[10px] text-muted-foreground/50 font-mono">
              vs {fmt(lastWeek)}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function EEGWeekCompareCard({ history }: EEGWeekCompareProps) {
  const focusComp  = computeWeekComparison(history, "focus");
  const stressComp = computeWeekComparison(history, "stress");
  const moodComp   = computeWeekComparison(history, "happiness");

  const focusSparkline = buildSparkline(history, "focus");
  const stressSparkline = buildSparkline(history, "stress");

  const hasData = history.length > 0;

  const metrics: MetricSummary[] = [
    { label: "Focus", ...focusComp, unit: "%", higherIsBetter: true },
    { label: "Stress", ...stressComp, unit: "%", higherIsBetter: false },
    { label: "Mood", ...moodComp, unit: "%", higherIsBetter: true },
  ];

  return (
    <div
      data-testid="eeg-week-compare-card"
      className="rounded-2xl border border-border/40 bg-card/60 p-4"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <Brain className="h-3.5 w-3.5 text-indigo-400" />
        <h2 className="text-sm font-semibold">Brain Trends</h2>
        <span className="text-[10px] font-mono text-indigo-400/70 bg-indigo-500/10 px-1.5 py-0.5 rounded-full ml-auto">
          7-day
        </span>
      </div>

      {!hasData ? (
        <div
          data-testid="eeg-week-compare-empty"
          className="rounded-xl border border-dashed border-border/60 p-4 text-center bg-card/40"
        >
          <p className="text-xs text-muted-foreground/50">
            Complete EEG sessions to see weekly brain trends
          </p>
        </div>
      ) : (
        <div className="space-y-2.5">
          {metrics.map((m) => (
            <MetricRow key={m.label} metric={m} />
          ))}

          {/* Mini sparklines for focus + stress */}
          {focusSparkline.some((v) => v !== null) && (
            <div className="pt-1 flex gap-4">
              <div>
                <p className="text-[9px] text-muted-foreground/50 mb-1 font-mono uppercase">Focus trend</p>
                <MiniSparkline data={focusSparkline} />
              </div>
              {stressSparkline.some((v) => v !== null) && (
                <div>
                  <p className="text-[9px] text-muted-foreground/50 mb-1 font-mono uppercase">Stress trend</p>
                  <MiniSparkline data={stressSparkline} />
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
