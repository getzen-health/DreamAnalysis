/**
 * EmotionStrip -- compact horizontal strip with valence/arousal/stress/focus bars.
 *
 * Reads from useCurrentEmotion. Designed for embedding in page headers.
 * Shows nothing if no emotion data is available.
 */

import { useCurrentEmotion } from "@/hooks/use-current-emotion";
import { cn } from "@/lib/utils";

interface EmotionStripProps {
  /** true = just bars, false = bars + labels */
  compact?: boolean;
}

interface Metric {
  label: string;
  value: number;
  /** Tailwind gradient stops for the bar */
  barColor: string;
  /** For bipolar metrics (valence), normalize to 0-1 range */
  normalize?: (v: number) => number;
}

const METRICS: Metric[] = [
  {
    label: "Valence",
    value: 0,
    barColor: "from-rose-500 to-cyan-500",
    normalize: (v: number) => (v + 1) / 2, // -1..1 -> 0..1
  },
  {
    label: "Arousal",
    value: 0,
    barColor: "from-indigo-500 to-orange-500",
  },
  {
    label: "Stress",
    value: 0,
    barColor: "from-cyan-500 to-rose-500",
  },
  {
    label: "Focus",
    value: 0,
    barColor: "from-gray-400 to-indigo-500",
  },
];

export function EmotionStrip({ compact = false }: EmotionStripProps) {
  const { emotion } = useCurrentEmotion();

  if (!emotion) return null;

  const values = [
    emotion.valence,
    emotion.arousal,
    emotion.stress,
    emotion.focus,
  ];

  return (
    <div className={cn("flex gap-3", compact ? "gap-2" : "gap-3")}>
      {METRICS.map((metric, i) => {
        const raw = values[i];
        const pct = (metric.normalize ? metric.normalize(raw) : raw) * 100;
        const clampedPct = Math.max(0, Math.min(100, pct));

        return (
          <div key={metric.label} className="flex-1 min-w-0">
            {!compact && (
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-[10px] text-muted-foreground truncate">
                  {metric.label}
                </span>
                <span className="text-[10px] font-mono text-muted-foreground/70">
                  {Math.round(clampedPct)}%
                </span>
              </div>
            )}
            <div className="h-1 w-full rounded-full bg-muted overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full bg-gradient-to-r transition-all duration-500",
                  metric.barColor,
                )}
                style={{ width: `${clampedPct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
