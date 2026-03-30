/**
 * DreamPatternsCard — shows longitudinal dream theme tracking (7/30/90 day).
 *
 * Displays top 5 recurring themes with trend arrows, emotion distribution
 * as a stacked bar, and a period selector. Pure presentational component;
 * receives a DreamPatternSummary from the parent.
 *
 * Issue #549.
 */

import { useState } from "react";
import { Moon, TrendingUp, TrendingDown, Minus } from "lucide-react";
import {
  analyzeDreamPatterns,
  type DreamEntry,
  type DreamPatternSummary,
  type DreamTheme,
} from "@/lib/dream-theme-tracker";

// ── Props ────────────────────────────────────────────────────────────────────

export interface DreamPatternsCardProps {
  dreams: DreamEntry[];
}

// ── Period pills ─────────────────────────────────────────────────────────────

const PERIODS = [
  { label: "7d", days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
] as const;

// ── Emotion dot colors (deterministic by emotion name hash) ──────────────────

const EMOTION_COLORS = [
  "bg-red-400",
  "bg-amber-400",
  "bg-emerald-400",
  "bg-blue-400",
  "bg-purple-400",
  "bg-pink-400",
  "bg-cyan-400",
  "bg-orange-400",
];

function emotionColor(emotion: string): string {
  let hash = 0;
  for (let i = 0; i < emotion.length; i++) {
    hash = (hash * 31 + emotion.charCodeAt(i)) | 0;
  }
  return EMOTION_COLORS[Math.abs(hash) % EMOTION_COLORS.length];
}

// ── Stacked bar colors for emotion distribution ──────────────────────────────

const BAR_COLORS = [
  "bg-indigo-500",
  "bg-emerald-500",
  "bg-amber-500",
  "bg-rose-500",
  "bg-cyan-500",
  "bg-purple-500",
  "bg-orange-500",
  "bg-teal-500",
];

// ── Trend icon helper ────────────────────────────────────────────────────────

function TrendIcon({ trend }: { trend: DreamTheme["trend"] }) {
  if (trend === "increasing") {
    return <TrendingUp className="h-3 w-3 text-emerald-400" />;
  }
  if (trend === "decreasing") {
    return <TrendingDown className="h-3 w-3 text-red-400" />;
  }
  return <Minus className="h-3 w-3 text-muted-foreground" />;
}

// ── Component ────────────────────────────────────────────────────────────────

export function DreamPatternsCard({ dreams }: DreamPatternsCardProps) {
  const [periodDays, setPeriodDays] = useState(30);
  const summary: DreamPatternSummary = analyzeDreamPatterns(dreams, periodDays);

  const isEmpty = summary.totalDreams === 0 || summary.topThemes.length === 0;

  // Sorted emotion entries for the stacked bar
  const emotionEntries = Object.entries(summary.emotionDistribution)
    .sort((a, b) => b[1] - a[1]);
  const totalEmotionCount = emotionEntries.reduce((s, [, c]) => s + c, 0);

  return (
    <div
      data-testid="dream-patterns-card"
      className="rounded-[14px] bg-card border border-border p-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Moon className="h-4 w-4 text-indigo-400" />
          <span className="text-sm font-medium text-muted-foreground">
            Dream Patterns
          </span>
          {summary.totalDreams > 0 && (
            <span className="text-[10px] text-muted-foreground/60">
              {summary.totalDreams} dream{summary.totalDreams !== 1 ? "s" : ""}
              {summary.lucidDreamCount > 0 && (
                <> &middot; {summary.lucidDreamCount} lucid</>
              )}
            </span>
          )}
        </div>

        {/* Period selector pills */}
        <div className="flex gap-1" data-testid="period-selector">
          {PERIODS.map((p) => (
            <button
              key={p.days}
              onClick={() => setPeriodDays(p.days)}
              className={`px-2.5 py-0.5 text-[10px] rounded-full transition-colors ${
                periodDays === p.days
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Empty state */}
      {isEmpty ? (
        <p
          data-testid="dream-patterns-empty"
          className="text-sm text-muted-foreground/60 py-4 text-center"
        >
          Log more dreams to discover patterns
        </p>
      ) : (
        <div className="space-y-3">
          {/* Top 5 themes */}
          <div className="space-y-1.5" data-testid="theme-list">
            {summary.topThemes.map((theme) => (
              <div
                key={theme.theme}
                className="flex items-center gap-2 py-1"
                data-testid={`theme-row-${theme.theme}`}
              >
                {/* Theme name */}
                <span className="text-xs font-medium capitalize min-w-[80px]">
                  {theme.theme}
                </span>

                {/* Count badge */}
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground font-mono">
                  {theme.count}
                </span>

                {/* Trend arrow */}
                <TrendIcon trend={theme.trend} />

                {/* Associated emotion dots */}
                <div className="flex items-center gap-0.5 ml-auto">
                  {theme.associatedEmotions.map((em) => (
                    <span
                      key={em}
                      title={em}
                      className={`h-2 w-2 rounded-full ${emotionColor(em)}`}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Emotion distribution stacked bar */}
          {emotionEntries.length > 0 && totalEmotionCount > 0 && (
            <div data-testid="emotion-bar">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">
                Emotion Distribution
              </p>
              <div className="flex h-2.5 rounded-full overflow-hidden">
                {emotionEntries.slice(0, 8).map(([emotion, count], i) => (
                  <div
                    key={emotion}
                    className={`${BAR_COLORS[i % BAR_COLORS.length]} transition-all`}
                    style={{ width: `${(count / totalEmotionCount) * 100}%` }}
                    title={`${emotion}: ${count}`}
                  />
                ))}
              </div>
              <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1">
                {emotionEntries.slice(0, 8).map(([emotion], i) => (
                  <div key={emotion} className="flex items-center gap-1">
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${BAR_COLORS[i % BAR_COLORS.length]}`}
                    />
                    <span className="text-[9px] text-muted-foreground capitalize">
                      {emotion}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
