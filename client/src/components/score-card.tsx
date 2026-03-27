/**
 * ScoreCard -- premium card for displaying a health score with context.
 *
 * Features:
 * - Left accent border in signature color
 * - Score gauge (sm) on the right
 * - Title + subtitle on the left
 * - Trend indicator (up/down/stable)
 * - Hover brightness increase
 */

import { type ReactNode } from "react";
import { TrendingUp, TrendingDown, Minus, HelpCircle } from "lucide-react";
import { ScoreGauge, type ScoreColor, SCORE_COLORS } from "./score-gauge";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ScoreCardProps {
  title: string;
  value: number | null;
  max?: number;
  color: ScoreColor;
  icon: ReactNode;
  subtitle?: string;    // e.g., "67ms HRV - 58 RHR"
  trend?: "up" | "down" | "stable";
  trendValue?: string;  // e.g., "+5% from yesterday"
  onInfoClick?: () => void;  // Opens Learn More overlay
}

// ── Component ─────────────────────────────────────────────────────────────────

export function ScoreCard({
  title,
  value,
  max = 100,
  color,
  icon,
  subtitle,
  trend,
  trendValue,
  onInfoClick,
}: ScoreCardProps) {
  const accentColor = SCORE_COLORS[color].from;

  return (
    <div
      className="relative group rounded-2xl p-4 transition-all duration-200 hover:brightness-110 active:scale-[0.98] bg-card border border-border"
      style={{
        borderLeft: `3px solid ${accentColor}`,
      }}
    >
      {/* Learn More button */}
      {onInfoClick && (
        <button
          onClick={(e) => { e.stopPropagation(); onInfoClick(); }}
          className="absolute top-2 right-2 p-1 rounded-full opacity-0 group-hover:opacity-60 transition-opacity"
          aria-label={`Learn more about ${title}`}
        >
          <HelpCircle className="w-3.5 h-3.5 text-muted-foreground" />
        </button>
      )}
      <div className="flex items-center justify-between gap-3">
        {/* Left: title + subtitle + icon */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            <span
              className="flex items-center justify-center w-7 h-7 rounded-lg shrink-0"
              style={{ background: `${accentColor}20` }}
            >
              {icon}
            </span>
            <span className="text-sm font-semibold truncate text-foreground">
              {title}
            </span>
          </div>

          {subtitle && (
            <p className="text-xs font-mono truncate mt-1 ml-9 text-muted-foreground">
              {subtitle}
            </p>
          )}

          {/* Trend indicator */}
          {trend && (
            <div className="flex items-center gap-1.5 mt-2 ml-9">
              {trend === "up" && (
                <TrendingUp className="h-3.5 w-3.5 text-cyan-400" />
              )}
              {trend === "down" && (
                <TrendingDown className="h-3.5 w-3.5 text-rose-400" />
              )}
              {trend === "stable" && (
                <Minus className="h-3.5 w-3.5 text-zinc-500" />
              )}
              {trendValue && (
                <span
                  className={`text-[11px] font-mono ${
                    trend === "up"
                      ? "text-cyan-400"
                      : trend === "down"
                      ? "text-rose-400"
                      : "text-muted-foreground"
                  }`}
                >
                  {trendValue}
                </span>
              )}
            </div>
          )}
        </div>

        {/* Right: score gauge */}
        <div className="shrink-0">
          <ScoreGauge
            value={value}
            max={max}
            label=""
            color={color}
            size="sm"
          />
        </div>
      </div>
    </div>
  );
}
