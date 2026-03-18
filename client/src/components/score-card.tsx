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
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
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
}: ScoreCardProps) {
  const accentColor = SCORE_COLORS[color].from;

  return (
    <div
      className="relative group rounded-[14px] p-4 transition-all duration-200 hover:brightness-110"
      style={{
        background: "#111827",
        border: "1px solid #1f2937",
        borderLeft: `3px solid ${accentColor}`,
      }}
    >
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
            <span
              className="text-sm font-semibold truncate"
              style={{ color: "#e8e0d4" }}
            >
              {title}
            </span>
          </div>

          {subtitle && (
            <p
              className="text-xs font-mono truncate mt-1 ml-9"
              style={{ color: "#8b8578" }}
            >
              {subtitle}
            </p>
          )}

          {/* Trend indicator */}
          {trend && (
            <div className="flex items-center gap-1.5 mt-2 ml-9">
              {trend === "up" && (
                <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />
              )}
              {trend === "down" && (
                <TrendingDown className="h-3.5 w-3.5 text-red-400" />
              )}
              {trend === "stable" && (
                <Minus className="h-3.5 w-3.5 text-zinc-500" />
              )}
              {trendValue && (
                <span
                  className="text-[11px] font-mono"
                  style={{
                    color:
                      trend === "up"
                        ? "#34d399"
                        : trend === "down"
                        ? "#f87171"
                        : "#8b8578",
                  }}
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
