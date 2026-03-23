/**
 * FoodScoreCard — Yuka-style health score display for food items.
 *
 * Shows a large animated circular score (0-100), rating label,
 * nutrient breakdown bars, brain impact analysis, and mood prediction.
 *
 * Used after scanning/logging food on the nutrition page.
 */

import { useState, useEffect, useMemo } from "react";
import { Brain, Heart, Zap, TrendingDown } from "lucide-react";
import {
  calculateFoodScore,
  type FoodScoreInput,
  type FoodScoreResult,
  type NutrientFlag,
} from "@/lib/food-score";

// ── Props ───────────────────────────────────────────────────────────────────

interface FoodScoreCardProps {
  food: FoodScoreInput;
  /** Optional meal name shown at the top */
  mealName?: string;
  /** Compact mode for inline display in meal cards */
  compact?: boolean;
  className?: string;
}

// ── Nutrient bar colors ─────────────────────────────────────────────────────

function nutrientBarColor(flag: NutrientFlag): string {
  // For nutrients where high = bad (sugar, sodium, fat, calories)
  const badWhenHigh = ["Sugar", "Sodium", "Fat", "Calories"];
  if (badWhenHigh.includes(flag.nutrient)) {
    if (flag.level === "high") return "#e879a8";  // rose
    if (flag.level === "moderate") return "#d4a017"; // amber
    return "#06b6d4"; // cyan -- good
  }
  // For nutrients where high = good (protein, fiber)
  if (flag.level === "high") return "#06b6d4"; // cyan
  if (flag.level === "moderate") return "#22c55e"; // green
  return "#e879a8"; // rose -- low is bad
}

function nutrientLevelLabel(flag: NutrientFlag): string {
  const badWhenHigh = ["Sugar", "Sodium", "Fat", "Calories"];
  if (badWhenHigh.includes(flag.nutrient)) {
    if (flag.level === "low") return "Good";
    if (flag.level === "moderate") return "OK";
    return "High";
  }
  // Good when high
  if (flag.level === "high") return "Great";
  if (flag.level === "moderate") return "OK";
  return "Low";
}

// ── Score ring component ────────────────────────────────────────────────────

function ScoreRing({
  score,
  color,
  size = 120,
}: {
  score: number;
  color: string;
  size?: number;
}) {
  const [animatedScore, setAnimatedScore] = useState(0);
  const strokeWidth = size > 100 ? 8 : 6;
  const r = (size - strokeWidth * 2) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * r;
  const dashOffset = circumference * (1 - animatedScore / 100);

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedScore(score), 80);
    return () => clearTimeout(timer);
  }, [score]);

  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Background track */}
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke="var(--border)"
          strokeWidth={strokeWidth}
          opacity={0.3}
        />
        {/* Score arc */}
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          transform={`rotate(-90 ${cx} ${cy})`}
          style={{
            transition: "stroke-dashoffset 1s cubic-bezier(0.34, 1.56, 0.64, 1)",
            filter: `drop-shadow(0 0 6px ${color}40)`,
          }}
        />
      </svg>
      {/* Center text */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <span
          style={{
            fontSize: size > 100 ? 36 : 24,
            fontWeight: 700,
            color: "var(--foreground)",
            lineHeight: 1,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {Math.round(animatedScore)}
        </span>
        <span
          style={{
            fontSize: size > 100 ? 10 : 8,
            color: "var(--muted-foreground)",
            fontWeight: 500,
            marginTop: 2,
            textTransform: "uppercase",
            letterSpacing: "0.5px",
          }}
        >
          Health Score
        </span>
      </div>
    </div>
  );
}

// ── Rating badge ────────────────────────────────────────────────────────────

const RATING_LABELS: Record<string, string> = {
  excellent: "Excellent",
  good: "Good",
  mediocre: "Mediocre",
  poor: "Poor",
};

// ── Main component ──────────────────────────────────────────────────────────

export function FoodScoreCard({
  food,
  mealName,
  compact = false,
  className,
}: FoodScoreCardProps) {
  const result: FoodScoreResult = useMemo(
    () => calculateFoodScore(food),
    [food.calories, food.protein_g, food.carbs_g, food.fat_g, food.fiber_g, food.sugar_g, food.sodium_mg]
  );

  if (compact) {
    return (
      <div
        className={className}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "8px 0",
        }}
      >
        <ScoreRing score={result.score} color={result.color} size={52} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span
              style={{
                fontSize: 11,
                fontWeight: 600,
                color: result.color,
              }}
            >
              {RATING_LABELS[result.rating]}
            </span>
          </div>
          <p
            style={{
              fontSize: 10,
              color: "var(--muted-foreground)",
              margin: "2px 0 0 0",
              lineHeight: 1.4,
              overflow: "hidden",
              textOverflow: "ellipsis",
              display: "-webkit-box",
              WebkitLineClamp: 2,
              WebkitBoxOrient: "vertical",
            }}
          >
            {result.brainImpact}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      className={className}
      style={{
        background: "var(--card)",
        border: "1px solid var(--border)",
        borderRadius: 16,
        padding: 16,
        boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
      }}
    >
      {/* Header */}
      {mealName && (
        <div
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: "var(--muted-foreground)",
            textTransform: "uppercase",
            letterSpacing: "0.5px",
            marginBottom: 12,
          }}
        >
          {mealName}
        </div>
      )}

      {/* Score circle + rating */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <ScoreRing score={result.score} color={result.color} size={120} />
        <div
          style={{
            marginTop: 8,
            padding: "3px 14px",
            borderRadius: 20,
            background: `${result.color}18`,
            display: "inline-flex",
            alignItems: "center",
            gap: 4,
          }}
        >
          <span
            style={{
              fontSize: 12,
              fontWeight: 700,
              color: result.color,
            }}
          >
            {RATING_LABELS[result.rating]}
          </span>
        </div>
      </div>

      {/* Nutrient breakdown bars */}
      <div style={{ marginBottom: 16 }}>
        <div
          style={{
            fontSize: 10,
            fontWeight: 600,
            color: "var(--muted-foreground)",
            textTransform: "uppercase",
            letterSpacing: "0.5px",
            marginBottom: 8,
          }}
        >
          Nutrient Breakdown
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {result.nutrientFlags.map((flag) => {
            const barColor = nutrientBarColor(flag);
            const barWidth = Math.min(flag.dailyPct, 100);
            return (
              <div key={flag.nutrient}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 3,
                  }}
                >
                  <span
                    style={{
                      fontSize: 11,
                      fontWeight: 500,
                      color: "var(--foreground)",
                    }}
                  >
                    {flag.nutrient}
                  </span>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span
                      style={{
                        fontSize: 10,
                        color: "var(--muted-foreground)",
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {flag.dailyPct}% daily
                    </span>
                    <span
                      style={{
                        fontSize: 9,
                        fontWeight: 600,
                        color: barColor,
                        padding: "1px 6px",
                        borderRadius: 4,
                        background: `${barColor}15`,
                      }}
                    >
                      {nutrientLevelLabel(flag)}
                    </span>
                  </div>
                </div>
                <div
                  style={{
                    height: 4,
                    borderRadius: 2,
                    background: "var(--border)",
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      borderRadius: 2,
                      background: barColor,
                      width: `${barWidth}%`,
                      transition: "width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)",
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Brain Impact section */}
      <div
        style={{
          background: "linear-gradient(135deg, rgba(124,58,237,0.06) 0%, rgba(6,182,212,0.06) 100%)",
          border: "1px solid rgba(124,58,237,0.12)",
          borderRadius: 12,
          padding: 12,
          marginBottom: 10,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            marginBottom: 6,
          }}
        >
          <Brain style={{ width: 14, height: 14, color: "#7c3aed" }} />
          <span
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: "var(--foreground)",
            }}
          >
            Brain Impact
          </span>
        </div>
        <p
          style={{
            fontSize: 11,
            color: "var(--muted-foreground)",
            margin: 0,
            lineHeight: 1.5,
          }}
        >
          {result.brainImpact}
        </p>
      </div>

      {/* Mood Prediction section */}
      <div
        style={{
          background: "linear-gradient(135deg, rgba(232,121,168,0.06) 0%, rgba(228,180,74,0.06) 100%)",
          border: "1px solid rgba(232,121,168,0.12)",
          borderRadius: 12,
          padding: 12,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            marginBottom: 6,
          }}
        >
          {result.score >= 50 ? (
            <Zap style={{ width: 14, height: 14, color: "#e8b94a" }} />
          ) : (
            <TrendingDown style={{ width: 14, height: 14, color: "#e879a8" }} />
          )}
          <span
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: "var(--foreground)",
            }}
          >
            Mood Prediction
          </span>
        </div>
        <p
          style={{
            fontSize: 11,
            color: "var(--muted-foreground)",
            margin: 0,
            lineHeight: 1.5,
          }}
        >
          {result.moodPrediction}
        </p>
      </div>
    </div>
  );
}
