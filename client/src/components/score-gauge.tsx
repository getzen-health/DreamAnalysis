/**
 * ScoreGauge -- premium circular SVG score gauge.
 *
 * Features:
 * - 270-degree arc with gradient stroke (gap at bottom)
 * - Animated fill on mount via framer-motion
 * - Three sizes: sm (80px), md (120px), lg (160px)
 * - Null state: dashed gray arc with em-dash
 * - Signature color per score type
 */

import { useId } from "react";
import { motion, useMotionValue, useTransform, animate } from "framer-motion";
import { useEffect } from "react";

// ── Types ─────────────────────────────────────────────────────────────────────

export type ScoreColor = "recovery" | "sleep" | "strain" | "stress" | "nutrition" | "energy";

export interface ScoreGaugeProps {
  value: number | null;
  max?: number;
  label: string;
  color: ScoreColor;
  size?: "sm" | "md" | "lg";
  subtitle?: string;
}

// ── Constants ─────────────────────────────────────────────────────────────────

export const SCORE_COLORS: Record<ScoreColor, { from: string; to: string }> = {
  recovery:  { from: "#0891b2", to: "#0e7490" },
  sleep:     { from: "#7c3aed", to: "#6d28d9" },
  strain:    { from: "#e879a8", to: "#be185d" },
  stress:    { from: "#ea580c", to: "#c2410c" },
  nutrition: { from: "#d4a017", to: "#a16207" },
  energy:    { from: "#d946ef", to: "#a21caf" },
};

const SIZES = {
  sm: { width: 80,  stroke: 6,  fontSize: 20, labelSize: 10 },
  md: { width: 120, stroke: 8,  fontSize: 28, labelSize: 12 },
  lg: { width: 160, stroke: 10, fontSize: 36, labelSize: 13 },
};

// ── Component ─────────────────────────────────────────────────────────────────

export function ScoreGauge({
  value,
  max = 100,
  label,
  color,
  size = "md",
  subtitle,
}: ScoreGaugeProps) {
  const gradientId = useId();
  const s = SIZES[size];
  const cx = s.width / 2;
  const cy = s.width / 2;
  const r = (s.width - s.stroke * 2) / 2;
  const circumference = 2 * Math.PI * r;
  const arcLength = circumference * 0.75; // 270 degrees
  const gapLength = circumference * 0.25; // 90 degrees

  const colors = SCORE_COLORS[color];
  const isNull = value === null || value === undefined;
  const pct = isNull ? 0 : Math.min(Math.max(value / max, 0), 1);

  // Animated dashoffset via framer-motion
  const progress = useMotionValue(0);
  const dashOffset = useTransform(progress, (v) => arcLength * (1 - v));

  useEffect(() => {
    if (isNull) {
      progress.set(0);
      return;
    }
    const controls = animate(progress, pct, {
      duration: 1.2,
      ease: [0.34, 1.56, 0.64, 1],
    });
    return () => controls.stop();
  }, [pct, isNull, progress, arcLength]);

  return (
    <div className="flex flex-col items-center">
      <svg
        width={s.width}
        height={s.width}
        viewBox={`0 0 ${s.width} ${s.width}`}
        className="drop-shadow-lg"
        aria-label={isNull ? `${label}: no data` : `${label}: ${value}`}
      >
        <defs>
          <linearGradient
            id={gradientId}
            gradientUnits="userSpaceOnUse"
            x1={s.width * 0.15}
            y1={s.width * 0.15}
            x2={s.width * 0.85}
            y2={s.width * 0.85}
          >
            <stop offset="0%" stopColor={colors.from} />
            <stop offset="100%" stopColor={colors.to} />
          </linearGradient>
        </defs>

        {/* Background track */}
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke="var(--border)"
          strokeWidth={s.stroke}
          strokeDasharray={isNull ? `${4} ${6}` : `${arcLength} ${gapLength}`}
          strokeLinecap="round"
          transform={isNull ? undefined : `rotate(135 ${cx} ${cy})`}
          opacity={0.5}
        />

        {/* Animated score arc (hidden when null) */}
        {!isNull && (
          <motion.circle
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke={`url(#${gradientId})`}
            strokeWidth={s.stroke}
            strokeDasharray={`${arcLength} ${gapLength}`}
            strokeLinecap="round"
            transform={`rotate(135 ${cx} ${cy})`}
            style={{ strokeDashoffset: dashOffset }}
          />
        )}

        {/* Score number / em-dash */}
        <text
          x={cx}
          y={cy - (subtitle ? s.fontSize * 0.15 : 2)}
          textAnchor="middle"
          dominantBaseline="central"
          fill={isNull ? "var(--muted-foreground)" : "var(--foreground)"}
          fontSize={s.fontSize}
          fontWeight="700"
          fontFamily="Inter, system-ui, sans-serif"
        >
          {isNull ? "\u2014" : Math.round(value!)}
        </text>

        {/* Label */}
        <text
          x={cx}
          y={cy + s.fontSize * 0.55}
          textAnchor="middle"
          dominantBaseline="central"
          fill="var(--muted-foreground)"
          fontSize={s.labelSize}
          fontFamily="Inter, system-ui, sans-serif"
        >
          {label}
        </text>

        {/* Optional subtitle (below label) */}
        {subtitle && (
          <text
            x={cx}
            y={cy + s.fontSize * 0.55 + s.labelSize + 4}
            textAnchor="middle"
            dominantBaseline="central"
            fill="var(--muted-foreground)"
            fontSize={s.labelSize - 1}
            fontFamily="Inter, system-ui, sans-serif"
            opacity={0.7}
          >
            {subtitle}
          </text>
        )}
      </svg>
    </div>
  );
}
