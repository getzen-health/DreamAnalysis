/**
 * EnergyBattery -- battery-shaped visualization for the Energy Bank score.
 *
 * Features:
 * - Horizontal battery with rounded rect body + terminal nub on right
 * - Gradient fill: pink when high, transitions to red when low
 * - Animated fill level via framer-motion
 * - Large number inside the battery
 * - "Energy Bank" label above
 */

import { useId } from "react";
import { motion, useMotionValue, useTransform, animate } from "framer-motion";
import { useEffect } from "react";

// ── Types ─────────────────────────────────────────────────────────────────────

interface EnergyBatteryProps {
  value: number | null;
  max?: number;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const BATTERY = {
  width: 240,
  height: 80,
  bodyWidth: 220,
  bodyHeight: 70,
  bodyRx: 12,
  terminalWidth: 10,
  terminalHeight: 28,
  terminalRx: 4,
  padding: 6,
};

// Color interpolation: magenta (#d946ef) when high, coral (#e879a8) when low
function getFillColors(pct: number): { from: string; to: string } {
  if (pct > 0.6) return { from: "#d946ef", to: "#a21caf" }; // electric magenta
  if (pct > 0.3) return { from: "#ea580c", to: "#c2410c" }; // burnt orange
  return { from: "#e879a8", to: "#be185d" }; // warm coral
}

// ── Component ─────────────────────────────────────────────────────────────────

export function EnergyBattery({ value, max = 100 }: EnergyBatteryProps) {
  const gradientId = useId();
  const isNull = value === null || value === undefined;
  const pct = isNull ? 0 : Math.min(Math.max(value / max, 0), 1);
  const colors = getFillColors(pct);

  const bodyX = (BATTERY.width - BATTERY.bodyWidth - BATTERY.terminalWidth) / 2;
  const bodyY = (BATTERY.height - BATTERY.bodyHeight) / 2;
  const terminalX = bodyX + BATTERY.bodyWidth;
  const terminalY = (BATTERY.height - BATTERY.terminalHeight) / 2;

  // Inner fill dimensions
  const fillX = bodyX + BATTERY.padding;
  const fillY = bodyY + BATTERY.padding;
  const maxFillWidth = BATTERY.bodyWidth - BATTERY.padding * 2;
  const fillHeight = BATTERY.bodyHeight - BATTERY.padding * 2;
  const fillRx = BATTERY.bodyRx - 3;

  // Animated fill width
  const progress = useMotionValue(0);
  const fillWidth = useTransform(progress, (v) => maxFillWidth * v);

  useEffect(() => {
    if (isNull) {
      progress.set(0);
      return;
    }
    const controls = animate(progress, pct, {
      duration: 1.0,
      ease: [0.34, 1.56, 0.64, 1],
    });
    return () => controls.stop();
  }, [pct, isNull, progress]);

  return (
    <div className="flex flex-col items-center gap-2">
      {/* Label */}
      <span className="text-xs font-semibold uppercase tracking-[0.1em] text-muted-foreground">
        Energy Bank
      </span>

      <svg
        width={BATTERY.width}
        height={BATTERY.height}
        viewBox={`0 0 ${BATTERY.width} ${BATTERY.height}`}
        aria-label={isNull ? "Energy Bank: no data" : `Energy Bank: ${value}`}
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={colors.from} />
            <stop offset="100%" stopColor={colors.to} />
          </linearGradient>
          {/* Glow filter */}
          <filter id={`${gradientId}-glow`}>
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Battery body outline */}
        <rect
          x={bodyX}
          y={bodyY}
          width={BATTERY.bodyWidth}
          height={BATTERY.bodyHeight}
          rx={BATTERY.bodyRx}
          fill="none"
          stroke="var(--border)"
          strokeWidth={2}
        />

        {/* Terminal nub */}
        <rect
          x={terminalX}
          y={terminalY}
          width={BATTERY.terminalWidth}
          height={BATTERY.terminalHeight}
          rx={BATTERY.terminalRx}
          fill="var(--border)"
        />

        {/* Animated fill */}
        {!isNull && (
          <motion.rect
            x={fillX}
            y={fillY}
            height={fillHeight}
            rx={fillRx}
            fill={`url(#${gradientId})`}
            style={{ width: fillWidth }}
            filter={pct > 0.5 ? `url(#${gradientId}-glow)` : undefined}
          />
        )}

        {/* Dashed interior when null */}
        {isNull && (
          <rect
            x={fillX}
            y={fillY}
            width={maxFillWidth}
            height={fillHeight}
            rx={fillRx}
            fill="none"
            stroke="var(--border)"
            strokeWidth={1}
            strokeDasharray="6 4"
            opacity={0.5}
          />
        )}

        {/* Large number inside battery */}
        <text
          x={bodyX + BATTERY.bodyWidth / 2}
          y={BATTERY.height / 2}
          textAnchor="middle"
          dominantBaseline="central"
          fill={isNull ? "var(--muted-foreground)" : "var(--foreground)"}
          fontSize={28}
          fontWeight="700"
          fontFamily="Inter, system-ui, sans-serif"
        >
          {isNull ? "\u2014" : Math.round(value!)}
        </text>

        {/* Percent sign (smaller, to the right of the number) */}
        {!isNull && (
          <text
            x={bodyX + BATTERY.bodyWidth / 2 + 24}
            y={BATTERY.height / 2}
            textAnchor="start"
            dominantBaseline="central"
            fill="var(--muted-foreground)"
            fontSize={14}
            fontWeight="500"
            fontFamily="Inter, system-ui, sans-serif"
          >
            %
          </text>
        )}
      </svg>
    </div>
  );
}
