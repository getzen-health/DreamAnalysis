/**
 * InnerScoreCard — Hero gauge for the adaptive Inner Score (0-100).
 *
 * 220px SVG arc (270-degree), tap-to-expand factor breakdown,
 * building state when no data, 7-day sparkline, delta indicator.
 */

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { getScoreLabel, getScoreColor, getTierConfidence, type Tier } from "@/lib/inner-score";

// ─── Types ───────────────────────────────────────────────────────────────────

export interface InnerScoreCardProps {
  score: number | null;
  tier: Tier;
  factors: Record<string, number>;
  narrative: string;
  delta: number | null;
  trend: (number | null)[];
}

// ─── Factor label mapping ────────────────────────────────────────────────────

const FACTOR_DISPLAY: Record<string, string> = {
  sleep_quality: "Sleep",
  stress_inverse: "Stress",
  valence: "Mood",
  energy: "Energy",
  hrv_trend: "HRV",
  activity: "Activity",
  brain_health: "Brain Health",
};

function factorBarColor(value: number): string {
  if (value >= 70) return "hsl(var(--success))";
  if (value >= 40) return "hsl(var(--warning))";
  return "hsl(var(--destructive))";
}

// ─── SVG Arc Helpers ─────────────────────────────────────────────────────────

const SIZE = 220;
const STROKE = 12;
const RADIUS = (SIZE - STROKE) / 2;
const CENTER = SIZE / 2;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
const ARC_FRACTION = 270 / 360;
const ARC_LENGTH = CIRCUMFERENCE * ARC_FRACTION;
const START_ANGLE = 135; // degrees — bottom-left start

function polarToCartesian(angle: number): { x: number; y: number } {
  const rad = ((angle - 90) * Math.PI) / 180;
  return { x: CENTER + RADIUS * Math.cos(rad), y: CENTER + RADIUS * Math.sin(rad) };
}

function describeArc(startAngle: number, endAngle: number): string {
  const start = polarToCartesian(endAngle);
  const end = polarToCartesian(startAngle);
  const largeArc = endAngle - startAngle <= 180 ? 0 : 1;
  return `M ${start.x} ${start.y} A ${RADIUS} ${RADIUS} 0 ${largeArc} 0 ${end.x} ${end.y}`;
}

// ─── Mini Sparkline ──────────────────────────────────────────────────────────

function MiniSparkline({ data }: { data: (number | null)[] }) {
  if (!data.length || data.every((d) => d == null)) return null;
  const w = 56;
  const h = 20;
  const points = data
    .map((v, i) => (v != null ? { x: (i / Math.max(data.length - 1, 1)) * w, y: h - (v / 100) * h } : null))
    .filter(Boolean) as { x: number; y: number }[];
  if (points.length < 2) return null;
  const d = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
  return (
    <svg width={w} height={h} className="opacity-50">
      <path d={d} fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" />
    </svg>
  );
}

// ─── Component ───────────────────────────────────────────────────────────────

export function InnerScoreCard({ score, tier, factors, narrative, delta, trend }: InnerScoreCardProps) {
  const [expanded, setExpanded] = useState(false);
  const isBuilding = score == null;
  const label = isBuilding ? "Building" : getScoreLabel(score);
  const color = isBuilding ? "var(--muted-foreground)" : getScoreColor(score);
  const confidence = getTierConfidence(tier);

  // Arc dash offset
  const fraction = isBuilding ? 0 : Math.max(0, Math.min(1, score / 100));
  const dashOffset = ARC_LENGTH * (1 - fraction);

  const backgroundArc = describeArc(START_ANGLE, START_ANGLE + 270);

  return (
    <div
      data-testid="inner-score-card"
      className="rounded-[14px] bg-card border border-border p-6 cursor-pointer select-none"
      onClick={() => !isBuilding && setExpanded((e) => !e)}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); !isBuilding && setExpanded((v) => !v); } }}
      tabIndex={0}
      role="button"
      aria-expanded={expanded}
    >
      {/* Gauge */}
      <div className="flex flex-col items-center">
        <svg
          width={SIZE}
          height={SIZE}
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          data-testid="inner-score-gauge"
          aria-label={isBuilding ? "Inner Score: building" : `Inner Score: ${score} out of 100, ${label}`}
          role="img"
        >
          <defs>
            <linearGradient id="inner-score-grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="hsl(var(--primary))" />
              <stop offset="100%" stopColor="hsl(var(--success))" />
            </linearGradient>
          </defs>

          {/* Background track */}
          <path
            d={backgroundArc}
            fill="none"
            stroke="hsl(var(--border))"
            strokeWidth={STROKE}
            strokeLinecap="round"
          />

          {/* Score arc */}
          {!isBuilding && (
            <circle
              cx={CENTER}
              cy={CENTER}
              r={RADIUS}
              fill="none"
              stroke="url(#inner-score-grad)"
              strokeWidth={STROKE}
              strokeLinecap="round"
              strokeDasharray={`${ARC_LENGTH} ${CIRCUMFERENCE}`}
              strokeDashoffset={dashOffset}
              transform={`rotate(${START_ANGLE} ${CENTER} ${CENTER})`}
              style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(0.34, 1.56, 0.64, 1)" }}
            />
          )}

          {/* Center text */}
          <text
            x={CENTER}
            y={CENTER - 8}
            textAnchor="middle"
            dominantBaseline="central"
            className="fill-foreground"
            style={{ fontSize: 52, fontWeight: 700, fontFamily: "var(--font-sans)" }}
          >
            {isBuilding ? "\u2014" : score}
          </text>
          <text
            x={CENTER}
            y={CENTER + 28}
            textAnchor="middle"
            dominantBaseline="central"
            className="fill-muted-foreground"
            style={{ fontSize: 13, fontWeight: 500, fontFamily: "var(--font-sans)" }}
          >
            Inner Score
          </text>
        </svg>

        {/* Building state */}
        {isBuilding && (
          <div data-testid="inner-score-building" className="text-center mt-2 animate-pulse">
            <p className="text-sm text-muted-foreground">
              Do a voice check-in to get your Inner Score
            </p>
          </div>
        )}

        {/* Label + confidence + delta */}
        {!isBuilding && (
          <div className="flex flex-col items-center gap-1 -mt-2">
            <span className="text-sm font-semibold" style={{ color }}>{label}</span>
            <span className="text-[10px] text-muted-foreground">{confidence}</span>
            <div className="flex items-center gap-2">
              {delta != null && (
                <span className={`text-xs font-mono font-medium ${delta >= 0 ? "text-emerald-500" : "text-red-400"}`}>
                  {delta >= 0 ? `+${delta}` : `${delta}`}
                </span>
              )}
              <MiniSparkline data={trend} />
            </div>
            <ChevronDown
              className={`h-4 w-4 text-muted-foreground transition-transform ${expanded ? "rotate-180" : ""}`}
            />
          </div>
        )}
      </div>

      {/* Expanded factor breakdown */}
      <AnimatePresence>
        {expanded && !isBuilding && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            className="overflow-hidden"
          >
            <div className="pt-4 space-y-3">
              {/* Narrative */}
              {narrative && (
                <p className="text-xs text-muted-foreground leading-relaxed italic">
                  {narrative}
                </p>
              )}

              {/* Factor bars */}
              {Object.entries(factors).map(([key, value]) => (
                <div key={key} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">{FACTOR_DISPLAY[key] ?? key}</span>
                    <span className="font-mono font-medium">{value}%</span>
                  </div>
                  <div
                    className="h-1.5 rounded-full bg-border overflow-hidden"
                    role="progressbar"
                    aria-valuenow={value}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  >
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{ width: `${Math.max(2, value)}%`, background: factorBarColor(value) }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
