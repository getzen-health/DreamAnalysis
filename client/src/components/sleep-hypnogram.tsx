/**
 * SleepHypnogram — SVG-based visualization of EEG-detected sleep stages over time.
 *
 * Renders the classic hypnogram shape:
 *   Wake at top, N1, N2, N3 (deep), REM at bottom.
 *   Horizontal bars show time spent in each stage.
 *   REM phases are highlighted in violet.
 *
 * This is a pure presentational component — pass stage history + total duration.
 */

import React from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

export type SleepStage = "Wake" | "N1" | "N2" | "N3" | "REM";

export interface StageEvent {
  /** Which stage started at this point */
  stage: SleepStage;
  /** Elapsed seconds from session start when this stage began */
  t: number;
}

interface SleepHypnogramProps {
  /** Ordered list of stage transitions (first entry should be t=0 or close to it) */
  stageHistory: StageEvent[];
  /** Total session duration in seconds */
  totalSeconds: number;
  /** Width of the SVG in pixels (default: 100% via viewBox) */
  width?: number;
  /** Height of the SVG in pixels */
  height?: number;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const STAGE_ORDER: SleepStage[] = ["Wake", "N1", "N2", "N3", "REM"];
const STAGE_COUNT = STAGE_ORDER.length;

const STAGE_COLORS: Record<SleepStage, string> = {
  Wake: "hsl(0, 0%, 55%)",
  N1:   "hsl(48, 90%, 55%)",
  N2:   "hsl(210, 80%, 55%)",
  N3:   "hsl(240, 65%, 55%)",
  REM:  "hsl(270, 75%, 60%)",
};

const STAGE_LABEL: Record<SleepStage, string> = {
  Wake: "Wake",
  N1:   "N1",
  N2:   "N2",
  N3:   "N3",
  REM:  "REM",
};

// ─── Component ────────────────────────────────────────────────────────────────

export function SleepHypnogram({
  stageHistory,
  totalSeconds,
  height = 80,
}: SleepHypnogramProps) {
  if (!stageHistory.length || totalSeconds <= 0) {
    return (
      <div
        data-testid="sleep-hypnogram-empty"
        className="rounded-xl border border-dashed border-border/40 p-4 text-center"
      >
        <p className="text-xs text-muted-foreground/50">No stage data recorded</p>
      </div>
    );
  }

  // SVG coordinate constants
  const LABEL_W = 32;   // left margin for stage labels
  const CHART_H = height;
  const ROW_H   = CHART_H / STAGE_COUNT;

  // Build list of {stage, startT, endT} segments from the transition events
  const segments: Array<{ stage: SleepStage; startT: number; endT: number }> = [];
  for (let i = 0; i < stageHistory.length; i++) {
    const startT = stageHistory[i].t;
    const endT   = i + 1 < stageHistory.length ? stageHistory[i + 1].t : totalSeconds;
    segments.push({ stage: stageHistory[i].stage, startT, endT });
  }

  // Map time to x-coordinate (0 to 100 in viewBox units, minus label width)
  const CHART_W = 100 - LABEL_W;
  const tToX = (t: number) => LABEL_W + (t / totalSeconds) * CHART_W;

  // Map stage to y-coordinate
  const stageToY = (stage: SleepStage) => STAGE_ORDER.indexOf(stage) * ROW_H;

  // Build the polyline points for the step chart
  const points: string[] = [];
  for (const seg of segments) {
    const x1 = tToX(seg.startT);
    const x2 = tToX(seg.endT);
    const y  = stageToY(seg.stage) + ROW_H / 2;
    // Horizontal line for this segment
    points.push(`${x1},${y}`);
    points.push(`${x2},${y}`);
  }

  return (
    <div data-testid="sleep-hypnogram" className="w-full">
      <svg
        viewBox={`0 0 100 ${CHART_H}`}
        width="100%"
        height={height}
        preserveAspectRatio="none"
        style={{ display: "block" }}
        aria-label="Sleep stage hypnogram"
      >
        {/* Stage label rows + horizontal grid lines */}
        {STAGE_ORDER.map((stage, i) => {
          const y = i * ROW_H;
          const midY = y + ROW_H / 2;
          return (
            <g key={stage}>
              {/* Subtle row background for REM */}
              {stage === "REM" && (
                <rect
                  x={LABEL_W}
                  y={y}
                  width={CHART_W}
                  height={ROW_H}
                  fill="hsl(270, 75%, 60%, 0.06)"
                />
              )}
              {/* Grid line */}
              <line
                x1={LABEL_W}
                y1={midY}
                x2={100}
                y2={midY}
                stroke="hsl(0,0%,100%,0.04)"
                strokeWidth={0.3}
              />
              {/* Stage label */}
              <text
                x={LABEL_W - 2}
                y={midY + 3}
                textAnchor="end"
                fontSize={4.5}
                fill="hsl(0,0%,100%,0.35)"
                fontFamily="monospace"
              >
                {STAGE_LABEL[stage]}
              </text>
            </g>
          );
        })}

        {/* Colored fill rectangles per segment */}
        {segments.map((seg, i) => {
          const x1 = tToX(seg.startT);
          const x2 = tToX(seg.endT);
          const y  = stageToY(seg.stage);
          return (
            <rect
              key={i}
              x={x1}
              y={y + 1}
              width={Math.max(0, x2 - x1)}
              height={ROW_H - 2}
              fill={STAGE_COLORS[seg.stage]}
              opacity={0.25}
              rx={0.5}
            />
          );
        })}

        {/* Step-chart polyline connecting midpoints */}
        {points.length > 0 && (
          <polyline
            points={points.join(" ")}
            fill="none"
            stroke="hsl(270, 75%, 70%)"
            strokeWidth={0.8}
            strokeLinejoin="round"
            opacity={0.9}
          />
        )}
      </svg>
    </div>
  );
}
