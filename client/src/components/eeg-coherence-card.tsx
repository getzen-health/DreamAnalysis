/**
 * EEGCoherenceCard — Brain region connectivity visualization.
 *
 * Shows the 4 BCI device electrode positions (TP9, AF7, AF8, TP10) as nodes
 * in a head silhouette and draws arcs between them proportional to the
 * Phase Locking Value (PLV) or coherence strength.
 *
 * Key connections:
 *   AF7 ↔ AF8 — inter-hemispheric frontal (emotion regulation, FAA source)
 *   TP9 ↔ TP10 — inter-hemispheric temporal (memory, language lateralization)
 *   AF7 ↔ TP9 — left fronto-temporal (left-hemisphere language/logic)
 *   AF8 ↔ TP10 — right fronto-temporal (right-hemisphere creativity/spatial)
 *   AF7 ↔ TP10 — cross-hemispheric (integration)
 *   AF8 ↔ TP9 — cross-hemispheric (integration)
 *
 * Unique EEG differentiator — no consumer wellness app shows brain connectivity.
 */

import { GitBranch } from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface EEGCoherenceCardProps {
  /** AF7 ↔ AF8 inter-hemispheric frontal PLV (0–1) */
  frontalPlv?: number | null;
  /** TP9 ↔ TP10 inter-hemispheric temporal PLV (0–1) */
  temporalPlv?: number | null;
  /** AF7↔TP9 + AF8↔TP10 fronto-temporal PLV mean (0–1) */
  leftFrontotemporalPlv?: number | null;
  /** Is EEG currently streaming? */
  isStreaming?: boolean;
}

// ─── Electrode Positions (in a 100×110 viewBox, centered on forehead-down head) ──

// Head circle: cx=50, cy=55, r=44
// TP9  = left ear area
// TP10 = right ear area
// AF7  = left frontal
// AF8  = right frontal
const NODES = {
  TP9:  { x: 14, y: 62, label: "TP9",  desc: "Left temporal" },
  AF7:  { x: 24, y: 22, label: "AF7",  desc: "Left frontal" },
  AF8:  { x: 76, y: 22, label: "AF8",  desc: "Right frontal" },
  TP10: { x: 86, y: 62, label: "TP10", desc: "Right temporal" },
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

function strengthColor(v: number): string {
  // Low (0–0.3): cool blue, Mid (0.3–0.6): teal, High (0.6–1): violet
  if (v >= 0.6) return "hsl(270,75%,65%)";
  if (v >= 0.3) return "hsl(175,70%,50%)";
  return "hsl(210,70%,55%)";
}

function strengthLabel(v: number): string {
  // Thresholds match strengthColor: 0.3 and 0.6
  if (v >= 0.6) return "Strong";
  if (v >= 0.3) return "Moderate";
  return "Weak";
}

// ─── Sub-components ───────────────────────────────────────────────────────────

interface ArcProps {
  from: { x: number; y: number };
  to: { x: number; y: number };
  strength: number; // 0–1
}

function CoherenceArc({ from, to, strength }: ArcProps) {
  if (strength < 0.05) return null;

  // Midpoint with a perpendicular offset for the bezier control point
  const mx = (from.x + to.x) / 2;
  const my = (from.y + to.y) / 2;
  // Push control point toward center of head (50, 55)
  const cx = mx + (50 - mx) * 0.3;
  const cy = my + (55 - my) * 0.3;

  const d = `M ${from.x} ${from.y} Q ${cx} ${cy} ${to.x} ${to.y}`;
  const color = strengthColor(strength);
  const strokeWidth = 0.8 + strength * 2.5;
  const opacity = 0.25 + strength * 0.65;

  return (
    <path
      d={d}
      stroke={color}
      strokeWidth={strokeWidth}
      fill="none"
      opacity={opacity}
      strokeLinecap="round"
    />
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function EEGCoherenceCard({
  frontalPlv,
  temporalPlv,
  leftFrontotemporalPlv,
  isStreaming = false,
}: EEGCoherenceCardProps) {
  const hasData =
    frontalPlv != null || temporalPlv != null || leftFrontotemporalPlv != null;

  const pairs: { from: keyof typeof NODES; to: keyof typeof NODES; strength: number }[] = [];
  if (frontalPlv != null)            pairs.push({ from: "AF7", to: "AF8",  strength: frontalPlv });
  if (temporalPlv != null)           pairs.push({ from: "TP9", to: "TP10", strength: temporalPlv });
  // fronto-temporal PLV is the mean of AF7↔TP9 and AF8↔TP10 — draw both arcs at the same strength
  if (leftFrontotemporalPlv != null) {
    pairs.push({ from: "AF7", to: "TP9",  strength: leftFrontotemporalPlv });
    pairs.push({ from: "AF8", to: "TP10", strength: leftFrontotemporalPlv });
  }

  return (
    <div
      data-testid="eeg-coherence-card"
      className="rounded-2xl border border-border/40 bg-card/60 p-4"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <GitBranch className="h-3.5 w-3.5 text-violet-400" />
        <h2 className="text-sm font-semibold">Brain Connectivity</h2>
        <span className="text-[10px] font-mono text-violet-400/70 bg-violet-500/10 px-1.5 py-0.5 rounded-full ml-auto">
          PLV
        </span>
      </div>

      <div className="flex gap-4 items-center">
        {/* SVG diagram */}
        <div className="shrink-0">
          <svg
            viewBox="0 0 100 100"
            width={90}
            height={90}
            aria-label="Brain connectivity diagram"
            data-testid="coherence-svg"
          >
            {/* Head circle */}
            <circle
              cx={50} cy={55} r={42}
              fill="hsl(240,15%,10%)"
              stroke="hsl(0,0%,100%,0.08)"
              strokeWidth={0.8}
            />
            {/* Frontal direction mark */}
            <path
              d="M 42 14 Q 50 10 58 14"
              fill="none"
              stroke="hsl(0,0%,100%,0.12)"
              strokeWidth={0.8}
            />

            {/* Coherence arcs */}
            {pairs.map(({ from, to, strength }) => (
              <CoherenceArc
                key={`${from}-${to}`}
                from={NODES[from]}
                to={NODES[to]}
                strength={strength}
              />
            ))}

            {/* Electrode nodes */}
            {(Object.keys(NODES) as (keyof typeof NODES)[]).map((key) => {
              const { x, y, label } = NODES[key];
              return (
                <g key={key}>
                  <circle
                    cx={x} cy={y} r={4.5}
                    fill="hsl(240,25%,18%)"
                    stroke={hasData ? "hsl(270,60%,60%)" : "hsl(210,40%,45%)"}
                    strokeWidth={1}
                  />
                  <text
                    x={x}
                    y={y + 8.5}
                    textAnchor="middle"
                    fontSize={5}
                    fill="hsl(0,0%,100%,0.5)"
                    fontFamily="monospace"
                  >
                    {label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Connectivity metrics */}
        <div className="flex-1 space-y-2 min-w-0">
          {!hasData ? (
            <div data-testid="coherence-empty">
              <p className="text-[11px] text-muted-foreground/60 leading-relaxed">
                {isStreaming
                  ? "Computing connectivity from live EEG..."
                  : "Connect EEG headset to see brain connectivity"}
              </p>
            </div>
          ) : (
            <>
              {frontalPlv != null && (
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-muted-foreground truncate">Frontal (FAA)</span>
                  <span
                    className="text-[10px] font-semibold font-mono"
                    style={{ color: strengthColor(frontalPlv) }}
                  >
                    {strengthLabel(frontalPlv)}
                  </span>
                </div>
              )}
              {temporalPlv != null && (
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-muted-foreground truncate">Temporal</span>
                  <span
                    className="text-[10px] font-semibold font-mono"
                    style={{ color: strengthColor(temporalPlv) }}
                  >
                    {strengthLabel(temporalPlv)}
                  </span>
                </div>
              )}
              {leftFrontotemporalPlv != null && (
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-muted-foreground truncate">Fronto-temporal</span>
                  <span
                    className="text-[10px] font-semibold font-mono"
                    style={{ color: strengthColor(leftFrontotemporalPlv) }}
                  >
                    {strengthLabel(leftFrontotemporalPlv)}
                  </span>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
