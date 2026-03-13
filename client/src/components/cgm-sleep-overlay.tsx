/**
 * CGMSleepOverlay — Recharts ComposedChart overlaying overnight glucose trace
 * (AreaChart in mg/dL) on sleep-stage bars.
 *
 * Glucose colour bands:
 *   green  70–120 mg/dL   (in range)
 *   yellow 120–140 mg/dL  (elevated)
 *   red    >140 mg/dL     (high)
 *
 * Excursion markers that coincide with awakenings are highlighted.
 *
 * Props:
 *   glucoseReadings   — array of { timestamp: string; value: number }
 *   sleepStages       — array of { time: string; stage: SleepStageKey; duration: number }
 *   awakeningTimes    — optional ISO timestamps of confirmed awakening events
 *   metrics           — optional precomputed summary metrics
 */

import {
  Area,
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// ─── Types ────────────────────────────────────────────────────────────────────

export type SleepStageKey = "Wake" | "N1" | "N2" | "N3" | "REM";

export interface GlucoseReading {
  timestamp: string; // ISO-8601
  value: number;     // mg/dL
}

export interface SleepStageEntry {
  time: string;           // ISO-8601 start of interval
  stage: SleepStageKey;
  duration: number;       // minutes
}

export interface CGMSleepMetrics {
  mean_glucose: number;
  glucose_std: number;
  time_in_range_pct: number;
  excursion_count: number;
}

interface CGMSleepOverlayProps {
  glucoseReadings: GlucoseReading[];
  sleepStages: SleepStageEntry[];
  awakeningTimes?: string[];
  metrics?: CGMSleepMetrics;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const GLUCOSE_LOW        = 70;
const GLUCOSE_TARGET_MAX = 120;
const GLUCOSE_HIGH       = 140;

const STAGE_NUMERIC: Record<SleepStageKey, number> = {
  Wake: 5,
  REM:  4,
  N1:   3,
  N2:   2,
  N3:   1,
};

const STAGE_COLOR: Record<SleepStageKey, string> = {
  Wake: "hsl(30, 90%, 55%)",
  N1:   "hsl(48, 90%, 55%)",
  N2:   "hsl(210, 80%, 55%)",
  N3:   "hsl(240, 65%, 55%)",
  REM:  "hsl(270, 75%, 60%)",
};

const STAGE_LABEL: Record<SleepStageKey, string> = {
  Wake: "Awake",
  N1:   "Light (N1)",
  N2:   "Core (N2)",
  N3:   "Deep (N3)",
  REM:  "REM",
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function glucoseColor(value: number): string {
  if (value > GLUCOSE_HIGH)       return "hsl(0, 75%, 55%)";    // red
  if (value > GLUCOSE_TARGET_MAX) return "hsl(43, 90%, 50%)";   // yellow
  return "hsl(142, 60%, 45%)";                                   // green
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return iso;
  }
}

/** Merge glucose and sleep-stage arrays into a single timeline. */
function buildChartData(
  glucoseReadings: GlucoseReading[],
  sleepStages: SleepStageEntry[],
  awakeningTimes: string[],
) {
  // Index sleep stages by minute bucket for lookup
  const stageByMinute = new Map<string, SleepStageKey>();
  for (const s of sleepStages) {
    const start = new Date(s.time).getTime();
    for (let m = 0; m < s.duration; m++) {
      const key = new Date(start + m * 60_000).toISOString().substring(0, 16);
      stageByMinute.set(key, s.stage);
    }
  }

  const awakeSet = new Set(awakeningTimes.map((t) => t.substring(0, 16)));

  return glucoseReadings.map((r) => {
    const bucket = r.timestamp.substring(0, 16);
    const stage  = stageByMinute.get(bucket) ?? null;
    const isAwakening = awakeSet.has(bucket);
    const isExcursion = r.value > GLUCOSE_TARGET_MAX;

    return {
      time:            formatTime(r.timestamp),
      glucose:         r.value,
      glucoseColor:    glucoseColor(r.value),
      stageNumeric:    stage ? STAGE_NUMERIC[stage] : null,
      stageName:       stage ?? "—",
      stageColor:      stage ? STAGE_COLOR[stage] : "transparent",
      isExcursion,
      isAwakening,
      excursionAwakening: isExcursion && isAwakening,
    };
  });
}

// ─── Custom tooltip ───────────────────────────────────────────────────────────

function CGMTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload ?? {};

  return (
    <div
      style={{
        background: "hsl(222, 47%, 11%)",
        border: "1px solid hsl(215, 28%, 17%)",
        borderRadius: 8,
        padding: "8px 12px",
        fontSize: 13,
        color: "hsl(210, 40%, 98%)",
        minWidth: 160,
      }}
    >
      <p style={{ fontWeight: 600, marginBottom: 4 }}>{label}</p>
      <p style={{ color: glucoseColor(d.glucose) }}>
        Glucose: <strong>{d.glucose} mg/dL</strong>
      </p>
      {d.stageName !== "—" && (
        <p style={{ color: d.stageColor }}>
          Stage: <strong>{STAGE_LABEL[d.stageName as SleepStageKey] ?? d.stageName}</strong>
        </p>
      )}
      {d.isExcursion && (
        <p style={{ color: "hsl(43, 90%, 50%)", fontSize: 11 }}>
          ⬆ Excursion {d.excursionAwakening ? "— coincides with awakening" : ""}
        </p>
      )}
      {d.isAwakening && !d.isExcursion && (
        <p style={{ color: "hsl(30, 90%, 55%)", fontSize: 11 }}>Awakening event</p>
      )}
    </div>
  );
}

// ─── Metric badge ─────────────────────────────────────────────────────────────

function MetricBadge({
  label,
  value,
  unit,
  color,
}: {
  label: string;
  value: string | number;
  unit?: string;
  color?: string;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        background: "hsl(222, 47%, 11%)",
        border: "1px solid hsl(215, 28%, 17%)",
        borderRadius: 8,
        padding: "8px 14px",
        minWidth: 90,
      }}
    >
      <span style={{ fontSize: 11, color: "hsl(215, 20%, 55%)", marginBottom: 2 }}>
        {label}
      </span>
      <span style={{ fontSize: 18, fontWeight: 700, color: color ?? "hsl(210, 40%, 98%)" }}>
        {value}
        {unit && (
          <span style={{ fontSize: 11, fontWeight: 400, color: "hsl(215, 20%, 55%)" }}>
            {" "}{unit}
          </span>
        )}
      </span>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function CGMSleepOverlay({
  glucoseReadings,
  sleepStages,
  awakeningTimes = [],
  metrics,
}: CGMSleepOverlayProps) {
  const data = buildChartData(glucoseReadings, sleepStages, awakeningTimes);

  const hasGlucose = glucoseReadings.length > 0;
  const hasSleep   = sleepStages.length > 0;

  if (!hasGlucose && !hasSleep) {
    return (
      <div
        style={{
          padding: 24,
          textAlign: "center",
          color: "hsl(215, 20%, 55%)",
          background: "hsl(222, 47%, 11%)",
          borderRadius: 12,
        }}
      >
        No CGM or sleep data available for this night.
      </div>
    );
  }

  return (
    <div
      style={{
        background: "hsl(222, 47%, 9%)",
        borderRadius: 12,
        padding: 20,
        fontFamily: "system-ui, sans-serif",
      }}
    >
      {/* Header */}
      <h3 style={{ color: "hsl(210, 40%, 98%)", margin: "0 0 16px", fontSize: 16 }}>
        Overnight Glucose + Sleep Stages
      </h3>

      {/* Metric badges */}
      {metrics && (
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 18 }}>
          <MetricBadge
            label="Mean Glucose"
            value={metrics.mean_glucose.toFixed(0)}
            unit="mg/dL"
            color={glucoseColor(metrics.mean_glucose)}
          />
          <MetricBadge
            label="Variability (SD)"
            value={metrics.glucose_std.toFixed(1)}
            unit="mg/dL"
            color={
              metrics.glucose_std > 20
                ? "hsl(0, 75%, 55%)"
                : metrics.glucose_std > 10
                ? "hsl(43, 90%, 50%)"
                : "hsl(142, 60%, 45%)"
            }
          />
          <MetricBadge
            label="Time in Range"
            value={metrics.time_in_range_pct.toFixed(0)}
            unit="%"
            color={
              metrics.time_in_range_pct >= 70
                ? "hsl(142, 60%, 45%)"
                : metrics.time_in_range_pct >= 50
                ? "hsl(43, 90%, 50%)"
                : "hsl(0, 75%, 55%)"
            }
          />
          {metrics.excursion_count > 0 && (
            <MetricBadge
              label="Excursions"
              value={metrics.excursion_count}
              color="hsl(43, 90%, 50%)"
            />
          )}
        </div>
      )}

      {/* Chart */}
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <defs>
            {/* Gradient fills for glucose area — will be colour-coded at runtime */}
            <linearGradient id="glucoseGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="hsl(142, 60%, 45%)" stopOpacity={0.3} />
              <stop offset="95%" stopColor="hsl(142, 60%, 45%)" stopOpacity={0}   />
            </linearGradient>
          </defs>

          <CartesianGrid
            strokeDasharray="3 3"
            stroke="hsl(215, 28%, 17%)"
            vertical={false}
          />

          <XAxis
            dataKey="time"
            tick={{ fill: "hsl(215, 20%, 55%)", fontSize: 11 }}
            tickLine={false}
            interval="preserveStartEnd"
          />

          {/* Left Y-axis: glucose in mg/dL */}
          <YAxis
            yAxisId="glucose"
            domain={[50, 200]}
            tick={{ fill: "hsl(215, 20%, 55%)", fontSize: 11 }}
            tickLine={false}
            label={{
              value: "mg/dL",
              angle: -90,
              position: "insideLeft",
              fill: "hsl(215, 20%, 55%)",
              fontSize: 11,
            }}
          />

          {/* Right Y-axis: sleep stage (numeric 1–5) */}
          <YAxis
            yAxisId="stage"
            orientation="right"
            domain={[0, 6]}
            ticks={[1, 2, 3, 4, 5]}
            tickFormatter={(v) => {
              const inv: Record<number, string> = {
                1: "N3", 2: "N2", 3: "N1", 4: "REM", 5: "Wake",
              };
              return inv[v] ?? "";
            }}
            tick={{ fill: "hsl(215, 20%, 55%)", fontSize: 10 }}
            tickLine={false}
          />

          <Tooltip content={<CGMTooltip />} />

          <Legend
            wrapperStyle={{ fontSize: 12, color: "hsl(215, 20%, 55%)" }}
            formatter={(value) =>
              value === "glucose" ? "Glucose (mg/dL)" : "Sleep Stage"
            }
          />

          {/* Target range shading: 70–120 mg/dL */}
          <ReferenceLine
            y={GLUCOSE_TARGET_MAX}
            yAxisId="glucose"
            stroke="hsl(142, 60%, 45%)"
            strokeDasharray="4 4"
            strokeOpacity={0.5}
            label={{ value: "120", fill: "hsl(142, 60%, 45%)", fontSize: 10 }}
          />
          <ReferenceLine
            y={GLUCOSE_LOW}
            yAxisId="glucose"
            stroke="hsl(210, 80%, 55%)"
            strokeDasharray="4 4"
            strokeOpacity={0.5}
            label={{ value: "70", fill: "hsl(210, 80%, 55%)", fontSize: 10 }}
          />
          <ReferenceLine
            y={GLUCOSE_HIGH}
            yAxisId="glucose"
            stroke="hsl(0, 75%, 55%)"
            strokeDasharray="4 4"
            strokeOpacity={0.4}
            label={{ value: "140", fill: "hsl(0, 75%, 55%)", fontSize: 10 }}
          />

          {/* Sleep stage bars */}
          {hasSleep && (
            <Bar
              yAxisId="stage"
              dataKey="stageNumeric"
              name="Sleep Stage"
              barSize={6}
              radius={[2, 2, 0, 0]}
              fill="hsl(215, 28%, 30%)"
              opacity={0.7}
            />
          )}

          {/* Glucose area */}
          {hasGlucose && (
            <Area
              yAxisId="glucose"
              type="monotone"
              dataKey="glucose"
              name="glucose"
              stroke="hsl(142, 60%, 45%)"
              strokeWidth={2}
              fill="url(#glucoseGradient)"
              dot={(props: any) => {
                const { cx, cy, payload } = props;
                if (payload.excursionAwakening) {
                  // Bold red diamond for excursion coinciding with awakening
                  return (
                    <polygon
                      key={`dot-${cx}-${cy}`}
                      points={`${cx},${cy - 6} ${cx + 5},${cy} ${cx},${cy + 6} ${cx - 5},${cy}`}
                      fill="hsl(0, 75%, 55%)"
                      stroke="hsl(0, 75%, 70%)"
                      strokeWidth={1}
                    />
                  );
                }
                if (payload.isExcursion) {
                  return (
                    <circle
                      key={`dot-${cx}-${cy}`}
                      cx={cx}
                      cy={cy}
                      r={4}
                      fill={
                        payload.glucose > GLUCOSE_HIGH
                          ? "hsl(0, 75%, 55%)"
                          : "hsl(43, 90%, 50%)"
                      }
                      stroke="none"
                    />
                  );
                }
                return <g key={`dot-${cx}-${cy}`} />;
              }}
              activeDot={{ r: 5, fill: "hsl(142, 60%, 45%)" }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend for glucose bands */}
      <div
        style={{
          display: "flex",
          gap: 16,
          marginTop: 12,
          fontSize: 11,
          color: "hsl(215, 20%, 55%)",
          flexWrap: "wrap",
        }}
      >
        <span>
          <span
            style={{
              display: "inline-block",
              width: 10,
              height: 10,
              borderRadius: 2,
              background: "hsl(142, 60%, 45%)",
              marginRight: 4,
            }}
          />
          In range (70–120)
        </span>
        <span>
          <span
            style={{
              display: "inline-block",
              width: 10,
              height: 10,
              borderRadius: 2,
              background: "hsl(43, 90%, 50%)",
              marginRight: 4,
            }}
          />
          Elevated (120–140)
        </span>
        <span>
          <span
            style={{
              display: "inline-block",
              width: 10,
              height: 10,
              borderRadius: 2,
              background: "hsl(0, 75%, 55%)",
              marginRight: 4,
            }}
          />
          High (&gt;140)
        </span>
        <span>
          <span style={{ marginRight: 4 }}>&#9670;</span>
          Excursion + awakening
        </span>
      </div>
    </div>
  );
}
