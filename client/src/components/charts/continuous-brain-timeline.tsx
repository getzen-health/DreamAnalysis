/**
 * ContinuousBrainTimeline — Apple Health-style timeline backed by TimescaleDB.
 *
 * Features
 * --------
 * - Period tabs: Today / Week / Month / 3M / Year
 * - ← Earlier / Later → pan buttons (shift window by one period)
 * - Metric dropdown: Focus / Stress / Relaxation / Flow / Valence
 * - Recharts LineChart with connectNulls=false (gaps show device-off periods)
 * - TanStack Query: 30s stale, 60s refetch
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

/* ── Config ─────────────────────────────────────────────────── */

const PERIODS = [
  { label: "Today",  days: 1,   bucket: "1m" },
  { label: "Week",   days: 7,   bucket: "1h" },
  { label: "Month",  days: 30,  bucket: "1d" },
  { label: "3M",     days: 90,  bucket: "1d" },
  { label: "Year",   days: 365, bucket: "1d" },
] as const;

type PeriodLabel = typeof PERIODS[number]["label"];

const METRICS = [
  { value: "focus_index",     label: "Focus",      color: "hsl(200, 70%, 55%)" },
  { value: "stress_index",    label: "Stress",     color: "hsl(38,  85%, 58%)" },
  { value: "relaxation_idx",  label: "Relaxation", color: "hsl(152, 60%, 48%)" },
  { value: "flow_score",      label: "Flow",       color: "hsl(262, 45%, 65%)" },
  { value: "valence",         label: "Valence",    color: "hsl(330, 60%, 60%)" },
] as const;

type MetricValue = typeof METRICS[number]["value"];

/* ── Types ──────────────────────────────────────────────────── */
interface TimelineBucket {
  time: string;
  value: number | null;
  dominant_emotion?: string | null;
}

interface TimelineResponse {
  buckets: TimelineBucket[];
}

/* ── Helpers ─────────────────────────────────────────────────── */
function formatBucketTime(iso: string, days: number): string {
  const d = new Date(iso);
  if (days <= 1)  return d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
  if (days <= 7)  return d.toLocaleDateString("en-US", { weekday: "short", hour: "numeric" });
  if (days <= 31) return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  return d.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
}

async function fetchTimeline(params: {
  userId: string;
  fromTs: number;
  toTs: number;
  metric: string;
  bucket: string;
}): Promise<TimelineResponse> {
  const url = new URL("/api/ml/brain/timeline", window.location.origin);
  url.searchParams.set("user_id", params.userId);
  url.searchParams.set("from_ts", String(params.fromTs));
  url.searchParams.set("to_ts",   String(params.toTs));
  url.searchParams.set("metric",  params.metric);
  url.searchParams.set("bucket",  params.bucket);
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/* ── Component ──────────────────────────────────────────────── */
interface Props {
  userId?: string;
  defaultMetric?: MetricValue;
  title?: string;
}

export function ContinuousBrainTimeline({
  userId = "default",
  defaultMetric = "focus_index",
  title = "Brain Trends",
}: Props) {
  const [periodLabel, setPeriodLabel] = useState<PeriodLabel>("Week");
  const [metric, setMetric] = useState<MetricValue>(defaultMetric);
  const [panOffset, setPanOffset] = useState(0); // in number-of-periods to the past

  const period = PERIODS.find((p) => p.label === periodLabel) ?? PERIODS[1];
  const windowSec = period.days * 86400;
  const nowSec   = Math.floor(Date.now() / 1000);
  const toTs     = nowSec  - panOffset * windowSec;
  const fromTs   = toTs    - windowSec;

  const metricConfig = METRICS.find((m) => m.value === metric) ?? METRICS[0];

  const { data, isLoading, isError } = useQuery<TimelineResponse>({
    queryKey: ["brain-timeline", userId, metric, period.bucket, fromTs, toTs],
    queryFn: () =>
      fetchTimeline({ userId, fromTs, toTs, metric, bucket: period.bucket }),
    staleTime: 30_000,
    refetchInterval: 60_000,
    retry: 1,
  });

  const chartData = (data?.buckets ?? []).map((b) => ({
    time:  formatBucketTime(b.time, period.days),
    value: b.value !== null ? Math.round((b.value ?? 0) * 100) : null,
    emotion: b.dominant_emotion,
  }));

  const hasData = chartData.some((d) => d.value !== null);

  function handlePeriodChange(label: PeriodLabel) {
    setPeriodLabel(label);
    setPanOffset(0); // reset pan on period change
  }

  return (
    <Card className="glass-card p-5 hover-glow">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
        <h3 className="text-sm font-semibold">{title}</h3>

        {/* Metric picker */}
        <Select value={metric} onValueChange={(v) => setMetric(v as MetricValue)}>
          <SelectTrigger className="h-7 w-36 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {METRICS.map((m) => (
              <SelectItem key={m.value} value={m.value} className="text-xs">
                {m.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Period tabs */}
      <div className="flex items-center gap-1 mb-4">
        {PERIODS.map((p) => (
          <button
            key={p.label}
            onClick={() => handlePeriodChange(p.label)}
            className={`px-2 py-0.5 rounded text-xs font-medium transition-colors ${
              periodLabel === p.label
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-muted"
            }`}
          >
            {p.label}
          </button>
        ))}

        {/* Pan buttons */}
        <div className="ml-auto flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={() => setPanOffset((o) => o + 1)}
            title="Earlier"
          >
            <ChevronLeft className="h-3 w-3" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            disabled={panOffset === 0}
            onClick={() => setPanOffset((o) => Math.max(0, o - 1))}
            title="Later"
          >
            <ChevronRight className="h-3 w-3" />
          </Button>
        </div>
      </div>

      {/* Chart */}
      {isLoading ? (
        <div className="h-[160px] flex items-center justify-center text-xs text-muted-foreground">
          Loading…
        </div>
      ) : isError ? (
        <div className="h-[160px] flex items-center justify-center text-xs text-destructive">
          Failed to load data — is the ML backend running?
        </div>
      ) : !hasData ? (
        <div className="h-[160px] flex items-center justify-center text-xs text-muted-foreground border border-dashed border-border/30 rounded-lg">
          No data for this period — connect Muse 2 to start recording
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={chartData}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="hsl(220, 18%, 15%)"
              opacity={0.4}
            />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 9, fill: "hsl(220, 12%, 42%)" }}
              axisLine={false}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fontSize: 9, fill: "hsl(220, 12%, 42%)" }}
              axisLine={false}
              tickLine={false}
              width={24}
            />
            <Tooltip
              cursor={{ stroke: "hsl(220, 12%, 55%)", strokeWidth: 1, strokeDasharray: "4 4" }}
              contentStyle={{
                background: "hsl(220, 22%, 9%)",
                border: "1px solid hsl(220, 18%, 20%)",
                borderRadius: 10,
                fontSize: 11,
              }}
              formatter={(val: number, _name: string, props: { payload?: { emotion?: string } }) => [
                `${val}%`,
                `${metricConfig.label}${props?.payload?.emotion ? ` · ${props.payload.emotion}` : ""}`,
              ]}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke={metricConfig.color}
              strokeWidth={2}
              dot={false}
              connectNulls={false}   // gaps show where device was off
              activeDot={{ r: 4, fill: metricConfig.color }}
              name={metricConfig.label}
            />
          </LineChart>
        </ResponsiveContainer>
      )}

      {/* Legend */}
      <div className="flex items-center gap-1.5 mt-2">
        <div
          className="w-3 h-0.5 rounded"
          style={{ background: metricConfig.color }}
        />
        <span className="text-[10px] text-muted-foreground">{metricConfig.label}</span>
        {panOffset > 0 && (
          <span className="ml-auto text-[10px] text-muted-foreground">
            Viewing {panOffset} {period.label.toLowerCase()}
            {panOffset > 1 ? "s" : ""} ago
          </span>
        )}
      </div>
    </Card>
  );
}
