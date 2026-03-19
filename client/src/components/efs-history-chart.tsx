/**
 * EFSHistoryChart — EFS history timeline with 30d/60d/90d toggles.
 *
 * Fetches data using getEmotionalFitness and renders a Recharts AreaChart
 * with gradient fill (cyan), showing EFS score over time.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp } from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { getEmotionalFitness } from "@/lib/ml-api";

// ── Props ─────────────────────────────────────────────────────────────────────

interface EFSHistoryChartProps {
  userId: string;
}

// ── Day options ───────────────────────────────────────────────────────────────

const DAY_OPTIONS = [30, 60, 90] as const;
type DayOption = (typeof DAY_OPTIONS)[number];

// ── Component ─────────────────────────────────────────────────────────────────

export function EFSHistoryChart({ userId }: EFSHistoryChartProps) {
  const [days, setDays] = useState<DayOption>(30);

  const { data, isLoading } = useQuery({
    queryKey: ["efs-history", userId, days],
    queryFn: () => getEmotionalFitness(userId, false, days),
    staleTime: 5 * 60_000,
    retry: 1,
  });

  // Build chart data from vitals history — use resilience as proxy for overall,
  // or the EFS score itself if available from the first vital's history dates
  const chartData: { date: string; score: number }[] = [];
  if (data?.vitals) {
    // Find the vital with the most history data points
    const vitals = Object.values(data.vitals);
    let longestHistory: { date: string; score: number }[] = [];
    for (const v of vitals) {
      if (v.history.length > longestHistory.length) {
        longestHistory = v.history;
      }
    }
    // Use the longest vital history as a proxy timeline
    for (const point of longestHistory) {
      chartData.push({ date: point.date, score: point.score });
    }
  }

  return (
    <Card className="bg-card rounded-xl border border-border/50 shadow-sm">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm text-zinc-300 flex items-center gap-2">
            <TrendingUp className="h-4 w-4" /> EFS Trend
          </CardTitle>
          <div className="flex gap-1">
            {DAY_OPTIONS.map((d) => (
              <button
                key={d}
                onClick={() => setDays(d)}
                className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                  days === d
                    ? "bg-cyan-500/20 text-cyan-400 font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/30"
                }`}
              >
                {d}d
              </button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading && (
          <div className="h-[200px] flex items-center justify-center">
            <div className="h-5 w-5 rounded-full border-2 border-primary border-t-transparent animate-spin" />
          </div>
        )}

        {!isLoading && chartData.length < 2 && (
          <div className="h-[200px] flex items-center justify-center">
            <p className="text-xs text-muted-foreground">Not enough data for trend chart</p>
          </div>
        )}

        {!isLoading && chartData.length >= 2 && (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="efsTrendGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#0891b2" stopOpacity={0.3} />
                  <stop offset="50%" stopColor="#0891b2" stopOpacity={0.12} />
                  <stop offset="100%" stopColor="#0891b2" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="date"
                tick={{ fontSize: 10, fill: "#71717a" }}
                tickFormatter={(val: string) => {
                  const d = new Date(val);
                  return `${d.getMonth() + 1}/${d.getDate()}`;
                }}
              />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#71717a" }} />
              <Tooltip
                contentStyle={{
                  background: "#18181b",
                  border: "1px solid #3f3f46",
                  borderRadius: 8,
                  fontSize: 11,
                }}
                labelStyle={{ color: "#a1a1aa" }}
                itemStyle={{ color: "#0891b2" }}
                formatter={(v: number) => [v, "EFS"]}
              />
              <Area
                type="monotone"
                dataKey="score"
                stroke="#0891b2"
                fill="url(#efsTrendGrad)"
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 4, fill: "#0891b2" }}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
