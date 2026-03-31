import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Sparkles } from "lucide-react";
import {
  qualityLabel,
  qualityColorClass,
  qualityBgClass,
  type DreamQualityTrend,
} from "@/lib/dream-quality-score";

interface Props {
  data: DreamQualityTrend;
}

export function DreamQualityCard({ data }: Props) {
  const trend = data.trend ?? [];
  if (data.totalDreams === 0 && trend.length === 0) return null;

  const score = data.current;
  const hasSparkline = trend.length >= 2;

  // Format YYYY-MM-DD to short label (Mon, Tue …)
  function shortLabel(dateStr: string): string {
    const d = new Date(dateStr + "T12:00:00");
    return d.toLocaleDateString("en-US", { weekday: "short" });
  }

  const chartData = trend.map((p) => ({ day: shortLabel(p.date), score: p.score }));

  return (
    <Card className="glass-card p-5 hover-glow border-secondary/20">
      <CardHeader className="p-0 pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-secondary" />
          Dream Quality Score
          {data.totalDreams > 0 && (
            <span className="ml-auto text-[10px] text-muted-foreground">
              {data.totalDreams} dream{data.totalDreams !== 1 ? "s" : ""}
            </span>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="p-0 space-y-4">
        {/* Current score badge */}
        {score != null ? (
          <div className={`rounded-xl px-4 py-3 flex items-center gap-3 ${qualityBgClass(score)}`}>
            <span className={`text-3xl font-bold tabular-nums ${qualityColorClass(score)}`}>
              {score}
            </span>
            <div>
              <p className={`text-sm font-semibold ${qualityColorClass(score)}`}>
                {qualityLabel(score)}
              </p>
              {data.avgScore != null && (
                <p className="text-[10px] text-muted-foreground">
                  {trend.length}-day avg: {data.avgScore}
                </p>
              )}
            </div>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            Record a dream with sleep quality to see your score.
          </p>
        )}

        {/* Sparkline */}
        {hasSparkline && (
          <div className="h-24">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="dqGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(var(--secondary))" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="hsl(var(--secondary))" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="day"
                  tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis domain={[0, 100]} hide />
                <Tooltip
                  contentStyle={{
                    background: "hsl(var(--background))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "6px",
                    fontSize: "11px",
                    padding: "4px 8px",
                  }}
                  formatter={(v: number) => [`${v} — ${qualityLabel(v)}`, "Score"]}
                />
                <Area
                  type="monotone"
                  dataKey="score"
                  stroke="hsl(var(--secondary))"
                  strokeWidth={1.5}
                  fill="url(#dqGrad)"
                  dot={false}
                  activeDot={{ r: 3, strokeWidth: 0 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Score breakdown legend */}
        <div className="grid grid-cols-2 gap-1.5 text-[10px] text-muted-foreground">
          <span>Sleep quality ±20</span>
          <span>Lucidity bonus +15</span>
          <span>Nightmare penalty −25</span>
          <span>Dream recall +10</span>
        </div>
      </CardContent>
    </Card>
  );
}
