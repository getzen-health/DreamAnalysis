import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { TrendingUp } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import {
  aggregateArcTrend,
  computeArcSummary,
  topArcPatterns,
  arcLabel,
  ARC_LABEL_COLOR,
  ARC_LABEL_BG,
  type DreamForArc,
} from "@/lib/emotional-arc-tracker";

interface Props {
  userId: string;
}

// Recharts tooltip
function ArcTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ value: number; payload: { dreamCount: number } }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const valence = payload[0].value;
  const label_ = arcLabel(valence);
  return (
    <div className="bg-background/95 border border-white/10 rounded px-2.5 py-1.5 text-xs shadow-lg">
      <p className="text-muted-foreground">{label}</p>
      <p className={ARC_LABEL_COLOR[label_]}>
        {label_} ({valence >= 0 ? "+" : ""}{valence.toFixed(2)})
      </p>
      <p className="text-muted-foreground/60">{payload[0].payload.dreamCount} dream{payload[0].payload.dreamCount !== 1 ? "s" : ""}</p>
    </div>
  );
}

export function EmotionalArcTrendCard({ userId }: Props) {
  const { data: rawData, isLoading } = useQuery<DreamForArc[]>({
    queryKey: ["dream-analysis", userId],
    queryFn: async () => {
      const res = await apiRequest("GET", `/api/dream-analysis/${userId}`); return res.json();
    },
    staleTime: 2 * 60 * 1000,
  });

  const raw: DreamForArc[] = Array.isArray(rawData) ? rawData : [];

  const trend  = aggregateArcTrend(raw);
  const summary = computeArcSummary(raw);
  const patterns = topArcPatterns(raw, 4);

  const hasData = trend.length > 0;
  const overallLabel = summary.meanValence !== null ? arcLabel(summary.meanValence) : null;

  // Format date for x-axis tick
  function shortDate(iso: string) {
    const [, m, d] = iso.split("-");
    const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    return `${months[parseInt(m, 10) - 1]} ${parseInt(d, 10)}`;
  }

  return (
    <Card className="glass-card p-5 hover-glow border-secondary/20">
      <CardHeader className="p-0 pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-secondary" />
          Emotional Arc Trend
          {overallLabel && (
            <span className={`ml-auto text-[10px] px-2 py-0.5 rounded-full ${ARC_LABEL_BG[overallLabel]} ${ARC_LABEL_COLOR[overallLabel]}`}>
              overall: {overallLabel}
            </span>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="p-0 space-y-3">
        {isLoading ? (
          <p className="text-xs text-muted-foreground py-4 text-center">Loading arc trend…</p>
        ) : !hasData ? (
          <p className="text-xs text-muted-foreground py-4 text-center">
            No emotional arc data yet — arcs are captured during dream analysis.
          </p>
        ) : (
          <>
            {/* Sparkline */}
            <ResponsiveContainer width="100%" height={110}>
              <LineChart data={trend} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
                <XAxis
                  dataKey="date"
                  tickFormatter={shortDate}
                  tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                  axisLine={false}
                  tickLine={false}
                  interval="preserveStartEnd"
                />
                <YAxis
                  domain={[-1, 1]}
                  ticks={[-1, 0, 1]}
                  tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip content={<ArcTooltip />} />
                <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeOpacity={0.2} strokeDasharray="3 3" />
                <Line
                  type="monotone"
                  dataKey="avgValence"
                  stroke="hsl(var(--secondary))"
                  strokeWidth={2}
                  dot={{ r: 3, fill: "hsl(var(--secondary))", strokeWidth: 0 }}
                  activeDot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>

            {/* Summary stats */}
            {summary.arcCount > 0 && (
              <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                  <p className="text-[10px] text-muted-foreground/60">Uplifting</p>
                  <p className="text-xs text-emerald-400">
                    {Math.round(summary.positiveRate * 100)}%
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground/60">Mixed</p>
                  <p className="text-xs text-amber-400">
                    {Math.round((1 - summary.positiveRate - summary.negativeRate) * 100)}%
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground/60">Distressing</p>
                  <p className="text-xs text-red-400">
                    {Math.round(summary.negativeRate * 100)}%
                  </p>
                </div>
              </div>
            )}

            {/* Top arc patterns */}
            {patterns.length > 0 && (
              <div>
                <p className="text-[10px] text-muted-foreground/60 mb-1.5">Recurring arcs</p>
                <div className="flex flex-wrap gap-1.5">
                  {patterns.map(({ pattern, count }) => {
                    const v = 0; // neutral display for patterns — no valence pre-computed
                    return (
                      <span
                        key={pattern}
                        className="text-[10px] px-2 py-0.5 rounded-full bg-white/8 border border-white/10 text-muted-foreground"
                      >
                        {pattern}
                        {count > 1 && (
                          <span className="ml-1 text-muted-foreground/50">×{count}</span>
                        )}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
