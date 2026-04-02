import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Moon } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { getIntentionHistory } from "@/lib/sleep-intention";
import {
  computeLucidityPrediction,
  lucidityTrend,
  BAND_LABEL,
  BAND_COLOR,
  BAND_BG,
  type DreamForLucidity,
  type IntentionForLucidity,
} from "@/lib/lucidity-predictor";
import {
  LineChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  ReferenceLine,
} from "recharts";

interface Props {
  userId: string;
}

export function LucidityPredictorCard({ userId }: Props) {
  const { data: rawData, isLoading } = useQuery<DreamForLucidity[]>({
    queryKey: ["dream-analysis", userId],
    queryFn: async () => {
      const res = await apiRequest("GET", `/api/dream-analysis/${userId}`); return res.json();
    },
    staleTime: 2 * 60 * 1000,
  });

  const dreams: DreamForLucidity[] = Array.isArray(rawData) ? rawData : [];

  // Intention history is localStorage — safe to read synchronously
  const rawIntentions = getIntentionHistory();
  const intentions: IntentionForLucidity[] = rawIntentions.map((i) => ({
    date: i.date,
    text: i.text,
  }));

  const prediction = computeLucidityPrediction(dreams, intentions);
  const trend = lucidityTrend(dreams, 14);

  const bandColor = BAND_COLOR[prediction.band];
  const bandBg    = BAND_BG[prediction.band];

  return (
    <Card className="glass-card p-5 hover-glow border-secondary/20">
      <CardHeader className="p-0 pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Moon className="h-4 w-4 text-secondary" />
          Tonight's Lucid Dream Potential
        </CardTitle>
      </CardHeader>

      <CardContent className="p-0 space-y-4">
        {isLoading ? (
          <p className="text-xs text-muted-foreground py-4 text-center">Computing…</p>
        ) : (
          <>
            {/* Score + band badge */}
            <div className="flex items-center gap-3">
              <span className={`text-4xl font-bold tabular-nums ${bandColor}`}>
                {prediction.likelihood}
                <span className="text-lg font-normal">%</span>
              </span>
              <div>
                <span className={`text-[10px] px-2 py-0.5 rounded-full border font-medium ${bandBg} ${bandColor}`}>
                  {BAND_LABEL[prediction.band]}
                </span>
                <p className="text-[11px] text-muted-foreground/70 mt-1 max-w-52 leading-snug">
                  {prediction.summary}
                </p>
              </div>
            </div>

            {/* 14-day sparkline (only if enough data) */}
            {trend.length >= 3 && (
              <div className="h-14">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trend} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
                    <ReferenceLine y={50} stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
                    <Tooltip
                      contentStyle={{ background: "#1a1a2e", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, fontSize: 10 }}
                      formatter={(v: number) => [`${v}`, "Lucidity"]}
                      labelFormatter={(l) => l}
                    />
                    <Line
                      type="monotone"
                      dataKey="avgScore"
                      stroke="hsl(var(--secondary))"
                      strokeWidth={1.5}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Factor bars */}
            <div className="space-y-2">
              {prediction.factors.map((f) => (
                <div key={f.label} className="space-y-0.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-muted-foreground/70">{f.label}</span>
                    <span className="text-[10px] text-muted-foreground/50">{f.value}%</span>
                  </div>
                  <div className="h-1 rounded-full bg-white/8 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-secondary/50 transition-all"
                      style={{ width: `${f.value}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Tonight's recommendation */}
            <div className="rounded-lg p-2.5 bg-secondary/5 border border-secondary/15">
              <p className="text-[10px] text-muted-foreground/50 mb-0.5 uppercase tracking-wide">Tonight</p>
              <p className="text-[11px] text-muted-foreground/80 leading-relaxed">
                {prediction.recommendation}
              </p>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
