import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CalendarDays, Flame } from "lucide-react";
import { resolveUrl } from "@/lib/queryClient";
import {
  buildRecallCalendar,
  computeRecallStreak,
  computeRecallRate,
  recallWeeklyTrend,
  recallTrendDirection,
  shortDate,
  recallCellClass,
} from "@/lib/dream-recall";

interface Props {
  userId: string;
}

const WEEKDAY_LABELS = ["M", "T", "W", "T", "F", "S", "S"];
const TREND_COLOR = {
  improving:    "text-emerald-400",
  stable:       "text-amber-400",
  declining:    "text-red-400",
  insufficient: "text-muted-foreground/50",
};
const TREND_LABEL = {
  improving:    "improving",
  stable:       "stable",
  declining:    "declining",
  insufficient: "—",
};

export function DreamRecallHeatmap({ userId }: Props) {
  const { data: rawData, isLoading } = useQuery<{ timestamp: string }[]>({
    queryKey: ["dream-analysis", userId],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/dream-analysis/${userId}`));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    },
    staleTime: 2 * 60 * 1000,
  });

  const raw: { timestamp: string }[] = Array.isArray(rawData) ? rawData : [];

  // 28-day calendar (4 weeks × 7 days)
  const DAYS = 28;
  const calendar = buildRecallCalendar(raw, DAYS);
  const streak = computeRecallStreak(raw);
  const rate7  = computeRecallRate(raw, 7);
  const rate28 = computeRecallRate(raw, 28);
  const trend  = recallTrendDirection(raw);
  const weekly = recallWeeklyTrend(raw, 4);

  // Arrange calendar into 4-week rows (each row = 7 days)
  const weeks: typeof calendar[] = [];
  for (let i = 0; i < calendar.length; i += 7) {
    weeks.push(calendar.slice(i, i + 7));
  }

  return (
    <Card className="glass-card p-5 hover-glow border-secondary/20">
      <CardHeader className="p-0 pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <CalendarDays className="h-4 w-4 text-secondary" />
          Dream Recall
          {streak > 0 && (
            <span className="flex items-center gap-1 ml-auto text-[10px] text-amber-400">
              <Flame className="h-3 w-3" />
              {streak}-day streak
            </span>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="p-0 space-y-3">
        {isLoading ? (
          <p className="text-xs text-muted-foreground py-4 text-center">Loading recall data…</p>
        ) : (
          <>
            {/* Heatmap grid */}
            <div className="space-y-1">
              {/* Weekday labels */}
              <div className="grid grid-cols-7 gap-1">
                {WEEKDAY_LABELS.map((l, i) => (
                  <div key={i} className="text-center text-[9px] text-muted-foreground/40">{l}</div>
                ))}
              </div>

              {/* Week rows */}
              {weeks.map((week, wi) => (
                <div key={wi} className="grid grid-cols-7 gap-1">
                  {week.map((day) => (
                    <div
                      key={day.date}
                      title={`${shortDate(day.date)}: ${day.count === 0 ? "no dream" : day.count === 1 ? "1 dream" : `${day.count} dreams`}`}
                      className={`h-5 rounded-sm transition-colors ${recallCellClass(day.count, day.isToday)}`}
                    />
                  ))}
                </div>
              ))}

              {/* Month labels below grid */}
              {weeks.length > 0 && (
                <div className="flex justify-between px-0.5 pt-0.5">
                  <span className="text-[9px] text-muted-foreground/40">
                    {shortDate(weeks[0][0].date)}
                  </span>
                  <span className="text-[9px] text-muted-foreground/40">
                    {shortDate(weeks[weeks.length - 1][weeks[weeks.length - 1].length - 1].date)}
                  </span>
                </div>
              )}
            </div>

            {/* Stats row */}
            <div className="grid grid-cols-3 gap-2 text-center pt-1 border-t border-white/5">
              <div>
                <p className="text-xs text-secondary">{Math.round(rate7 * 100)}%</p>
                <p className="text-[10px] text-muted-foreground/50">7-day rate</p>
              </div>
              <div>
                <p className="text-xs text-secondary">{Math.round(rate28 * 100)}%</p>
                <p className="text-[10px] text-muted-foreground/50">28-day rate</p>
              </div>
              <div>
                <p className={`text-xs ${TREND_COLOR[trend]}`}>{TREND_LABEL[trend]}</p>
                <p className="text-[10px] text-muted-foreground/50">trend</p>
              </div>
            </div>

            {/* Weekly bars */}
            {weekly.length > 0 && (
              <div>
                <p className="text-[10px] text-muted-foreground/50 mb-1.5">Weekly recall</p>
                <div className="flex items-end gap-1.5 h-10">
                  {weekly.map((pt) => {
                    const heightPct = Math.max(4, Math.round(pt.rate * 100));
                    return (
                      <div
                        key={pt.weekStart}
                        className="flex-1 flex flex-col items-center gap-0.5"
                        title={`Week of ${shortDate(pt.weekStart)}: ${pt.dreamDays}/${pt.totalDays} days`}
                      >
                        <div
                          className="w-full rounded-sm bg-secondary/50"
                          style={{ height: `${heightPct}%` }}
                        />
                        <span className="text-[9px] text-muted-foreground/40">
                          {shortDate(pt.weekStart).split(" ")[1]}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Legend */}
            <div className="flex items-center gap-2 text-[9px] text-muted-foreground/40 justify-end">
              <span>fewer</span>
              {[0, 1, 2, 3].map((n) => (
                <div key={n} className={`h-3 w-3 rounded-sm ${recallCellClass(n, false)}`} />
              ))}
              <span>more</span>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
