import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Clock,
  Download,
  Trash2,
  FileText,
  TrendingUp,
  Brain,
  Eye,
  Activity,
  Sparkles,
  Moon,
} from "lucide-react";
import {
  LineChart,
  Line,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";
import { listSessions, deleteSession, exportSession, type SessionSummary } from "@/lib/ml-api";

/* ── Period tabs ─────────────────────────────────────────── */
const PERIODS = [
  { label: "Today", days: 1 },
  { label: "Week", days: 7 },
  { label: "Month", days: 30 },
  { label: "3 Months", days: 90 },
  { label: "Year", days: 365 },
];

/* ── Helpers ─────────────────────────────────────────────── */
function filterByPeriod(sessions: SessionSummary[], days: number): SessionSummary[] {
  const cutoff = Date.now() / 1000 - days * 86400;
  return sessions.filter((s) => (s.start_time ?? 0) >= cutoff);
}

function formatDuration(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function formatHour(h: number): string {
  if (h === 0) return "12 AM";
  if (h < 12) return `${h} AM`;
  if (h === 12) return "12 PM";
  return `${h - 12} PM`;
}

function getTimeOfDay(h: number): string {
  if (h >= 5 && h < 12) return "morning";
  if (h >= 12 && h < 17) return "afternoon";
  if (h >= 17 && h < 21) return "evening";
  return "night";
}

function findPeakHour(
  sessions: SessionSummary[],
  metric: "avg_focus" | "avg_flow" | "avg_relaxation"
): { hour: number; value: number } | null {
  const hourMap: Record<number, { sum: number; count: number }> = {};
  for (const s of sessions) {
    const val = s.summary?.[metric];
    if (val == null || isNaN(val)) continue;
    const hour = new Date((s.start_time ?? 0) * 1000).getHours();
    if (!hourMap[hour]) hourMap[hour] = { sum: 0, count: 0 };
    hourMap[hour].sum += val;
    hourMap[hour].count += 1;
  }
  const hours = Object.entries(hourMap);
  if (!hours.length) return null;
  const best = hours.reduce((a, b) =>
    a[1].sum / a[1].count > b[1].sum / b[1].count ? a : b
  );
  return { hour: Number(best[0]), value: best[1].sum / best[1].count };
}

function getDominantEmotion(sessions: SessionSummary[]): string {
  const counts: Record<string, number> = {};
  for (const s of sessions) {
    const e = s.summary?.dominant_emotion;
    if (e) counts[e] = (counts[e] ?? 0) + 1;
  }
  if (!Object.keys(counts).length) return "";
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
}

function buildTrendData(sessions: SessionSummary[]) {
  return sessions
    .filter((s) => s.summary?.avg_focus != null)
    .slice(-30)
    .map((s) => ({
      date: new Date((s.start_time ?? 0) * 1000).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      }),
      focus: Math.round((s.summary.avg_focus ?? 0) * 100),
      stress: Math.round((s.summary.avg_stress ?? 0) * 100),
      flow: Math.round((s.summary.avg_flow ?? 0) * 100),
      creativity: Math.round((s.summary.avg_creativity ?? 0) * 100),
    }));
}

function groupByDay(sessions: SessionSummary[]): Record<string, SessionSummary[]> {
  const groups: Record<string, SessionSummary[]> = {};
  for (const session of sessions) {
    const date = new Date((session.start_time ?? 0) * 1000);
    const key = date.toLocaleDateString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
    });
    if (!groups[key]) groups[key] = [];
    groups[key].push(session);
  }
  return groups;
}

/* ── Component ───────────────────────────────────────────── */
export default function SessionHistory() {
  const [periodDays, setPeriodDays] = useState(7);
  const [selectedSession, setSelectedSession] = useState<string | null>(null);

  const { data: allSessions = [], isLoading, refetch } = useQuery<SessionSummary[]>({
    queryKey: ["sessions"],
    queryFn: () => listSessions(),
    retry: false,
    refetchInterval: 30_000,
  });

  const periodSessions = filterByPeriod(allSessions, periodDays);
  const selectedPeriodLabel =
    PERIODS.find((p) => p.days === periodDays)?.label?.toLowerCase() ?? "period";

  /* Averages */
  const count = Math.max(periodSessions.length, 1);
  const avgFocus = periodSessions.reduce((s, x) => s + (x.summary?.avg_focus ?? 0), 0) / count;
  const avgStress = periodSessions.reduce((s, x) => s + (x.summary?.avg_stress ?? 0), 0) / count;
  const avgFlow = periodSessions.reduce((s, x) => s + (x.summary?.avg_flow ?? 0), 0) / count;
  const avgCreativity = periodSessions.reduce((s, x) => s + (x.summary?.avg_creativity ?? 0), 0) / count;
  const totalMinutes =
    periodSessions.reduce((s, x) => s + (x.summary?.duration_sec ?? 0), 0) / 60;

  const dominantEmotion = getDominantEmotion(periodSessions);
  const peakFocus = findPeakHour(periodSessions, "avg_focus");
  const peakFlow = findPeakHour(periodSessions, "avg_flow");
  const peakRelax = findPeakHour(periodSessions, "avg_relaxation");
  const trendData = buildTrendData(periodSessions);

  /* Day groups */
  const dayGroups = groupByDay(periodSessions);
  const sortedDays = Object.keys(dayGroups).sort((a, b) => {
    const aTime = dayGroups[a][0].start_time ?? 0;
    const bTime = dayGroups[b][0].start_time ?? 0;
    return bTime - aTime;
  });

  /* Summary text */
  function buildSummaryText(): string {
    if (!periodSessions.length) return "No sessions recorded yet.";
    const stressPct = Math.round(avgStress * 100);
    const focusPct = Math.round(avgFocus * 100);
    const flowPct = Math.round(avgFlow * 100);
    const totalMin = Math.round(totalMinutes);

    let text = `You had ${periodSessions.length} session${periodSessions.length > 1 ? "s" : ""} (${totalMin}m total). `;
    if (stressPct < 30 && focusPct > 60) {
      text += "A productive period — low stress, strong focus.";
    } else if (stressPct > 60) {
      text += "Stress was elevated. Consider more breaks or breathing exercises.";
    } else if (flowPct > 60) {
      text += "You spent solid time in flow state — great for deep work.";
    } else if (focusPct > 50) {
      text += "Focus was solid throughout this period.";
    } else {
      text += "Your neural patterns were in a balanced, transitional state.";
    }
    return text;
  }

  const handleDelete = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deleteSession(sessionId);
      refetch();
      if (selectedSession === sessionId) setSelectedSession(null);
    } catch {
      /* ignore */
    }
  };

  const handleExport = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const data = await exportSession(sessionId);
      const blob = new Blob([data], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `session_${sessionId}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      /* ignore */
    }
  };

  return (
    <main className="p-4 md:p-6 space-y-6 max-w-5xl">
      {/* Header + Period Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Clock className="h-5 w-5 text-primary" />
          <h2 className="text-lg font-semibold">Session History</h2>
        </div>
        <div className="flex gap-1 flex-wrap">
          {PERIODS.map((p) => (
            <button
              key={p.days}
              onClick={() => setPeriodDays(p.days)}
              className={`px-3 py-1 text-xs rounded-full transition-colors ${
                periodDays === p.days
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Loading */}
      {isLoading && (
        <p className="text-sm text-muted-foreground">Loading sessions...</p>
      )}

      {/* Empty state — no sessions at all */}
      {!isLoading && allSessions.length === 0 && (
        <Card className="glass-card p-10 text-center">
          <Brain className="h-10 w-10 text-muted-foreground/30 mx-auto mb-3" />
          <p className="text-sm font-medium text-foreground/60">No sessions recorded yet</p>
          <p className="text-xs text-muted-foreground mt-1 max-w-xs mx-auto">
            Connect your Muse 2 and start streaming — sessions are recorded automatically.
          </p>
        </Card>
      )}

      {/* Period Analysis */}
      {periodSessions.length > 0 && (
        <>
          {/* "How your [period] went" summary card */}
          <Card className="glass-card p-5 hover-glow">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-xl bg-primary/10 flex items-center justify-center shrink-0">
                <Sparkles className="h-4 w-4 text-primary" />
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium text-foreground mb-1">
                  How your {selectedPeriodLabel} went
                </p>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {buildSummaryText()}
                </p>
                <div className="flex flex-wrap gap-2 mt-3">
                  <span className="px-2.5 py-1 rounded-full text-[10px] font-medium bg-primary/10 text-primary">
                    Focus {Math.round(avgFocus * 100)}%
                  </span>
                  <span className="px-2.5 py-1 rounded-full text-[10px] font-medium bg-warning/10 text-warning">
                    Stress {Math.round(avgStress * 100)}%
                  </span>
                  <span className="px-2.5 py-1 rounded-full text-[10px] font-medium bg-success/10 text-success">
                    Flow {Math.round(avgFlow * 100)}%
                  </span>
                  <span className="px-2.5 py-1 rounded-full text-[10px] font-medium bg-secondary/10 text-secondary">
                    Creativity {Math.round(avgCreativity * 100)}%
                  </span>
                  {dominantEmotion && (
                    <span className="px-2.5 py-1 rounded-full text-[10px] font-medium bg-muted text-muted-foreground capitalize">
                      Mostly {dominantEmotion}
                    </span>
                  )}
                </div>
              </div>
            </div>
          </Card>

          {/* Peak Time cards */}
          {(peakFocus || peakFlow || peakRelax) && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {peakFocus && (
                <Card className="glass-card p-4 hover-glow">
                  <div className="flex items-center gap-2 mb-2">
                    <Eye className="h-4 w-4 text-primary" />
                    <span className="text-xs text-muted-foreground">Peak Focus</span>
                  </div>
                  <p className="text-lg font-semibold">{formatHour(peakFocus.hour)}</p>
                  <p className="text-xs text-muted-foreground capitalize mt-0.5">
                    {getTimeOfDay(peakFocus.hour)} · {Math.round(peakFocus.value * 100)}% avg
                  </p>
                </Card>
              )}
              {peakFlow && (
                <Card className="glass-card p-4 hover-glow">
                  <div className="flex items-center gap-2 mb-2">
                    <Activity className="h-4 w-4 text-success" />
                    <span className="text-xs text-muted-foreground">Peak Flow</span>
                  </div>
                  <p className="text-lg font-semibold">{formatHour(peakFlow.hour)}</p>
                  <p className="text-xs text-muted-foreground capitalize mt-0.5">
                    {getTimeOfDay(peakFlow.hour)} · {Math.round(peakFlow.value * 100)}% avg
                  </p>
                </Card>
              )}
              {peakRelax && (
                <Card className="glass-card p-4 hover-glow">
                  <div className="flex items-center gap-2 mb-2">
                    <Moon className="h-4 w-4 text-secondary" />
                    <span className="text-xs text-muted-foreground">Peak Relaxation</span>
                  </div>
                  <p className="text-lg font-semibold">{formatHour(peakRelax.hour)}</p>
                  <p className="text-xs text-muted-foreground capitalize mt-0.5">
                    {getTimeOfDay(peakRelax.hour)} · {Math.round(peakRelax.value * 100)}% avg
                  </p>
                </Card>
              )}
            </div>
          )}

          {/* Trend Chart */}
          {trendData.length >= 2 && (
            <Card className="glass-card p-5 hover-glow">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="h-4 w-4 text-primary" />
                <h3 className="text-sm font-medium">Metrics Over Time</h3>
              </div>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trendData}>
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 10 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis hide domain={[0, 100]} />
                    <Tooltip
                      contentStyle={{
                        background: "var(--card)",
                        border: "1px solid var(--border)",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                      formatter={(v: number) => `${v}%`}
                    />
                    <Line
                      type="monotone"
                      dataKey="focus"
                      name="Focus"
                      stroke="hsl(152, 60%, 48%)"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="stress"
                      name="Stress"
                      stroke="hsl(38, 85%, 58%)"
                      strokeWidth={1.5}
                      dot={false}
                      strokeDasharray="4 4"
                    />
                    <Line
                      type="monotone"
                      dataKey="flow"
                      name="Flow"
                      stroke="hsl(200, 70%, 55%)"
                      strokeWidth={1.5}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="creativity"
                      name="Creativity"
                      stroke="hsl(262, 45%, 65%)"
                      strokeWidth={1.5}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="flex gap-4 mt-2 flex-wrap">
                {[
                  { label: "Focus", color: "hsl(152, 60%, 48%)" },
                  { label: "Stress", color: "hsl(38, 85%, 58%)", dashed: true },
                  { label: "Flow", color: "hsl(200, 70%, 55%)" },
                  { label: "Creativity", color: "hsl(262, 45%, 65%)" },
                ].map((l) => (
                  <div key={l.label} className="flex items-center gap-1.5">
                    <svg width="20" height="8">
                      <line
                        x1="0"
                        y1="4"
                        x2="20"
                        y2="4"
                        stroke={l.color}
                        strokeWidth="2"
                        strokeDasharray={l.dashed ? "4 3" : "0"}
                      />
                    </svg>
                    <span className="text-[10px] text-muted-foreground">{l.label}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}

      {/* Session list */}
      {allSessions.length > 0 && (
        <div className="space-y-5">
          <p className="text-xs font-medium text-muted-foreground">
            {periodSessions.length === 0
              ? "No sessions in this period"
              : `${periodSessions.length} session${periodSessions.length > 1 ? "s" : ""}`}
          </p>

          {periodSessions.length === 0 && allSessions.length > 0 && (
            <Card className="glass-card p-5 text-center">
              <FileText className="h-7 w-7 text-muted-foreground/30 mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">
                No sessions in this period. Select a longer time range above.
              </p>
            </Card>
          )}

          {sortedDays.map((day) => (
            <div key={day}>
              <p className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wide">
                {day}
              </p>
              <div className="space-y-2">
                {dayGroups[day].map((session) => (
                  <Card
                    key={session.session_id}
                    className={`glass-card p-4 cursor-pointer transition-all hover-glow ${
                      selectedSession === session.session_id
                        ? "border-primary/30 bg-primary/5"
                        : ""
                    }`}
                    onClick={() =>
                      setSelectedSession(
                        selectedSession === session.session_id ? null : session.session_id
                      )
                    }
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-3 min-w-0">
                        <div className="shrink-0">
                          <p className="text-xs text-muted-foreground">
                            {new Date((session.start_time ?? 0) * 1000).toLocaleTimeString(
                              "en-US",
                              { hour: "numeric", minute: "2-digit" }
                            )}
                          </p>
                          {session.summary?.duration_sec && (
                            <p className="text-sm font-medium">
                              {formatDuration(session.summary.duration_sec)}
                            </p>
                          )}
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {session.summary?.avg_focus != null && (
                            <span className="px-2 py-0.5 rounded-full text-[10px] bg-primary/10 text-primary">
                              F {Math.round(session.summary.avg_focus * 100)}%
                            </span>
                          )}
                          {session.summary?.avg_stress != null && (
                            <span className="px-2 py-0.5 rounded-full text-[10px] bg-warning/10 text-warning">
                              S {Math.round(session.summary.avg_stress * 100)}%
                            </span>
                          )}
                          {session.summary?.avg_flow != null && (
                            <span className="px-2 py-0.5 rounded-full text-[10px] bg-success/10 text-success">
                              Fl {Math.round(session.summary.avg_flow * 100)}%
                            </span>
                          )}
                          {session.summary?.dominant_emotion && (
                            <span className="px-2 py-0.5 rounded-full text-[10px] bg-muted text-muted-foreground capitalize">
                              {session.summary.dominant_emotion}
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex gap-1 shrink-0">
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground"
                          onClick={(e) => handleExport(session.session_id, e)}
                          title="Export CSV"
                        >
                          <Download className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-7 w-7 p-0 text-destructive"
                          onClick={(e) => handleDelete(session.session_id, e)}
                          title="Delete session"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </main>
  );
}
