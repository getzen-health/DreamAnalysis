import React, { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Clock,
  Download,
  Trash2,
  Brain,
  Eye,
  Activity,
  Sparkles,
  Moon,
  Heart,
  Zap,
  BookOpen,
  BarChart2,
  TrendingUp,
} from "lucide-react";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import {
  listSessions,
  deleteSession,
  exportSession,
  getEmotionHistory,
  type SessionSummary,
  type StoredEmotionReading,
} from "@/lib/ml-api";

/* ── Shared tooltip style (theme-aware, uses CSS vars) ───────── */
const tooltipStyle = {
  contentStyle: {
    background: "var(--popover)",
    border: "1px solid var(--border)",
    borderRadius: 10,
    fontSize: 11,
    color: "var(--popover-foreground)",
  },
  labelStyle: { color: "var(--muted-foreground)", marginBottom: 4, fontSize: 10 },
  itemStyle: { padding: "1px 0" },
};

/* ── Types ───────────────────────────────────────────────────── */
interface DreamEntry {
  id: string;
  dreamText: string;
  aiAnalysis?: string;
  symbols?: string[];
  emotions?: { emotion: string; intensity: number }[];
  timestamp: string;
}

interface HealthEntry {
  id: string;
  heartRate: number;
  stressLevel: number;
  sleepQuality: number;
  neuralActivity: number;
  sleepDuration?: number;
  timestamp: string;
}

/* ── Periods ─────────────────────────────────────────────────── */
const PERIODS = [
  { label: "Today", days: 1 },
  { label: "Week", days: 7 },
  { label: "Month", days: 30 },
  { label: "3 Months", days: 90 },
  { label: "Year", days: 365 },
];

type Tab = "sessions" | "emotions" | "dreams" | "health";

/* ── Helpers ─────────────────────────────────────────────────── */
const CURRENT_USER = "default";

function cutoffUnix(days: number) {
  return Date.now() / 1000 - days * 86400;
}

function cutoffDate(days: number) {
  return new Date(Date.now() - days * 86400 * 1000);
}

function fmt(sec: number) {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function fmtTime(ts: string | number) {
  const d = typeof ts === "number" ? new Date(ts * 1000) : new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function fmtDate(ts: string | number) {
  const d = typeof ts === "number" ? new Date(ts * 1000) : new Date(ts);
  return d.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" });
}

function avg(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

/* ── Component ───────────────────────────────────────────────── */
export default function DataHub() {
  const [periodDays, setPeriodDays] = useState(1); // default: today
  const [tab, setTab] = useState<Tab>("sessions");
  const [selectedSession, setSelectedSession] = useState<string | null>(null);

  /* — Data fetches — */
  const { data: allSessions = [], isLoading: sessionsLoading, refetch: refetchSessions } =
    useQuery<SessionSummary[]>({
      queryKey: ["sessions"],
      queryFn: () => listSessions(),
      retry: false,
      refetchInterval: 30_000,
    });

  const { data: emotions = [], isLoading: emotionsLoading } =
    useQuery<StoredEmotionReading[]>({
      queryKey: ["emotions", CURRENT_USER, periodDays],
      queryFn: () => getEmotionHistory(CURRENT_USER, periodDays),
      staleTime: 30_000,
    });

  const { data: dreams = [], isLoading: dreamsLoading } =
    useQuery<DreamEntry[]>({
      queryKey: ["dreams", CURRENT_USER],
      queryFn: async () => {
        const res = await fetch(`/api/dream-analysis/${CURRENT_USER}`);
        if (!res.ok) return [];
        return res.json();
      },
      staleTime: 60_000,
    });

  const { data: health = [], isLoading: healthLoading } =
    useQuery<HealthEntry[]>({
      queryKey: ["health", CURRENT_USER],
      queryFn: async () => {
        const res = await fetch(`/api/health-metrics/${CURRENT_USER}`);
        if (!res.ok) return [];
        return res.json();
      },
      staleTime: 60_000,
    });

  /* — Period filtering — */
  const cutoffU = cutoffUnix(periodDays);
  const cutoffD = cutoffDate(periodDays);

  const periodSessions = allSessions.filter((s) => (s.start_time ?? 0) >= cutoffU);
  const periodDreams = dreams.filter((d) => new Date(d.timestamp) >= cutoffD);
  const periodHealth = health.filter((h) => new Date(h.timestamp) >= cutoffD);
  // emotions are already filtered server-side by days param

  /* — Summary metrics — */
  const n = Math.max(periodSessions.length, 1);
  const avgFocus = periodSessions.reduce((s, x) => s + (x.summary?.avg_focus ?? 0), 0) / n;
  const avgStress = periodSessions.reduce((s, x) => s + (x.summary?.avg_stress ?? 0), 0) / n;
  const avgFlow = periodSessions.reduce((s, x) => s + (x.summary?.avg_flow ?? 0), 0) / n;

  /* — Emotion chart data — */
  const emotionChartData = emotions
    .filter((_, i, arr) => {
      const step = Math.max(1, Math.floor(arr.length / 200));
      return i % step === 0;
    })
    .map((r) => ({
      time: fmtTime(r.timestamp),
      stress: Math.round(r.stress * 100),
      focus: Math.round(r.focus * 100),
      happiness: Math.round(r.happiness * 100),
    }));

  /* — Trend chart data for sessions — */
  const sessionTrendData = (() => {
    const map: Record<string, { focus: number[]; stress: number[]; ts: number }> = {};
    for (const s of periodSessions) {
      if (s.summary?.avg_focus == null) continue;
      const d = new Date((s.start_time ?? 0) * 1000);
      const key = periodDays <= 1
        ? d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })
        : d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
      if (!map[key]) map[key] = { focus: [], stress: [], ts: d.getTime() };
      map[key].focus.push((s.summary.avg_focus ?? 0) * 100);
      map[key].stress.push((s.summary.avg_stress ?? 0) * 100);
    }
    return Object.entries(map)
      .sort((a, b) => a[1].ts - b[1].ts)
      .map(([date, data]) => ({
        date,
        focus: Math.round(avg(data.focus)),
        stress: Math.round(avg(data.stress)),
      }));
  })();

  /* — Handlers — */
  const handleDelete = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deleteSession(sessionId);
      refetchSessions();
      if (selectedSession === sessionId) setSelectedSession(null);
    } catch { /* ignore */ }
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
    } catch { /* ignore */ }
  };

  /* — Day-grouped sessions — */
  const dayGroups: Record<string, SessionSummary[]> = {};
  for (const s of periodSessions) {
    const key = fmtDate(s.start_time ?? 0);
    if (!dayGroups[key]) dayGroups[key] = [];
    dayGroups[key].push(s);
  }
  const sortedDays = Object.keys(dayGroups).sort((a, b) => {
    return (dayGroups[b][0].start_time ?? 0) - (dayGroups[a][0].start_time ?? 0);
  });

  const tabs: { id: Tab; label: string; icon: React.ReactNode; count: number }[] = [
    { id: "sessions", label: "Sessions", icon: <Brain className="h-3.5 w-3.5" />, count: periodSessions.length },
    { id: "emotions", label: "Emotions", icon: <Activity className="h-3.5 w-3.5" />, count: emotions.length },
    { id: "dreams", label: "Dreams", icon: <Moon className="h-3.5 w-3.5" />, count: periodDreams.length },
    { id: "health", label: "Health", icon: <Heart className="h-3.5 w-3.5" />, count: periodHealth.length },
  ];

  return (
    <main className="p-4 md:p-6 space-y-5 max-w-5xl">
      {/* ── Header ── */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <BarChart2 className="h-5 w-5 text-primary" />
          <h2 className="text-lg font-semibold">Data Hub</h2>
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

      {/* ── Summary chips ── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "Sessions", value: periodSessions.length, icon: Brain, color: "text-primary", bg: "bg-primary/10" },
          { label: "Readings", value: emotions.length, icon: Activity, color: "text-cyan-500", bg: "bg-cyan-500/10" },
          { label: "Dreams", value: periodDreams.length, icon: Moon, color: "text-secondary", bg: "bg-secondary/10" },
          { label: "Health", value: periodHealth.length, icon: Heart, color: "text-rose-500", bg: "bg-rose-500/10" },
        ].map(({ label, value, icon: Icon, color, bg }) => (
          <Card key={label} className="glass-card p-4">
            <div className={`w-8 h-8 rounded-lg ${bg} flex items-center justify-center mb-2`}>
              <Icon className={`h-4 w-4 ${color}`} />
            </div>
            <p className="text-2xl font-bold">{value}</p>
            <p className="text-xs text-muted-foreground">{label}</p>
          </Card>
        ))}
      </div>

      {/* ── Session summary card ── */}
      {periodSessions.length > 0 && (
        <Card className="glass-card p-5">
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-xl bg-primary/10 flex items-center justify-center shrink-0">
              <Sparkles className="h-4 w-4 text-primary" />
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium mb-1">
                {periodSessions.length} session{periodSessions.length !== 1 ? "s" : ""} recorded
              </p>
              <div className="flex flex-wrap gap-2 mt-2">
                <span className="px-2.5 py-1 rounded-full text-[10px] font-medium bg-primary/10 text-primary">
                  Focus {Math.round(avgFocus * 100)}%
                </span>
                <span className="px-2.5 py-1 rounded-full text-[10px] font-medium bg-warning/10 text-warning">
                  Stress {Math.round(avgStress * 100)}%
                </span>
                <span className="px-2.5 py-1 rounded-full text-[10px] font-medium bg-success/10 text-success">
                  Flow {Math.round(avgFlow * 100)}%
                </span>
                {emotions.length > 0 && (
                  <span className="px-2.5 py-1 rounded-full text-[10px] font-medium bg-cyan-500/10 text-cyan-500">
                    {emotions.length} emotion readings
                  </span>
                )}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* ── Tabs ── */}
      <div className="flex gap-1 border-b border-border/30 pb-0">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-1.5 px-4 py-2 text-xs font-medium rounded-t-lg transition-colors border-b-2 -mb-px ${
              tab === t.id
                ? "border-primary text-primary bg-primary/5"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {t.icon}
            {t.label}
            {t.count > 0 && (
              <span className={`ml-1 px-1.5 py-0.5 rounded-full text-[9px] ${
                tab === t.id ? "bg-primary/20 text-primary" : "bg-muted text-muted-foreground"
              }`}>
                {t.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* ══ Sessions Tab ══════════════════════════════════════════ */}
      {tab === "sessions" && (
        <div className="space-y-4">
          {sessionsLoading && <p className="text-sm text-muted-foreground">Loading sessions...</p>}

          {!sessionsLoading && periodSessions.length === 0 && (
            <Card className="glass-card p-10 text-center">
              <Brain className="h-10 w-10 text-muted-foreground/30 mx-auto mb-3" />
              <p className="text-sm text-muted-foreground">No sessions in this period.</p>
            </Card>
          )}

          {/* Trend chart */}
          {sessionTrendData.length >= 2 && (
            <Card className="glass-card p-5">
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp className="h-4 w-4 text-primary" />
                <h3 className="text-sm font-medium">Focus & Stress Over Time</h3>
              </div>
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={sessionTrendData}>
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis hide domain={[0, 100]} />
                  <Tooltip {...tooltipStyle} formatter={(v: number) => [`${v}%`]} />
                  <Line type="monotone" dataKey="focus" name="Focus" stroke="hsl(152,60%,48%)" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="stress" name="Stress" stroke="hsl(38,85%,58%)" strokeWidth={1.5} dot={false} strokeDasharray="4 4" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Session cards */}
          {sortedDays.map((day) => (
            <div key={day}>
              <p className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wide">{day}</p>
              <div className="space-y-2">
                {dayGroups[day].map((session) => (
                  <Card
                    key={session.session_id}
                    className={`glass-card p-4 cursor-pointer hover-glow transition-all ${
                      selectedSession === session.session_id ? "border-primary/40" : ""
                    }`}
                    onClick={() => setSelectedSession(
                      selectedSession === session.session_id ? null : session.session_id
                    )}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-3 min-w-0">
                        <div className="shrink-0">
                          <p className="text-xs text-muted-foreground">{fmtTime(session.start_time ?? 0)}</p>
                          {session.summary?.duration_sec && (
                            <p className="text-sm font-medium">{fmt(session.summary.duration_sec)}</p>
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
                        <Button size="sm" variant="ghost" className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground"
                          onClick={(e) => handleExport(session.session_id, e)} title="Export CSV">
                          <Download className="h-3.5 w-3.5" />
                        </Button>
                        <Button size="sm" variant="ghost" className="h-7 w-7 p-0 text-destructive"
                          onClick={(e) => handleDelete(session.session_id, e)} title="Delete">
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

      {/* ══ Emotions Tab ══════════════════════════════════════════ */}
      {tab === "emotions" && (
        <div className="space-y-4">
          {emotionsLoading && <p className="text-sm text-muted-foreground">Loading readings...</p>}

          {!emotionsLoading && emotions.length === 0 && (
            <Card className="glass-card p-10 text-center">
              <Activity className="h-10 w-10 text-muted-foreground/30 mx-auto mb-3" />
              <p className="text-sm text-muted-foreground">No emotion readings in this period.</p>
              <p className="text-xs text-muted-foreground mt-1">Connect your Muse 2 to start recording.</p>
            </Card>
          )}

          {emotionChartData.length >= 2 && (
            <Card className="glass-card p-5">
              <div className="flex items-center gap-2 mb-3">
                <Activity className="h-4 w-4 text-cyan-500" />
                <h3 className="text-sm font-medium">Emotion Timeline</h3>
                <span className="text-xs text-muted-foreground">{emotions.length} readings</span>
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={emotionChartData} margin={{ left: -24 }}>
                  <defs>
                    <linearGradient id="stressG" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(340,70%,55%)" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(340,70%,55%)" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="focusG" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(200,70%,55%)" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(200,70%,55%)" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="happyG" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(38,85%,58%)" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(38,85%,58%)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                  <Tooltip {...tooltipStyle} formatter={(v: number) => [`${v}%`]} />
                  <Area type="monotone" dataKey="stress" name="Stress" stroke="hsl(340,70%,55%)" fill="url(#stressG)" strokeWidth={1.5} dot={false} />
                  <Area type="monotone" dataKey="focus" name="Focus" stroke="hsl(200,70%,55%)" fill="url(#focusG)" strokeWidth={1.5} dot={false} />
                  <Area type="monotone" dataKey="happiness" name="Happiness" stroke="hsl(38,85%,58%)" fill="url(#happyG)" strokeWidth={1.5} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          )}

          {emotions.length > 0 && (
            <Card className="glass-card p-5">
              <p className="text-xs font-medium text-muted-foreground mb-3">Recent readings (latest 50)</p>
              <div className="space-y-1.5 max-h-96 overflow-y-auto">
                {emotions.slice(-50).reverse().map((r) => (
                  <div key={r.id} className="flex items-center gap-3 py-1.5 border-b border-border/20 last:border-0">
                    <span className="text-[10px] text-muted-foreground w-14 shrink-0">{fmtTime(r.timestamp)}</span>
                    <span className="text-xs capitalize font-medium w-20 shrink-0">{r.dominantEmotion}</span>
                    <div className="flex gap-2 flex-wrap">
                      <span className="text-[10px] text-primary">F {Math.round(r.focus * 100)}%</span>
                      <span className="text-[10px] text-rose-400">S {Math.round(r.stress * 100)}%</span>
                      <span className="text-[10px] text-amber-400">H {Math.round(r.happiness * 100)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ══ Dreams Tab ════════════════════════════════════════════ */}
      {tab === "dreams" && (
        <div className="space-y-4">
          {dreamsLoading && <p className="text-sm text-muted-foreground">Loading dreams...</p>}

          {!dreamsLoading && periodDreams.length === 0 && (
            <Card className="glass-card p-10 text-center">
              <Moon className="h-10 w-10 text-muted-foreground/30 mx-auto mb-3" />
              <p className="text-sm text-muted-foreground">No dreams recorded in this period.</p>
            </Card>
          )}

          {periodDreams.map((dream) => (
            <Card key={dream.id} className="glass-card p-5 hover-glow">
              <div className="flex items-start justify-between gap-3 mb-3">
                <div className="flex items-center gap-2">
                  <Moon className="h-4 w-4 text-secondary" />
                  <span className="text-xs text-muted-foreground">{fmtDate(dream.timestamp)} · {fmtTime(dream.timestamp)}</span>
                </div>
              </div>
              <p className="text-sm text-foreground/80 leading-relaxed mb-3 line-clamp-3">{dream.dreamText}</p>
              {dream.aiAnalysis && (
                <div className="ai-insight-card text-xs text-muted-foreground leading-relaxed mb-3">
                  <div className="flex items-center gap-1.5 mb-1">
                    <Sparkles className="h-3 w-3 text-primary" />
                    <span className="font-medium text-foreground">AI Analysis</span>
                  </div>
                  {dream.aiAnalysis}
                </div>
              )}
              {dream.symbols && dream.symbols.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {dream.symbols.map((s) => (
                    <span key={s} className="px-2 py-0.5 rounded-full text-[10px] bg-secondary/10 text-secondary capitalize">{s}</span>
                  ))}
                </div>
              )}
            </Card>
          ))}
        </div>
      )}

      {/* ══ Health Tab ════════════════════════════════════════════ */}
      {tab === "health" && (
        <div className="space-y-4">
          {healthLoading && <p className="text-sm text-muted-foreground">Loading health data...</p>}

          {!healthLoading && periodHealth.length === 0 && (
            <Card className="glass-card p-10 text-center">
              <Heart className="h-10 w-10 text-muted-foreground/30 mx-auto mb-3" />
              <p className="text-sm text-muted-foreground">No health data in this period.</p>
            </Card>
          )}

          {periodHealth.length > 0 && (
            <Card className="glass-card p-5">
              <div className="flex items-center gap-2 mb-4">
                <Heart className="h-4 w-4 text-rose-500" />
                <h3 className="text-sm font-medium">Health Metrics</h3>
                <span className="text-xs text-muted-foreground">{periodHealth.length} entries</span>
              </div>
              <div className="space-y-2 max-h-[500px] overflow-y-auto">
                {[...periodHealth].reverse().map((h) => (
                  <div key={h.id} className="flex items-center gap-4 py-2 border-b border-border/20 last:border-0">
                    <span className="text-[10px] text-muted-foreground w-28 shrink-0">
                      {fmtDate(h.timestamp)} {fmtTime(h.timestamp)}
                    </span>
                    <div className="flex gap-3 flex-wrap">
                      <span className="text-xs"><span className="text-rose-400">HR</span> {h.heartRate} bpm</span>
                      <span className="text-xs"><span className="text-warning">Stress</span> {h.stressLevel}%</span>
                      <span className="text-xs"><span className="text-primary">Sleep</span> {h.sleepQuality}%</span>
                      <span className="text-xs"><span className="text-secondary">Neural</span> {h.neuralActivity}%</span>
                      {h.sleepDuration && (
                        <span className="text-xs"><span className="text-cyan-400">Duration</span> {h.sleepDuration.toFixed(1)}h</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </main>
  );
}
