import { useState, useEffect, useMemo, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Link, useLocation } from "wouter";
import { InsightEngine, type StoredInsight } from "@/lib/insight-engine";
import { MorningBriefingCard } from "@/components/morning-briefing-card";
import { Card } from "@/components/ui/card";
import {
  Sparkles, Brain, Heart, UtensilsCrossed,
  TrendingUp, TrendingDown, Minus,
  Activity,
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine,
} from "recharts";
import { pageTransition } from "@/lib/animations";
import { getParticipantId } from "@/lib/participant";
import { EEGPeakHours } from "@/components/eeg-peak-hours";
import { fetchLatestDreamContext } from "@/lib/dream-briefing-context";
import { sbGetGeneric } from "@/lib/supabase-store";

// ── Types ──────────────────────────────────────────────────────────────────

interface EmotionEntry {
  stress: number;
  happiness: number;
  focus: number;
  dominantEmotion: string;
  timestamp: string;
  valence?: number;
  arousal?: number;
}


interface PersonalBaseline {
  avgStress: number;
  avgFocus: number;
  avgHappiness: number;
  avgCalories: number;
  sampleCount: number;
}

// ── Insight engine ─────────────────────────────────────────────────────────

function computeBaseline(entries: EmotionEntry[]): PersonalBaseline | null {
  if (entries.length < 3) return null;
  const n = entries.length;
  return {
    avgStress: entries.reduce((s, e) => s + e.stress, 0) / n,
    avgFocus: entries.reduce((s, e) => s + e.focus, 0) / n,
    avgHappiness: entries.reduce((s, e) => s + e.happiness, 0) / n,
    avgCalories: 0,
    sampleCount: n,
  };
}

function pct(v: number): string {
  return `${Math.round(v * 100)}%`;
}


// ── Weekly stats ──────────────────────────────────────────────────────────

function computeWeeklyStats(entries: EmotionEntry[]) {
  const week = entries.filter(e => {
    const d = new Date(e.timestamp);
    return Date.now() - d.getTime() < 7 * 86400000;
  });
  if (!week.length) return null;
  return {
    avgStress: week.reduce((s, e) => s + e.stress, 0) / week.length,
    avgFocus: week.reduce((s, e) => s + e.focus, 0) / week.length,
    avgHappiness: week.reduce((s, e) => s + e.happiness, 0) / week.length,
    checkIns: week.length,
    topEmotion: (() => {
      const c: Record<string, number> = {};
      week.forEach(e => { c[e.dominantEmotion] = (c[e.dominantEmotion] ?? 0) + 1; });
      return Object.entries(c).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "—";
    })(),
  };
}

// ── Metric chip ───────────────────────────────────────────────────────────

function MetricChip({
  label,
  value,
  delta,
  color,
}: {
  label: string;
  value: string;
  delta?: number;
  color: string;
}) {
  const DeltaIcon = delta === undefined ? Minus : delta > 0.02 ? TrendingUp : delta < -0.02 ? TrendingDown : Minus;
  return (
    <div className="flex flex-col gap-1 p-3 rounded-[14px] bg-card border border-border">
      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p>
      <p className="text-lg font-bold" style={{ color }}>{value}</p>
      {delta !== undefined && (
        <div className="flex items-center gap-0.5 text-[10px] text-muted-foreground">
          <DeltaIcon className="h-3 w-3" />
          <span>{Math.abs(delta) < 0.02 ? "on track" : `${delta > 0 ? "+" : ""}${Math.round(delta * 100)}pts`}</span>
        </div>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────

const PARTICIPANT = getParticipantId();

export default function Insights() {
  const [emotionHistory, setEmotionHistory] = useState<EmotionEntry[]>([]);
  const [trendPeriod, setTrendPeriod] = useState<"7d" | "30d">("7d");

  // InsightEngine integration
  const { data: user } = useQuery<{ id: string }>({ queryKey: ["/api/user"] });
  const userId = user?.id ?? "anonymous";
  const engineRef = useRef<InsightEngine | null>(null);
  const [insights, setInsights] = useState<StoredInsight[]>([]);
  const [briefing, setBriefing] = useState<import("@/lib/insight-engine").BriefingResponse | null>(null);
  const [briefingLoading, setBriefingLoading] = useState(false);
  const [, navigate] = useLocation();

  useEffect(() => {
    engineRef.current = new InsightEngine(userId);
    setBriefing(engineRef.current.getMorningBriefing());
    engineRef.current.getStoredInsights().then(setInsights);
  }, [userId]);

  const handleGenerateBriefing = async () => {
    setBriefingLoading(true);
    try {
      const emotionHistoryRaw: any[] = sbGetGeneric("ndw_emotion_history") ?? [];
      const yesterday = emotionHistoryRaw.filter((e: any) => {
        const d = new Date(e.timestamp);
        const now = new Date();
        return d.toISOString().slice(0, 10) < now.toISOString().slice(0, 10);
      });
      const avgStress = yesterday.length > 0
        ? yesterday.reduce((a: number, e: any) => a + (e.stress || 0.4), 0) / yesterday.length
        : 0.4;
      const avgFocus = yesterday.length > 0 ? yesterday.reduce((a: number, e: any) => a + (e.focus || 0.55), 0) / yesterday.length : 0.55;
      const avgValence = yesterday.length > 0 ? yesterday.reduce((a: number, e: any) => a + (e.valence || 0.55), 0) / yesterday.length : 0.55;
      if (!engineRef.current) return;

      // Fetch last night's dream analysis to enrich briefing
      const dreamContext = await fetchLatestDreamContext(userId);

      const newBriefing = await engineRef.current.generateMorningBriefing({
        sleepData: { totalHours: null, deepHours: null, remHours: null, efficiency: null, dataAvailability: "none" },
        morningHrv: null, hrvRange: null,
        emotionSummary: {
          readingCount: yesterday.length, avgStress, avgFocus,
          avgValence, dominantLabel: "neutral", dominantMinutes: 60,
        },
        patternSummaries: insights.map(i => i.headline),
        yesterdaySummary: `${yesterday.length} readings. Avg stress ${(avgStress * 100).toFixed(0)}%.`,
        dreamContext,
      });
      setBriefing(newBriefing);
    } catch (e) {
      console.warn("Briefing generation failed", e);
    } finally {
      setBriefingLoading(false);
    }
  };

  // Fetch emotion history from Express API (persisted DB data)
  const { data: apiHistory } = useQuery<any[]>({
    queryKey: [`/api/brain/history/${PARTICIPANT}?days=30`],
    staleTime: 5 * 60_000,
  });

  // Load data from localStorage + merge with API data
  useEffect(() => {
    let local: EmotionEntry[] = sbGetGeneric("ndw_emotion_history") ?? [];

    // Also try last emotion
    try {
      const last = sbGetGeneric<any>("ndw_last_emotion");
      if (last) {
        if (last?.result) {
          const entry: EmotionEntry = {
            stress: last.result.stress_index ?? 0.5,
            happiness: Math.max(0, Math.min(1, (last.result.valence ?? 0 + 1) / 2)),
            focus: last.result.focus_index ?? 0.5,
            dominantEmotion: last.result.emotion ?? "neutral",
            timestamp: new Date(last.timestamp).toISOString(),
            valence: last.result.valence,
            arousal: last.result.arousal,
          };
          if (!local.some(e => e.timestamp === entry.timestamp)) {
            local = [...local, entry];
          }
        }
      }
    } catch { /* ignore */ }

    // Merge API data (deduplicate by timestamp)
    if (apiHistory && apiHistory.length > 0) {
      const localTimestamps = new Set(local.map(e => e.timestamp));
      const apiEntries: EmotionEntry[] = apiHistory
        .filter((r: any) => !localTimestamps.has(r.timestamp ?? r.created_at))
        .map((r: any) => ({
          stress: r.stress ?? 0.5,
          happiness: r.happiness ?? r.mood ?? 0.5,
          focus: r.focus ?? 0.5,
          dominantEmotion: r.dominantEmotion ?? r.dominant_emotion ?? r.emotion ?? "neutral",
          timestamp: r.timestamp ?? r.created_at,
          valence: r.valence,
          arousal: r.arousal,
        }));
      local = [...local, ...apiEntries];
    }

    // Sort by timestamp and deduplicate
    local.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
    setEmotionHistory(local.slice(-1000));
  }, [apiHistory]);


  const baseline = useMemo(() => computeBaseline(emotionHistory), [emotionHistory]);
  const weeklyStats = useMemo(() => computeWeeklyStats(emotionHistory), [emotionHistory]);
  const latest = emotionHistory[emotionHistory.length - 1] ?? null;

  // Daily-bucketed trend data for chart — average all readings within each calendar day
  const trendChartData = useMemo(() => {
    const daysBack = trendPeriod === "7d" ? 7 : 30;
    const now = new Date();
    const days: { date: string; label: string; ms: number }[] = [];
    for (let i = daysBack - 1; i >= 0; i--) {
      const d = new Date(now);
      d.setDate(d.getDate() - i);
      days.push({
        date: d.toDateString(),
        label: daysBack <= 7
          ? d.toLocaleDateString([], { weekday: "short" })
          : d.toLocaleDateString([], { month: "short", day: "numeric" }),
        ms: d.getTime(),
      });
    }
    const cutoff = now.getTime() - daysBack * 86_400_000;
    const inRange = emotionHistory.filter(e => new Date(e.timestamp).getTime() >= cutoff);

    return days.map(day => {
      const entries = inRange.filter(e => new Date(e.timestamp).toDateString() === day.date);
      if (entries.length === 0) return { label: day.label, stress: null, focus: null, happiness: null };
      const n = entries.length;
      return {
        label: day.label,
        stress: Math.round(entries.reduce((s, e) => s + e.stress, 0) / n * 100),
        focus: Math.round(entries.reduce((s, e) => s + e.focus, 0) / n * 100),
        happiness: Math.round(entries.reduce((s, e) => s + e.happiness, 0) / n * 100),
      };
    });
  }, [emotionHistory, trendPeriod]);

  const hasData = emotionHistory.length > 0;

  return (
    <motion.main
      {...pageTransition}
      className="min-h-screen px-4 py-5 pb-8 max-w-lg mx-auto"
    >
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-1">
          <div
            className="w-8 h-8 rounded-xl flex items-center justify-center bg-emerald-500/15"
          >
            <Sparkles className="h-4 w-4 text-emerald-500" />
          </div>
          <h1 className="text-xl font-bold text-foreground">Insights</h1>
        </div>
        <p className="text-xs text-muted-foreground ml-10">
          {hasData
            ? `${emotionHistory.length} readings · personal baseline active`
            : "Start a voice check-in to build your insight engine"}
        </p>
      </div>

      {/* Morning Briefing */}
      <MorningBriefingCard
        loading={briefingLoading}
        briefing={briefing}
        onGenerate={handleGenerateBriefing}
      />

      {/* No data state */}
      {!hasData && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center py-16 px-4"
        >
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4 bg-emerald-500/10"
          >
            <Brain className="h-8 w-8 text-emerald-500" />
          </div>
          <p className="text-base font-semibold text-foreground mb-2">Your insight engine is waiting</p>
          <p className="text-sm text-muted-foreground mb-6 leading-relaxed max-w-xs mx-auto">
            Complete a few voice check-ins and the system will start detecting patterns, baselines, and cross-domain correlations — like Oura, but built for your emotional data.
          </p>
          <Link href="/">
            <button
              className="px-5 py-2.5 rounded-xl text-sm font-semibold text-white bg-emerald-500 hover:bg-emerald-600 transition-colors"
            >
              Go to Today's check-in
            </button>
          </Link>
        </motion.div>
      )}

      {hasData && (
        <div className="space-y-6">
          {/* Weekly stats bar */}
          {weeklyStats && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-[14px] bg-card border border-border p-4"
            >
              <div className="flex items-center justify-between mb-3">
                <div>
                  <p className="text-xs font-medium text-emerald-500 uppercase tracking-wider">This Week</p>
                  <p className="text-sm text-muted-foreground">{weeklyStats.checkIns} readings · top emotion: <span className="text-foreground font-medium capitalize">{weeklyStats.topEmotion}</span></p>
                </div>
                <div className="text-right">
                  <p className="text-[10px] text-muted-foreground">Personal baseline</p>
                  <p className="text-xs text-emerald-500 font-medium">{baseline ? `${baseline.sampleCount} data points` : "Building..."}</p>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <MetricChip
                  label="Stress"
                  value={pct(weeklyStats.avgStress)}
                  delta={baseline ? weeklyStats.avgStress - baseline.avgStress : undefined}
                  color="#f87171"
                />
                <MetricChip
                  label="Focus"
                  value={pct(weeklyStats.avgFocus)}
                  delta={baseline ? weeklyStats.avgFocus - baseline.avgFocus : undefined}
                  color="#60a5fa"
                />
                <MetricChip
                  label="Happiness"
                  value={pct(weeklyStats.avgHappiness)}
                  delta={baseline ? weeklyStats.avgHappiness - baseline.avgHappiness : undefined}
                  color="#4ade80"
                />
              </div>
            </motion.div>
          )}

          {/* ── Oura-style Trends Chart ──────────────────────────────────── */}
          {trendChartData.some(d => d.stress !== null) && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.08 }}
              className="rounded-[14px] bg-card border border-border p-4"
            >
              {/* Header + period switcher */}
              <div className="flex items-center justify-between mb-3">
                <div>
                  <p className="text-xs font-semibold text-foreground">Trends</p>
                  <p className="text-[10px] text-muted-foreground">Daily averages · personal baseline</p>
                </div>
                <div
                  className="flex gap-1 rounded-xl p-1 bg-muted/30"
                >
                  {(["7d", "30d"] as const).map(p => (
                    <button
                      key={p}
                      onClick={() => setTrendPeriod(p)}
                      className={`text-[11px] font-semibold px-3 py-1 rounded-lg border-none cursor-pointer transition-all duration-150 ${
                        trendPeriod === p
                          ? "bg-emerald-500/20 text-emerald-500"
                          : "text-muted-foreground"
                      }`}
                    >
                      {p.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              {/* Legend */}
              <div className="flex gap-4 mb-3">
                {[
                  { label: "Stress", color: "#f87171" },
                  { label: "Focus", color: "#60a5fa" },
                  { label: "Happiness", color: "#4ade80" },
                ].map(({ label, color }) => (
                  <div key={label} className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full" style={{ background: color }} />
                    <span className="text-[10px] text-muted-foreground">{label}</span>
                  </div>
                ))}
              </div>

              {/* Chart */}
              <ResponsiveContainer width="100%" height={140}>
                <AreaChart
                  data={trendChartData}
                  margin={{ left: -28, right: 4, top: 4, bottom: 0 }}
                >
                  <defs>
                    <linearGradient id="insGradStress" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f87171" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#f87171" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="insGradFocus" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#60a5fa" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#60a5fa" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="insGradHappy" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4ade80" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#4ade80" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 2" stroke="hsl(var(--border))" strokeOpacity={0.5} vertical={false} />
                  <XAxis
                    dataKey="label"
                    tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
                    axisLine={false}
                    tickLine={false}
                    interval={trendPeriod === "30d" ? 4 : 0}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
                    axisLine={false}
                    tickLine={false}
                    tickCount={3}
                    tickFormatter={v => `${v}%`}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "hsl(var(--card))", border: "1px solid hsl(var(--border))",
                      borderRadius: 10, fontSize: 11, padding: "6px 10px",
                    }}
                    formatter={(v: number, name: string) => [`${v}%`, name]}
                    labelStyle={{ color: "hsl(var(--muted-foreground))", fontSize: 10 }}
                  />
                  {/* Personal baseline lines */}
                  {baseline && (
                    <>
                      <ReferenceLine y={Math.round(baseline.avgStress * 100)} stroke="#f87171" strokeDasharray="4 3" strokeOpacity={0.4} strokeWidth={1.5} />
                      <ReferenceLine y={Math.round(baseline.avgFocus * 100)} stroke="#60a5fa" strokeDasharray="4 3" strokeOpacity={0.4} strokeWidth={1.5} />
                      <ReferenceLine y={Math.round(baseline.avgHappiness * 100)} stroke="#4ade80" strokeDasharray="4 3" strokeOpacity={0.3} strokeWidth={1.5} />
                    </>
                  )}
                  <Area type="monotone" dataKey="stress" name="Stress" stroke="#f87171" strokeWidth={1.5} fill="url(#insGradStress)" dot={false} connectNulls />
                  <Area type="monotone" dataKey="focus" name="Focus" stroke="#60a5fa" strokeWidth={1.5} fill="url(#insGradFocus)" dot={false} connectNulls />
                  <Area type="monotone" dataKey="happiness" name="Happiness" stroke="#4ade80" strokeWidth={1.5} fill="url(#insGradHappy)" dot={false} connectNulls />
                </AreaChart>
              </ResponsiveContainer>

              {baseline && (
                <p className="text-[10px] text-muted-foreground mt-2 text-center">
                  Dashed lines = your personal baseline · {baseline.sampleCount} total readings
                </p>
              )}
            </motion.div>
          )}

          {/* Today's snapshot (if latest reading is from today) */}
          {latest && new Date(latest.timestamp).toDateString() === new Date().toDateString() && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
              className="rounded-[14px] bg-card border border-border p-4"
            >
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">Latest Reading</p>
              <div className="flex items-center gap-4">
                <div
                  className="w-12 h-12 rounded-2xl flex items-center justify-center text-lg bg-emerald-500/10"
                >
                  {latest.dominantEmotion === "happy" ? "😊" :
                   latest.dominantEmotion === "sad" ? "😔" :
                   latest.dominantEmotion === "angry" ? "😤" :
                   latest.dominantEmotion === "fear" ? "😨" :
                   latest.dominantEmotion === "surprise" ? "😲" : "😐"}
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold capitalize text-foreground">{latest.dominantEmotion}</p>
                  <p className="text-xs text-muted-foreground">
                    Stress {pct(latest.stress)} · Focus {pct(latest.focus)}
                    {baseline && (
                      <span className="ml-1 text-emerald-500">
                        · {latest.stress < baseline.avgStress ? "↓ below" : "↑ above"} your avg stress
                      </span>
                    )}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-[10px] text-muted-foreground">
                    {new Date(latest.timestamp).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          {/* InsightEngine pattern-discovered insights */}
          {insights.length > 0 && (
            <div className="space-y-3">
              <p className="text-sm font-semibold text-foreground">Discovered Patterns</p>
              {insights.map(insight => (
                <Card key={insight.id} className="glass-card p-4">
                  <div className="flex items-start gap-3">
                    <div className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${
                      insight.priority === "high" ? "bg-red-400" : insight.priority === "medium" ? "bg-amber-400" : "bg-green-400"
                    }`} />
                    <div className="flex-1">
                      <p className="text-sm font-medium">{insight.headline}</p>
                      <p className="text-xs text-muted-foreground mt-0.5">{insight.context}</p>
                      <button
                        onClick={() => navigate(insight.actionHref)}
                        className="text-xs text-primary mt-2 hover:underline"
                      >
                        {insight.action} →
                      </button>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}

          {/* Cross-domain explorer */}
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-[14px] bg-card border border-border p-4"
          >
            {emotionHistory.length > 0 && (
              <div className="mb-4">
                <EEGPeakHours history={emotionHistory} />
              </div>
            )}
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">Explore Patterns</p>
            <div className="grid grid-cols-2 gap-2">
              {[
                { label: "Emotion Trends", icon: Heart, href: "/discover", color: "#e879a8" },
                { label: "Stress History", icon: Activity, href: "/discover", color: "#f87171" },
                { label: "Food & Mood", icon: UtensilsCrossed, href: "/food-emotion", color: "#4ade80" },
                { label: "Brain States", icon: Brain, href: "/brain-monitor", color: "#60a5fa" },
              ].map(({ label, icon: Icon, href, color }) => (
                <Link key={label} href={href}>
                  <div
                    className="flex items-center gap-2.5 p-3 rounded-xl cursor-pointer active:scale-95 transition-transform"
                    style={{ background: `${color}11`, border: `1px solid ${color}22` }}
                  >
                    <Icon className="h-4 w-4" style={{ color }} />
                    <span className="text-xs font-medium text-foreground">{label}</span>
                  </div>
                </Link>
              ))}
            </div>
          </motion.div>

          {/* Data quality note */}
          <div className="text-center py-2">
            <p className="text-[10px] text-muted-foreground">
              Insights improve with every check-in · {emotionHistory.length} readings recorded
            </p>
          </div>
        </div>
      )}
    </motion.main>
  );
}
