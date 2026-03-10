import { useQuery } from "@tanstack/react-query";
import { listSessions, type SessionSummary } from "@/lib/ml-api";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Trophy,
  Brain,
  Zap,
  Wind,
  Moon,
  Flame,
  TrendingUp,
  Activity,
  CalendarDays,
} from "lucide-react";

/* ── helpers ──────────────────────────────────────────────────── */

function pct(v: number | undefined): number {
  return Math.round((v ?? 0) * 100);
}

function fmtDuration(secs: number): string {
  const h = Math.floor(secs / 3600);
  const m = Math.round((secs % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

function fmtDate(ts: number): string {
  return new Date(ts * 1000).toLocaleDateString([], {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function longestStreak(sessions: SessionSummary[]): number {
  if (!sessions.length) return 0;
  const dayMs = 86_400_000;
  const days = new Set(
    sessions.map((s) => {
      const d = new Date((s.start_time ?? 0) * 1000);
      d.setHours(0, 0, 0, 0);
      return d.getTime();
    }),
  );
  const sorted = Array.from(days).sort((a, b) => a - b);
  let best = 1;
  let cur = 1;
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] - sorted[i - 1] <= dayMs) {
      cur++;
      best = Math.max(best, cur);
    } else {
      cur = 1;
    }
  }
  return best;
}

/* ── Record card ──────────────────────────────────────────────── */

interface RecordCardProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  sub?: string;
  color?: string;
  empty?: boolean;
}

function RecordCard({ icon: Icon, label, value, sub, color = "text-primary", empty }: RecordCardProps) {
  return (
    <Card className="glass-card p-5 hover-glow flex flex-col gap-2">
      <div className="flex items-center gap-2 text-muted-foreground text-xs uppercase tracking-wider">
        <Icon className={`h-3.5 w-3.5 ${color}`} />
        {label}
      </div>
      {empty ? (
        <p className="text-sm text-muted-foreground/50">—</p>
      ) : (
        <>
          <p className={`text-2xl font-bold font-mono ${color}`}>{value}</p>
          {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
        </>
      )}
    </Card>
  );
}

/* ── Top-5 sessions table ─────────────────────────────────────── */

interface TopRowProps {
  rank: number;
  session: SessionSummary;
  metric: string;
  value: number;
}

function TopRow({ rank, session, metric, value }: TopRowProps) {
  const medals = ["🥇", "🥈", "🥉", "4.", "5."];
  return (
    <div className="flex items-center gap-3 py-2 border-b border-border/20 last:border-0">
      <span className="text-sm w-6 shrink-0">{medals[rank] ?? `${rank + 1}.`}</span>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate capitalize">
          {session.session_type || "General"} session
        </p>
        <p className="text-xs text-muted-foreground">{fmtDate(session.start_time)}</p>
      </div>
      <Badge variant="secondary" className="font-mono shrink-0">
        {value}% {metric}
      </Badge>
    </div>
  );
}

/* ── Main page ────────────────────────────────────────────────── */

export default function PersonalRecords() {
  const { data: sessions = [], isLoading } = useQuery<SessionSummary[]>({
    queryKey: ["sessions-records"],
    queryFn: () => listSessions(),
    staleTime: 2 * 60_000,
    retry: false,
  });

  const withData = sessions.filter((s) => s.summary?.avg_focus != null);
  const hasData = withData.length > 0;

  /* ── all-time bests ── */
  const peakFocus = withData.reduce((m, s) => Math.max(m, pct(s.summary?.avg_focus)), 0);
  const peakFocusSession = withData.find((s) => pct(s.summary?.avg_focus) === peakFocus);

  const peakFlow = withData.reduce((m, s) => Math.max(m, pct(s.summary?.avg_flow)), 0);
  const peakFlowSession = withData.find((s) => pct(s.summary?.avg_flow) === peakFlow);

  const peakRelax = withData.reduce((m, s) => Math.max(m, pct(s.summary?.avg_relaxation)), 0);
  const peakRelaxSession = withData.find((s) => pct(s.summary?.avg_relaxation) === peakRelax);

  const peakCreativity = withData.reduce((m, s) => Math.max(m, pct(s.summary?.avg_creativity)), 0);

  const longestSec = withData.reduce((m, s) => Math.max(m, s.summary?.duration_sec ?? 0), 0);
  const longestSession = withData.find((s) => (s.summary?.duration_sec ?? 0) === longestSec);

  const streak = longestStreak(sessions);
  const totalSessions = withData.length;
  const totalMinutes = Math.round(
    withData.reduce((s, r) => s + (r.summary?.duration_sec ?? 0), 0) / 60,
  );

  /* ── top-5 by focus ── */
  const top5Focus = [...withData]
    .sort((a, b) => pct(b.summary?.avg_focus) - pct(a.summary?.avg_focus))
    .slice(0, 5);

  return (
    <main className="p-4 md:p-6 space-y-6 max-w-3xl">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Trophy className="h-6 w-6 text-amber-400" />
        <div>
          <h1 className="text-xl font-semibold">Personal Records</h1>
          <p className="text-xs text-muted-foreground">Your all-time bests across every session</p>
        </div>
      </div>

      {/* Empty state */}
      {!isLoading && !hasData && (
        <Card className="glass-card p-6 text-center space-y-2">
          <Trophy className="h-8 w-8 text-muted-foreground/30 mx-auto" />
          <p className="text-sm font-medium">No records yet</p>
          <p className="text-xs text-muted-foreground">
            Complete voice check-ins or health syncs to build your records. Connect Muse 2 for EEG session records.
          </p>
        </Card>
      )}

      {/* Records grid */}
      {hasData && (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <RecordCard
              icon={Brain}
              label="Peak Focus"
              value={`${peakFocus}%`}
              sub={peakFocusSession ? fmtDate(peakFocusSession.start_time) : undefined}
              color="text-blue-400"
            />
            <RecordCard
              icon={Zap}
              label="Peak Flow"
              value={`${peakFlow}%`}
              sub={peakFlowSession ? fmtDate(peakFlowSession.start_time) : undefined}
              color="text-primary"
            />
            <RecordCard
              icon={Wind}
              label="Peak Relaxation"
              value={`${peakRelax}%`}
              sub={peakRelaxSession ? fmtDate(peakRelaxSession.start_time) : undefined}
              color="text-emerald-400"
            />
            <RecordCard
              icon={Activity}
              label="Peak Creativity"
              value={`${peakCreativity}%`}
              color="text-purple-400"
              empty={peakCreativity === 0}
            />
            <RecordCard
              icon={Moon}
              label="Longest Session"
              value={longestSec > 0 ? fmtDuration(longestSec) : "—"}
              sub={longestSession ? fmtDate(longestSession.start_time) : undefined}
              color="text-indigo-400"
              empty={longestSec === 0}
            />
            <RecordCard
              icon={Flame}
              label="Best Streak"
              value={streak > 0 ? `${streak} days` : "—"}
              color="text-orange-400"
              empty={streak === 0}
            />
          </div>

          {/* Totals */}
          <Card className="glass-card p-5 hover-glow">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3 flex items-center gap-2">
              <CalendarDays className="h-3.5 w-3.5" />
              All-time totals
            </h3>
            <div className="grid grid-cols-3 gap-4">
              {[
                { label: "Sessions", value: totalSessions },
                { label: "Minutes", value: totalMinutes },
                { label: "Days active", value: new Set(withData.map((s) => {
                    const d = new Date((s.start_time ?? 0) * 1000);
                    d.setHours(0, 0, 0, 0);
                    return d.getTime();
                  })).size },
              ].map(({ label, value }) => (
                <div key={label} className="text-center">
                  <p className="text-2xl font-bold font-mono text-foreground">{value}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{label}</p>
                </div>
              ))}
            </div>
          </Card>

          {/* Top sessions by focus */}
          {top5Focus.length > 0 && (
            <Card className="glass-card p-5 hover-glow">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3 flex items-center gap-2">
                <TrendingUp className="h-3.5 w-3.5" />
                Top sessions by focus
              </h3>
              {top5Focus.map((s, i) => (
                <TopRow
                  key={s.session_id}
                  rank={i}
                  session={s}
                  metric="focus"
                  value={pct(s.summary?.avg_focus)}
                />
              ))}
            </Card>
          )}
        </>
      )}
    </main>
  );
}
