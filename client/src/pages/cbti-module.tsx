import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  Moon,
  Clock,
  CheckCircle,
  XCircle,
  TrendingUp,
  AlertTriangle,
  BedDouble,
  Sun,
} from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────

interface EnrollmentData {
  enrolledAt: string;       // ISO-8601
  baselineSleepHours: number;
  prescribedBedtime: string;    // "HH:MM"
  prescribedWakeTime: string;   // "HH:MM"
  windowHours: number;
  currentWeek: number;
}

interface DailyCheckin {
  bedtimeAdherence: boolean | null;
  getUpIfAwake: boolean | null;
  bedForSleepOnly: boolean | null;
}

interface WeeklyProgressEntry {
  week: number;
  efficiency: number;   // 0–100
  windowHours: number;
  compliance: number;   // 0–100
}

// ─── Demo / mock data (replace with real API calls) ──────────────────────────

const MOCK_ENROLLMENT: EnrollmentData = {
  enrolledAt: new Date(Date.now() - 22 * 24 * 60 * 60 * 1000).toISOString(),
  baselineSleepHours: 5.8,
  prescribedBedtime: "23:30",
  prescribedWakeTime: "06:30",
  windowHours: 5.8,
  currentWeek: 4,
};

const MOCK_WEEKLY: WeeklyProgressEntry[] = [
  { week: 1, efficiency: 73, windowHours: 5.8, compliance: 62 },
  { week: 2, efficiency: 81, windowHours: 5.8, compliance: 74 },
  { week: 3, efficiency: 86, windowHours: 6.1, compliance: 83 },
  { week: 4, efficiency: 89, windowHours: 6.4, compliance: 88 },
];

// ─── Helper ───────────────────────────────────────────────────────────────────

function fmtTime(hhmm: string): string {
  const [h, m] = hhmm.split(":").map(Number);
  const ampm = h < 12 ? "AM" : "PM";
  const hour12 = h % 12 === 0 ? 12 : h % 12;
  return `${hour12}:${String(m).padStart(2, "0")} ${ampm}`;
}

function efficiencyColor(pct: number): string {
  if (pct >= 85) return "text-cyan-400";
  if (pct >= 80) return "text-amber-400";
  return "text-rose-400";
}

function adjustmentLabel(data: WeeklyProgressEntry[], week: number): string {
  const entry = data.find((d) => d.week === week);
  if (!entry) return "";
  if (entry.efficiency >= 85) return "Expand next week";
  if (entry.efficiency < 80) return "Contract next week";
  return "Maintain next week";
}

function adjustmentBadgeStyle(data: WeeklyProgressEntry[], week: number): string {
  const entry = data.find((d) => d.week === week);
  if (!entry) return "";
  if (entry.efficiency >= 85)
    return "border-cyan-500/40 text-cyan-400 bg-cyan-500/10";
  if (entry.efficiency < 80)
    return "border-rose-500/40 text-rose-400 bg-rose-500/10";
  return "border-amber-500/40 text-amber-400 bg-amber-500/10";
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function DisclaimerBanner() {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-xs text-amber-300">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>
        This is a <strong>wellness tool only</strong>, not a medical device. Sleep
        restriction therapy may be contraindicated for people with seizure disorders,
        bipolar disorder, untreated sleep apnea, or safety-critical occupations.
        Consult a licensed clinician before starting.
      </span>
    </div>
  );
}

interface CheckinRowProps {
  label: string;
  value: boolean | null;
  onChange: (v: boolean) => void;
}

function CheckinRow({ label, value, onChange }: CheckinRowProps) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-zinc-800 bg-zinc-900/60 px-4 py-3">
      <span className="text-sm text-zinc-200">{label}</span>
      <div className="flex gap-2">
        <button
          onClick={() => onChange(true)}
          className={`flex h-8 w-8 items-center justify-center rounded-full transition-colors ${
            value === true
              ? "bg-cyan-500/20 text-cyan-400"
              : "text-zinc-600 hover:text-zinc-400"
          }`}
          aria-label="Yes"
        >
          <CheckCircle className="h-5 w-5" />
        </button>
        <button
          onClick={() => onChange(false)}
          className={`flex h-8 w-8 items-center justify-center rounded-full transition-colors ${
            value === false
              ? "bg-rose-500/20 text-rose-400"
              : "text-zinc-600 hover:text-zinc-400"
          }`}
          aria-label="No"
        >
          <XCircle className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export default function CbtiModule() {
  // State: enrollment
  const [enrolled, setEnrolled] = useState(false);
  const [enrollment, setEnrollment] = useState<EnrollmentData | null>(null);

  // State: daily check-in
  const [checkin, setCheckin] = useState<DailyCheckin>({
    bedtimeAdherence: null,
    getUpIfAwake: null,
    bedForSleepOnly: null,
  });
  const [checkinSaved, setCheckinSaved] = useState(false);

  // Derive compliance score from current responses
  const complianceScore = useMemo(() => {
    const weights = { bedtimeAdherence: 0.4, getUpIfAwake: 0.35, bedForSleepOnly: 0.25 };
    let score = 0;
    for (const [key, weight] of Object.entries(weights)) {
      if (checkin[key as keyof DailyCheckin] === true) score += weight;
    }
    return Math.round(score * 100);
  }, [checkin]);

  // Weekly chart data (use mock for now)
  const weeklyData = enrollment ? MOCK_WEEKLY.slice(0, enrollment.currentWeek) : MOCK_WEEKLY;

  function handleEnroll() {
    // In production: call API to initialise the 6-week program from user's
    // last 7 days of healthSamples data. Here we use the mock.
    setEnrollment(MOCK_ENROLLMENT);
    setEnrolled(true);
  }

  function saveCheckin() {
    // In production: POST checkin responses + compliance score to API
    setCheckinSaved(true);
  }

  // ── Not enrolled ──────────────────────────────────────────────────────────
  if (!enrolled) {
    return (
      <div className="mx-auto max-w-lg space-y-5 px-4 py-6">
        <h1 className="text-xl font-semibold text-foreground">CBT-i Sleep Program</h1>
        <DisclaimerBanner />

        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <BedDouble className="h-5 w-5 text-indigo-400" />
              6-Week Sleep Restriction Program
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm text-zinc-300">
            <p>
              <strong>Sleep Restriction Therapy (SRT)</strong> is the most effective
              behavioral treatment for chronic insomnia. It works by temporarily
              limiting time in bed to match your actual sleep time, consolidating
              sleep and rebuilding the drive to sleep deeply.
            </p>

            <ul className="space-y-2 text-zinc-400">
              {[
                "Week 1: Baseline — initial sleep window set from your last 7 days",
                "Weeks 2–5: Adjust window weekly based on sleep efficiency",
                "Week 6: Final review — taper toward natural sleep schedule",
              ].map((item) => (
                <li key={item} className="flex gap-2">
                  <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-400" />
                  {item}
                </li>
              ))}
            </ul>

            <p className="text-xs text-zinc-500">
              Your initial sleep window will be calculated from your Apple Health
              sleep data. You'll receive a prescribed bedtime and wake time to follow
              strictly throughout the program.
            </p>

            <Button onClick={handleEnroll} className="w-full">
              Start 6-Week Program
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // ── Enrolled ──────────────────────────────────────────────────────────────
  const enr = enrollment!;
  const currentWeekData = weeklyData.find((d) => d.week === enr.currentWeek);
  const latestEfficiency = currentWeekData?.efficiency ?? 0;

  return (
    <div className="mx-auto max-w-lg space-y-5 px-4 py-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-foreground">CBT-i Sleep Program</h1>
        <Badge variant="outline" className="text-xs text-indigo-300 border-indigo-500/40">
          Week {enr.currentWeek} / 6
        </Badge>
      </div>

      {/* ── Evening reminder ─────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 rounded-xl border border-indigo-500/30 bg-indigo-500/10 px-4 py-3">
        <Moon className="h-5 w-5 shrink-0 text-indigo-400" />
        <div>
          <p className="text-[11px] uppercase tracking-wider text-indigo-400">
            Your bedtime tonight
          </p>
          <p className="text-2xl font-bold tabular-nums text-foreground">
            {fmtTime(enr.prescribedBedtime)}
          </p>
        </div>
        <div className="ml-auto text-right">
          <p className="text-[11px] uppercase tracking-wider text-zinc-500">Wake up</p>
          <p className="text-lg font-semibold tabular-nums text-zinc-300">
            {fmtTime(enr.prescribedWakeTime)}
          </p>
        </div>
      </div>

      {/* ── Current week dashboard ──────────────────────────────────────── */}
      <Card className="glass-card">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-sm font-medium text-zinc-400">
            <TrendingUp className="h-4 w-4" />
            Week {enr.currentWeek} at a Glance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-3 text-center">
            {/* Sleep window */}
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-2 py-3">
              <Clock className="mx-auto mb-1 h-4 w-4 text-zinc-500" />
              <p className="text-lg font-bold tabular-nums text-foreground">
                {enr.windowHours.toFixed(1)}h
              </p>
              <p className="text-[10px] text-zinc-500">Window</p>
            </div>

            {/* Sleep efficiency */}
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-2 py-3">
              <Moon className="mx-auto mb-1 h-4 w-4 text-zinc-500" />
              <p className={`text-lg font-bold tabular-nums ${efficiencyColor(latestEfficiency)}`}>
                {latestEfficiency}%
              </p>
              <p className="text-[10px] text-zinc-500">Efficiency</p>
            </div>

            {/* Next week adjustment */}
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-2 py-3">
              <Sun className="mx-auto mb-1 h-4 w-4 text-zinc-500" />
              <Badge
                variant="outline"
                className={`text-[10px] px-1.5 py-0.5 ${adjustmentBadgeStyle(weeklyData, enr.currentWeek)}`}
              >
                {adjustmentLabel(weeklyData, enr.currentWeek).split(" ")[0]}
              </Badge>
              <p className="mt-1 text-[10px] text-zinc-500">Next week</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ── Daily check-in ──────────────────────────────────────────────── */}
      <Card className="glass-card">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center justify-between text-sm font-medium">
            <span className="flex items-center gap-2 text-zinc-400">
              <CheckCircle className="h-4 w-4" />
              Daily Analysis
            </span>
            {checkin.bedtimeAdherence !== null && (
              <span className="text-xs text-zinc-500">
                Compliance:{" "}
                <span className={efficiencyColor(complianceScore)}>
                  {complianceScore}%
                </span>
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <CheckinRow
            label="Did you go to bed at your prescribed time?"
            value={checkin.bedtimeAdherence}
            onChange={(v) => setCheckin((c) => ({ ...c, bedtimeAdherence: v }))}
          />
          <CheckinRow
            label="Did you get up if awake > 20 min?"
            value={checkin.getUpIfAwake}
            onChange={(v) => setCheckin((c) => ({ ...c, getUpIfAwake: v }))}
          />
          <CheckinRow
            label="Did you use your bed only for sleep?"
            value={checkin.bedForSleepOnly}
            onChange={(v) => setCheckin((c) => ({ ...c, bedForSleepOnly: v }))}
          />

          {checkinSaved ? (
            <p className="pt-1 text-center text-xs text-cyan-400">
              Analysis saved. Keep it up!
            </p>
          ) : (
            <Button
              onClick={saveCheckin}
              variant="outline"
              size="sm"
              className="mt-2 w-full text-xs"
              disabled={
                checkin.bedtimeAdherence === null &&
                checkin.getUpIfAwake === null &&
                checkin.bedForSleepOnly === null
              }
            >
              Save Analysis
            </Button>
          )}
        </CardContent>
      </Card>

      {/* ── Progress chart ──────────────────────────────────────────────── */}
      <Card className="glass-card">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-sm font-medium text-zinc-400">
            <TrendingUp className="h-4 w-4" />
            6-Week Progress
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* Sleep efficiency trend */}
          <div>
            <p className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">
              Sleep efficiency %
            </p>
            <ResponsiveContainer width="100%" height={120}>
              <LineChart data={weeklyData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="week"
                  tickFormatter={(v) => `W${v}`}
                  tick={{ fontSize: 10, fill: "#71717a" }}
                />
                <YAxis
                  domain={[50, 100]}
                  tick={{ fontSize: 10, fill: "#71717a" }}
                  tickFormatter={(v) => `${v}%`}
                />
                <Tooltip
                  contentStyle={{
                    background: "#18181b",
                    border: "1px solid #3f3f46",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(v: number) => [`${v}%`, "Efficiency"]}
                  labelFormatter={(l) => `Week ${l}`}
                />
                <ReferenceLine y={85} stroke="#0891b2" strokeDasharray="4 3" strokeOpacity={0.5} />
                <ReferenceLine y={80} stroke="#d4a017" strokeDasharray="4 3" strokeOpacity={0.4} />
                <Line
                  type="monotone"
                  dataKey="efficiency"
                  stroke="#818cf8"
                  strokeWidth={2}
                  dot={{ r: 3, fill: "#818cf8" }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
            <p className="mt-1 text-[10px] text-zinc-600">
              Green line = 85% expand threshold · Amber = 80% maintain threshold
            </p>
          </div>

          {/* Sleep window bars */}
          <div>
            <p className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">
              Sleep window (hours)
            </p>
            <ResponsiveContainer width="100%" height={100}>
              <BarChart data={weeklyData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="week"
                  tickFormatter={(v) => `W${v}`}
                  tick={{ fontSize: 10, fill: "#71717a" }}
                />
                <YAxis domain={[0, 9]} tick={{ fontSize: 10, fill: "#71717a" }} />
                <Tooltip
                  contentStyle={{
                    background: "#18181b",
                    border: "1px solid #3f3f46",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(v: number) => [`${v}h`, "Window"]}
                  labelFormatter={(l) => `Week ${l}`}
                />
                <Bar dataKey="windowHours" fill="#6366f1" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Compliance per week */}
          <div>
            <p className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">
              Compliance %
            </p>
            <ResponsiveContainer width="100%" height={100}>
              <BarChart data={weeklyData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="week"
                  tickFormatter={(v) => `W${v}`}
                  tick={{ fontSize: 10, fill: "#71717a" }}
                />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#71717a" }} tickFormatter={(v) => `${v}%`} />
                <Tooltip
                  contentStyle={{
                    background: "#18181b",
                    border: "1px solid #3f3f46",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(v: number) => [`${v}%`, "Compliance"]}
                  labelFormatter={(l) => `Week ${l}`}
                />
                <Bar dataKey="compliance" fill="#0891b2" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
