import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  LineChart,
  Line,
  Area,
  AreaChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Heart,
  Calendar,
  Smile,
  Frown,
  Meh,
  Laugh,
  Angry,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { motion } from "framer-motion";
import { useToast } from "@/hooks/use-toast";

/* ---------- types ---------- */

interface CycleEntry {
  id: string;
  userId: string | null;
  date: string;
  flowLevel: string | null;
  symptoms: string[] | null;
  phase: string | null;
  contraception: string | null;
  basalTemp: string | null;
  notes: string | null;
}

interface PhaseInfo {
  currentPhase: string;
  dayOfCycle: number;
  avgCycleLength: number;
  lastPeriodStart: string | null;
  nextPeriodDate: string | null;
  periodStartCount: number;
}

interface MoodLog {
  id: string;
  userId: string | null;
  moodScore: string;
  energyLevel: string | null;
  notes: string | null;
  loggedAt: string | null;
}

/* ---------- constants ---------- */

const FLOW_LEVELS = [
  { value: "none", label: "None", color: "bg-muted" },
  { value: "light", label: "Light", color: "bg-pink-300" },
  { value: "medium", label: "Medium", color: "bg-pink-500" },
  { value: "heavy", label: "Heavy", color: "bg-red-600" },
];

const SYMPTOMS = [
  "cramps", "bloating", "headache", "mood_swings", "fatigue",
  "acne", "cravings", "insomnia", "nausea", "breast_tenderness",
  "back_pain", "joint_pain", "anxiety", "irritability", "sadness",
  "brain_fog", "hot_flashes", "dizziness", "constipation", "diarrhea",
];

const PHASE_INFO: Record<string, { label: string; color: string; bg: string; description: string }> = {
  menstrual: { label: "Menstrual", color: "text-rose-400", bg: "bg-rose-500/10 border-rose-500/20", description: "Day 1-5 of your cycle" },
  follicular: { label: "Follicular", color: "text-ndw-recovery", bg: "bg-ndw-recovery/10 border-ndw-recovery/20", description: "Estrogen rising, energy increasing" },
  ovulatory: { label: "Ovulatory", color: "text-ndw-stress", bg: "bg-ndw-stress/10 border-ndw-stress/20", description: "Peak fertility window" },
  luteal: { label: "Luteal", color: "text-ndw-sleep", bg: "bg-ndw-sleep/10 border-ndw-sleep/20", description: "Progesterone dominant, winding down" },
  late: { label: "Late", color: "text-orange-400", bg: "bg-orange-500/10 border-orange-500/20", description: "Period may be coming soon" },
  unknown: { label: "Unknown", color: "text-muted-foreground", bg: "bg-muted border-border", description: "Log more data to predict phases" },
};

const MOOD_FACES = [
  { min: 1, max: 2, icon: Angry, label: "Awful" },
  { min: 3, max: 4, icon: Frown, label: "Bad" },
  { min: 5, max: 6, icon: Meh, label: "Okay" },
  { min: 7, max: 8, icon: Smile, label: "Good" },
  { min: 9, max: 10, icon: Laugh, label: "Great" },
];

function getMoodFace(score: number) {
  return MOOD_FACES.find(f => score >= f.min && score <= f.max) ?? MOOD_FACES[2];
}

function formatSymptom(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

/* ---------- helpers ---------- */

function getToday(): string {
  return new Date().toISOString().slice(0, 10);
}

function getMonthDays(year: number, month: number): { date: string; day: number; inMonth: boolean }[] {
  const firstDay = new Date(year, month, 1);
  const startDow = firstDay.getDay(); // 0=Sun
  const daysInMonth = new Date(year, month + 1, 0).getDate();

  const days: { date: string; day: number; inMonth: boolean }[] = [];

  // Padding days from previous month
  for (let i = startDow - 1; i >= 0; i--) {
    const d = new Date(year, month, -i);
    days.push({
      date: d.toISOString().slice(0, 10),
      day: d.getDate(),
      inMonth: false,
    });
  }

  // Current month days
  for (let i = 1; i <= daysInMonth; i++) {
    const d = new Date(year, month, i);
    days.push({
      date: d.toISOString().slice(0, 10),
      day: i,
      inMonth: true,
    });
  }

  // Padding days from next month to fill the grid
  while (days.length % 7 !== 0) {
    const d = new Date(year, month + 1, days.length - startDow - daysInMonth + 1);
    days.push({
      date: d.toISOString().slice(0, 10),
      day: d.getDate(),
      inMonth: false,
    });
  }

  return days;
}

/* ========== Cycle Tab ========== */

function CycleTab() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const today = getToday();

  const [viewDate, setViewDate] = useState(() => {
    const now = new Date();
    return { year: now.getFullYear(), month: now.getMonth() };
  });
  const [logDialogOpen, setLogDialogOpen] = useState(false);
  const [selectedDate, setSelectedDate] = useState(today);
  const [flowLevel, setFlowLevel] = useState("none");
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [cycleNotes, setCycleNotes] = useState("");

  // Fetch cycle data
  const { data: cycleData = [] } = useQuery<CycleEntry[]>({
    queryKey: [`/api/cycle/${user?.id}?days=365`],
    enabled: !!user?.id,
    retry: false,
    staleTime: 30_000,
  });

  // Fetch phase prediction
  const { data: phaseInfo } = useQuery<PhaseInfo>({
    queryKey: [`/api/cycle/${user?.id}/phase`],
    enabled: !!user?.id,
    retry: false,
    staleTime: 60_000,
  });

  // Map cycle data by date
  const cycleByDate = useMemo(() => {
    const map = new Map<string, CycleEntry>();
    for (const entry of cycleData) {
      map.set(entry.date, entry);
    }
    return map;
  }, [cycleData]);

  // Log cycle mutation
  const logCycleMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/cycle", {
        date: selectedDate,
        flowLevel,
        symptoms: selectedSymptoms.length > 0 ? selectedSymptoms : null,
        notes: cycleNotes || null,
      });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/api/cycle/${user?.id}?days=365`] });
      queryClient.invalidateQueries({ queryKey: [`/api/cycle/${user?.id}/phase`] });
      setLogDialogOpen(false);
      toast({ title: "Cycle data logged" });
    },
    onError: (err: Error) => {
      toast({ title: "Error", description: err.message, variant: "destructive" });
    },
  });

  function openLogDialog(date: string) {
    setSelectedDate(date);
    const existing = cycleByDate.get(date);
    if (existing) {
      setFlowLevel(existing.flowLevel ?? "none");
      setSelectedSymptoms(existing.symptoms ?? []);
      setCycleNotes(existing.notes ?? "");
    } else {
      setFlowLevel("none");
      setSelectedSymptoms([]);
      setCycleNotes("");
    }
    setLogDialogOpen(true);
  }

  function toggleSymptom(s: string) {
    setSelectedSymptoms(prev =>
      prev.includes(s) ? prev.filter(x => x !== s) : [...prev, s]
    );
  }

  const monthDays = getMonthDays(viewDate.year, viewDate.month);
  const monthLabel = new Date(viewDate.year, viewDate.month).toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
  });

  const phase = phaseInfo ? PHASE_INFO[phaseInfo.currentPhase] ?? PHASE_INFO.unknown : PHASE_INFO.unknown;

  return (
    <div className="space-y-5">
      {/* Phase indicator */}
      {phaseInfo && phaseInfo.currentPhase !== "unknown" && (
        <motion.div
          className={`rounded-xl border p-4 ${phase.bg}`}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold">Current Phase</p>
              <p className={`text-lg font-bold ${phase.color}`}>{phase.label}</p>
              <p className="text-xs text-muted-foreground">{phase.description}</p>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold text-foreground">Day {phaseInfo.dayOfCycle}</p>
              <p className="text-[10px] text-muted-foreground">
                Avg cycle: {phaseInfo.avgCycleLength} days
              </p>
              {phaseInfo.nextPeriodDate && (
                <p className="text-[10px] text-muted-foreground">
                  Next period: {new Date(phaseInfo.nextPeriodDate + "T12:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                </p>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* Calendar */}
      <div className="rounded-xl border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-3">
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0"
            onClick={() => {
              setViewDate(prev => {
                const d = new Date(prev.year, prev.month - 1);
                return { year: d.getFullYear(), month: d.getMonth() };
              });
            }}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <p className="text-sm font-semibold">{monthLabel}</p>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0"
            onClick={() => {
              setViewDate(prev => {
                const d = new Date(prev.year, prev.month + 1);
                return { year: d.getFullYear(), month: d.getMonth() };
              });
            }}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        {/* Day headers */}
        <div className="grid grid-cols-7 gap-1 mb-1">
          {["S", "M", "T", "W", "T", "F", "S"].map((d, i) => (
            <div key={i} className="text-center text-[10px] text-muted-foreground font-medium py-1">{d}</div>
          ))}
        </div>

        {/* Day cells */}
        <div className="grid grid-cols-7 gap-1">
          {monthDays.map(({ date, day, inMonth }) => {
            const entry = cycleByDate.get(date);
            const flow = entry?.flowLevel;
            const isToday = date === today;

            let bgColor = "";
            if (flow === "heavy") bgColor = "bg-red-600/80";
            else if (flow === "medium") bgColor = "bg-pink-500/70";
            else if (flow === "light") bgColor = "bg-pink-300/60";

            return (
              <button
                key={date}
                onClick={() => openLogDialog(date)}
                className={`aspect-square rounded-lg flex items-center justify-center text-xs font-medium transition-colors
                  ${!inMonth ? "opacity-30" : ""}
                  ${bgColor || "hover:bg-muted/50"}
                  ${isToday && !bgColor ? "ring-1 ring-primary" : ""}
                  ${flow && flow !== "none" ? "text-white" : "text-foreground"}
                `}
              >
                {day}
              </button>
            );
          })}
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 mt-3 justify-center">
          {FLOW_LEVELS.filter(f => f.value !== "none").map(f => (
            <div key={f.value} className="flex items-center gap-1">
              <div className={`w-2.5 h-2.5 rounded-full ${f.color}`} />
              <span className="text-[9px] text-muted-foreground">{f.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Quick log button for today */}
      <Button
        variant="outline"
        className="w-full gap-2"
        onClick={() => openLogDialog(today)}
      >
        <Calendar className="h-4 w-4" />
        Log Today
      </Button>

      {/* Log dialog */}
      <Dialog open={logDialogOpen} onOpenChange={setLogDialogOpen}>
        <DialogContent className="max-w-sm max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {new Date(selectedDate + "T12:00:00").toLocaleDateString(undefined, {
                weekday: "short",
                month: "short",
                day: "numeric",
              })}
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            {/* Flow level */}
            <div>
              <Label className="text-xs font-medium">Flow Level</Label>
              <div className="grid grid-cols-4 gap-2 mt-2">
                {FLOW_LEVELS.map(f => (
                  <button
                    key={f.value}
                    onClick={() => setFlowLevel(f.value)}
                    className={`py-2 rounded-lg text-xs font-medium border transition-colors ${
                      flowLevel === f.value
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border hover:bg-muted/50"
                    }`}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Symptoms */}
            <div>
              <Label className="text-xs font-medium">Symptoms</Label>
              <div className="flex flex-wrap gap-1.5 mt-2">
                {SYMPTOMS.map(s => (
                  <button
                    key={s}
                    onClick={() => toggleSymptom(s)}
                    className={`px-2 py-1 rounded-full text-[10px] font-medium border transition-colors ${
                      selectedSymptoms.includes(s)
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border text-muted-foreground hover:bg-muted/50"
                    }`}
                  >
                    {formatSymptom(s)}
                  </button>
                ))}
              </div>
            </div>

            {/* Notes */}
            <div>
              <Label className="text-xs font-medium">Notes</Label>
              <Textarea
                value={cycleNotes}
                onChange={e => setCycleNotes(e.target.value)}
                placeholder="How are you feeling?"
                className="mt-1 text-sm"
                rows={2}
              />
            </div>

            <Button
              onClick={() => logCycleMutation.mutate()}
              disabled={logCycleMutation.isPending}
              className="w-full"
            >
              Save
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

/* ========== Mood Tab ========== */

function MoodTab() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const [moodScore, setMoodScore] = useState(5);
  const [energyLevel, setEnergyLevel] = useState(5);
  const [moodNotes, setMoodNotes] = useState("");

  // Fetch mood logs
  const { data: moodLogs = [] } = useQuery<MoodLog[]>({
    queryKey: [`/api/mood/${user?.id}?days=30`],
    enabled: !!user?.id,
    retry: false,
    staleTime: 30_000,
  });

  // Log mood mutation
  const logMoodMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/mood", {
        moodScore,
        energyLevel,
        notes: moodNotes || null,
      });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/api/mood/${user?.id}?days=30`] });
      setMoodNotes("");
      toast({ title: "Mood logged" });
    },
    onError: (err: Error) => {
      toast({ title: "Error", description: err.message, variant: "destructive" });
    },
  });

  // Chart data
  const chartData = useMemo(() => {
    return [...moodLogs]
      .reverse()
      .map(log => ({
        date: log.loggedAt
          ? new Date(log.loggedAt).toLocaleDateString(undefined, { month: "short", day: "numeric" })
          : "",
        mood: parseFloat(log.moodScore),
        energy: log.energyLevel ? parseFloat(log.energyLevel) : null,
      }));
  }, [moodLogs]);

  // Mood by time-of-day pattern
  const timePattern = useMemo(() => {
    const buckets: Record<string, { total: number; count: number }> = {
      Morning: { total: 0, count: 0 },
      Afternoon: { total: 0, count: 0 },
      Evening: { total: 0, count: 0 },
      Night: { total: 0, count: 0 },
    };

    for (const log of moodLogs) {
      if (!log.loggedAt) continue;
      const hour = new Date(log.loggedAt).getHours();
      const score = parseFloat(log.moodScore);
      if (hour >= 5 && hour < 12) {
        buckets.Morning.total += score;
        buckets.Morning.count++;
      } else if (hour >= 12 && hour < 17) {
        buckets.Afternoon.total += score;
        buckets.Afternoon.count++;
      } else if (hour >= 17 && hour < 21) {
        buckets.Evening.total += score;
        buckets.Evening.count++;
      } else {
        buckets.Night.total += score;
        buckets.Night.count++;
      }
    }

    return Object.entries(buckets)
      .filter(([, v]) => v.count > 0)
      .map(([label, v]) => ({
        label,
        avg: Math.round((v.total / v.count) * 10) / 10,
        count: v.count,
      }));
  }, [moodLogs]);

  const face = getMoodFace(moodScore);
  const FaceIcon = face.icon;

  return (
    <div className="space-y-5">
      {/* Mood input card */}
      <motion.div
        className="rounded-xl border border-border bg-card p-4 space-y-4"
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
      >
        <p className="text-sm font-semibold text-foreground">How are you feeling?</p>

        {/* Mood slider with face */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="text-xs text-muted-foreground">Mood</Label>
            <div className="flex items-center gap-1.5">
              <motion.div
                key={face.label}
                initial={{ scale: 0.5, rotate: -10 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ type: "spring", stiffness: 400, damping: 15 }}
              >
                <FaceIcon className="h-6 w-6 text-primary" />
              </motion.div>
              <span className="text-sm font-bold text-foreground">{moodScore}</span>
              <span className="text-xs text-muted-foreground">/ 10</span>
              <span className="text-[10px] text-muted-foreground ml-1">({face.label})</span>
            </div>
          </div>
          <Slider
            value={[moodScore]}
            onValueChange={([v]) => setMoodScore(v)}
            min={1}
            max={10}
            step={1}
            className="py-2"
          />
          <div className="flex justify-between text-[9px] text-muted-foreground/60">
            <span>Awful</span>
            <span>Great</span>
          </div>
        </div>

        {/* Energy slider */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs text-muted-foreground">Energy</Label>
            <span className="text-sm font-bold">{energyLevel}<span className="text-xs text-muted-foreground font-normal"> / 10</span></span>
          </div>
          <Slider
            value={[energyLevel]}
            onValueChange={([v]) => setEnergyLevel(v)}
            min={1}
            max={10}
            step={1}
            className="py-2"
          />
          <div className="flex justify-between text-[9px] text-muted-foreground/60">
            <span>Exhausted</span>
            <span>Energized</span>
          </div>
        </div>

        {/* Notes */}
        <div>
          <Label className="text-xs text-muted-foreground">Notes (optional)</Label>
          <Textarea
            value={moodNotes}
            onChange={e => setMoodNotes(e.target.value)}
            placeholder="What's on your mind?"
            className="mt-1 text-sm"
            rows={2}
          />
        </div>

        <Button
          onClick={() => logMoodMutation.mutate()}
          disabled={logMoodMutation.isPending}
          className="w-full"
        >
          Log Mood
        </Button>
      </motion.div>

      {/* Mood over time chart */}
      {chartData.length > 2 && (
        <motion.div
          className="rounded-xl border border-border bg-card p-4"
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, delay: 0.1 }}
        >
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
            Mood over time
          </p>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="moodGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0891b2" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#0891b2" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
                  tickLine={false}
                  axisLine={false}
                  interval="preserveStartEnd"
                />
                <YAxis
                  domain={[1, 10]}
                  tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
                  tickLine={false}
                  axisLine={false}
                  width={24}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--card)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    fontSize: 11,
                    color: "var(--foreground)",
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="mood"
                  stroke="#0891b2"
                  fill="url(#moodGradient)"
                  strokeWidth={2}
                  dot={{ r: 2, fill: "#0891b2" }}
                  name="Mood"
                />
                <Area
                  type="monotone"
                  dataKey="energy"
                  stroke="#d4a017"
                  fill="transparent"
                  strokeWidth={1.5}
                  dot={{ r: 1.5 }}
                  strokeDasharray="4 2"
                  name="Energy"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      )}

      {/* Mood by time of day */}
      {timePattern.length > 0 && (
        <div className="rounded-xl border border-border bg-card p-4">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
            Mood by time of day
          </p>
          <div className="grid grid-cols-4 gap-2">
            {["Morning", "Afternoon", "Evening", "Night"].map(label => {
              const bucket = timePattern.find(b => b.label === label);
              return (
                <div key={label} className="text-center">
                  <p className="text-[10px] text-muted-foreground mb-1">{label}</p>
                  {bucket ? (
                    <>
                      <p className="text-lg font-bold">{bucket.avg}</p>
                      <p className="text-[9px] text-muted-foreground">{bucket.count} logs</p>
                    </>
                  ) : (
                    <p className="text-sm text-muted-foreground/40">--</p>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Recent mood entries */}
      {moodLogs.length > 0 && (
        <div className="rounded-xl border border-border bg-card p-4">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
            Recent entries
          </p>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {moodLogs.slice(0, 10).map(log => {
              const f = getMoodFace(parseFloat(log.moodScore));
              const Icon = f.icon;
              return (
                <div key={log.id} className="flex items-center gap-3 py-1.5">
                  <Icon className="h-4 w-4 text-primary shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">
                        Mood {parseFloat(log.moodScore)}
                        {log.energyLevel && <span className="text-muted-foreground font-normal"> / Energy {parseFloat(log.energyLevel)}</span>}
                      </span>
                    </div>
                    {log.notes && (
                      <p className="text-[10px] text-muted-foreground truncate">{log.notes}</p>
                    )}
                  </div>
                  <span className="text-[10px] text-muted-foreground shrink-0">
                    {log.loggedAt
                      ? new Date(log.loggedAt).toLocaleDateString(undefined, { month: "short", day: "numeric" })
                      : ""}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

/* ========== Main Wellness Page ========== */

export default function Wellness() {
  const [tab, setTab] = useState("cycle");

  return (
    <div className="max-w-lg mx-auto px-4 py-6 space-y-5">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <h1 className="text-xl font-bold text-foreground flex items-center gap-2">
          <Heart className="h-5 w-5 text-ndw-energy" />
          Wellness
        </h1>
        <p className="text-xs text-muted-foreground mt-0.5">
          Cycle tracking and mood logging
        </p>
      </motion.div>

      {/* Tabs */}
      <Tabs value={tab} onValueChange={setTab} className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="cycle" className="gap-1.5">
            <Calendar className="h-3.5 w-3.5" />
            Cycle
          </TabsTrigger>
          <TabsTrigger value="mood" className="gap-1.5">
            <Smile className="h-3.5 w-3.5" />
            Mood
          </TabsTrigger>
        </TabsList>

        <TabsContent value="cycle" className="mt-4">
          <CycleTab />
        </TabsContent>

        <TabsContent value="mood" className="mt-4">
          <MoodTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
