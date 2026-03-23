import { useState, useEffect, useMemo, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { syncMoodLogToML } from "@/lib/ml-api";
import { useAuth } from "@/hooks/use-auth";
import { getCycleData as sbGetCycleData, getMoodLogs as sbGetMoodLogs, saveCycleData as sbSaveCycleData, saveMoodLog as sbSaveMoodLog, sbGetGeneric, sbGetSetting, sbSaveGeneric } from "../lib/supabase-store";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
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
  Settings2,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
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
  { value: "heavy", label: "Heavy", color: "bg-rose-600" },
];

const SYMPTOMS = [
  "cramps", "bloating", "headache", "mood_swings", "fatigue",
  "acne", "cravings", "insomnia", "nausea", "breast_tenderness",
  "back_pain", "joint_pain", "anxiety", "irritability", "sadness",
  "brain_fog", "hot_flashes", "dizziness", "constipation", "diarrhea",
];

const SEVERITY_LABELS = ["Mild", "Moderate", "Severe"] as const;

const HORMONE_PHASES: Record<string, { hormones: string; emoji: string; description: string }> = {
  menstrual: {
    hormones: "Estrogen + Progesterone low",
    emoji: "🩸",
    description: "Energy may be lower. Rest and gentle movement recommended.",
  },
  follicular: {
    hormones: "Estrogen rising",
    emoji: "🌱",
    description: "Energy increasing. Great time for new activities and social plans.",
  },
  ovulatory: {
    hormones: "Estrogen peak, LH surge",
    emoji: "🌸",
    description: "Peak energy and confidence. Fertility window open.",
  },
  luteal: {
    hormones: "Progesterone high, then drops",
    emoji: "🍂",
    description: "Winding down. Cravings may increase. Self-care is key.",
  },
  late: {
    hormones: "Progesterone dropping",
    emoji: "🌙",
    description: "PMS symptoms may appear. Be gentle with yourself.",
  },
  unknown: {
    hormones: "Log more data to determine",
    emoji: "🔮",
    description: "Track a few cycles to get hormone phase insights.",
  },
};

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

/* ---------- localStorage cycle data ---------- */

interface LocalCycleData {
  lastPeriodStart: string; // ISO date string
  cycleLength: number;
  periodLength: number;
}

const CYCLE_STORAGE_KEY = "ndw_cycle_data";

function getLocalCycleData(): LocalCycleData | null {
  try {
    const parsed = sbGetGeneric<LocalCycleData>(CYCLE_STORAGE_KEY);
    if (parsed && parsed.lastPeriodStart && parsed.cycleLength && parsed.periodLength) {
      return parsed;
    }
    return null;
  } catch {
    return null;
  }
}

function setLocalCycleData(data: LocalCycleData): void {
  sbSaveGeneric(CYCLE_STORAGE_KEY, data);
  // Also persist to Supabase (fire-and-forget)
  sbSaveCycleData("local", {
    last_period_start: data.lastPeriodStart,
    cycle_length: data.cycleLength,
    period_length: data.periodLength,
  }).catch(() => {});
}

/* ---------- cycle phase computation ---------- */

const CYCLE_PHASES = [
  { key: "menstrual", label: "Menstrual", color: "#e879a8" },
  { key: "follicular", label: "Follicular", color: "#0891b2" },
  { key: "ovulation", label: "Ovulation", color: "#d4a017" },
  { key: "luteal", label: "Luteal", color: "#7c3aed" },
] as const;

type CyclePhaseKey = typeof CYCLE_PHASES[number]["key"];

interface ComputedCycleInfo {
  currentPhase: CyclePhaseKey;
  dayOfCycle: number;
  cycleLength: number;
  periodLength: number;
  nextPeriodDate: string;
  phaseRanges: { key: CyclePhaseKey; startDay: number; endDay: number }[];
}

function computeCycleInfo(data: LocalCycleData): ComputedCycleInfo {
  const { lastPeriodStart, cycleLength, periodLength } = data;
  const startDate = new Date(lastPeriodStart + "T12:00:00");
  const now = new Date();
  const diffMs = now.getTime() - startDate.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  const dayOfCycle = ((diffDays % cycleLength) + cycleLength) % cycleLength + 1;

  const ovulationDay = cycleLength - 14;
  const follicularEnd = ovulationDay - 1;
  const ovulationEnd = ovulationDay + 1;

  const phaseRanges: ComputedCycleInfo["phaseRanges"] = [
    { key: "menstrual", startDay: 1, endDay: periodLength },
    { key: "follicular", startDay: periodLength + 1, endDay: follicularEnd },
    { key: "ovulation", startDay: ovulationDay, endDay: ovulationEnd },
    { key: "luteal", startDay: ovulationEnd + 1, endDay: cycleLength },
  ];

  let currentPhase: CyclePhaseKey = "luteal";
  if (dayOfCycle <= periodLength) {
    currentPhase = "menstrual";
  } else if (dayOfCycle <= follicularEnd) {
    currentPhase = "follicular";
  } else if (dayOfCycle <= ovulationEnd) {
    currentPhase = "ovulation";
  } else {
    currentPhase = "luteal";
  }

  // Compute next period date
  const daysUntilNextPeriod = cycleLength - dayOfCycle + 1;
  const nextPeriod = new Date(now);
  nextPeriod.setDate(nextPeriod.getDate() + daysUntilNextPeriod);
  const nextPeriodDate = nextPeriod.toISOString().slice(0, 10);

  return { currentPhase, dayOfCycle, cycleLength, periodLength, nextPeriodDate, phaseRanges };
}

/* ---------- SVG Cycle Wheel ---------- */

function CycleWheel({ cycleInfo }: { cycleInfo: ComputedCycleInfo }) {
  const size = 220;
  const cx = size / 2;
  const cy = size / 2;
  const outerR = 90;
  const innerR = 58;
  const { cycleLength, dayOfCycle, phaseRanges, currentPhase } = cycleInfo;

  function dayToAngle(day: number): number {
    return ((day - 1) / cycleLength) * 360 - 90;
  }

  function polarToCartesian(centerX: number, centerY: number, radius: number, angleDeg: number) {
    const rad = (angleDeg * Math.PI) / 180;
    return { x: centerX + radius * Math.cos(rad), y: centerY + radius * Math.sin(rad) };
  }

  function arcPath(startAngle: number, endAngle: number, rOuter: number, rInner: number): string {
    let sweep = endAngle - startAngle;
    if (sweep <= 0) sweep += 360;
    const largeArc = sweep > 180 ? 1 : 0;

    const outerStart = polarToCartesian(cx, cy, rOuter, startAngle);
    const outerEnd = polarToCartesian(cx, cy, rOuter, endAngle);
    const innerStart = polarToCartesian(cx, cy, rInner, endAngle);
    const innerEnd = polarToCartesian(cx, cy, rInner, startAngle);

    return [
      `M ${outerStart.x} ${outerStart.y}`,
      `A ${rOuter} ${rOuter} 0 ${largeArc} 1 ${outerEnd.x} ${outerEnd.y}`,
      `L ${innerStart.x} ${innerStart.y}`,
      `A ${rInner} ${rInner} 0 ${largeArc} 0 ${innerEnd.x} ${innerEnd.y}`,
      "Z",
    ].join(" ");
  }

  // Current day marker position
  const markerAngle = dayToAngle(dayOfCycle);
  const markerPos = polarToCartesian(cx, cy, outerR + 8, markerAngle);

  const phaseColorMap: Record<string, string> = {
    menstrual: "#e879a8",
    follicular: "#0891b2",
    ovulation: "#d4a017",
    luteal: "#7c3aed",
  };

  const currentPhaseData = CYCLE_PHASES.find(p => p.key === currentPhase);

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Phase arcs */}
        {phaseRanges.map((range) => {
          const startAngle = dayToAngle(range.startDay);
          const endAngle = dayToAngle(range.endDay + 1);
          const isActive = range.key === currentPhase;
          const color = phaseColorMap[range.key];
          return (
            <motion.path
              key={range.key}
              d={arcPath(startAngle, endAngle, outerR, innerR)}
              fill={color}
              opacity={isActive ? 1 : 0.25}
              stroke="var(--background)"
              strokeWidth={2.5}
              initial={{ opacity: 0 }}
              animate={{ opacity: isActive ? 1 : 0.25 }}
              transition={{ duration: 0.5 }}
            />
          );
        })}

        {/* Phase labels around the wheel */}
        {phaseRanges.map((range) => {
          const midDay = (range.startDay + range.endDay) / 2;
          const labelAngle = dayToAngle(midDay);
          const labelPos = polarToCartesian(cx, cy, (outerR + innerR) / 2, labelAngle);
          const isActive = range.key === currentPhase;
          return (
            <text
              key={`label-${range.key}`}
              x={labelPos.x}
              y={labelPos.y}
              textAnchor="middle"
              dominantBaseline="central"
              fill={isActive ? "#fff" : "var(--muted-foreground)"}
              fontSize={isActive ? 9 : 8}
              fontWeight={isActive ? 700 : 400}
              className="select-none pointer-events-none"
            >
              {CYCLE_PHASES.find(p => p.key === range.key)?.label ?? ""}
            </text>
          );
        })}

        {/* Current day marker dot */}
        <motion.circle
          cx={markerPos.x}
          cy={markerPos.y}
          r={5}
          fill={phaseColorMap[currentPhase]}
          stroke="var(--background)"
          strokeWidth={2}
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", stiffness: 300, damping: 15, delay: 0.3 }}
        />

        {/* Center text */}
        <text x={cx} y={cy - 10} textAnchor="middle" fill="var(--foreground)" fontSize={22} fontWeight={700}>
          Day {dayOfCycle}
        </text>
        <text x={cx} y={cy + 10} textAnchor="middle" fill="var(--muted-foreground)" fontSize={11}>
          of {cycleLength}
        </text>
      </svg>

      {/* Current phase label below wheel */}
      <motion.div
        className="mt-2 text-center"
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.3 }}
      >
        <span
          className="inline-block px-3 py-1 rounded-full text-xs font-semibold text-white"
          style={{ backgroundColor: currentPhaseData?.color }}
        >
          {currentPhaseData?.label} Phase
        </span>
      </motion.div>
    </div>
  );
}

/* ---------- Cycle Setup Prompt ---------- */

function CycleSetupPrompt({ onComplete, initialData }: { onComplete: (data: LocalCycleData) => void; initialData?: LocalCycleData | null }) {
  const [lastPeriodDate, setLastPeriodDate] = useState(initialData?.lastPeriodStart ?? "");
  const [cycleLengthStr, setCycleLengthStr] = useState(String(initialData?.cycleLength ?? 28));
  const [periodLengthStr, setPeriodLengthStr] = useState(String(initialData?.periodLength ?? 5));
  const { toast } = useToast();

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!lastPeriodDate) {
      toast({ title: "Please enter your last period start date", variant: "destructive" });
      return;
    }
    const cycleLength = parseInt(cycleLengthStr) || 28;
    const periodLength = parseInt(periodLengthStr) || 5;
    if (cycleLength < 20 || cycleLength > 45) {
      toast({ title: "Cycle length should be between 20 and 45 days", variant: "destructive" });
      return;
    }
    if (periodLength < 1 || periodLength > 10) {
      toast({ title: "Period length should be between 1 and 10 days", variant: "destructive" });
      return;
    }
    const data: LocalCycleData = {
      lastPeriodStart: lastPeriodDate,
      cycleLength,
      periodLength,
    };
    setLocalCycleData(data);
    onComplete(data);
    toast({ title: "Cycle data saved" });
  }

  return (
    <motion.div
      className="rounded-xl border border-border bg-card p-5 space-y-5" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div>
        <h3 className="text-sm font-bold text-foreground">Set Up Menstrual Cycle Tracking</h3>
        <p className="text-xs text-muted-foreground mt-1">
          Enter your cycle details to get phase predictions, next period estimates, and symptom tracking.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <Label htmlFor="lastPeriodDate" className="text-xs font-medium">
            When did your last period start?
          </Label>
          <Input
            id="lastPeriodDate"
            type="date"
            value={lastPeriodDate}
            onChange={e => setLastPeriodDate(e.target.value)}
            max={getToday()}
            className="mt-1"
          />
        </div>

        <div>
          <Label htmlFor="cycleLength" className="text-xs font-medium">
            Average cycle length (days)
          </Label>
          <div className="flex items-center gap-2 mt-1">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 w-9 p-0 shrink-0"
              onClick={() => {
                const v = Math.max(20, (parseInt(cycleLengthStr) || 28) - 1);
                setCycleLengthStr(String(v));
              }}
            >
              -
            </Button>
            <Input
              id="cycleLength"
              type="number"
              min={20}
              max={45}
              value={cycleLengthStr}
              onChange={e => setCycleLengthStr(e.target.value)}
              onBlur={() => {
                const v = parseInt(cycleLengthStr);
                if (isNaN(v)) setCycleLengthStr("28");
                else setCycleLengthStr(String(Math.max(20, Math.min(45, v))));
              }}
              className="text-center"
            />
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 w-9 p-0 shrink-0"
              onClick={() => {
                const v = Math.min(45, (parseInt(cycleLengthStr) || 28) + 1);
                setCycleLengthStr(String(v));
              }}
            >
              +
            </Button>
          </div>
          <p className="text-[10px] text-muted-foreground mt-1">Most cycles are 24-35 days. Default: 28.</p>
        </div>

        <div>
          <Label htmlFor="periodLength" className="text-xs font-medium">
            Average period length (days)
          </Label>
          <div className="flex items-center gap-2 mt-1">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 w-9 p-0 shrink-0"
              onClick={() => {
                const v = Math.max(1, (parseInt(periodLengthStr) || 5) - 1);
                setPeriodLengthStr(String(v));
              }}
            >
              -
            </Button>
            <Input
              id="periodLength"
              type="number"
              min={1}
              max={10}
              value={periodLengthStr}
              onChange={e => setPeriodLengthStr(e.target.value)}
              onBlur={() => {
                const v = parseInt(periodLengthStr);
                if (isNaN(v)) setPeriodLengthStr("5");
                else setPeriodLengthStr(String(Math.max(1, Math.min(10, v))));
              }}
              className="text-center"
            />
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 w-9 p-0 shrink-0"
              onClick={() => {
                const v = Math.min(10, (parseInt(periodLengthStr) || 5) + 1);
                setPeriodLengthStr(String(v));
              }}
            >
              +
            </Button>
          </div>
          <p className="text-[10px] text-muted-foreground mt-1">Average period lasts 3-7 days. Min: 1, Max: 10.</p>
        </div>

        <Button type="submit" className="w-full">
          {initialData ? "Update Settings" : "Start Tracking"}
        </Button>
      </form>
    </motion.div>
  );
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

  // Local cycle data from localStorage
  const [localCycleData, setLocalCycleData_] = useState<LocalCycleData | null>(() => getLocalCycleData());
  const [showSettings, setShowSettings] = useState(false);

  const handleCycleSetup = useCallback((data: LocalCycleData) => {
    setLocalCycleData_(data);
    setShowSettings(false);
  }, []);

  // Computed cycle info from localStorage data
  const localCycleInfo = useMemo(() => {
    if (!localCycleData) return null;
    return computeCycleInfo(localCycleData);
  }, [localCycleData]);

  const [viewDate, setViewDate] = useState(() => {
    const now = new Date();
    return { year: now.getFullYear(), month: now.getMonth() };
  });
  const [logDialogOpen, setLogDialogOpen] = useState(false);
  const [selectedDate, setSelectedDate] = useState(today);
  const [flowLevel, setFlowLevel] = useState("none");
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [symptomSeverity, setSymptomSeverity] = useState<Record<string, number>>({});
  const [cycleNotes, setCycleNotes] = useState("");

  // Fetch cycle data — API with localStorage fallback
  const { data: cycleData = [] } = useQuery<CycleEntry[]>({
    queryKey: [`/api/cycle/${user?.id ?? "local"}?days=365`],
    queryFn: async () => {
      if (user?.id) {
        try {
          const res = await fetch(`/api/cycle/${user.id}?days=365`, {
            headers: { Authorization: `Bearer ${localStorage.getItem("auth_token") ?? ""}` },
          });
          if (res.ok) return res.json();
        } catch { /* API unavailable */ }
      }
      // Fallback: read from localStorage
      try {
        const raw = sbGetGeneric("ndw_cycle_logs") ?? {};
        return Object.entries(raw).map(([date, entry]: [string, any]) => ({
          id: date, userId: null, date, flowLevel: entry.flowLevel,
          symptoms: entry.symptoms, phase: null, contraception: null,
          basalTemp: null, notes: entry.notes,
        }));
      } catch { return []; }
    },
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

  // Log cycle mutation — API first, localStorage fallback when auth unavailable
  // Auto-detect last period start from logged cycle data and update localStorage
  // This runs whenever cycleData changes so calendar-logged period days update the phase
  useEffect(() => {
    if (cycleData.length === 0) return;
    // Find the most recent date with flow (not "none")
    const periodDates = cycleData
      .filter(e => e.flowLevel && e.flowLevel !== "none")
      .map(e => e.date)
      .sort(); // ascending ISO dates
    if (periodDates.length === 0) return;

    // Find consecutive period groups — the last group's start is the last period start
    const groups: string[][] = [];
    let currentGroup: string[] = [periodDates[0]];
    for (let i = 1; i < periodDates.length; i++) {
      const prevDate = new Date(periodDates[i - 1] + "T12:00:00");
      const currDate = new Date(periodDates[i] + "T12:00:00");
      const diffDays = Math.round((currDate.getTime() - prevDate.getTime()) / (1000 * 60 * 60 * 24));
      if (diffDays <= 2) {
        currentGroup.push(periodDates[i]);
      } else {
        groups.push(currentGroup);
        currentGroup = [periodDates[i]];
      }
    }
    groups.push(currentGroup);

    const lastGroup = groups[groups.length - 1];
    const detectedStart = lastGroup[0];

    // Update localStorage cycle data if the detected start is different
    const existing = getLocalCycleData();
    if (existing && existing.lastPeriodStart !== detectedStart) {
      const updated: LocalCycleData = {
        ...existing,
        lastPeriodStart: detectedStart,
      };
      setLocalCycleData(updated);
      setLocalCycleData_(updated);
    } else if (!existing && detectedStart) {
      // No setup done yet but user logged period days — create initial data
      const newData: LocalCycleData = {
        lastPeriodStart: detectedStart,
        cycleLength: 28,
        periodLength: lastGroup.length,
      };
      setLocalCycleData(newData);
      setLocalCycleData_(newData);
    }
  }, [cycleData]);

  const logCycleMutation = useMutation({
    mutationFn: async () => {
      const entry = {
        date: selectedDate,
        flowLevel,
        symptoms: selectedSymptoms.length > 0 ? selectedSymptoms : null,
        notes: cycleNotes || null,
      };
      try {
        const res = await apiRequest("POST", "/api/cycle", entry);
        if (!res.ok) throw new Error(await res.text());
        return res.json();
      } catch {
        // API unavailable or auth missing — save to localStorage
        const key = "ndw_cycle_logs";
        const existing: Record<string, typeof entry> = sbGetGeneric(key) ?? {};
        existing[selectedDate] = entry;
        sbSaveGeneric(key, existing);
        return entry;
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/api/cycle/${user?.id ?? "local"}?days=365`] });
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
      // Restore severity from notes metadata if stored as JSON suffix
      setSymptomSeverity({});
    } else {
      setFlowLevel("none");
      setSelectedSymptoms([]);
      setSymptomSeverity({});
      setCycleNotes("");
    }
    setLogDialogOpen(true);
  }

  function toggleSymptom(s: string) {
    setSelectedSymptoms(prev => {
      if (prev.includes(s)) {
        // Remove severity too
        setSymptomSeverity(sev => {
          const copy = { ...sev };
          delete copy[s];
          return copy;
        });
        return prev.filter(x => x !== s);
      }
      // Default severity = 1 (mild) when toggled on
      setSymptomSeverity(sev => ({ ...sev, [s]: 1 }));
      return [...prev, s];
    });
  }

  function setSeverity(s: string, level: number) {
    setSymptomSeverity(prev => ({ ...prev, [s]: level }));
  }

  // Computed cycle insights
  const cycleInsights = useMemo(() => {
    if (!phaseInfo) return null;
    const avgLen = phaseInfo.avgCycleLength || 28;
    const dayOfCycle = phaseInfo.dayOfCycle || 0;
    const ovulationDay = Math.round(avgLen - 14); // ~14 days before next period
    const daysUntilOvulation = Math.max(0, ovulationDay - dayOfCycle);
    const fertileWindowStart = Math.max(1, ovulationDay - 5);
    const fertileWindowEnd = ovulationDay + 1;
    const inFertileWindow = dayOfCycle >= fertileWindowStart && dayOfCycle <= fertileWindowEnd;
    return { avgLen, ovulationDay, daysUntilOvulation, fertileWindowStart, fertileWindowEnd, inFertileWindow };
  }, [phaseInfo]);

  // Basal body temperature from cycle entries
  const tempChartData = useMemo(() => {
    return cycleData
      .filter(e => e.basalTemp !== null && e.basalTemp !== undefined)
      .map(e => ({
        date: new Date(e.date + "T12:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" }),
        temp: parseFloat(e.basalTemp!),
      }))
      .slice(-30);
  }, [cycleData]);

  const monthDays = getMonthDays(viewDate.year, viewDate.month);

  // Compute predicted period date set for next 3 months (for calendar highlighting)
  const predictedPeriodDates = useMemo(() => {
    const dates = new Set<string>();
    if (!localCycleData) return dates;
    const { lastPeriodStart, cycleLength, periodLength } = localCycleData;
    const start = new Date(lastPeriodStart + "T12:00:00");
    // Generate predicted periods for up to 6 cycles ahead (covers ~6 months)
    for (let cycle = 0; cycle <= 6; cycle++) {
      const periodStart = new Date(start);
      periodStart.setDate(periodStart.getDate() + cycle * cycleLength);
      for (let d = 0; d < periodLength; d++) {
        const periodDay = new Date(periodStart);
        periodDay.setDate(periodDay.getDate() + d);
        dates.add(periodDay.toISOString().slice(0, 10));
      }
    }
    return dates;
  }, [localCycleData]);

  const monthLabel = new Date(viewDate.year, viewDate.month).toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
  });

  const phase = phaseInfo ? PHASE_INFO[phaseInfo.currentPhase] ?? PHASE_INFO.unknown : PHASE_INFO.unknown;

  // Determine which cycle info source to use: prefer server data, fall back to local
  const effectiveCycleInfo = useMemo(() => {
    if (phaseInfo && phaseInfo.currentPhase !== "unknown") {
      return {
        source: "server" as const,
        currentPhase: phaseInfo.currentPhase as CyclePhaseKey,
        dayOfCycle: phaseInfo.dayOfCycle,
        cycleLength: phaseInfo.avgCycleLength || 28,
        nextPeriodDate: phaseInfo.nextPeriodDate,
      };
    }
    if (localCycleInfo) {
      return {
        source: "local" as const,
        currentPhase: localCycleInfo.currentPhase,
        dayOfCycle: localCycleInfo.dayOfCycle,
        cycleLength: localCycleInfo.cycleLength,
        nextPeriodDate: localCycleInfo.nextPeriodDate,
      };
    }
    return null;
  }, [phaseInfo, localCycleInfo]);

  return (
    <div className="space-y-5">
      {/* Setup prompt — show when no cycle data exists anywhere */}
      {!effectiveCycleInfo && !showSettings && (
        <CycleSetupPrompt onComplete={handleCycleSetup} initialData={localCycleData} />
      )}

      {/* Settings edit dialog — re-enter cycle data */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <CycleSetupPrompt onComplete={handleCycleSetup} initialData={localCycleData} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Menstrual Cycle header + settings toggle */}
      {effectiveCycleInfo && (
        <motion.div
          className="flex items-center justify-between"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <div>
            <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold">
              Menstrual Cycle
            </p>
            <p className="text-xs text-muted-foreground mt-0.5">
              Day {effectiveCycleInfo.dayOfCycle} of {effectiveCycleInfo.cycleLength}
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            onClick={() => setShowSettings(prev => !prev)}
            title="Edit cycle settings"
          >
            <Settings2 className="h-3.5 w-3.5 text-muted-foreground" />
          </Button>
        </motion.div>
      )}

      {/* Cycle Wheel — Withings-style circular phase visualization */}
      {localCycleInfo && effectiveCycleInfo && (
        <motion.div
          className="rounded-xl border border-border bg-card p-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <CycleWheel cycleInfo={localCycleInfo} />
        </motion.div>
      )}

      {/* Phase details + next period card */}
      {effectiveCycleInfo && (
        <motion.div
          className="rounded-xl border p-4"
          style={{
            borderColor: CYCLE_PHASES.find(p => p.key === effectiveCycleInfo.currentPhase)?.color + "33",
            backgroundColor: CYCLE_PHASES.find(p => p.key === effectiveCycleInfo.currentPhase)?.color + "0a",
          }}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, delay: 0.1 }}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold">
                Current Phase
              </p>
              <p className="text-lg font-bold" style={{ color: CYCLE_PHASES.find(p => p.key === effectiveCycleInfo.currentPhase)?.color }}>
                {CYCLE_PHASES.find(p => p.key === effectiveCycleInfo.currentPhase)?.label}
              </p>
              <p className="text-xs text-muted-foreground">
                {HORMONE_PHASES[effectiveCycleInfo.currentPhase]?.description
                  ?? HORMONE_PHASES[effectiveCycleInfo.currentPhase === "ovulation" ? "ovulatory" : effectiveCycleInfo.currentPhase]?.description
                  ?? ""}
              </p>
            </div>
            <div className="text-right">
              <p className="text-3xl font-bold text-foreground">Day {effectiveCycleInfo.dayOfCycle}</p>
              <p className="text-[10px] text-muted-foreground">
                of {effectiveCycleInfo.cycleLength} days
              </p>
              {effectiveCycleInfo.nextPeriodDate && (
                <p className="text-[10px] text-muted-foreground mt-0.5">
                  Next period:{" "}
                  {new Date(effectiveCycleInfo.nextPeriodDate + "T12:00:00").toLocaleDateString(undefined, {
                    month: "short",
                    day: "numeric",
                  })}
                </p>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* Hormone Phase Indicator */}
      {effectiveCycleInfo && (() => {
        const phaseKey = effectiveCycleInfo.currentPhase === "ovulation" ? "ovulatory" : effectiveCycleInfo.currentPhase;
        const hp = HORMONE_PHASES[phaseKey] ?? HORMONE_PHASES.unknown;
        return (
          <motion.div
            className="rounded-xl border border-border bg-card p-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.15 }}
          >
            <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-2">Hormone Phase</p>
            <div className="flex items-start gap-3">
              <div>
                <p className="text-sm font-semibold text-foreground">{hp.hormones}</p>
                <p className="text-[11px] text-muted-foreground mt-0.5">{hp.description}</p>
              </div>
            </div>
          </motion.div>
        );
      })()}

      {/* Cycle Insights + Fertility Window */}
      {cycleInsights && phaseInfo && phaseInfo.periodStartCount >= 2 && (
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-xl border border-border bg-card p-3.5" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-medium mb-1">Avg Cycle</p>
            <span className="text-2xl font-bold text-foreground">{cycleInsights.avgLen}</span>
            <span className="text-xs text-muted-foreground ml-1">days</span>
          </div>
          <div className="rounded-xl border border-border bg-card p-3.5" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-medium mb-1">Ovulation In</p>
            <span className="text-2xl font-bold text-foreground">{cycleInsights.daysUntilOvulation}</span>
            <span className="text-xs text-muted-foreground ml-1">days</span>
          </div>
          {cycleInsights.inFertileWindow && (
            <div className="col-span-2 rounded-xl border border-ndw-stress/30 bg-ndw-stress/5 p-3.5 flex items-center gap-2">
              <div>
                <p className="text-xs font-semibold text-ndw-stress">Fertile Window</p>
                <p className="text-[10px] text-muted-foreground">
                  Days {cycleInsights.fertileWindowStart}--{cycleInsights.fertileWindowEnd} of your cycle
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Phase Legend — horizontal row of all 4 phases */}
      {effectiveCycleInfo && (
        <div className="grid grid-cols-4 gap-2">
          {CYCLE_PHASES.map((p) => {
            const isActive = p.key === effectiveCycleInfo.currentPhase;
            return (
              <motion.div
                key={p.key}
                className={`rounded-lg border p-2 text-center transition-colors ${
                  isActive ? "border-opacity-100" : "border-border opacity-50"
                }`}
                style={{
                  borderColor: isActive ? p.color : undefined,
                  backgroundColor: isActive ? p.color + "15" : undefined,
                }}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: isActive ? 1 : 0.5, y: 0 }}
                transition={{ duration: 0.3, delay: 0.2 }}
              >
                <div
                  className="w-2.5 h-2.5 rounded-full mx-auto mb-1"
                  style={{ backgroundColor: p.color }}
                />
                <p className="text-[9px] font-semibold" style={{ color: isActive ? p.color : "var(--muted-foreground)" }}>
                  {p.label}
                </p>
              </motion.div>
            );
          })}
        </div>
      )}

      {/* Basal Body Temperature Chart */}
      {tempChartData.length > 3 && (
        <div className="rounded-xl border border-border bg-card p-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}>
          <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-3">
            Basal Body Temperature
          </p>
          <div className="h-36">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={tempChartData}>
                <defs>
                  <linearGradient id="bbtGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#ea580c" stopOpacity={0.3} />
                    <stop offset="50%" stopColor="#ea580c" stopOpacity={0.12} />
                    <stop offset="100%" stopColor="#ea580c" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 8, fill: "var(--muted-foreground)" }}
                  tickLine={false}
                  axisLine={false}
                  interval="preserveStartEnd"
                />
                <YAxis
                  domain={["auto", "auto"]}
                  tick={{ fontSize: 8, fill: "var(--muted-foreground)" }}
                  tickLine={false}
                  axisLine={false}
                  width={32}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--card)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    fontSize: 11,
                    color: "var(--foreground)",
                  }}
                  formatter={(value: number) => [`${value.toFixed(1)} C`, "Temp"]}
                />
                <Area
                  type="monotone"
                  dataKey="temp"
                  stroke="#ea580c"
                  strokeWidth={2}
                  fill="url(#bbtGrad)"
                  dot={{ r: 2, fill: "#ea580c" }}
                  name="Temp (C)"
                  isAnimationActive={true}
                  animationDuration={1200}
                  animationEasing="ease-out"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[9px] text-muted-foreground mt-2 text-center">
            BBT rises ~0.2-0.5 C after ovulation due to progesterone
          </p>
        </div>
      )}

      {/* Calendar */}
      <div className="rounded-xl border border-border bg-card p-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}>
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
            const isPredictedPeriod = !flow && predictedPeriodDates.has(date);

            let bgColor = "";
            if (flow === "heavy") bgColor = "bg-rose-600/80";
            else if (flow === "medium") bgColor = "bg-pink-500/70";
            else if (flow === "light") bgColor = "bg-pink-300/60";
            else if (isPredictedPeriod) bgColor = "bg-pink-200/40";

            return (
              <button
                key={date}
                onClick={() => openLogDialog(date)}
                className={`aspect-square rounded-lg flex items-center justify-center text-xs font-medium transition-colors
                  ${!inMonth ? "opacity-30" : ""}
                  ${bgColor || "hover:bg-muted/50"}
                  ${isToday && !bgColor ? "ring-1 ring-primary" : ""}
                  ${isPredictedPeriod ? "ring-1 ring-pink-300/50 ring-dashed text-pink-400" : ""}
                  ${flow && flow !== "none" ? "text-white" : isPredictedPeriod ? "" : "text-foreground"}
                `}
              >
                {day}
              </button>
            );
          })}
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 mt-3 justify-center flex-wrap">
          {FLOW_LEVELS.filter(f => f.value !== "none").map(f => (
            <div key={f.value} className="flex items-center gap-1">
              <div className={`w-2.5 h-2.5 rounded-full ${f.color}`} />
              <span className="text-[9px] text-muted-foreground">{f.label}</span>
            </div>
          ))}
          {predictedPeriodDates.size > 0 && (
            <div className="flex items-center gap-1">
              <div className="w-2.5 h-2.5 rounded-full bg-pink-200/40 ring-1 ring-pink-300/50" />
              <span className="text-[9px] text-muted-foreground">Predicted</span>
            </div>
          )}
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

              {/* Severity for selected symptoms */}
              {selectedSymptoms.length > 0 && (
                <div className="mt-3 space-y-2">
                  <Label className="text-[10px] text-muted-foreground uppercase tracking-wide">Severity</Label>
                  {selectedSymptoms.map(s => (
                    <div key={s} className="flex items-center gap-2">
                      <span className="text-[10px] text-foreground w-24 truncate">{formatSymptom(s)}</span>
                      <div className="flex gap-1 flex-1">
                        {SEVERITY_LABELS.map((label, i) => (
                          <button
                            key={label}
                            onClick={() => setSeverity(s, i + 1)}
                            className={`flex-1 py-1 rounded text-[9px] font-medium border transition-colors ${
                              (symptomSeverity[s] ?? 1) === i + 1
                                ? i === 0 ? "border-cyan-500 bg-cyan-500/10 text-cyan-500"
                                : i === 1 ? "border-yellow-500 bg-yellow-500/10 text-yellow-500"
                                : "border-rose-500 bg-rose-500/10 text-rose-500"
                                : "border-border text-muted-foreground"
                            }`}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Notes */}
            <div>
              <Label className="text-xs font-medium">Notes</Label>
              <Textarea
                value={cycleNotes}
                onChange={e => setCycleNotes(e.target.value)}
                placeholder="Any notes about your state?"
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

  // Always load mood history from localStorage (works without auth)
  const [localMoodLogs, setLocalMoodLogs] = useState<MoodLog[]>(() => {
    try {
      return sbGetGeneric("ndw_mood_logs") ?? [];
    } catch { return []; }
  });

  // Re-read localStorage on mount and after mutations; also try Supabase
  const refreshLocalLogs = useCallback(() => {
    try {
      setLocalMoodLogs(sbGetGeneric("ndw_mood_logs") ?? []);
    } catch { setLocalMoodLogs([]); }
    // Also fetch from Supabase in background (updates localStorage cache)
    sbGetMoodLogs(user?.id ?? "local", 30).then((logs) => {
      if (logs.length > 0) setLocalMoodLogs(logs);
    }).catch(() => {});
  }, [user?.id]);

  // Fetch mood logs from API when authenticated
  const { data: apiMoodLogs = [] } = useQuery<MoodLog[]>({
    queryKey: [`/api/mood/${user?.id}?days=30`],
    enabled: !!user?.id,
    retry: false,
    staleTime: 30_000,
    queryFn: async () => {
      try {
        const res = await fetch(`/api/mood/${user?.id}?days=30`);
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data)) return data;
        }
      } catch { /* fall through */ }
      return [];
    },
  });

  // Merge: prefer API data when available, fall back to localStorage
  const moodLogs = useMemo(() => {
    if (apiMoodLogs.length > 0) return apiMoodLogs;
    return localMoodLogs;
  }, [apiMoodLogs, localMoodLogs]);

  // Log mood mutation — tries API first, falls back to localStorage
  const logMoodMutation = useMutation({
    mutationFn: async () => {
      try {
        const res = await apiRequest("POST", "/api/mood", {
          moodScore,
          energyLevel,
          notes: moodNotes || null,
        });
        if (!res.ok) throw new Error("API error");
        return res.json();
      } catch {
        // Fallback: save to localStorage
        const key = "ndw_mood_logs";
        const existing = sbGetGeneric(key) ?? [];
        const entry = {
          id: `local_${Date.now()}`,
          userId: user?.id,
          moodScore: String(moodScore),
          energyLevel: String(energyLevel),
          notes: moodNotes || null,
          loggedAt: new Date().toISOString(),
        };
        existing.unshift(entry);
        if (existing.length > 100) existing.length = 100;
        sbSaveGeneric(key, existing);
        return entry;
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/api/mood/${user?.id}?days=30`] });
      // Persist to Supabase + localStorage via supabase-store
      sbSaveMoodLog(user?.id ?? "local", {
        mood: moodScore,
        energy: energyLevel,
        notes: moodNotes || undefined,
      }).catch(() => {});
      refreshLocalLogs(); // Update local mood history display
      setMoodNotes("");
      toast({ title: "Mood logged" });
      // Sync to Railway ML backend for session history + retraining
      if (user?.id) {
        syncMoodLogToML({
          user_id: user.id,
          mood_score: moodScore,
          energy_level: energyLevel,
          notes: moodNotes || undefined,
        });
      }
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
        className="rounded-xl border border-border bg-card p-4 space-y-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
      >
        <p className="text-sm font-semibold text-foreground">How are you feeling?</p>
        <p className="text-[11px] text-muted-foreground mt-0.5">Track your mood and energy over time</p>

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
          <input
            type="range"
            value={moodScore}
            onChange={e => setMoodScore(parseInt(e.target.value))}
            min={1}
            max={10}
            step={1}
            className="w-full py-2 accent-primary"
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
          <input
            type="range"
            value={energyLevel}
            onChange={e => setEnergyLevel(parseInt(e.target.value))}
            min={1}
            max={10}
            step={1}
            className="w-full py-2 accent-primary"
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
          className="rounded-xl border border-border bg-card p-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
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
                    <stop offset="0%" stopColor="#0891b2" stopOpacity={0.3} />
                    <stop offset="50%" stopColor="#0891b2" stopOpacity={0.12} />
                    <stop offset="100%" stopColor="#0891b2" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="energyGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#d4a017" stopOpacity={0.3} />
                    <stop offset="50%" stopColor="#d4a017" stopOpacity={0.12} />
                    <stop offset="100%" stopColor="#d4a017" stopOpacity={0} />
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
                  isAnimationActive={true}
                  animationDuration={1200}
                  animationEasing="ease-out"
                />
                <Area
                  type="monotone"
                  dataKey="energy"
                  stroke="#d4a017"
                  fill="url(#energyGradient)"
                  strokeWidth={2.5}
                  dot={{ r: 1.5 }}
                  strokeDasharray="4 2"
                  name="Energy"
                  isAnimationActive={true}
                  animationDuration={1200}
                  animationEasing="ease-out"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      )}

      {/* Mood by time of day */}
      {timePattern.length > 0 && (
        <div className="rounded-xl border border-border bg-card p-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}>
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
            Average mood by time of day
          </p>
          <div className="space-y-2.5">
            {["Morning", "Afternoon", "Evening", "Night"].map(label => {
              const bucket = timePattern.find(b => b.label === label);
              const timeRanges: Record<string, string> = {
                Morning: "5am - 12pm",
                Afternoon: "12pm - 5pm",
                Evening: "5pm - 9pm",
                Night: "9pm - 5am",
              };
              const avg = bucket?.avg ?? 0;
              const barPct = bucket ? Math.round((avg / 10) * 100) : 0;
              const barColor = avg >= 7 ? "#0891b2" : avg >= 5 ? "#d4a017" : "#e879a8";
              return (
                <div key={label} className="flex items-center gap-3">
                  <div className="w-[72px] shrink-0">
                    <p className="text-[11px] font-medium text-foreground leading-tight">{label}</p>
                    <p className="text-[9px] text-muted-foreground/60">{timeRanges[label]}</p>
                  </div>
                  {bucket ? (
                    <div className="flex-1 flex items-center gap-2">
                      <div className="flex-1 h-5 rounded-full bg-muted/30 overflow-hidden">
                        <motion.div
                          className="h-full rounded-full"
                          style={{ backgroundColor: barColor, width: `${barPct}%` }}
                          initial={{ width: 0 }}
                          animate={{ width: `${barPct}%` }}
                          transition={{ duration: 0.6, ease: "easeOut" }}
                        />
                      </div>
                      <span className="text-sm font-bold text-foreground w-8 text-right">{avg}</span>
                      <span className="text-[9px] text-muted-foreground/50 w-12 text-right">{bucket.count} {bucket.count === 1 ? "log" : "logs"}</span>
                    </div>
                  ) : (
                    <div className="flex-1 flex items-center gap-2">
                      <div className="flex-1 h-5 rounded-full bg-muted/30" />
                      <span className="text-[10px] text-muted-foreground/40 w-8 text-right">--</span>
                      <span className="w-12" />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          <p className="text-[9px] text-muted-foreground/50 mt-2.5 text-center">
            Based on when you log -- bars show average mood out of 10
          </p>
        </div>
      )}

      {/* Recent mood entries */}
      {moodLogs.length > 0 && (
        <div className="rounded-xl border border-border bg-card p-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}>
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
            Recent entries
          </p>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {moodLogs.slice(0, 10).map(log => {
              const f = getMoodFace(parseFloat(log.moodScore));
              const Icon = f.icon;
              const moodVal = parseFloat(log.moodScore);
              const energyVal = log.energyLevel ? parseFloat(log.energyLevel) : null;
              return (
                <div key={log.id} className="flex items-center gap-3 py-2 border-b border-border/30 last:border-0">
                  <Icon className="h-5 w-5 text-primary shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] text-muted-foreground w-10">Mood</span>
                        <div className="w-16 h-2 rounded-full bg-muted/30 overflow-hidden">
                          <div className="h-full rounded-full bg-[#0891b2]" style={{ width: `${moodVal * 10}%` }} />
                        </div>
                        <span className="text-xs font-bold w-4">{moodVal}</span>
                      </div>
                      {energyVal !== null && (
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] text-muted-foreground w-10">Energy</span>
                          <div className="w-16 h-2 rounded-full bg-muted/30 overflow-hidden">
                            <div className="h-full rounded-full bg-[#d4a017]" style={{ width: `${energyVal * 10}%` }} />
                          </div>
                          <span className="text-xs font-bold w-4">{energyVal}</span>
                        </div>
                      )}
                    </div>
                    {log.notes && (
                      <p className="text-[10px] text-muted-foreground truncate mt-0.5">{log.notes}</p>
                    )}
                  </div>
                  <span className="text-[10px] text-muted-foreground shrink-0">
                    {log.loggedAt
                      ? new Date(log.loggedAt).toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
                        " " +
                        new Date(log.loggedAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
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
          Cycle tracking and optional mood logging
        </p>
      </motion.div>

      {/* Tabs */}
      <Tabs value={tab} onValueChange={setTab} className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="cycle" className="gap-1.5">
            <Calendar className="h-3.5 w-3.5" />
            Menstrual Cycle
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
