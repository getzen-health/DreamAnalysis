import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ListChecks,
  Plus,
  Flame,
  Droplets,
  Sun,
  Monitor,
  Coffee,
  Brain,
  Dumbbell,
  Check,
  Trash2,
} from "lucide-react";
import { motion } from "framer-motion";
import { listItemVariants, pageTransition, staggerContainer, staggerChild } from "@/lib/animations";
import { useToast } from "@/hooks/use-toast";
import { EmotionStrip } from "@/components/emotion-strip";
import { HabitStreakCard } from "@/components/habit-streak-card";
import { HabitHeatmap } from "@/components/habit-heatmap";
import { HabitAnalytics } from "@/components/habit-analytics";

/* ---------- types ---------- */

interface Habit {
  id: string;
  userId: string | null;
  name: string;
  category: string | null;
  icon: string | null;
  targetValue: string | null;
  unit: string | null;
  isActive: boolean | null;
  createdAt: string | null;
}

interface HabitLog {
  id: string;
  userId: string | null;
  habitId: string | null;
  value: string;
  note: string | null;
  loggedAt: string | null;
}

/* ---------- prebuilt habits ---------- */

const PREBUILT_HABITS = [
  { name: "Water", category: "health", icon: "droplets", targetValue: 8, unit: "glasses", iconBg: "bg-indigo-500/15", iconColor: "text-indigo-400" },
  { name: "Sunlight", category: "health", icon: "sun", targetValue: 30, unit: "min", iconBg: "bg-ndw-stress/15", iconColor: "text-ndw-stress" },
  { name: "Screen Time", category: "wellness", icon: "monitor", targetValue: 2, unit: "hrs", iconBg: "bg-ndw-sleep/15", iconColor: "text-ndw-sleep" },
  { name: "Caffeine", category: "nutrition", icon: "coffee", targetValue: 3, unit: "cups", iconBg: "bg-ndw-nutrition/15", iconColor: "text-ndw-nutrition" },
  { name: "Meditation", category: "mindfulness", icon: "brain", targetValue: 15, unit: "min", iconBg: "bg-ndw-recovery/15", iconColor: "text-ndw-recovery" },
  { name: "Exercise", category: "fitness", icon: "dumbbell", targetValue: 30, unit: "min", iconBg: "bg-ndw-strain/15", iconColor: "text-ndw-strain" },
];

const CATEGORIES = ["health", "wellness", "fitness", "nutrition", "mindfulness", "other"];

const ICON_MAP: Record<string, typeof Droplets> = {
  droplets: Droplets,
  sun: Sun,
  monitor: Monitor,
  coffee: Coffee,
  brain: Brain,
  dumbbell: Dumbbell,
  flame: Flame,
  check: ListChecks,
};

function getIconComponent(iconName: string | null) {
  if (!iconName) return ListChecks;
  return ICON_MAP[iconName] ?? ListChecks;
}

/* ---------- helpers ---------- */

function getToday(): string {
  return new Date().toISOString().slice(0, 10);
}

function getDateStr(date: string | Date | null): string {
  if (!date) return "";
  return new Date(date).toISOString().slice(0, 10);
}

function getLast7Days(): string[] {
  const days: string[] = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    days.push(d.toISOString().slice(0, 10));
  }
  return days;
}

/* ========== Component ========== */

export default function Habits() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [logDialogOpen, setLogDialogOpen] = useState(false);
  const [selectedHabit, setSelectedHabit] = useState<Habit | null>(null);

  // Add habit form
  const [newName, setNewName] = useState("");
  const [newCategory, setNewCategory] = useState("health");
  const [newIcon, setNewIcon] = useState("check");
  const [newTarget, setNewTarget] = useState("");
  const [newUnit, setNewUnit] = useState("");

  // Log form
  const [logValue, setLogValue] = useState("");
  const [logNote, setLogNote] = useState("");

  // Fetch habits
  const { data: userHabits = [] } = useQuery<Habit[]>({
    queryKey: [`/api/habits/${user?.id}`],
    enabled: !!user?.id,
    retry: false,
    staleTime: 30_000,
  });

  // Fetch habit logs (last 30 days — for habit list 7-day tracker)
  const { data: habitLogData = [] } = useQuery<HabitLog[]>({
    queryKey: [`/api/habit-logs/${user?.id}`],
    enabled: !!user?.id,
    retry: false,
    staleTime: 30_000,
  });

  // Fetch habit logs (last 90 days — for heatmap + analytics)
  const { data: extendedLogData = [] } = useQuery<HabitLog[]>({
    queryKey: [`/api/habit-logs/${user?.id}?days=90`],
    enabled: !!user?.id,
    retry: false,
    staleTime: 60_000,
  });

  // Fetch streaks
  const { data: streaks = {} } = useQuery<Record<string, number>>({
    queryKey: [`/api/habit-logs/${user?.id}/streaks`],
    enabled: !!user?.id,
    retry: false,
    staleTime: 60_000,
  });

  // Compute today's logged habits
  const today = getToday();
  const todayLogs = useMemo(() => {
    const logsByHabit = new Map<string, HabitLog[]>();
    for (const log of habitLogData) {
      if (!log.habitId) continue;
      const dateStr = getDateStr(log.loggedAt);
      if (dateStr === today) {
        if (!logsByHabit.has(log.habitId)) logsByHabit.set(log.habitId, []);
        logsByHabit.get(log.habitId)!.push(log);
      }
    }
    return logsByHabit;
  }, [habitLogData, today]);

  // Build 7-day history per habit
  const last7 = getLast7Days();
  const weekHistory = useMemo(() => {
    const result = new Map<string, Set<string>>();
    for (const log of habitLogData) {
      if (!log.habitId) continue;
      const dateStr = getDateStr(log.loggedAt);
      if (!result.has(log.habitId)) result.set(log.habitId, new Set());
      result.get(log.habitId)!.add(dateStr);
    }
    return result;
  }, [habitLogData]);

  // Create habit mutation
  const createHabitMutation = useMutation({
    mutationFn: async (data: { name: string; category: string; icon: string; targetValue: number | null; unit: string }) => {
      const res = await apiRequest("POST", "/api/habits", data);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/api/habits/${user?.id}`] });
      setAddDialogOpen(false);
      setNewName("");
      setNewTarget("");
      setNewUnit("");
      toast({ title: "Habit created" });
    },
    onError: (err: Error) => {
      toast({ title: "Error", description: err.message, variant: "destructive" });
    },
  });

  // Log habit mutation
  const logHabitMutation = useMutation({
    mutationFn: async (data: { habitId: string; value: number; note: string }) => {
      const res = await apiRequest("POST", "/api/habit-logs", data);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/api/habit-logs/${user?.id}`] });
      queryClient.invalidateQueries({ queryKey: [`/api/habit-logs/${user?.id}/streaks`] });
      setLogDialogOpen(false);
      setLogValue("");
      setLogNote("");
      setSelectedHabit(null);
      toast({ title: "Habit logged" });
    },
    onError: (err: Error) => {
      toast({ title: "Error", description: err.message, variant: "destructive" });
    },
  });

  // Delete habit mutation
  const deleteHabitMutation = useMutation({
    mutationFn: async (habitId: string) => {
      const res = await apiRequest("DELETE", `/api/habits/${habitId}`);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/api/habits/${user?.id}`] });
      queryClient.invalidateQueries({ queryKey: [`/api/habit-logs/${user?.id}/streaks`] });
      toast({ title: "Habit removed" });
    },
    onError: (err: Error) => {
      toast({ title: "Error", description: err.message, variant: "destructive" });
    },
  });

  function handleQuickLog(habit: Habit) {
    setSelectedHabit(habit);
    setLogValue(habit.targetValue ? String(parseFloat(habit.targetValue)) : "1");
    setLogDialogOpen(true);
  }

  function handleAddPrebuilt(p: typeof PREBUILT_HABITS[0]) {
    createHabitMutation.mutate({
      name: p.name,
      category: p.category,
      icon: p.icon,
      targetValue: p.targetValue,
      unit: p.unit,
    });
  }

  function handleCreateCustom() {
    const target = newTarget ? parseFloat(newTarget) : null;
    createHabitMutation.mutate({
      name: newName,
      category: newCategory,
      icon: newIcon,
      targetValue: target,
      unit: newUnit,
    });
  }

  function handleLogSubmit() {
    if (!selectedHabit) return;
    const val = parseFloat(logValue);
    if (isNaN(val) || val < 0) {
      toast({ title: "Error", description: "Enter a valid value", variant: "destructive" });
      return;
    }
    logHabitMutation.mutate({
      habitId: selectedHabit.id,
      value: val,
      note: logNote,
    });
  }

  // Day labels for 7-day grid
  const dayLabels = last7.map(d => {
    const date = new Date(d + "T12:00:00");
    return date.toLocaleDateString(undefined, { weekday: "narrow" });
  });

  return (
    <div className="max-w-lg mx-auto px-4 py-6 space-y-6">
      {/* Header */}
      <motion.div
        className="space-y-2"
        initial={pageTransition.initial}
        animate={pageTransition.animate}
        transition={pageTransition.transition}
      >
        <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-foreground flex items-center gap-2">
            <ListChecks className="h-5 w-5 text-primary" />
            Habits
          </h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Track daily habits and build streaks
          </p>
        </div>
        <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
          <DialogTrigger asChild>
            <Button size="sm" variant="outline" className="gap-1">
              <Plus className="h-4 w-4" /> Add
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-sm">
            <DialogHeader>
              <DialogTitle>Add Habit</DialogTitle>
            </DialogHeader>

            {/* Prebuilt habits */}
            <div className="space-y-2">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Quick Add</p>
              <div className="grid grid-cols-2 gap-2">
                {PREBUILT_HABITS.map(p => {
                  const already = userHabits.some(h => h.name === p.name);
                  const Icon = ICON_MAP[p.icon] ?? ListChecks;
                  return (
                    <button
                      key={p.name}
                      disabled={already || createHabitMutation.isPending}
                      onClick={() => handleAddPrebuilt(p)}
                      className="flex items-center gap-2.5 p-2.5 rounded-lg border border-border text-left hover:bg-muted/50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    >
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${p.iconBg}`}>
                        <Icon className={`h-4 w-4 ${p.iconColor}`} />
                      </div>
                      <div>
                        <div className="text-sm font-medium text-foreground">{p.name}</div>
                        <div className="text-[10px] text-muted-foreground">{p.targetValue} {p.unit}/day</div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Custom habit form */}
            <div className="space-y-3 pt-3 border-t">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Custom Habit</p>
              <div>
                <Label className="text-xs">Name</Label>
                <Input
                  value={newName}
                  onChange={e => setNewName(e.target.value)}
                  placeholder="e.g., Read"
                  className="mt-1"
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <Label className="text-xs">Category</Label>
                  <Select value={newCategory} onValueChange={setNewCategory}>
                    <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {CATEGORIES.map(c => (
                        <SelectItem key={c} value={c}>{c}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label className="text-xs">Icon</Label>
                  <Select value={newIcon} onValueChange={setNewIcon}>
                    <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {Object.keys(ICON_MAP).map(k => (
                        <SelectItem key={k} value={k}>{k}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <Label className="text-xs">Target</Label>
                  <Input
                    type="number"
                    value={newTarget}
                    onChange={e => setNewTarget(e.target.value)}
                    placeholder="e.g., 8"
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label className="text-xs">Unit</Label>
                  <Input
                    value={newUnit}
                    onChange={e => setNewUnit(e.target.value)}
                    placeholder="e.g., glasses"
                    className="mt-1"
                  />
                </div>
              </div>
              <Button
                onClick={handleCreateCustom}
                disabled={!newName.trim() || createHabitMutation.isPending}
                className="w-full"
                size="sm"
              >
                Create Habit
              </Button>
            </div>
          </DialogContent>
        </Dialog>
        </div>
        <EmotionStrip />
      </motion.div>

      {/* Streaks section — horizontal scrollable cards */}
      {userHabits.length > 0 && (
        <motion.div
          className="space-y-2"
          variants={staggerContainer}
          initial="hidden"
          animate="show"
        >
          <motion.h2
            className="text-xs font-medium text-muted-foreground uppercase tracking-wide"
            variants={staggerChild}
          >
            Streaks
          </motion.h2>
          <div className="flex gap-3 overflow-x-auto pb-2 -mx-4 px-4 scrollbar-hide">
            {userHabits.map((habit, idx) => (
              <HabitStreakCard
                key={habit.id}
                habit={habit}
                currentStreak={streaks[habit.id] ?? 0}
                logs={extendedLogData}
                index={idx}
              />
            ))}
          </div>
        </motion.div>
      )}

      {/* Activity heatmap */}
      {userHabits.length > 0 && (
        <HabitHeatmap habits={userHabits} logs={extendedLogData} />
      )}

      {/* Analytics */}
      {userHabits.length > 0 && (
        <HabitAnalytics habits={userHabits} logs={extendedLogData} />
      )}

      {/* Habits list */}
      {userHabits.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          <ListChecks className="h-10 w-10 mx-auto mb-3 opacity-40" />
          <p className="text-sm font-medium">No habits yet</p>
          <p className="text-xs mt-1">Tap "Add" to create your first habit</p>
        </div>
      ) : (
        <div className="space-y-3">
          {userHabits.map((habit, idx) => {
            const Icon = getIconComponent(habit.icon);
            const isLoggedToday = todayLogs.has(habit.id);
            const streak = streaks[habit.id] ?? 0;
            const habitWeek = weekHistory.get(habit.id) ?? new Set();

            return (
              <motion.div
                key={habit.id}
                className="rounded-xl border border-border bg-card p-3.5 space-y-2.5"
                custom={idx}
                initial="hidden"
                animate="visible"
                variants={listItemVariants}
              >
                {/* Top row: icon, name, streak, actions */}
                <div className="flex items-center gap-3">
                  <div className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${
                    isLoggedToday ? "bg-primary/15 text-primary" : "bg-muted text-muted-foreground"
                  }`}>
                    <Icon className="h-4.5 w-4.5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold truncate">{habit.name}</span>
                      {streak > 0 && (
                        <span className="flex items-center gap-0.5 text-[10px] font-semibold text-ndw-stress">
                          <Flame className="h-3 w-3 text-ndw-stress" />
                          {streak}d
                        </span>
                      )}
                    </div>
                    {habit.targetValue && (
                      <p className="text-[10px] text-muted-foreground">
                        Goal: {parseFloat(habit.targetValue)} {habit.unit ?? ""}/day
                      </p>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Button
                      size="sm"
                      variant={isLoggedToday ? "secondary" : "default"}
                      className="h-8 text-xs gap-1"
                      onClick={() => handleQuickLog(habit)}
                    >
                      {isLoggedToday ? <><Check className="h-3 w-3" /> Done</> : "Log"}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                      onClick={() => deleteHabitMutation.mutate(habit.id)}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>

                {/* 7-day history grid */}
                <div className="flex items-center gap-1.5 justify-between px-0.5">
                  {last7.map((day, i) => {
                    const logged = habitWeek.has(day);
                    return (
                      <div key={day} className="flex flex-col items-center gap-1">
                        <motion.div
                          className={`w-7 h-7 rounded-lg flex items-center justify-center text-[10px] font-medium transition-colors ${
                            logged
                              ? "bg-ndw-recovery/20 text-ndw-recovery"
                              : day === today
                              ? "border border-dashed border-muted-foreground/30 text-muted-foreground/50"
                              : "bg-muted/30 text-muted-foreground/30"
                          }`}
                          whileTap={{ scale: 0.9 }}
                        >
                          {logged ? (
                            <motion.div
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              transition={{ type: "spring", stiffness: 260, damping: 20 }}
                            >
                              <Check className="h-3.5 w-3.5" />
                            </motion.div>
                          ) : ""}
                        </motion.div>
                        <span className="text-[9px] text-muted-foreground/60">{dayLabels[i]}</span>
                      </div>
                    );
                  })}
                </div>
              </motion.div>
            );
          })}
        </div>
      )}

      {/* Log dialog */}
      <Dialog open={logDialogOpen} onOpenChange={setLogDialogOpen}>
        <DialogContent className="max-w-xs">
          <DialogHeader>
            <DialogTitle>Log: {selectedHabit?.name}</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div>
              <Label className="text-xs">
                Value {selectedHabit?.unit ? `(${selectedHabit.unit})` : ""}
              </Label>
              <Input
                type="number"
                value={logValue}
                onChange={e => setLogValue(e.target.value)}
                placeholder={selectedHabit?.targetValue ?? "1"}
                className="mt-1"
                autoFocus
              />
            </div>
            <div>
              <Label className="text-xs">Note (optional)</Label>
              <Input
                value={logNote}
                onChange={e => setLogNote(e.target.value)}
                placeholder="How did it go?"
                className="mt-1"
              />
            </div>
            <Button
              onClick={handleLogSubmit}
              disabled={logHabitMutation.isPending}
              className="w-full"
              size="sm"
            >
              Save
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
