import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { apiRequest } from "@/lib/queryClient";
import { useAuth } from "@/hooks/use-auth";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Timer,
  Plus,
  Check,
  Dumbbell,
  Trophy,
  Pause,
  Play,
  Search,
  X,
  ChevronDown,
  ChevronUp,
  Trash2,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import type { Exercise, InProgressExercise, InProgressSet } from "@/lib/workout-types";
import {
  saveActiveWorkout,
  loadActiveWorkout,
  clearActiveWorkout,
  formatDuration,
  calculateVolume,
  getMuscleGroupColor,
  RPE_OPTIONS,
  REST_TIMER_OPTIONS,
} from "@/lib/workout-types";

/* ---------- Rest Timer Circle ---------- */

function RestTimerCircle({
  total,
  remaining,
  onDismiss,
}: {
  total: number;
  remaining: number;
  onDismiss: () => void;
}) {
  const progress = remaining / total;
  const radius = 50;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - progress);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm"
      onClick={onDismiss}
    >
      <div
        className="flex flex-col items-center gap-4"
        onClick={(e) => e.stopPropagation()}
      >
        <p className="text-sm font-medium text-muted-foreground">Rest Timer</p>
        <div className="relative w-32 h-32">
          <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
            <circle
              cx="60"
              cy="60"
              r={radius}
              stroke="currentColor"
              strokeWidth="6"
              fill="none"
              className="text-muted/20"
            />
            <circle
              cx="60"
              cy="60"
              r={radius}
              stroke="currentColor"
              strokeWidth="6"
              fill="none"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              strokeLinecap="round"
              className="text-primary transition-all duration-1000 ease-linear"
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-3xl font-bold text-foreground tabular-nums">
              {formatDuration(remaining)}
            </span>
          </div>
        </div>
        <Button variant="outline" size="sm" onClick={onDismiss}>
          Skip Rest
        </Button>
      </div>
    </motion.div>
  );
}

/* ---------- Set Row ---------- */

function SetRow({
  set,
  onUpdate,
  onComplete,
}: {
  set: InProgressSet;
  onUpdate: (field: keyof InProgressSet, value: string) => void;
  onComplete: () => void;
}) {
  return (
    <div
      className={`flex items-center gap-2 px-2 py-1.5 rounded-lg transition-colors ${
        set.completed ? "bg-primary/5" : ""
      }`}
    >
      <span className="text-[11px] font-semibold text-muted-foreground w-6 text-center tabular-nums">
        {set.setNumber}
      </span>
      <Input
        type="number"
        inputMode="decimal"
        value={set.weightKg}
        onChange={(e) => onUpdate("weightKg", e.target.value)}
        placeholder="kg"
        className="h-8 w-20 text-center text-sm tabular-nums bg-card"
        disabled={set.completed}
      />
      <span className="text-[10px] text-muted-foreground">x</span>
      <Input
        type="number"
        inputMode="numeric"
        value={set.reps}
        onChange={(e) => onUpdate("reps", e.target.value)}
        placeholder="reps"
        className="h-8 w-16 text-center text-sm tabular-nums bg-card"
        disabled={set.completed}
      />
      <Select
        value={set.rpe || ""}
        onValueChange={(v) => onUpdate("rpe", v)}
        disabled={set.completed}
      >
        <SelectTrigger className="h-8 w-16 text-xs bg-card">
          <SelectValue placeholder="RPE" />
        </SelectTrigger>
        <SelectContent>
          {RPE_OPTIONS.map((rpe) => (
            <SelectItem key={rpe} value={rpe}>
              RPE {rpe}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <button
        onClick={onComplete}
        className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 transition-colors ${
          set.completed
            ? "bg-primary text-primary-foreground"
            : "border border-border text-muted-foreground hover:border-primary hover:text-primary"
        }`}
      >
        <Check className="h-4 w-4" />
      </button>
    </div>
  );
}

/* ---------- Exercise Block ---------- */

function ExerciseBlock({
  exercise,
  onAddSet,
  onUpdateSet,
  onCompleteSet,
  onRemoveExercise,
  collapsed,
  onToggleCollapse,
}: {
  exercise: InProgressExercise;
  onAddSet: () => void;
  onUpdateSet: (
    setIdx: number,
    field: keyof InProgressSet,
    value: string
  ) => void;
  onCompleteSet: (setIdx: number) => void;
  onRemoveExercise: () => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}) {
  const completedSets = exercise.sets.filter((s) => s.completed).length;

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      {/* Exercise header */}
      <button
        onClick={onToggleCollapse}
        className="w-full flex items-center gap-3 p-3 hover:bg-muted/30 transition-colors"
      >
        <div className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0 bg-primary/10">
          <Dumbbell className="h-4 w-4 text-primary" />
        </div>
        <div className="flex-1 text-left min-w-0">
          <p className="text-sm font-semibold text-foreground truncate">
            {exercise.exerciseName}
          </p>
          <p className="text-[10px] text-muted-foreground">
            {completedSets}/{exercise.sets.length} sets
          </p>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onRemoveExercise();
          }}
          className="w-7 h-7 rounded-lg flex items-center justify-center text-muted-foreground hover:text-destructive transition-colors"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
        {collapsed ? (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronUp className="h-4 w-4 text-muted-foreground" />
        )}
      </button>

      {/* Sets */}
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3 space-y-1">
              {/* Header */}
              <div className="flex items-center gap-2 px-2 py-1 text-[10px] text-muted-foreground font-medium uppercase tracking-wider">
                <span className="w-6 text-center">Set</span>
                <span className="w-20 text-center">Weight</span>
                <span className="w-4" />
                <span className="w-16 text-center">Reps</span>
                <span className="w-16 text-center">RPE</span>
                <span className="w-8" />
              </div>

              {exercise.sets.map((set, setIdx) => (
                <SetRow
                  key={setIdx}
                  set={set}
                  onUpdate={(field, value) => onUpdateSet(setIdx, field, value)}
                  onComplete={() => onCompleteSet(setIdx)}
                />
              ))}

              <Button
                variant="ghost"
                size="sm"
                className="w-full h-8 text-xs gap-1 text-muted-foreground"
                onClick={onAddSet}
              >
                <Plus className="h-3 w-3" /> Add Set
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ---------- Workout Summary Dialog ---------- */

function WorkoutSummary({
  open,
  onClose,
  exercises,
  durationSec,
  newPRs,
}: {
  open: boolean;
  onClose: () => void;
  exercises: InProgressExercise[];
  durationSec: number;
  newPRs: string[];
}) {
  const totalSets = exercises.reduce(
    (acc, e) => acc + e.sets.filter((s) => s.completed).length,
    0
  );
  const totalVolume = exercises.reduce(
    (acc, e) => acc + calculateVolume(e.sets),
    0
  );

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Trophy className="h-5 w-5 text-amber-400" />
            Workout Complete
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* Stats grid */}
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl bg-muted/30 p-3 text-center">
              <p className="text-2xl font-bold text-foreground">
                {formatDuration(durationSec)}
              </p>
              <p className="text-[10px] text-muted-foreground">Duration</p>
            </div>
            <div className="rounded-xl bg-muted/30 p-3 text-center">
              <p className="text-2xl font-bold text-foreground">
                {exercises.length}
              </p>
              <p className="text-[10px] text-muted-foreground">Exercises</p>
            </div>
            <div className="rounded-xl bg-muted/30 p-3 text-center">
              <p className="text-2xl font-bold text-foreground">{totalSets}</p>
              <p className="text-[10px] text-muted-foreground">Sets</p>
            </div>
            <div className="rounded-xl bg-muted/30 p-3 text-center">
              <p className="text-2xl font-bold text-foreground">
                {totalVolume > 0 ? `${Math.round(totalVolume).toLocaleString()}` : "---"}
              </p>
              <p className="text-[10px] text-muted-foreground">Volume (kg)</p>
            </div>
          </div>

          {/* New PRs */}
          {newPRs.length > 0 && (
            <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3">
              <p className="text-xs font-semibold text-amber-400 mb-1">
                New Personal Records!
              </p>
              {newPRs.map((pr) => (
                <p key={pr} className="text-xs text-muted-foreground">
                  {pr}
                </p>
              ))}
            </div>
          )}

          <Button className="w-full" onClick={onClose}>
            Done
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/* ========== Main Component ========== */

export default function ActiveWorkout() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [, setLocation] = useLocation();

  // Workout state
  const [exercises, setExercises] = useState<InProgressExercise[]>([]);
  const [startedAt, setStartedAt] = useState<string>("");
  const [elapsedSec, setElapsedSec] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const [collapsedExercises, setCollapsedExercises] = useState<Set<number>>(
    new Set()
  );

  // Rest timer
  const [restTotal, setRestTotal] = useState(0);
  const [restRemaining, setRestRemaining] = useState(0);
  const [restTimerActive, setRestTimerActive] = useState(false);
  const restIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Exercise picker sheet
  const [showExercisePicker, setShowExercisePicker] = useState(false);
  const [exerciseSearch, setExerciseSearch] = useState("");

  // Summary dialog
  const [showSummary, setShowSummary] = useState(false);
  const [summaryDuration, setSummaryDuration] = useState(0);
  const [summaryPRs, setSummaryPRs] = useState<string[]>([]);

  // Saving state
  const [isSaving, setIsSaving] = useState(false);

  // Fetch exercise library
  const { data: exerciseLibrary = [] } = useQuery<Exercise[]>({
    queryKey: ["/api/exercises"],
    staleTime: 5 * 60_000,
  });

  // Filter exercise library
  const filteredLibrary = useMemo(() => {
    if (!exerciseSearch.trim()) return exerciseLibrary;
    const q = exerciseSearch.toLowerCase();
    return exerciseLibrary.filter(
      (e) =>
        e.name.toLowerCase().includes(q) ||
        e.muscleGroups.some((mg) => mg.toLowerCase().includes(q))
    );
  }, [exerciseLibrary, exerciseSearch]);

  // Initialize workout (load from localStorage or start fresh)
  useEffect(() => {
    const saved = loadActiveWorkout();
    if (saved) {
      setExercises(saved.exercises);
      setStartedAt(saved.startedAt);
    } else {
      // Check if we were passed a template via sessionStorage
      try {
        const templateRaw = sessionStorage.getItem("ndw_workout_template");
        if (templateRaw) {
          sessionStorage.removeItem("ndw_workout_template");
          const templateExercises = JSON.parse(templateRaw) as Array<{
            exerciseId: string;
            name: string;
            sets: number;
            reps: number;
          }>;
          const preloaded: InProgressExercise[] = templateExercises.map(
            (te) => ({
              exerciseId: te.exerciseId,
              exerciseName: te.name,
              sets: Array.from({ length: te.sets }, (_, i) => ({
                exerciseId: te.exerciseId,
                exerciseName: te.name,
                setNumber: i + 1,
                weightKg: "",
                reps: String(te.reps),
                rpe: "",
                completed: false,
              })),
            })
          );
          setExercises(preloaded);
        }
      } catch {
        // noop
      }
      setStartedAt(new Date().toISOString());
    }
  }, []);

  // Elapsed time ticker
  useEffect(() => {
    if (!startedAt || isPaused) return;
    const id = setInterval(() => {
      setElapsedSec(
        Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000)
      );
    }, 1000);
    return () => clearInterval(id);
  }, [startedAt, isPaused]);

  // Persist to localStorage on every change
  useEffect(() => {
    if (startedAt && exercises.length > 0) {
      saveActiveWorkout({
        startedAt,
        exercises,
        lastUpdated: new Date().toISOString(),
      });
    }
  }, [exercises, startedAt]);

  // Rest timer effect
  useEffect(() => {
    if (!restTimerActive || restRemaining <= 0) {
      if (restIntervalRef.current) {
        clearInterval(restIntervalRef.current);
        restIntervalRef.current = null;
      }
      if (restRemaining <= 0 && restTimerActive) {
        setRestTimerActive(false);
      }
      return;
    }

    restIntervalRef.current = setInterval(() => {
      setRestRemaining((prev) => {
        if (prev <= 1) {
          setRestTimerActive(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => {
      if (restIntervalRef.current) clearInterval(restIntervalRef.current);
    };
  }, [restTimerActive, restRemaining]);

  // ── Exercise Management ─────────────────────────────────────────────

  const addExercise = useCallback(
    (exercise: Exercise) => {
      setExercises((prev) => [
        ...prev,
        {
          exerciseId: exercise.id,
          exerciseName: exercise.name,
          sets: [
            {
              exerciseId: exercise.id,
              exerciseName: exercise.name,
              setNumber: 1,
              weightKg: "",
              reps: "",
              rpe: "",
              completed: false,
            },
          ],
        },
      ]);
      setShowExercisePicker(false);
      setExerciseSearch("");
    },
    []
  );

  const removeExercise = useCallback((exerciseIdx: number) => {
    setExercises((prev) => prev.filter((_, i) => i !== exerciseIdx));
  }, []);

  const addSet = useCallback((exerciseIdx: number) => {
    setExercises((prev) => {
      const updated = [...prev];
      const ex = { ...updated[exerciseIdx] };
      const lastSet = ex.sets[ex.sets.length - 1];
      ex.sets = [
        ...ex.sets,
        {
          exerciseId: ex.exerciseId,
          exerciseName: ex.exerciseName,
          setNumber: ex.sets.length + 1,
          weightKg: lastSet?.weightKg ?? "",
          reps: lastSet?.reps ?? "",
          rpe: "",
          completed: false,
        },
      ];
      updated[exerciseIdx] = ex;
      return updated;
    });
  }, []);

  const updateSet = useCallback(
    (
      exerciseIdx: number,
      setIdx: number,
      field: keyof InProgressSet,
      value: string
    ) => {
      setExercises((prev) => {
        const updated = [...prev];
        const ex = { ...updated[exerciseIdx] };
        const sets = [...ex.sets];
        sets[setIdx] = { ...sets[setIdx], [field]: value };
        ex.sets = sets;
        updated[exerciseIdx] = ex;
        return updated;
      });
    },
    []
  );

  const completeSet = useCallback(
    (exerciseIdx: number, setIdx: number) => {
      setExercises((prev) => {
        const updated = [...prev];
        const ex = { ...updated[exerciseIdx] };
        const sets = [...ex.sets];
        const wasCompleted = sets[setIdx].completed;
        sets[setIdx] = { ...sets[setIdx], completed: !wasCompleted };
        ex.sets = sets;
        updated[exerciseIdx] = ex;
        return updated;
      });

      // Start rest timer if marking as completed
      const set = exercises[exerciseIdx]?.sets[setIdx];
      if (set && !set.completed) {
        setRestTotal(90);
        setRestRemaining(90);
        setRestTimerActive(true);
      }
    },
    [exercises]
  );

  const toggleCollapse = useCallback((idx: number) => {
    setCollapsedExercises((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }, []);

  // ── Finish Workout ──────────────────────────────────────────────────

  const finishWorkout = useCallback(async () => {
    if (!user?.id || isSaving) return;

    const completedExercises = exercises.filter(
      (e) => e.sets.some((s) => s.completed)
    );
    if (completedExercises.length === 0) {
      toast({
        title: "No completed sets",
        description: "Complete at least one set before finishing.",
        variant: "destructive",
      });
      return;
    }

    setIsSaving(true);

    try {
      const endedAt = new Date().toISOString();
      const durationSec = Math.floor(
        (new Date(endedAt).getTime() - new Date(startedAt).getTime()) / 1000
      );
      const durationMin = durationSec / 60;

      // 1. Create workout
      const workoutRes = await apiRequest("POST", "/api/workouts", {
        name: "Strength Training",
        workoutType: "strength",
        startedAt,
        source: "manual",
      });
      const workout = await workoutRes.json();

      // 2. Add all completed sets
      for (const ex of completedExercises) {
        for (const set of ex.sets) {
          if (!set.completed) continue;
          await apiRequest("POST", `/api/workouts/${workout.id}/sets`, {
            exerciseId: set.exerciseId,
            setNumber: set.setNumber,
            reps: parseInt(set.reps, 10) || 0,
            weightKg: parseFloat(set.weightKg) || 0,
            rpe: set.rpe ? parseFloat(set.rpe) : null,
          });
        }
      }

      // 3. Finalize workout (triggers exercise history + strain computation)
      await apiRequest("PUT", `/api/workouts/${workout.id}`, {
        endedAt,
        notes: null,
      });

      // 4. Clear localStorage
      clearActiveWorkout();

      // 5. Invalidate queries
      queryClient.invalidateQueries({
        queryKey: [`/api/workouts/${user.id}`],
      });
      queryClient.invalidateQueries({
        queryKey: [`/api/exercise-history/${user.id}/prs`],
      });

      // 6. Show summary
      setSummaryDuration(durationSec);
      setSummaryPRs([]); // PRs computed server-side; could fetch but keeping it simple
      setShowSummary(true);
    } catch (err) {
      toast({
        title: "Failed to save workout",
        description: err instanceof Error ? err.message : "Unknown error",
        variant: "destructive",
      });
    } finally {
      setIsSaving(false);
    }
  }, [user?.id, exercises, startedAt, isSaving, queryClient, toast]);

  const handleSummaryClose = useCallback(() => {
    setShowSummary(false);
    setLocation("/workout");
  }, [setLocation]);

  // ── Rest Timer Presets ──────────────────────────────────────────────

  const startRestTimer = useCallback((seconds: number) => {
    setRestTotal(seconds);
    setRestRemaining(seconds);
    setRestTimerActive(true);
  }, []);

  return (
    <div className="max-w-lg mx-auto px-4 py-6 space-y-4 pb-32">
      {/* Header + Timer */}
      <motion.div
        className="space-y-3"
        initial={pageTransition.initial}
        animate={pageTransition.animate}
        transition={pageTransition.transition}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-gradient-to-br from-indigo-500 to-purple-600">
              <Dumbbell className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-foreground">
                Workout
              </h1>
              <div className="flex items-center gap-2 mt-0.5">
                <Timer className="h-3 w-3 text-muted-foreground" />
                <span className="text-sm font-mono text-primary tabular-nums">
                  {formatDuration(elapsedSec)}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              className="h-9 w-9 p-0"
              onClick={() => setIsPaused(!isPaused)}
            >
              {isPaused ? (
                <Play className="h-4 w-4" />
              ) : (
                <Pause className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>

        {/* Rest timer presets */}
        <div className="flex gap-2">
          {REST_TIMER_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => startRestTimer(opt.value)}
              className="px-3 py-1.5 rounded-full text-xs font-medium bg-card border border-border text-muted-foreground hover:text-foreground transition-colors"
            >
              Rest {opt.label}
            </button>
          ))}
        </div>
      </motion.div>

      {/* Exercise Blocks */}
      {exercises.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          <Dumbbell className="h-10 w-10 mx-auto mb-3 opacity-40" />
          <p className="text-sm font-medium">No exercises yet</p>
          <p className="text-xs mt-1">
            Add an exercise to start your workout
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {exercises.map((exercise, exIdx) => (
            <ExerciseBlock
              key={`${exercise.exerciseId}-${exIdx}`}
              exercise={exercise}
              onAddSet={() => addSet(exIdx)}
              onUpdateSet={(setIdx, field, value) =>
                updateSet(exIdx, setIdx, field, value)
              }
              onCompleteSet={(setIdx) => completeSet(exIdx, setIdx)}
              onRemoveExercise={() => removeExercise(exIdx)}
              collapsed={collapsedExercises.has(exIdx)}
              onToggleCollapse={() => toggleCollapse(exIdx)}
            />
          ))}
        </div>
      )}

      {/* Bottom Action Buttons */}
      <div className="fixed bottom-20 left-0 right-0 px-4 pb-4 bg-gradient-to-t from-background via-background to-transparent pt-6">
        <div className="max-w-lg mx-auto flex gap-2">
          <Button
            variant="outline"
            className="flex-1 h-12 gap-2"
            onClick={() => setShowExercisePicker(true)}
          >
            <Plus className="h-4 w-4" />
            Add Exercise
          </Button>
          <Button
            className="flex-1 h-12 gap-2"
            onClick={finishWorkout}
            disabled={isSaving}
          >
            {isSaving ? (
              <div className="h-4 w-4 rounded-full border-2 border-primary-foreground/30 border-t-primary-foreground animate-spin" />
            ) : (
              <Check className="h-4 w-4" />
            )}
            Finish Workout
          </Button>
        </div>
      </div>

      {/* Exercise Picker Sheet */}
      <Sheet
        open={showExercisePicker}
        onOpenChange={setShowExercisePicker}
      >
        <SheetContent side="bottom" className="max-h-[80vh] rounded-t-2xl">
          <SheetHeader>
            <SheetTitle>Add Exercise</SheetTitle>
          </SheetHeader>

          <div className="space-y-3 mt-3">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                value={exerciseSearch}
                onChange={(e) => setExerciseSearch(e.target.value)}
                placeholder="Search exercises..."
                className="pl-9 h-10"
                autoFocus
              />
              {exerciseSearch && (
                <button
                  className="absolute right-3 top-1/2 -translate-y-1/2"
                  onClick={() => setExerciseSearch("")}
                >
                  <X className="h-4 w-4 text-muted-foreground" />
                </button>
              )}
            </div>

            {/* Exercise list */}
            <div className="max-h-[50vh] overflow-y-auto space-y-1">
              {filteredLibrary.map((ex) => {
                const alreadyAdded = exercises.some(
                  (e) => e.exerciseId === ex.id
                );
                return (
                  <button
                    key={ex.id}
                    onClick={() => addExercise(ex)}
                    disabled={alreadyAdded}
                    className="w-full flex items-center gap-3 p-2.5 rounded-lg hover:bg-muted/50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-left"
                  >
                    <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 bg-primary/10">
                      <Dumbbell className="h-4 w-4 text-primary" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-foreground truncate">
                        {ex.name}
                      </p>
                      <div className="flex gap-1 mt-0.5">
                        {ex.muscleGroups.slice(0, 2).map((mg) => (
                          <span
                            key={mg}
                            className={`text-[9px] px-1 py-0.5 rounded ${getMuscleGroupColor(mg)}`}
                          >
                            {mg}
                          </span>
                        ))}
                      </div>
                    </div>
                    {alreadyAdded && (
                      <Badge variant="secondary" className="text-[10px]">
                        Added
                      </Badge>
                    )}
                  </button>
                );
              })}

              {filteredLibrary.length === 0 && (
                <p className="text-center text-sm text-muted-foreground py-8">
                  No exercises found
                </p>
              )}
            </div>
          </div>
        </SheetContent>
      </Sheet>

      {/* Rest Timer Overlay */}
      <AnimatePresence>
        {restTimerActive && restRemaining > 0 && (
          <RestTimerCircle
            total={restTotal}
            remaining={restRemaining}
            onDismiss={() => {
              setRestTimerActive(false);
              setRestRemaining(0);
            }}
          />
        )}
      </AnimatePresence>

      {/* Workout Summary */}
      <WorkoutSummary
        open={showSummary}
        onClose={handleSummaryClose}
        exercises={exercises}
        durationSec={summaryDuration}
        newPRs={summaryPRs}
      />
    </div>
  );
}
