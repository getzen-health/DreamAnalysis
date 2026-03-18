import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { Checkbox } from "@/components/ui/checkbox";
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
  DialogTrigger,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { useAuth } from "@/hooks/use-auth";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { motion } from "framer-motion";
import { EmotionStrip } from "@/components/emotion-strip";
import {
  Dumbbell,
  Play,
  Square,
  Plus,
  Minus,
  Search,
  Timer,
  Clock,
  ChevronDown,
  ChevronUp,
  Flame,
  Trophy,
  X,
  RotateCcw,
} from "lucide-react";

/* ---------- types ---------- */

interface Exercise {
  id: string;
  name: string;
  category: string;
  muscleGroups: string[];
  equipment: string | null;
  instructions: string | null;
  videoUrl: string | null;
  isCustom: boolean;
  createdBy: string | null;
}

interface WorkoutSet {
  id?: string;
  localId: string; // client-side key before persisted
  exerciseId: string;
  setNumber: number;
  setType: string;
  reps: number | null;
  weightKg: number | null;
  durationSec: number | null;
  distanceM: number | null;
  rpe: number | null;
  completed: boolean;
}

interface WorkoutExercise {
  exercise: Exercise;
  sets: WorkoutSet[];
}

interface Workout {
  id: string;
  userId: string;
  name: string | null;
  workoutType: string;
  startedAt: string;
  endedAt: string | null;
  durationMin: string | null;
  totalStrain: string | null;
  avgHr: string | null;
  maxHr: string | null;
  caloriesBurned: string | null;
  hrZones: unknown;
  hrRecovery: string | null;
  source: string;
  eegSessionId: string | null;
  notes: string | null;
  createdAt: string;
  sets?: Array<{
    id: string;
    workoutId: string;
    exerciseId: string;
    setNumber: number;
    setType: string;
    reps: number | null;
    weightKg: string | null;
    durationSec: number | null;
    distanceM: string | null;
    rpe: string | null;
    completed: boolean;
    createdAt: string;
  }>;
}

/* ---------- constants ---------- */

const WORKOUT_TYPES = ["Strength", "Cardio", "HIIT", "Flexibility", "Mixed"] as const;
const SET_TYPES = ["normal", "warmup", "dropset"] as const;
const REST_OPTIONS = [60, 90, 120] as const;
const CATEGORIES = ["strength", "cardio", "flexibility", "hiit"] as const;

const SET_TYPE_COLORS: Record<string, string> = {
  normal: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  warmup: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  dropset: "bg-purple-500/20 text-purple-400 border-purple-500/30",
};

const MUSCLE_GROUP_COLORS: Record<string, string> = {
  chest: "bg-red-500/15 text-red-400",
  back: "bg-emerald-500/15 text-emerald-400",
  shoulders: "bg-cyan-500/15 text-cyan-400",
  biceps: "bg-orange-500/15 text-orange-400",
  triceps: "bg-yellow-500/15 text-yellow-400",
  legs: "bg-violet-500/15 text-violet-400",
  quads: "bg-violet-500/15 text-violet-400",
  hamstrings: "bg-fuchsia-500/15 text-fuchsia-400",
  glutes: "bg-pink-500/15 text-pink-400",
  calves: "bg-indigo-500/15 text-indigo-400",
  core: "bg-teal-500/15 text-teal-400",
  abs: "bg-teal-500/15 text-teal-400",
  forearms: "bg-lime-500/15 text-lime-400",
  traps: "bg-sky-500/15 text-sky-400",
  default: "bg-muted text-muted-foreground",
};

function getMuscleColor(muscle: string): string {
  return MUSCLE_GROUP_COLORS[muscle.toLowerCase()] ?? MUSCLE_GROUP_COLORS.default;
}

/* ---------- helpers ---------- */

function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function todayDateStr(): string {
  return new Date().toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

let nextLocalId = 0;
function genLocalId(): string {
  return `local_${Date.now()}_${nextLocalId++}`;
}

/* ---------- Rest Timer Hook ---------- */

function useRestTimer(defaultSeconds: number) {
  const [restSeconds, setRestSeconds] = useState(defaultSeconds);
  const [remaining, setRemaining] = useState(0);
  const [isResting, setIsResting] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startRest = useCallback(() => {
    setRemaining(restSeconds);
    setIsResting(true);
  }, [restSeconds]);

  const stopRest = useCallback(() => {
    setIsResting(false);
    setRemaining(0);
    if (intervalRef.current) clearInterval(intervalRef.current);
  }, []);

  useEffect(() => {
    if (!isResting) return;
    intervalRef.current = setInterval(() => {
      setRemaining((prev) => {
        if (prev <= 1) {
          setIsResting(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isResting]);

  return { remaining, isResting, startRest, stopRest, restSeconds, setRestSeconds };
}

/* ---------- Workout Duration Timer Hook ---------- */

function useWorkoutTimer(startedAt: string | null) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!startedAt) { setElapsed(0); return; }
    const startMs = new Date(startedAt).getTime();
    const tick = () => setElapsed(Math.floor((Date.now() - startMs) / 1000));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [startedAt]);

  return elapsed;
}

/* ========================================================================== */
/*  COMPONENT                                                                 */
/* ========================================================================== */

export default function WorkoutPage() {
  const { user } = useAuth();
  const { toast } = useToast();
  const qc = useQueryClient();

  /* ---- active workout state ---- */
  const [activeWorkoutId, setActiveWorkoutId] = useState<string | null>(null);
  const [activeWorkoutStartedAt, setActiveWorkoutStartedAt] = useState<string | null>(null);
  const [workoutName, setWorkoutName] = useState("");
  const [workoutType, setWorkoutType] = useState<string>("Strength");
  const [exercises, setExercises] = useState<WorkoutExercise[]>([]);
  const [finishNotes, setFinishNotes] = useState("");
  const [showSummary, setShowSummary] = useState(false);
  const [lastSummary, setLastSummary] = useState<{
    duration: string;
    exerciseCount: number;
    totalVolume: number;
    totalSets: number;
    strain: string | null;
  } | null>(null);

  /* ---- exercise picker state ---- */
  const [pickerOpen, setPickerOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [equipmentFilter, setEquipmentFilter] = useState<string>("all");

  /* ---- rest timer ---- */
  const restTimer = useRestTimer(90);

  /* ---- workout timer ---- */
  const elapsedSeconds = useWorkoutTimer(activeWorkoutStartedAt);

  /* ---- history expansion ---- */
  const [expandedWorkoutId, setExpandedWorkoutId] = useState<string | null>(null);

  /* ---- tab ---- */
  const [tab, setTab] = useState("workout");

  /* ================================================================ */
  /*  QUERIES                                                         */
  /* ================================================================ */

  const { data: exerciseList = [] } = useQuery<Exercise[]>({
    queryKey: ["/api/exercises"],
    enabled: !!user,
  });

  const { data: workoutHistory = [] } = useQuery<Workout[]>({
    queryKey: [`/api/workouts/${user?.id}`],
    enabled: !!user?.id,
  });

  /* equipment list derived from exercises */
  const equipmentOptions = useMemo(() => {
    const set = new Set<string>();
    for (const e of exerciseList) {
      if (e.equipment) set.add(e.equipment);
    }
    return Array.from(set).sort();
  }, [exerciseList]);

  /* filtered exercises for picker */
  const filteredExercises = useMemo(() => {
    let list = exerciseList;
    if (categoryFilter !== "all") {
      list = list.filter((e) => e.category.toLowerCase() === categoryFilter);
    }
    if (equipmentFilter !== "all") {
      list = list.filter((e) => e.equipment?.toLowerCase() === equipmentFilter.toLowerCase());
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter(
        (e) =>
          e.name.toLowerCase().includes(q) ||
          e.muscleGroups.some((m) => m.toLowerCase().includes(q)),
      );
    }
    return list;
  }, [exerciseList, categoryFilter, equipmentFilter, searchQuery]);

  /* ================================================================ */
  /*  MUTATIONS                                                       */
  /* ================================================================ */

  const createWorkoutMut = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/workouts", {
        name: workoutName || `Workout ${todayDateStr()}`,
        workoutType,
        startedAt: new Date().toISOString(),
      });
      return res.json() as Promise<Workout>;
    },
    onSuccess: (data) => {
      setActiveWorkoutId(data.id);
      setActiveWorkoutStartedAt(data.startedAt);
      toast({ title: "Workout started", description: data.name ?? "Let's go!" });
    },
    onError: (err: Error) => {
      toast({ title: "Failed to start workout", description: err.message, variant: "destructive" });
    },
  });

  const addSetMut = useMutation({
    mutationFn: async (payload: {
      workoutId: string;
      exerciseId: string;
      setNumber: number;
      setType: string;
      reps: number | null;
      weightKg: number | null;
    }) => {
      const res = await apiRequest("POST", `/api/workouts/${payload.workoutId}/sets`, payload);
      return res.json();
    },
    onError: (err: Error) => {
      toast({ title: "Failed to save set", description: err.message, variant: "destructive" });
    },
  });

  const updateSetMut = useMutation({
    mutationFn: async (payload: { setId: string; data: Record<string, unknown> }) => {
      const res = await apiRequest("PUT", `/api/workout-sets/${payload.setId}`, payload.data);
      return res.json();
    },
  });

  const finishWorkoutMut = useMutation({
    mutationFn: async () => {
      if (!activeWorkoutId) throw new Error("No active workout");
      const res = await apiRequest("PUT", `/api/workouts/${activeWorkoutId}`, {
        endedAt: new Date().toISOString(),
        notes: finishNotes || null,
      });
      return res.json() as Promise<Workout>;
    },
    onSuccess: (data) => {
      // Compute summary
      const totalSets = exercises.reduce((sum, we) => sum + we.sets.filter((s) => s.completed).length, 0);
      const totalVolume = exercises.reduce(
        (sum, we) =>
          sum +
          we.sets
            .filter((s) => s.completed)
            .reduce((v, s) => v + (s.reps ?? 0) * (s.weightKg ?? 0), 0),
        0,
      );
      setLastSummary({
        duration: formatDuration(elapsedSeconds),
        exerciseCount: exercises.length,
        totalVolume: Math.round(totalVolume),
        totalSets,
        strain: data.totalStrain,
      });
      setShowSummary(true);
      // Reset active workout
      setActiveWorkoutId(null);
      setActiveWorkoutStartedAt(null);
      setExercises([]);
      setFinishNotes("");
      // Refresh history
      qc.invalidateQueries({ queryKey: [`/api/workouts/${user?.id}`] });
      toast({ title: "Workout complete!" });
    },
    onError: (err: Error) => {
      toast({ title: "Failed to finish workout", description: err.message, variant: "destructive" });
    },
  });

  /* ================================================================ */
  /*  HANDLERS                                                        */
  /* ================================================================ */

  function addExerciseToWorkout(exercise: Exercise) {
    // Prevent duplicates
    if (exercises.some((we) => we.exercise.id === exercise.id)) {
      toast({ title: "Already added", description: `${exercise.name} is already in this workout.` });
      return;
    }
    setExercises((prev) => [
      ...prev,
      {
        exercise,
        sets: [
          {
            localId: genLocalId(),
            exerciseId: exercise.id,
            setNumber: 1,
            setType: "normal",
            reps: null,
            weightKg: null,
            durationSec: null,
            distanceM: null,
            rpe: null,
            completed: false,
          },
        ],
      },
    ]);
    setPickerOpen(false);
  }

  function removeExercise(exerciseId: string) {
    setExercises((prev) => prev.filter((we) => we.exercise.id !== exerciseId));
  }

  function addSet(exerciseId: string) {
    setExercises((prev) =>
      prev.map((we) => {
        if (we.exercise.id !== exerciseId) return we;
        const newSetNum = we.sets.length + 1;
        // Copy reps/weight from last set as default
        const lastSet = we.sets[we.sets.length - 1];
        return {
          ...we,
          sets: [
            ...we.sets,
            {
              localId: genLocalId(),
              exerciseId,
              setNumber: newSetNum,
              setType: "normal",
              reps: lastSet?.reps ?? null,
              weightKg: lastSet?.weightKg ?? null,
              durationSec: null,
              distanceM: null,
              rpe: null,
              completed: false,
            },
          ],
        };
      }),
    );
  }

  function removeSet(exerciseId: string, localId: string) {
    setExercises((prev) =>
      prev.map((we) => {
        if (we.exercise.id !== exerciseId) return we;
        const filtered = we.sets.filter((s) => s.localId !== localId);
        // Re-number
        return {
          ...we,
          sets: filtered.map((s, i) => ({ ...s, setNumber: i + 1 })),
        };
      }),
    );
  }

  function updateSetField(
    exerciseId: string,
    localId: string,
    field: keyof WorkoutSet,
    value: unknown,
  ) {
    setExercises((prev) =>
      prev.map((we) => {
        if (we.exercise.id !== exerciseId) return we;
        return {
          ...we,
          sets: we.sets.map((s) => (s.localId === localId ? { ...s, [field]: value } : s)),
        };
      }),
    );
  }

  async function completeSet(exerciseId: string, localId: string) {
    const we = exercises.find((w) => w.exercise.id === exerciseId);
    const set = we?.sets.find((s) => s.localId === localId);
    if (!set || !activeWorkoutId) return;

    const newCompleted = !set.completed;
    updateSetField(exerciseId, localId, "completed", newCompleted);

    // Persist to server
    if (newCompleted && !set.id) {
      try {
        const result = await addSetMut.mutateAsync({
          workoutId: activeWorkoutId,
          exerciseId: set.exerciseId,
          setNumber: set.setNumber,
          setType: set.setType,
          reps: set.reps,
          weightKg: set.weightKg,
        });
        // Store server ID
        setExercises((prev) =>
          prev.map((w) => ({
            ...w,
            sets: w.sets.map((s) =>
              s.localId === localId ? { ...s, id: result.id } : s,
            ),
          })),
        );
        // Start rest timer
        restTimer.startRest();
      } catch {
        // Revert
        updateSetField(exerciseId, localId, "completed", false);
      }
    } else if (newCompleted && set.id) {
      // Already persisted, just update completed
      updateSetMut.mutate({ setId: set.id, data: { completed: true, reps: set.reps, weightKg: set.weightKg } });
      restTimer.startRest();
    } else if (!newCompleted && set.id) {
      updateSetMut.mutate({ setId: set.id, data: { completed: false } });
    }
  }

  /* ---- expand workout in history ---- */
  const { data: expandedWorkoutData } = useQuery<Workout>({
    queryKey: [`/api/workouts/${user?.id}/${expandedWorkoutId}`],
    enabled: !!user?.id && !!expandedWorkoutId,
  });

  /* ================================================================ */
  /*  RENDER: Start Workout                                           */
  /* ================================================================ */

  function renderStartWorkout() {
    return (
      <Card className="border-border/50 bg-card/80">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Dumbbell className="h-5 w-5 text-primary" />
            Start Workout
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="workout-name" className="text-sm text-muted-foreground">
              Workout Name (optional)
            </Label>
            <Input
              id="workout-name"
              placeholder={`Workout ${todayDateStr()}`}
              value={workoutName}
              onChange={(e) => setWorkoutName(e.target.value)}
              className="h-12 text-base"
            />
          </div>

          <div className="space-y-2">
            <Label className="text-sm text-muted-foreground">Workout Type</Label>
            <div className="flex flex-wrap gap-2">
              {WORKOUT_TYPES.map((t) => (
                <Button
                  key={t}
                  variant={workoutType === t ? "default" : "outline"}
                  size="sm"
                  onClick={() => setWorkoutType(t)}
                  className="min-h-[40px] px-4"
                >
                  {t}
                </Button>
              ))}
            </div>
          </div>

          <Button
            className="w-full h-14 text-base font-semibold gap-2"
            onClick={() => createWorkoutMut.mutate()}
            disabled={createWorkoutMut.isPending}
          >
            <Play className="h-5 w-5" />
            {createWorkoutMut.isPending ? "Starting..." : "Start Workout"}
          </Button>
        </CardContent>
      </Card>
    );
  }

  /* ================================================================ */
  /*  RENDER: Exercise Picker Dialog                                   */
  /* ================================================================ */

  function renderExercisePicker() {
    return (
      <Dialog open={pickerOpen} onOpenChange={setPickerOpen}>
        <DialogTrigger asChild>
          <Button variant="outline" className="w-full h-12 gap-2 text-base border-dashed border-2">
            <Plus className="h-5 w-5" />
            Add Exercise
          </Button>
        </DialogTrigger>
        <DialogContent className="max-w-lg max-h-[85vh] flex flex-col">
          <DialogHeader>
            <DialogTitle>Add Exercise</DialogTitle>
          </DialogHeader>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search exercises..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 h-11"
            />
          </div>

          {/* Filters */}
          <div className="flex gap-2">
            <Select value={categoryFilter} onValueChange={setCategoryFilter}>
              <SelectTrigger className="h-9 w-[130px]">
                <SelectValue placeholder="Category" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Categories</SelectItem>
                {CATEGORIES.map((c) => (
                  <SelectItem key={c} value={c}>
                    {c.charAt(0).toUpperCase() + c.slice(1)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={equipmentFilter} onValueChange={setEquipmentFilter}>
              <SelectTrigger className="h-9 w-[130px]">
                <SelectValue placeholder="Equipment" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Equipment</SelectItem>
                {equipmentOptions.map((eq) => (
                  <SelectItem key={eq} value={eq}>
                    {eq}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Exercise List */}
          <div className="flex-1 overflow-y-auto space-y-1 min-h-0 -mx-1 px-1">
            {filteredExercises.length === 0 && (
              <p className="text-center text-sm text-muted-foreground py-8">
                {exerciseList.length === 0
                  ? "No exercises in the database yet."
                  : "No exercises match your filters."}
              </p>
            )}
            {filteredExercises.map((ex) => (
              <button
                key={ex.id}
                onClick={() => addExerciseToWorkout(ex)}
                className="w-full text-left px-3 py-3 rounded-lg hover:bg-muted/60 active:bg-muted transition-colors"
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-sm">{ex.name}</span>
                  <Plus className="h-4 w-4 text-muted-foreground" />
                </div>
                <div className="flex flex-wrap gap-1 mt-1.5">
                  {ex.muscleGroups.map((mg) => (
                    <Badge
                      key={mg}
                      variant="outline"
                      className={`text-[10px] px-1.5 py-0 border-0 ${getMuscleColor(mg)}`}
                    >
                      {mg}
                    </Badge>
                  ))}
                  {ex.equipment && (
                    <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                      {ex.equipment}
                    </Badge>
                  )}
                </div>
              </button>
            ))}
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  /* ================================================================ */
  /*  RENDER: Active Workout View                                      */
  /* ================================================================ */

  function renderActiveWorkout() {
    return (
      <div className="space-y-4">
        {/* Header: duration + rest timer + finish */}
        <Card className="border-border/50 bg-card/80">
          <CardContent className="pt-4 pb-3">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h2 className="text-lg font-semibold">
                  {workoutName || `Workout ${todayDateStr()}`}
                </h2>
                <div className="flex items-center gap-3 mt-1">
                  <Badge variant="outline" className="text-xs">{workoutType}</Badge>
                  <span className="flex items-center gap-1 text-sm font-mono text-primary">
                    <Clock className="h-3.5 w-3.5" />
                    {formatDuration(elapsedSeconds)}
                  </span>
                </div>
              </div>
              <Button
                variant="destructive"
                size="sm"
                className="h-10 px-4 gap-1.5 font-semibold"
                onClick={() => finishWorkoutMut.mutate()}
                disabled={finishWorkoutMut.isPending}
              >
                <Square className="h-4 w-4" />
                Finish
              </Button>
            </div>

            {/* Rest Timer -- circular countdown */}
            {restTimer.isResting && (
              <motion.div
                className="rounded-xl bg-ndw-strain/10 border border-ndw-strain/20 p-4 flex items-center justify-between"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.25 }}
              >
                <div className="flex items-center gap-4">
                  {/* Circular countdown */}
                  <div className="relative w-14 h-14 shrink-0">
                    <svg className="w-14 h-14 -rotate-90" viewBox="0 0 56 56">
                      <circle
                        cx="28" cy="28" r="24"
                        fill="none"
                        stroke="var(--border)"
                        strokeWidth="4"
                      />
                      <circle
                        cx="28" cy="28" r="24"
                        fill="none"
                        stroke="#f87171"
                        strokeWidth="4"
                        strokeLinecap="round"
                        strokeDasharray={2 * Math.PI * 24}
                        strokeDashoffset={2 * Math.PI * 24 * (1 - restTimer.remaining / restTimer.restSeconds)}
                        className="transition-all duration-1000 ease-linear"
                      />
                    </svg>
                    <span className="absolute inset-0 flex items-center justify-center text-sm font-mono font-bold text-ndw-strain">
                      {restTimer.remaining}
                    </span>
                  </div>
                  <div>
                    <span className="text-sm font-semibold text-foreground">Rest</span>
                    <p className="text-xs text-muted-foreground">{formatDuration(restTimer.remaining)} remaining</p>
                  </div>
                </div>
                <Button variant="ghost" size="sm" onClick={restTimer.stopRest} className="h-8 text-muted-foreground hover:text-foreground">
                  <X className="h-4 w-4 mr-1" />
                  Skip
                </Button>
              </motion.div>
            )}

            {/* Rest duration selector */}
            {!restTimer.isResting && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Timer className="h-3.5 w-3.5" />
                <span>Rest:</span>
                {REST_OPTIONS.map((sec) => (
                  <button
                    key={sec}
                    onClick={() => restTimer.setRestSeconds(sec)}
                    className={`px-2 py-1 rounded-md transition-colors ${
                      restTimer.restSeconds === sec
                        ? "bg-primary/20 text-primary font-medium"
                        : "hover:bg-muted"
                    }`}
                  >
                    {sec}s
                  </button>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Exercise List */}
        {exercises.map((we, idx) => (
          <motion.div
            key={we.exercise.id}
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: idx * 0.05 }}
          >
          <Card className="border-border/50 bg-card/80 overflow-hidden">
            <CardHeader className="pb-2 pt-3 px-4">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-base">{we.exercise.name}</CardTitle>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {we.exercise.muscleGroups.map((mg) => (
                      <Badge
                        key={mg}
                        variant="outline"
                        className={`text-[10px] px-1.5 py-0 border-0 ${getMuscleColor(mg)}`}
                      >
                        {mg}
                      </Badge>
                    ))}
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-muted-foreground hover:text-destructive"
                  onClick={() => removeExercise(we.exercise.id)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>

            <CardContent className="px-4 pb-3">
              {/* Set Headers */}
              <div className="grid grid-cols-[40px_1fr_1fr_60px_36px] gap-2 text-[11px] text-muted-foreground font-medium mb-1 px-1">
                <span>Set</span>
                <span>kg</span>
                <span>Reps</span>
                <span>Type</span>
                <span></span>
              </div>

              {/* Sets */}
              {we.sets.map((set) => (
                <div
                  key={set.localId}
                  className={`grid grid-cols-[40px_1fr_1fr_60px_36px] gap-2 items-center py-1.5 px-1 rounded-md transition-colors ${
                    set.completed ? "bg-primary/5" : ""
                  }`}
                >
                  {/* Set number + checkbox */}
                  <div className="flex items-center gap-1">
                    <Checkbox
                      checked={set.completed}
                      onCheckedChange={() => completeSet(we.exercise.id, set.localId)}
                      className="h-5 w-5"
                    />
                    <span className="text-xs text-muted-foreground">{set.setNumber}</span>
                  </div>

                  {/* Weight */}
                  <Input
                    type="number"
                    inputMode="decimal"
                    placeholder="0"
                    value={set.weightKg ?? ""}
                    onChange={(e) =>
                      updateSetField(
                        we.exercise.id,
                        set.localId,
                        "weightKg",
                        e.target.value ? parseFloat(e.target.value) : null,
                      )
                    }
                    className="h-11 text-base text-center font-mono [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                    disabled={set.completed}
                  />

                  {/* Reps */}
                  <Input
                    type="number"
                    inputMode="numeric"
                    placeholder="0"
                    value={set.reps ?? ""}
                    onChange={(e) =>
                      updateSetField(
                        we.exercise.id,
                        set.localId,
                        "reps",
                        e.target.value ? parseInt(e.target.value, 10) : null,
                      )
                    }
                    className="h-11 text-base text-center font-mono [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                    disabled={set.completed}
                  />

                  {/* Set type */}
                  <Select
                    value={set.setType}
                    onValueChange={(val) =>
                      updateSetField(we.exercise.id, set.localId, "setType", val)
                    }
                    disabled={set.completed}
                  >
                    <SelectTrigger className="h-9 text-[10px] px-1.5">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {SET_TYPES.map((st) => (
                        <SelectItem key={st} value={st} className="text-xs">
                          {st}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>

                  {/* Remove set */}
                  {!set.completed && we.sets.length > 1 && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-muted-foreground hover:text-destructive"
                      onClick={() => removeSet(we.exercise.id, set.localId)}
                    >
                      <Minus className="h-3.5 w-3.5" />
                    </Button>
                  )}
                  {(set.completed || we.sets.length <= 1) && <div />}
                </div>
              ))}

              {/* Add Set Button */}
              <Button
                variant="ghost"
                size="sm"
                className="w-full mt-1 text-xs h-9 text-muted-foreground hover:text-foreground gap-1"
                onClick={() => addSet(we.exercise.id)}
              >
                <Plus className="h-3.5 w-3.5" />
                Add Set
              </Button>
            </CardContent>
          </Card>
          </motion.div>
        ))}

        {/* Add Exercise */}
        {renderExercisePicker()}

        {/* Finish Notes */}
        {exercises.length > 0 && (
          <div className="space-y-2">
            <Label className="text-sm text-muted-foreground">Notes (optional)</Label>
            <Textarea
              placeholder="How did the workout feel?"
              value={finishNotes}
              onChange={(e) => setFinishNotes(e.target.value)}
              className="min-h-[60px] resize-none"
            />
          </div>
        )}
      </div>
    );
  }

  /* ================================================================ */
  /*  RENDER: Workout Summary                                          */
  /* ================================================================ */

  function renderSummary() {
    if (!lastSummary) return null;
    return (
      <Dialog open={showSummary} onOpenChange={setShowSummary}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Trophy className="h-5 w-5 text-amber-400" />
              Workout Complete
            </DialogTitle>
          </DialogHeader>

          <div className="grid grid-cols-2 gap-4 py-2">
            <div className="text-center">
              <p className="text-2xl font-bold font-mono">{lastSummary.duration}</p>
              <p className="text-xs text-muted-foreground">Duration</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold">{lastSummary.exerciseCount}</p>
              <p className="text-xs text-muted-foreground">Exercises</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold">{lastSummary.totalSets}</p>
              <p className="text-xs text-muted-foreground">Sets</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold">
                {lastSummary.totalVolume.toLocaleString()}
                <span className="text-sm font-normal text-muted-foreground ml-0.5">kg</span>
              </p>
              <p className="text-xs text-muted-foreground">Total Volume</p>
            </div>
          </div>

          {lastSummary.strain && (
            <div className="flex items-center justify-center gap-2 py-2 rounded-lg bg-orange-500/10 border border-orange-500/20">
              <Flame className="h-5 w-5 text-orange-400" />
              <span className="text-lg font-bold text-orange-400">
                {parseFloat(lastSummary.strain).toFixed(1)}
              </span>
              <span className="text-xs text-muted-foreground">Strain</span>
            </div>
          )}

          <Button className="w-full mt-2" onClick={() => setShowSummary(false)}>
            Done
          </Button>
        </DialogContent>
      </Dialog>
    );
  }

  /* ================================================================ */
  /*  RENDER: Workout History                                          */
  /* ================================================================ */

  function renderHistory() {
    if (workoutHistory.length === 0) {
      return (
        <div className="text-center py-12 text-muted-foreground">
          <Dumbbell className="h-10 w-10 mx-auto mb-3 opacity-30" />
          <p className="text-sm">No workouts yet.</p>
          <p className="text-xs mt-1">Start your first workout to see history here.</p>
        </div>
      );
    }

    return (
      <div className="space-y-2">
        {workoutHistory.map((w) => {
          const isExpanded = expandedWorkoutId === w.id;
          const startDate = new Date(w.startedAt);
          const dateStr = startDate.toLocaleDateString("en-US", {
            weekday: "short",
            month: "short",
            day: "numeric",
          });
          const durationMin = w.durationMin ? parseFloat(w.durationMin) : null;

          return (
            <Card key={w.id} className="border-border/50 bg-card/80">
              <button
                className="w-full text-left px-4 py-3"
                onClick={() => setExpandedWorkoutId(isExpanded ? null : w.id)}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-sm">{w.name || "Workout"}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs text-muted-foreground">{dateStr}</span>
                      <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                        {w.workoutType}
                      </Badge>
                      {durationMin !== null && (
                        <span className="text-xs text-muted-foreground">
                          {durationMin < 60
                            ? `${Math.round(durationMin)}m`
                            : `${Math.floor(durationMin / 60)}h ${Math.round(durationMin % 60)}m`}
                        </span>
                      )}
                    </div>
                  </div>
                  {isExpanded ? (
                    <ChevronUp className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                  )}
                </div>
              </button>

              {isExpanded && expandedWorkoutData?.id === w.id && expandedWorkoutData.sets && (
                <CardContent className="pt-0 pb-3 px-4">
                  <Separator className="mb-3" />
                  {expandedWorkoutData.sets.length === 0 ? (
                    <p className="text-xs text-muted-foreground">No sets recorded.</p>
                  ) : (
                    <div className="space-y-1">
                      {/* Group sets by exerciseId */}
                      {Array.from(
                        expandedWorkoutData.sets.reduce((map, s) => {
                          const key = s.exerciseId || "unknown";
                          if (!map.has(key)) map.set(key, []);
                          map.get(key)!.push(s);
                          return map;
                        }, new Map<string, typeof expandedWorkoutData.sets>()),
                      ).map(([exerciseId, sets]) => {
                        const ex = exerciseList.find((e) => e.id === exerciseId);
                        return (
                          <div key={exerciseId} className="mb-2">
                            <p className="text-xs font-medium mb-1">
                              {ex?.name ?? "Unknown exercise"}
                            </p>
                            {sets.map((s) => (
                              <div
                                key={s.id}
                                className="flex items-center gap-3 text-xs text-muted-foreground pl-2"
                              >
                                <span className="w-10">Set {s.setNumber}</span>
                                <span>
                                  {s.weightKg ?? "—"}kg x {s.reps ?? "—"}
                                </span>
                                <Badge
                                  variant="outline"
                                  className={`text-[9px] px-1 py-0 border ${
                                    SET_TYPE_COLORS[s.setType] ?? ""
                                  }`}
                                >
                                  {s.setType}
                                </Badge>
                              </div>
                            ))}
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {w.notes && (
                    <p className="text-xs text-muted-foreground mt-2 italic">
                      {w.notes}
                    </p>
                  )}
                </CardContent>
              )}
            </Card>
          );
        })}
      </div>
    );
  }

  /* ================================================================ */
  /*  MAIN RENDER                                                      */
  /* ================================================================ */

  return (
    <div className="max-w-lg mx-auto px-4 py-6 space-y-4">
      <motion.div
        className="space-y-2 mb-2"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-gradient-to-br from-ndw-recovery to-ndw-stress">
            <Dumbbell className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-foreground">Strength Builder</h1>
            <p className="text-xs text-muted-foreground">Log workouts and track progress</p>
          </div>
        </div>
        <EmotionStrip />
      </motion.div>

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="w-full">
          <TabsTrigger value="workout" className="flex-1">
            <Dumbbell className="h-4 w-4 mr-1.5" />
            Workout
          </TabsTrigger>
          <TabsTrigger value="history" className="flex-1">
            <Clock className="h-4 w-4 mr-1.5" />
            History
          </TabsTrigger>
        </TabsList>

        <TabsContent value="workout" className="mt-4 space-y-4">
          {activeWorkoutId ? renderActiveWorkout() : renderStartWorkout()}
        </TabsContent>

        <TabsContent value="history" className="mt-4">
          {renderHistory()}
        </TabsContent>
      </Tabs>

      {/* Summary Dialog */}
      {renderSummary()}
    </div>
  );
}
