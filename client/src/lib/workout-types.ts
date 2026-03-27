/**
 * workout-types.ts — Shared types and helpers for the Strength Builder feature.
 *
 * Types mirror the DB schema in shared/schema.ts (exercises, workouts, workout_sets,
 * workout_templates, exercise_history).
 *
 * localStorage helpers provide crash recovery for in-progress workouts.
 */

// ── Exercise ──────────────────────────────────────────────────────────────

export interface Exercise {
  id: string;
  name: string;
  category: string;
  muscleGroups: string[];
  equipment: string | null;
  instructions: string | null;
  videoUrl: string | null;
  isCustom: boolean | null;
  createdBy: string | null;
}

// ── Workout ───────────────────────────────────────────────────────────────

export interface Workout {
  id: string;
  userId: string | null;
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
  createdAt: string | null;
}

export interface WorkoutWithSets extends Workout {
  sets: WorkoutSet[];
}

// ── Workout Set ───────────────────────────────────────────────────────────

export interface WorkoutSet {
  id: string;
  workoutId: string | null;
  exerciseId: string | null;
  setNumber: number;
  setType: string | null;
  reps: number | null;
  weightKg: string | null;
  durationSec: number | null;
  restSec: number | null;
  rpe: string | null;
  completed: boolean | null;
  createdAt: string | null;
}

// ── Workout Template ──────────────────────────────────────────────────────

export interface TemplateExercise {
  exerciseId: string;
  name: string;
  sets: number;
  reps: number;
  restSec: number;
}

export interface WorkoutTemplate {
  id: string;
  userId: string | null;
  name: string;
  description: string | null;
  exercises: TemplateExercise[];
  isAiGenerated: boolean | null;
  timesUsed: number | null;
  createdAt: string | null;
}

// ── Exercise History / Personal Records ───────────────────────────────────

export interface ExerciseHistoryRecord {
  date: string;
  bestWeightKg: number | null;
  bestReps: number | null;
  estimated1rm: number | null;
  totalVolume: number | null;
}

export interface PersonalRecord {
  exerciseId: string | null;
  exerciseName: string;
  estimated1rm: number | null;
  bestWeightKg: number | null;
  bestReps: number | null;
  date: string | null;
}

// ── In-Progress Workout (localStorage crash recovery) ─────────────────────

export interface InProgressSet {
  exerciseId: string;
  exerciseName: string;
  setNumber: number;
  weightKg: string;
  reps: string;
  rpe: string;
  completed: boolean;
}

export interface InProgressExercise {
  exerciseId: string;
  exerciseName: string;
  sets: InProgressSet[];
}

export interface InProgressWorkout {
  startedAt: string;
  exercises: InProgressExercise[];
  lastUpdated: string;
}

const STORAGE_KEY = "ndw_active_workout";

export function saveActiveWorkout(workout: InProgressWorkout): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      ...workout,
      lastUpdated: new Date().toISOString(),
    }));
  } catch {
    // localStorage may be full or unavailable
  }
}

export function loadActiveWorkout(): InProgressWorkout | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as InProgressWorkout;
  } catch {
    return null;
  }
}

export function clearActiveWorkout(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // noop
  }
}

// ── Category / Muscle Group Constants ─────────────────────────────────────

export const EXERCISE_CATEGORIES = [
  "All",
  "Strength",
  "Cardio",
  "Flexibility",
  "Compound",
  "Isolation",
] as const;

export const MUSCLE_GROUPS = [
  "Chest",
  "Back",
  "Legs",
  "Shoulders",
  "Arms",
  "Core",
] as const;

export const RPE_OPTIONS = ["6", "7", "8", "9", "10"] as const;

export const REST_TIMER_OPTIONS = [
  { label: "60s", value: 60 },
  { label: "90s", value: 90 },
  { label: "120s", value: 120 },
] as const;

// ── Prebuilt Templates ────────────────────────────────────────────────────

export const PREBUILT_TEMPLATES = [
  {
    name: "Push Day",
    description: "Chest, shoulders, triceps",
    muscleGroups: ["Chest", "Shoulders", "Arms"],
    estimatedMin: 60,
    exercises: [
      { name: "Bench Press", sets: 4, reps: 8 },
      { name: "Overhead Press", sets: 3, reps: 10 },
      { name: "Incline Dumbbell Press", sets: 3, reps: 10 },
      { name: "Lateral Raises", sets: 3, reps: 15 },
      { name: "Tricep Pushdowns", sets: 3, reps: 12 },
    ],
  },
  {
    name: "Pull Day",
    description: "Back, biceps, rear delts",
    muscleGroups: ["Back", "Arms"],
    estimatedMin: 55,
    exercises: [
      { name: "Deadlift", sets: 3, reps: 5 },
      { name: "Pull-ups", sets: 4, reps: 8 },
      { name: "Barbell Rows", sets: 3, reps: 10 },
      { name: "Face Pulls", sets: 3, reps: 15 },
      { name: "Bicep Curls", sets: 3, reps: 12 },
    ],
  },
  {
    name: "Leg Day",
    description: "Quads, hamstrings, glutes, calves",
    muscleGroups: ["Legs"],
    estimatedMin: 65,
    exercises: [
      { name: "Squats", sets: 4, reps: 8 },
      { name: "Romanian Deadlift", sets: 3, reps: 10 },
      { name: "Leg Press", sets: 3, reps: 12 },
      { name: "Walking Lunges", sets: 3, reps: 12 },
      { name: "Calf Raises", sets: 4, reps: 15 },
    ],
  },
  {
    name: "Upper Body",
    description: "Full upper body compound + isolation",
    muscleGroups: ["Chest", "Back", "Shoulders", "Arms"],
    estimatedMin: 60,
    exercises: [
      { name: "Bench Press", sets: 4, reps: 8 },
      { name: "Pull-ups", sets: 4, reps: 8 },
      { name: "Overhead Press", sets: 3, reps: 10 },
      { name: "Barbell Rows", sets: 3, reps: 10 },
      { name: "Bicep Curls", sets: 2, reps: 12 },
      { name: "Tricep Pushdowns", sets: 2, reps: 12 },
    ],
  },
  {
    name: "Full Body",
    description: "Hit every major muscle group",
    muscleGroups: ["Chest", "Back", "Legs", "Shoulders"],
    estimatedMin: 70,
    exercises: [
      { name: "Squats", sets: 3, reps: 8 },
      { name: "Bench Press", sets: 3, reps: 8 },
      { name: "Barbell Rows", sets: 3, reps: 10 },
      { name: "Overhead Press", sets: 3, reps: 10 },
      { name: "Romanian Deadlift", sets: 3, reps: 10 },
      { name: "Plank", sets: 3, reps: 60 },
    ],
  },
] as const;

// ── Helpers ───────────────────────────────────────────────────────────────

export function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function calculateVolume(sets: Array<{ weightKg: string; reps: string; completed: boolean }>): number {
  return sets.reduce((total, s) => {
    if (!s.completed) return total;
    const w = parseFloat(s.weightKg) || 0;
    const r = parseInt(s.reps, 10) || 0;
    return total + w * r;
  }, 0);
}

/** Muscle group badge colors */
export function getMuscleGroupColor(group: string): string {
  const map: Record<string, string> = {
    Chest: "bg-rose-500/15 text-rose-400 border-rose-500/20",
    Back: "bg-blue-500/15 text-blue-400 border-blue-500/20",
    Legs: "bg-green-500/15 text-green-400 border-green-500/20",
    Shoulders: "bg-amber-500/15 text-amber-400 border-amber-500/20",
    Arms: "bg-purple-500/15 text-purple-400 border-purple-500/20",
    Core: "bg-orange-500/15 text-orange-400 border-orange-500/20",
    Glutes: "bg-pink-500/15 text-pink-400 border-pink-500/20",
    Calves: "bg-teal-500/15 text-teal-400 border-teal-500/20",
  };
  return map[group] ?? "bg-zinc-500/15 text-zinc-400 border-zinc-500/20";
}

/** Category badge color */
export function getCategoryColor(category: string): string {
  const map: Record<string, string> = {
    Strength: "bg-indigo-500/15 text-indigo-400",
    Cardio: "bg-rose-500/15 text-rose-400",
    Flexibility: "bg-emerald-500/15 text-emerald-400",
    Compound: "bg-amber-500/15 text-amber-400",
    Isolation: "bg-cyan-500/15 text-cyan-400",
  };
  return map[category] ?? "bg-zinc-500/15 text-zinc-400";
}
