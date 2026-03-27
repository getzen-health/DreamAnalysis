import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  Search,
  Dumbbell,
  ChevronRight,
  ArrowLeft,
  X,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import type { Exercise } from "@/lib/workout-types";
import {
  EXERCISE_CATEGORIES,
  MUSCLE_GROUPS,
  getMuscleGroupColor,
  getCategoryColor,
} from "@/lib/workout-types";

/* ---------- Component ---------- */

export default function ExerciseLibrary() {
  const [, setLocation] = useLocation();
  const [search, setSearch] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [selectedMuscle, setSelectedMuscle] = useState<string | null>(null);
  const [previewExercise, setPreviewExercise] = useState<Exercise | null>(null);

  // Fetch all exercises
  const { data: exercises = [], isLoading } = useQuery<Exercise[]>({
    queryKey: ["/api/exercises"],
    staleTime: 5 * 60_000,
  });

  // Filter exercises
  const filtered = useMemo(() => {
    let result = exercises;

    // Category filter
    if (selectedCategory !== "All") {
      result = result.filter(
        (e) => e.category.toLowerCase() === selectedCategory.toLowerCase()
      );
    }

    // Muscle group filter
    if (selectedMuscle) {
      result = result.filter((e) =>
        e.muscleGroups.some(
          (mg) => mg.toLowerCase() === selectedMuscle.toLowerCase()
        )
      );
    }

    // Search filter
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(
        (e) =>
          e.name.toLowerCase().includes(q) ||
          e.muscleGroups.some((mg) => mg.toLowerCase().includes(q)) ||
          (e.equipment && e.equipment.toLowerCase().includes(q))
      );
    }

    return result;
  }, [exercises, selectedCategory, selectedMuscle, search]);

  return (
    <div className="max-w-lg mx-auto px-4 py-6 space-y-4 pb-24">
      {/* Header */}
      <motion.div
        className="space-y-3"
        initial={pageTransition.initial}
        animate={pageTransition.animate}
        transition={pageTransition.transition}
      >
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            className="h-9 w-9 p-0"
            onClick={() => setLocation("/workout")}
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-foreground flex items-center gap-2">
              <Dumbbell className="h-5 w-5 text-primary" />
              Exercise Library
            </h1>
            <p className="text-xs text-muted-foreground mt-0.5">
              {exercises.length} exercises available
            </p>
          </div>
        </div>

        {/* Search bar */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search exercises..."
            className="pl-9 h-10 bg-card border-border"
          />
          {search && (
            <button
              className="absolute right-3 top-1/2 -translate-y-1/2"
              onClick={() => setSearch("")}
            >
              <X className="h-4 w-4 text-muted-foreground" />
            </button>
          )}
        </div>

        {/* Category filter chips */}
        <div className="flex gap-2 overflow-x-auto no-scrollbar pb-1">
          {EXERCISE_CATEGORIES.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-colors ${
                selectedCategory === cat
                  ? "bg-primary text-primary-foreground"
                  : "bg-card border border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* Muscle group filter chips */}
        <div className="flex gap-2 overflow-x-auto no-scrollbar pb-1">
          {MUSCLE_GROUPS.map((mg) => (
            <button
              key={mg}
              onClick={() =>
                setSelectedMuscle(selectedMuscle === mg ? null : mg)
              }
              className={`px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-colors ${
                selectedMuscle === mg
                  ? "bg-primary text-primary-foreground"
                  : "bg-card border border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              {mg}
            </button>
          ))}
        </div>
      </motion.div>

      {/* Exercise Grid */}
      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="rounded-xl border border-border bg-card p-4 animate-pulse"
            >
              <div className="h-4 bg-muted rounded w-1/3 mb-2" />
              <div className="h-3 bg-muted rounded w-1/2" />
            </div>
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          <Dumbbell className="h-10 w-10 mx-auto mb-3 opacity-40" />
          <p className="text-sm font-medium">No exercises found</p>
          <p className="text-xs mt-1">
            {search
              ? "Try a different search term"
              : "No exercises match the selected filters"}
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          <AnimatePresence mode="popLayout">
            {filtered.map((exercise, idx) => (
              <motion.div
                key={exercise.id}
                custom={idx}
                initial="hidden"
                animate="visible"
                exit="hidden"
                variants={cardVariants}
                layout
                onClick={() => setPreviewExercise(exercise)}
                className="rounded-xl border border-border bg-card p-3.5 flex items-center gap-3 cursor-pointer hover:bg-card/80 transition-colors active:scale-[0.98]"
              >
                {/* Exercise icon */}
                <div className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0 bg-primary/10">
                  <Dumbbell className="h-5 w-5 text-primary" />
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-foreground truncate">
                    {exercise.name}
                  </p>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {exercise.muscleGroups.slice(0, 3).map((mg) => (
                      <span
                        key={mg}
                        className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium border ${getMuscleGroupColor(mg)}`}
                      >
                        {mg}
                      </span>
                    ))}
                    {exercise.muscleGroups.length > 3 && (
                      <span className="text-[10px] text-muted-foreground">
                        +{exercise.muscleGroups.length - 3}
                      </span>
                    )}
                  </div>
                  {exercise.equipment && (
                    <p className="text-[10px] text-muted-foreground mt-0.5">
                      {exercise.equipment}
                    </p>
                  )}
                </div>

                <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}

      {/* Exercise Detail Sheet (quick preview) */}
      <Sheet
        open={!!previewExercise}
        onOpenChange={() => setPreviewExercise(null)}
      >
        <SheetContent side="bottom" className="max-h-[80vh] rounded-t-2xl">
          {previewExercise && (
            <div className="space-y-4 pb-6">
              <SheetHeader>
                <SheetTitle className="text-left text-lg">
                  {previewExercise.name}
                </SheetTitle>
              </SheetHeader>

              {/* Category badge */}
              <div className="flex flex-wrap gap-2">
                <Badge
                  variant="secondary"
                  className={getCategoryColor(previewExercise.category)}
                >
                  {previewExercise.category}
                </Badge>
                {previewExercise.equipment && (
                  <Badge variant="outline" className="text-xs">
                    {previewExercise.equipment}
                  </Badge>
                )}
              </div>

              {/* Muscle groups */}
              <div>
                <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em] mb-1.5">
                  Muscle Groups
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {previewExercise.muscleGroups.map((mg) => (
                    <span
                      key={mg}
                      className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-medium border ${getMuscleGroupColor(mg)}`}
                    >
                      {mg}
                    </span>
                  ))}
                </div>
              </div>

              {/* Instructions */}
              {previewExercise.instructions && (
                <div>
                  <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em] mb-1.5">
                    Instructions
                  </p>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {previewExercise.instructions}
                  </p>
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-2 pt-2">
                <Button
                  className="flex-1"
                  onClick={() => {
                    setPreviewExercise(null);
                    setLocation(`/exercises/${previewExercise.id}`);
                  }}
                >
                  View Progression
                </Button>
              </div>
            </div>
          )}
        </SheetContent>
      </Sheet>
    </div>
  );
}
