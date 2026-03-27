import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { apiRequest } from "@/lib/queryClient";
import { useAuth } from "@/hooks/use-auth";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  LayoutGrid,
  Play,
  Plus,
  ArrowLeft,
  Clock,
  Dumbbell,
  Trash2,
  Sparkles,
} from "lucide-react";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import type { WorkoutTemplate } from "@/lib/workout-types";
import { PREBUILT_TEMPLATES, getMuscleGroupColor } from "@/lib/workout-types";

/* ========== Component ========== */

export default function WorkoutTemplates() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [, setLocation] = useLocation();
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDescription, setNewDescription] = useState("");

  // Fetch user templates
  const { data: templates = [], isLoading } = useQuery<WorkoutTemplate[]>({
    queryKey: [`/api/workout-templates/${user?.id}`],
    enabled: !!user?.id,
    staleTime: 30_000,
  });

  // Delete template
  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      const res = await apiRequest("DELETE", `/api/workout-templates/${id}`);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: [`/api/workout-templates/${user?.id}`],
      });
      toast({ title: "Template deleted" });
    },
    onError: (err: Error) => {
      toast({
        title: "Error",
        description: err.message,
        variant: "destructive",
      });
    },
  });

  // Start workout from template
  function startFromTemplate(exerciseList: Array<{ exerciseId?: string; name: string; sets: number; reps: number }>) {
    // Store template exercises in sessionStorage for active-workout to pick up
    sessionStorage.setItem(
      "ndw_workout_template",
      JSON.stringify(
        exerciseList.map((e) => ({
          exerciseId: e.exerciseId ?? "",
          name: e.name,
          sets: e.sets,
          reps: e.reps,
        }))
      )
    );
    setLocation("/active-workout");
  }

  // Create empty template
  const createMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/workout-templates", {
        name: newName.trim(),
        description: newDescription.trim() || null,
        exercises: [],
      });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: [`/api/workout-templates/${user?.id}`],
      });
      setShowCreate(false);
      setNewName("");
      setNewDescription("");
      toast({ title: "Template created" });
    },
    onError: (err: Error) => {
      toast({
        title: "Error",
        description: err.message,
        variant: "destructive",
      });
    },
  });

  return (
    <div className="max-w-lg mx-auto px-4 py-6 space-y-6 pb-24">
      {/* Header */}
      <motion.div
        className="space-y-2"
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
          <div className="flex-1">
            <h1 className="text-xl font-bold tracking-tight text-foreground flex items-center gap-2">
              <LayoutGrid className="h-5 w-5 text-primary" />
              Workout Templates
            </h1>
            <p className="text-xs text-muted-foreground mt-0.5">
              Quick-start a workout from a template
            </p>
          </div>
          <Button
            size="sm"
            variant="outline"
            className="gap-1"
            onClick={() => setShowCreate(true)}
          >
            <Plus className="h-4 w-4" /> New
          </Button>
        </div>
      </motion.div>

      {/* User Templates */}
      {templates.length > 0 && (
        <div className="space-y-2">
          <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em]">
            Your Templates
          </p>
          {templates.map((template, idx) => {
            const exerciseList = Array.isArray(template.exercises) ? template.exercises : [];
            return (
              <motion.div
                key={template.id}
                custom={idx}
                initial="hidden"
                animate="visible"
                variants={cardVariants}
                className="rounded-xl border border-border bg-card p-4 space-y-3"
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-foreground truncate">
                      {template.name}
                    </p>
                    {template.description && (
                      <p className="text-[11px] text-muted-foreground mt-0.5">
                        {template.description}
                      </p>
                    )}
                    <div className="flex items-center gap-3 mt-1.5">
                      <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                        <Dumbbell className="h-3 w-3" />
                        {exerciseList.length} exercises
                      </span>
                      {template.timesUsed != null && template.timesUsed > 0 && (
                        <span className="text-[10px] text-muted-foreground">
                          Used {template.timesUsed}x
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Button
                      size="sm"
                      className="h-8 gap-1 text-xs"
                      onClick={() => startFromTemplate(exerciseList as Array<{ exerciseId: string; name: string; sets: number; reps: number }>)}
                    >
                      <Play className="h-3 w-3" /> Start
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                      onClick={() => deleteMutation.mutate(template.id)}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      )}

      {/* Prebuilt Templates */}
      <div className="space-y-2">
        <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em] flex items-center gap-1">
          <Sparkles className="h-3 w-3" />
          Prebuilt Programs
        </p>
        {PREBUILT_TEMPLATES.map((template, idx) => (
          <motion.div
            key={template.name}
            custom={idx + templates.length}
            initial="hidden"
            animate="visible"
            variants={cardVariants}
            className="rounded-xl border border-border bg-card p-4 space-y-3"
          >
            <div className="flex items-center justify-between">
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-foreground truncate">
                  {template.name}
                </p>
                <p className="text-[11px] text-muted-foreground mt-0.5">
                  {template.description}
                </p>
                <div className="flex items-center gap-3 mt-1.5">
                  <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                    <Dumbbell className="h-3 w-3" />
                    {template.exercises.length} exercises
                  </span>
                  <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                    <Clock className="h-3 w-3" />~{template.estimatedMin}min
                  </span>
                </div>
                <div className="flex flex-wrap gap-1 mt-2">
                  {template.muscleGroups.map((mg) => (
                    <span
                      key={mg}
                      className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium border ${getMuscleGroupColor(mg)}`}
                    >
                      {mg}
                    </span>
                  ))}
                </div>
              </div>
              <Button
                size="sm"
                variant="outline"
                className="h-8 gap-1 text-xs shrink-0 ml-3"
                onClick={() => startFromTemplate(template.exercises.map(e => ({ ...e, exerciseId: "" })))}
              >
                <Play className="h-3 w-3" /> Start
              </Button>
            </div>

            {/* Exercise list preview */}
            <div className="border-t border-border pt-2 space-y-1">
              {template.exercises.map((ex, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between text-xs text-muted-foreground"
                >
                  <span className="truncate">{ex.name}</span>
                  <span className="text-[10px] tabular-nums shrink-0 ml-2">
                    {ex.sets}x{ex.reps}
                  </span>
                </div>
              ))}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Create Template Dialog */}
      <Dialog open={showCreate} onOpenChange={setShowCreate}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>Create Template</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div>
              <label className="text-xs font-medium text-muted-foreground">
                Name
              </label>
              <Input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="e.g., My Push Day"
                className="mt-1"
                autoFocus
              />
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground">
                Description (optional)
              </label>
              <Input
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                placeholder="e.g., Chest, shoulders, triceps"
                className="mt-1"
              />
            </div>
            <Button
              onClick={() => createMutation.mutate()}
              disabled={!newName.trim() || createMutation.isPending}
              className="w-full"
              size="sm"
            >
              Create Template
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
