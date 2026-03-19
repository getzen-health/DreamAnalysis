/**
 * Health -- Consolidated health page with tabs.
 *
 * Tabs:
 * - Body (body metrics from body-metrics.tsx)
 * - Workouts (workout from workout.tsx)
 * - Scores (health scores from scores-dashboard.tsx)
 */

import { lazy, Suspense } from "react";
import { motion } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Scale, Dumbbell, Activity } from "lucide-react";

// Lazy-load tab contents from existing pages
const BodyMetrics = lazy(() => import("@/pages/body-metrics"));
const WorkoutPage = lazy(() => import("@/pages/workout"));
const ScoresDashboard = lazy(() => import("@/pages/scores-dashboard"));

/* ---------- Loading fallback ---------- */

function TabLoader() {
  return (
    <div className="flex items-center justify-center py-12">
      <div className="h-5 w-5 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
    </div>
  );
}

/* ---------- Main component ---------- */

export default function Health() {
  return (
    <div className="max-w-2xl mx-auto px-4 py-6 pb-24">
      {/* Header */}
      <motion.div
        initial={pageTransition.initial}
        animate={pageTransition.animate}
        transition={pageTransition.transition}
        className="mb-5"
      >
        <h1 className="text-xl font-bold tracking-tight text-foreground">
          Health
        </h1>
        <p className="text-sm mt-1 text-muted-foreground">
          Body metrics, workouts, and health scores
        </p>
      </motion.div>

      <Tabs defaultValue="body" className="w-full">
        <TabsList className="w-full">
          <TabsTrigger value="body" className="flex-1 gap-1.5">
            <Scale className="h-3.5 w-3.5" />
            Body
          </TabsTrigger>
          <TabsTrigger value="workouts" className="flex-1 gap-1.5">
            <Dumbbell className="h-3.5 w-3.5" />
            Workouts
          </TabsTrigger>
          <TabsTrigger value="scores" className="flex-1 gap-1.5">
            <Activity className="h-3.5 w-3.5" />
            Scores
          </TabsTrigger>
        </TabsList>

        <TabsContent value="body">
          <Suspense fallback={<TabLoader />}>
            <BodyMetrics />
          </Suspense>
        </TabsContent>

        <TabsContent value="workouts">
          <Suspense fallback={<TabLoader />}>
            <WorkoutPage />
          </Suspense>
        </TabsContent>

        <TabsContent value="scores">
          <Suspense fallback={<TabLoader />}>
            <ScoresDashboard />
          </Suspense>
        </TabsContent>
      </Tabs>
    </div>
  );
}
