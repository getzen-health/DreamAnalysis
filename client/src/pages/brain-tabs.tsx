/**
 * BrainTabs -- Consolidated Brain Monitor page with tabs.
 *
 * Tabs:
 * - Live EEG (existing brain monitor content)
 * - Neurofeedback (existing neurofeedback.tsx)
 * - Connectivity (existing brain-connectivity.tsx)
 * - Study (link to research study)
 */

import { lazy, Suspense } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Activity, Radio, Network, GraduationCap } from "lucide-react";
import { Button } from "@/components/ui/button";

// Lazy-load tab contents from existing pages
const BrainMonitor = lazy(() => import("@/pages/brain-monitor"));
const Neurofeedback = lazy(() => import("@/pages/neurofeedback"));
const BrainConnectivity = lazy(() => import("@/pages/brain-connectivity"));

/* ---------- Loading fallback ---------- */

function TabLoader() {
  return (
    <div className="flex items-center justify-center py-12">
      <div className="h-5 w-5 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
    </div>
  );
}

/* ---------- Study tab content ---------- */

function StudyTab() {
  const [, navigate] = useLocation();

  return (
    <div className="rounded-2xl p-8 border border-border bg-card text-center">
      <GraduationCap className="h-10 w-10 mx-auto text-primary mb-3" />
      <h3 className="text-lg font-semibold text-foreground mb-2">
        Research Study
      </h3>
      <p className="text-sm text-muted-foreground mb-4 max-w-sm mx-auto">
        Participate in our neuroscience research study. Your data helps advance
        brain-computer interface technology.
      </p>
      <Button
        onClick={() => navigate("/study")}
        className="bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30"
      >
        View Study Details
      </Button>
    </div>
  );
}

/* ---------- Main component ---------- */

export default function BrainTabs() {
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
          Brain
        </h1>
        <p className="text-sm mt-1 text-muted-foreground">
          EEG monitor, neurofeedback, and connectivity
        </p>
      </motion.div>

      <Tabs defaultValue="eeg" className="w-full">
        <TabsList className="w-full">
          <TabsTrigger value="eeg" className="flex-1 gap-1.5">
            <Activity className="h-3.5 w-3.5" />
            Live EEG
          </TabsTrigger>
          <TabsTrigger value="neurofeedback" className="flex-1 gap-1.5">
            <Radio className="h-3.5 w-3.5" />
            Train
          </TabsTrigger>
          <TabsTrigger value="connectivity" className="flex-1 gap-1.5">
            <Network className="h-3.5 w-3.5" />
            Connect
          </TabsTrigger>
          <TabsTrigger value="study" className="flex-1 gap-1.5">
            <GraduationCap className="h-3.5 w-3.5" />
            Study
          </TabsTrigger>
        </TabsList>

        <TabsContent value="eeg">
          <Suspense fallback={<TabLoader />}>
            <BrainMonitor />
          </Suspense>
        </TabsContent>

        <TabsContent value="neurofeedback">
          <Suspense fallback={<TabLoader />}>
            <Neurofeedback />
          </Suspense>
        </TabsContent>

        <TabsContent value="connectivity">
          <Suspense fallback={<TabLoader />}>
            <BrainConnectivity />
          </Suspense>
        </TabsContent>

        <TabsContent value="study">
          <StudyTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
