/**
 * BrainTabs -- Consolidated Brain Monitor page with tabs.
 *
 * Tabs:
 * - Live EEG (existing brain monitor content)
 * - Neurofeedback (existing neurofeedback.tsx)
 * - Connectivity (existing brain-connectivity.tsx)
 */

import { lazy, Suspense } from "react";
import { motion } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Activity, Radio, Network } from "lucide-react";

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
      </Tabs>
    </div>
  );
}
