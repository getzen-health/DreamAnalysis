/**
 * Sleep -- Consolidated sleep page with tabs.
 *
 * Tabs:
 * - Sleep Data (sleep metrics from health sync)
 * - Dreams (dream journal from dream-journal.tsx)
 * - Music (sleep music from sleep-stories.tsx)
 */

import { lazy, Suspense, useState } from "react";
import { motion } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useHealthSync } from "@/hooks/use-health-sync";
import { Moon, PenLine, Music } from "lucide-react";

// Lazy-load tab contents from existing pages
const DreamJournal = lazy(() => import("@/pages/dream-journal"));
const SleepMusic = lazy(() => import("@/components/sleep-stories"));

/* ---------- Sleep Data tab content ---------- */

function SleepDataTab() {
  const { latestPayload } = useHealthSync();

  const sleepHours = latestPayload?.sleep_total_hours ?? null;
  const sleepEfficiency = latestPayload?.sleep_efficiency ?? null;
  const deepSleepMins = latestPayload?.sleep_deep_hours != null
    ? Math.round(latestPayload.sleep_deep_hours * 60)
    : null;
  const remSleepMins = latestPayload?.sleep_rem_hours != null
    ? Math.round(latestPayload.sleep_rem_hours * 60)
    : null;
  // Light sleep = total - deep - REM (approximate)
  const lightSleepMins =
    sleepHours != null && deepSleepMins != null && remSleepMins != null
      ? Math.max(0, Math.round(sleepHours * 60) - deepSleepMins - remSleepMins)
      : null;

  const sleepLabel =
    sleepHours != null
      ? `${Math.floor(sleepHours)}h ${Math.round((sleepHours % 1) * 60)}m`
      : null;

  const stages = [
    { label: "Deep", mins: deepSleepMins, color: "#7c3aed" },
    { label: "REM", mins: remSleepMins, color: "#2dd4bf" },
    { label: "Light", mins: lightSleepMins, color: "#3b82f6" },
  ];

  const totalMins = stages.reduce((sum, s) => sum + (s.mins ?? 0), 0);

  return (
    <div className="space-y-4">
      {/* Sleep Duration hero */}
      <div className="rounded-2xl p-6 border border-border bg-card text-center" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}>
        <Moon className="h-8 w-8 mx-auto text-indigo-400 mb-3" />
        <div className="text-3xl font-bold text-foreground">
          {sleepLabel ?? "\u2014"}
        </div>
        <div className="text-xs text-muted-foreground mt-1">
          Last night's sleep
        </div>
        {sleepEfficiency != null && (
          <div className="mt-3 inline-flex items-center gap-1.5 rounded-full px-3 py-1 bg-indigo-500/10 text-indigo-400 text-xs font-medium">
            {Math.round(sleepEfficiency)}% efficiency
          </div>
        )}
      </div>

      {/* Sleep Stages */}
      {totalMins > 0 && (
        <div className="rounded-2xl p-4 border border-border bg-card" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}>
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
            Sleep Stages
          </div>
          {/* Stacked bar */}
          <div className="flex rounded-full overflow-hidden h-3 mb-3">
            {stages.map(
              (s) =>
                s.mins != null &&
                s.mins > 0 && (
                  <div
                    key={s.label}
                    style={{
                      width: `${(s.mins / totalMins) * 100}%`,
                      background: s.color,
                    }}
                  />
                )
            )}
          </div>
          <div className="flex justify-between">
            {stages.map((s) => (
              <div key={s.label} className="flex items-center gap-1.5">
                <div
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ background: s.color }}
                />
                <span className="text-[10px] text-muted-foreground">
                  {s.label}
                </span>
                <span className="text-[10px] font-mono text-foreground">
                  {s.mins != null ? `${s.mins}m` : "\u2014"}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {sleepHours == null && (
        <div className="rounded-2xl p-8 border border-border bg-card text-center" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}>
          <Moon className="h-10 w-10 mx-auto text-muted-foreground/30 mb-3" />
          <p className="text-sm text-muted-foreground">
            Connect Apple Health or Google Fit to see your sleep data
          </p>
        </div>
      )}
    </div>
  );
}

/* ---------- Loading fallback ---------- */

function TabLoader() {
  return (
    <div className="flex items-center justify-center py-12">
      <div className="h-5 w-5 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
    </div>
  );
}

/* ---------- Main component ---------- */

export default function Sleep() {
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
          Sleep
        </h1>
        <p className="text-sm mt-1 text-muted-foreground">
          Sleep data, dreams, and relaxation music
        </p>
      </motion.div>

      <Tabs defaultValue="data" className="w-full">
        <TabsList className="w-full">
          <TabsTrigger value="data" className="flex-1 gap-1.5">
            <Moon className="h-3.5 w-3.5" />
            Sleep Data
          </TabsTrigger>
          <TabsTrigger value="dreams" className="flex-1 gap-1.5">
            <PenLine className="h-3.5 w-3.5" />
            Dreams
          </TabsTrigger>
          <TabsTrigger value="music" className="flex-1 gap-1.5">
            <Music className="h-3.5 w-3.5" />
            Music
          </TabsTrigger>
        </TabsList>

        <TabsContent value="data">
          <SleepDataTab />
        </TabsContent>

        <TabsContent value="dreams">
          <Suspense fallback={<TabLoader />}>
            <DreamJournal />
          </Suspense>
        </TabsContent>

        <TabsContent value="music">
          <Suspense fallback={<TabLoader />}>
            <SleepMusic />
          </Suspense>
        </TabsContent>
      </Tabs>
    </div>
  );
}
