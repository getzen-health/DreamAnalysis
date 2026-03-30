/**
 * Emotional Fitness — Full page showing the EFS composite score,
 * 5 vital cards (resilience, regulation, awareness, range, stability),
 * daily insight banner, history chart, and PNG share button.
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { getEmotionalFitness, type EFSData } from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";
import { EFSHeroScore } from "@/components/efs-hero-score";
import { EFSVitalCard } from "@/components/efs-vital-card";
import { EFSInsightBanner } from "@/components/efs-insight-banner";
import { EFSHistoryChart } from "@/components/efs-history-chart";
import { exportEFSCard } from "@/components/efs-share-card";
import { Shield, Gauge, Eye, Palette, Anchor, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConfidenceMeter } from "@/components/confidence-meter";

// Map EFS confidence tier to a numeric value for ConfidenceMeter
function efsConfidenceToNumber(tier: EFSData["confidence"]): number {
  switch (tier) {
    case "full": return 0.9;
    case "early_estimate": return 0.55;
    case "building": return 0.25;
    default: return 0.5;
  }
}

// ── Animation variants ──────────────────────────────────────────────────────

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

const staggerContainer = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.08 },
  },
};

const vitalCardVariant = {
  hidden: { opacity: 0, y: 16, scale: 0.97 },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { type: "spring", stiffness: 300, damping: 24 },
  },
};

// ── Vital icon mapping ──────────────────────────────────────────────────────

const VITAL_ICONS: Record<string, React.ElementType> = {
  resilience: Shield,
  regulation: Gauge,
  awareness: Eye,
  range: Palette,
  stability: Anchor,
};

const VITAL_ORDER = ["resilience", "regulation", "awareness", "range", "stability"];

// ── Loading skeleton ────────────────────────────────────────────────────────

function EFSSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      {/* Hero skeleton */}
      <div className="flex justify-center">
        <div className="w-[180px] h-[180px] rounded-full bg-muted/30" />
      </div>
      {/* Vital cards skeleton */}
      <div className="grid grid-cols-2 gap-3">
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="h-24 rounded-xl bg-muted/20" />
        ))}
      </div>
      {/* Insight skeleton */}
      <div className="h-20 rounded-xl bg-muted/20" />
      {/* Chart skeleton */}
      <div className="h-[280px] rounded-xl bg-muted/20" />
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────────

export default function EmotionalFitness() {
  const userId = useMemo(() => getParticipantId(), []);

  const { data, isLoading } = useQuery<EFSData>({
    queryKey: ["emotional-fitness", userId],
    queryFn: () => getEmotionalFitness(userId),
    staleTime: 5 * 60_000,
  });

  return (
    <main className="min-h-screen bg-background pb-24 max-w-2xl mx-auto">
      {/* Header */}
      <motion.div
        className="px-4 pt-6 pb-4"
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
      >
        <h1 className="text-xl font-bold text-foreground tracking-tight">
          Emotional Fitness
        </h1>
        <p className="text-xs text-muted-foreground mt-1">Your emotional health, measured daily</p>
      </motion.div>
      <div className="px-4">

      {isLoading && <EFSSkeleton />}

      {!isLoading && data && (
        <div className="space-y-6">
          {/* Hero score arc — fade in from above */}
          <motion.div
            className="flex justify-center"
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5, ease: "easeOut" }}
          >
            <EFSHeroScore
              score={data.score}
              color={data.color}
              label={data.label}
              confidence={data.confidence}
              trend={data.trend}
              progress={data.progress}
            />
          </motion.div>

          {/* Confidence meter for the overall emotional fitness reading */}
          <div className="px-2">
            <ConfidenceMeter
              confidence={efsConfidenceToNumber(data.confidence)}
              size="sm"
              showLabel
            />
          </div>

          {/* 5 vital cards — staggered entrance, 5th card spans full width */}
          <motion.div
            className="grid grid-cols-2 gap-3"
            variants={staggerContainer}
            initial="hidden"
            animate="visible"
          >
            {VITAL_ORDER.map((key, index) => {
              const vital = data.vitals[key];
              if (!vital) return null;
              const Icon = VITAL_ICONS[key] ?? Shield;
              const isLast = index === VITAL_ORDER.length - 1 && VITAL_ORDER.length % 2 !== 0;
              return (
                <motion.div
                  key={key}
                  variants={vitalCardVariant}
                  className={isLast ? "col-span-2" : undefined}
                >
                  <EFSVitalCard name={key} icon={Icon} vital={vital} />
                </motion.div>
              );
            })}
          </motion.div>

          {/* Daily insight banner — fade up */}
          <motion.div
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.4, delay: 0.5, ease: "easeOut" }}
          >
            <EFSInsightBanner insight={data.dailyInsight} />
          </motion.div>

          {/* History chart — fade up */}
          <motion.div
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.4, delay: 0.65, ease: "easeOut" }}
          >
            <EFSHistoryChart userId={userId} />
          </motion.div>

          {/* Share button — fade up */}
          <motion.div
            className="flex justify-center pt-2"
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.3, delay: 0.8, ease: "easeOut" }}
          >
            <Button
              variant="outline"
              size="sm"
              className="gap-2 text-muted-foreground hover:text-foreground"
              onClick={() => exportEFSCard(data)}
            >
              <Download className="h-4 w-4" />
              Share as image
            </Button>
          </motion.div>
        </div>
      )}

      {!isLoading && !data && (
        <div className="flex flex-col items-center justify-center min-h-[40vh] gap-3">
          <p className="text-sm text-muted-foreground">
            No emotional fitness data available yet.
          </p>
          <p className="text-xs text-muted-foreground/60">
            Record voice check-ins to build your score.
          </p>
        </div>
      )}
      </div>
    </main>
  );
}
