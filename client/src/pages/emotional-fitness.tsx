/**
 * Emotional Fitness — Full page showing the EFS composite score,
 * 5 vital cards (resilience, regulation, awareness, range, stability),
 * daily insight banner, history chart, and PNG share button.
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { getEmotionalFitness, type EFSData } from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";
import { EFSHeroScore } from "@/components/efs-hero-score";
import { EFSVitalCard } from "@/components/efs-vital-card";
import { EFSInsightBanner } from "@/components/efs-insight-banner";
import { EFSHistoryChart } from "@/components/efs-history-chart";
import { exportEFSCard } from "@/components/efs-share-card";
import { Shield, Gauge, Eye, Palette, Anchor, Download } from "lucide-react";
import { Button } from "@/components/ui/button";

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
      <div className="px-4 pt-6 pb-4">
        <h1 className="text-xl font-bold text-foreground tracking-tight">
          Emotional Fitness
        </h1>
        <p className="text-xs text-muted-foreground mt-1">Your emotional health, measured daily</p>
      </div>
      <div className="px-4">

      {isLoading && <EFSSkeleton />}

      {!isLoading && data && (
        <div className="space-y-6">
          {/* Hero score arc */}
          <div className="flex justify-center">
            <EFSHeroScore
              score={data.score}
              color={data.color}
              label={data.label}
              confidence={data.confidence}
              trend={data.trend}
              progress={data.progress}
            />
          </div>

          {/* 5 vital cards — 2-column grid */}
          <div className="grid grid-cols-2 gap-3">
            {VITAL_ORDER.map((key) => {
              const vital = data.vitals[key];
              if (!vital) return null;
              const Icon = VITAL_ICONS[key] ?? Shield;
              return (
                <EFSVitalCard key={key} name={key} icon={Icon} vital={vital} />
              );
            })}
          </div>

          {/* Daily insight banner */}
          <EFSInsightBanner insight={data.dailyInsight} />

          {/* History chart */}
          <EFSHistoryChart userId={userId} />

          {/* Share button — PNG export */}
          <div className="flex justify-center pt-2">
            <Button
              variant="outline"
              size="sm"
              className="gap-2 text-muted-foreground hover:text-foreground"
              onClick={() => exportEFSCard(data)}
            >
              <Download className="h-4 w-4" />
              Share as image
            </Button>
          </div>
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
