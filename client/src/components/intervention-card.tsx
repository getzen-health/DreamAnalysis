/**
 * InterventionCard — compact "Things you can try" card showing 2-3
 * suggested interventions from the intervention engine.
 *
 * Every page showing mood/emotion data must include this component.
 * Supportive framing: "Things you can try" not "You should do this."
 *
 * @see Issue #524
 */

import { useLocation } from "wouter";
import {
  Wind,
  Lightbulb,
  Brain,
  Music,
  PenLine,
  Heart,
  Sparkles,
  Eye,
  BarChart,
  Hand,
  type LucideIcon,
} from "lucide-react";
import {
  suggestInterventions,
  type Intervention,
} from "@/lib/intervention-engine";

// ── Icon mapping ────────────────────────────────────────────────────────────

const ICON_MAP: Record<string, LucideIcon> = {
  wind: Wind,
  lightbulb: Lightbulb,
  brain: Brain,
  music: Music,
  "pen-line": PenLine,
  heart: Heart,
  sparkles: Sparkles,
  eye: Eye,
  "bar-chart": BarChart,
  hand: Hand,
};

// ── Tier badge colors ───────────────────────────────────────────────────────

const TIER_COLORS: Record<string, { bg: string; text: string }> = {
  breathing: { bg: "bg-cyan-500/10", text: "text-cyan-400" },
  reappraisal: { bg: "bg-amber-500/10", text: "text-amber-400" },
  neurofeedback: { bg: "bg-indigo-500/10", text: "text-indigo-400" },
};

// ── Props ───────────────────────────────────────────────────────────────────

export interface InterventionCardProps {
  /** Primary emotion label */
  emotion?: string;
  /** Stress index 0-1 */
  stress?: number;
  /** Whether a Muse headband is connected */
  hasHeadband?: boolean;
  /** Compact mode — smaller card */
  compact?: boolean;
}

// ── Component ───────────────────────────────────────────────────────────────

export function InterventionCard({
  emotion,
  stress,
  hasHeadband = false,
  compact = false,
}: InterventionCardProps) {
  const [, navigate] = useLocation();

  const interventions = suggestInterventions(
    emotion ?? "neutral",
    stress ?? 0,
    hasHeadband,
  );

  if (interventions.length === 0) return null;

  return (
    <div
      data-testid="intervention-card"
      className={`rounded-xl border border-border/20 bg-card/50 ${compact ? "p-3" : "p-4"} space-y-3`}
    >
      {/* Header */}
      <p className="text-xs font-medium text-muted-foreground">
        Things you can try
      </p>

      {/* Intervention rows */}
      <div className="space-y-2">
        {interventions.map((intervention, i) => (
          <InterventionRow
            key={`${intervention.tier}-${i}`}
            intervention={intervention}
            compact={compact}
            onNavigate={navigate}
          />
        ))}
      </div>
    </div>
  );
}

// ── Row sub-component ───────────────────────────────────────────────────────

function InterventionRow({
  intervention,
  compact,
  onNavigate,
}: {
  intervention: Intervention;
  compact: boolean;
  onNavigate: (route: string) => void;
}) {
  const Icon = ICON_MAP[intervention.icon] ?? Brain;
  const colors = TIER_COLORS[intervention.tier] ?? TIER_COLORS.breathing;

  return (
    <button
      onClick={() => onNavigate(intervention.route)}
      className={`flex items-center gap-3 w-full rounded-lg transition-colors hover:bg-muted/30 text-left ${compact ? "p-2" : "p-2.5"}`}
      data-testid="intervention-row"
    >
      {/* Icon */}
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${colors.bg}`}
      >
        <Icon className={`h-4 w-4 ${colors.text}`} />
      </div>

      {/* Title + description */}
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium text-foreground leading-tight truncate">
          {intervention.title}
        </p>
        {!compact && (
          <p className="text-[10px] text-muted-foreground leading-relaxed mt-0.5 line-clamp-2">
            {intervention.description}
          </p>
        )}
      </div>

      {/* Duration badge */}
      <span
        className={`shrink-0 rounded-md px-2 py-0.5 text-[10px] font-medium ${colors.bg} ${colors.text}`}
      >
        {intervention.duration}
      </span>
    </button>
  );
}
