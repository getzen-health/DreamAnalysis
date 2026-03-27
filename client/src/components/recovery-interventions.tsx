/**
 * RecoveryInterventions — Bevel-style actionable suggestions when recovery is low.
 * Shows horizontally scrollable cards with contextual health advice.
 * Only renders when there are interventions to show.
 */

import { useLocation } from "wouter";
import { motion } from "framer-motion";
import {
  HeartPulse, BatteryLow, Moon, TrendingDown, Wind, TreePine, Zap,
  type LucideIcon,
} from "lucide-react";
import {
  suggestRecoveryInterventions,
  type RecoveryContext,
  type RecoveryIntervention,
} from "@/lib/intervention-engine";

const ICON_MAP: Record<string, LucideIcon> = {
  "heart-pulse": HeartPulse,
  "battery-low": BatteryLow,
  moon: Moon,
  "trending-down": TrendingDown,
  wind: Wind,
  "tree-pine": TreePine,
  zap: Zap,
};

const PRIORITY_COLORS: Record<string, string> = {
  high: "border-l-rose-500 bg-rose-500/5",
  medium: "border-l-amber-500 bg-amber-500/5",
  low: "border-l-emerald-500 bg-emerald-500/5",
};

const PRIORITY_ICON_COLORS: Record<string, string> = {
  high: "text-rose-400",
  medium: "text-amber-400",
  low: "text-emerald-400",
};

interface Props {
  recovery?: number;
  sleepHours?: number;
  strain?: number;
  stress?: number;
  hrvTrend?: "up" | "down" | "stable";
}

export function RecoveryInterventions({ recovery, sleepHours, strain, stress, hrvTrend }: Props) {
  const [, navigate] = useLocation();

  const ctx: RecoveryContext = { recovery, sleepHours, strain, stress, hrvTrend };
  const interventions = suggestRecoveryInterventions(ctx);

  if (interventions.length === 0) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-muted-foreground px-1">
        Recovery Insights
      </h3>
      <div className="flex gap-3 overflow-x-auto pb-2 -mx-1 px-1 scrollbar-hide">
        {interventions.map((item, i) => (
          <InterventionCard
            key={i}
            item={item}
            index={i}
            onNavigate={navigate}
          />
        ))}
      </div>
    </div>
  );
}

function InterventionCard({
  item,
  index,
  onNavigate,
}: {
  item: RecoveryIntervention;
  index: number;
  onNavigate: (path: string) => void;
}) {
  const Icon = ICON_MAP[item.icon] ?? HeartPulse;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.08, duration: 0.3 }}
      className={`
        flex-shrink-0 w-[260px] rounded-xl border-l-4 p-3 space-y-2
        ${PRIORITY_COLORS[item.priority]}
      `}
    >
      <div className="flex items-center gap-2">
        <Icon className={`w-4 h-4 ${PRIORITY_ICON_COLORS[item.priority]}`} />
        <span className="text-sm font-semibold">{item.title}</span>
      </div>
      <p className="text-xs text-muted-foreground leading-relaxed">
        {item.description}
      </p>
      {item.action && (
        <button
          onClick={() => onNavigate(item.action!.route)}
          className="text-xs font-medium text-primary hover:underline"
        >
          {item.action.label} &rarr;
        </button>
      )}
    </motion.div>
  );
}
