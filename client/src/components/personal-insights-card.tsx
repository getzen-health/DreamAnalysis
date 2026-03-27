/**
 * PersonalInsightsCard — Displays cross-modal correlation insights
 * like "You sleep 23% better on nights you don't eat after 9pm".
 *
 * Shows up to 3 insights with category icons and confidence badges.
 * Empty state guides the user to keep logging data.
 */

import { Moon, Utensils, Footprints, Clock, Brain, Sparkles } from "lucide-react";
import type { PersonalInsight, InsightCategory } from "@/lib/cross-modal-insights";

export interface PersonalInsightsCardProps {
  insights: PersonalInsight[];
}

const CATEGORY_ICON: Record<InsightCategory, React.ElementType> = {
  food_sleep: Moon,
  exercise_mood: Footprints,
  sleep_focus: Brain,
  time_pattern: Clock,
  streak_mood: Brain,
  food_stress: Utensils,
};

const CATEGORY_LABEL: Record<InsightCategory, string> = {
  food_sleep: "Sleep",
  exercise_mood: "Activity",
  sleep_focus: "Focus",
  time_pattern: "Routine",
  streak_mood: "Focus",
  food_stress: "Food",
};

const CONFIDENCE_STYLE: Record<PersonalInsight["confidence"], { bg: string; text: string; label: string }> = {
  strong: {
    bg: "bg-emerald-500/15",
    text: "text-emerald-400",
    label: "Strong",
  },
  moderate: {
    bg: "bg-amber-500/15",
    text: "text-amber-400",
    label: "Moderate",
  },
  weak: {
    bg: "bg-zinc-500/15",
    text: "text-muted-foreground",
    label: "Weak",
  },
};

export function PersonalInsightsCard({ insights }: PersonalInsightsCardProps) {
  const displayInsights = insights.slice(0, 3);

  return (
    <div
      data-testid="personal-insights-card"
      className="rounded-[14px] bg-card border border-border p-5"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-4 h-4 text-foreground" />
        <h3 className="text-sm font-semibold text-foreground tracking-wide">
          Your Patterns
        </h3>
      </div>

      {displayInsights.length === 0 ? (
        /* Empty state */
        <div className="flex flex-col gap-3">
          <p className="text-sm text-muted-foreground leading-relaxed">
            Keep logging to discover your patterns
          </p>
          <div className="flex flex-col gap-2">
            {[
              { icon: Moon, label: "Sleep data", needed: true },
              { icon: Utensils, label: "Food logs", needed: true },
              { icon: Footprints, label: "Step counts", needed: true },
              { icon: Brain, label: "Mood check-ins", needed: true },
            ].map(({ icon: Icon, label }) => (
              <div key={label} className="flex items-center gap-2">
                <Icon className="w-3.5 h-3.5 text-muted-foreground" />
                <span className="text-xs text-muted-foreground">{label}</span>
                <div className="flex-1 h-1.5 rounded-full bg-muted/50 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-muted-foreground/20"
                    style={{ width: "0%" }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        /* Insight rows */
        <div className="flex flex-col gap-3">
          {displayInsights.map((insight) => {
            const Icon = CATEGORY_ICON[insight.category];
            const conf = CONFIDENCE_STYLE[insight.confidence];

            return (
              <div
                key={insight.id}
                className="flex items-start gap-3"
              >
                <div className="mt-0.5 shrink-0">
                  <Icon className="w-4 h-4 text-muted-foreground" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-foreground leading-snug">
                    {insight.text}
                  </p>
                  <div className="flex items-center gap-2 mt-1">
                    <span
                      className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full ${conf.bg} ${conf.text}`}
                    >
                      {conf.label}
                    </span>
                    <span className="text-[10px] text-muted-foreground">
                      {insight.dataPoints} data points
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
