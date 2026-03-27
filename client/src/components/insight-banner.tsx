/**
 * InsightBanner — bottom-slide real-time deviation alert.
 *
 * Shows when the InsightEngine detects a sustained deviation from baseline.
 * 5-minute cooldown between banners. Dismiss or tap CTA to navigate.
 */

import { motion, AnimatePresence } from "framer-motion";
import { X, Zap } from "lucide-react";
import type { DeviationEvent } from "@/lib/insight-engine";

interface Props {
  events: DeviationEvent[];
  onDismiss: () => void;
  onCTA: (href: string) => void;
  suggestedLabel?: string;
}

const METRIC_LABELS: Record<string, string> = {
  stress: "stress", focus: "focus", valence: "mood", arousal: "arousal",
  hrv: "HRV", sleep: "sleep", steps: "activity", energy: "energy",
};

const CTA_MAP: Record<string, { label: string; href: string }> = {
  stress:  { label: "Box breathing", href: "/biofeedback" },
  focus:   { label: "Neurofeedback", href: "/neurofeedback" },
  valence: { label: "AI Companion", href: "/ai-companion" },
  arousal: { label: "Breathing", href: "/biofeedback" },
};

export function InsightBanner({ events, onDismiss, onCTA, suggestedLabel }: Props) {
  const event = events[0];
  if (!event) return null;

  const metricLabel = suggestedLabel || METRIC_LABELS[event.metric] || event.metric;
  const dir = event.direction === "high" ? "elevated" : "low";
  const durationText = event.durationMinutes >= 1 ? `${Math.round(event.durationMinutes)} min` : "just now";
  const cta = CTA_MAP[event.metric] || { label: "See insights", href: "/insights" };

  return (
    <AnimatePresence>
      <motion.div
        key="insight-banner"
        data-testid="insight-banner"
        initial={{ y: 80, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 80, opacity: 0 }}
        transition={{ type: "spring", damping: 20 }}
        role="status"
        aria-label={`${metricLabel} deviation detected`}
        className="fixed bottom-20 left-4 right-4 z-50 rounded-xl bg-card border border-border/30 shadow-xl p-4 flex items-center gap-3"
      >
        <Zap className="h-4 w-4 text-primary shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium leading-tight">
            {metricLabel.charAt(0).toUpperCase() + metricLabel.slice(1)} is {dir} — {durationText}
          </p>
          <p className="text-xs text-muted-foreground mt-0.5">
            {(event.currentValue * 100).toFixed(0)}% vs your usual {(event.baselineMean * 100).toFixed(0)}%
          </p>
        </div>
        <button
          onClick={() => onCTA(cta.href)}
          aria-label={`${cta.label} for ${metricLabel}`}
          className="text-xs font-medium text-primary whitespace-nowrap hover:underline"
        >
          {cta.label}
        </button>
        <button
          onClick={onDismiss}
          aria-label="dismiss"
          className="p-1 rounded hover:bg-muted/50 text-muted-foreground"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </motion.div>
    </AnimatePresence>
  );
}
