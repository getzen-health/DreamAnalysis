/**
 * InsightBanner — bottom-slide real-time deviation alert.
 *
 * Shows when the InsightEngine detects a sustained deviation from baseline.
 * 5-minute cooldown between banners. Dismiss or tap CTA to navigate.
 */

import { motion } from "framer-motion";
import { AlertTriangle, X, ChevronRight } from "lucide-react";
import type { DeviationEvent } from "@/lib/insight-engine";

export interface InsightBannerProps {
  events: DeviationEvent[];
  onDismiss: () => void;
  onCTA?: (href: string) => void;
}

export function InsightBanner({ events, onDismiss, onCTA }: InsightBannerProps) {
  if (events.length === 0) return null;

  // Show the most significant event (highest absolute z-score)
  const primary = events.reduce((a, b) =>
    Math.abs(b.zScore) > Math.abs(a.zScore) ? b : a,
  );

  return (
    <motion.div
      data-testid="insight-banner"
      initial={{ y: 100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: 100, opacity: 0 }}
      transition={{ type: "spring", stiffness: 300, damping: 25 }}
      className="fixed bottom-20 left-4 right-4 z-50 rounded-[14px] bg-card border border-warning/30 p-4 shadow-lg"
    >
      <div className="flex items-start gap-3">
        <AlertTriangle className="h-5 w-5 text-warning shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-foreground">
            {primary.message}
          </p>
          <p className="text-xs text-muted-foreground mt-0.5">
            {primary.durationMinutes > 0
              ? `${primary.durationMinutes} min · z=${primary.zScore}`
              : `z=${primary.zScore}`}
          </p>
          {primary.cta && primary.ctaHref && onCTA && (
            <button
              onClick={() => onCTA(primary.ctaHref!)}
              className="inline-flex items-center gap-1 mt-2 text-xs font-medium text-emerald-500 hover:text-emerald-400"
            >
              {primary.cta}
              <ChevronRight className="h-3 w-3" />
            </button>
          )}
        </div>
        <button
          onClick={onDismiss}
          className="text-muted-foreground hover:text-foreground p-1"
          aria-label="Dismiss"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </motion.div>
  );
}
