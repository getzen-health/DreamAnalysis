/**
 * DreamFusionCard — Shows dream content fused with overnight biometric data.
 *
 * Displays a headline, narrative body, biometric highlight pills, and sleep context.
 * Empty state guides the user to log a dream.
 */

import { Moon } from "lucide-react";
import type { DreamFusionInsight, BiometricHighlight } from "@/lib/dream-biometric-fusion";

export interface DreamFusionCardProps {
  insight: DreamFusionInsight | null;
}

const STATUS_DOT: Record<BiometricHighlight["status"], string> = {
  normal: "bg-emerald-400",
  elevated: "bg-amber-400",
  low: "bg-red-400",
};

export function DreamFusionCard({ insight }: DreamFusionCardProps) {
  return (
    <div
      data-testid="dream-fusion-card"
      className="rounded-[14px] bg-card border border-border p-4"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <Moon className="h-4 w-4 text-indigo-400" />
        <span className="text-sm font-medium text-muted-foreground">
          Last Night&apos;s Dream
        </span>
      </div>

      {!insight ? (
        /* Empty state */
        <p
          data-testid="dream-fusion-empty"
          className="text-sm text-muted-foreground/60"
        >
          Connect your BCI device and sleep — dream analysis generates automatically
        </p>
      ) : (
        <>
          {/* Headline */}
          <h3
            data-testid="dream-fusion-headline"
            className="text-base font-semibold mb-2"
          >
            {insight.headline}
          </h3>

          {/* Body narrative */}
          <p
            data-testid="dream-fusion-body"
            className="text-sm text-muted-foreground leading-relaxed mb-3"
          >
            {insight.body}
          </p>

          {/* Biometric highlights */}
          {insight.biometricHighlights.length > 0 && (
            <div
              data-testid="dream-fusion-highlights"
              className="flex flex-wrap gap-2 mb-3"
            >
              {insight.biometricHighlights.map((h) => (
                <span
                  key={h.label}
                  className="inline-flex items-center gap-1.5 rounded-full bg-muted/50 px-2.5 py-1 text-xs font-medium"
                >
                  <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[h.status]}`} />
                  {h.label}: {h.value}
                </span>
              ))}
            </div>
          )}

          {/* Sleep context */}
          <p
            data-testid="dream-fusion-sleep-context"
            className="text-xs text-muted-foreground/60"
          >
            {insight.sleepContext}
          </p>
        </>
      )}
    </div>
  );
}
