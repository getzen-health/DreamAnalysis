/**
 * InterventionSummary — before/after comparison card shown at end of a
 * breathing or biofeedback session.
 *
 * Displays stress, focus, and HRV metrics side-by-side with colour-coded
 * arrows and percentage-change labels.  A "Share" button copies a plain-text
 * summary to the clipboard.
 */

import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  TrendingDown,
  TrendingUp,
  Minus,
  Share2,
  CheckCircle2,
} from "lucide-react";
import { useState } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

export interface InterventionMetrics {
  /** 0–1 normalised stress index */
  stress: number;
  /** 0–1 normalised focus index */
  focus: number;
  /** Heart-rate variability in ms (or null when unavailable) */
  hrv: number | null;
}

export interface InterventionSummaryProps {
  beforeMetrics: InterventionMetrics;
  afterMetrics: InterventionMetrics;
  /** Session length in seconds */
  duration: number;
  /** Human-readable label, e.g. "Coherence breathing" */
  type: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Returns percentage change, capped at ±999 to avoid absurd display values. */
function pctChange(before: number, after: number): number {
  if (before === 0) return 0;
  return Math.max(-999, Math.min(999, Math.round(((after - before) / before) * 100)));
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m === 0) return `${s}s`;
  return s === 0 ? `${m} min` : `${m} min ${s}s`;
}

// ── Sub-component: a single metric row ────────────────────────────────────────

interface MetricRowProps {
  label: string;
  before: number | null;
  after: number | null;
  /** If true, a decrease is good (e.g. stress). If false, an increase is good (e.g. focus, HRV). */
  lowerIsBetter: boolean;
  unit?: string;
  /** How many decimal places to show for the raw values */
  decimals?: number;
}

function MetricRow({
  label,
  before,
  after,
  lowerIsBetter,
  unit = "",
  decimals = 0,
}: MetricRowProps) {
  if (before === null || after === null) {
    return (
      <div
        className="flex items-center justify-between py-2 border-b border-border/10 last:border-0"
        data-testid={`metric-row-${label.toLowerCase().replace(/\s+/g, "-")}`}
      >
        <span className="text-xs text-muted-foreground">{label}</span>
        <span className="text-xs text-muted-foreground">—</span>
      </div>
    );
  }

  const pct = pctChange(before, after);
  const improved = lowerIsBetter ? pct < 0 : pct > 0;
  const unchanged = pct === 0;

  const trendColor = unchanged
    ? "text-muted-foreground"
    : improved
      ? "text-cyan-400"
      : "text-rose-400";

  const TrendIcon = unchanged ? Minus : improved ? TrendingDown : TrendingUp;

  const beforeDisplay = before.toFixed(decimals);
  const afterDisplay = after.toFixed(decimals);
  const pctLabel = unchanged
    ? "no change"
    : `${pct > 0 ? "+" : ""}${pct}%`;

  return (
    <div
      className="flex items-center justify-between py-2.5 border-b border-border/10 last:border-0"
      data-testid={`metric-row-${label.toLowerCase().replace(/\s+/g, "-")}`}
    >
      {/* Label */}
      <span className="text-xs text-muted-foreground w-16 shrink-0">{label}</span>

      {/* Before */}
      <div className="text-center w-16">
        <span className="text-sm font-mono font-semibold text-foreground/70">
          {beforeDisplay}
          {unit}
        </span>
      </div>

      {/* Arrow + pct change */}
      <div className={`flex items-center gap-1 w-24 justify-center ${trendColor}`}>
        <TrendIcon
          className="h-3.5 w-3.5 shrink-0"
          data-testid={
            unchanged
              ? `icon-unchanged-${label.toLowerCase()}`
              : improved
                ? `icon-improved-${label.toLowerCase()}`
                : `icon-worsened-${label.toLowerCase()}`
          }
        />
        <span className="text-xs font-mono">{pctLabel}</span>
      </div>

      {/* After */}
      <div className="text-center w-16">
        <span className={`text-sm font-mono font-semibold ${trendColor}`}>
          {afterDisplay}
          {unit}
        </span>
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export function InterventionSummary({
  beforeMetrics,
  afterMetrics,
  duration,
  type,
}: InterventionSummaryProps) {
  const [copied, setCopied] = useState(false);

  const handleShare = async () => {
    const stressPct = pctChange(beforeMetrics.stress, afterMetrics.stress);
    const focusPct = pctChange(beforeMetrics.focus, afterMetrics.focus);
    const hrvPct =
      beforeMetrics.hrv !== null && afterMetrics.hrv !== null
        ? pctChange(beforeMetrics.hrv, afterMetrics.hrv)
        : null;

    const lines: string[] = [
      `AntarAI — ${type}`,
      `Duration: ${formatDuration(duration)}`,
      ``,
      `Stress:  ${Math.round(beforeMetrics.stress * 100)} → ${Math.round(afterMetrics.stress * 100)}  (${stressPct > 0 ? "+" : ""}${stressPct}%)`,
      `Focus:   ${Math.round(beforeMetrics.focus * 100)} → ${Math.round(afterMetrics.focus * 100)}  (${focusPct > 0 ? "+" : ""}${focusPct}%)`,
    ];
    if (hrvPct !== null && beforeMetrics.hrv !== null && afterMetrics.hrv !== null) {
      lines.push(
        `HRV:     ${beforeMetrics.hrv.toFixed(1)} ms → ${afterMetrics.hrv.toFixed(1)} ms  (${hrvPct > 0 ? "+" : ""}${hrvPct}%)`
      );
    }

    try {
      await navigator.clipboard.writeText(lines.join("\n"));
    } catch {
      // Clipboard API blocked (e.g. non-secure context in test env) — silently skip
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  return (
    <Card
      className="glass-card rounded-xl overflow-hidden"
      data-testid="intervention-summary"
    >
      {/* Header */}
      <CardHeader className="pb-2 pt-5 px-5">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-sm font-semibold">Session Summary</CardTitle>
            <p className="text-xs text-muted-foreground mt-0.5">
              {type} · {formatDuration(duration)}
            </p>
          </div>
          <Badge
            variant="outline"
            className="text-[10px] border-cyan-500/30 text-cyan-400 bg-cyan-500/5"
          >
            Complete
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="px-5 pb-5 space-y-4">
        {/* Column headers */}
        <div className="flex items-center justify-between text-[10px] font-medium uppercase tracking-wider text-muted-foreground/50">
          <span className="w-16">Metric</span>
          <span className="w-16 text-center">Before</span>
          <span className="w-24 text-center">Change</span>
          <span className="w-16 text-center">After</span>
        </div>

        {/* Metric rows */}
        <div>
          <MetricRow
            label="Stress"
            before={beforeMetrics.stress}
            after={afterMetrics.stress}
            lowerIsBetter
            decimals={2}
          />
          <MetricRow
            label="Focus"
            before={beforeMetrics.focus}
            after={afterMetrics.focus}
            lowerIsBetter={false}
            decimals={2}
          />
          <MetricRow
            label="HRV"
            before={beforeMetrics.hrv}
            after={afterMetrics.hrv}
            lowerIsBetter={false}
            unit=" ms"
            decimals={1}
          />
        </div>

        {/* Share button */}
        <div className="pt-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleShare}
            className="w-full border border-border/30 text-xs gap-2 hover:bg-muted/30"
            data-testid="share-button"
          >
            {copied ? (
              <>
                <CheckCircle2 className="h-3.5 w-3.5 text-cyan-400" />
                <span className="text-cyan-400">Copied to clipboard</span>
              </>
            ) : (
              <>
                <Share2 className="h-3.5 w-3.5" />
                Share results
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
