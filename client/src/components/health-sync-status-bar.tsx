/**
 * HealthSyncStatusBar -- shows sync status, data summary, Sync Now button,
 * and guidance when health data is empty.
 *
 * Designed to sit at the top of the Health page (health.tsx) so users know
 * what data has been synced and what to do if nothing appears.
 */

import { useMemo } from "react";
import { RefreshCw, Smartphone, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  buildSyncSummary,
  formatSyncSummary,
  getEmptyDataGuidance,
  type BiometricPayload,
  type HealthSyncStatus,
} from "@/lib/health-sync";

// ── Helpers ─────────────────────────────────────────────────────────────────

function formatSyncTime(d: Date | null): string {
  if (!d) return "never";
  const diffMin = Math.round((Date.now() - d.getTime()) / 60_000);
  if (diffMin < 1) return "just now";
  if (diffMin === 1) return "1 min ago";
  if (diffMin < 60) return `${diffMin} min ago`;
  const h = Math.round(diffMin / 60);
  return h === 1 ? "1h ago" : `${h}h ago`;
}

// ── Props ───────────────────────────────────────────────────────────────────

interface HealthSyncStatusBarProps {
  status: HealthSyncStatus;
  lastSyncAt: Date | null;
  latestPayload: BiometricPayload | null;
  onSyncNow: () => void;
  platform: "ios" | "android" | "web";
}

// ── Component ───────────────────────────────────────────────────────────────

export function HealthSyncStatusBar({
  status,
  lastSyncAt,
  latestPayload,
  onSyncNow,
  platform,
}: HealthSyncStatusBarProps) {
  const isSyncing = status === "syncing";
  const summary = useMemo(() => buildSyncSummary(latestPayload), [latestPayload]);
  const summaryText = useMemo(() => formatSyncSummary(summary), [summary]);
  const guidance = useMemo(
    () => (!summary.hasData ? getEmptyDataGuidance(platform) : null),
    [summary.hasData, platform],
  );

  return (
    <div
      className="rounded-2xl p-4 bg-card border border-border shadow-[0_2px_16px_rgba(0,0,0,0.06)] space-y-3"
      data-testid="health-sync-status-bar"
    >
      {/* Header row: icon + title + last sync */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Smartphone className="h-4 w-4 text-primary" />
          <p className="text-[13px] font-semibold text-foreground">Health Sync</p>
        </div>
        <span className="text-[10px] text-muted-foreground">
          Last synced: {formatSyncTime(lastSyncAt)}
        </span>
      </div>

      {/* Data summary line */}
      <p className="text-xs text-muted-foreground">{summaryText}</p>

      {/* Empty data guidance */}
      {guidance && (
        <div className="flex items-start gap-2 p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20">
          <AlertCircle className="h-3.5 w-3.5 text-amber-500 shrink-0 mt-0.5" />
          <p className="text-[11px] text-amber-600 dark:text-amber-400 leading-snug">
            {guidance}
          </p>
        </div>
      )}

      {/* Sync Now button */}
      <Button
        onClick={onSyncNow}
        disabled={isSyncing}
        className="w-full h-10 text-sm font-semibold gap-2"
        variant="outline"
      >
        <RefreshCw className={`h-4 w-4 ${isSyncing ? "animate-spin" : ""}`} />
        {isSyncing ? "Syncing..." : "Sync Now"}
      </Button>
    </div>
  );
}
