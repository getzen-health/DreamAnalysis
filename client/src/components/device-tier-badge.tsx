/**
 * device-tier-badge.tsx — Shows the current device tier as a compact badge.
 *
 * Color coding:
 *   eeg_full   → green   (full pipeline)
 *   eeg_basic  → amber   (limited staging)
 *   phone_only → blue    (accelerometer + mic)
 *   none       → gray    (journal only)
 */

import { type DeviceTier, tierLabel } from "@/lib/dream-pipeline";

const TIER_STYLES: Record<DeviceTier, string> = {
  eeg_full:   "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30",
  eeg_basic:  "bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30",
  phone_only: "bg-blue-500/15 text-blue-600 dark:text-blue-400 border-blue-500/30",
  none:       "bg-muted text-muted-foreground border-border",
};

interface DeviceTierBadgeProps {
  tier: DeviceTier;
  className?: string;
}

export default function DeviceTierBadge({ tier, className = "" }: DeviceTierBadgeProps) {
  return (
    <span
      data-testid="device-tier-badge"
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium leading-tight ${TIER_STYLES[tier]} ${className}`}
    >
      {tierLabel(tier)}
    </span>
  );
}
