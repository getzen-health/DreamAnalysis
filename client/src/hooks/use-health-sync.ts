/**
 * use-health-sync.ts — React hook for Apple HealthKit / Google Health Connect.
 *
 * Initializes the health sync manager on mount, starts auto-sync every 15 min,
 * and exposes current sync state + a manual trigger.
 *
 * Usage:
 *   const { status, lastSyncAt, latestPayload, syncNow } = useHealthSync();
 */

import { useState, useEffect, useCallback } from "react";
import { healthSync, type HealthSyncState } from "@/lib/health-sync";

export interface UseHealthSyncReturn {
  status: HealthSyncState["status"];
  lastSyncAt: Date | null;
  latestPayload: HealthSyncState["latestPayload"];
  error: string | null;
  /** Manually trigger an immediate sync. */
  syncNow: () => Promise<void>;
  /** True if health data is available on this platform (iOS or Android). */
  isAvailable: boolean;
}

export function useHealthSync(): UseHealthSyncReturn {
  const [state, setState] = useState<HealthSyncState>(healthSync.getState());

  useEffect(() => {
    // Subscribe to state changes
    const unsubscribe = healthSync.subscribe((s) => setState(s));

    // Initialize and start syncing — try regardless of errors
    (async () => {
      try {
        await healthSync.initialize();
      } catch {
        // Permissions denied or platform unavailable — still try sync for cached data
      }
      try {
        healthSync.startAutoSync();
      } catch {
        // Auto-sync failed — at least we have cached data from localStorage
      }
    })();

    return () => {
      unsubscribe();
      healthSync.stopAutoSync();
    };
  }, []);

  const syncNow = useCallback(() => healthSync.syncNow(), []);

  return {
    status: state.status,
    lastSyncAt: state.lastSyncAt,
    latestPayload: state.latestPayload,
    error: state.error,
    syncNow,
    isAvailable: state.status !== "unavailable",
  };
}
