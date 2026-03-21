/**
 * useFusedState — React hook that subscribes to the data fusion layer.
 *
 * Returns the current fused state from EEG + voice + health sources,
 * auto-updating when any source changes.
 *
 * Usage:
 *   const { fusedState, source, isReady } = useFusedState();
 */

import { useState, useEffect } from "react";
import { dataFusionBus, type FusedState, type FusionSource } from "@/lib/data-fusion";

export interface UseFusedStateReturn {
  /** The current fused state, or null if no data from any source. */
  fusedState: FusedState | null;
  /** Which source is dominant: "eeg", "voice", "health", or "fused" (multiple). */
  source: FusionSource | null;
  /** True when at least one data source is providing readings. */
  isReady: boolean;
}

export function useFusedState(): UseFusedStateReturn {
  const [state, setState] = useState<FusedState | null>(() => dataFusionBus.getState());

  useEffect(() => {
    // Initialize the bus (idempotent — safe to call multiple times)
    dataFusionBus.initialize();

    // Subscribe to state updates
    const unsubscribe = dataFusionBus.subscribe((newState) => {
      setState(newState);
    });

    return unsubscribe;
  }, []);

  return {
    fusedState: state,
    source: state?.source ?? null,
    isReady: state !== null,
  };
}
