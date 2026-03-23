/**
 * useInterventionTriggers — React hook for the EEG intervention trigger engine.
 *
 * Checks triggers every 5 seconds during active EEG sessions.
 * Manages trigger state for the toast component and logs events.
 *
 * @see Issue #504
 */

import { useState, useEffect, useRef, useCallback } from "react";
import {
  checkTriggers,
  loadTriggerConfig,
  type InterventionTrigger,
  type TriggerState,
} from "@/lib/eeg-intervention-trigger";

const CHECK_INTERVAL_MS = 5_000; // 5 seconds

export interface TriggerEvent {
  trigger: InterventionTrigger;
  timestamp: number;
  state: TriggerState;
}

export function useInterventionTriggers(
  isStreaming: boolean,
  getState: () => TriggerState | null,
) {
  const [activeTrigger, setActiveTrigger] = useState<InterventionTrigger | null>(null);
  const [triggerLog, setTriggerLog] = useState<TriggerEvent[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const dismiss = useCallback(() => {
    setActiveTrigger(null);
  }, []);

  useEffect(() => {
    if (!isStreaming) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    intervalRef.current = setInterval(() => {
      const state = getState();
      if (!state) return;

      const config = loadTriggerConfig();
      const trigger = checkTriggers(state, config);

      if (trigger) {
        setActiveTrigger(trigger);
        setTriggerLog((prev) => [
          ...prev,
          { trigger, timestamp: Date.now(), state },
        ]);
      }
    }, CHECK_INTERVAL_MS);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isStreaming, getState]);

  return { activeTrigger, dismiss, triggerLog };
}
