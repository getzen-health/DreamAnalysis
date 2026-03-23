/**
 * use-sync-status.ts — React hook exposing sync queue status for UI display.
 *
 * Shows: "Synced", "Pending 3 items", "Offline — data saved locally"
 *
 * Usage:
 *   const { state, pendingCount, displayText, enqueue } = useSyncStatus();
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { SyncQueue, type QueuedWrite, type SyncState } from "@/lib/sync-queue";

export interface UseSyncStatusReturn {
  state: SyncState;
  pendingCount: number;
  displayText: string;
  enqueue: (write: QueuedWrite) => void;
}

export function useSyncStatus(): UseSyncStatusReturn {
  const queueRef = useRef<SyncQueue>(new SyncQueue());

  const getSnapshot = () => {
    const q = queueRef.current;
    return {
      state: q.getStatus().state,
      pendingCount: q.getStatus().pendingCount,
      displayText: q.getDisplayText(),
    };
  };

  const [snapshot, setSnapshot] = useState(getSnapshot);

  const refresh = useCallback(() => {
    setSnapshot(getSnapshot());
  }, []);

  const enqueue = useCallback(
    (write: QueuedWrite) => {
      queueRef.current.enqueue(write);
      refresh();
    },
    [refresh],
  );

  useEffect(() => {
    const handleOnline = () => refresh();
    const handleOffline = () => refresh();

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, [refresh]);

  return {
    ...snapshot,
    enqueue,
  };
}
