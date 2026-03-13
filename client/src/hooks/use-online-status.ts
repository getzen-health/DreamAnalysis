/**
 * useOnlineStatus — tracks browser online/offline state.
 *
 * Listens to the window "online" and "offline" events and reflects the
 * current connectivity state. Initialises from navigator.onLine so the
 * first render is always correct.
 *
 * Usage:
 *   const { isOnline } = useOnlineStatus();
 */

import { useState, useEffect } from "react";

export interface UseOnlineStatusReturn {
  isOnline: boolean;
}

export function useOnlineStatus(): UseOnlineStatusReturn {
  const [isOnline, setIsOnline] = useState<boolean>(
    typeof navigator !== "undefined" ? navigator.onLine : true
  );

  useEffect(() => {
    const handleOnline  = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener("online",  handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("online",  handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  return { isOnline };
}
