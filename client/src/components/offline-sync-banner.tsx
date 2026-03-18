/**
 * OfflineSyncBanner — persistent thin banner when offline.
 *
 * Shows:
 *  - "offline" state: yellow, WifiOff icon, queue size
 *  - "syncing" state: blue, spinner
 *  - "synced" state: green, checkmark, item count (auto-hides after 3 s)
 *
 * Also listens for SW_BACKGROUND_SYNC messages from the service worker
 * and triggers syncAll() in response.
 */

import { useState, useEffect } from "react";
import { WifiOff, RefreshCw, CheckCircle2 } from "lucide-react";
import { syncAll, getOfflineQueueSize } from "@/lib/offline-store";
import { useOnlineStatus } from "@/hooks/use-online-status";
import { getParticipantId } from "@/lib/participant";

type BannerState = "online" | "offline" | "syncing" | "synced";

export default function OfflineSyncBanner() {
  const { isOnline } = useOnlineStatus();
  const [bannerState, setBannerState] = useState<BannerState>(
    isOnline ? "online" : "offline"
  );
  const [syncCount, setSyncCount] = useState(0);
  const [queueSize, setQueueSize] = useState(0);

  // Refresh queue size periodically while offline
  useEffect(() => {
    if (!isOnline) {
      getOfflineQueueSize().then(setQueueSize).catch(() => {});
      const interval = setInterval(() => {
        getOfflineQueueSize().then(setQueueSize).catch(() => {});
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [isOnline]);

  // Handle online → sync
  useEffect(() => {
    if (isOnline && bannerState === "offline") {
      runSync();
    } else if (!isOnline) {
      setBannerState("offline");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOnline]);

  // Listen for SW_BACKGROUND_SYNC messages
  useEffect(() => {
    if (!("serviceWorker" in navigator)) return;

    const handleMessage = (event: MessageEvent) => {
      if (event.data?.type === "SW_BACKGROUND_SYNC" && navigator.onLine) {
        runSync();
      }
    };

    navigator.serviceWorker.addEventListener("message", handleMessage);
    return () => navigator.serviceWorker.removeEventListener("message", handleMessage);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function runSync() {
    setBannerState("syncing");
    try {
      const userId = getParticipantId();
      const result = await syncAll(userId);
      const total = result.dreams + result.sessions + result.metrics + result.voice + result.food;
      setSyncCount(total);
      setBannerState("synced");
      setTimeout(() => setBannerState("online"), 3000);
    } catch {
      setBannerState("online");
    }
  }

  if (bannerState === "online") return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className={`fixed top-0 left-0 right-0 z-[200] flex items-center justify-center gap-2 px-4 py-1.5 text-xs font-medium transition-all ${
        bannerState === "offline"
          ? "bg-yellow-500/90 text-yellow-950"
          : bannerState === "syncing"
            ? "bg-indigo-500/90 text-white"
            : "bg-cyan-600/90 text-white"
      }`}
    >
      {bannerState === "offline" && (
        <>
          <WifiOff className="h-3 w-3 shrink-0" />
          Offline — data saved locally and will sync when reconnected
          {queueSize > 0 && (
            <span className="ml-1 rounded-full bg-yellow-900/30 px-1.5 py-0.5 tabular-nums">
              {queueSize} queued
            </span>
          )}
        </>
      )}
      {bannerState === "syncing" && (
        <>
          <RefreshCw className="h-3 w-3 shrink-0 animate-spin" />
          Syncing queued data...
        </>
      )}
      {bannerState === "synced" && (
        <>
          <CheckCircle2 className="h-3 w-3 shrink-0" />
          {syncCount > 0
            ? `Synced ${syncCount} item${syncCount !== 1 ? "s" : ""}`
            : "Back online"}
        </>
      )}
    </div>
  );
}
