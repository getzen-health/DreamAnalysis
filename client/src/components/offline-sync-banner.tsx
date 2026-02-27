/**
 * OfflineSyncBanner — shows a thin banner when offline.
 * Auto-syncs queued data when connection is restored.
 */

import { useState, useEffect } from "react";
import { WifiOff, RefreshCw, CheckCircle2 } from "lucide-react";
import { syncAll } from "@/lib/offline-store";
import { getParticipantId } from "@/lib/participant";

type BannerState = "online" | "offline" | "syncing" | "synced";

export default function OfflineSyncBanner() {
  const [bannerState, setBannerState] = useState<BannerState>(
    navigator.onLine ? "online" : "offline"
  );
  const [syncCount, setSyncCount] = useState(0);

  useEffect(() => {
    const handleOffline = () => setBannerState("offline");

    const handleOnline = async () => {
      setBannerState("syncing");
      try {
        const userId = getParticipantId();
        const result = await syncAll(userId);
        const total = result.dreams + result.sessions + result.metrics;
        setSyncCount(total);
        setBannerState("synced");
        // Auto-hide after 3s
        setTimeout(() => setBannerState("online"), 3000);
      } catch {
        setBannerState("online");
      }
    };

    window.addEventListener("offline", handleOffline);
    window.addEventListener("online", handleOnline);
    return () => {
      window.removeEventListener("offline", handleOffline);
      window.removeEventListener("online", handleOnline);
    };
  }, []);

  if (bannerState === "online") return null;

  return (
    <div
      className={`fixed top-0 left-0 right-0 z-[200] flex items-center justify-center gap-2 px-4 py-1.5 text-xs font-medium transition-all ${
        bannerState === "offline"
          ? "bg-yellow-500/90 text-yellow-950"
          : bannerState === "syncing"
            ? "bg-blue-500/90 text-white"
            : "bg-green-500/90 text-white"
      }`}
    >
      {bannerState === "offline" && (
        <>
          <WifiOff className="h-3 w-3" />
          Offline — data saved locally and will sync when reconnected
        </>
      )}
      {bannerState === "syncing" && (
        <>
          <RefreshCw className="h-3 w-3 animate-spin" />
          Syncing queued data...
        </>
      )}
      {bannerState === "synced" && (
        <>
          <CheckCircle2 className="h-3 w-3" />
          {syncCount > 0 ? `Synced ${syncCount} item${syncCount !== 1 ? "s" : ""}` : "Back online"}
        </>
      )}
    </div>
  );
}
