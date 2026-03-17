/**
 * HealthSyncDashboard — unified status panel for all 6 health data sources.
 *
 * Shows: icon, name, connected/disconnected badge, last sync time, data types,
 * color-coded freshness (green <1h, yellow 1-24h, red >24h or disconnected),
 * and per-source Sync Now / Disconnect buttons.
 */
import { useState, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Heart,
  Smartphone,
  Activity,
  Zap,
  Brain,
  RefreshCw,
  Unplug,
  Wifi,
  WifiOff,
} from "lucide-react";
import { getMLApiUrl } from "@/lib/ml-api";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface HealthSourceStatus {
  source: string;
  connected: boolean;
  last_sync: string | null; // ISO datetime or null
  data_types: string[];
  freshness: "fresh" | "stale" | "old" | "disconnected";
}

interface HealthSyncStatusResponse {
  sources: HealthSourceStatus[];
  fetched_at: string;
}

// ── Static source definitions (icon, label, canonical data types) ─────────────

const SOURCE_META: Record<
  string,
  { label: string; icon: React.ReactNode; color: string }
> = {
  apple_health: {
    label: "Apple HealthKit",
    icon: <Heart className="h-5 w-5" />,
    color: "text-red-400",
  },
  google_health: {
    label: "Google Health Connect",
    icon: <Smartphone className="h-5 w-5" />,
    color: "text-green-400",
  },
  oura: {
    label: "Oura Ring",
    icon: <Activity className="h-5 w-5" />,
    color: "text-purple-400",
  },
  garmin: {
    label: "Garmin",
    icon: <Zap className="h-5 w-5" />,
    color: "text-blue-400",
  },
  whoop: {
    label: "Whoop",
    icon: <Activity className="h-5 w-5" />,
    color: "text-orange-400",
  },
  muse_eeg: {
    label: "Muse 2 EEG",
    icon: <Brain className="h-5 w-5" />,
    color: "text-indigo-400",
  },
};

const _SOURCE_DATA_TYPES: Record<string, string[]> = {
  apple_health: ["HR", "HRV", "sleep", "steps"],
  google_health: ["steps", "HR", "calories"],
  oura: ["sleep", "readiness", "activity"],
  garmin: ["Body Battery", "stress", "HRV"],
  whoop: ["recovery", "strain", "sleep"],
  muse_eeg: ["connection", "battery", "signal"],
};

const SOURCE_ORDER = [
  "apple_health",
  "google_health",
  "oura",
  "garmin",
  "whoop",
  "muse_eeg",
];

// ── Freshness helpers ─────────────────────────────────────────────────────────

function freshnessColor(freshness: HealthSourceStatus["freshness"]): string {
  switch (freshness) {
    case "fresh":
      return "bg-green-500/20 text-green-400 border-green-500/30";
    case "stale":
      return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
    case "old":
    case "disconnected":
    default:
      return "bg-red-500/20 text-red-400 border-red-500/30";
  }
}

function formatLastSync(isoString: string | null): string {
  if (!isoString) return "Never";
  const date = new Date(isoString);
  if (isNaN(date.getTime())) return "Unknown";
  const diffMs = Date.now() - date.getTime();
  const diffMins = Math.floor(diffMs / 60_000);
  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
}

// ── API calls ─────────────────────────────────────────────────────────────────

async function fetchSyncStatus(): Promise<HealthSyncStatusResponse> {
  const url = `${getMLApiUrl()}/api/health/sync-status`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Health sync status fetch failed: ${res.status}`);
  return res.json();
}

async function triggerSync(source: string): Promise<void> {
  const url = `${getMLApiUrl()}/api/health/sync/${source}`;
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw new Error(`Sync failed for ${source}: ${res.status}`);
}

async function disconnectSource(source: string): Promise<void> {
  const url = `${getMLApiUrl()}/api/health/disconnect/${source}`;
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw new Error(`Disconnect failed for ${source}: ${res.status}`);
}

// ── Sub-component: single source row ─────────────────────────────────────────

interface SourceRowProps {
  status: HealthSourceStatus;
  onSync: (source: string) => void;
  onDisconnect: (source: string) => void;
  isSyncing: boolean;
}

function SourceRow({ status, onSync, onDisconnect, isSyncing }: SourceRowProps) {
  const meta = SOURCE_META[status.source] ?? {
    label: status.source,
    icon: <Activity className="h-5 w-5" />,
    color: "text-muted-foreground",
  };

  return (
    <div
      className="flex flex-col sm:flex-row sm:items-center gap-3 p-3 rounded-lg border border-border/40 bg-muted/20"
      data-testid={`health-source-${status.source}`}
    >
      {/* Icon + name + status */}
      <div className="flex items-center gap-3 flex-1 min-w-0">
        <div className={`shrink-0 ${meta.color}`}>{meta.icon}</div>
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-medium truncate">{meta.label}</span>
            <Badge
              className={`text-xs shrink-0 ${freshnessColor(status.freshness)}`}
              data-testid={`badge-${status.source}`}
            >
              {status.connected ? (
                <>
                  <Wifi className="h-2.5 w-2.5 mr-1" />
                  Connected
                </>
              ) : (
                <>
                  <WifiOff className="h-2.5 w-2.5 mr-1" />
                  Disconnected
                </>
              )}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground mt-0.5">
            Last sync: {formatLastSync(status.last_sync)}
          </p>
          {status.data_types.length > 0 && (
            <p className="text-xs text-muted-foreground mt-0.5 truncate">
              {status.data_types.join(", ")}
            </p>
          )}
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-2 shrink-0">
        <Button
          size="sm"
          variant="outline"
          className="h-7 px-2 text-xs"
          onClick={() => onSync(status.source)}
          disabled={isSyncing || !status.connected}
          data-testid={`btn-sync-${status.source}`}
        >
          <RefreshCw className={`h-3 w-3 mr-1 ${isSyncing ? "animate-spin" : ""}`} />
          Sync
        </Button>
        {status.connected && (
          <Button
            size="sm"
            variant="outline"
            className="h-7 px-2 text-xs text-destructive border-destructive/30 hover:bg-destructive/10"
            onClick={() => onDisconnect(status.source)}
            data-testid={`btn-disconnect-${status.source}`}
          >
            <Unplug className="h-3 w-3 mr-1" />
            Disconnect
          </Button>
        )}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function HealthSyncDashboard() {
  const queryClient = useQueryClient();
  const [syncingSource, setSyncingSource] = useState<string | null>(null);

  const { data, isLoading, isError } = useQuery<HealthSyncStatusResponse>({
    queryKey: ["health-sync-status"],
    queryFn: fetchSyncStatus,
    staleTime: 60_000,
    retry: 1,
  });

  const syncMutation = useMutation({
    mutationFn: triggerSync,
    onMutate: (source) => setSyncingSource(source),
    onSettled: () => {
      setSyncingSource(null);
      queryClient.invalidateQueries({ queryKey: ["health-sync-status"] });
    },
  });

  const disconnectMutation = useMutation({
    mutationFn: disconnectSource,
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["health-sync-status"] });
    },
  });

  const handleSync = useCallback(
    (source: string) => syncMutation.mutate(source),
    [syncMutation]
  );

  const handleDisconnect = useCallback(
    (source: string) => disconnectMutation.mutate(source),
    [disconnectMutation]
  );

  // Build ordered list — merge ML backend data with local device state
  const sourceMap = new Map<string, HealthSourceStatus>(
    (data?.sources ?? []).map((s) => [s.source, s])
  );

  // Check local flags for on-device health connections (Capacitor)
  const localGoogleConnected = (() => {
    try { return localStorage.getItem("ndw_health_connect_granted") === "true"; } catch { return false; }
  })();
  const localAppleConnected = (() => {
    try { return localStorage.getItem("ndw_apple_health_granted") === "true"; } catch { return false; }
  })();

  const orderedSources: HealthSourceStatus[] = SOURCE_ORDER.map((key) => {
    const fromServer = sourceMap.get(key);
    if (fromServer) return fromServer;
    // Override with local device state if server doesn't know
    const localConnected =
      (key === "google_health" && localGoogleConnected) ||
      (key === "apple_health" && localAppleConnected);
    return {
      source: key,
      connected: localConnected,
      last_sync: null,
      data_types: localConnected ? (_SOURCE_DATA_TYPES[key] ?? []) : [],
      freshness: localConnected ? ("stale" as const) : ("disconnected" as const),
    };
  });

  // Summary counts
  const connectedCount = orderedSources.filter((s) => s.connected).length;

  return (
    <Card data-testid="health-sync-dashboard">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <Heart className="h-5 w-5" />
            Health Sync Status
          </CardTitle>
          <Badge
            className={
              connectedCount > 0
                ? "bg-green-500/20 text-green-400 border-green-500/30"
                : "bg-muted text-muted-foreground"
            }
          >
            {connectedCount}/{SOURCE_ORDER.length} connected
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Manage connections to health data sources. Color indicates data freshness: green
          (&lt;1h), yellow (1–24h), red (&gt;24h or disconnected).
        </p>
      </CardHeader>

      <CardContent>
        {isLoading && (
          <div className="text-sm text-muted-foreground text-center py-4">
            Loading health source status…
          </div>
        )}

        {isError && !orderedSources.some(s => s.connected) && (
          <div className="text-sm text-muted-foreground text-center py-4">
            Server sync status unavailable. Showing on-device connection status.
          </div>
        )}

        <div className="space-y-2">
          {orderedSources.map((source) => (
            <SourceRow
              key={source.source}
              status={source}
              onSync={handleSync}
              onDisconnect={handleDisconnect}
              isSyncing={syncingSource === source.source}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
