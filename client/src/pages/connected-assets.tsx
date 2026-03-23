import { useState, useEffect, useCallback, useRef } from "react";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { useDevice } from "@/hooks/use-device";
import { useQuery } from "@tanstack/react-query";
import { getParticipantId } from "@/lib/participant";
import { resolveUrl } from "@/lib/queryClient";
import { ingestHealthData } from "@/lib/ml-api";
import { requestHealthWritePermissions } from "@/lib/health-connect";
import {
  Heart,
  Brain,
  Watch,
  Apple,
  Smartphone,
  Upload,
  CheckCircle2,
  XCircle,
  Info,
  Link,
  Unlink,
  Loader2,
  RefreshCw,
  ChevronRight,
} from "lucide-react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { sbGetSetting, sbSaveSetting } from "../lib/supabase-store";

/* -- Types -------------------------------------------------- */

const USER_ID = getParticipantId();

interface DeviceInfo {
  id: string;
  provider: string;
  lastSyncAt: string | null;
  syncStatus: string;
  errorMessage: string | null;
  connectedAt: string;
  scopes: string[] | null;
}

/* -- Platform Detection ------------------------------------- */

function usePlatform() {
  const [platform, setPlatform] = useState<"web" | "ios" | "android">("web");
  useEffect(() => {
    import("@capacitor/core")
      .then(({ Capacitor }) => {
        const p = Capacitor.getPlatform();
        setPlatform(p === "ios" ? "ios" : p === "android" ? "android" : "web");
      })
      .catch(() => {});
  }, []);
  return platform;
}

/* -- Status indicator --------------------------------------- */

function StatusDot({ connected }: { connected: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      {connected ? (
        <>
          <CheckCircle2 className="h-3 w-3 text-green-500" />
          <span className="text-xs text-green-500">Connected</span>
        </>
      ) : (
        <>
          <XCircle className="h-3 w-3 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">Not connected</span>
        </>
      )}
    </div>
  );
}

/* -- Main Component ----------------------------------------- */

export default function ConnectedAssets() {
  const platform = usePlatform();
  const { toast } = useToast();
  const [, setLocation] = useLocation();
  const { state: deviceState, deviceStatus } = useDevice();
  const fileRef = useRef<HTMLInputElement>(null);

  // Health connect state
  const [isConnectingHealth, setIsConnectingHealth] = useState(false);
  const [healthConnected, setHealthConnected] = useState(false);
  const [uploading, setUploading] = useState(false);

  // Wearable state
  const [connectingProvider, setConnectingProvider] = useState<string | null>(null);
  const [syncingProvider, setSyncingProvider] = useState<string | null>(null);
  const [disconnectingProvider, setDisconnectingProvider] = useState<string | null>(null);

  // EEG status
  const isMuseConnected = deviceState === "connected" || deviceState === "streaming";

  // Health connect status
  useEffect(() => {
    if (platform === "android") {
      setHealthConnected(sbGetSetting("ndw_health_connect_granted") === "true");
    } else if (platform === "ios") {
      setHealthConnected(sbGetSetting("ndw_apple_health_granted") === "true");
    } else {
      const gfit = sbGetSetting("ndw_health_connect_granted") === "true";
      const apple = sbGetSetting("ndw_apple_health_granted") === "true";
      setHealthConnected(gfit || apple);
    }
  }, [platform]);

  // Wearable device query
  const { data: devicesData, refetch: refetchDevices } = useQuery({
    queryKey: ["devices", USER_ID],
    queryFn: async () => {
      const token = localStorage.getItem("auth_token");
      const res = await fetch(resolveUrl(`/api/devices/${USER_ID}`), {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) return { devices: [] as DeviceInfo[] };
      return res.json() as Promise<{ devices: DeviceInfo[] }>;
    },
    staleTime: 30_000,
  });

  const connectedDevices = devicesData?.devices ?? [];

  // Listen for OAuth popup completion
  useEffect(() => {
    function handleMessage(e: MessageEvent) {
      if (e.data?.type === "device-connected") {
        refetchDevices();
        toast({
          title: "Device connected",
          description: `${e.data.provider} has been connected and initial sync started.`,
        });
        setConnectingProvider(null);
      }
    }
    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, [refetchDevices, toast]);

  // -- Handlers -- //

  const handleConnectHealth = async () => {
    setIsConnectingHealth(true);
    try {
      if (platform === "android") {
        const { Health } = await import("capacitor-health");
        const available = await Health.isHealthAvailable();
        if (!available.available) {
          toast({ title: "Health Connect Not Available", description: "Install Google Health Connect from the Play Store.", variant: "destructive" });
          return;
        }
        await Health.requestHealthPermissions({
          permissions: ["READ_STEPS", "READ_HEART_RATE", "READ_ACTIVE_CALORIES", "READ_WORKOUTS", "READ_MINDFULNESS"],
        });
        requestHealthWritePermissions().catch(() => {});
        setHealthConnected(true);
        sbSaveSetting("ndw_health_connect_granted", "true");
        toast({ title: "Connected", description: "Google Health Connect linked." });
      } else if (platform === "ios") {
        await fetch(resolveUrl("/api/health/connect"), { method: "POST" });
        requestHealthWritePermissions().catch(() => {});
        setHealthConnected(true);
        sbSaveSetting("ndw_apple_health_granted", "true");
        toast({ title: "Connected", description: "Apple HealthKit linked." });
      } else {
        toast({ title: "Mobile Only", description: "Health Connect is available on the mobile app.", variant: "destructive" });
      }
    } catch (e) {
      toast({ title: "Connection Failed", description: String(e), variant: "destructive" });
    } finally {
      setIsConnectingHealth(false);
    }
  };

  const handleUpload = async (file: File) => {
    setUploading(true);
    try {
      const text = await file.text();
      let data: Record<string, unknown>;
      if (file.name.endsWith(".json")) {
        data = JSON.parse(text);
      } else {
        data = { raw_xml: text, filename: file.name };
      }
      const source = platform === "ios" ? "apple_health" : "google_fit";
      const result = await ingestHealthData(USER_ID, source, data);
      toast({ title: "Data Imported", description: `Imported ${result.stored} samples (${result.metrics.join(", ")}).` });
    } catch {
      toast({ title: "Import Failed", description: "Could not parse the export file.", variant: "destructive" });
    } finally {
      setUploading(false);
    }
  };

  const handleConnectWearable = async (provider: string) => {
    setConnectingProvider(provider);
    try {
      const token = localStorage.getItem("auth_token");
      const res = await fetch(resolveUrl(`/api/devices/connect/${provider}`), {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error((err as Record<string, string>).error ?? "Failed to start connection");
      }
      const { authUrl } = await res.json();
      const popup = window.open(authUrl, `connect-${provider}`, "width=600,height=700,scrollbars=yes");
      if (!popup) {
        toast({ title: "Popup blocked", description: "Allow popups for this site and try again.", variant: "destructive" });
        setConnectingProvider(null);
      }
    } catch (error) {
      toast({ title: "Connection failed", description: String(error), variant: "destructive" });
      setConnectingProvider(null);
    }
  };

  const handleSyncWearable = async (provider: string) => {
    setSyncingProvider(provider);
    try {
      const token = localStorage.getItem("auth_token");
      const res = await fetch(resolveUrl(`/api/devices/sync/${provider}`), {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error((err as Record<string, string>).error ?? "Sync failed");
      }
      const data = await res.json();
      toast({ title: "Sync complete", description: `Synced ${data.synced} health samples from ${provider}.` });
      refetchDevices();
    } catch (error) {
      toast({ title: "Sync failed", description: String(error), variant: "destructive" });
    } finally {
      setSyncingProvider(null);
    }
  };

  const handleDisconnectWearable = async (provider: string) => {
    setDisconnectingProvider(provider);
    try {
      const token = localStorage.getItem("auth_token");
      const res = await fetch(resolveUrl(`/api/devices/${provider}`), {
        method: "DELETE",
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error((err as Record<string, string>).error ?? "Failed to disconnect");
      }
      toast({ title: "Device disconnected", description: `${provider} has been disconnected.` });
      refetchDevices();
    } catch (error) {
      toast({ title: "Disconnect failed", description: String(error), variant: "destructive" });
    } finally {
      setDisconnectingProvider(null);
    }
  };

  // -- Device definitions -- //

  const healthName = platform === "android" ? "Google Health Connect" : platform === "ios" ? "Apple HealthKit" : "Health Connect";
  const HealthIcon = platform === "ios" ? Apple : Smartphone;

  interface DeviceRow {
    id: string;
    name: string;
    icon: React.ComponentType<{ className?: string }>;
    iconBg: string;
    iconColor: string;
    connected: boolean;
    description?: string;
    lastSync?: string | null;
    error?: string | null;
    onConnect: () => void;
    onSync?: () => void;
    onDisconnect?: () => void;
    connectLabel?: string;
    isConnecting?: boolean;
    isSyncing?: boolean;
    isDisconnecting?: boolean;
    showUpload?: boolean;
    webOnly?: string;
  }

  const ouraConnection = connectedDevices.find((d) => d.provider === "oura");
  const whoopConnection = connectedDevices.find((d) => d.provider === "whoop");
  const garminConnection = connectedDevices.find((d) => d.provider === "garmin");

  const devices: DeviceRow[] = [
    {
      id: "health",
      name: healthName,
      icon: HealthIcon,
      iconBg: "bg-rose-500/10",
      iconColor: "text-rose-400",
      connected: healthConnected,
      description: "Heart rate, sleep, steps, workouts, mindfulness",
      onConnect: handleConnectHealth,
      connectLabel: healthConnected ? "Reconnect" : "Connect",
      isConnecting: isConnectingHealth,
      showUpload: platform !== "web",
      webOnly: platform === "web" ? "Use the mobile app to connect" : undefined,
    },
    {
      id: "muse2",
      name: deviceStatus?.device_type ?? "Muse 2",
      icon: Brain,
      iconBg: "bg-indigo-500/10",
      iconColor: "text-indigo-400",
      connected: isMuseConnected,
      description: "EEG brain wave monitoring at 256 Hz",
      onConnect: () => setLocation("/device-setup"),
      connectLabel: isMuseConnected ? "Manage" : "Setup",
    },
    {
      id: "muse-s",
      name: "Muse S",
      icon: Brain,
      iconBg: "bg-violet-500/10",
      iconColor: "text-violet-400",
      connected: false,
      description: "Sleep-optimized EEG headband",
      onConnect: () => setLocation("/device-setup"),
      connectLabel: "Setup",
    },
    {
      id: "synthetic",
      name: "Synthetic Demo",
      icon: Brain,
      iconBg: "bg-emerald-500/10",
      iconColor: "text-emerald-400",
      connected: false,
      description: "Simulated EEG data for testing",
      onConnect: () => setLocation("/device-setup"),
      connectLabel: "Setup",
    },
    {
      id: "oura",
      name: "Oura Ring",
      icon: Watch,
      iconBg: "bg-indigo-400/10",
      iconColor: "text-indigo-400",
      connected: !!ouraConnection,
      description: "Readiness, sleep, activity, heart rate",
      lastSync: ouraConnection?.lastSyncAt,
      error: ouraConnection?.errorMessage,
      onConnect: () => handleConnectWearable("oura"),
      onSync: () => handleSyncWearable("oura"),
      onDisconnect: () => handleDisconnectWearable("oura"),
      isConnecting: connectingProvider === "oura",
      isSyncing: syncingProvider === "oura",
      isDisconnecting: disconnectingProvider === "oura",
    },
    {
      id: "whoop",
      name: "WHOOP",
      icon: Watch,
      iconBg: "bg-yellow-500/10",
      iconColor: "text-yellow-500",
      connected: !!whoopConnection,
      description: "Recovery, strain, sleep, HRV",
      lastSync: whoopConnection?.lastSyncAt,
      error: whoopConnection?.errorMessage,
      onConnect: () => handleConnectWearable("whoop"),
      onSync: () => handleSyncWearable("whoop"),
      onDisconnect: () => handleDisconnectWearable("whoop"),
      isConnecting: connectingProvider === "whoop",
      isSyncing: syncingProvider === "whoop",
      isDisconnecting: disconnectingProvider === "whoop",
    },
    {
      id: "garmin",
      name: "Garmin",
      icon: Watch,
      iconBg: "bg-cyan-400/10",
      iconColor: "text-cyan-400",
      connected: !!garminConnection,
      description: "Steps, stress, body battery, workouts",
      lastSync: garminConnection?.lastSyncAt,
      error: garminConnection?.errorMessage,
      onConnect: () => handleConnectWearable("garmin"),
      onSync: () => handleSyncWearable("garmin"),
      onDisconnect: () => handleDisconnectWearable("garmin"),
      isConnecting: connectingProvider === "garmin",
      isSyncing: syncingProvider === "garmin",
      isDisconnecting: disconnectingProvider === "garmin",
    },
  ];

  return (
    <main className="p-4 md:p-6 pb-24 max-w-3xl">
      {/* Header */}
      <div className="flex items-center gap-3 mb-5">
        <Link className="h-6 w-6 text-primary" />
        <div>
          <h1 className="text-xl font-semibold">Connected Assets</h1>
          <p className="text-xs text-muted-foreground">
            Manage your health, brain, and wearable connections
          </p>
        </div>
      </div>

      {/* Flat device list */}
      <div className="space-y-3">
        {devices.map((device) => {
          const DeviceIcon = device.icon;
          return (
            <div
              key={device.id}
              data-testid={`device-${device.id}`}
              className={`rounded-2xl border p-4 ${
                device.connected
                  ? "border-primary/20 bg-primary/[0.03]"
                  : "border-border/50 bg-card/50"
              }`}
              style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div
                    className={`w-10 h-10 rounded-xl flex items-center justify-center ${device.iconBg}`}
                  >
                    <DeviceIcon className={`h-5 w-5 ${device.iconColor}`} />
                  </div>
                  <div>
                    <p className="text-sm font-medium">{device.name}</p>
                    <StatusDot connected={device.connected} />
                    {device.description && (
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        {device.description}
                      </p>
                    )}
                    {device.lastSync && (
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        Last sync: {new Date(device.lastSync).toLocaleString()}
                      </p>
                    )}
                    {device.error && (
                      <p className="text-[10px] text-destructive mt-0.5">
                        {device.error}
                      </p>
                    )}
                    {device.webOnly && (
                      <div className="flex items-center gap-1 mt-0.5">
                        <Info className="h-3 w-3 text-muted-foreground" />
                        <span className="text-[10px] text-muted-foreground">{device.webOnly}</span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2 flex-shrink-0">
                  {/* Upload button for health connect */}
                  {device.showUpload && (
                    <>
                      <Input
                        ref={fileRef}
                        type="file"
                        accept=".xml,.zip,.json"
                        className="hidden"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) handleUpload(file);
                        }}
                      />
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={uploading}
                        onClick={() => fileRef.current?.click()}
                        className="h-8"
                      >
                        <Upload className="h-3.5 w-3.5 mr-1" />
                        {uploading ? "..." : "Upload"}
                      </Button>
                    </>
                  )}

                  {/* Connected wearable: sync + disconnect */}
                  {device.connected && device.onSync && (
                    <>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={device.onSync}
                        disabled={device.isSyncing}
                        className="h-8"
                      >
                        {device.isSyncing ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <RefreshCw className="h-3.5 w-3.5" />
                        )}
                        <span className="ml-1.5 text-xs">
                          {device.isSyncing ? "..." : "Sync"}
                        </span>
                      </Button>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-8 border-destructive/30 text-destructive hover:bg-destructive/10"
                            disabled={device.isDisconnecting}
                          >
                            {device.isDisconnecting ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            ) : (
                              <Unlink className="h-3.5 w-3.5" />
                            )}
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent className="glass-card">
                          <AlertDialogHeader>
                            <AlertDialogTitle>Disconnect {device.name}?</AlertDialogTitle>
                            <AlertDialogDescription>
                              This removes the connection. Previously synced data stays in your
                              profile. You can reconnect at any time.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>Cancel</AlertDialogCancel>
                            <AlertDialogAction
                              onClick={device.onDisconnect}
                              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                            >
                              Disconnect
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </>
                  )}

                  {/* Connect / Setup button */}
                  {!(device.connected && device.onSync) && (
                    <Button
                      size="sm"
                      onClick={device.onConnect}
                      disabled={device.isConnecting || (device.id === "health" && platform === "web")}
                      className="h-8"
                    >
                      {device.isConnecting ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />
                      ) : null}
                      <span className="text-xs">
                        {device.isConnecting ? "..." : device.connectLabel ?? "Connect"}
                      </span>
                      {!device.isConnecting && (device.id === "muse2" || device.id === "muse-s" || device.id === "synthetic") && (
                        <ChevronRight className="h-3.5 w-3.5 ml-1" />
                      )}
                    </Button>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </main>
  );
}
