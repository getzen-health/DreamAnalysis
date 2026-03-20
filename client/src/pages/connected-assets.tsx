import { useState, useEffect, useCallback, useRef } from "react";
import { useLocation } from "wouter";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { useDevice } from "@/hooks/use-device";
import { useQuery } from "@tanstack/react-query";
import { getParticipantId } from "@/lib/participant";
import { resolveUrl } from "@/lib/queryClient";
import { ingestHealthData } from "@/lib/ml-api";
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

/* ── Types ──────────────────────────────────────────────────── */

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

/* ── Platform Detection ─────────────────────────────────────── */

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

/* ── Section Header ─────────────────────────────────────────── */

function SectionHeader({
  icon: Icon,
  iconColor,
  title,
  subtitle,
}: {
  icon: React.ComponentType<{ className?: string }>;
  iconColor: string;
  title: string;
  subtitle: string;
}) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${iconColor}`}>
        <Icon className="h-5 w-5" />
      </div>
      <div>
        <h3 className="text-base font-semibold">{title}</h3>
        <p className="text-xs text-muted-foreground">{subtitle}</p>
      </div>
    </div>
  );
}

/* ── Connect Health Section ─────────────────────────────────── */

function ConnectHealthSection() {
  const platform = usePlatform();
  const { toast } = useToast();
  const [isConnecting, setIsConnecting] = useState(false);
  const [healthConnected, setHealthConnected] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // Check stored connection status
  useEffect(() => {
    if (platform === "android") {
      setHealthConnected(localStorage.getItem("ndw_health_connect_granted") === "true");
    } else if (platform === "ios") {
      setHealthConnected(localStorage.getItem("ndw_apple_health_granted") === "true");
    } else {
      const gfit = localStorage.getItem("ndw_health_connect_granted") === "true";
      const apple = localStorage.getItem("ndw_apple_health_granted") === "true";
      setHealthConnected(gfit || apple);
    }
  }, [platform]);

  const handleConnect = async () => {
    setIsConnecting(true);
    try {
      if (platform === "android") {
        const { Health } = await import("capacitor-health");
        const available = await Health.isHealthAvailable();
        if (!available.available) {
          toast({
            title: "Health Connect Not Available",
            description: "Install Google Health Connect from the Play Store.",
            variant: "destructive",
          });
          return;
        }
        await Health.requestHealthPermissions({
          permissions: [
            "READ_STEPS",
            "READ_HEART_RATE",
            "READ_ACTIVE_CALORIES",
            "READ_WORKOUTS",
            "READ_MINDFULNESS",
          ],
        });
        setHealthConnected(true);
        localStorage.setItem("ndw_health_connect_granted", "true");
        toast({
          title: "Connected",
          description: "Google Health Connect linked. Data will sync automatically.",
        });
      } else if (platform === "ios") {
        await fetch(resolveUrl("/api/health/connect"), { method: "POST" });
        setHealthConnected(true);
        localStorage.setItem("ndw_apple_health_granted", "true");
        toast({
          title: "Connected",
          description: "Apple HealthKit linked. Data will sync automatically.",
        });
      } else {
        toast({
          title: "Mobile Only",
          description: "Health Connect is available on the mobile app. Use Android or iOS to connect.",
        });
      }
    } catch (e) {
      toast({
        title: "Connection Failed",
        description: String(e),
        variant: "destructive",
      });
    } finally {
      setIsConnecting(false);
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
      toast({
        title: "Data Imported",
        description: `Imported ${result.stored} samples (${result.metrics.join(", ")}).`,
      });
    } catch {
      toast({
        title: "Import Failed",
        description: "Could not parse the export file.",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  const healthName =
    platform === "android"
      ? "Google Health Connect"
      : platform === "ios"
      ? "Apple HealthKit"
      : "Health Connect";

  const HealthIcon = platform === "ios" ? Apple : Smartphone;

  return (
    <Card className="glass-card p-5">
      <SectionHeader
        icon={Heart}
        iconColor="bg-rose-500/10 text-rose-400"
        title="Connect Health"
        subtitle={
          platform === "android"
            ? "Google Health Connect"
            : platform === "ios"
            ? "Apple HealthKit"
            : "Google Health Connect (Android) / Apple HealthKit (iOS)"
        }
      />

      <div className="rounded-lg border border-border/50 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-rose-500/10">
              <HealthIcon className="h-5 w-5 text-rose-400" />
            </div>
            <div>
              <p className="text-sm font-medium">{healthName}</p>
              <div className="flex items-center gap-1.5 mt-0.5">
                {healthConnected ? (
                  <>
                    <CheckCircle2 className="h-3 w-3 text-green-500" />
                    <span className="text-xs text-green-500">Connected</span>
                  </>
                ) : platform === "web" ? (
                  <>
                    <Info className="h-3 w-3 text-muted-foreground" />
                    <span className="text-xs text-muted-foreground">
                      Use the mobile app to connect
                    </span>
                  </>
                ) : (
                  <>
                    <XCircle className="h-3 w-3 text-muted-foreground" />
                    <span className="text-xs text-muted-foreground">Not connected</span>
                  </>
                )}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {platform !== "web" && (
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
                >
                  <Upload className="h-3.5 w-3.5 mr-1.5" />
                  {uploading ? "..." : "Upload"}
                </Button>
              </>
            )}
            <Button
              size="sm"
              onClick={handleConnect}
              disabled={isConnecting || (platform === "web")}
            >
              {isConnecting ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />
              ) : healthConnected ? (
                "Reconnect"
              ) : (
                "Connect"
              )}
            </Button>
          </div>
        </div>
      </div>

      <p className="text-xs text-muted-foreground mt-3">
        Syncs heart rate, sleep stages, steps, workouts, and mindfulness data.
      </p>
    </Card>
  );
}

/* ── Connect BCI / EEG Section ──────────────────────────────── */

function ConnectBCISection() {
  const [, setLocation] = useLocation();
  const { state: deviceState, deviceStatus } = useDevice();
  const isConnected = deviceState === "connected" || deviceState === "streaming";

  return (
    <Card className="glass-card p-5">
      <SectionHeader
        icon={Brain}
        iconColor="bg-indigo-500/10 text-indigo-400"
        title="BCI / EEG"
        subtitle="Brain-computer interface headbands"
      />

      <div className="space-y-3">
        {/* Current device status */}
        <div className="rounded-lg border border-border/50 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-indigo-500/10">
                <Brain className="h-5 w-5 text-indigo-400" />
              </div>
              <div>
                <p className="text-sm font-medium">
                  {deviceStatus?.device_type ?? "No device connected"}
                </p>
                <div className="flex items-center gap-1.5 mt-0.5">
                  {isConnected ? (
                    <>
                      <CheckCircle2 className="h-3 w-3 text-green-500" />
                      <span className="text-xs text-green-500 capitalize">{deviceState}</span>
                    </>
                  ) : (
                    <>
                      <XCircle className="h-3 w-3 text-muted-foreground" />
                      <span className="text-xs text-muted-foreground capitalize">{deviceState}</span>
                    </>
                  )}
                </div>
              </div>
            </div>
            <Button
              size="sm"
              onClick={() => setLocation("/device-setup")}
            >
              {isConnected ? "Manage" : "Setup"}
              <ChevronRight className="h-3.5 w-3.5 ml-1" />
            </Button>
          </div>
        </div>

        {/* Supported devices */}
        <div className="rounded-lg bg-muted/20 p-3">
          <p className="text-xs font-medium text-muted-foreground mb-2">Supported devices</p>
          <div className="flex flex-wrap gap-2">
            {["Muse 2", "Muse S", "Synthetic (Demo)"].map((device) => (
              <Badge key={device} variant="secondary" className="text-xs">
                {device}
              </Badge>
            ))}
          </div>
          <p className="text-[10px] text-muted-foreground/60 mt-2">
            More BCI devices coming soon. EEG headbands stream brain wave data at 256 Hz for real-time emotion, focus, and sleep analysis.
          </p>
        </div>
      </div>
    </Card>
  );
}

/* ── Connect Wearables Section ──────────────────────────────── */

const WEARABLE_PROVIDERS = [
  {
    id: "oura",
    name: "Oura Ring",
    description: "Readiness, sleep, activity, heart rate",
    color: "text-indigo-400",
    bgColor: "bg-indigo-400/10",
    borderColor: "border-indigo-400/30",
  },
  {
    id: "whoop",
    name: "WHOOP",
    description: "Recovery, strain, sleep, HRV",
    color: "text-yellow-500",
    bgColor: "bg-yellow-500/10",
    borderColor: "border-yellow-500/30",
  },
  {
    id: "garmin",
    name: "Garmin",
    description: "Steps, stress, body battery, workouts",
    color: "text-cyan-400",
    bgColor: "bg-cyan-400/10",
    borderColor: "border-cyan-400/30",
  },
];

function ConnectWearablesSection() {
  const { toast } = useToast();
  const [connectingProvider, setConnectingProvider] = useState<string | null>(null);
  const [syncingProvider, setSyncingProvider] = useState<string | null>(null);
  const [disconnectingProvider, setDisconnectingProvider] = useState<string | null>(null);

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

  const handleConnect = async (provider: string) => {
    setConnectingProvider(provider);
    try {
      const token = localStorage.getItem("auth_token");
      const res = await fetch(resolveUrl(`/api/devices/connect/${provider}`), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error((err as Record<string, string>).error ?? "Failed to start connection");
      }
      const { authUrl } = await res.json();
      const popup = window.open(authUrl, `connect-${provider}`, "width=600,height=700,scrollbars=yes");
      if (!popup) {
        toast({
          title: "Popup blocked",
          description: "Allow popups for this site and try again.",
          variant: "destructive",
        });
        setConnectingProvider(null);
      }
    } catch (error) {
      toast({
        title: "Connection failed",
        description: String(error),
        variant: "destructive",
      });
      setConnectingProvider(null);
    }
  };

  const handleSync = async (provider: string) => {
    setSyncingProvider(provider);
    try {
      const token = localStorage.getItem("auth_token");
      const res = await fetch(resolveUrl(`/api/devices/sync/${provider}`), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error((err as Record<string, string>).error ?? "Sync failed");
      }
      const data = await res.json();
      toast({
        title: "Sync complete",
        description: `Synced ${data.synced} health samples from ${provider}.`,
      });
      refetchDevices();
    } catch (error) {
      toast({
        title: "Sync failed",
        description: String(error),
        variant: "destructive",
      });
    } finally {
      setSyncingProvider(null);
    }
  };

  const handleDisconnect = async (provider: string) => {
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
      toast({
        title: "Device disconnected",
        description: `${provider} has been disconnected.`,
      });
      refetchDevices();
    } catch (error) {
      toast({
        title: "Disconnect failed",
        description: String(error),
        variant: "destructive",
      });
    } finally {
      setDisconnectingProvider(null);
    }
  };

  return (
    <Card className="glass-card p-5">
      <SectionHeader
        icon={Watch}
        iconColor="bg-amber-500/10 text-amber-400"
        title="Connect Wearable"
        subtitle="Oura, WHOOP, Garmin"
      />

      <div className="space-y-3">
        {WEARABLE_PROVIDERS.map((wp) => {
          const connection = connectedDevices.find((d) => d.provider === wp.id);
          const isConnected = !!connection;
          const isConnecting = connectingProvider === wp.id;
          const isSyncing = syncingProvider === wp.id;
          const isDisconnecting = disconnectingProvider === wp.id;

          return (
            <div
              key={wp.id}
              className={`rounded-lg border p-4 ${
                isConnected ? wp.borderColor : "border-border/50"
              } ${isConnected ? wp.bgColor : "bg-card/50"}`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div
                    className={`h-10 w-10 rounded-full flex items-center justify-center ${wp.bgColor}`}
                  >
                    <Watch className={`h-5 w-5 ${wp.color}`} />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{wp.name}</span>
                      {isConnected && (
                        <Badge
                          variant="outline"
                          className="text-xs border-cyan-500/40 text-cyan-500"
                        >
                          Connected
                        </Badge>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground">{wp.description}</p>
                    {connection?.lastSyncAt && (
                      <p className="text-xs text-muted-foreground mt-0.5">
                        Last sync: {new Date(connection.lastSyncAt).toLocaleString()}
                      </p>
                    )}
                    {connection?.errorMessage && (
                      <p className="text-xs text-destructive mt-0.5">
                        {connection.errorMessage}
                      </p>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {isConnected ? (
                    <>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleSync(wp.id)}
                        disabled={isSyncing}
                        className="h-8"
                      >
                        {isSyncing ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <RefreshCw className="h-3.5 w-3.5" />
                        )}
                        <span className="ml-1.5 text-xs">
                          {isSyncing ? "..." : "Sync"}
                        </span>
                      </Button>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-8 border-destructive/30 text-destructive hover:bg-destructive/10"
                            disabled={isDisconnecting}
                          >
                            {isDisconnecting ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            ) : (
                              <Unlink className="h-3.5 w-3.5" />
                            )}
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent className="glass-card">
                          <AlertDialogHeader>
                            <AlertDialogTitle>Disconnect {wp.name}?</AlertDialogTitle>
                            <AlertDialogDescription>
                              This removes the connection. Previously synced data stays in your
                              profile. You can reconnect at any time.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>Cancel</AlertDialogCancel>
                            <AlertDialogAction
                              onClick={() => handleDisconnect(wp.id)}
                              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                            >
                              Disconnect
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </>
                  ) : (
                    <Button
                      size="sm"
                      onClick={() => handleConnect(wp.id)}
                      disabled={isConnecting}
                      className="h-8"
                    >
                      {isConnecting ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />
                      ) : (
                        <Link className="h-3.5 w-3.5 mr-1.5" />
                      )}
                      <span className="text-xs">
                        {isConnecting ? "..." : "Connect"}
                      </span>
                    </Button>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

/* ── Main Page ──────────────────────────────────────────────── */

export default function ConnectedAssets() {
  return (
    <main className="p-4 md:p-6 pb-24 space-y-6 max-w-3xl">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Link className="h-6 w-6 text-primary" />
        <div>
          <h1 className="text-xl font-semibold">Connected Assets</h1>
          <p className="text-xs text-muted-foreground">
            Manage your health, brain, and wearable connections
          </p>
        </div>
      </div>

      {/* Connect Health */}
      <ConnectHealthSection />

      {/* Connect BCI / EEG */}
      <ConnectBCISection />

      {/* Connect Wearables */}
      <ConnectWearablesSection />
    </main>
  );
}
