import { getParticipantId } from "@/lib/participant";
import { resolveUrl } from "@/lib/queryClient";
import { useState, useEffect, useCallback, useRef } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
import { AlertTriangle, Download, Apple, Smartphone, Upload, CheckCircle2, XCircle, Info, Server, Bell, BellOff, Cpu, Heart, Brain, LogOut, User, Shield } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import { useTheme } from "@/hooks/use-theme";
import { useLocation } from "wouter";
import { useDevice } from "@/hooks/use-device";
import { useAuth } from "@/hooks/use-auth";
import { useQuery } from "@tanstack/react-query";
const USER_ID = getParticipantId();
import { useToast } from "@/hooks/use-toast";
import { ingestHealthData, addBaselineFrame, getBaselineStatus, resetBaselineCalibration, getCalibrationStatus, getPersonalStatus, triggerPersonalFineTune } from "@/lib/ml-api";
import HealthSyncDashboard from "@/components/health-sync-dashboard";
import { useMutation, useQueryClient } from "@tanstack/react-query";

interface SettingsState {
  chartAnimations: boolean;
  neuralFlowEffects: boolean;
  healthAlerts: boolean;
  localProcessing: boolean;
  dataEncryption: boolean;
  anonymousAnalytics: boolean;
}

const defaultSettings: SettingsState = {
  chartAnimations: true,
  neuralFlowEffects: true,
  healthAlerts: true,
  localProcessing: true,
  dataEncryption: true,
  anonymousAnalytics: false,
};

interface HealthConnectionStatus {
  apple_health: boolean;
  google_fit: boolean;
}

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const userId = USER_ID;
  const { toast } = useToast();
  const [, setLocation] = useLocation();
  const { state: deviceState, deviceStatus } = useDevice();
  const [settings, setSettings] = useState<SettingsState>(defaultSettings);
  const [healthStatus, setHealthStatus] = useState<HealthConnectionStatus>({
    apple_health: false,
    google_fit: false,
  });
  const [isConnectingApple, setIsConnectingApple] = useState(false);
  const [isConnectingGoogle, setIsConnectingGoogle] = useState(false);
  const [uploading, setUploading] = useState<string | null>(null);
  const [exportingHealthkit, setExportingHealthkit] = useState(false);
  const [seedingDemo, setSeedingDemo] = useState(false);

  const appleFileRef = useRef<HTMLInputElement>(null);
  const googleFileRef = useRef<HTMLInputElement>(null);

  // Load settings from API on mount
  useEffect(() => {
    async function loadSettings() {
      try {
        const response = await fetch(resolveUrl(`/api/settings/${userId}`));
        if (response.ok) {
          const data = await response.json();
          setSettings((prev) => ({ ...prev, ...data }));
        }
      } catch (error) {
        console.error("Failed to load settings:", error);
      }
    }
    loadSettings();
  }, [userId]);

  // Fetch health connection status on mount
  const refetchHealthStatus = useCallback(async () => {
    try {
      const res = await fetch(resolveUrl("/api/health/status"));
      if (res.ok) {
        const data = await res.json();
        setHealthStatus((prev) => ({ ...prev, ...data }));
      }
    } catch {
      // ignore — not critical
    }
  }, []);

  useEffect(() => {
    refetchHealthStatus();
  }, [refetchHealthStatus]);

  const handleAppleHealthConnect = async () => {
    setIsConnectingApple(true);
    try {
      await fetch(resolveUrl("/api/health/connect"), { method: "POST" });
      refetchHealthStatus();
    } catch {
      // Silently fail — not critical
    } finally {
      setIsConnectingApple(false);
    }
  };

  const handleGoogleFitConnect = async () => {
    setIsConnectingGoogle(true);
    try {
      // Use the capacitor-health plugin (same as health-sync.ts Android path)
      const { Capacitor } = await import("@capacitor/core");
      const platform = Capacitor.getPlatform();

      if (platform === "android") {
        const { Health } = await import("capacitor-health");
        // Check if Health Connect is available
        const available = await Health.isHealthAvailable();
        if (!available.available) {
          toast({
            title: "Health Connect Not Available",
            description: "Please install Google Health Connect from the Play Store.",
            variant: "destructive",
          });
          return;
        }
        // Request permissions
        await Health.requestHealthPermissions({
          permissions: [
            "READ_STEPS",
            "READ_HEART_RATE",
            "READ_ACTIVE_CALORIES",
            "READ_WORKOUTS",
            "READ_MINDFULNESS",
          ],
        });
        setHealthStatus((prev) => ({ ...prev, google_fit: true }));
        toast({
          title: "Google Health Connect",
          description: "Connected successfully. Health data will sync automatically.",
        });
        // Also notify the server
        await fetch(resolveUrl("/api/health/connect"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source: "google_fit" }),
        }).catch(() => {});
      } else if (platform === "web") {
        toast({
          title: "Health Connect",
          description: "Google Health Connect is available on Android devices only. Use the mobile app to connect.",
        });
      } else {
        toast({
          title: "Health Connect",
          description: "Google Health Connect is available on Android only. Use Apple Health on iOS.",
        });
      }
    } catch (e) {
      toast({
        title: "Connection Failed",
        description: `Could not connect to Health Connect: ${String(e)}`,
        variant: "destructive",
      });
    } finally {
      setIsConnectingGoogle(false);
    }
  };

  // Save settings to API
  const saveSettings = useCallback(
    async (updated: SettingsState) => {
      try {
        await fetch(resolveUrl(`/api/settings/${userId}`), {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(updated),
        });
      } catch (error) {
        console.error("Failed to save settings:", error);
      }
    },
    [userId]
  );

  const updateSetting = <K extends keyof SettingsState>(
    key: K,
    value: SettingsState[K]
  ) => {
    const updated = { ...settings, [key]: value };
    setSettings(updated);
    saveSettings(updated);
  };

  const handleHealthUpload = async (source: "apple_health" | "google_fit", file: File) => {
    setUploading(source);
    try {
      const text = await file.text();
      let data: Record<string, unknown>;

      if (file.name.endsWith(".json")) {
        data = JSON.parse(text);
      } else {
        // XML or other — wrap raw text for backend parsing
        data = { raw_xml: text, filename: file.name };
      }

      const result = await ingestHealthData(userId, source, data);
      setHealthStatus((prev) => ({ ...prev, [source]: true }));
      toast({
        title: "Health Data Imported",
        description: `Successfully imported ${result.stored} samples (${result.metrics.join(", ")}).`,
      });
    } catch (error) {
      console.error("Health upload failed:", error);
      toast({
        title: "Import Failed",
        description: "Could not parse the health export file. Please try again.",
        variant: "destructive",
      });
    } finally {
      setUploading(null);
    }
  };

  const handleHealthkitExport = async () => {
    setExportingHealthkit(true);
    try {
      const res = await fetch(resolveUrl(`/api/ml/health/export-to-healthkit/${userId}`), { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `healthkit_export_${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast({
        title: "HealthKit export ready",
        description: `${data.count} samples exported. Import with Shortcuts or Health app on iOS.`,
      });
    } catch (err) {
      toast({ title: "Export failed", description: String(err), variant: "destructive" });
    } finally {
      setExportingHealthkit(false);
    }
  };

  const handleDataExport = async () => {
    try {
      const response = await fetch(resolveUrl(`/api/export/${userId}`));
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "neural_data.csv";
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Export failed:", error);
    }
  };

  const handleDreamExport = async () => {
    try {
      const response = await fetch(resolveUrl(`/api/export/${userId}?type=dreams`));
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "dream_analysis.csv";
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Dream export failed:", error);
    }
  };

  const handleSeedDemo = async () => {
    setSeedingDemo(true);
    try {
      const token = localStorage.getItem("auth_token");
      const res = await fetch(resolveUrl("/api/seed-demo"), {
        method: "POST",
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      const data = await res.json();
      if (res.ok) {
        toast({ title: "Demo data loaded", description: data.message });
      } else {
        toast({ title: "Error", description: data.error || "Failed to seed", variant: "destructive" });
      }
    } catch {
      toast({ title: "Error", description: "Network error", variant: "destructive" });
    } finally {
      setSeedingDemo(false);
    }
  };

  const handleClearAllData = async () => {
    try {
      await fetch(resolveUrl(`/api/settings/${userId}/data`), {
        method: "DELETE",
      });
      setSettings(defaultSettings);
      setHealthStatus({ apple_health: false, google_fit: false });
      toast({
        title: "Data Cleared",
        description: "All your data has been permanently removed.",
      });
    } catch (error) {
      console.error("Failed to clear data:", error);
      toast({
        title: "Error",
        description: "Failed to clear data. Please try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <main className="p-4 md:p-6 pb-24 space-y-6">
      {/* Connected Devices */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Cpu className="h-5 w-5" />
            Connected Devices
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">
                {deviceStatus?.device_type ?? "No device connected"}
              </p>
              <p className="text-xs text-muted-foreground capitalize">
                Status: {deviceState}
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setLocation("/device-setup")}
            >
              {deviceState === "disconnected" ? "Connect" : "Manage"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Health Integrations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Heart className="h-5 w-5" />
            Health Integrations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {/* Apple Health */}
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Apple Health</p>
                <p className="text-xs text-muted-foreground">
                  {healthStatus.apple_health ? "Connected" : "Not connected"}
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleAppleHealthConnect}
                disabled={isConnectingApple}
              >
                {isConnectingApple
                  ? "Connecting..."
                  : healthStatus.apple_health
                  ? "Disconnect"
                  : "Connect"}
              </Button>
            </div>
            {/* Google Health Connect */}
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Google Health Connect</p>
                <p className="text-xs text-muted-foreground">
                  {healthStatus.google_fit ? "Connected" : "Not connected"}
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleGoogleFitConnect}
                disabled={isConnectingGoogle}
              >
                {isConnectingGoogle
                  ? "Connecting..."
                  : healthStatus.google_fit
                  ? "Disconnect"
                  : "Connect"}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Health Sync Status Dashboard */}
      <HealthSyncDashboard />

      {/* ML Backend URL */}
      <MLBackendCard />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Health Connections */}
        <Card className="glass-card p-6 rounded-xl">
          <h3 className="text-lg font-semibold mb-6">
            Health Connections
          </h3>
          <div className="space-y-5">
            {/* Apple Health */}
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-destructive/10">
                  <Apple className="h-5 w-5 text-destructive" />
                </div>
                <div>
                  <p className="text-sm font-medium">Apple Health</p>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    {healthStatus.apple_health ? (
                      <>
                        <CheckCircle2 className="h-3 w-3 text-success" />
                        <span className="text-xs text-success">Connected</span>
                      </>
                    ) : (
                      <>
                        <XCircle className="h-3 w-3 text-muted-foreground" />
                        <span className="text-xs text-muted-foreground">Not Connected</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
              <div>
                <Input
                  ref={appleFileRef}
                  type="file"
                  accept=".xml,.zip,.json"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleHealthUpload("apple_health", file);
                  }}
                />
                <Button
                  variant="outline"
                  size="sm"
                  disabled={uploading === "apple_health"}
                  onClick={() => appleFileRef.current?.click()}
                >
                  <Upload className="h-3.5 w-3.5 mr-1.5" />
                  {uploading === "apple_health" ? "Uploading..." : "Upload Export"}
                </Button>
              </div>
            </div>

            {/* Google Health Connect */}
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-primary/10">
                  <Smartphone className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <p className="text-sm font-medium">Google Health Connect</p>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    {healthStatus.google_fit ? (
                      <>
                        <CheckCircle2 className="h-3 w-3 text-success" />
                        <span className="text-xs text-success">Connected</span>
                      </>
                    ) : (
                      <>
                        <XCircle className="h-3 w-3 text-muted-foreground" />
                        <span className="text-xs text-muted-foreground">Not Connected</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
              <div>
                <Input
                  ref={googleFileRef}
                  type="file"
                  accept=".xml,.zip,.json"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleHealthUpload("google_fit", file);
                  }}
                />
                <Button
                  variant="outline"
                  size="sm"
                  disabled={uploading === "google_fit"}
                  onClick={() => googleFileRef.current?.click()}
                >
                  <Upload className="h-3.5 w-3.5 mr-1.5" />
                  {uploading === "google_fit" ? "Uploading..." : "Upload Export"}
                </Button>
              </div>
            </div>

            {/* Apple HealthKit export */}
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-start gap-2">
                <Info className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
                <p className="text-xs text-muted-foreground">
                  Export your brain sessions as HealthKit-formatted JSON. Import on iOS via Shortcuts or the Health app.
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                disabled={exportingHealthkit}
                onClick={handleHealthkitExport}
              >
                <Download className="h-3.5 w-3.5 mr-1.5" />
                {exportingHealthkit ? "Exporting..." : "Export to HealthKit"}
              </Button>
            </div>
          </div>
        </Card>

        {/* Interface Settings */}
        <Card className="glass-card p-6 rounded-xl">
          <h3 className="text-lg font-semibold mb-6">
            Interface Settings
          </h3>
          <div className="space-y-6">
            <div>
              <Label className="text-sm font-medium text-foreground/80 mb-2">
                Theme
              </Label>
              <div className="flex space-x-3">
                <Button
                  variant={theme === "dark" ? "default" : "outline"}
                  className={`flex-1 ${theme === "dark" ? "bg-primary/20 border-primary/30 text-primary" : ""}`}
                  onClick={() => setTheme("dark")}
                  data-testid="button-theme-dark"
                >
                  Dark Mode
                </Button>
                <Button
                  variant={theme === "light" ? "default" : "outline"}
                  className={`flex-1 ${theme === "light" ? "bg-primary/20 border-primary/30 text-primary" : ""}`}
                  onClick={() => setTheme("light")}
                  data-testid="button-theme-light"
                >
                  Light Mode
                </Button>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label className="text-sm">Chart Animations</Label>
                <Switch
                  checked={settings.chartAnimations}
                  onCheckedChange={(checked) =>
                    updateSetting("chartAnimations", checked)
                  }
                  data-testid="switch-chart-animations"
                />
              </div>
              <div className="flex items-center justify-between">
                <Label className="text-sm">Neural Flow Effects</Label>
                <Switch
                  checked={settings.neuralFlowEffects}
                  onCheckedChange={(checked) =>
                    updateSetting("neuralFlowEffects", checked)
                  }
                  data-testid="switch-neural-effects"
                />
              </div>
              <div className="flex items-center justify-between">
                <Label className="text-sm">Health Alerts</Label>
                <Switch
                  checked={settings.healthAlerts}
                  onCheckedChange={(checked) =>
                    updateSetting("healthAlerts", checked)
                  }
                  data-testid="switch-health-alerts"
                />
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Data Export & Privacy */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="glass-card p-6 rounded-xl">
          <h3 className="text-lg font-semibold mb-6">
            Data Export
          </h3>
          <div className="space-y-4">
            <Button
              onClick={handleDataExport}
              className="w-full bg-success/10 border border-success/30 text-success hover:bg-success/20"
              data-testid="button-export-health-data"
            >
              <Download className="mr-2 h-4 w-4" />
              Export Health Data (CSV)
            </Button>
            <Button
              variant="outline"
              className="w-full bg-secondary/10 border border-secondary/30 text-secondary hover:bg-secondary/20"
              onClick={handleDreamExport}
              data-testid="button-export-dream-analysis"
            >
              <Download className="mr-2 h-4 w-4" />
              Export Dream Analysis
            </Button>
          </div>
          <div className="mt-6 text-xs text-foreground/50">
            <p>Last export: {new Date().toLocaleDateString()}</p>
            <p>Data retention: 90 days</p>
          </div>
        </Card>

        <Card className="glass-card p-6 rounded-xl">
          <h3 className="text-lg font-semibold mb-6">
            Privacy & Security
          </h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label className="text-sm">Local Data Processing</Label>
              <Switch
                checked={settings.localProcessing}
                onCheckedChange={(checked) =>
                  updateSetting("localProcessing", checked)
                }
                data-testid="switch-local-processing"
              />
            </div>
            <div className="flex items-center justify-between">
              <Label className="text-sm">Data Encryption</Label>
              <Switch
                checked={settings.dataEncryption}
                onCheckedChange={(checked) =>
                  updateSetting("dataEncryption", checked)
                }
                data-testid="switch-data-encryption"
              />
            </div>
            <div className="flex items-center justify-between">
              <Label className="text-sm">Anonymous Analytics</Label>
              <Switch
                checked={settings.anonymousAnalytics}
                onCheckedChange={(checked) =>
                  updateSetting("anonymousAnalytics", checked)
                }
                data-testid="switch-anonymous-analytics"
              />
            </div>

            <Button
              variant="outline"
              className="w-full mt-6 border-primary/30 text-primary hover:bg-primary/10"
              onClick={handleSeedDemo}
              disabled={seedingDemo}
            >
              <Download className="mr-2 h-4 w-4" />
              {seedingDemo ? "Loading demo data…" : "Load demo data"}
            </Button>

            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  variant="destructive"
                  className="w-full mt-2 bg-destructive/10 border border-destructive/30 text-destructive hover:bg-destructive/20"
                  data-testid="button-clear-data"
                >
                  <AlertTriangle className="mr-2 h-4 w-4" />
                  Clear All Data
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent className="glass-card border-destructive/30">
                <AlertDialogHeader>
                  <AlertDialogTitle className="font-medium text-destructive">
                    Clear All Data
                  </AlertDialogTitle>
                  <AlertDialogDescription>
                    This action cannot be undone. This will permanently delete
                    all your neural data, dream analysis records, health
                    metrics, and settings. Are you sure you want to continue?
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={handleClearAllData}
                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  >
                    Yes, Clear Everything
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </Card>
      </div>

      {/* Baseline Calibration */}
      <PersonalizationCard userId={userId} />

      {/* Baseline Calibration */}
      <BaselineCalibrationCard userId={userId} />

      {/* Personal Model Personalization */}
      <PersonalModelCard userId={userId} />

      {/* Notifications */}
      <NotificationsCard userId={userId} />

      {/* Export Brain Data */}
      <ExportBrainDataCard userId={userId} />

      {/* Data & Privacy (GDPR) */}
      <DataPrivacyCard userId={userId} />

      {/* Account & Sign Out */}
      <AccountCard />

    </main>
  );
}

/* ── Baseline Calibration Card ───────────────────────────────── */

const CALIBRATION_TOTAL_FRAMES = 30;

function PersonalizationCard({ userId }: { userId: string }) {
  const { data: status } = useQuery({
    queryKey: ["personalization-status", userId],
    queryFn: () => getCalibrationStatus(userId),
    staleTime: 30_000,
    retry: false,
  });

  const progress = status?.personalization_progress_pct ?? 0;
  const active = status?.personal_model_active ?? false;
  const threshold = status?.activation_threshold_sessions ?? 5;
  const sessions = status?.total_sessions ?? 0;
  const improvement = status?.accuracy_improvement_pct ?? 0;
  const blend = status?.personal_blend_weight_pct ?? 70;
  const priors = status?.feature_priors;

  return (
    <Card className="glass-card p-6 rounded-xl">
      <div className="flex items-center justify-between gap-3 mb-3">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Brain className="h-4 w-4 text-primary" />
            Model Personalization
          </h3>
          <p className="text-sm text-muted-foreground mt-1">
            Your emotion model becomes personal after {threshold} corrected sessions.
          </p>
        </div>
        <Badge className={active ? "bg-green-500/20 text-green-400 border-green-500/30" : "bg-yellow-500/20 text-yellow-400 border-yellow-500/30"}>
          {active ? `${progress}% Personalized` : `${progress}% Ready`}
        </Badge>
      </div>

      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-xs text-muted-foreground mb-1">
            <span>{sessions} / {threshold} corrected sessions</span>
            <span>{progress}%</span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="rounded-lg bg-muted/40 p-3">
            <p className="text-xs text-muted-foreground">Corrected sessions</p>
            <p className="text-lg font-semibold">{sessions}</p>
          </div>
          <div className="rounded-lg bg-muted/40 p-3">
            <p className="text-xs text-muted-foreground">Labeled epochs</p>
            <p className="text-lg font-semibold">{status?.total_labeled_epochs ?? 0}</p>
          </div>
          <div className="rounded-lg bg-muted/40 p-3">
            <p className="text-xs text-muted-foreground">Accuracy lift</p>
            <p className="text-lg font-semibold">{improvement}%</p>
          </div>
          <div className="rounded-lg bg-muted/40 p-3">
            <p className="text-xs text-muted-foreground">Blend</p>
            <p className="text-lg font-semibold">{blend}/{100 - blend}</p>
          </div>
        </div>

        {priors && (
          <div className="rounded-lg border border-border/40 bg-muted/20 p-3">
            <p className="text-xs text-muted-foreground mb-2">Stored personal priors</p>
            <div className="grid grid-cols-3 gap-3 text-sm">
              <div>
                <p className="text-muted-foreground text-xs">Alpha</p>
                <p className="font-mono">{priors.alpha_mean.toFixed(3)}</p>
              </div>
              <div>
                <p className="text-muted-foreground text-xs">Beta</p>
                <p className="font-mono">{priors.beta_mean.toFixed(3)}</p>
              </div>
              <div>
                <p className="text-muted-foreground text-xs">Theta</p>
                <p className="font-mono">{priors.theta_mean.toFixed(3)}</p>
              </div>
            </div>
          </div>
        )}

        <p className="text-xs text-muted-foreground">
          {status?.message ?? "Correct labels in Emotion Lab to start training your personal model."}
        </p>
      </div>
    </Card>
  );
}

function BaselineCalibrationCard({ userId }: { userId: string }) {
  const { toast } = useToast();
  const { latestFrame, state: deviceState } = useDevice();

  const [frames, setFrames] = useState(0);
  const [isReady, setIsReady] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fetch calibration status on mount
  useEffect(() => {
    getBaselineStatus(userId)
      .then((status) => {
        setFrames(Math.min(status.n_frames, CALIBRATION_TOTAL_FRAMES));
        setIsReady(status.ready);
      })
      .catch(() => {
        // Backend unreachable — stay at defaults
      });
  }, [userId]);

  function stopCalibration() {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsCalibrating(false);
  }

  async function startCalibration() {
    if (!latestFrame?.signals) {
      toast({
        title: "No EEG signal",
        description: "Connect an EEG device and start streaming before calibrating.",
        variant: "destructive",
      });
      return;
    }

    setIsCalibrating(true);
    setFrames(0);
    setIsReady(false);

    let collected = 0;

    intervalRef.current = setInterval(async () => {
      if (!latestFrame?.signals) {
        stopCalibration();
        toast({
          title: "Calibration stopped",
          description: "EEG signal lost. Reconnect your device and try again.",
          variant: "destructive",
        });
        return;
      }

      try {
        const result = await addBaselineFrame(latestFrame.signals, userId, 256);
        collected = Math.min(result.n_frames, CALIBRATION_TOTAL_FRAMES);
        setFrames(collected);

        if (result.ready) {
          setIsReady(true);
          stopCalibration();
          toast({
            title: "Calibration complete",
            description: "Baseline recorded. EEG features will now be normalized against your resting state.",
          });
        } else if (collected >= CALIBRATION_TOTAL_FRAMES) {
          stopCalibration();
        }
      } catch {
        stopCalibration();
        toast({
          title: "Calibration error",
          description: "Could not send frame to ML backend. Is it running?",
          variant: "destructive",
        });
      }
    }, 1000);
  }

  async function handleReset() {
    stopCalibration();
    try {
      await resetBaselineCalibration(userId);
      setFrames(0);
      setIsReady(false);
      toast({ title: "Calibration reset", description: "Baseline cleared. Run calibration again before a new session." });
    } catch {
      toast({ title: "Reset failed", description: "Could not reach ML backend.", variant: "destructive" });
    }
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const deviceStreaming = deviceState === "streaming";
  const progressPct = Math.round((frames / CALIBRATION_TOTAL_FRAMES) * 100);

  let statusBadge: JSX.Element;
  if (isReady) {
    statusBadge = <Badge className="bg-green-500/20 text-green-400 border-green-500/30">Ready</Badge>;
  } else if (isCalibrating) {
    statusBadge = <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">Calibrating…</Badge>;
  } else {
    statusBadge = <Badge className="bg-red-500/20 text-red-400 border-red-500/30">Not calibrated</Badge>;
  }

  return (
    <Card className="glass-card p-6 rounded-xl">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Brain className="h-4 w-4 text-primary" />
          Optional EEG Calibration
        </h3>
        {statusBadge}
      </div>
      <p className="text-sm text-muted-foreground mb-4">
        If you use Muse 2, record 30 seconds of resting-state EEG so the ML backend can normalize live headset signals.
        This only affects the optional EEG layer.
      </p>

      <div className="flex items-start gap-2 p-3 rounded-lg bg-muted/40 border border-border/30 text-xs text-muted-foreground mb-4">
        <Info className="h-4 w-4 shrink-0 mt-0.5 text-primary/60" />
        <span>
          Sit quietly with eyes closed for 30 seconds. Minimize jaw movement.
          Keep your Muse headband firmly in place before starting if you are using EEG.
        </span>
      </div>

      {frames > 0 && (
        <div className="mb-4">
          <div className="flex justify-between text-xs text-muted-foreground mb-1">
            <span>{frames} / {CALIBRATION_TOTAL_FRAMES} frames</span>
            <span>{progressPct}%</span>
          </div>
          <Progress value={progressPct} className="h-2" />
        </div>
      )}

      <div className="flex gap-2">
        {!isCalibrating ? (
          <Button
            size="sm"
            onClick={startCalibration}
            disabled={!deviceStreaming}
            className="bg-primary/10 border border-primary/30 text-primary hover:bg-primary/20"
          >
            {isReady ? "Recalibrate" : "Start Calibration"}
          </Button>
        ) : (
          <Button size="sm" variant="outline" onClick={stopCalibration}>
            Stop
          </Button>
        )}
        {(frames > 0 || isReady) && !isCalibrating && (
          <Button size="sm" variant="outline" onClick={handleReset}>
            Reset
          </Button>
        )}
      </div>

      {!deviceStreaming && (
        <p className="text-xs text-muted-foreground mt-2">
          EEG calibration only appears when a headset is connected and streaming.
        </p>
      )}
    </Card>
  );
}

/* ── ML Backend URL Card ─────────────────────────────────────── */

function MLBackendCard() {
  const { toast } = useToast();
  const [url, setUrl] = useState(() => {
    try { return localStorage.getItem("ml_backend_url") || ""; } catch { return ""; }
  });
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<"ok" | "fail" | null>(null);

  // Auto-fill URL from query string — set by ml/start.sh
  useEffect(() => {
    try {
      const params = new URLSearchParams(window.location.search);
      const mlBackend = params.get("ml_backend");
      if (mlBackend) {
        const trimmed = mlBackend.trim().replace(/\/$/, "");
        setUrl(trimmed);
        localStorage.setItem("ml_backend_url", trimmed);
        // Remove the query param from the URL bar without a reload
        const clean = window.location.pathname;
        window.history.replaceState({}, "", clean);
        toast({
          title: "ML Backend URL saved",
          description: `Auto-filled from startup script: ${trimmed}`,
        });
      }
    } catch { /* no-op */ }
  }, [toast]);

  function save() {
    try {
      const trimmed = url.trim().replace(/\/$/, "");
      if (trimmed) {
        localStorage.setItem("ml_backend_url", trimmed);
      } else {
        localStorage.removeItem("ml_backend_url");
      }
      toast({ title: "Saved", description: "ML backend URL updated. Reconnect your device." });
      setTestResult(null);
    } catch {
      toast({ title: "Error", description: "Could not save to localStorage.", variant: "destructive" });
    }
  }

  async function testConnection() {
    const target = url.trim().replace(/\/$/, "") || "http://localhost:8080";
    setTesting(true);
    setTestResult(null);
    try {
      const res = await fetch(`${target}/health`, { signal: AbortSignal.timeout(5000) });
      setTestResult(res.ok ? "ok" : "fail");
    } catch {
      setTestResult("fail");
    } finally {
      setTesting(false);
    }
  }

  const ML_DEFAULT = import.meta.env.VITE_ML_API_URL || "http://localhost:8080";

  function resetToDefault() {
    try { localStorage.removeItem("ml_backend_url"); } catch { /* ok */ }
    setUrl("");
    setTestResult(null);
    toast({ title: "Reset", description: "Using the default ML backend." });
  }

  return (
    <Card className="glass-card p-6 rounded-xl">
      <h3 className="text-lg font-semibold mb-1 flex items-center gap-2">
        <Server className="h-4 w-4 text-primary" />
        ML Backend
      </h3>
      <p className="text-sm text-muted-foreground mb-4">
        By default the app uses the deployed ML backend on Render. Leave the field below
        blank to use it. For local development you can override with your own server
        (expose via <strong>ngrok</strong> if needed).
      </p>

      <div className="mb-4 p-3 rounded-xl bg-muted/30 border border-border/30 text-xs text-muted-foreground font-mono">
        Default: {ML_DEFAULT}
      </div>

      <div className="flex gap-2">
        <Input
          value={url}
          onChange={(e) => { setUrl(e.target.value); setTestResult(null); }}
          placeholder={`Leave blank to use ${ML_DEFAULT}`}
          className="flex-1 font-mono text-sm"
        />
        <Button variant="outline" onClick={testConnection} disabled={testing}>
          {testing ? "Testing…" : "Test"}
        </Button>
        <Button onClick={save}>Save</Button>
      </div>
      {url && (
        <button
          onClick={resetToDefault}
          className="mt-1.5 text-xs text-muted-foreground/60 hover:text-muted-foreground underline"
        >
          Reset to Render default
        </button>
      )}

      {testResult === "ok" && (
        <p className="text-xs text-success mt-2 flex items-center gap-1">
          <CheckCircle2 className="h-3 w-3" /> Backend reachable — save and reconnect your device.
        </p>
      )}
      {testResult === "fail" && (
        <p className="text-xs text-destructive mt-2 flex items-center gap-1">
          <XCircle className="h-3 w-3" /> Could not reach backend. Check the URL and that the server is running.
        </p>
      )}
      <p className="text-[11px] text-muted-foreground mt-3">
        Current: <span className="font-mono">{url.trim() || "http://localhost:8080 (default)"}</span>
      </p>
    </Card>
  );
}

/* ── Notifications Card ──────────────────────────────────────── */

function NotificationsCard({ userId }: { userId: string }) {
  const { toast } = useToast();
  const [permission, setPermission] = useState<NotificationPermission>(
    typeof Notification !== "undefined" ? Notification.permission : "default"
  );
  const [subscribing, setSubscribing] = useState(false);

  const supported =
    typeof window !== "undefined" &&
    "serviceWorker" in navigator &&
    "Notification" in window;

  async function enableNotifications() {
    if (!supported) return;
    setSubscribing(true);
    try {
      // Request notification permission
      const perm = await Notification.requestPermission();
      setPermission(perm);
      if (perm !== "granted") {
        toast({ title: "Permission denied", description: "Enable notifications in your browser settings.", variant: "destructive" });
        return;
      }

      // Register the service worker
      const reg = await navigator.serviceWorker.register("/sw.js");
      await navigator.serviceWorker.ready;

      // Try to get VAPID key from server; gracefully handle missing key
      let sub: PushSubscription | null = null;
      try {
        const vapidRes = await fetch(resolveUrl("/api/notifications/vapid-public-key"));
        if (vapidRes.ok) {
          const { publicKey } = await vapidRes.json();
          sub = await reg.pushManager.subscribe({
            userVisibleOnly: true,
            applicationServerKey: publicKey,
          });
        }
      } catch {
        // VAPID not configured — SW is registered, local notifications work
      }

      if (sub) {
        const { endpoint, keys } = sub.toJSON() as { endpoint: string; keys: Record<string, string> };
        await fetch(resolveUrl("/api/notifications/subscribe"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ userId, endpoint, keys }),
        });
        toast({ title: "Notifications enabled", description: "You'll get a morning brain report reminder at 8 am." });
      } else {
        // SW registered but no VAPID — in-browser only
        toast({ title: "Notifications ready", description: "Service worker registered. Push delivery requires backend VAPID setup." });
      }
    } catch (err) {
      toast({ title: "Could not enable notifications", description: String(err), variant: "destructive" });
    } finally {
      setSubscribing(false);
    }
  }

  async function disableNotifications() {
    if (!supported) return;
    try {
      const reg = await navigator.serviceWorker.getRegistration("/sw.js");
      if (reg) {
        const sub = await reg.pushManager.getSubscription();
        if (sub) await sub.unsubscribe();
      }
      setPermission("default");
      toast({ title: "Notifications disabled" });
    } catch {
      toast({ title: "Could not disable", variant: "destructive" });
    }
  }

  return (
    <Card className="glass-card p-6 rounded-xl">
      <h3 className="text-lg font-semibold mb-1 flex items-center gap-2">
        <Bell className="h-4 w-4 text-primary" />
        Morning Reminders
      </h3>
      <p className="text-sm text-muted-foreground mb-4">
        Get a daily push notification at 8 am with your Brain Report — sleep quality, focus forecast, and top recommended action.
      </p>

      {!supported && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-muted/50 border border-border text-xs text-muted-foreground">
          <Info className="h-4 w-4 shrink-0 mt-0.5" />
          <span>Push notifications are not supported in this browser.</span>
        </div>
      )}

      {supported && permission === "denied" && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-xs text-destructive">
          <BellOff className="h-4 w-4 shrink-0 mt-0.5" />
          <span>Notifications are blocked. Enable them in your browser's site settings, then reload.</span>
        </div>
      )}

      {supported && permission !== "denied" && (
        <div className="flex items-center justify-between">
          <div>
            {permission === "granted" ? (
              <div className="flex items-center gap-1.5">
                <CheckCircle2 className="h-3.5 w-3.5 text-success" />
                <span className="text-sm text-success">Enabled</span>
              </div>
            ) : (
              <span className="text-sm text-muted-foreground">Not enabled</span>
            )}
          </div>
          {permission === "granted" ? (
            <Button variant="outline" size="sm" onClick={disableNotifications}>
              <BellOff className="h-3.5 w-3.5 mr-1.5" />
              Disable
            </Button>
          ) : (
            <Button size="sm" onClick={enableNotifications} disabled={subscribing}>
              <Bell className="h-3.5 w-3.5 mr-1.5" />
              {subscribing ? "Enabling…" : "Enable notifications"}
            </Button>
          )}
        </div>
      )}
    </Card>
  );
}

/* ── Export Brain Data Card ──────────────────────────────────── */
const EXPORT_METRICS = [
  { id: "focus_index",    label: "Focus"       },
  { id: "stress_index",   label: "Stress"      },
  { id: "relaxation_idx", label: "Relaxation"  },
  { id: "flow_score",     label: "Flow"        },
  { id: "valence",        label: "Valence"     },
  { id: "alpha",          label: "Alpha band"  },
  { id: "beta",           label: "Beta band"   },
  { id: "theta",          label: "Theta band"  },
];

function ExportBrainDataCard({ userId }: { userId: string }) {
  const { toast } = useToast();
  const today = new Date().toISOString().slice(0, 10);
  const weekAgo = new Date(Date.now() - 7 * 86400_000).toISOString().slice(0, 10);

  const [fromDate, setFromDate] = useState(weekAgo);
  const [toDate,   setToDate]   = useState(today);
  const [format,   setFormat]   = useState<"csv" | "json">("csv");
  const [selected, setSelected] = useState<Set<string>>(
    new Set(["focus_index", "stress_index", "relaxation_idx", "flow_score", "valence"])
  );
  const [exporting, setExporting] = useState(false);

  function toggleMetric(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  async function handleExport() {
    if (selected.size === 0) {
      toast({ title: "Select at least one metric", variant: "destructive" });
      return;
    }
    setExporting(true);
    try {
      const fromTs = Math.floor(new Date(fromDate).getTime() / 1000);
      const toTs   = Math.floor(new Date(toDate + "T23:59:59").getTime() / 1000);
      const url = new URL("/api/ml/brain/export", window.location.origin);
      url.searchParams.set("user_id", userId);
      url.searchParams.set("from_ts", String(fromTs));
      url.searchParams.set("to_ts",   String(toTs));
      url.searchParams.set("format",  format);
      url.searchParams.set("metrics", Array.from(selected).join(","));

      const res = await fetch(url.toString());
      if (!res.ok) throw new Error(await res.text());

      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = `brain_export_${fromDate}_${toDate}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(objectUrl);

      toast({ title: "Export complete", description: `${format.toUpperCase()} file downloaded.` });
    } catch (err) {
      toast({
        title: "Export failed",
        description: String(err),
        variant: "destructive",
      });
    } finally {
      setExporting(false);
    }
  }

  return (
    <Card className="glass-card p-6 rounded-xl">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Download className="h-4 w-4 text-primary" />
        Export Brain Data
      </h3>

      {/* Date range */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <Label className="text-xs text-muted-foreground mb-1 block">From</Label>
          <Input
            type="date"
            value={fromDate}
            max={toDate}
            onChange={(e) => setFromDate(e.target.value)}
          />
        </div>
        <div>
          <Label className="text-xs text-muted-foreground mb-1 block">To</Label>
          <Input
            type="date"
            value={toDate}
            min={fromDate}
            max={today}
            onChange={(e) => setToDate(e.target.value)}
          />
        </div>
      </div>

      {/* Metric checkboxes */}
      <div className="mb-4">
        <Label className="text-xs text-muted-foreground mb-2 block">Metrics</Label>
        <div className="grid grid-cols-2 gap-2">
          {EXPORT_METRICS.map((m) => (
            <div key={m.id} className="flex items-center gap-2">
              <Checkbox
                id={`metric-${m.id}`}
                checked={selected.has(m.id)}
                onCheckedChange={() => toggleMetric(m.id)}
              />
              <Label htmlFor={`metric-${m.id}`} className="text-xs cursor-pointer">
                {m.label}
              </Label>
            </div>
          ))}
        </div>
      </div>

      {/* Format */}
      <div className="mb-4">
        <Label className="text-xs text-muted-foreground mb-1 block">Format</Label>
        <Select value={format} onValueChange={(v) => setFormat(v as "csv" | "json")}>
          <SelectTrigger className="h-8 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="csv">CSV</SelectItem>
            <SelectItem value="json">JSON</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <Button
        onClick={handleExport}
        disabled={exporting}
        className="w-full bg-primary/10 border border-primary/30 text-primary hover:bg-primary/20"
      >
        <Download className="mr-2 h-4 w-4" />
        {exporting ? "Exporting…" : "Export Brain Data"}
      </Button>

      <p className="text-[10px] text-muted-foreground mt-3">
        Exports raw 1Hz readings from TimescaleDB. Requires DATABASE_URL.
      </p>
    </Card>
  );
}

/* ── Account & Sign Out Card ────────────────────────────────────── */

function AccountCard() {
  const { user, logout } = useAuth();
  const [signingOut, setSigningOut] = useState(false);
  const [, setLocation] = useLocation();

  async function handleSignOut() {
    setSigningOut(true);
    try {
      await logout();
      setLocation("/auth");
    } catch {
      setSigningOut(false);
    }
  }

  const memberSince = user?.createdAt
    ? new Date(user.createdAt).toLocaleDateString(undefined, {
        year: "numeric",
        month: "long",
        day: "numeric",
      })
    : null;

  return (
    <Card className="glass-card p-6 rounded-xl">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <User className="h-4 w-4 text-primary" />
        Account
      </h3>

      {user && (
        <div className="space-y-2 mb-5">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Username</span>
            <span className="text-sm font-medium">{user.username}</span>
          </div>
          {user.email && (
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Email</span>
              <span className="text-sm font-medium">{user.email}</span>
            </div>
          )}
          {memberSince && (
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Member since</span>
              <span className="text-sm font-medium">{memberSince}</span>
            </div>
          )}
        </div>
      )}

      <Separator className="my-4" />

      <Button
        variant="destructive"
        className="w-full"
        onClick={handleSignOut}
        disabled={signingOut}
      >
        <LogOut className="mr-2 h-4 w-4" />
        {signingOut ? "Signing out..." : "Sign Out"}
      </Button>
    </Card>
  );
}

/* ── Personal Model Card (#203) ─────────────────────────────────── */

function PersonalModelCard({ userId }: { userId: string }) {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: status, isLoading } = useQuery({
    queryKey: ["personal-status", userId],
    queryFn: () => getPersonalStatus(userId),
    refetchInterval: 30_000,
    retry: false,
  });

  const fineTune = useMutation({
    mutationFn: () => triggerPersonalFineTune(userId),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["personal-status", userId] });
      toast({
        title: data.status === "fine_tuned" ? "Model fine-tuned" : "Not enough data yet",
        description: data.status === "fine_tuned"
          ? `Validation accuracy: ${data.val_accuracy_pct}%`
          : "Keep rating predictions to collect more labeled epochs.",
      });
    },
    onError: () => {
      toast({ title: "Fine-tune failed", description: "ML backend unreachable.", variant: "destructive" });
    },
  });

  const MILESTONE = status?.next_milestone ?? 30;
  const collected = status?.buffer_size ?? 0;
  const pct = Math.min(100, Math.round((collected / MILESTONE) * 100));
  const isActive = status?.personal_model_active ?? false;

  return (
    <Card className="p-5 space-y-4">
      <div className="flex items-center gap-2">
        <Brain className="h-4 w-4 text-violet-400" />
        <h3 className="text-sm font-semibold">Personal Model</h3>
        {isActive ? (
          <Badge className="text-xs bg-green-500/10 text-green-400 border-green-500/30">Active</Badge>
        ) : (
          <Badge className="text-xs bg-zinc-500/10 text-zinc-400 border-zinc-500/30">Inactive</Badge>
        )}
      </div>

      {isLoading && (
        <p className="text-xs text-muted-foreground animate-pulse">Loading personalization status…</p>
      )}

      {status && (
        <>
          <p className="text-xs text-muted-foreground">{status.message}</p>

          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Labeled epochs</span>
              <span>{collected} / {MILESTONE}</span>
            </div>
            <Progress value={pct} className="h-2" />
          </div>

          <div className="grid grid-cols-3 gap-3 text-xs">
            <div className="rounded-lg bg-muted/30 p-2 text-center">
              <p className="text-muted-foreground">Sessions</p>
              <p className="font-semibold text-foreground">{status.total_sessions}</p>
            </div>
            <div className="rounded-lg bg-muted/30 p-2 text-center">
              <p className="text-muted-foreground">Accuracy</p>
              <p className="font-semibold text-foreground">
                {isActive ? `${status.head_accuracy_pct}%` : "—"}
              </p>
            </div>
            <div className="rounded-lg bg-muted/30 p-2 text-center">
              <p className="text-muted-foreground">Baseline</p>
              <p className={`font-semibold ${status.baseline_ready ? "text-green-400" : "text-muted-foreground"}`}>
                {status.baseline_ready ? "Ready" : `${status.baseline_frames}/30`}
              </p>
            </div>
          </div>

          <Button
            size="sm"
            variant="outline"
            disabled={fineTune.isPending || collected < 10}
            onClick={() => fineTune.mutate()}
            className="w-full text-xs"
          >
            {fineTune.isPending ? "Fine-tuning…" : "Fine-Tune Now"}
          </Button>
          {collected < 10 && (
            <p className="text-[10px] text-muted-foreground text-center">
              Rate emotion predictions in the Emotions page to collect labeled epochs.
            </p>
          )}
        </>
      )}
    </Card>
  );
}

/* ── Data & Privacy Card (GDPR Art. 15 / 17 / 20) ───────────────────────── */

function DataPrivacyCard({ userId }: { userId: string }) {
  const { toast } = useToast();
  const [isExporting, setIsExporting] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const { data: historyData } = useQuery({
    queryKey: ["export-history", userId],
    queryFn: async () => {
      const token = localStorage.getItem("auth_token");
      const res = await fetch(resolveUrl(`/api/user/${userId}/export-history`), {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) return { exportHistory: [] as string[] };
      return res.json() as Promise<{ exportHistory: string[] }>;
    },
    staleTime: 60_000,
    retry: false,
  });

  const lastExport = historyData?.exportHistory?.at(-1);

  const handleFullExport = async () => {
    setIsExporting(true);
    try {
      const token = localStorage.getItem("auth_token");
      const res = await fetch(resolveUrl(`/api/user/${userId}/export-all`), {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error((err as any).error ?? "Export failed");
      }
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `neural_dream_export_${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      const totalRows = Object.values(data.metadata?.rowCounts ?? {}).reduce(
        (acc: number, v) => acc + (v as number),
        0
      );
      toast({
        title: "Data export ready",
        description: `Downloaded ${totalRows} records across ${data.metadata?.dataCategories?.length ?? 0} categories.`,
      });
    } catch (err) {
      toast({
        title: "Export failed",
        description: String(err),
        variant: "destructive",
      });
    } finally {
      setIsExporting(false);
    }
  };

  const handleDeleteAccount = async () => {
    setIsDeleting(true);
    try {
      const token = localStorage.getItem("auth_token");
      const res = await fetch(resolveUrl(`/api/user/${userId}`), {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ confirm: true }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error((data as any).error ?? "Deletion request failed");
      }
      toast({
        title: "Deletion request submitted",
        description: `Your account is scheduled for permanent deletion on ${new Date((data as any).scheduledDeletionDate).toLocaleDateString()}. You have 30 days to cancel by contacting support.`,
      });
    } catch (err) {
      toast({
        title: "Request failed",
        description: String(err),
        variant: "destructive",
      });
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <Card className="glass-card p-6 rounded-xl">
      <div className="flex items-center gap-2 mb-4">
        <Shield className="h-5 w-5 text-primary" />
        <h3 className="text-lg font-semibold">Data &amp; Privacy</h3>
      </div>
      <p className="text-sm text-muted-foreground mb-5">
        Under GDPR Art. 20 you can download all your data, and under Art. 17 you can
        request permanent deletion. Deletion requests have a 30-day grace period.
      </p>

      <div className="space-y-3">
        <Button
          className="w-full bg-primary/10 border border-primary/30 text-primary hover:bg-primary/20"
          onClick={handleFullExport}
          disabled={isExporting}
        >
          <Download className="mr-2 h-4 w-4" />
          {isExporting ? "Preparing export…" : "Download My Data (JSON)"}
        </Button>

        {lastExport && (
          <p className="text-xs text-muted-foreground text-center">
            Last exported: {new Date(lastExport).toLocaleString()}
          </p>
        )}

        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button
              variant="destructive"
              className="w-full bg-destructive/10 border border-destructive/30 text-destructive hover:bg-destructive/20"
              disabled={isDeleting}
            >
              <AlertTriangle className="mr-2 h-4 w-4" />
              {isDeleting ? "Submitting request…" : "Delete My Account"}
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent className="glass-card border-destructive/30">
            <AlertDialogHeader>
              <AlertDialogTitle className="font-medium text-destructive">
                Request Account Deletion
              </AlertDialogTitle>
              <AlertDialogDescription>
                This will schedule permanent deletion of your account and all associated
                data — EEG sessions, dream journals, health metrics, emotion readings,
                and AI chat history. You have a <strong>30-day grace period</strong> to
                cancel by contacting support. After that, all data is permanently removed
                and cannot be recovered.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={handleDeleteAccount}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                Yes, Request Deletion
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </Card>
  );
}
