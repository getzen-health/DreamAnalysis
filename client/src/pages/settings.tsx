import { useState, useEffect, useCallback, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
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
import { AlertTriangle, Download, Apple, Smartphone, Upload, CheckCircle2, XCircle, Info } from "lucide-react";
import { useTheme } from "@/hooks/use-theme";
const USER_ID = "default";
import { useToast } from "@/hooks/use-toast";
import { ingestHealthData } from "@/lib/ml-api";

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
  const [settings, setSettings] = useState<SettingsState>(defaultSettings);
  const [healthStatus, setHealthStatus] = useState<HealthConnectionStatus>({
    apple_health: false,
    google_fit: false,
  });
  const [uploading, setUploading] = useState<string | null>(null);

  const appleFileRef = useRef<HTMLInputElement>(null);
  const googleFileRef = useRef<HTMLInputElement>(null);

  // Load settings from API on mount
  useEffect(() => {
    async function loadSettings() {
      try {
        const response = await fetch(`/api/settings/${userId}`);
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

  // Save settings to API
  const saveSettings = useCallback(
    async (updated: SettingsState) => {
      try {
        await fetch(`/api/settings/${userId}`, {
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

  const handleDataExport = async () => {
    try {
      const response = await fetch(`/api/export/${userId}`);
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
      const response = await fetch(`/api/export/${userId}?type=dreams`);
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

  const handleClearAllData = async () => {
    try {
      await fetch(`/api/settings/${userId}/data`, {
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
    <main className="p-4 md:p-6 space-y-6">
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

            {/* Google Fit */}
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-primary/10">
                  <Smartphone className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <p className="text-sm font-medium">Google Fit</p>
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

            {/* Coming soon note */}
            <div className="flex items-start gap-2 p-3 rounded-lg bg-muted/50 border border-border">
              <Info className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
              <p className="text-xs text-muted-foreground">
                Direct API connection coming soon. For now, export your data from the Health app and upload the file here.
              </p>
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

            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  variant="destructive"
                  className="w-full mt-6 bg-destructive/10 border border-destructive/30 text-destructive hover:bg-destructive/20"
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

    </main>
  );
}
