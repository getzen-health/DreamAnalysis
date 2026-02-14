import { useState, useEffect, useCallback } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
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
import { AlertTriangle, Download, BarChart3, Brain, Cpu } from "lucide-react";
import { useTheme } from "@/hooks/use-theme";
import { useMetrics } from "@/hooks/use-metrics";
import { useToast } from "@/hooks/use-toast";
import { useInference } from "@/hooks/use-inference";
import { getModelsBenchmarks, getCalibrationStatus, type BenchmarkResult, type CalibrationStatus } from "@/lib/ml-api";
import { CalibrationWizard } from "@/components/calibration-wizard";

interface SettingsState {
  electrodeCount: string;
  samplingRate: string;
  stressAlertThreshold: number;
  chartAnimations: boolean;
  neuralFlowEffects: boolean;
  healthAlerts: boolean;
  localProcessing: boolean;
  dataEncryption: boolean;
  anonymousAnalytics: boolean;
}

const defaultSettings: SettingsState = {
  electrodeCount: "64",
  samplingRate: "500",
  stressAlertThreshold: 75,
  chartAnimations: true,
  neuralFlowEffects: true,
  healthAlerts: true,
  localProcessing: true,
  dataEncryption: true,
  anonymousAnalytics: false,
};

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const { userId } = useMetrics();
  const { toast } = useToast();
  const [settings, setSettings] = useState<SettingsState>(defaultSettings);
  const [benchmarks, setBenchmarks] = useState<Record<string, BenchmarkResult>>({});
  const [calibrationStatus, setCalibrationStatus] = useState<CalibrationStatus | null>(null);
  const { isLocal, latencyMs, isReady } = useInference();

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
        // Use defaults if API unavailable
        console.error("Failed to load settings:", error);
      }
    }
    loadSettings();
  }, [userId]);

  // Load model benchmarks and calibration status
  useEffect(() => {
    async function loadBenchmarks() {
      try {
        const data = await getModelsBenchmarks();
        setBenchmarks(data);
      } catch {
        // Benchmarks not available
      }
    }
    async function loadCalibration() {
      try {
        const status = await getCalibrationStatus();
        setCalibrationStatus(status);
      } catch {
        // Calibration not available
      }
    }
    loadBenchmarks();
    loadCalibration();
  }, []);

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
        {/* BCI Configuration */}
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-futuristic font-semibold mb-6">
            BCI Configuration
          </h3>
          <div className="space-y-6">
            <div>
              <Label className="text-sm font-medium text-foreground/80 mb-2">
                Electrode Count
              </Label>
              <Select
                value={settings.electrodeCount}
                onValueChange={(value) =>
                  updateSetting("electrodeCount", value)
                }
              >
                <SelectTrigger className="w-full bg-card/50 border border-primary/30">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="32">32 Channels</SelectItem>
                  <SelectItem value="64">64 Channels</SelectItem>
                  <SelectItem value="128">128 Channels</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-sm font-medium text-foreground/80 mb-2">
                Sampling Rate
              </Label>
              <Select
                value={settings.samplingRate}
                onValueChange={(value) =>
                  updateSetting("samplingRate", value)
                }
              >
                <SelectTrigger className="w-full bg-card/50 border border-primary/30">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="250">250 Hz</SelectItem>
                  <SelectItem value="500">500 Hz</SelectItem>
                  <SelectItem value="1000">1000 Hz</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-sm font-medium text-foreground/80 mb-4">
                Alert Thresholds
              </Label>
              <div className="space-y-3">
                <div>
                  <Label className="text-xs text-foreground/60">
                    Stress Level Alert
                  </Label>
                  <Slider
                    value={[settings.stressAlertThreshold]}
                    onValueChange={([value]) =>
                      updateSetting("stressAlertThreshold", value)
                    }
                    max={100}
                    step={1}
                    className="w-full mt-2"
                  />
                  <div className="flex justify-between text-xs text-foreground/50 mt-1">
                    <span>0%</span>
                    <span>{settings.stressAlertThreshold}%</span>
                    <span>100%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Card>

        {/* Interface Settings */}
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-futuristic font-semibold mb-6">
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
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-futuristic font-semibold mb-6">
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

        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-futuristic font-semibold mb-6">
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
                  <AlertDialogTitle className="font-futuristic text-destructive">
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

      {/* Personal Calibration & Edge Inference */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Personal Calibration (Phase 9) */}
        <div>
          <CalibrationWizard
            onComplete={(result) => {
              setCalibrationStatus({
                calibrated: result.calibrated,
                n_samples: 3,
                personal_accuracy: result.personal_accuracy,
                classes: ["relaxed", "focused", "stressed"],
              });
            }}
          />
          {calibrationStatus?.calibrated && (
            <Card className="glass-card p-4 rounded-xl mt-3">
              <div className="flex items-center gap-2 mb-2">
                <Brain className="h-4 w-4 text-success" />
                <span className="text-sm font-medium text-success">Calibrated</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs text-foreground/60">
                <span>Samples: {calibrationStatus.n_samples}</span>
                <span>Accuracy: {(calibrationStatus.personal_accuracy * 100).toFixed(0)}%</span>
              </div>
            </Card>
          )}
        </div>

        {/* Edge Inference (Phase 12) */}
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-futuristic font-semibold mb-4 flex items-center gap-2">
            <Cpu className="h-5 w-5 text-primary" />
            Edge Inference
          </h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label className="text-sm">Inference Mode</Label>
              <span className={`text-sm font-mono ${isLocal ? "text-success" : "text-foreground/60"}`}>
                {isReady ? (isLocal ? "Local (ONNX)" : "Server API") : "Initializing..."}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <Label className="text-sm">Latency</Label>
              <span className="text-sm font-mono text-foreground/60">
                {latencyMs > 0 ? `${latencyMs.toFixed(1)}ms` : "—"}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <Label className="text-sm">Local Processing</Label>
              <Switch
                checked={settings.localProcessing}
                onCheckedChange={(checked) =>
                  updateSetting("localProcessing", checked)
                }
                data-testid="switch-edge-inference"
              />
            </div>
            <p className="text-xs text-foreground/40">
              When ONNX models are available in /public/models/, inference runs locally in the browser with &lt;10ms latency. Otherwise, falls back to the server API.
            </p>
          </div>
        </Card>
      </div>

      {/* Model Performance */}
      {Object.keys(benchmarks).length > 0 && (
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-futuristic font-semibold mb-6 flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            Model Performance
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 text-foreground/60 font-medium">Model</th>
                  <th className="text-left py-2 text-foreground/60 font-medium">Dataset</th>
                  <th className="text-right py-2 text-foreground/60 font-medium">Accuracy</th>
                  <th className="text-right py-2 text-foreground/60 font-medium">F1 Score</th>
                  <th className="text-right py-2 text-foreground/60 font-medium">Inference</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(benchmarks).map(([key, bench]) => (
                  <tr key={key} className="border-b border-border/50">
                    <td className="py-3 font-mono text-foreground/80">
                      {bench.model_name.replace(/_/g, " ")}
                    </td>
                    <td className="py-3 text-foreground/60">
                      {bench.dataset}
                    </td>
                    <td className="py-3 text-right font-mono">
                      <span className={bench.accuracy >= 0.8 ? "text-success" : bench.accuracy >= 0.6 ? "text-warning" : "text-destructive"}>
                        {(bench.accuracy * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3 text-right font-mono text-foreground/80">
                      {(bench.f1_macro * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 text-right font-mono text-foreground/60">
                      {bench.inference_time_ms !== undefined
                        ? `${bench.inference_time_ms.toFixed(1)}ms`
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Per-class breakdown */}
          {Object.entries(benchmarks).map(([key, bench]) => (
            bench.per_class && Object.keys(bench.per_class).length > 0 && (
              <details key={`detail-${key}`} className="mt-4">
                <summary className="cursor-pointer text-sm text-foreground/60 hover:text-foreground/80">
                  {bench.model_name.replace(/_/g, " ")} — per-class details
                </summary>
                <div className="mt-2 grid grid-cols-2 md:grid-cols-3 gap-2">
                  {Object.entries(bench.per_class).map(([cls, metrics]) => (
                    <div key={cls} className="p-2 bg-card/50 rounded border border-border/30 text-xs">
                      <p className="font-medium text-foreground/80">{cls}</p>
                      <p className="text-foreground/50">
                        P: {(metrics.precision * 100).toFixed(0)}% |
                        R: {(metrics.recall * 100).toFixed(0)}% |
                        F1: {(metrics.f1 * 100).toFixed(0)}%
                      </p>
                    </div>
                  ))}
                </div>
              </details>
            )
          ))}
        </Card>
      )}
    </main>
  );
}
