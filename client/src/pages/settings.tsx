import { useState, useEffect, useCallback, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
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
import { AlertTriangle, Download, Apple, Smartphone, Upload, CheckCircle2, XCircle, Info, Server } from "lucide-react";
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

      {/* Export Brain Data */}
      <ExportBrainDataCard userId={userId} />

    </main>
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
    const target = url.trim().replace(/\/$/, "") || "http://localhost:8000";
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

  return (
    <Card className="glass-card p-6 rounded-xl">
      <h3 className="text-lg font-semibold mb-1 flex items-center gap-2">
        <Server className="h-4 w-4 text-primary" />
        ML Backend
      </h3>
      <p className="text-sm text-muted-foreground mb-4">
        The ML backend runs locally on your machine to read EEG from your Muse headset.
        When using the hosted app (Vercel), you need to expose your local backend via{" "}
        <strong>ngrok</strong> so the browser can reach it.
      </p>

      {/* ngrok instructions */}
      <div className="mb-4 p-4 rounded-xl bg-muted/40 border border-border/40 text-sm space-y-2">
        <p className="font-medium text-xs text-muted-foreground uppercase tracking-wide">Quick setup</p>
        <ol className="space-y-1.5 text-sm text-muted-foreground list-none">
          {[
            <>Install ngrok: <code className="text-xs bg-muted px-1 rounded">brew install ngrok</code> (or download from ngrok.com)</>,
            <>Start your ML backend: <code className="text-xs bg-muted px-1 rounded">cd ~/NeuralDreamWorkshop/ml && uvicorn main:app --port 8000</code></>,
            <>Expose it: <code className="text-xs bg-muted px-1 rounded">ngrok http 8000</code></>,
            <>Copy the <code className="text-xs bg-muted px-1 rounded">https://xxxx.ngrok-free.app</code> URL and paste it below</>,
          ].map((step, i) => (
            <li key={i} className="flex items-start gap-2">
              <span className="text-primary font-semibold shrink-0">{i + 1}.</span>
              <span>{step}</span>
            </li>
          ))}
        </ol>
      </div>

      <div className="flex gap-2">
        <Input
          value={url}
          onChange={(e) => { setUrl(e.target.value); setTestResult(null); }}
          placeholder="https://xxxx.ngrok-free.app  (leave blank for localhost:8000)"
          className="flex-1 font-mono text-sm"
        />
        <Button variant="outline" onClick={testConnection} disabled={testing}>
          {testing ? "Testing…" : "Test"}
        </Button>
        <Button onClick={save}>Save</Button>
      </div>

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
        Current: <span className="font-mono">{url.trim() || "http://localhost:8000 (default)"}</span>
      </p>
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
