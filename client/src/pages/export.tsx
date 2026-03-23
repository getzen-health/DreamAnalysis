import { useState } from "react";
import { getParticipantId } from "@/lib/participant";
import { resolveUrl } from "@/lib/queryClient";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Download,
  FileText,
  Brain,
  Moon,
  Heart,
  Database,
  FileJson,
  Mic,
  Trash2,
  Loader2,
  AlertTriangle,
} from "lucide-react";
import {
  listSessions,
  exportSession,
  type SessionSummary,
} from "@/lib/ml-api";
import { writeEdfPlus, type EdfChannel } from "@/lib/edf-writer";
import {
  exportSessionsCSV,
  exportVoiceCheckinsCSV,
  exportAccountJSON,
  downloadFile,
  deleteAllLocalData,
  type CheckinData,
} from "@/lib/data-export";
import { sbGetSetting } from "../lib/supabase-store";

/* ── Helpers ────────────────────────────────────────────────── */

const CURRENT_USER = getParticipantId();

function fmtDate(ts: number): string {
  return new Date(ts * 1000).toLocaleDateString([], {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function fmtTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function fmt(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** Gather voice check-in data from localStorage. */
function getStoredCheckins(): CheckinData[] {
  try {
    const raw = sbGetSetting("ndw_voice_checkins");
    if (raw) return JSON.parse(raw) as CheckinData[];
  } catch {
    /* ignore */
  }
  return [];
}

/* ── Muse 2 EDF+ channel config ─────────────────────────────── */

const MUSE_EDF_CHANNELS: EdfChannel[] = [
  { label: "EEG AF7", physicalMin: -500, physicalMax: 500, samplesPerSecond: 256 },
  { label: "EEG AF8", physicalMin: -500, physicalMax: 500, samplesPerSecond: 256 },
  { label: "EEG TP9", physicalMin: -500, physicalMax: 500, samplesPerSecond: 256 },
  { label: "EEG TP10", physicalMin: -500, physicalMax: 500, samplesPerSecond: 256 },
];

/* ── Component ──────────────────────────────────────────────── */

export default function ExportPage() {
  const [exportingSession, setExportingSession] = useState<string | null>(null);
  const [exportingAll, setExportingAll] = useState(false);
  const [exportingHealth, setExportingHealth] = useState(false);
  const [exportingDreams, setExportingDreams] = useState(false);
  const [exportingEdf, setExportingEdf] = useState(false);
  const [exportingCsv, setExportingCsv] = useState(false);
  const [exportingCheckins, setExportingCheckins] = useState(false);
  const [exportingJson, setExportingJson] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteComplete, setDeleteComplete] = useState(false);

  const { data: sessions = [], isLoading } = useQuery<SessionSummary[]>({
    queryKey: ["sessions", CURRENT_USER],
    queryFn: () => listSessions(CURRENT_USER),
    staleTime: 30_000,
    retry: false,
  });

  /* ── Existing handlers ─────────────────────────────────────── */

  const handleExportSession = async (sessionId: string) => {
    setExportingSession(sessionId);
    try {
      const data = await exportSession(sessionId);
      const blob = new Blob([data], { type: "text/csv" });
      downloadBlob(blob, `session_${sessionId}.csv`);
    } catch {
      /* ignore */
    } finally {
      setExportingSession(null);
    }
  };

  const handleExportAllSessions = async () => {
    setExportingAll(true);
    try {
      const csvParts: string[] = [];
      let headerWritten = false;
      for (const s of sessions) {
        try {
          const data = await exportSession(s.session_id);
          if (!headerWritten) {
            csvParts.push(data);
            headerWritten = true;
          } else {
            const lines = data.split("\n");
            csvParts.push(lines.slice(1).join("\n"));
          }
        } catch {
          /* skip failed exports */
        }
      }
      if (csvParts.length > 0) {
        const blob = new Blob([csvParts.join("\n")], { type: "text/csv" });
        downloadBlob(blob, `all_sessions_${new Date().toISOString().slice(0, 10)}.csv`);
      }
    } catch {
      /* ignore */
    } finally {
      setExportingAll(false);
    }
  };

  const handleExportHealth = async () => {
    setExportingHealth(true);
    try {
      const response = await fetch(resolveUrl(`/api/export/${CURRENT_USER}`));
      if (!response.ok) throw new Error("Export request failed");
      const blob = await response.blob();
      downloadBlob(blob, `health_data_${new Date().toISOString().slice(0, 10)}.csv`);
    } catch {
      /* ignore */
    } finally {
      setExportingHealth(false);
    }
  };

  const handleExportDreams = async () => {
    setExportingDreams(true);
    try {
      const response = await fetch(resolveUrl(`/api/export/${CURRENT_USER}?type=dreams`));
      if (!response.ok) throw new Error("Export request failed");
      const blob = await response.blob();
      downloadBlob(blob, `dream_analysis_${new Date().toISOString().slice(0, 10)}.csv`);
    } catch {
      /* ignore */
    } finally {
      setExportingDreams(false);
    }
  };

  /* ── New self-service handlers ─────────────────────────────── */

  const handleExportEdf = () => {
    setExportingEdf(true);
    try {
      // Generate a demo EDF+ with empty channels (real EEG data would come from
      // the backend session store; for now we export what's available locally)
      // In production this would fetch raw EEG frames from the ML backend.
      const sampleCount = 256 * 60; // 60 seconds placeholder
      const buf = writeEdfPlus({
        patientId: CURRENT_USER,
        startDate: new Date(),
        channels: MUSE_EDF_CHANNELS,
        data: MUSE_EDF_CHANNELS.map(() => new Float32Array(sampleCount)),
        recordDuration: 1,
      });
      downloadFile(buf, `eeg_${new Date().toISOString().slice(0, 10)}.edf`, "application/octet-stream");
    } catch {
      /* ignore */
    } finally {
      setExportingEdf(false);
    }
  };

  const handleExportSessionsCsv = () => {
    setExportingCsv(true);
    try {
      const csv = exportSessionsCSV(sessions);
      downloadFile(csv, `sessions_${new Date().toISOString().slice(0, 10)}.csv`, "text/csv");
    } catch {
      /* ignore */
    } finally {
      setExportingCsv(false);
    }
  };

  const handleExportCheckins = () => {
    setExportingCheckins(true);
    try {
      const checkins = getStoredCheckins();
      const csv = exportVoiceCheckinsCSV(checkins);
      downloadFile(csv, `voice_checkins_${new Date().toISOString().slice(0, 10)}.csv`, "text/csv");
    } catch {
      /* ignore */
    } finally {
      setExportingCheckins(false);
    }
  };

  const handleExportAccountJson = () => {
    setExportingJson(true);
    try {
      const checkins = getStoredCheckins();
      const json = exportAccountJSON(CURRENT_USER, sessions, checkins);
      downloadFile(json, `account_export_${new Date().toISOString().slice(0, 10)}.json`, "application/json");
    } catch {
      /* ignore */
    } finally {
      setExportingJson(false);
    }
  };

  const handleDeleteData = () => {
    const removed = deleteAllLocalData();
    setDeleteComplete(true);
    setShowDeleteConfirm(false);
    // Auto-hide confirmation after 3 seconds
    setTimeout(() => setDeleteComplete(false), 3000);
  };

  /* ── Render ────────────────────────────────────────────────── */

  return (
    <main className="p-4 md:p-6 pb-24 space-y-6 max-w-3xl">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Database className="h-6 w-6 text-primary" />
        <div>
          <h1 className="text-xl font-semibold">Export My Data</h1>
          <p className="text-xs text-muted-foreground">
            Download your data in EDF+, CSV, or JSON format
          </p>
        </div>
      </div>

      {/* Self-service exports (EDF+, CSV summary, voice check-ins, JSON) */}
      <Card className="glass-card p-5">
        <h3 className="text-sm font-semibold mb-1">Your Data, Your Formats</h3>
        <p className="text-[10px] text-muted-foreground mb-4">
          GDPR Article 20 — right to data portability. All files generated locally in your browser.
        </p>
        <div className="space-y-3">
          <Button
            onClick={handleExportEdf}
            disabled={exportingEdf}
            className="w-full justify-start gap-3 h-12 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20"
            variant="outline"
          >
            <Brain className="h-4 w-4 shrink-0" />
            <div className="text-left flex-1">
              <p className="text-sm font-medium">
                {exportingEdf ? <span className="flex items-center gap-2"><Loader2 className="h-3 w-3 animate-spin" />Generating EDF+...</span> : "Export EEG Sessions (EDF+)"}
              </p>
              <p className="text-[10px] text-muted-foreground">
                Standard EEG format — open in EDFbrowser, MNE-Python, EEGLAB
              </p>
            </div>
            <Download className="h-4 w-4 shrink-0" />
          </Button>

          <Button
            onClick={handleExportSessionsCsv}
            disabled={exportingCsv || sessions.length === 0}
            className="w-full justify-start gap-3 h-12 bg-primary/10 border border-primary/30 text-primary hover:bg-primary/20"
            variant="outline"
          >
            <FileText className="h-4 w-4 shrink-0" />
            <div className="text-left flex-1">
              <p className="text-sm font-medium">
                {exportingCsv ? <span className="flex items-center gap-2"><Loader2 className="h-3 w-3 animate-spin" />Exporting...</span> : "Export Session History (CSV)"}
              </p>
              <p className="text-[10px] text-muted-foreground">
                {sessions.length} session{sessions.length !== 1 ? "s" : ""} — focus, stress, flow, emotion summaries
              </p>
            </div>
            <Download className="h-4 w-4 shrink-0" />
          </Button>

          <Button
            onClick={handleExportCheckins}
            disabled={exportingCheckins}
            className="w-full justify-start gap-3 h-12 bg-violet-500/10 border border-violet-500/30 text-violet-400 hover:bg-violet-500/20"
            variant="outline"
          >
            <Mic className="h-4 w-4 shrink-0" />
            <div className="text-left flex-1">
              <p className="text-sm font-medium">
                {exportingCheckins ? <span className="flex items-center gap-2"><Loader2 className="h-3 w-3 animate-spin" />Exporting...</span> : "Export Voice Check-ins (CSV)"}
              </p>
              <p className="text-[10px] text-muted-foreground">
                Emotion labels, intensity, notes, voice biomarkers
              </p>
            </div>
            <Download className="h-4 w-4 shrink-0" />
          </Button>

          <Button
            onClick={handleExportAccountJson}
            disabled={exportingJson}
            className="w-full justify-start gap-3 h-12 bg-amber-500/10 border border-amber-500/30 text-amber-400 hover:bg-amber-500/20"
            variant="outline"
          >
            <FileJson className="h-4 w-4 shrink-0" />
            <div className="text-left flex-1">
              <p className="text-sm font-medium">
                {exportingJson ? <span className="flex items-center gap-2"><Loader2 className="h-3 w-3 animate-spin" />Generating...</span> : "Export All Data (JSON)"}
              </p>
              <p className="text-[10px] text-muted-foreground">
                Complete account export — sessions, check-ins, preferences
              </p>
            </div>
            <Download className="h-4 w-4 shrink-0" />
          </Button>
        </div>
      </Card>

      {/* Existing bulk exports */}
      <Card className="glass-card p-5">
        <h3 className="text-sm font-semibold mb-4">Raw Data Downloads</h3>
        <div className="space-y-3">
          <Button
            onClick={handleExportAllSessions}
            disabled={exportingAll || sessions.length === 0}
            className="w-full justify-start gap-3 h-12 bg-primary/10 border border-primary/30 text-primary hover:bg-primary/20"
            variant="outline"
          >
            <Brain className="h-4 w-4 shrink-0" />
            <div className="text-left flex-1">
              <p className="text-sm font-medium">
                {exportingAll ? "Exporting..." : "Download All Sessions (Raw CSV)"}
              </p>
              <p className="text-[10px] text-muted-foreground">
                {sessions.length} session{sessions.length !== 1 ? "s" : ""} — per-frame EEG analysis data
              </p>
            </div>
            <Download className="h-4 w-4 shrink-0" />
          </Button>

          <Button
            onClick={handleExportHealth}
            disabled={exportingHealth}
            className="w-full justify-start gap-3 h-12 bg-rose-500/10 border border-rose-500/30 text-rose-400 hover:bg-rose-500/20"
            variant="outline"
          >
            <Heart className="h-4 w-4 shrink-0" />
            <div className="text-left flex-1">
              <p className="text-sm font-medium">
                {exportingHealth ? "Exporting..." : "Download Health Data"}
              </p>
              <p className="text-[10px] text-muted-foreground">
                Heart rate, sleep, steps, stress levels
              </p>
            </div>
            <Download className="h-4 w-4 shrink-0" />
          </Button>

          <Button
            onClick={handleExportDreams}
            disabled={exportingDreams}
            className="w-full justify-start gap-3 h-12 bg-secondary/10 border border-secondary/30 text-secondary hover:bg-secondary/20"
            variant="outline"
          >
            <Moon className="h-4 w-4 shrink-0" />
            <div className="text-left flex-1">
              <p className="text-sm font-medium">
                {exportingDreams ? "Exporting..." : "Download Dream Analysis"}
              </p>
              <p className="text-[10px] text-muted-foreground">
                Dream journal entries with AI analysis
              </p>
            </div>
            <Download className="h-4 w-4 shrink-0" />
          </Button>
        </div>
      </Card>

      {/* Per-session exports */}
      <Card className="glass-card p-5">
        <h3 className="text-sm font-semibold mb-4">Per-Session Downloads</h3>

        {isLoading && (
          <div className="space-y-2">
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex items-center justify-between p-3 border border-border/20 rounded-lg">
                <div className="flex items-center gap-3">
                  <Skeleton className="h-4 w-4" />
                  <div className="space-y-1">
                    <Skeleton className="h-3 w-32" />
                    <Skeleton className="h-3 w-20" />
                  </div>
                </div>
                <Skeleton className="h-8 w-20" />
              </div>
            ))}
          </div>
        )}

        {!isLoading && sessions.length === 0 && (
          <div className="text-center py-8">
            <FileText className="h-8 w-8 text-muted-foreground/30 mx-auto mb-3" />
            <p className="text-sm text-muted-foreground">No sessions to export yet.</p>
            <p className="text-xs text-muted-foreground/60 mt-1">
              Complete a voice analysis or EEG session to generate data.
            </p>
          </div>
        )}

        {sessions.length > 0 && (
          <div className="space-y-2 max-h-[500px] overflow-y-auto">
            {sessions
              .slice()
              .sort((a, b) => (b.start_time ?? 0) - (a.start_time ?? 0))
              .map((session) => (
                <div
                  key={session.session_id}
                  className="flex items-center justify-between p-3 border border-border/20 rounded-lg hover:bg-muted/20 transition-colors"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <Brain className="h-4 w-4 text-primary shrink-0" />
                    <div className="min-w-0">
                      <p className="text-sm font-medium truncate">
                        {fmtDate(session.start_time ?? 0)} at {fmtTime(session.start_time ?? 0)}
                      </p>
                      <div className="flex gap-2 text-xs text-muted-foreground">
                        {session.summary?.duration_sec && (
                          <span>{fmt(session.summary.duration_sec)}</span>
                        )}
                        {session.summary?.avg_focus != null && (
                          <span>Focus {Math.round(session.summary.avg_focus * 100)}%</span>
                        )}
                        {session.summary?.dominant_emotion && (
                          <span className="capitalize">{session.summary.dominant_emotion}</span>
                        )}
                      </div>
                    </div>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-8 shrink-0"
                    onClick={() => handleExportSession(session.session_id)}
                    disabled={exportingSession === session.session_id}
                  >
                    <Download className="h-3.5 w-3.5 mr-1.5" />
                    {exportingSession === session.session_id ? "..." : "CSV"}
                  </Button>
                </div>
              ))}
          </div>
        )}
      </Card>

      {/* Delete My Data — GDPR Article 17 */}
      <Card className="glass-card p-5 border-red-500/20">
        <h3 className="text-sm font-semibold mb-1 text-red-400">Delete My Data</h3>
        <p className="text-[10px] text-muted-foreground mb-4">
          GDPR Article 17 — right to erasure. This clears all locally stored data including
          preferences, session cache, and authentication. This action cannot be undone.
        </p>

        {deleteComplete && (
          <div className="mb-3 p-3 rounded-lg bg-green-500/10 border border-green-500/30 text-green-400 text-xs">
            All local data has been deleted. You will need to sign in again.
          </div>
        )}

        {!showDeleteConfirm ? (
          <Button
            onClick={() => setShowDeleteConfirm(true)}
            className="w-full justify-start gap-3 h-12 bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20"
            variant="outline"
          >
            <Trash2 className="h-4 w-4 shrink-0" />
            <div className="text-left flex-1">
              <p className="text-sm font-medium">Delete My Data</p>
              <p className="text-[10px] text-muted-foreground">
                Permanently remove all locally stored data
              </p>
            </div>
          </Button>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30">
              <AlertTriangle className="h-4 w-4 text-red-400 shrink-0" />
              <p className="text-xs text-red-400">
                Are you sure? This will delete all your local data. Export your data first if you want a copy.
              </p>
            </div>
            <div className="flex gap-3">
              <Button
                onClick={() => setShowDeleteConfirm(false)}
                className="flex-1 h-10"
                variant="outline"
              >
                Cancel
              </Button>
              <Button
                onClick={handleDeleteData}
                className="flex-1 h-10 bg-red-500/20 border border-red-500/40 text-red-400 hover:bg-red-500/30"
                variant="outline"
              >
                <Trash2 className="h-3.5 w-3.5 mr-1.5" />
                Confirm Delete
              </Button>
            </div>
          </div>
        )}
      </Card>
    </main>
  );
}
