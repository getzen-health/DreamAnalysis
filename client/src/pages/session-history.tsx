import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Clock, Download, Trash2, Play, FileText } from "lucide-react";
import { listSessions, getSession, deleteSession, exportSession, type SessionSummary } from "@/lib/ml-api";

export default function SessionHistory() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [sessionDetail, setSessionDetail] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);

  const loadSessions = async () => {
    try {
      const data = await listSessions();
      setSessions(data);
    } catch {
      // API not available
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSessions();
  }, []);

  const handleView = async (sessionId: string) => {
    setSelectedSession(sessionId);
    try {
      const data = await getSession(sessionId);
      setSessionDetail(data);
    } catch {
      console.error("Failed to load session");
    }
  };

  const handleExport = async (sessionId: string) => {
    try {
      const data = await exportSession(sessionId);
      const blob = new Blob([data], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `session_${sessionId}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      console.error("Export failed");
    }
  };

  const handleDelete = async (sessionId: string) => {
    try {
      await deleteSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.session_id !== sessionId));
      if (selectedSession === sessionId) {
        setSelectedSession(null);
        setSessionDetail(null);
      }
    } catch {
      console.error("Delete failed");
    }
  };

  const formatDate = (ts: number) => {
    return new Date(ts * 1000).toLocaleString();
  };

  const formatDuration = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    return `${m}m ${s}s`;
  };

  const typeColors: Record<string, string> = {
    sleep: "text-secondary",
    meditation: "text-primary",
    neurofeedback: "text-warning",
    general: "text-foreground/70",
  };

  return (
    <main className="p-4 md:p-6 space-y-6">
      <div className="flex items-center gap-3">
        <Clock className="h-6 w-6 text-primary" />
        <h2 className="text-xl font-futuristic font-bold">Session History</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Session List */}
        <div className="lg:col-span-1 space-y-3">
          {loading && (
            <p className="text-sm text-foreground/50">Loading sessions...</p>
          )}

          {!loading && sessions.length === 0 && (
            <Card className="glass-card p-6 rounded-xl text-center">
              <FileText className="h-8 w-8 text-foreground/30 mx-auto mb-2" />
              <p className="text-sm text-foreground/50">No recorded sessions yet.</p>
              <p className="text-xs text-foreground/40 mt-1">
                Use the Record button in Brain Monitor to start.
              </p>
            </Card>
          )}

          {sessions.map((session) => (
            <Card
              key={session.session_id}
              className={`glass-card p-4 rounded-xl cursor-pointer transition-all ${
                selectedSession === session.session_id
                  ? "border-primary/30 bg-primary/5"
                  : "hover:bg-card/50"
              }`}
              onClick={() => handleView(session.session_id)}
            >
              <div className="flex items-center justify-between mb-2">
                <span className={`text-xs font-mono uppercase ${typeColors[session.session_type] || "text-foreground/70"}`}>
                  {session.session_type}
                </span>
                <span className="text-xs text-foreground/50">
                  {session.start_time ? formatDate(session.start_time) : "—"}
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-foreground/70">
                  {session.summary?.duration_sec
                    ? formatDuration(session.summary.duration_sec)
                    : "—"}
                </span>
                <span className="text-xs text-foreground/40">
                  {session.summary?.n_channels || 0}ch
                </span>
              </div>
              <div className="flex gap-2 mt-3">
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-7 text-xs text-primary"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleExport(session.session_id);
                  }}
                >
                  <Download className="h-3 w-3 mr-1" />
                  CSV
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-7 text-xs text-destructive"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(session.session_id);
                  }}
                >
                  <Trash2 className="h-3 w-3 mr-1" />
                  Delete
                </Button>
              </div>
            </Card>
          ))}
        </div>

        {/* Session Detail */}
        <div className="lg:col-span-2">
          {selectedSession && sessionDetail ? (
            <Card className="glass-card p-6 rounded-xl hover-glow">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-futuristic font-semibold">
                  Session Detail
                </h3>
                <span className="text-xs font-mono text-foreground/50">
                  ID: {selectedSession}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="text-center">
                  <p className="text-xs text-foreground/50">Duration</p>
                  <p className="text-lg font-mono font-bold">
                    {sessionDetail.summary &&
                    typeof sessionDetail.summary === "object" &&
                    "duration_sec" in (sessionDetail.summary as Record<string, unknown>)
                      ? formatDuration(
                          (sessionDetail.summary as Record<string, number>).duration_sec
                        )
                      : "—"}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-xs text-foreground/50">Frames</p>
                  <p className="text-lg font-mono font-bold">
                    {sessionDetail.summary &&
                    typeof sessionDetail.summary === "object" &&
                    "n_frames" in (sessionDetail.summary as Record<string, unknown>)
                      ? String(
                          (sessionDetail.summary as Record<string, number>).n_frames
                        )
                      : "—"}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-xs text-foreground/50">Channels</p>
                  <p className="text-lg font-mono font-bold">
                    {sessionDetail.summary &&
                    typeof sessionDetail.summary === "object" &&
                    "n_channels" in (sessionDetail.summary as Record<string, unknown>)
                      ? String(
                          (sessionDetail.summary as Record<string, number>).n_channels
                        )
                      : "—"}
                  </p>
                </div>
              </div>

              {/* Analysis Timeline */}
              {Array.isArray(sessionDetail.analysis_timeline) &&
                (sessionDetail.analysis_timeline as Array<Record<string, unknown>>).length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium text-foreground/70 mb-3">Analysis Timeline</h4>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {(sessionDetail.analysis_timeline as Array<Record<string, unknown>>).map(
                        (entry, idx) => (
                          <div
                            key={idx}
                            className="flex items-center gap-3 text-xs p-2 bg-card/30 rounded"
                          >
                            <span className="font-mono text-foreground/50 w-16">
                              {typeof entry.time_offset === "number"
                                ? `${entry.time_offset.toFixed(1)}s`
                                : "—"}
                            </span>
                            {entry.band_powers &&
                              typeof entry.band_powers === "object" ? (
                                <span className="text-primary">
                                  α:{" "}
                                  {(
                                    entry.band_powers as Record<string, number>
                                  ).alpha?.toFixed(2) || "—"}
                                </span>
                              ) : null}
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}
            </Card>
          ) : (
            <Card className="glass-card p-12 rounded-xl text-center">
              <Play className="h-8 w-8 text-foreground/20 mx-auto mb-3" />
              <p className="text-sm text-foreground/40">Select a session to view details</p>
            </Card>
          )}
        </div>
      </div>
    </main>
  );
}
