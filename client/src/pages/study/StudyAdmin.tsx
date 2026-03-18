import { useState, useEffect, useCallback, useRef } from "react";
import { useLocation } from "wouter";
import { resolveUrl } from "@/lib/queryClient";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { Loader2, RefreshCw, Download, ChevronDown, ChevronUp, Lock, Radio, Users, Activity, CheckCircle2, AlertCircle, Trash2 } from "lucide-react";

// ── Types ────────────────────────────────────────────────────────────────────

interface SurveyScore {
  label: string;
  value: number | string;
}

interface EegSnapshot {
  stress_level?: number;
  [key: string]: unknown;
}

interface AdminSession {
  id: number;
  blockType: "stress" | "food";
  interventionTriggered: boolean;
  partial: boolean;
  surveyJson: Record<string, unknown> | null;
  preEegJson: EegSnapshot | null;
  postEegJson: EegSnapshot | null;
  checkpointAt: string | null;
  createdAt: string;
  participantCode: string;
  dataQualityScore: number | null;
  durationSeconds: number | null;
}

type SessionStatus = "recording" | "complete" | "partial";

function sessionStatus(s: AdminSession): SessionStatus {
  if (s.partial) return "partial";
  if (s.surveyJson) return "complete";
  return "recording";
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}

function qualityBadgeClass(score: number): string {
  if (score >= 70) return "border-cyan-500/50 text-cyan-400 text-xs";
  if (score >= 40) return "border-yellow-500/50 text-yellow-400 text-xs";
  return "border-rose-500/50 text-rose-400 text-xs";
}

function StressBar({ pre, post }: { pre: number | undefined; post: number | undefined }) {
  if (pre === undefined && post === undefined) return null;
  const preW  = Math.round((pre  ?? 0) * 100);
  const postW = Math.round((post ?? 0) * 100);
  const dropped = (pre ?? 0) - (post ?? 0) > 0.05;
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-muted-foreground w-5">pre</span>
      <div className="relative h-2 w-20 rounded bg-muted overflow-hidden">
        <div className="h-full bg-orange-400/70 rounded" style={{ width: `${preW}%` }} />
      </div>
      <span className="text-muted-foreground">→</span>
      <div className="relative h-2 w-20 rounded bg-muted overflow-hidden">
        <div
          className={`h-full rounded ${dropped ? "bg-cyan-500/70" : "bg-orange-400/70"}`}
          style={{ width: `${postW}%` }}
        />
      </div>
      <span className="text-muted-foreground w-5">post</span>
      {dropped && <span className="text-cyan-400 text-[10px]">↓</span>}
    </div>
  );
}

interface AdminStats {
  total_participants: number;
  total_sessions: number;
  stress_sessions: number;
  food_sessions: number;
  complete_sessions: number;
  partial_sessions: number;
  avg_quality_score: number;
  avg_duration_seconds: number;
  avg_stress_reduction: number;
}

interface AdminParticipant {
  participantCode: string;
  age: number | null;
  dietType: string | null;
  hasAppleWatch: boolean;
  createdAt: string;
  sessions: AdminSession[];
  researcherNotes: string | null;
}

// ── Survey score extractor ────────────────────────────────────────────────────

function extractScores(session: AdminSession): SurveyScore[] {
  if (!session.surveyJson) return [];
  const s = session.surveyJson;
  const scores: SurveyScore[] = [];

  if (session.blockType === "food") {
    if (typeof s.what_ate === "string") scores.push({ label: "Ate", value: s.what_ate });
    if (typeof s.pre_hunger === "number") scores.push({ label: "Pre-hunger", value: s.pre_hunger });
    if (typeof s.pre_mood === "number") scores.push({ label: "Pre-mood", value: s.pre_mood });
    if (typeof s.food_healthy === "number") scores.push({ label: "Healthy", value: s.food_healthy });
    if (typeof s.post_energy === "number") scores.push({ label: "Post-energy", value: s.post_energy });
    if (typeof s.post_mood === "number") scores.push({ label: "Post-mood", value: s.post_mood });
    if (typeof s.post_satisfied === "number") scores.push({ label: "Satisfied", value: s.post_satisfied });
  } else {
    // stress session — generic display
    for (const [k, v] of Object.entries(s)) {
      if (typeof v === "number" || typeof v === "string") {
        scores.push({ label: k.replace(/_/g, " "), value: v });
      }
    }
  }

  return scores;
}

// ── Expandable participant row ────────────────────────────────────────────────

function ParticipantRow({ participant, onDeleteSession }: { participant: AdminParticipant; onDeleteSession: (sessionId: number) => void }) {
  const [expanded, setExpanded] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [notes, setNotes] = useState(participant.researcherNotes ?? "");
  const [savedNotes, setSavedNotes] = useState(participant.researcherNotes ?? "");
  const [showSaved, setShowSaved] = useState(false);
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleNotesBlur = async () => {
    if (notes === savedNotes) return;
    try {
      const res = await fetch(
        `/api/study/admin/participant/${participant.participantCode}/notes`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({ notes }),
        }
      );
      if (!res.ok) return;
      setSavedNotes(notes);
      setShowSaved(true);
      if (savedTimerRef.current) clearTimeout(savedTimerRef.current);
      savedTimerRef.current = setTimeout(() => setShowSaved(false), 2000);
    } catch {
      // silently ignore network errors
    }
  };

  const handleDeleteSession = async (sessionId: number) => {
    setDeleting(true);
    try {
      const res = await fetch(resolveUrl(`/api/study/admin/session/${sessionId}`), {
        method: "DELETE",
        credentials: "include",
      });
      if (res.ok) {
        onDeleteSession(sessionId);
      }
    } catch {
      // silently ignore network errors
    } finally {
      setDeleting(false);
      setConfirmDeleteId(null);
    }
  };

  const stressSessions = participant.sessions.filter((s) => s.blockType === "stress");
  const foodSessions = participant.sessions.filter((s) => s.blockType === "food");
  const totalSessions = participant.sessions.length;

  return (
    <>
      <TableRow
        className="cursor-pointer hover:bg-muted/40 transition-colors"
        onClick={() => setExpanded((e) => !e)}
      >
        <TableCell className="font-mono font-semibold text-primary whitespace-nowrap">
          {participant.participantCode}
        </TableCell>
        <TableCell className="tabular-nums text-sm">
          {participant.age ?? "—"}
        </TableCell>
        <TableCell className="text-sm capitalize">
          {participant.dietType ?? "—"}
        </TableCell>
        <TableCell>
          {participant.hasAppleWatch ? (
            <Badge variant="outline" className="border-indigo-500/50 text-indigo-400 text-xs">Yes</Badge>
          ) : (
            <span className="text-xs text-muted-foreground">No</span>
          )}
        </TableCell>
        <TableCell>
          <div className="flex items-center gap-1.5">
            <Badge variant="outline" className="text-xs tabular-nums">
              {totalSessions}
            </Badge>
            <span className="text-xs text-muted-foreground">
              ({stressSessions.length}S / {foodSessions.length}F)
            </span>
          </div>
        </TableCell>
        <TableCell className="text-xs text-muted-foreground whitespace-nowrap">
          {new Date(participant.createdAt).toLocaleDateString()}
        </TableCell>
        <TableCell>
          {expanded ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </TableCell>
      </TableRow>

      {/* Expanded sessions */}
      {expanded && participant.sessions.length > 0 && (
        <TableRow className="bg-muted/20">
          <TableCell colSpan={7} className="py-3 px-6">
            <div className="space-y-3">
              {participant.sessions.map((session) => {
                const scores = extractScores(session);
                return (
                  <div
                    key={session.id}
                    className="rounded-md border border-border/50 bg-background p-3 space-y-2"
                  >
                    {confirmDeleteId === session.id ? (
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-rose-400">Delete this session?</span>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 px-2 text-xs text-muted-foreground"
                          onClick={(e) => { e.stopPropagation(); setConfirmDeleteId(null); }}
                          disabled={deleting}
                        >
                          Cancel
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 px-2 text-xs text-rose-400 hover:text-rose-300 hover:bg-rose-500/10"
                          onClick={(e) => { e.stopPropagation(); void handleDeleteSession(session.id); }}
                          disabled={deleting}
                        >
                          {deleting ? <Loader2 className="h-3 w-3 animate-spin" /> : "Delete"}
                        </Button>
                      </div>
                    ) : (
                    <div className="flex items-center gap-2 flex-wrap">
                      <Badge
                        variant="outline"
                        className={
                          session.blockType === "stress"
                            ? "border-orange-500/50 text-orange-400 text-xs capitalize"
                            : "border-indigo-500/50 text-indigo-400 text-xs capitalize"
                        }
                      >
                        {session.blockType}
                      </Badge>
                      {/* Status badge */}
                      {sessionStatus(session) === "recording" && (
                        <Badge variant="outline" className="border-yellow-500/50 text-yellow-400 text-xs gap-1">
                          <Radio className="h-2.5 w-2.5" />recording
                        </Badge>
                      )}
                      {sessionStatus(session) === "complete" && (
                        <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 text-xs">complete</Badge>
                      )}
                      {sessionStatus(session) === "partial" && (
                        <Badge variant="outline" className="border-amber-500/50 text-amber-400 text-xs">partial</Badge>
                      )}
                      <Badge
                        variant="outline"
                        className={
                          session.interventionTriggered
                            ? "border-rose-500/50 text-rose-400 text-xs"
                            : "border-muted text-muted-foreground text-xs"
                        }
                      >
                        Intervention: {session.interventionTriggered ? "yes" : "no"}
                      </Badge>
                      {session.dataQualityScore != null && (
                        <Badge variant="outline" className={qualityBadgeClass(session.dataQualityScore)}>
                          Quality: {session.dataQualityScore}
                        </Badge>
                      )}
                      {session.durationSeconds != null && (
                        <span className="text-xs text-muted-foreground">
                          {formatDuration(session.durationSeconds)}
                        </span>
                      )}
                      <span className="text-xs text-muted-foreground ml-auto">
                        {new Date(session.createdAt).toLocaleString()}
                      </span>
                      <button
                        type="button"
                        className="p-1 rounded text-rose-400/60 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                        onClick={(e) => { e.stopPropagation(); setConfirmDeleteId(session.id); }}
                        title="Delete session"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                    )}

                    {/* Stress sparkline */}
                    {(session.preEegJson?.stress_level !== undefined || session.postEegJson?.stress_level !== undefined) && (
                      <StressBar
                        pre={session.preEegJson?.stress_level}
                        post={session.postEegJson?.stress_level}
                      />
                    )}

                    {scores.length > 0 && (
                      <div className="flex flex-wrap gap-2 pt-1">
                        {scores.map((sc, i) => (
                          <div
                            key={i}
                            className="text-xs rounded bg-muted px-2 py-0.5 flex items-center gap-1"
                          >
                            <span className="text-muted-foreground">{sc.label}:</span>
                            <span className="font-medium">{sc.value}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}

              {/* Researcher notes */}
              <div className="pt-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-medium text-muted-foreground">Researcher Notes</span>
                  {showSaved && (
                    <span className="text-xs text-cyan-400 animate-in fade-in duration-200">Saved</span>
                  )}
                </div>
                <Textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  onBlur={handleNotesBlur}
                  placeholder="Add researcher notes..."
                  rows={3}
                  className="min-h-0 text-xs resize-y bg-background"
                />
              </div>
            </div>
          </TableCell>
        </TableRow>
      )}

      {expanded && participant.sessions.length === 0 && (
        <TableRow className="bg-muted/20">
          <TableCell colSpan={7} className="py-3 px-6">
            <p className="text-center text-xs text-muted-foreground mb-3">
              No sessions recorded yet.
            </p>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-medium text-muted-foreground">Researcher Notes</span>
                {showSaved && (
                  <span className="text-xs text-cyan-400 animate-in fade-in duration-200">Saved</span>
                )}
              </div>
              <Textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                onBlur={handleNotesBlur}
                placeholder="Add researcher notes..."
                rows={3}
                className="min-h-0 text-xs resize-y bg-background"
              />
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function StudyAdmin() {
  const [, navigate] = useLocation();

  const [participants, setParticipants] = useState<AdminParticipant[]>([]);
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isUnauthorized, setIsUnauthorized] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);
  const unauthorizedRef = useRef(false);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setIsUnauthorized(false);

    try {
      const [pRes, sRes, stRes] = await Promise.all([
        fetch(resolveUrl("/api/study/admin/participants"), { credentials: "include" }),
        fetch(resolveUrl("/api/study/admin/sessions"), { credentials: "include" }),
        fetch(resolveUrl("/api/study/admin/stats"), { credentials: "include" }),
      ]);

      if (pRes.status === 401 || sRes.status === 401 || stRes.status === 401) {
        setIsUnauthorized(true);
        unauthorizedRef.current = true;
        return;
      }

      if (!pRes.ok) {
        throw new Error(`Participants fetch failed: ${pRes.status}`);
      }
      if (!sRes.ok) {
        throw new Error(`Sessions fetch failed: ${sRes.status}`);
      }
      if (!stRes.ok) {
        throw new Error(`Stats fetch failed: ${stRes.status}`);
      }

      const pData: Omit<AdminParticipant, "sessions">[] = await pRes.json();
      const sData: AdminSession[] = await sRes.json();
      const stData: AdminStats = await stRes.json();
      setStats(stData);

      // Merge sessions into participants
      const sessionsByCode: Record<string, AdminSession[]> = {};
      for (const s of sData) {
        if (!sessionsByCode[s.participantCode]) {
          sessionsByCode[s.participantCode] = [];
        }
        sessionsByCode[s.participantCode].push(s);
      }

      const merged: AdminParticipant[] = pData.map((p) => ({
        ...p,
        sessions: sessionsByCode[p.participantCode] ?? [],
      }));

      setParticipants(merged);
      setLastRefreshed(new Date());
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchData();
    const interval = setInterval(() => {
      if (!unauthorizedRef.current) void fetchData();
    }, 15_000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // ── Derived stats ────────────────────────────────────────────────────────────

  const allSessions = participants.flatMap((p) => p.sessions);
  const stressCount = allSessions.filter((s) => s.blockType === "stress").length;
  const foodCount = allSessions.filter((s) => s.blockType === "food").length;
  const completeSessions = allSessions.filter((s) => sessionStatus(s) === "complete").length;
  const partialSessions = allSessions.filter((s) => sessionStatus(s) === "partial").length;
  const activeSessions = allSessions.filter((s) => sessionStatus(s) === "recording").length;
  const completionRate = allSessions.length > 0 ? Math.round((completeSessions / allSessions.length) * 100) : 0;
  const bothDone = participants.filter((p) => {
    const types = new Set(p.sessions.filter((s) => sessionStatus(s) === "complete").map((s) => s.blockType));
    return types.has("stress") && types.has("food");
  }).length;

  // ── Unauthorized ───────────────────────────────────────────────────────────

  if (isUnauthorized) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center space-y-4 max-w-sm mx-auto px-4">
          <Lock className="h-10 w-10 text-muted-foreground mx-auto" />
          <h1 className="text-xl font-semibold">Access Restricted</h1>
          <p className="text-sm text-muted-foreground">
            You need to be logged in to view the admin dashboard.
          </p>
          <Button onClick={() => navigate("/auth")}>Log In</Button>
        </div>
      </div>
    );
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-5xl mx-auto px-4 py-10 space-y-6">

        {/* Header */}
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="space-y-1">
            <h1 className="text-2xl font-bold">Study Admin</h1>
            <p className="text-sm text-muted-foreground">
              AntarAI pilot study — participant data
            </p>
          </div>
          <div className="flex items-center gap-2">
            {lastRefreshed && (
              <span className="text-xs text-muted-foreground hidden sm:inline">
                Updated {lastRefreshed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                {" "}· auto-refresh 15s
              </span>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={fetchData}
              disabled={isLoading}
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              <span className="ml-1.5">Refresh</span>
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => window.open("/api/study/admin/export-csv")}
            >
              <Download className="h-4 w-4 mr-1.5" />
              Export CSV
            </Button>
          </div>
        </div>

        {/* Real-time tracker */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Card>
            <CardContent className="pt-4 pb-4 text-center">
              <Users className="h-4 w-4 text-primary mx-auto mb-1" />
              <p className="text-2xl font-bold tabular-nums">{participants.length}</p>
              <p className="text-xs text-muted-foreground mt-0.5">enrolled</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 pb-4 text-center">
              <CheckCircle2 className="h-4 w-4 text-cyan-400 mx-auto mb-1" />
              <p className="text-2xl font-bold tabular-nums text-cyan-400">{bothDone}</p>
              <p className="text-xs text-muted-foreground mt-0.5">both sessions done</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 pb-4 text-center">
              <Activity className="h-4 w-4 text-yellow-400 mx-auto mb-1" />
              <p className="text-2xl font-bold tabular-nums text-yellow-400">{activeSessions}</p>
              <p className="text-xs text-muted-foreground mt-0.5">recording now</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 pb-4 text-center">
              <p className="text-2xl font-bold tabular-nums">{completionRate}%</p>
              <p className="text-xs text-muted-foreground mt-0.5">completion rate</p>
            </CardContent>
          </Card>
        </div>

        {/* Session breakdown */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Card className="border-orange-500/20">
            <CardContent className="pt-3 pb-3 text-center">
              <p className="text-lg font-bold tabular-nums text-orange-400">{stressCount}</p>
              <p className="text-xs text-muted-foreground">stress sessions</p>
            </CardContent>
          </Card>
          <Card className="border-indigo-500/20">
            <CardContent className="pt-3 pb-3 text-center">
              <p className="text-lg font-bold tabular-nums text-indigo-400">{foodCount}</p>
              <p className="text-xs text-muted-foreground">food sessions</p>
            </CardContent>
          </Card>
          <Card className="border-cyan-500/20">
            <CardContent className="pt-3 pb-3 text-center">
              <p className="text-lg font-bold tabular-nums text-cyan-400">{completeSessions}</p>
              <p className="text-xs text-muted-foreground">completed</p>
            </CardContent>
          </Card>
          <Card className="border-amber-500/20">
            <CardContent className="pt-3 pb-3 text-center">
              <div className="flex items-center justify-center gap-1">
                <p className="text-lg font-bold tabular-nums text-amber-400">{partialSessions}</p>
                {partialSessions > 0 && <AlertCircle className="h-3.5 w-3.5 text-amber-400" />}
              </div>
              <p className="text-xs text-muted-foreground">partial/dropped</p>
            </CardContent>
          </Card>
        </div>

        {/* Aggregate stats */}
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Card>
              <CardContent className="pt-3 pb-3 text-center">
                <p className={`text-lg font-bold tabular-nums ${
                  Math.round(stats.avg_quality_score) >= 70
                    ? "text-cyan-400"
                    : Math.round(stats.avg_quality_score) >= 40
                      ? "text-yellow-400"
                      : "text-rose-400"
                }`}>
                  {Math.round(stats.avg_quality_score)}
                </p>
                <p className="text-xs text-muted-foreground">avg quality</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-3 pb-3 text-center">
                <p className="text-lg font-bold tabular-nums">
                  {formatDuration(Math.round(stats.avg_duration_seconds))}
                </p>
                <p className="text-xs text-muted-foreground">avg duration</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-3 pb-3 text-center">
                <p className={`text-lg font-bold tabular-nums ${
                  stats.avg_stress_reduction > 0 ? "text-cyan-400" : "text-muted-foreground"
                }`}>
                  {stats.avg_stress_reduction > 0 ? "" : ""}
                  {Math.round(stats.avg_stress_reduction * 100)}%
                </p>
                <p className="text-xs text-muted-foreground">stress reduction</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-3 pb-3 text-center">
                <p className="text-lg font-bold tabular-nums">
                  <span className="text-cyan-400">{stats.complete_sessions}</span>
                  <span className="text-muted-foreground"> / </span>
                  <span className="text-amber-400">{stats.partial_sessions}</span>
                </p>
                <p className="text-xs text-muted-foreground">complete / partial</p>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Error */}
        {error && (
          <Card className="border-destructive/50 bg-destructive/5">
            <CardContent className="pt-4 pb-4">
              <p className="text-sm text-destructive">{error}</p>
            </CardContent>
          </Card>
        )}

        {/* Table */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Participants</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {isLoading && participants.length === 0 ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : participants.length === 0 ? (
              <div className="text-center py-12 text-sm text-muted-foreground">
                No participants enrolled yet.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="whitespace-nowrap">Code</TableHead>
                      <TableHead className="whitespace-nowrap">Age</TableHead>
                      <TableHead className="whitespace-nowrap">Diet</TableHead>
                      <TableHead className="whitespace-nowrap">Apple Watch</TableHead>
                      <TableHead className="whitespace-nowrap">Sessions Done</TableHead>
                      <TableHead className="whitespace-nowrap">Joined</TableHead>
                      <TableHead className="w-8" />
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {participants.map((p) => (
                      <ParticipantRow
                        key={p.participantCode}
                        participant={p}
                        onDeleteSession={(sessionId) =>
                          setParticipants((prev) =>
                            prev.map((pt) => ({
                              ...pt,
                              sessions: pt.sessions.filter((s) => s.id !== sessionId),
                            }))
                          )
                        }
                      />
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
