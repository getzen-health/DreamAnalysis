import { useState, useEffect, useCallback } from "react";
import { useLocation } from "wouter";
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
import { Loader2, RefreshCw, Download, ChevronDown, ChevronUp, Lock } from "lucide-react";

// ── Types ────────────────────────────────────────────────────────────────────

interface SurveyScore {
  label: string;
  value: number | string;
}

interface AdminSession {
  id: number;
  block_type: "stress" | "food";
  intervention_triggered: boolean;
  survey_json: Record<string, unknown> | null;
  created_at: string;
}

interface AdminParticipant {
  participant_code: string;
  age: number | null;
  diet_type: string | null;
  has_apple_watch: boolean;
  joined_at: string;
  sessions: AdminSession[];
}

// ── Survey score extractor ────────────────────────────────────────────────────

function extractScores(session: AdminSession): SurveyScore[] {
  if (!session.survey_json) return [];
  const s = session.survey_json;
  const scores: SurveyScore[] = [];

  if (session.block_type === "food") {
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

function ParticipantRow({ participant }: { participant: AdminParticipant }) {
  const [expanded, setExpanded] = useState(false);

  const stressSessions = participant.sessions.filter((s) => s.block_type === "stress");
  const foodSessions = participant.sessions.filter((s) => s.block_type === "food");
  const totalSessions = participant.sessions.length;

  return (
    <>
      <TableRow
        className="cursor-pointer hover:bg-muted/40 transition-colors"
        onClick={() => setExpanded((e) => !e)}
      >
        <TableCell className="font-mono font-semibold text-primary whitespace-nowrap">
          {participant.participant_code}
        </TableCell>
        <TableCell className="tabular-nums text-sm">
          {participant.age ?? "—"}
        </TableCell>
        <TableCell className="text-sm capitalize">
          {participant.diet_type ?? "—"}
        </TableCell>
        <TableCell>
          {participant.has_apple_watch ? (
            <Badge variant="outline" className="border-blue-500/50 text-blue-400 text-xs">Yes</Badge>
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
          {new Date(participant.joined_at).toLocaleDateString()}
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
                    <div className="flex items-center gap-2 flex-wrap">
                      <Badge
                        variant="outline"
                        className={
                          session.block_type === "stress"
                            ? "border-orange-500/50 text-orange-400 text-xs capitalize"
                            : "border-blue-500/50 text-blue-400 text-xs capitalize"
                        }
                      >
                        {session.block_type}
                      </Badge>
                      <Badge
                        variant="outline"
                        className={
                          session.intervention_triggered
                            ? "border-red-500/50 text-red-400 text-xs"
                            : "border-muted text-muted-foreground text-xs"
                        }
                      >
                        Intervention: {session.intervention_triggered ? "yes" : "no"}
                      </Badge>
                      <span className="text-xs text-muted-foreground ml-auto">
                        {new Date(session.created_at).toLocaleString()}
                      </span>
                    </div>

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
            </div>
          </TableCell>
        </TableRow>
      )}

      {expanded && participant.sessions.length === 0 && (
        <TableRow className="bg-muted/20">
          <TableCell colSpan={7} className="py-3 text-center text-xs text-muted-foreground">
            No sessions recorded yet.
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
  const [isLoading, setIsLoading] = useState(false);
  const [isUnauthorized, setIsUnauthorized] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setIsUnauthorized(false);

    try {
      const [pRes, sRes] = await Promise.all([
        fetch("/api/study/admin/participants", { credentials: "include" }),
        fetch("/api/study/admin/sessions", { credentials: "include" }),
      ]);

      if (pRes.status === 401 || sRes.status === 401) {
        setIsUnauthorized(true);
        return;
      }

      if (!pRes.ok) {
        throw new Error(`Participants fetch failed: ${pRes.status}`);
      }
      if (!sRes.ok) {
        throw new Error(`Sessions fetch failed: ${sRes.status}`);
      }

      const pData: Omit<AdminParticipant, "sessions">[] = await pRes.json();
      const sData: (AdminSession & { participant_code: string })[] = await sRes.json();

      // Merge sessions into participants
      const sessionsByCode: Record<string, AdminSession[]> = {};
      for (const s of sData) {
        if (!sessionsByCode[s.participant_code]) {
          sessionsByCode[s.participant_code] = [];
        }
        sessionsByCode[s.participant_code].push(s);
      }

      const merged: AdminParticipant[] = pData.map((p) => ({
        ...p,
        sessions: sessionsByCode[p.participant_code] ?? [],
      }));

      setParticipants(merged);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchData();
  }, [fetchData]);

  // ── Derived stats ────────────────────────────────────────────────────────────

  const allSessions = participants.flatMap((p) => p.sessions);
  const stressCount = allSessions.filter((s) => s.block_type === "stress").length;
  const foodCount = allSessions.filter((s) => s.block_type === "food").length;

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
              Neural Dream Workshop pilot study — participant data
            </p>
          </div>
          <div className="flex items-center gap-2">
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

        {/* Summary row */}
        <div className="grid grid-cols-3 gap-3">
          <Card>
            <CardContent className="pt-4 pb-4 text-center">
              <p className="text-2xl font-bold tabular-nums">{participants.length}</p>
              <p className="text-xs text-muted-foreground mt-0.5">participants enrolled</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 pb-4 text-center">
              <p className="text-2xl font-bold tabular-nums text-orange-400">{stressCount}</p>
              <p className="text-xs text-muted-foreground mt-0.5">stress sessions</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 pb-4 text-center">
              <p className="text-2xl font-bold tabular-nums text-blue-400">{foodCount}</p>
              <p className="text-xs text-muted-foreground mt-0.5">food sessions</p>
            </CardContent>
          </Card>
        </div>

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
                      <ParticipantRow key={p.participant_code} participant={p} />
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
