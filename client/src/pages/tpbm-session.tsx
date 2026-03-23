/**
 * tpbm-session.tsx — Transcranial Photobiomodulation (tPBM) session tracker.
 *
 * Logs tPBM sessions with:
 *   - Session date and duration
 *   - Device used (e.g., Vielight Neuro Gamma, Joovv Go, custom)
 *   - Pre/post mood rating (1-10 scale)
 *   - Optional gamma power comparison if EEG data is available
 *
 * Data stored in localStorage (key: ndw_tpbm_sessions).
 */

import { useState, useEffect, useMemo } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Lightbulb,
  Clock,
  Smile,
  Plus,
  Brain,
  TrendingUp,
  TrendingDown,
  Minus,
} from "lucide-react";

// ── Types ──────────────────────────────────────────────────────────────────

interface TPBMSession {
  id: string;
  date: string; // ISO date string
  durationMinutes: number;
  device: string;
  preMood: number; // 1-10
  postMood: number; // 1-10
  gammaPowerPre?: number; // optional EEG gamma power before
  gammaPowerPost?: number; // optional EEG gamma power after
  notes?: string;
}

// ── Constants ──────────────────────────────────────────────────────────────

const STORAGE_KEY = "ndw_tpbm_sessions";
const MAX_SESSIONS = 200;

const COMMON_DEVICES = [
  "Vielight Neuro Gamma",
  "Vielight Neuro Alpha",
  "Joovv Go",
  "Joovv Mini",
  "Custom / DIY",
  "Other",
];

// ── localStorage helpers ───────────────────────────────────────────────────

function loadSessions(): TPBMSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as TPBMSession[];
  } catch {
    return [];
  }
}

function saveSessions(sessions: TPBMSession[]): void {
  try {
    const capped = sessions.slice(0, MAX_SESSIONS);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(capped));
  } catch {
    // localStorage full or unavailable
  }
}

// ── Mood emoji helper ──────────────────────────────────────────────────────

function moodLabel(value: number): string {
  if (value <= 2) return "Very low";
  if (value <= 4) return "Low";
  if (value <= 6) return "Moderate";
  if (value <= 8) return "Good";
  return "Excellent";
}

// ── Read gamma from latest EEG frame in localStorage ───────────────────────

function getLatestGammaPower(): number | null {
  try {
    const raw = localStorage.getItem("ndw_last_eeg_emotion");
    if (!raw) return null;
    const data = JSON.parse(raw);
    // gamma_power may be stored as part of band powers
    const gamma = data?.gamma_power ?? data?.gamma ?? data?.band_powers?.gamma ?? null;
    if (typeof gamma === "number" && isFinite(gamma)) return gamma;
    return null;
  } catch {
    return null;
  }
}

// ── Component ──────────────────────────────────────────────────────────────

export default function TPBMSessionPage() {
  const [sessions, setSessions] = useState<TPBMSession[]>([]);
  const [showForm, setShowForm] = useState(false);

  // Form state
  const [duration, setDuration] = useState("20");
  const [device, setDevice] = useState(COMMON_DEVICES[0]);
  const [preMood, setPreMood] = useState(5);
  const [postMood, setPostMood] = useState(5);
  const [notes, setNotes] = useState("");

  useEffect(() => {
    setSessions(loadSessions());
  }, []);

  // Stats
  const stats = useMemo(() => {
    if (sessions.length === 0) return null;
    const totalSessions = sessions.length;
    const avgDuration = sessions.reduce((s, x) => s + x.durationMinutes, 0) / totalSessions;
    const avgMoodChange =
      sessions.reduce((s, x) => s + (x.postMood - x.preMood), 0) / totalSessions;
    const sessionsWithGamma = sessions.filter(
      (s) => s.gammaPowerPre != null && s.gammaPowerPost != null,
    );
    const avgGammaChange =
      sessionsWithGamma.length > 0
        ? sessionsWithGamma.reduce(
            (s, x) => s + ((x.gammaPowerPost ?? 0) - (x.gammaPowerPre ?? 0)),
            0,
          ) / sessionsWithGamma.length
        : null;

    return { totalSessions, avgDuration, avgMoodChange, avgGammaChange };
  }, [sessions]);

  function handleSave() {
    const gammaPre = getLatestGammaPower();
    const newSession: TPBMSession = {
      id: `tpbm_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      date: new Date().toISOString(),
      durationMinutes: Math.max(1, parseInt(duration, 10) || 20),
      device,
      preMood,
      postMood,
      gammaPowerPre: gammaPre ?? undefined,
      notes: notes.trim() || undefined,
    };

    const updated = [newSession, ...sessions];
    setSessions(updated);
    saveSessions(updated);
    setShowForm(false);
    setDuration("20");
    setPreMood(5);
    setPostMood(5);
    setNotes("");
  }

  function handleDelete(id: string) {
    const updated = sessions.filter((s) => s.id !== id);
    setSessions(updated);
    saveSessions(updated);
  }

  return (
    <main className="p-4 md:p-6 pb-24 space-y-6" data-testid="tpbm-session-page">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold flex items-center gap-2">
          <Lightbulb className="h-5 w-5 text-amber-400" aria-hidden="true" />
          tPBM Sessions
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Track transcranial photobiomodulation sessions and mood changes
        </p>
      </div>

      {/* Stats Overview */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Card className="p-3">
            <p className="text-xs text-muted-foreground">Total Sessions</p>
            <p className="text-lg font-bold">{stats.totalSessions}</p>
          </Card>
          <Card className="p-3">
            <p className="text-xs text-muted-foreground">Avg Duration</p>
            <p className="text-lg font-bold">{Math.round(stats.avgDuration)} min</p>
          </Card>
          <Card className="p-3">
            <p className="text-xs text-muted-foreground">Avg Mood Change</p>
            <p className="text-lg font-bold flex items-center gap-1">
              {stats.avgMoodChange > 0 ? (
                <TrendingUp className="h-4 w-4 text-green-400" aria-hidden="true" />
              ) : stats.avgMoodChange < 0 ? (
                <TrendingDown className="h-4 w-4 text-red-400" aria-hidden="true" />
              ) : (
                <Minus className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
              )}
              {stats.avgMoodChange >= 0 ? "+" : ""}
              {stats.avgMoodChange.toFixed(1)}
            </p>
          </Card>
          {stats.avgGammaChange !== null && (
            <Card className="p-3">
              <p className="text-xs text-muted-foreground">Avg Gamma Change</p>
              <p className="text-lg font-bold flex items-center gap-1">
                <Brain className="h-4 w-4 text-violet-400" aria-hidden="true" />
                {stats.avgGammaChange >= 0 ? "+" : ""}
                {stats.avgGammaChange.toFixed(3)}
              </p>
            </Card>
          )}
        </div>
      )}

      {/* New Session Form */}
      {showForm ? (
        <Card className="p-5 space-y-4">
          <h2 className="text-base font-semibold">Log New Session</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="tpbm-duration" className="text-sm">
                Duration (minutes)
              </Label>
              <Input
                id="tpbm-duration"
                type="number"
                min={1}
                max={120}
                value={duration}
                onChange={(e) => setDuration(e.target.value)}
                aria-label="Session duration in minutes"
              />
            </div>
            <div>
              <Label htmlFor="tpbm-device" className="text-sm">
                Device
              </Label>
              <Select value={device} onValueChange={setDevice}>
                <SelectTrigger id="tpbm-device" aria-label="Select tPBM device">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {COMMON_DEVICES.map((d) => (
                    <SelectItem key={d} value={d}>
                      {d}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label className="text-sm">Pre-session Mood ({preMood}/10)</Label>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs text-muted-foreground">1</span>
                <input
                  type="range"
                  min={1}
                  max={10}
                  value={preMood}
                  onChange={(e) => setPreMood(parseInt(e.target.value, 10))}
                  className="flex-1"
                  aria-label="Pre-session mood rating"
                />
                <span className="text-xs text-muted-foreground">10</span>
              </div>
              <p className="text-xs text-muted-foreground mt-1">{moodLabel(preMood)}</p>
            </div>
            <div>
              <Label className="text-sm">Post-session Mood ({postMood}/10)</Label>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs text-muted-foreground">1</span>
                <input
                  type="range"
                  min={1}
                  max={10}
                  value={postMood}
                  onChange={(e) => setPostMood(parseInt(e.target.value, 10))}
                  className="flex-1"
                  aria-label="Post-session mood rating"
                />
                <span className="text-xs text-muted-foreground">10</span>
              </div>
              <p className="text-xs text-muted-foreground mt-1">{moodLabel(postMood)}</p>
            </div>
          </div>

          <div>
            <Label htmlFor="tpbm-notes" className="text-sm">
              Notes (optional)
            </Label>
            <Input
              id="tpbm-notes"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="How did the session feel?"
              aria-label="Session notes"
            />
          </div>

          <div className="flex gap-2">
            <Button onClick={handleSave} aria-label="Save tPBM session">
              Save Session
            </Button>
            <Button
              variant="outline"
              onClick={() => setShowForm(false)}
              aria-label="Cancel new session"
            >
              Cancel
            </Button>
          </div>
        </Card>
      ) : (
        <Button
          onClick={() => setShowForm(true)}
          className="w-full"
          aria-label="Log new tPBM session"
        >
          <Plus className="h-4 w-4 mr-2" aria-hidden="true" />
          Log New Session
        </Button>
      )}

      {/* Session History */}
      <div>
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
          Session History
        </h2>
        {sessions.length === 0 ? (
          <Card className="p-6 text-center">
            <Lightbulb className="h-8 w-8 text-muted-foreground mx-auto mb-2" aria-hidden="true" />
            <p className="text-sm text-muted-foreground">
              No tPBM sessions logged yet. Tap above to log your first session.
            </p>
          </Card>
        ) : (
          <div className="space-y-3">
            {sessions.map((session) => {
              const moodChange = session.postMood - session.preMood;
              const hasGamma =
                session.gammaPowerPre != null && session.gammaPowerPost != null;
              const gammaChange = hasGamma
                ? (session.gammaPowerPost ?? 0) - (session.gammaPowerPre ?? 0)
                : null;

              return (
                <Card key={session.id} className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <Clock className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
                        <span className="text-sm font-medium">
                          {new Date(session.date).toLocaleDateString(undefined, {
                            month: "short",
                            day: "numeric",
                            year: "numeric",
                          })}
                        </span>
                        <Badge variant="outline" className="text-xs">
                          {session.durationMinutes} min
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground">{session.device}</p>

                      <div className="flex items-center gap-4 mt-2">
                        <div className="flex items-center gap-1">
                          <Smile className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
                          <span className="text-xs">
                            Mood: {session.preMood} → {session.postMood}
                          </span>
                          <span
                            className={`text-xs font-medium ${
                              moodChange > 0
                                ? "text-green-400"
                                : moodChange < 0
                                ? "text-red-400"
                                : "text-muted-foreground"
                            }`}
                          >
                            ({moodChange >= 0 ? "+" : ""}
                            {moodChange})
                          </span>
                        </div>

                        {gammaChange !== null && (
                          <div className="flex items-center gap-1">
                            <Brain className="h-3.5 w-3.5 text-violet-400" aria-hidden="true" />
                            <span className="text-xs">
                              Gamma: {gammaChange >= 0 ? "+" : ""}
                              {gammaChange.toFixed(3)}
                            </span>
                          </div>
                        )}
                      </div>

                      {session.notes && (
                        <p className="text-xs text-muted-foreground mt-1 italic">
                          {session.notes}
                        </p>
                      )}
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-xs text-muted-foreground hover:text-destructive"
                      onClick={() => handleDelete(session.id)}
                      aria-label={`Delete session from ${new Date(session.date).toLocaleDateString()}`}
                    >
                      Delete
                    </Button>
                  </div>
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
}
