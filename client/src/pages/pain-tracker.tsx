/**
 * Pain Tracker — log pain episodes with severity, location, duration.
 * If EEG data is available, shows theta/alpha power changes
 * (elevated theta = pain marker per Scientific Reports 2024).
 *
 * Issue #538
 */

import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import {
  AlertTriangle, Plus, Clock, MapPin, Activity, TrendingUp,
  TrendingDown, Brain, Trash2,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { sbGetSetting, sbSaveGeneric } from "../lib/supabase-store";

// ── Types ──────────────────────────────────────────────────────────────────

interface PainEpisode {
  id: string;
  severity: number;        // 1-10
  location: string;
  durationMinutes: number;
  timestamp: number;
  eegData?: {
    thetaPower: number;
    alphaPower: number;
    thetaAlphaRatio: number;
  };
}

const PAIN_LOCATIONS = [
  "Head - Frontal",
  "Head - Temporal",
  "Head - Occipital",
  "Head - Full",
  "Neck",
  "Back - Upper",
  "Back - Lower",
  "Shoulders",
  "Other",
];

const STORAGE_KEY = "ndw_pain_episodes";

// ── Helpers ────────────────────────────────────────────────────────────────

function loadEpisodes(): PainEpisode[] {
  try {
    const raw = sbGetSetting(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveEpisodes(episodes: PainEpisode[]): void {
  sbSaveGeneric(STORAGE_KEY, episodes);
}

function getSeverityColor(severity: number): string {
  if (severity <= 3) return "#22c55e";
  if (severity <= 6) return "#d4a017";
  return "#f43f5e";
}

function getSeverityLabel(severity: number): string {
  if (severity <= 3) return "Mild";
  if (severity <= 6) return "Moderate";
  return "Severe";
}

function formatTimeAgo(timestamp: number): string {
  const diff = Date.now() - timestamp;
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function getEegPainMarkers(): PainEpisode["eegData"] | undefined {
  try {
    const raw = sbGetSetting("ndw_last_eeg_features");
    if (!raw) return undefined;
    const data = JSON.parse(raw);
    const theta = data.theta_power ?? data.theta ?? 0;
    const alpha = data.alpha_power ?? data.alpha ?? 0;
    if (theta === 0 && alpha === 0) return undefined;
    return {
      thetaPower: theta,
      alphaPower: alpha,
      thetaAlphaRatio: alpha > 0 ? theta / alpha : 0,
    };
  } catch {
    return undefined;
  }
}

// ── Component ──────────────────────────────────────────────────────────────

export default function PainTracker() {
  const [episodes, setEpisodes] = useState<PainEpisode[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [severity, setSeverity] = useState(5);
  const [location, setLocation] = useState(PAIN_LOCATIONS[0]);
  const [duration, setDuration] = useState(30);

  useEffect(() => {
    setEpisodes(loadEpisodes());
  }, []);

  const handleAdd = useCallback(() => {
    const episode: PainEpisode = {
      id: `pain_${Date.now()}`,
      severity,
      location,
      durationMinutes: duration,
      timestamp: Date.now(),
      eegData: getEegPainMarkers(),
    };
    const updated = [episode, ...episodes].slice(0, 100); // Keep last 100
    setEpisodes(updated);
    saveEpisodes(updated);
    setShowForm(false);
    setSeverity(5);
    setDuration(30);
  }, [severity, location, duration, episodes]);

  const handleDelete = useCallback((id: string) => {
    const updated = episodes.filter((e) => e.id !== id);
    setEpisodes(updated);
    saveEpisodes(updated);
  }, [episodes]);

  // Average severity over last 7 days
  const weekAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
  const recentEpisodes = episodes.filter((e) => e.timestamp > weekAgo);
  const avgSeverity = recentEpisodes.length > 0
    ? recentEpisodes.reduce((s, e) => s + e.severity, 0) / recentEpisodes.length
    : 0;

  // EEG pattern analysis: compare theta power in pain vs non-pain episodes
  const withEeg = episodes.filter((e) => e.eegData);
  const avgTheta = withEeg.length > 0
    ? withEeg.reduce((s, e) => s + (e.eegData?.thetaPower ?? 0), 0) / withEeg.length
    : null;

  return (
    <motion.div
      initial={pageTransition.initial}
      animate={pageTransition.animate}
      transition={pageTransition.transition}
      className="max-w-2xl mx-auto px-4 py-6 pb-24"
    >
      {/* Header */}
      <div className="flex items-center gap-3 mb-5">
        <div className="w-10 h-10 rounded-xl bg-red-500/10 flex items-center justify-center">
          <AlertTriangle className="w-5 h-5 text-red-500" />
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-tight text-foreground">
            Pain Tracker
          </h1>
          <p className="text-sm text-muted-foreground">
            Track pain episodes and EEG patterns
          </p>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 gap-3 mb-4" data-testid="pain-summary">
        <Card>
          <CardContent className="p-4">
            <div className="text-xs text-muted-foreground mb-1">This Week</div>
            <div className="text-2xl font-bold">{recentEpisodes.length}</div>
            <div className="text-xs text-muted-foreground">episodes</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-xs text-muted-foreground mb-1">Avg Severity</div>
            <div
              className="text-2xl font-bold"
              style={{ color: avgSeverity > 0 ? getSeverityColor(avgSeverity) : undefined }}
            >
              {avgSeverity > 0 ? avgSeverity.toFixed(1) : "---"}
            </div>
            <div className="text-xs text-muted-foreground">
              {avgSeverity > 0 ? getSeverityLabel(Math.round(avgSeverity)) : "No data"}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* EEG Pain Markers */}
      {avgTheta !== null && (
        <Card className="mb-4" data-testid="eeg-pain-markers">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Brain className="w-4 h-4 text-indigo-500" />
              EEG Pain Patterns
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground mb-2">
              Elevated frontal theta during pain episodes is a known biomarker.
            </p>
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-1">
                <TrendingUp className="w-3 h-3 text-violet-500" />
                <span className="text-muted-foreground">Avg Theta:</span>
                <span className="font-medium">{avgTheta.toFixed(3)}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Log Button */}
      {!showForm && (
        <Button
          onClick={() => setShowForm(true)}
          className="w-full mb-4"
          data-testid="log-pain-button"
        >
          <Plus className="w-4 h-4 mr-2" />
          Log Pain Episode
        </Button>
      )}

      {/* Log Form */}
      {showForm && (
        <Card className="mb-4" data-testid="pain-form">
          <CardContent className="p-4 space-y-4">
            <div>
              <Label htmlFor="severity">Severity (1-10)</Label>
              <div className="flex items-center gap-3 mt-1">
                <Input
                  id="severity"
                  type="range"
                  min={1}
                  max={10}
                  value={severity}
                  onChange={(e) => setSeverity(Number(e.target.value))}
                  className="flex-1"
                />
                <span
                  className="text-lg font-bold w-8 text-center"
                  style={{ color: getSeverityColor(severity) }}
                >
                  {severity}
                </span>
              </div>
            </div>

            <div>
              <Label htmlFor="location">Location</Label>
              <Select value={location} onValueChange={setLocation}>
                <SelectTrigger id="location" className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PAIN_LOCATIONS.map((loc) => (
                    <SelectItem key={loc} value={loc}>{loc}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="duration">Duration (minutes)</Label>
              <Input
                id="duration"
                type="number"
                min={1}
                max={1440}
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
                className="mt-1"
              />
            </div>

            <div className="flex gap-2">
              <Button onClick={handleAdd} className="flex-1">Save</Button>
              <Button
                variant="outline"
                onClick={() => setShowForm(false)}
                className="flex-1"
              >
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* History */}
      <h2 className="text-sm font-semibold text-foreground mb-2">
        Pain History
      </h2>

      {episodes.length === 0 ? (
        <Card>
          <CardContent className="p-6 text-center text-muted-foreground text-sm" data-testid="no-episodes">
            No pain episodes logged
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-2" data-testid="pain-history">
          {episodes.slice(0, 20).map((ep) => (
            <Card key={ep.id}>
              <CardContent className="p-3 flex items-center gap-3">
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold text-white"
                  style={{ backgroundColor: getSeverityColor(ep.severity) }}
                >
                  {ep.severity}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">
                    <MapPin className="w-3 h-3 inline mr-1" />
                    {ep.location}
                  </div>
                  <div className="text-xs text-muted-foreground flex items-center gap-2">
                    <span><Clock className="w-3 h-3 inline mr-0.5" />{ep.durationMinutes}min</span>
                    <span>{formatTimeAgo(ep.timestamp)}</span>
                  </div>
                </div>
                {ep.eegData && (
                  <div className="text-xs text-muted-foreground flex items-center gap-1">
                    <Activity className="w-3 h-3 text-violet-500" />
                    <span>{ep.eegData.thetaAlphaRatio.toFixed(2)}</span>
                  </div>
                )}
                <button
                  onClick={() => handleDelete(ep.id)}
                  className="p-1 rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive"
                  aria-label="Delete episode"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </motion.div>
  );
}
