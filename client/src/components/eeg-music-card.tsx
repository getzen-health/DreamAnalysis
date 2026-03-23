/**
 * eeg-music-card.tsx
 *
 * Compact card showing EEG-adaptive music recommendation.
 * Auto-updates when EEG state changes (subscribes to localStorage events).
 * Falls back to static categories when no EEG data is available.
 */

import { useState, useEffect, useCallback } from "react";
import {
  Music, Headphones, Moon, Zap, Brain, Wind,
  type LucideIcon,
} from "lucide-react";
import { recommendMusic, type MusicRecommendation, type EegMusicState } from "@/lib/eeg-music";
import { createBinauralBeat, type BinauralHandle } from "@/lib/binaural";
import { sbGetSetting } from "../lib/supabase-store";

// ── Category icons ───────────────────────────────────────────────────────────

const CATEGORY_ICONS: Record<MusicRecommendation["category"], LucideIcon> = {
  focus: Brain,
  calm: Wind,
  energy: Zap,
  sleep: Moon,
  meditation: Headphones,
};

const CATEGORY_LABELS: Record<MusicRecommendation["category"], string> = {
  focus: "Focus",
  calm: "Calm",
  energy: "Energy",
  sleep: "Sleep",
  meditation: "Meditation",
};

// ── Static fallback categories (when no EEG data) ───────────────────────────

const STATIC_CATEGORIES: {
  icon: LucideIcon;
  title: string;
  color: string;
  url: string;
}[] = [
  { icon: Brain, title: "Focus", color: "#6366f1", url: "https://open.spotify.com/playlist/37i9dQZF1DWZeKCadgRdKQ" },
  { icon: Wind, title: "Calm", color: "#0891b2", url: "https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO" },
  { icon: Moon, title: "Sleep", color: "#7c3aed", url: "https://open.spotify.com/playlist/37i9dQZF1DWZd79rJ6a7lp" },
  { icon: Zap, title: "Energize", color: "#ea580c", url: "https://open.spotify.com/playlist/37i9dQZF1DX76Wlfdnj7AP" },
  { icon: Headphones, title: "Meditate", color: "#a78bfa", url: "https://open.spotify.com/playlist/37i9dQZF1DWYcDQ1hSjOpY" },
];

// ── Hook: read EEG state from localStorage ──────────────────────────────────

function useEegMusicState(): EegMusicState | null {
  const [state, setState] = useState<EegMusicState | null>(null);

  useEffect(() => {
    const read = (): EegMusicState | null => {
      try {
        const raw = sbGetSetting("ndw_last_emotion");
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        const result = parsed?.result ?? parsed;
        if (!result?.emotion) return null;

        // Determine dominant band from band powers if available,
        // otherwise infer from stress/focus/arousal
        let dominantBand = "alpha";
        if (result.band_powers) {
          const bands = result.band_powers as Record<string, number>;
          let maxPower = 0;
          for (const [band, power] of Object.entries(bands)) {
            if (power > maxPower) {
              maxPower = power;
              dominantBand = band;
            }
          }
        } else {
          // Heuristic inference when band_powers not available
          const stress = result.stress_index ?? 0;
          const focus = result.focus_index ?? 0;
          const relaxation = result.relaxation_index ?? (1 - stress);
          if (relaxation > 0.6 && stress < 0.3) dominantBand = "alpha";
          else if (stress > 0.6) dominantBand = "beta";
          else if (relaxation > 0.7 && focus < 0.3) dominantBand = "theta";
        }

        return {
          dominantBand,
          arousal: result.arousal ?? (result.stress_index ?? 0.5),
          stress: result.stress_index ?? 0,
          focus: result.focus_index ?? 0.5,
          emotion: result.emotion ?? "neutral",
        };
      } catch {
        return null;
      }
    };

    setState(read());

    const handler = () => setState(read());
    window.addEventListener("ndw-voice-updated", handler);
    window.addEventListener("ndw-emotion-update", handler);
    window.addEventListener("storage", handler);
    return () => {
      window.removeEventListener("ndw-voice-updated", handler);
      window.removeEventListener("ndw-emotion-update", handler);
      window.removeEventListener("storage", handler);
    };
  }, []);

  return state;
}

// ── Main component ──────────────────────────────────────────────────────────

export function EegMusicCard() {
  const eegState = useEegMusicState();
  const [binauralActive, setBinauralActive] = useState(false);
  const [binauralHandle, setBinauralHandle] = useState<BinauralHandle | null>(null);

  // Clean up binaural on unmount
  useEffect(() => {
    return () => {
      if (binauralHandle) {
        binauralHandle.stop();
      }
    };
  }, [binauralHandle]);

  const recommendation = eegState ? recommendMusic(eegState) : null;

  const handleListen = useCallback((query: string) => {
    const url = `https://open.spotify.com/search/${encodeURIComponent(query)}`;
    window.open(url, "_blank", "noopener");
  }, []);

  const toggleBinaural = useCallback((freq: number) => {
    if (binauralActive && binauralHandle) {
      binauralHandle.stop();
      setBinauralHandle(null);
      setBinauralActive(false);
    } else {
      try {
        const ctx = new AudioContext();
        const handle = createBinauralBeat(200, freq, ctx);
        handle.start();
        setBinauralHandle(handle);
        setBinauralActive(true);
      } catch {
        // AudioContext not available
      }
    }
  }, [binauralActive, binauralHandle]);

  // ── EEG-adaptive mode ─────────────────────────────────────────────────────
  if (recommendation) {
    const Icon = CATEGORY_ICONS[recommendation.category];
    return (
      <div style={{ marginBottom: 16 }}>
        <div style={{
          fontSize: 11, fontWeight: 700, color: "var(--muted-foreground)",
          textTransform: "uppercase" as const, letterSpacing: "0.8px", marginBottom: 8,
          display: "flex", alignItems: "center", gap: 6,
        }}>
          <Music style={{ width: 12, height: 12 }} />
          EEG Music
        </div>

        <div style={{
          background: `linear-gradient(135deg, var(--card) 0%, ${recommendation.colorAccent}08 100%)`,
          border: `1px solid ${recommendation.colorAccent}20`,
          borderLeft: `3px solid ${recommendation.colorAccent}`,
          borderRadius: 20,
          padding: "14px 16px",
          boxShadow: "0 2px 16px rgba(0,0,0,0.06), 0 0 0 0.5px rgba(255,255,255,0.04)",
        }}>
          {/* Header: icon + category */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <Icon style={{ width: 20, height: 20, color: recommendation.colorAccent }} />
            <span style={{ fontSize: 13, fontWeight: 600, color: "var(--foreground)" }}>
              {CATEGORY_LABELS[recommendation.category]}
            </span>
          </div>

          {/* Reason text */}
          <p style={{
            fontSize: 11, color: "var(--muted-foreground)", margin: "0 0 10px 0",
            lineHeight: 1.4,
          }}>
            {recommendation.reason}
          </p>

          {/* Action buttons */}
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" as const }}>
            <button
              onClick={() => handleListen(recommendation.spotifyQuery)}
              style={{
                background: recommendation.colorAccent,
                color: "#fff",
                border: "none",
                borderRadius: 12,
                padding: "6px 14px",
                fontSize: 11,
                fontWeight: 600,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: 4,
              }}
            >
              <Music style={{ width: 12, height: 12 }} />
              Listen
            </button>

            {recommendation.binauralFreq && (
              <button
                onClick={() => toggleBinaural(recommendation.binauralFreq!)}
                style={{
                  background: binauralActive ? recommendation.colorAccent + "30" : "var(--card)",
                  color: binauralActive ? recommendation.colorAccent : "var(--muted-foreground)",
                  border: `1px solid ${recommendation.colorAccent}30`,
                  borderRadius: 12,
                  padding: "6px 14px",
                  fontSize: 11,
                  fontWeight: 500,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                }}
              >
                <Headphones style={{ width: 12, height: 12 }} />
                {binauralActive ? "Stop" : `${recommendation.binauralFreq} Hz binaural`}
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ── Static fallback — no EEG data ─────────────────────────────────────────
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{
        fontSize: 11, fontWeight: 700, color: "var(--muted-foreground)",
        textTransform: "uppercase" as const, letterSpacing: "0.8px", marginBottom: 8,
        display: "flex", alignItems: "center", gap: 6,
      }}>
        <Music style={{ width: 12, height: 12 }} />
        Quick Listen
      </div>
      <div style={{ display: "flex", gap: 8, overflowX: "auto" as const, paddingBottom: 4 }}>
        {STATIC_CATEGORIES.map((cat) => (
          <a
            key={cat.title}
            href={cat.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              background: `linear-gradient(135deg, var(--card) 0%, ${cat.color}06 100%)`,
              border: `1px solid ${cat.color}20`,
              borderRadius: 16,
              padding: "10px 14px",
              minWidth: 80,
              flexShrink: 0,
              textAlign: "center" as const,
              textDecoration: "none",
              cursor: "pointer",
              display: "flex",
              flexDirection: "column" as const,
              alignItems: "center",
              gap: 4,
            }}
          >
            <cat.icon style={{ width: 20, height: 20, color: cat.color }} />
            <span style={{ fontSize: 10, fontWeight: 600, color: "var(--foreground)" }}>
              {cat.title}
            </span>
          </a>
        ))}
      </div>
    </div>
  );
}
