/**
 * SpotifyConnect — shows connection status and provides connect/disconnect.
 * Used in the biofeedback Music tab and Settings.
 *
 * When connected: shows username + "Play calm" / "Play focus" buttons.
 * When disconnected: shows "Connect Spotify" link to OAuth flow.
 * When Spotify not configured server-side: hides silently.
 */

import { useState, useEffect, useCallback } from "react";
import { Music, LogIn, LogOut, Play, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { apiRequest } from "@/lib/queryClient";

interface SpotifyStatus {
  connected: boolean;
  enabled: boolean;
  username: string | null;
}

interface PlayResult {
  ok?: boolean;
  error?: string;
  message?: string;
  authUrl?: string;
}

interface SpotifyConnectProps {
  /** If provided, auto-plays this mood when connection is detected */
  autoPlayMood?: "calm" | "focus" | "sleep";
  compact?: boolean;
}

export default function SpotifyConnect({ autoPlayMood, compact = false }: SpotifyConnectProps) {
  const [status, setStatus] = useState<SpotifyStatus | null>(null);
  const [playing, setPlaying] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await apiRequest("GET", "/api/spotify/status");
      if (res.ok) setStatus(await res.json());
    } catch { /* server offline */ }
  }, []);

  useEffect(() => {
    fetchStatus();

    // Check if we just returned from OAuth (URL contains spotify=connected)
    const params = new URLSearchParams(window.location.search);
    if (params.get("spotify") === "connected") {
      // Clean up URL
      const url = new URL(window.location.href);
      url.searchParams.delete("spotify");
      window.history.replaceState({}, "", url.toString());
      // Auto-play if requested
      if (autoPlayMood) {
        setTimeout(() => playMood(autoPlayMood), 500);
      }
    }
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const playMood = useCallback(async (mood: "calm" | "focus" | "sleep") => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await apiRequest("POST", "/api/spotify/play", { mood });
      const data: PlayResult = await res.json();
      if (data.ok) {
        setPlaying(mood);
        setTimeout(() => setPlaying(null), 5000);
      } else if (data.error === "no_active_device") {
        setError("Open Spotify on any device first.");
      } else if (data.error === "not_connected") {
        setStatus((s) => s ? { ...s, connected: false } : null);
      } else {
        setError(data.message ?? "Playback failed.");
      }
    } catch {
      setError("Could not reach server.");
    }
    setIsLoading(false);
  }, []);

  const disconnect = useCallback(async () => {
    await apiRequest("POST", "/api/spotify/disconnect");
    setStatus((s) => s ? { ...s, connected: false, username: null } : null);
  }, []);

  // Server hasn't responded yet — show nothing
  if (!status) return null;

  // Spotify not configured (no env vars) — hide entirely
  if (!status.enabled) return null;

  if (compact) {
    // Minimal version: just a pill-shaped status indicator
    return status.connected ? (
      <div className="flex items-center gap-1.5 text-xs text-cyan-400">
        <Music className="h-3 w-3" />
        <span>Spotify</span>
      </div>
    ) : null;
  }

  if (!status.connected) {
    return (
      <div className="flex items-center gap-3 p-3 rounded-xl border border-[#1DB954]/20 bg-[#1DB954]/5">
        <Music className="h-4 w-4 text-[#1DB954] shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium">Connect Spotify</p>
          <p className="text-xs text-muted-foreground">
            Auto-play calming music when stress is detected
          </p>
        </div>
        <a href="/api/spotify/auth">
          <Button
            size="sm"
            className="bg-[#1DB954] hover:bg-[#1ed760] text-black font-semibold shrink-0"
          >
            <LogIn className="h-3.5 w-3.5 mr-1.5" />
            Connect
          </Button>
        </a>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-[#1DB954]/20 bg-[#1DB954]/5 p-3 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Music className="h-4 w-4 text-[#1DB954]" />
          <div>
            <p className="text-sm font-medium text-[#1DB954]">Spotify connected</p>
            {status.username && (
              <p className="text-xs text-muted-foreground">{status.username}</p>
            )}
          </div>
        </div>
        <button
          onClick={disconnect}
          className="text-muted-foreground hover:text-foreground transition-colors"
          title="Disconnect Spotify"
        >
          <LogOut className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="flex gap-2">
        <Button
          size="sm"
          variant="outline"
          className="flex-1 border-[#1DB954]/30 hover:bg-[#1DB954]/10"
          onClick={() => playMood("calm")}
          disabled={isLoading}
        >
          <Play className="h-3 w-3 mr-1.5" />
          {playing === "calm" ? "Playing..." : "Calm"}
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="flex-1 border-indigo-500/30 hover:bg-indigo-500/10"
          onClick={() => playMood("focus")}
          disabled={isLoading}
        >
          <Play className="h-3 w-3 mr-1.5" />
          {playing === "focus" ? "Playing..." : "Focus"}
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="flex-1 border-purple-500/30 hover:bg-purple-500/10"
          onClick={() => playMood("sleep")}
          disabled={isLoading}
        >
          <Play className="h-3 w-3 mr-1.5" />
          {playing === "sleep" ? "Playing..." : "Sleep"}
        </Button>
      </div>

      {error && (
        <div className="flex items-center gap-1.5 text-xs text-yellow-500">
          <AlertCircle className="h-3 w-3 shrink-0" />
          {error}
        </div>
      )}
    </div>
  );
}

/**
 * Trigger Spotify playback programmatically (called from intervention system).
 * Silent failure if Spotify is not connected.
 */
export async function triggerSpotifyPlay(mood: "calm" | "focus" | "sleep"): Promise<boolean> {
  try {
    const res = await apiRequest("POST", "/api/spotify/play", { mood });
    const data = await res.json();
    return !!data.ok;
  } catch {
    return false;
  }
}
