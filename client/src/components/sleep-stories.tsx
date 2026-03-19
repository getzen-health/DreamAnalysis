/**
 * sleep-stories.tsx → sleep-music.tsx (component kept at same path for lazy import)
 *
 * Curated sleep music playlists from Spotify and YouTube.
 * Opens external links in new tabs — no built-in audio player.
 *
 * Used as the content of the /sleep-music route.
 */

import { Moon, Music, ExternalLink } from "lucide-react";
import { Card } from "@/components/ui/card";

// ─── Playlist data ──────────────────────────────────────────────────────────

interface Playlist {
  id: string;
  title: string;
  description: string;
  url: string;
  source: "spotify" | "youtube";
  gradient: string;
}

const PLAYLISTS: Playlist[] = [
  {
    id: "deep-sleep",
    title: "Deep Sleep",
    description: "Ambient drones and slow pads for deep rest",
    url: "https://open.spotify.com/playlist/37i9dQZF1DWZd79rJ6a7lp",
    source: "spotify",
    gradient: "from-indigo-600/30 to-blue-900/30",
  },
  {
    id: "sleep-sounds",
    title: "Sleep Sounds",
    description: "White noise, fans, and soft textures",
    url: "https://open.spotify.com/playlist/37i9dQZF1DWYcDQ1hSjOpY",
    source: "spotify",
    gradient: "from-violet-600/30 to-indigo-900/30",
  },
  {
    id: "peaceful-piano",
    title: "Peaceful Piano",
    description: "Gentle piano pieces for winding down",
    url: "https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO",
    source: "spotify",
    gradient: "from-cyan-600/30 to-blue-900/30",
  },
  {
    id: "nature-sounds",
    title: "Nature Sounds",
    description: "Rain, ocean, forest, and birdsong",
    url: "https://open.spotify.com/playlist/37i9dQZF1DX4PP3DA4J0N8",
    source: "spotify",
    gradient: "from-emerald-600/30 to-teal-900/30",
  },
  {
    id: "yt-sleep-music",
    title: "8 Hour Sleep Music",
    description: "Long-play ambient tracks on YouTube",
    url: "https://www.youtube.com/results?search_query=8+hour+sleep+music",
    source: "youtube",
    gradient: "from-red-600/30 to-rose-900/30",
  },
  {
    id: "yt-rain",
    title: "Rain Sounds",
    description: "Hours of rain for sleeping",
    url: "https://www.youtube.com/results?search_query=rain+sounds+for+sleeping",
    source: "youtube",
    gradient: "from-slate-600/30 to-gray-900/30",
  },
];

// ─── Playlist card ──────────────────────────────────────────────────────────

function PlaylistCard({ playlist }: { playlist: Playlist }) {
  return (
    <a
      href={playlist.url}
      target="_blank"
      rel="noopener noreferrer"
      className="block group"
    >
      <Card
        className={`glass-card overflow-hidden transition-all hover:border-primary/30 hover:scale-[1.02]`}
      >
        {/* Gradient header */}
        <div
          className={`h-24 bg-gradient-to-br ${playlist.gradient} flex items-center justify-center`}
        >
          <Music className="h-8 w-8 text-white/70 group-hover:text-white/90 transition-colors" />
        </div>

        {/* Info */}
        <div className="p-4">
          <div className="flex items-center justify-between mb-1">
            <p className="text-sm font-medium truncate">{playlist.title}</p>
            <ExternalLink className="h-3.5 w-3.5 text-muted-foreground shrink-0 ml-2" />
          </div>
          <p className="text-[11px] text-muted-foreground line-clamp-2">
            {playlist.description}
          </p>
          <span
            className={`inline-block mt-2 text-[10px] font-medium px-2 py-0.5 rounded-full ${
              playlist.source === "spotify"
                ? "bg-cyan-500/15 text-cyan-400"
                : "bg-rose-500/15 text-rose-400"
            }`}
          >
            {playlist.source === "spotify" ? "Spotify" : "YouTube"}
          </span>
        </div>
      </Card>
    </a>
  );
}

// ─── Main component ─────────────────────────────────────────────────────────

export default function SleepMusic() {
  return (
    <main className="p-4 md:p-6 space-y-6 max-w-3xl mx-auto">
      {/* Page header */}
      <div className="flex items-center gap-3">
        <Moon className="h-6 w-6 text-primary" />
        <div>
          <h2 className="text-xl font-semibold">Sleep Music</h2>
          <p className="text-xs text-muted-foreground">
            Curated playlists to help you wind down and drift off
          </p>
        </div>
      </div>

      {/* Playlist grid */}
      <div className="grid grid-cols-2 gap-3">
        {PLAYLISTS.map((playlist) => (
          <PlaylistCard key={playlist.id} playlist={playlist} />
        ))}
      </div>

      {/* Tip */}
      <Card className="glass-card p-4">
        <p className="text-[13px] text-muted-foreground leading-relaxed">
          <span className="font-medium text-foreground">Tip:</span> Listening to
          slow-tempo music (60-80 BPM) before bed can reduce heart rate and
          promote the transition to sleep. Set a sleep timer in Spotify or
          YouTube to auto-stop playback.
        </p>
      </Card>
    </main>
  );
}
