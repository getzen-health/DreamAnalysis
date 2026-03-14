/**
 * sleep-stories.tsx
 *
 * Sleep story library / listing page.
 * Renders a grid of sleep story cards. Selecting one expands an inline
 * SleepStoryPlayer. Includes a "How it works" explainer for EEG auto-fade.
 *
 * Used as the content of the /sleep-stories route.
 */

import { useState } from "react";
import {
  Moon,
  BrainCircuit,
  Play,
  Clock,
  ChevronDown,
  ChevronUp,
  Waves,
  CloudRain,
  Trees,
  Star,
  Droplets,
  Flame,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useDevice } from "@/hooks/use-device";
import { SleepStoryPlayer } from "@/components/sleep-story-player";

// ─── Story catalogue ──────────────────────────────────────────────────────────

import type { AmbientType } from "@/lib/ambient-audio";

interface StoryMeta {
  id: string;
  title: string;
  duration: string; // display string e.g. "32 min"
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  /** Which procedural ambient generator to use (Web Audio API) */
  audioType: AmbientType;
}

const STORIES: StoryMeta[] = [
  {
    id: "ocean-waves",
    title: "Ocean Waves",
    duration: "45 min",
    description: "Slow, rhythmic waves lapping a quiet shore at night.",
    icon: Waves,
    color: "hsl(210, 80%, 55%)",
    audioType: "ocean",
  },
  {
    id: "mountain-rain",
    title: "Mountain Rain",
    duration: "38 min",
    description: "Steady rain on pine trees in an alpine clearing.",
    icon: CloudRain,
    color: "hsl(230, 60%, 58%)",
    audioType: "rain",
  },
  {
    id: "forest-walk",
    title: "Forest Walk",
    duration: "30 min",
    description: "Birdsong, distant wind, and the creak of tall oaks.",
    icon: Trees,
    color: "hsl(152, 60%, 46%)",
    audioType: "forest",
  },
  {
    id: "night-sky",
    title: "Night Sky",
    duration: "50 min",
    description: "Deep silence with occasional distant owl calls and soft wind.",
    icon: Star,
    color: "hsl(260, 65%, 60%)",
    audioType: "night",
  },
  {
    id: "gentle-stream",
    title: "Gentle Stream",
    duration: "35 min",
    description: "A babbling brook flowing over smooth stones.",
    icon: Droplets,
    color: "hsl(190, 70%, 50%)",
    audioType: "stream",
  },
  {
    id: "campfire",
    title: "Campfire",
    duration: "40 min",
    description: "The soft crackle and warmth of a campfire under open stars.",
    icon: Flame,
    color: "hsl(30, 85%, 55%)",
    audioType: "campfire",
  },
];

// ─── Story card ───────────────────────────────────────────────────────────────

interface StoryCardProps {
  story: StoryMeta;
  isSelected: boolean;
  eegConnected: boolean;
  onSelect: (id: string) => void;
  onSleepDetected: (latencyMs: number) => void;
}

function StoryCard({
  story,
  isSelected,
  eegConnected,
  onSelect,
  onSleepDetected,
}: StoryCardProps) {
  const Icon = story.icon;

  return (
    <div className="space-y-2">
      {/* Card header — always visible */}
      <Card
        className={`glass-card p-4 cursor-pointer transition-all hover:border-primary/30 ${
          isSelected ? "border-primary/40 bg-primary/5" : ""
        }`}
        onClick={() => onSelect(story.id)}
      >
        <div className="flex items-center gap-4">
          {/* Icon */}
          <div
            className="w-11 h-11 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: story.color + "18", border: `1px solid ${story.color}30` }}
          >
            <Icon
              className="h-5 w-5"
              style={{ color: story.color } as React.CSSProperties}
            />
          </div>

          {/* Text */}
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate">{story.title}</p>
            <p className="text-[11px] text-muted-foreground mt-0.5 truncate">
              {story.description}
            </p>
          </div>

          {/* Right side */}
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-[11px] text-muted-foreground font-mono">
              {story.duration}
            </span>

            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 rounded-full"
              style={{
                background: story.color + "18",
                color: story.color,
              }}
              aria-label={isSelected ? "Collapse player" : "Open player"}
            >
              {isSelected ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <Play className="h-3.5 w-3.5 ml-0.5" />
              )}
            </Button>
          </div>
        </div>
      </Card>

      {/* Inline player — shown when this story is selected */}
      {isSelected && (
        <SleepStoryPlayer
          audioType={story.audioType}
          title={story.title}
          eegConnected={eegConnected}
          onSleepDetected={onSleepDetected}
        />
      )}
    </div>
  );
}

// ─── How it works explainer ───────────────────────────────────────────────────

function HowItWorks() {
  const [open, setOpen] = useState(false);

  return (
    <Card className="glass-card p-4">
      <button
        className="w-full flex items-center justify-between text-left"
        onClick={() => setOpen((o) => !o)}
      >
        <div className="flex items-center gap-2">
          <BrainCircuit className="h-4 w-4 text-primary" />
          <span className="text-sm font-medium">How EEG auto-fade works</span>
        </div>
        {open ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        )}
      </button>

      {open && (
        <div className="mt-4 space-y-3 text-[13px] text-muted-foreground leading-relaxed">
          <p>
            When your Muse 2 is connected, the ML backend classifies your EEG
            in real time and tracks which sleep stage you're in.
          </p>
          <p>
            The transition from{" "}
            <span className="text-foreground font-medium">N1 (light sleep)</span> to{" "}
            <span className="text-foreground font-medium">N2 (core sleep)</span> is
            the clearest signal that you have fully fallen asleep. N2 sleep is
            characterized by sleep spindles (12–15 Hz bursts) and K-complexes
            detected across the Muse frontal channels.
          </p>
          <p>
            Once the N1→N2 transition is detected, the story fades out over
            90 seconds using an exponential volume curve — so the volume drop
            feels natural rather than mechanical.
          </p>
          <p>
            Without an EEG device, you can set a timer (15 / 30 / 45 min)
            instead. Your sleep latency is saved and shown each morning as a
            personal record.
          </p>
          <div className="flex flex-wrap gap-2 pt-1">
            <Badge variant="secondary" className="text-[10px]">
              Sleep spindles (12–15 Hz) detected
            </Badge>
            <Badge variant="secondary" className="text-[10px]">
              92.98% staging accuracy
            </Badge>
            <Badge variant="secondary" className="text-[10px]">
              90s exponential fade
            </Badge>
          </div>
        </div>
      )}
    </Card>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function SleepStories() {
  const { state: deviceState } = useDevice();
  const eegConnected = deviceState === "streaming";

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [lastLatencyMs, setLastLatencyMs] = useState<number | null>(null);

  const handleSelect = (id: string) => {
    setSelectedId((prev) => (prev === id ? null : id));
  };

  const handleSleepDetected = (latencyMs: number) => {
    setLastLatencyMs(latencyMs);
  };

  return (
    <main className="p-4 md:p-6 space-y-6 max-w-3xl mx-auto">
      {/* Page header */}
      <div className="flex items-center gap-3">
        <Moon className="h-6 w-6 text-primary" />
        <div>
          <h2 className="text-xl font-semibold">Sleep Stories</h2>
          <p className="text-xs text-muted-foreground">
            Ambient soundscapes that fade when you drift off
          </p>
        </div>
      </div>

      {/* EEG status notice */}
      {eegConnected ? (
        <div className="flex items-center gap-3 p-3 rounded-xl border border-primary/30 bg-primary/5 text-sm text-primary">
          <BrainCircuit className="h-4 w-4 shrink-0" />
          EEG connected — stories will auto-fade when N1→N2 sleep transition is detected.
        </div>
      ) : (
        <div className="flex items-center gap-3 p-3 rounded-xl border border-yellow-500/30 bg-yellow-500/5 text-sm text-yellow-500">
          <Clock className="h-4 w-4 shrink-0 opacity-70" />
          No EEG connected — using timer mode. Connect your Muse 2 for automatic fade-out.
        </div>
      )}

      {/* Post-sleep latency summary */}
      {lastLatencyMs !== null && (
        <Card className="glass-card p-4 border-primary/20 bg-primary/5">
          <div className="flex items-center gap-3">
            <Moon className="h-5 w-5 text-primary shrink-0" />
            <div>
              <p className="text-sm font-medium">Sleep detected</p>
              <p className="text-xs text-muted-foreground mt-0.5">
                You fell asleep{" "}
                <span className="text-primary font-semibold">
                  {Math.floor(lastLatencyMs / 60_000)}m{" "}
                  {Math.round((lastLatencyMs % 60_000) / 1000)}s
                </span>{" "}
                into the story. Audio has faded to silence.
              </p>
            </div>
          </div>
        </Card>
      )}

      {/* Story grid */}
      <div className="space-y-3">
        {STORIES.map((story) => (
          <StoryCard
            key={story.id}
            story={story}
            isSelected={selectedId === story.id}
            eegConnected={eegConnected}
            onSelect={handleSelect}
            onSleepDetected={handleSleepDetected}
          />
        ))}
      </div>

      {/* Explainer */}
      <HowItWorks />
    </main>
  );
}
