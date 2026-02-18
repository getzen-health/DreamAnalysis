import { Music, Play, Pause, Volume2, VolumeX, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { useMoodMusic } from "@/hooks/use-mood-music";

interface MoodMusicPlayerProps {
  emotion?: string;
  isStreaming?: boolean;
}

export function MoodMusicPlayer({ emotion, isStreaming }: MoodMusicPlayerProps) {
  const music = useMoodMusic(emotion, isStreaming);
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <Card className="glass-card rounded-xl hover-glow overflow-hidden">
        <CollapsibleTrigger asChild>
          <button className="w-full p-4 flex items-center justify-between text-left">
            <div className="flex items-center gap-3">
              <Music className="h-4 w-4 text-violet-400" />
              <span className="text-sm font-medium">Mood Music</span>
              {music.isEnabled && (
                <Badge variant="secondary" className="text-[10px]">
                  {music.currentSoundscape.name}
                </Badge>
              )}
            </div>
            {isOpen ? (
              <ChevronUp className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            )}
          </button>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div className="px-4 pb-4 space-y-4">
            {!music.isEnabled ? (
              <Button
                variant="outline"
                size="sm"
                className="w-full"
                onClick={music.enable}
              >
                <Volume2 className="h-3.5 w-3.5 mr-2" />
                Enable Audio
              </Button>
            ) : (
              <>
                {/* Transport controls */}
                <div className="flex items-center gap-3">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={music.toggle}
                  >
                    {music.isPlaying ? (
                      <Pause className="h-4 w-4" />
                    ) : (
                      <Play className="h-4 w-4" />
                    )}
                  </Button>

                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={music.toggleMute}
                  >
                    {music.isMuted ? (
                      <VolumeX className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <Volume2 className="h-4 w-4" />
                    )}
                  </Button>

                  <Slider
                    className="flex-1"
                    min={0}
                    max={100}
                    step={1}
                    value={[Math.round(music.volume * 100)]}
                    onValueChange={([v]) => music.setVolume(v / 100)}
                  />
                </div>

                {/* Layer indicators */}
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  <span className={music.binauralActive ? "text-cyan-400" : ""}>
                    Binaural {music.binauralActive ? "ON" : "OFF"}
                  </span>
                  <span className={music.droneActive ? "text-violet-400" : ""}>
                    Drone {music.droneActive ? "ON" : "OFF"}
                  </span>
                </div>

                {/* Current mood */}
                <div className="text-xs text-muted-foreground">
                  Mood: <span className="capitalize text-foreground">{music.currentSoundscape.emotion}</span>
                </div>
              </>
            )}
          </div>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
