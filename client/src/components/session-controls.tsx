import { useState, useEffect, useRef } from "react";
import { Circle, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { startSession, stopSession } from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";

interface SessionControlsProps {
  onRecordingChange?: (recording: boolean) => void;
}

export function SessionControls({ onRecordingChange }: SessionControlsProps) {
  const userId = useRef(getParticipantId());
  const [isRecording, setIsRecording] = useState(false);
  const [sessionType, setSessionType] = useState("general");
  const [elapsed, setElapsed] = useState(0);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
      setElapsed(0);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRecording]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m.toString().padStart(2, "0")}:${sec.toString().padStart(2, "0")}`;
  };

  const handleStart = async () => {
    try {
      const result = await startSession(sessionType, userId.current);
      setSessionId(result.session_id);
      setIsRecording(true);
      onRecordingChange?.(true);
    } catch (e) {
      console.error("Failed to start recording:", e);
    }
  };

  const handleStop = async () => {
    try {
      await stopSession(userId.current);
      setIsRecording(false);
      setSessionId(null);
      onRecordingChange?.(false);
    } catch (e) {
      console.error("Failed to stop recording:", e);
    }
  };

  return (
    <div className="flex items-center gap-3">
      {!isRecording && (
        <Select value={sessionType} onValueChange={setSessionType}>
          <SelectTrigger className="w-32 h-8 text-xs bg-card/50 border-primary/20">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="general">General</SelectItem>
            <SelectItem value="sleep">Sleep</SelectItem>
            <SelectItem value="meditation">Meditation</SelectItem>
            <SelectItem value="neurofeedback">Neurofeedback</SelectItem>
          </SelectContent>
        </Select>
      )}

      {isRecording ? (
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1">
            <Circle className="h-3 w-3 text-destructive fill-destructive animate-pulse" />
            <span className="text-xs font-mono text-destructive">REC</span>
          </span>
          <span className="text-xs font-mono text-foreground/70">
            {formatTime(elapsed)}
          </span>
          <Button
            size="sm"
            variant="outline"
            className="h-7 text-xs border-destructive/30 text-destructive hover:bg-destructive/10"
            onClick={handleStop}
          >
            <Square className="h-3 w-3 mr-1" />
            Stop
          </Button>
        </div>
      ) : (
        <Button
          size="sm"
          variant="outline"
          className="h-7 text-xs border-primary/30 text-primary hover:bg-primary/10"
          onClick={handleStart}
        >
          <Circle className="h-3 w-3 mr-1 fill-destructive text-destructive" />
          Record
        </Button>
      )}
    </div>
  );
}
