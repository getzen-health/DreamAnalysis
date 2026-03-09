/**
 * VoiceCheckinCard — 10-second voice mood check-in, prompted 3× daily.
 *
 * Appears on the dashboard at 08:00, 13:00, 20:00 (±2h window).
 * Outside those windows it still renders in a compact "not time yet" state
 * so users can always do a manual check-in.
 */
import { useState, useRef, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Mic, MicOff, CheckCircle, Clock, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import {
  submitVoiceCheckin,
  getCheckinDailySummary,
  type CheckinDailySummary,
} from "@/lib/ml-api";

// ── helpers ───────────────────────────────────────────────────────────────────

const CHECKIN_SLOTS: { hour: number; label: string; display: string }[] = [
  { hour: 8, label: "morning", display: "Morning" },
  { hour: 13, label: "afternoon", display: "Afternoon" },
  { hour: 20, label: "evening", display: "Evening" },
];

/** Returns the label of the current active slot (±2h), or null. */
function activeSlot(): { label: string; display: string } | null {
  const h = new Date().getHours();
  for (const slot of CHECKIN_SLOTS) {
    if (Math.abs(h - slot.hour) <= 2) return slot;
  }
  return null;
}

/** Convert a Float32Array PCM buffer to a base64-encoded WAV string. */
function pcmToWavB64(pcm: Float32Array, sampleRate: number): string {
  const numSamples = pcm.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);     // PCM
  view.setUint16(22, 1, true);     // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, numSamples * 2, true);

  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

const EMOTION_EMOJI: Record<string, string> = {
  happy: "😊", sad: "😢", angry: "😠", fear: "😨", surprise: "😲", neutral: "😐",
};

const TREND_ICON = {
  improving: TrendingUp,
  declining: TrendingDown,
  stable: Minus,
  insufficient_data: Minus,
};

const TREND_COLOR = {
  improving: "text-green-400",
  declining: "text-red-400",
  stable: "text-yellow-400",
  insufficient_data: "text-muted-foreground",
};

// ── component ─────────────────────────────────────────────────────────────────

interface Props {
  userId: string;
}

export function VoiceCheckinCard({ userId }: Props) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const slot = activeSlot();

  // Recording state
  const [recording, setRecording] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(10);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);

  const { data: summary } = useQuery<CheckinDailySummary>({
    queryKey: ["checkin-summary", userId],
    queryFn: () => getCheckinDailySummary(userId),
    refetchInterval: 120_000,
    retry: false,
  });

  const submit = useMutation({
    mutationFn: (audio_b64: string) =>
      submitVoiceCheckin({ user_id: userId, audio_b64, sample_rate: 16000 }),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["checkin-summary", userId] });
      toast({
        title: `Check-in recorded — ${EMOTION_EMOJI[data.emotion] ?? ""} ${data.emotion}`,
        description: `Valence ${data.valence > 0 ? "+" : ""}${(data.valence * 100).toFixed(0)}%  ·  Stress ${(data.stress_index * 100).toFixed(0)}%`,
      });
    },
    onError: () => {
      toast({ title: "Check-in failed", description: "Could not reach ML backend.", variant: "destructive" });
    },
  });

  // ── recording flow ────────────────────────────────────────────────────────

  const stopRecording = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (mediaRef.current && mediaRef.current.state !== "inactive") {
      mediaRef.current.stop();
    }
    setRecording(false);
    setSecondsLeft(10);
  }, []);

  const startRecording = useCallback(async () => {
    if (recording) { stopRecording(); return; }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
    } catch {
      toast({ title: "Microphone access denied", variant: "destructive" });
      return;
    }

    chunksRef.current = [];
    const mr = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
    mediaRef.current = mr;

    mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };

    mr.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });

      // Decode to PCM via AudioContext, then encode as WAV
      try {
        const arrayBuf = await blob.arrayBuffer();
        const ctx = new AudioContext({ sampleRate: 16000 });
        audioCtxRef.current = ctx;
        const decoded = await ctx.decodeAudioData(arrayBuf);
        const pcm = decoded.getChannelData(0);
        const b64 = pcmToWavB64(pcm, 16000);
        await ctx.close();
        submit.mutate(b64);
      } catch {
        toast({ title: "Audio encoding failed", description: "Could not process the recording.", variant: "destructive" });
      }
    };

    mr.start(100);
    setRecording(true);
    setSecondsLeft(10);

    let s = 10;
    timerRef.current = setInterval(() => {
      s -= 1;
      setSecondsLeft(s);
      if (s <= 0) stopRecording();
    }, 1000);
  }, [recording, stopRecording, submit, toast]);

  // ── derived display ───────────────────────────────────────────────────────

  const alreadyDoneToday = summary && summary.checkin_count > 0;
  const slotDone = slot && summary?.slots_completed.includes(slot.label);
  const TrendIcon = TREND_ICON[(summary?.trend ?? "insufficient_data") as keyof typeof TREND_ICON];
  const trendColor = TREND_COLOR[(summary?.trend ?? "insufficient_data") as keyof typeof TREND_COLOR];

  return (
    <Card className="glass-card hover-glow">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Mic className="h-4 w-4 text-accent" />
            Voice Check-In
          </CardTitle>
          {slot && !slotDone ? (
            <Badge variant="outline" className="text-[10px] border-accent/40 text-accent">
              {slot.display} check-in
            </Badge>
          ) : (
            <Badge variant="outline" className="text-[10px] text-muted-foreground">
              {alreadyDoneToday ? `${summary!.checkin_count}/3 today` : "Optional"}
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        {/* Daily summary row */}
        {summary && summary.checkin_count > 0 && (
          <div className="flex items-center gap-3 p-2 rounded-lg bg-muted/40 text-xs">
            <span className="text-lg leading-none">
              {EMOTION_EMOJI[summary.dominant_emotion ?? "neutral"] ?? "😐"}
            </span>
            <div className="flex-1 min-w-0">
              <p className="font-medium text-foreground capitalize">{summary.dominant_emotion ?? "—"}</p>
              <p className="text-muted-foreground">
                {summary.avg_valence !== null
                  ? `Valence ${summary.avg_valence > 0 ? "+" : ""}${(summary.avg_valence * 100).toFixed(0)}%`
                  : ""}
                {summary.avg_stress !== null
                  ? `  ·  Stress ${(summary.avg_stress * 100).toFixed(0)}%`
                  : ""}
              </p>
            </div>
            <div className={`flex items-center gap-1 ${trendColor}`}>
              <TrendIcon className="h-3.5 w-3.5" />
              <span className="capitalize text-[10px]">{summary.trend.replace("_", " ")}</span>
            </div>
          </div>
        )}

        {/* Slot progress dots */}
        <div className="flex items-center gap-1.5">
          {CHECKIN_SLOTS.map((s) => {
            const done = summary?.slots_completed.includes(s.label);
            const active = slot?.label === s.label;
            return (
              <div
                key={s.label}
                className={`h-1.5 flex-1 rounded-full transition-colors ${
                  done ? "bg-accent" : active ? "bg-accent/40" : "bg-muted"
                }`}
                title={`${s.display}${done ? " ✓" : ""}`}
              />
            );
          })}
        </div>
        <div className="flex justify-between text-[10px] text-muted-foreground px-0.5">
          <span>8am</span><span>1pm</span><span>8pm</span>
        </div>

        {/* Record button */}
        {slotDone ? (
          <div className="flex items-center gap-2 text-xs text-muted-foreground py-1">
            <CheckCircle className="h-3.5 w-3.5 text-accent" />
            {slot?.display} check-in complete
          </div>
        ) : (
          <Button
            size="sm"
            variant={recording ? "destructive" : "outline"}
            className={`w-full text-xs ${recording ? "" : "border-accent/40 hover:border-accent"}`}
            onClick={startRecording}
            disabled={submit.isPending}
          >
            {submit.isPending ? (
              <span className="flex items-center gap-1.5">
                <Clock className="h-3.5 w-3.5 animate-pulse" /> Processing…
              </span>
            ) : recording ? (
              <span className="flex items-center gap-1.5">
                <MicOff className="h-3.5 w-3.5" /> Stop ({secondsLeft}s)
              </span>
            ) : (
              <span className="flex items-center gap-1.5">
                <Mic className="h-3.5 w-3.5" />
                {slot && !slotDone ? `Record ${slot.display} Check-In` : "Record Check-In"}
              </span>
            )}
          </Button>
        )}

        {recording && (
          <div className="flex items-center gap-2">
            <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-red-500 transition-all"
                style={{ width: `${((10 - secondsLeft) / 10) * 100}%` }}
              />
            </div>
            <span className="text-[10px] text-muted-foreground w-6 text-right">{secondsLeft}s</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
