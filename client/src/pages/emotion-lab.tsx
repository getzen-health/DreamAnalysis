import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis,
} from "recharts";
import { Brain, Heart, Activity, TrendingUp, Zap, Radio } from "lucide-react";
import { EmotionWheel } from "@/components/emotion-wheel";
import { BrainBands } from "@/components/brain-bands";
import { useDevice } from "@/hooks/use-device";

interface EmotionState {
  emotion: string;
  confidence: number;
  valence: number;
  arousal: number;
  stress_index: number;
  focus_index: number;
  relaxation_index: number;
  band_powers: Record<string, number>;
  probabilities: Record<string, number>;
}

interface HistoryEntry extends EmotionState {
  time: string;
}

export default function EmotionLab() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;

  const [emotionHistory, setEmotionHistory] = useState<HistoryEntry[]>([]);

  // Build current emotion from live Muse 2 data
  const emotions = analysis?.emotions;
  const bandPowers = analysis?.band_powers ?? {};

  const currentEmotion: EmotionState = emotions
    ? {
        emotion: emotions.emotion ?? "unknown",
        confidence: emotions.confidence ?? 0,
        valence: emotions.valence ?? 0,
        arousal: emotions.arousal ?? 0,
        stress_index: (emotions.stress_index ?? 0) * 100,
        focus_index: (emotions.focus_index ?? 0) * 100,
        relaxation_index: (emotions.relaxation_index ?? 0) * 100,
        band_powers: bandPowers,
        probabilities: emotions.probabilities ?? {},
      }
    : {
        emotion: "—",
        confidence: 0,
        valence: 0,
        arousal: 0,
        stress_index: 0,
        focus_index: 0,
        relaxation_index: 0,
        band_powers: {},
        probabilities: {},
      };

  // Accumulate history from live stream
  useEffect(() => {
    if (!isStreaming || !emotions) return;

    const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    setEmotionHistory((prev) => [
      ...prev.slice(-60),
      { ...currentEmotion, time: now },
    ]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  const vaData = emotionHistory.map((e, i) => ({
    valence: e.valence,
    arousal: e.arousal,
    emotion: e.emotion,
    size: 30 + i * 2,
  }));

  return (
    <main className="p-6 space-y-6 max-w-5xl">
      {/* Connection status */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          Connect your Muse 2 from the sidebar to see live emotion data. Showing empty state.
        </div>
      )}

      {/* Top Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Emotion Wheel */}
        <Card className="glass-card p-6">
          <h3 className="text-sm font-medium mb-4 flex items-center gap-2">
            <Heart className="h-4 w-4 text-primary" />
            Current Emotion
            {isStreaming && (
              <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </h3>
          <EmotionWheel
            probabilities={currentEmotion.probabilities}
            dominantEmotion={currentEmotion.emotion}
            confidence={currentEmotion.confidence}
          />
        </Card>

        {/* Brain Bands */}
        <Card className="glass-card p-6">
          <h3 className="text-sm font-medium mb-4 flex items-center gap-2">
            <Brain className="h-4 w-4 text-primary" />
            Brain Waves
            {isStreaming && (
              <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </h3>
          <BrainBands bandPowers={currentEmotion.band_powers} />
        </Card>

        {/* Mental State */}
        <Card className="glass-card p-6">
          <h3 className="text-sm font-medium mb-4 flex items-center gap-2">
            <Activity className="h-4 w-4 text-secondary" />
            Mental State
          </h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between mb-1 text-sm">
                <span className="text-muted-foreground">Stress</span>
                <span className="font-mono text-warning">{Math.round(currentEmotion.stress_index)}</span>
              </div>
              <Progress value={currentEmotion.stress_index} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between mb-1 text-sm">
                <span className="text-muted-foreground">Focus</span>
                <span className="font-mono text-primary">{Math.round(currentEmotion.focus_index)}</span>
              </div>
              <Progress value={currentEmotion.focus_index} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between mb-1 text-sm">
                <span className="text-muted-foreground">Relaxation</span>
                <span className="font-mono text-success">{Math.round(currentEmotion.relaxation_index)}</span>
              </div>
              <Progress value={currentEmotion.relaxation_index} className="h-2" />
            </div>
            <div className="pt-3 border-t border-border text-sm space-y-1">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Valence</span>
                <span className={`font-mono ${currentEmotion.valence >= 0 ? "text-success" : "text-destructive"}`}>
                  {currentEmotion.valence >= 0 ? "+" : ""}{currentEmotion.valence.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Arousal</span>
                <span className="font-mono text-secondary">{currentEmotion.arousal.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Timeline */}
        <Card className="glass-card p-6">
          <h3 className="text-sm font-medium mb-4 flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-primary" />
            Timeline
            <span className="ml-auto text-[10px] text-muted-foreground">
              {emotionHistory.length} samples
            </span>
          </h3>
          {emotionHistory.length < 2 ? (
            <div className="h-[220px] flex items-center justify-center text-sm text-muted-foreground">
              {isStreaming ? "Collecting data..." : "Connect device to see timeline"}
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={emotionHistory.slice(-30)}>
                <defs>
                  <linearGradient id="tlStressGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(38, 85%, 58%)" stopOpacity={0.25} />
                    <stop offset="100%" stopColor="hsl(38, 85%, 58%)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="tlFocusGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(152, 60%, 48%)" stopOpacity={0.25} />
                    <stop offset="100%" stopColor="hsl(152, 60%, 48%)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="tlRelaxGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(200, 70%, 55%)" stopOpacity={0.25} />
                    <stop offset="100%" stopColor="hsl(200, 70%, 55%)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 22%)" opacity={0.6} />
                <XAxis dataKey="time" tick={{ fontSize: 9, fill: "hsl(220, 12%, 52%)" }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "hsl(220, 12%, 52%)" }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, fontSize: 12, color: "hsl(38, 20%, 92%)" }} labelStyle={{ color: "hsl(38, 20%, 92%)" }} itemStyle={{ color: "hsl(220, 12%, 75%)" }} />
                <Line type="monotone" dataKey="stress_index" stroke="hsl(38, 85%, 58%)" strokeWidth={2.5} dot={false} name="Stress" />
                <Line type="monotone" dataKey="focus_index" stroke="hsl(152, 60%, 48%)" strokeWidth={2.5} dot={false} name="Focus" />
                <Line type="monotone" dataKey="relaxation_index" stroke="hsl(200, 70%, 55%)" strokeWidth={2.5} dot={false} name="Relaxation" />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        {/* Valence-Arousal */}
        <Card className="glass-card p-6">
          <h3 className="text-sm font-medium mb-4 flex items-center gap-2">
            <Zap className="h-4 w-4 text-warning" />
            Valence-Arousal Space
          </h3>
          {vaData.length < 2 ? (
            <div className="h-[220px] flex items-center justify-center text-sm text-muted-foreground">
              {isStreaming ? "Collecting data..." : "Connect device to see V-A plot"}
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <ScatterChart>
                <defs>
                  <radialGradient id="vaGlow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="hsl(152, 60%, 55%)" stopOpacity={0.9} />
                    <stop offset="70%" stopColor="hsl(200, 70%, 55%)" stopOpacity={0.6} />
                    <stop offset="100%" stopColor="hsl(262, 45%, 65%)" stopOpacity={0.3} />
                  </radialGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 22%)" opacity={0.6} />
                <XAxis type="number" dataKey="valence" domain={[-1, 1]} name="Valence" tick={{ fontSize: 10, fill: "hsl(220, 12%, 55%)" }} axisLine={{ stroke: "hsl(220, 18%, 25%)" }} tickLine={false} />
                <YAxis type="number" dataKey="arousal" domain={[0, 1]} name="Arousal" tick={{ fontSize: 10, fill: "hsl(220, 12%, 55%)" }} axisLine={{ stroke: "hsl(220, 18%, 25%)" }} tickLine={false} />
                <ZAxis type="number" dataKey="size" range={[40, 200]} />
                <Tooltip
                  contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, fontSize: 12, color: "hsl(38, 20%, 92%)" }}
                  labelStyle={{ color: "hsl(38, 20%, 92%)" }}
                  itemStyle={{ color: "hsl(220, 12%, 75%)" }}
                  formatter={(value: number, name: string) => [value.toFixed(2), name]}
                />
                <Scatter data={vaData} fill="url(#vaGlow)" stroke="hsl(152, 60%, 55%)" strokeWidth={1} fillOpacity={0.85} />
              </ScatterChart>
            </ResponsiveContainer>
          )}
          <div className="flex justify-between text-xs text-muted-foreground mt-2 px-4">
            <span>Negative</span>
            <span>Valence</span>
            <span>Positive</span>
          </div>
        </Card>
      </div>
    </main>
  );
}
