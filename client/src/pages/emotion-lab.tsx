import { useState, useEffect, useCallback } from "react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis,
} from "recharts";
import { Brain, Heart, Activity, TrendingUp, Zap } from "lucide-react";
import { EmotionWheel } from "@/components/emotion-wheel";
import { BrainBands } from "@/components/brain-bands";

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

export default function EmotionLab() {
  const [currentEmotion, setCurrentEmotion] = useState<EmotionState>({
    emotion: "relaxed",
    confidence: 0.72,
    valence: 0.35,
    arousal: 0.42,
    stress_index: 28,
    focus_index: 65,
    relaxation_index: 72,
    band_powers: { delta: 0.15, theta: 0.2, alpha: 0.35, beta: 0.2, gamma: 0.1 },
    probabilities: { happy: 0.15, sad: 0.05, angry: 0.03, fearful: 0.05, relaxed: 0.52, focused: 0.20 },
  });

  const [emotionHistory, setEmotionHistory] = useState<Array<EmotionState & { time: string }>>([]);

  const updateEmotions = useCallback(() => {
    setCurrentEmotion(prev => {
      const newBands: Record<string, number> = {};
      let total = 0;
      for (const band of ["delta", "theta", "alpha", "beta", "gamma"]) {
        const val = Math.max(0.01, (prev.band_powers[band] || 0.2) + (Math.random() - 0.5) * 0.05);
        newBands[band] = val;
        total += val;
      }
      for (const band in newBands) newBands[band] /= total;

      const alpha = newBands.alpha || 0;
      const beta = newBands.beta || 0;
      const theta = newBands.theta || 0;
      const gamma = newBands.gamma || 0;

      const stress = Math.max(0, Math.min(100, prev.stress_index + (Math.random() - 0.5) * 8));
      const focus = Math.max(0, Math.min(100, prev.focus_index + (Math.random() - 0.5) * 6));
      const relaxation = Math.max(0, Math.min(100, alpha * 100 + (Math.random() - 0.5) * 10));

      const probs: Record<string, number> = {};
      let probTotal = 0;
      probs.happy = Math.max(0, 0.15 + (alpha - beta) * 0.3 + Math.random() * 0.05);
      probs.sad = Math.max(0, 0.05 + (newBands.delta - alpha) * 0.2 + Math.random() * 0.03);
      probs.angry = Math.max(0, 0.03 + (beta - alpha) * 0.2 + Math.random() * 0.02);
      probs.fearful = Math.max(0, 0.05 + gamma * 0.15 + Math.random() * 0.03);
      probs.relaxed = Math.max(0, alpha * 0.6 + theta * 0.2 + Math.random() * 0.05);
      probs.focused = Math.max(0, beta * 0.4 + gamma * 0.2 + Math.random() * 0.05);
      for (const k in probs) probTotal += probs[k];
      for (const k in probs) probs[k] /= probTotal;

      const topEmotion = Object.entries(probs).sort(([, a], [, b]) => b - a)[0];

      return {
        emotion: topEmotion[0],
        confidence: topEmotion[1],
        valence: Math.tanh((alpha - beta) * 2),
        arousal: Math.min(1, beta + gamma),
        stress_index: stress,
        focus_index: focus,
        relaxation_index: relaxation,
        band_powers: newBands,
        probabilities: probs,
      };
    });
  }, []);

  useEffect(() => {
    const interval = setInterval(updateEmotions, 3000);
    return () => clearInterval(interval);
  }, [updateEmotions]);

  useEffect(() => {
    const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    setEmotionHistory(prev => [...prev.slice(-20), { ...currentEmotion, time: now }]);
  }, [currentEmotion]);

  const vaData = emotionHistory.map((e, i) => ({
    valence: e.valence,
    arousal: e.arousal,
    emotion: e.emotion,
    size: 50 + i * 5,
  }));

  return (
    <main className="p-6 space-y-6 max-w-5xl">
      {/* Top Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Emotion Wheel */}
        <Card className="glass-card p-6">
          <h3 className="text-sm font-medium mb-4 flex items-center gap-2">
            <Heart className="h-4 w-4 text-primary" />
            Current Emotion
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
          </h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={emotionHistory.slice(-15)}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
              <XAxis dataKey="time" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
              <Tooltip contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8 }} />
              <Line type="monotone" dataKey="stress_index" stroke="hsl(var(--warning))" strokeWidth={1.5} dot={false} name="Stress" />
              <Line type="monotone" dataKey="focus_index" stroke="hsl(var(--primary))" strokeWidth={1.5} dot={false} name="Focus" />
              <Line type="monotone" dataKey="relaxation_index" stroke="hsl(var(--success))" strokeWidth={1.5} dot={false} name="Relaxation" />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* Valence-Arousal */}
        <Card className="glass-card p-6">
          <h3 className="text-sm font-medium mb-4 flex items-center gap-2">
            <Zap className="h-4 w-4 text-warning" />
            Valence-Arousal Space
          </h3>
          <ResponsiveContainer width="100%" height={220}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
              <XAxis type="number" dataKey="valence" domain={[-1, 1]} name="Valence" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
              <YAxis type="number" dataKey="arousal" domain={[0, 1]} name="Arousal" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
              <ZAxis type="number" dataKey="size" range={[30, 150]} />
              <Tooltip
                contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8 }}
                formatter={(value: number, name: string) => [value.toFixed(2), name]}
              />
              <Scatter data={vaData} fill="hsl(var(--primary))" fillOpacity={0.5} />
            </ScatterChart>
          </ResponsiveContainer>
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
