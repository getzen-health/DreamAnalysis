import { useState, useEffect, useCallback } from "react";
import { Link } from "wouter";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Heart,
  Brain,
  Activity,
  AlertCircle,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Minus,
} from "lucide-react";

interface EmotionState {
  emotion: string;
  confidence: number;
  valence: number;
  arousal: number;
  stress_index: number;
  focus_index: number;
  relaxation_index: number;
}

const EMOTION_LABELS: Record<string, string> = {
  happy: "Happy",
  sad: "Sad",
  angry: "Angry",
  fearful: "Anxious",
  relaxed: "Relaxed",
  focused: "Focused",
};

export default function Dashboard() {
  const [emotion, setEmotion] = useState<EmotionState>({
    emotion: "relaxed",
    confidence: 0.72,
    valence: 0.35,
    arousal: 0.42,
    stress_index: 28,
    focus_index: 65,
    relaxation_index: 72,
  });

  const [shift, setShift] = useState<{
    detected: boolean;
    type: string;
    description: string;
  } | null>(null);

  const [prevEmotion, setPrevEmotion] = useState("relaxed");

  // Simulate real-time updates
  const update = useCallback(() => {
    setEmotion((prev) => {
      const stress = Math.max(
        0,
        Math.min(100, prev.stress_index + (Math.random() - 0.5) * 6)
      );
      const focus = Math.max(
        0,
        Math.min(100, prev.focus_index + (Math.random() - 0.5) * 5)
      );
      const relaxation = Math.max(
        0,
        Math.min(100, prev.relaxation_index + (Math.random() - 0.5) * 5)
      );

      const emotions = ["happy", "sad", "relaxed", "focused", "fearful"];
      const weights = [
        relaxation * 0.3 + (100 - stress) * 0.2,
        stress * 0.15 + (100 - relaxation) * 0.1,
        relaxation * 0.4,
        focus * 0.35,
        stress * 0.2,
      ];
      const maxIdx = weights.indexOf(Math.max(...weights));
      const topEmotion = emotions[maxIdx];

      setPrevEmotion(prev.emotion);

      // Detect emotional shift
      if (topEmotion !== prev.emotion) {
        if (stress > 60 && prev.stress_index <= 50) {
          setShift({
            detected: true,
            type: "approaching_anxiety",
            description:
              "Your body is tensing before you feel it. Take a slow breath.",
          });
        } else if (relaxation > 70 && prev.relaxation_index <= 55) {
          setShift({
            detected: true,
            type: "approaching_calm",
            description: "A wave of calm is settling in. Let it happen.",
          });
        } else {
          setShift(null);
        }
      }

      return {
        emotion: topEmotion,
        confidence: 0.5 + Math.random() * 0.4,
        valence: Math.tanh((relaxation - stress) * 0.02),
        arousal: Math.min(1, (stress + focus) * 0.008),
        stress_index: stress,
        focus_index: focus,
        relaxation_index: relaxation,
      };
    });
  }, []);

  useEffect(() => {
    const interval = setInterval(update, 3000);
    return () => clearInterval(interval);
  }, [update]);

  // Dismiss shift after 8 seconds
  useEffect(() => {
    if (shift?.detected) {
      const timer = setTimeout(() => setShift(null), 8000);
      return () => clearTimeout(timer);
    }
  }, [shift]);

  const emotionLabel = EMOTION_LABELS[emotion.emotion] || emotion.emotion;
  const valenceIcon =
    emotion.valence > 0.1 ? (
      <TrendingUp className="h-4 w-4 text-success" />
    ) : emotion.valence < -0.1 ? (
      <TrendingDown className="h-4 w-4 text-destructive" />
    ) : (
      <Minus className="h-4 w-4 text-muted-foreground" />
    );

  return (
    <main className="p-6 space-y-6 max-w-4xl">
      {/* Emotional Shift Alert */}
      {shift?.detected && (
        <div className="bg-warning/10 border border-warning/20 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-warning mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium text-foreground">
              Emotional Shift Detected
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              {shift.description}
            </p>
          </div>
        </div>
      )}

      {/* Current Emotion */}
      <Card className="glass-card p-6">
        <div className="flex items-center gap-3 mb-5">
          <Heart className="h-5 w-5 text-primary" />
          <h3 className="text-base font-medium">How You Feel</h3>
        </div>

        <div className="flex items-baseline gap-3 mb-1">
          <span className="text-3xl font-semibold text-foreground">
            {emotionLabel}
          </span>
          <span className="text-sm text-muted-foreground">
            {Math.round(emotion.confidence * 100)}% confidence
          </span>
        </div>

        <div className="flex items-center gap-4 mt-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-1.5">
            {valenceIcon}
            <span>
              Valence {emotion.valence >= 0 ? "+" : ""}
              {emotion.valence.toFixed(2)}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <Activity className="h-4 w-4" />
            <span>Arousal {emotion.arousal.toFixed(2)}</span>
          </div>
        </div>
      </Card>

      {/* Mental State */}
      <Card className="glass-card p-6">
        <div className="flex items-center gap-3 mb-5">
          <Brain className="h-5 w-5 text-secondary" />
          <h3 className="text-base font-medium">Mental State</h3>
        </div>

        <div className="space-y-4">
          <div>
            <div className="flex justify-between mb-1.5 text-sm">
              <span className="text-muted-foreground">Stress</span>
              <span className="font-mono">
                {Math.round(emotion.stress_index)}
              </span>
            </div>
            <Progress value={emotion.stress_index} className="h-2" />
          </div>

          <div>
            <div className="flex justify-between mb-1.5 text-sm">
              <span className="text-muted-foreground">Focus</span>
              <span className="font-mono">
                {Math.round(emotion.focus_index)}
              </span>
            </div>
            <Progress value={emotion.focus_index} className="h-2" />
          </div>

          <div>
            <div className="flex justify-between mb-1.5 text-sm">
              <span className="text-muted-foreground">Relaxation</span>
              <span className="font-mono">
                {Math.round(emotion.relaxation_index)}
              </span>
            </div>
            <Progress value={emotion.relaxation_index} className="h-2" />
          </div>
        </div>
      </Card>

      {/* Quick Links */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Link href="/emotions">
          <Card className="glass-card p-4 hover:bg-muted/50 transition-colors cursor-pointer">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Heart className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">Deep Emotion Analysis</span>
              </div>
              <ArrowRight className="h-4 w-4 text-muted-foreground" />
            </div>
          </Card>
        </Link>
        <Link href="/inner-energy">
          <Card className="glass-card p-4 hover:bg-muted/50 transition-colors cursor-pointer">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Activity className="h-4 w-4 text-secondary" />
                <span className="text-sm font-medium">Inner Energy Map</span>
              </div>
              <ArrowRight className="h-4 w-4 text-muted-foreground" />
            </div>
          </Card>
        </Link>
      </div>
    </main>
  );
}
