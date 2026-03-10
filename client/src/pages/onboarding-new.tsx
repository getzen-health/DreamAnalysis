import { useEffect, useState } from "react";
import { Link, useLocation } from "wouter";
import { Brain, ChevronRight, Mic, Sparkles, Watch, Waves } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useVoiceEmotion } from "@/hooks/use-voice-emotion";
import { useAuth } from "@/hooks/use-auth";

type Stage = "choose" | "voice" | "result";

export default function OnboardingNew() {
  const [, navigate] = useLocation();
  const { user } = useAuth();
  const [stage, setStage] = useState<Stage>("choose");
  const voiceEmotion = useVoiceEmotion({ durationMs: 10000, userId: "default" });

  useEffect(() => {
    if (voiceEmotion.lastResult) {
      setStage("result");
    }
  }, [voiceEmotion.lastResult]);

  const finishPath = user ? "/" : "/auth";

  if (stage === "voice") {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center p-6">
        <div className="max-w-md w-full space-y-6">
          <div className="text-center space-y-3">
            <Badge variant="outline" className="border-primary/40 text-primary">
              Recommended path
            </Badge>
            <h1 className="text-3xl font-semibold">Voice + Health setup</h1>
            <p className="text-sm text-muted-foreground">
              Record a 10-second check-in to start with emotion inference now. EEG stays optional.
            </p>
          </div>

          <Card className="glass-card p-6 space-y-4">
            <div className="flex items-start gap-3">
              <Mic className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">Voice check-in</p>
                <p className="text-sm text-muted-foreground">
                  Uses the live `voice-watch` pipeline and caches the result for downstream pages.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <Watch className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">Health data when available</p>
                <p className="text-sm text-muted-foreground">
                  Apple Health or wearable biometrics can refine stress and readiness estimates later.
                </p>
              </div>
            </div>
          </Card>

          <div className="space-y-3">
            <Button
              className="w-full"
              onClick={voiceEmotion.startRecording}
              disabled={voiceEmotion.isRecording || voiceEmotion.isAnalyzing}
            >
              {voiceEmotion.isRecording
                ? "Recording..."
                : voiceEmotion.isAnalyzing
                ? "Analyzing..."
                : "Start 10-second voice check-in"}
            </Button>
            <Button variant="outline" className="w-full" onClick={() => navigate(finishPath)}>
              Skip for now
            </Button>
            {voiceEmotion.error && (
              <p className="text-sm text-destructive text-center">{voiceEmotion.error}</p>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (stage === "result" && voiceEmotion.lastResult) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center p-6">
        <div className="max-w-md w-full space-y-6">
          <div className="text-center space-y-3">
            <Badge variant="outline" className="border-primary/40 text-primary">
              Check-in complete
            </Badge>
            <h1 className="text-3xl font-semibold">Your first state read is ready</h1>
          </div>

          <Card className="glass-card p-6 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Emotion</span>
              <span className="font-semibold capitalize">{voiceEmotion.lastResult.emotion}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Confidence</span>
              <span className="font-mono">{Math.round(voiceEmotion.lastResult.confidence * 100)}%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Valence</span>
              <span className="font-mono">{voiceEmotion.lastResult.valence >= 0 ? "+" : ""}{voiceEmotion.lastResult.valence.toFixed(2)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Model</span>
              <span className="font-mono text-xs">{voiceEmotion.lastResult.model_type}</span>
            </div>
          </Card>

          <div className="grid gap-3 sm:grid-cols-2">
            <Card className="p-4 border-primary/20">
              <div className="flex items-start gap-3">
                <Sparkles className="h-4 w-4 text-primary mt-0.5" />
                <div>
                  <p className="text-sm font-medium">Continue without EEG</p>
                  <p className="text-xs text-muted-foreground">
                    Use voice and health inputs first, then add EEG later if needed.
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-4 border-primary/20">
              <div className="flex items-start gap-3">
                <Waves className="h-4 w-4 text-primary mt-0.5" />
                <div>
                  <p className="text-sm font-medium">Add Muse 2 later</p>
                  <p className="text-xs text-muted-foreground">
                    EEG unlocks live neural features and calibration, but it is not required to start.
                  </p>
                </div>
              </div>
            </Card>
          </div>

          <div className="space-y-3">
            <Button className="w-full" onClick={() => navigate(finishPath)}>
              Continue
              <ChevronRight className="ml-2 h-4 w-4" />
            </Button>
            <Button variant="outline" className="w-full" onClick={() => navigate("/onboarding")}>
              I have a Muse 2
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex items-center justify-center p-6">
      <div className="max-w-4xl w-full space-y-8">
        <div className="text-center space-y-3">
          <Badge variant="outline" className="border-primary/40 text-primary">
            Start here
          </Badge>
          <h1 className="text-4xl font-semibold">Choose your setup path</h1>
          <p className="text-sm text-muted-foreground max-w-2xl mx-auto">
            The app works without EEG. Start with voice and health for the fastest setup, or go
            straight to Muse 2 if you want live neural features from the beginning.
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <Card className="glass-card p-6 space-y-4 border-primary/30">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Mic className="h-5 w-5 text-primary" />
                <h2 className="text-xl font-semibold">Voice + Health</h2>
              </div>
              <Badge>Recommended</Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              Start with a 10-second voice check-in. Add health data and EEG later as optional inputs.
            </p>
            <ul className="text-sm space-y-2 text-foreground/90">
              <li>Fastest way to get emotion and readiness output</li>
              <li>No headset required</li>
              <li>Uses the same live voice path as the rest of the app</li>
            </ul>
            <Button className="w-full" onClick={() => setStage("voice")}>
              Start with voice
            </Button>
          </Card>

          <Card className="glass-card p-6 space-y-4">
            <div className="flex items-center gap-3">
              <Brain className="h-5 w-5 text-primary" />
              <h2 className="text-xl font-semibold">Muse 2 EEG</h2>
            </div>
            <p className="text-sm text-muted-foreground">
              Use the existing calibration onboarding if you already have a headset and want live EEG.
            </p>
            <ul className="text-sm space-y-2 text-foreground/90">
              <li>Live stress, focus, and relaxation from neural signals</li>
              <li>Baseline calibration improves model accuracy</li>
              <li>Best for neurofeedback and continuous sessions</li>
            </ul>
            <Link href="/onboarding">
              <Button variant="outline" className="w-full">
                I have a Muse 2
              </Button>
            </Link>
          </Card>
        </div>

        <div className="text-center">
          <Button variant="ghost" onClick={() => navigate(finishPath)}>
            Skip for now
          </Button>
        </div>
      </div>
    </div>
  );
}
