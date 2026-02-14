import { useState, useEffect, useRef, useCallback } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Brain, CheckCircle, Loader2 } from "lucide-react";
import {
  startCalibration,
  submitCalibration,
  simulateEEG,
} from "@/lib/ml-api";

type WizardStep = "idle" | "recording" | "processing" | "complete";

interface CalibrationStep {
  step: number;
  instruction: string;
  label: string;
  duration_sec: number;
}

interface CalibrationWizardProps {
  onComplete?: (result: { calibrated: boolean; personal_accuracy: number }) => void;
}

export function CalibrationWizard({ onComplete }: CalibrationWizardProps) {
  const [wizardState, setWizardState] = useState<WizardStep>("idle");
  const [steps, setSteps] = useState<CalibrationStep[]>([]);
  const [currentStepIdx, setCurrentStepIdx] = useState(0);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<{ calibrated: boolean; personal_accuracy: number } | null>(null);
  const [recordedSignals, setRecordedSignals] = useState<number[][][]>([]);
  const [labels, setLabels] = useState<string[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const handleStart = async () => {
    try {
      const data = await startCalibration();
      setSteps(data.steps);
      setCurrentStepIdx(0);
      setRecordedSignals([]);
      setLabels([]);
      setWizardState("recording");
      startStepRecording(data.steps[0]);
    } catch (e) {
      console.error("Failed to start calibration:", e);
    }
  };

  const startStepRecording = useCallback((step: CalibrationStep) => {
    setProgress(0);
    let elapsed = 0;

    if (timerRef.current) clearInterval(timerRef.current);

    timerRef.current = setInterval(async () => {
      elapsed += 1;
      setProgress((elapsed / step.duration_sec) * 100);

      if (elapsed >= step.duration_sec) {
        if (timerRef.current) clearInterval(timerRef.current);

        // Simulate EEG for this step (in real use, this would come from the device)
        try {
          const stateMap: Record<string, string> = {
            relaxed: "rest",
            focused: "focused",
            stressed: "stressed",
          };
          const sim = await simulateEEG(
            stateMap[step.label] || "rest",
            step.duration_sec,
            256,
            1
          );

          setRecordedSignals((prev) => [...prev, sim.signals]);
          setLabels((prev) => [...prev, step.label]);

          // Move to next step or finish
          setCurrentStepIdx((prevIdx) => {
            const nextIdx = prevIdx + 1;
            if (nextIdx < steps.length) {
              // We need to start recording the next step
              // Use setTimeout to let state update first
              setTimeout(() => {
                startStepRecording(steps[nextIdx]);
              }, 500);
            } else {
              finishCalibration();
            }
            return nextIdx;
          });
        } catch {
          console.error("Failed to record calibration step");
        }
      }
    }, 1000);
  }, [steps]);

  const finishCalibration = async () => {
    setWizardState("processing");
    try {
      const calResult = await submitCalibration(
        recordedSignals,
        labels,
        256
      );
      setResult(calResult);
      setWizardState("complete");
      onComplete?.(calResult);
    } catch {
      setWizardState("idle");
    }
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  if (wizardState === "idle") {
    return (
      <Card className="p-6 glass-card rounded-xl">
        <div className="text-center space-y-4">
          <Brain className="h-12 w-12 text-primary mx-auto" />
          <h3 className="text-lg font-futuristic font-semibold">Personal Calibration</h3>
          <p className="text-sm text-foreground/60 max-w-md mx-auto">
            Train a personal model by recording 3 brain states (30 seconds each).
            This improves prediction accuracy by adapting to your unique EEG patterns.
          </p>
          <Button onClick={handleStart} className="bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30">
            Start Calibration
          </Button>
        </div>
      </Card>
    );
  }

  if (wizardState === "recording" && steps.length > 0) {
    const step = steps[Math.min(currentStepIdx, steps.length - 1)];
    return (
      <Card className="p-6 glass-card rounded-xl">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-futuristic font-semibold">
              Step {step.step} of {steps.length}
            </h3>
            <span className="text-xs font-mono text-foreground/50">
              {step.label.toUpperCase()}
            </span>
          </div>
          <p className="text-sm text-foreground/80">{step.instruction}</p>
          <Progress value={progress} className="h-2" />
          <p className="text-xs text-foreground/50 text-center">
            Recording... {Math.round(progress)}%
          </p>
        </div>
      </Card>
    );
  }

  if (wizardState === "processing") {
    return (
      <Card className="p-6 glass-card rounded-xl">
        <div className="text-center space-y-4">
          <Loader2 className="h-12 w-12 text-primary mx-auto animate-spin" />
          <h3 className="text-lg font-futuristic font-semibold">Training Personal Model</h3>
          <p className="text-sm text-foreground/60">
            Analyzing your brain patterns...
          </p>
        </div>
      </Card>
    );
  }

  // Complete
  return (
    <Card className="p-6 glass-card rounded-xl">
      <div className="text-center space-y-4">
        <CheckCircle className="h-12 w-12 text-success mx-auto" />
        <h3 className="text-lg font-futuristic font-semibold">Calibration Complete</h3>
        {result && (
          <div className="space-y-2">
            <p className="text-sm text-foreground/80">
              Personal model trained with{" "}
              <span className="text-primary font-mono">
                {(result.personal_accuracy * 100).toFixed(0)}%
              </span>{" "}
              estimated accuracy improvement.
            </p>
          </div>
        )}
        <Button
          variant="outline"
          onClick={() => setWizardState("idle")}
          className="border-primary/30 text-primary"
        >
          Re-calibrate
        </Button>
      </div>
    </Card>
  );
}
