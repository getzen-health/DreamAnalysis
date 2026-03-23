import { useState, useEffect, useRef, useCallback } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Headphones, Play, Square, Volume2, VolumeX, Trophy, Target, Timer, Radio } from "lucide-react";
import {
  getNeurofeedbackProtocols,
  startNeurofeedback,
  evaluateNeurofeedback,
  stopNeurofeedback,
} from "@/lib/ml-api";
import { writeMindfulSession } from "@/lib/health-connect";
import { useDevice } from "@/hooks/use-device";
import { useToast } from "@/hooks/use-toast";
import { NeurofeedbackScheduleCard } from "@/components/neurofeedback-schedule-card";
import {
  getSessionSchedule,
  getSessionHistory,
  recordNeurofeedbackSession,
} from "@/lib/neurofeedback-schedule";
import SubstanceQuestionnaire, { SubstanceContextNote } from "@/components/substance-questionnaire";
import {
  getLatestSubstanceLog,
  getBaselineAdjustment,
  hasAnsweredToday,
} from "@/lib/substance-context";

type SessionPhase = "idle" | "calibrating" | "training" | "summary";

interface SessionStats {
  total_rewards: number;
  reward_rate: number;
  avg_score: number;
  max_streak: number;
  total_evaluations: number;
}

export default function Neurofeedback() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const bandPowers = latestFrame?.analysis?.band_powers;

  const [phase, setPhase] = useState<SessionPhase>("idle");
  const [protocols, setProtocols] = useState<Record<string, { name: string; description: string }>>({});
  const [selectedProtocol, setSelectedProtocol] = useState("alpha_up");
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [score, setScore] = useState(0);
  const [reward, setReward] = useState(false);
  const [streak, setStreak] = useState(0);
  const [rewardCount, setRewardCount] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [stats, setStats] = useState<SessionStats | null>(null);
  const [showTooSoonDialog, setShowTooSoonDialog] = useState(false);
  const [showSubstanceQ, setShowSubstanceQ] = useState(() => !hasAnsweredToday());
  const substanceNote = (() => {
    const log = getLatestSubstanceLog();
    const adj = getBaselineAdjustment(log);
    return adj?.note ?? "";
  })();

  // Session scheduling — recalculate when phase changes back to idle
  const [sessionHistory, setSessionHistory] = useState<Date[]>(() =>
    getSessionHistory()
  );
  const schedule = getSessionSchedule(sessionHistory);

  const { toast } = useToast();
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const calibrationTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sessionStartRef = useRef<Date | null>(null);

  // Load protocols on mount
  useEffect(() => {
    getNeurofeedbackProtocols().then(setProtocols).catch(() => {});
  }, []);

  // Calibration timeout — skip to training after 30 seconds
  useEffect(() => {
    if (phase === "calibrating") {
      calibrationTimeoutRef.current = setTimeout(() => {
        if (phase === "calibrating") {
          toast({
            title: "Calibration timed out",
            description: "Skipping to training phase with default baseline.",
            variant: "destructive",
          });
          setPhase("training");
        }
      }, 30_000);
    } else {
      if (calibrationTimeoutRef.current) {
        clearTimeout(calibrationTimeoutRef.current);
        calibrationTimeoutRef.current = null;
      }
    }
    return () => {
      if (calibrationTimeoutRef.current) {
        clearTimeout(calibrationTimeoutRef.current);
        calibrationTimeoutRef.current = null;
      }
    };
  }, [phase, toast]);

  const playRewardTone = useCallback(() => {
    if (!audioEnabled) return;
    try {
      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContext();
      }
      const ctx = audioCtxRef.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = 523.25; // C5
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.3);
    } catch {
      // Audio not available
    }
  }, [audioEnabled]);

  const handleStartRequest = () => {
    if (!isStreaming) return;
    // If session is too soon, show a gentle suggestion (not a gate)
    if (schedule.isTooSoon) {
      setShowTooSoonDialog(true);
      return;
    }
    doStart();
  };

  const doStart = async () => {
    if (!isStreaming) return;
    setShowTooSoonDialog(false);
    sessionStartRef.current = new Date();
    try {
      const result = await startNeurofeedback(selectedProtocol, true);
      if (result.status === "calibrating") {
        setPhase("calibrating");
        setCalibrationProgress(0);
        startEvalLoop();
      } else {
        setPhase("training");
        startEvalLoop();
      }
      startTimer();
    } catch (e) {
      console.error("Failed to start neurofeedback:", e);
    }
  };

  const startEvalLoop = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(async () => {
      // Use real band powers from live Muse 2 data
      const liveBands = bandPowers ?? {
        delta: 0,
        theta: 0,
        alpha: 0,
        beta: 0,
        gamma: 0,
      };

      try {
        const evalResult = await evaluateNeurofeedback(liveBands);

        if (evalResult.status === "calibrating") {
          setCalibrationProgress((evalResult.progress || 0) * 100);
        } else if (evalResult.status === "calibration_complete") {
          setPhase("training");
        } else if (evalResult.status === "active") {
          setScore(evalResult.score || 0);
          setStreak(evalResult.streak || 0);
          if (evalResult.reward) {
            setReward(true);
            setRewardCount((c) => c + 1);
            playRewardTone();
            setTimeout(() => setReward(false), 500);
          }
        }
      } catch {
        // Evaluation failed
      }
    }, 1000);
  };

  const startTimer = () => {
    setElapsed(0);
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
  };

  const handleStop = async () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (timerRef.current) clearInterval(timerRef.current);

    // Write mindful session to HealthKit / Health Connect (fire-and-forget)
    const sessionEnd = new Date();
    const sessionStart = sessionStartRef.current ?? new Date(sessionEnd.getTime() - elapsed * 1000);
    const durationMin = Math.max(1, Math.round(elapsed / 60));
    writeMindfulSession(sessionStart, sessionEnd, durationMin).catch(() => {});

    try {
      const result = await stopNeurofeedback();
      // Record completed session for scheduling
      recordNeurofeedbackSession();
      setSessionHistory(getSessionHistory());
      setStats(result.stats);
      setPhase("summary");
    } catch {
      setPhase("idle");
    }
  };

  const handleReset = () => {
    setPhase("idle");
    setStats(null);
    setScore(0);
    setRewardCount(0);
    setStreak(0);
    setElapsed(0);
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  const circumference = 2 * Math.PI * 80;
  const dashOffset = circumference * (1 - score / 100);

  return (
    <main className="p-4 md:p-6 pb-24 space-y-6">
      {/* Connection Banner */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          Neurofeedback is an EEG-only mode. Use voice and health features without hardware, or connect Muse to train here.
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Headphones className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold font-bold">Neurofeedback Training</h2>
        </div>
        {phase !== "idle" && phase !== "summary" && (
          <div className="flex items-center gap-3">
            <Timer className="h-4 w-4 text-foreground/50" />
            <span className="text-sm font-mono text-foreground/70">{formatTime(elapsed)}</span>
            <Button
              size="sm"
              variant="destructive"
              onClick={handleStop}
              className="bg-destructive/10 border border-destructive/30 text-destructive"
            >
              <Square className="h-3 w-3 mr-1" />
              Stop
            </Button>
          </div>
        )}
      </div>

      {/* Session Schedule Card — shown in idle phase */}
      {phase === "idle" && (
        <NeurofeedbackScheduleCard schedule={schedule} />
      )}

      {/* Pre-session substance questionnaire */}
      {phase === "idle" && showSubstanceQ && (
        <SubstanceQuestionnaire
          onComplete={() => setShowSubstanceQ(false)}
        />
      )}

      {/* Substance context note */}
      {phase === "idle" && !showSubstanceQ && substanceNote && (
        <SubstanceContextNote note={substanceNote} />
      )}

      {/* Idle: Protocol Selection */}
      {phase === "idle" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="glass-card p-6 rounded-xl hover-glow">
            <h3 className="text-lg font-semibold mb-4">Select Protocol</h3>
            <div className="space-y-4">
              <Select value={selectedProtocol} onValueChange={setSelectedProtocol}>
                <SelectTrigger className="w-full bg-card/50 border border-primary/30">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(protocols).map(([key, proto]) => (
                    <SelectItem key={key} value={key}>
                      {proto.name}
                    </SelectItem>
                  ))}
                  {Object.keys(protocols).length === 0 && (
                    <>
                      <SelectItem value="alpha_up">Alpha Enhancement</SelectItem>
                      <SelectItem value="smr_up">SMR Training</SelectItem>
                      <SelectItem value="theta_beta_ratio">Theta/Beta Ratio</SelectItem>
                      <SelectItem value="alpha_asymmetry">Alpha Asymmetry</SelectItem>
                    </>
                  )}
                </SelectContent>
              </Select>

              {protocols[selectedProtocol] && (
                <p className="text-sm text-foreground/60">
                  {protocols[selectedProtocol].description}
                </p>
              )}

              <div className="flex items-center justify-between">
                <Label className="text-sm">Audio Feedback</Label>
                <div className="flex items-center gap-2">
                  {audioEnabled ? <Volume2 className="h-4 w-4 text-primary" /> : <VolumeX className="h-4 w-4 text-foreground/50" />}
                  <Switch checked={audioEnabled} onCheckedChange={setAudioEnabled} />
                </div>
              </div>

              <Button
                onClick={handleStartRequest}
                disabled={!isStreaming}
                className="w-full bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30 disabled:opacity-50"
              >
                <Play className="h-4 w-4 mr-2" />
                {isStreaming ? "Start Training" : "EEG Required For Training"}
              </Button>
            </div>
          </Card>

          <Card className="glass-card p-6 rounded-xl hover-glow">
            <h3 className="text-lg font-semibold mb-4">How It Works</h3>
            <div className="space-y-3 text-sm text-foreground/70">
              <p>1. <strong>Connect Muse 2</strong> from EEG Setup and start streaming</p>
              <p>2. <strong>Calibration</strong> (30s): Measure your baseline brain activity</p>
              <p>3. <strong>Training</strong>: Watch the gauge and try to increase your score</p>
              <p>4. <strong>Rewards</strong>: Audio/visual feedback when you hit the target</p>
            </div>
            {isStreaming && bandPowers && (
              <div className="mt-4 pt-4 border-t border-border/30">
                <p className="text-xs text-muted-foreground mb-2">Current Band Powers:</p>
                <div className="grid grid-cols-5 gap-2">
                  {Object.entries(bandPowers).map(([band, power]) => (
                    <div key={band} className="text-center">
                      <p className="text-[10px] text-muted-foreground capitalize">{band}</p>
                      <p className="text-xs font-mono text-primary">{((power as number) * 100).toFixed(0)}%</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </Card>
        </div>
      )}

      {/* Calibrating */}
      {phase === "calibrating" && (
        <Card className="glass-card p-8 rounded-xl hover-glow max-w-lg mx-auto text-center" aria-live="polite">
          <h3 className="text-lg font-semibold mb-4">Calibrating Baseline</h3>
          <p className="text-sm text-foreground/60 mb-6">
            Relax and breathe normally. Measuring your baseline brain activity from your Muse 2...
          </p>
          <Progress value={calibrationProgress} className="h-3 mb-4" />
          <p className="text-xs font-mono text-foreground/50">
            {calibrationProgress.toFixed(0)}%
          </p>
        </Card>
      )}

      {/* Training */}
      {phase === "training" && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Score Gauge */}
          <div className="lg:col-span-2 flex justify-center" aria-live="polite" aria-label="Neurofeedback score">
            <Card
              className={`glass-card p-8 rounded-xl hover-glow w-full flex flex-col items-center transition-colors duration-300 ${
                reward ? "bg-success/5 border-success/30" : ""
              }`}
            >
              <svg width="200" height="200" viewBox="0 0 200 200">
                <circle cx="100" cy="100" r="80" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="12" />
                <circle
                  cx="100"
                  cy="100"
                  r="80"
                  fill="none"
                  stroke={score >= 70 ? "hsl(142, 71%, 45%)" : score >= 40 ? "hsl(48, 96%, 53%)" : "hsl(215, 20%, 65%)"}
                  strokeWidth="12"
                  strokeDasharray={circumference}
                  strokeDashoffset={dashOffset}
                  strokeLinecap="round"
                  transform="rotate(-90 100 100)"
                  className="transition-all duration-300"
                />
                <text x="100" y="95" textAnchor="middle" fill="white" fontSize="36" fontFamily="monospace" fontWeight="bold">
                  {score.toFixed(0)}
                </text>
                <text x="100" y="120" textAnchor="middle" fill="rgba(255,255,255,0.5)" fontSize="12" fontFamily="monospace">
                  SCORE
                </text>
              </svg>
            </Card>
          </div>

          {/* Stats Panel */}
          <div className="space-y-4">
            <Card className="glass-card p-4 rounded-xl">
              <div className="flex items-center gap-2 mb-2">
                <Trophy className="h-4 w-4 text-warning" />
                <span className="text-sm font-medium">Rewards</span>
              </div>
              <p className="text-2xl font-mono font-bold text-warning">{rewardCount}</p>
            </Card>

            <Card className="glass-card p-4 rounded-xl">
              <div className="flex items-center gap-2 mb-2">
                <Target className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">Streak</span>
              </div>
              <p className="text-2xl font-mono font-bold text-primary">{streak}</p>
            </Card>

            <Card className="glass-card p-4 rounded-xl">
              <div className="flex items-center gap-2 mb-2">
                <Timer className="h-4 w-4 text-foreground/60" />
                <span className="text-sm font-medium">Session</span>
              </div>
              <p className="text-2xl font-mono font-bold">{formatTime(elapsed)}</p>
            </Card>
          </div>
        </div>
      )}

      {/* Summary */}
      {phase === "summary" && stats && (
        <Card className="glass-card p-8 rounded-xl hover-glow max-w-lg mx-auto">
          <div className="text-center space-y-6">
            <Trophy className="h-12 w-12 text-warning mx-auto" />
            <h3 className="text-xl font-semibold">Session Complete</h3>

            <div className="grid grid-cols-2 gap-4 text-center">
              <div>
                <p className="text-xs text-foreground/50">Total Rewards</p>
                <p className="text-2xl font-mono font-bold text-warning">{stats.total_rewards}</p>
              </div>
              <div>
                <p className="text-xs text-foreground/50">Reward Rate</p>
                <p className="text-2xl font-mono font-bold text-primary">
                  {(stats.reward_rate * 100).toFixed(0)}%
                </p>
              </div>
              <div>
                <p className="text-xs text-foreground/50">Avg Score</p>
                <p className="text-2xl font-mono font-bold">{stats.avg_score.toFixed(0)}</p>
              </div>
              <div>
                <p className="text-xs text-foreground/50">Best Streak</p>
                <p className="text-2xl font-mono font-bold text-success">{stats.max_streak}</p>
              </div>
            </div>

            <div className="flex gap-3 justify-center">
              <Button onClick={handleReset} className="bg-primary/20 border border-primary/30 text-primary">
                New Session
              </Button>
            </div>
          </div>
        </Card>
      )}
      {/* Too-Soon Dialog — gentle suggestion, never blocks */}
      <Dialog open={showTooSoonDialog} onOpenChange={setShowTooSoonDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Your brain benefits from rest</DialogTitle>
            <DialogDescription>
              Research shows 2-3 day spacing between sessions leads to better
              long-term results. Want to do a quick meditation instead?
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex gap-2 sm:justify-end">
            <Button
              variant="outline"
              onClick={doStart}
              className="border-primary/30 text-foreground"
            >
              Start anyway
            </Button>
            <Button
              onClick={() => {
                setShowTooSoonDialog(false);
                window.location.href = "/inner-energy";
              }}
              className="bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30"
            >
              Try meditation
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </main>
  );
}
