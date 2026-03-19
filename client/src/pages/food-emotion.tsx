import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient } from "@/lib/queryClient";
import {
  predictFoodEmotion,
  calibrateFoodEmotion,
  type FoodEmotionResult,
  type FoodImageAnalysisResult,
} from "@/lib/ml-api";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Utensils, RefreshCw, CheckCircle } from "lucide-react";
import { FoodCapture } from "@/components/food-capture";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { useDevice } from "@/hooks/use-device";
import { useVoiceData } from "@/hooks/use-voice-data";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const STATE_LABELS: Record<string, string> = {
  craving_carbs: "Craving Carbs",
  appetite_suppressed: "Appetite Suppressed",
  comfort_seeking: "Comfort Seeking",
  balanced: "Balanced",
  stress_eating: "Stress Eating",
  mindful_eating: "Mindful Eating",
};

const STATE_COLORS: Record<string, string> = {
  craving_carbs: "#d4a017",
  appetite_suppressed: "#6366f1",
  comfort_seeking: "#d946ef",
  balanced: "#0891b2",
  stress_eating: "#e879a8",
  mindful_eating: "#3b82f6",
};

/** Map live EEG emotion analysis to food-state soft probabilities. */
function deriveFoodStates(emotions: {
  stress_index?: number;
  relaxation_index?: number;
  focus_index?: number;
  valence?: number;
}): Record<string, number> {
  const stress     = emotions.stress_index     ?? 0;
  const relaxation = emotions.relaxation_index ?? 0;
  const focus      = emotions.focus_index      ?? 0;
  const valence    = emotions.valence          ?? 0;

  const raw: Record<string, number> = {
    stress_eating:        Math.max(0, stress - 0.35) * 2.5,
    mindful_eating:       Math.max(0, relaxation - 0.35) * 2.5,
    comfort_seeking:      Math.max(0, -valence - 0.1) * 1.8,
    balanced:             Math.max(0, 1 - Math.abs(stress - 0.35) * 2 - Math.abs(relaxation - 0.35) * 2),
    craving_carbs:        Math.max(0, stress - 0.25) * Math.max(0, 0.55 - focus) * 4,
    appetite_suppressed:  Math.max(0, focus - 0.45) * Math.max(0, 0.45 - stress) * 4,
  };

  const total = Object.values(raw).reduce((a, b) => a + b, 0) || 1;
  return Object.fromEntries(
    Object.entries(raw).map(([k, v]) => [k, Math.max(0, v) / total])
  );
}

/** Pick the dominant state from a probability map. */
function topState(probs: Record<string, number>): string {
  return Object.entries(probs).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "balanced";
}

// ---------------------------------------------------------------------------
// FoodMoodInsights — pattern detection from localStorage food logs
// ---------------------------------------------------------------------------

function FoodMoodInsights() {
  const [insights, setInsights] = useState<string[]>([]);

  useEffect(() => {
    const patterns: string[] = [];

    try {
      const keys = Object.keys(localStorage).filter(k =>
        k.startsWith("ndw_food_log_")
      );
      const logs: Array<{
        emotion?: string;
        calories?: number;
        carbs?: number;
        protein?: number;
        fat?: number;
        timestamp: number;
      }> = [];

      keys.forEach(k => {
        try {
          const raw = JSON.parse(localStorage.getItem(k) || "{}");
          if (raw.timestamp) logs.push(raw);
        } catch { /* skip */ }
      });

      if (logs.length < 3) {
        setInsights([]);
        return;
      }

      // Pattern 1: High-calorie meals during stress
      const stressedMeals = logs.filter(
        l => l.emotion === "angry" || l.emotion === "fear"
      );
      const calmMeals = logs.filter(
        l => l.emotion === "happy" || l.emotion === "neutral"
      );
      if (stressedMeals.length >= 2 && calmMeals.length >= 2) {
        const avgStressCalories =
          stressedMeals.reduce((a, b) => a + (b.calories || 0), 0) /
          stressedMeals.length;
        const avgCalmCalories =
          calmMeals.reduce((a, b) => a + (b.calories || 0), 0) /
          calmMeals.length;
        if (avgStressCalories > avgCalmCalories * 1.2) {
          patterns.push(
            `You eat ~${Math.round(avgStressCalories - avgCalmCalories)} more calories when stressed`
          );
        }
      }

      // Pattern 2: Carb preference during negative emotions
      const negMeals = logs.filter(
        l =>
          l.emotion === "sad" ||
          l.emotion === "angry" ||
          l.emotion === "fear"
      );
      const posMeals = logs.filter(
        l => l.emotion === "happy" || l.emotion === "surprise"
      );
      if (negMeals.length >= 2 && posMeals.length >= 2) {
        const avgNegCarbs =
          negMeals.reduce((a, b) => a + (b.carbs || 0), 0) / negMeals.length;
        const avgPosCarbs =
          posMeals.reduce((a, b) => a + (b.carbs || 0), 0) / posMeals.length;
        if (avgNegCarbs > avgPosCarbs * 1.3) {
          patterns.push("You tend to eat more carbs when feeling down");
        }
      }

      // Pattern 3: Timing
      const morningLogs = logs.filter(
        l => new Date(l.timestamp).getHours() < 10
      );
      const eveningLogs = logs.filter(
        l => new Date(l.timestamp).getHours() >= 19
      );
      if (
        eveningLogs.length > morningLogs.length * 2 &&
        eveningLogs.length >= 3
      ) {
        patterns.push("Most of your meals are logged in the evening");
      }

      // Pattern 4: Protein on focus days
      const focusMeals = logs.filter(
        l => l.emotion === "neutral" || l.emotion === "surprise"
      );
      if (focusMeals.length >= 2) {
        const avgProtein =
          focusMeals.reduce((a, b) => a + (b.protein || 0), 0) /
          focusMeals.length;
        if (avgProtein > 20) {
          patterns.push("Your focused days tend to include more protein");
        }
      }
    } catch { /* ignore */ }

    setInsights(patterns);
  }, []);

  if (insights.length === 0) return null;

  return (
    <div
      style={{
        background: "var(--card)",
        border: "1px solid var(--border)",
        borderRadius: 14,
        padding: "12px 14px",
        marginBottom: 12,
      }}
    >
      <div
        style={{
          fontSize: 11,
          fontWeight: 600,
          color: "var(--muted-foreground)",
          textTransform: "uppercase" as const,
          letterSpacing: "0.5px",
          marginBottom: 8,
        }}
      >
        Food-Mood Patterns
      </div>
      {insights.map((insight, i) => (
        <div
          key={i}
          style={{
            fontSize: 12,
            color: "var(--foreground)",
            padding: "6px 0",
            borderTop: i > 0 ? "1px solid var(--border)" : "none",
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <span style={{ fontSize: 16 }}>
            {insight.includes("calorie")
              ? "🔥"
              : insight.includes("carb")
              ? "🍞"
              : insight.includes("evening")
              ? "🌙"
              : "💪"}
          </span>
          {insight}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// MindfulEatingPrompt — pause-before-eating check-in
// ---------------------------------------------------------------------------

function MindfulEatingPrompt({ onDismiss }: { onDismiss: () => void }) {
  return (
    <div
      style={{
        background:
          "linear-gradient(135deg, hsl(165 30% 12% / 0.9), hsl(200 30% 10% / 0.9))",
        border: "1px solid hsl(165 60% 45% / 0.2)",
        borderRadius: 14,
        padding: "14px 16px",
        marginBottom: 12,
        textAlign: "center" as const,
      }}
    >
      <div style={{ fontSize: 22, marginBottom: 6 }}>🧘</div>
      <div
        style={{
          fontSize: 13,
          fontWeight: 600,
          color: "var(--foreground)",
          marginBottom: 4,
        }}
      >
        Pause before eating
      </div>
      <div
        style={{
          fontSize: 11,
          color: "var(--muted-foreground)",
          marginBottom: 10,
          lineHeight: 1.4,
        }}
      >
        Are you physically hungry, or eating in response to an emotion?
      </div>
      <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
        <button
          onClick={onDismiss}
          style={{
            fontSize: 11,
            fontWeight: 600,
            padding: "6px 14px",
            borderRadius: 8,
            background: "hsl(165 60% 45% / 0.15)",
            color: "hsl(165 60% 55%)",
            border: "1px solid hsl(165 60% 45% / 0.3)",
            cursor: "pointer",
          }}
        >
          I'm hungry
        </button>
        <button
          onClick={onDismiss}
          style={{
            fontSize: 11,
            fontWeight: 600,
            padding: "6px 14px",
            borderRadius: 8,
            background: "hsl(38 85% 52% / 0.15)",
            color: "hsl(38 85% 62%)",
            border: "1px solid hsl(38 85% 52% / 0.3)",
            cursor: "pointer",
          }}
        >
          Emotional eating
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function FoodEmotion() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const [mealResult, setMealResult] = useState<FoodImageAnalysisResult | null>(null);
  const [showMindfulPrompt, setShowMindfulPrompt] = useState(true);
  const liveEmotions = latestFrame?.analysis?.emotions;
  const voiceData = useVoiceData();

  /** Save a food-mood correlation entry to localStorage after a meal is captured. */
  function saveFoodLog(result: FoodImageAnalysisResult) {
    try {
      const emotionRaw = localStorage.getItem("ndw_last_emotion");
      const emotion = emotionRaw
        ? (JSON.parse(emotionRaw)?.result?.emotion as string | undefined)
        : undefined;
      const logKey = `ndw_food_log_${Date.now()}`;
      localStorage.setItem(
        logKey,
        JSON.stringify({
          emotion,
          calories: result.total_calories || 0,
          carbs: result.total_carbs_g || 0,
          protein: result.total_protein_g || 0,
          fat: result.total_fat_g || 0,
          timestamp: Date.now(),
        })
      );
    } catch { /* ignore */ }
  }

  // Only call ML API when streaming with real signals — never simulate
  const { data, isLoading } = useQuery<FoodEmotionResult>({
    queryKey: ["food-emotion"],
    queryFn: () => predictFoodEmotion(),
    // Only fetch when NOT streaming (to get calibration status); no simulation fallback
    enabled: !isStreaming,
    refetchInterval: false,
    retry: 1,
  });

  const calibrate = useMutation({
    mutationFn: () => calibrateFoodEmotion(),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["food-emotion"] }),
  });

  // --- Determine displayed state + probabilities ----------------------------
  // Only show real EEG-derived food states when the device is streaming.
  // Never show simulated/hardcoded values.
  const liveProbs: Record<string, number> | null =
    isStreaming && liveEmotions
      ? deriveFoodStates(liveEmotions)
      : null;

  // Only use real EEG data — no simulation fallback
  const stateProbabilities: Record<string, number> = liveProbs ?? {};

  const state      = liveProbs ? topState(liveProbs) : "balanced";
  const stateLabel = STATE_LABELS[state] ?? state;
  const stateColor = STATE_COLORS[state] ?? "#0891b2";

  const confidence = liveProbs ? (stateProbabilities[state] ?? 0) : 0;

  // --- Chart data -----------------------------------------------------------
  const chartData = Object.entries(stateProbabilities)
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([key, value]) => ({
      key,
      name: STATE_LABELS[key] ?? key,
      value: Math.round(value * 100),
      color: STATE_COLORS[key] ?? "#888",
    }));

  const hasChartData = isStreaming && liveEmotions && chartData.some(d => d.value > 0);
  void isLoading;

  return (
    <div className="p-6 space-y-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-amber-500/10">
          <Utensils className="h-6 w-6 text-amber-500" />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold">Food &amp; Cravings</h1>
            {!isStreaming && data?.simulation_mode && (
              <Badge variant="outline" className="text-xs border-yellow-500/40 text-yellow-400 bg-yellow-500/10">
                Simulation Mode
              </Badge>
            )}
            {isStreaming && (
              <Badge variant="outline" className="text-xs border-cyan-500/40 text-cyan-400 bg-cyan-500/10">
                Live EEG
              </Badge>
            )}
          </div>
          <p className="text-muted-foreground text-sm">
            Appetite and eating-state analysis from your latest voice analysis, with EEG depth when available
          </p>
        </div>
      </div>

      {/* Mindful eating check-in — shown once per page load */}
      {showMindfulPrompt && (
        <MindfulEatingPrompt onDismiss={() => setShowMindfulPrompt(false)} />
      )}

      {/* Food-Mood pattern insights derived from past logs */}
      <FoodMoodInsights />

      {/* Current State + Calibration */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Current Food State */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Current Food State
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {!isStreaming ? (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Your food state is derived from your latest voice analysis. Tap the mic button in the bottom tabs to update.
                </p>
                <FoodCapture
                  onAnalyzed={(result) => {
                    setMealResult(result);
                    saveFoodLog(result);
                  }}
                />
                {voiceData && (
                  <div className="rounded-md bg-muted/40 p-3 text-xs space-y-1">
                    <div className="flex justify-between gap-3">
                      <span className="text-muted-foreground">Emotion</span>
                      <span className="font-medium capitalize">{voiceData.emotion ?? "—"}</span>
                    </div>
                    <div className="flex justify-between gap-3">
                      <span className="text-muted-foreground">Confidence</span>
                      <span className="font-mono">{Math.round((voiceData.confidence ?? 0) * 100)}%</span>
                    </div>
                  </div>
                )}
                {!voiceData && (
                  <p className="text-xs text-muted-foreground italic">
                    No voice analysis yet. Use the mic button in the bottom tabs to analyze your emotional state.
                  </p>
                )}
              </div>
            ) : (
              <>
              <div className="flex items-center gap-3">
                <Badge
                  style={{
                    backgroundColor: `${stateColor}20`,
                    color: stateColor,
                    borderColor: `${stateColor}40`,
                  }}
                  variant="outline"
                  className="text-base px-3 py-1 font-semibold"
                >
                  {liveEmotions ? stateLabel : "Waiting for EEG…"}
                </Badge>
              </div>
              {liveEmotions && (
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Confidence</span>
                  <span className="font-mono">{(confidence * 100).toFixed(1)}%</span>
                </div>
                <Progress value={confidence * 100} className="h-2" />
              </div>
              )}
              </>
            )}

            {/* Live EEG indices */}
            {isStreaming && liveEmotions && (
              <div className="pt-1 space-y-1 text-xs text-muted-foreground">
                <div className="flex justify-between">
                  <span>Stress index</span>
                  <span className="font-mono">{((liveEmotions.stress_index ?? 0) * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Relaxation index</span>
                  <span className="font-mono">{((liveEmotions.relaxation_index ?? 0) * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Focus index</span>
                  <span className="font-mono">{((liveEmotions.focus_index ?? 0) * 100).toFixed(0)}%</span>
                </div>
              </div>
            )}

          </CardContent>
        </Card>

        {/* Calibration */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Calibration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-2">
              {data?.is_calibrated ? (
                <>
                  <CheckCircle className="h-4 w-4 text-cyan-500" />
                  <span className="text-sm text-cyan-500 font-medium">
                    Calibrated
                  </span>
                </>
              ) : (
                <span className="text-sm text-muted-foreground">
                  {Math.round((data?.calibration_progress ?? 0) * 100)}%
                  complete
                </span>
              )}
            </div>
            {!data?.is_calibrated && (
              <Progress
                value={(data?.calibration_progress ?? 0) * 100}
                className="h-2"
              />
            )}
            <Button
              size="sm"
              variant="outline"
              onClick={() => calibrate.mutate()}
              disabled={calibrate.isPending}
              className="w-full"
            >
              {calibrate.isPending ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Calibrating…
                </>
              ) : (
                "Calibrate Now"
              )}
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* ── Meal nutritional breakdown (shown after image capture) ───────────── */}
      {mealResult && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Utensils className="h-4 w-4 text-amber-400" />
              Meal Captured
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-xs text-muted-foreground">{mealResult.summary}</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs text-center">
              <div className="rounded-md bg-muted/30 p-2">
                <p className="font-semibold tabular-nums">{Math.round(mealResult.total_calories)}</p>
                <p className="text-muted-foreground">kcal</p>
              </div>
              <div className="rounded-md bg-muted/30 p-2">
                <p className="font-semibold tabular-nums">{mealResult.total_protein_g}g</p>
                <p className="text-muted-foreground">protein</p>
              </div>
              <div className="rounded-md bg-muted/30 p-2">
                <p className="font-semibold tabular-nums">{mealResult.total_carbs_g}g</p>
                <p className="text-muted-foreground">carbs</p>
              </div>
              <div className="rounded-md bg-muted/30 p-2">
                <p className="font-semibold tabular-nums">{mealResult.total_fat_g}g</p>
                <p className="text-muted-foreground">fat</p>
              </div>
            </div>
            <div className="flex gap-1.5 flex-wrap">
              <Badge variant="outline" className="text-xs">
                GI: {mealResult.glycemic_impact}
              </Badge>
              <Badge variant="outline" className="text-xs">
                {mealResult.dominant_macro}-dominant
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Food State Distribution Chart (US-011 / US-012) ─────────────────── */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">
            {isStreaming ? "Live EEG Food State Distribution" : "Food Craving Pattern"}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {!hasChartData ? (
            <p className="text-xs text-muted-foreground py-6 text-center">
              {isStreaming
                ? "Waiting for EEG data — keep your Muse on for a few seconds."
                : "Log a meal and use the mic button in the bottom tabs for voice analysis, or connect Muse for real-time EEG-based food state analysis."}
            </p>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart
                data={chartData}
                margin={{ top: 4, right: 8, left: -20, bottom: 0 }}
              >
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                  tickLine={false}
                  axisLine={false}
                  interval={0}
                  angle={-25}
                  textAnchor="end"
                  height={48}
                />
                <YAxis
                  tickFormatter={(v) => `${v}%`}
                  tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  formatter={(value: number) => [`${value}%`, "Probability"]}
                  contentStyle={{
                    background: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={48}>
                  {chartData.map((entry) => (
                    <Cell key={entry.key} fill={entry.color} opacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      {/* Dietary Recommendations */}
      {data?.recommendations && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Utensils className="h-4 w-4" />
              Dietary Guidance
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <p className="text-xs font-semibold text-cyan-500 uppercase mb-2">
                  Prefer
                </p>
                <ul className="space-y-1">
                  {data.recommendations.prefer.map((item, i) => (
                    <li key={i} className="text-sm flex items-start gap-1">
                      <span className="text-cyan-500 mt-0.5">•</span> {item}
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="text-xs font-semibold text-rose-500 uppercase mb-2">
                  Avoid
                </p>
                <ul className="space-y-1">
                  {data.recommendations.avoid.map((item, i) => (
                    <li key={i} className="text-sm flex items-start gap-1">
                      <span className="text-rose-500 mt-0.5">•</span> {item}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            <div className="rounded-md bg-muted/40 p-3 space-y-2">
              <p className="text-sm">
                <span className="font-medium">Strategy:</span>{" "}
                {data.recommendations.strategy}
              </p>
              <p className="text-sm text-muted-foreground italic">
                {data.recommendations.mindfulness_tip}
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
