import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient } from "@/lib/queryClient";
import {
  predictFoodEmotion,
  calibrateFoodEmotion,
  type FoodEmotionResult,
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
  craving_carbs: "#f59e0b",
  appetite_suppressed: "#6366f1",
  comfort_seeking: "#ec4899",
  balanced: "#10b981",
  stress_eating: "#ef4444",
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
// Page
// ---------------------------------------------------------------------------

export default function FoodEmotion() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const liveEmotions = latestFrame?.analysis?.emotions;

  const { data, isLoading } = useQuery<FoodEmotionResult>({
    queryKey: ["food-emotion"],
    queryFn: () => predictFoodEmotion(),
    refetchInterval: 3000,
    retry: 1,
  });

  const calibrate = useMutation({
    mutationFn: () => calibrateFoodEmotion(),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["food-emotion"] }),
  });

  // --- Determine displayed state + probabilities ----------------------------
  // When EEG is streaming, derive food states from live EEG indices.
  // When no device, fall back to ML API simulation result.
  const liveProbs: Record<string, number> | null =
    isStreaming && liveEmotions
      ? deriveFoodStates(liveEmotions)
      : null;

  const stateProbabilities: Record<string, number> =
    liveProbs ?? data?.state_probabilities ?? {};

  const state      = liveProbs ? topState(liveProbs) : (data?.food_state ?? "balanced");
  const stateLabel = STATE_LABELS[state] ?? state;
  const stateColor = STATE_COLORS[state] ?? "#10b981";

  // Confidence: from live EEG use top-state probability; from ML API use as-is
  const confidence = liveProbs
    ? (stateProbabilities[state] ?? 0)
    : (data?.confidence ?? 0);

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

  const hasChartData = chartData.some(d => d.value > 0);

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
              <Badge variant="outline" className="text-xs border-emerald-500/40 text-emerald-400 bg-emerald-500/10">
                Live EEG
              </Badge>
            )}
          </div>
          <p className="text-muted-foreground text-sm">
            EEG-based appetite and eating-state analysis
          </p>
        </div>
      </div>

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
                {stateLabel}
              </Badge>
              {isLoading && !isStreaming && (
                <RefreshCw className="h-4 w-4 animate-spin text-muted-foreground" />
              )}
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Confidence</span>
                <span className="font-mono">
                  {isStreaming && !liveEmotions
                    ? "--"
                    : `${(confidence * 100).toFixed(1)}%`}
                </span>
              </div>
              <Progress value={confidence * 100} className="h-2" />
            </div>

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

            {isStreaming && !liveEmotions && (
              <p className="text-xs text-muted-foreground">
                Waiting for EEG data…
              </p>
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
                  <CheckCircle className="h-4 w-4 text-emerald-500" />
                  <span className="text-sm text-emerald-500 font-medium">
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
                ? "Connect your Muse and wait for EEG data to populate the chart."
                : "Log food sessions to see your craving patterns."}
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
                <p className="text-xs font-semibold text-emerald-500 uppercase mb-2">
                  Prefer
                </p>
                <ul className="space-y-1">
                  {data.recommendations.prefer.map((item, i) => (
                    <li key={i} className="text-sm flex items-start gap-1">
                      <span className="text-emerald-500 mt-0.5">•</span> {item}
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
