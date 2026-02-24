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
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Utensils, Activity, RefreshCw, CheckCircle } from "lucide-react";

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

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function BiomarkerGauge({
  label,
  value,
  min = 0,
  max = 1,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
}) {
  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono text-xs">{value.toFixed(3)}</span>
      </div>
      <Progress value={pct} className="h-2" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function FoodEmotion() {
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

  const state = data?.food_state ?? "balanced";
  const stateLabel = STATE_LABELS[state] ?? state;
  const stateColor = STATE_COLORS[state] ?? "#10b981";

  const probData = data
    ? Object.entries(data.state_probabilities).map(([k, v]) => ({
        name: STATE_LABELS[k] ?? k,
        value: Math.round(v * 100),
        color: STATE_COLORS[k] ?? "#6b7280",
      }))
    : [];

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
            {data?.simulation_mode && (
              <Badge variant="outline" className="text-xs border-yellow-500/40 text-yellow-400 bg-yellow-500/10">
                Simulation Mode
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
              {isLoading && (
                <RefreshCw className="h-4 w-4 animate-spin text-muted-foreground" />
              )}
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Confidence</span>
                <span className="font-mono">
                  {((data?.confidence ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
              <Progress
                value={(data?.confidence ?? 0) * 100}
                className="h-2"
              />
            </div>
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

      {/* Biomarkers */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Neural Biomarkers
          </CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <BiomarkerGauge
            label="Frontal Alpha Asymmetry (FAA)"
            value={data?.components.faa ?? 0}
            min={-1}
            max={1}
          />
          <BiomarkerGauge
            label="High-Beta (Stress / Anxiety)"
            value={data?.components.high_beta ?? 0}
            min={0}
            max={1}
          />
          <BiomarkerGauge
            label="Prefrontal Theta (Hunger / Urge)"
            value={data?.components.prefrontal_theta ?? 0}
            min={0}
            max={1}
          />
          <BiomarkerGauge
            label="Delta (Satiety / Deep Inhibition)"
            value={data?.components.delta ?? 0}
            min={0}
            max={1}
          />
        </CardContent>
      </Card>

      {/* State Probabilities */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">
            State Probabilities
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart
              data={probData}
              margin={{ top: 4, right: 8, left: 0, bottom: 4 }}
            >
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11 }} />
              <Tooltip
                formatter={(v) => [`${v}%`, "Probability"]}
                contentStyle={{ fontSize: 12 }}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {probData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
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
