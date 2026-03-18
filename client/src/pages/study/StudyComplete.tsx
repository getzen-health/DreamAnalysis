import { useEffect } from "react";
import { useLocation, useSearch } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, CartesianGrid,
} from "recharts";
import { CheckCircle2, ChevronRight, Brain, Zap, FlaskConical, Compass } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";

// ── Interpretation helper ────────────────────────────────────────────────────

function interpretStress(pre: number, post: number): string {
  if (pre === 0 && post === 0) return "Your EEG data has been saved for analysis.";
  const diff = pre - post;
  const pct  = pre > 0 ? Math.round(Math.abs(diff) / pre * 100) : 0;
  if (diff > 0.05)  return `Your stress dropped ${pct}% after the breathing exercise — the intervention worked.`;
  if (diff < -0.05) return `Stress increased ${pct}% — this can happen. Your data is still valuable for the study.`;
  return "Stress stayed stable throughout the session — a solid baseline for comparison.";
}

// ── Stress arc chart ─────────────────────────────────────────────────────────

function StressArcChart({ pre, peak, post }: { pre: number; peak: number; post: number }) {
  const data = [
    { label: "Baseline", stress: pre },
    { label: "Peak",     stress: peak },
    { label: "Recovery", stress: post },
  ];

  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground text-center">Stress arc (0 = calm, 1 = high)</p>
      <ResponsiveContainer width="100%" height={120}>
        <LineChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted)/0.3)" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            formatter={(v: number) => [`${(v * 100).toFixed(0)}%`, "Stress"]}
            contentStyle={{ background: "hsl(var(--background))", border: "1px solid hsl(var(--border))", fontSize: 12 }}
          />
          <ReferenceLine y={0.65} stroke="hsl(var(--destructive)/0.4)" strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="stress"
            stroke="hsl(var(--primary))"
            strokeWidth={2.5}
            dot={{ fill: "hsl(var(--primary))", r: 4 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function StudyComplete() {
  const [, navigate] = useLocation();
  const search = useSearch();

  const params          = new URLSearchParams(search);
  const participantCode = params.get("code")  ?? "";
  const done            = params.get("done")   ?? "";
  const isPartial       = params.get("partial") === "true";
  const preStress       = parseFloat(params.get("pre_stress")  ?? "0");
  const peakStress      = parseFloat(params.get("peak_stress") ?? "0");
  const postStress      = parseFloat(params.get("post_stress") ?? "0");
  const hasStressData   = preStress > 0 || postStress > 0;

  // ── Persist completed session to localStorage ────────────────────────────

  useEffect(() => {
    if (!participantCode) return;
    if (done === "stress") localStorage.setItem(`study_completed_stress_${participantCode}`, "true");
    if (done === "food")   localStorage.setItem(`study_completed_food_${participantCode}`, "true");
  }, [participantCode, done]);

  // ── Derive completion state ───────────────────────────────────────────────

  const stressDone = done === "stress" ||
    localStorage.getItem(`study_completed_stress_${participantCode}`) === "true";
  const foodDone   = done === "food" ||
    localStorage.getItem(`study_completed_food_${participantCode}`)   === "true";
  const bothDone   = stressDone && foodDone;

  // ── Navigation ────────────────────────────────────────────────────────────

  function goNextSession() {
    const block = !stressDone ? "stress" : "food";
    navigate(`/study/session?code=${encodeURIComponent(participantCode)}&block=${block}`);
  }

  async function exploreApp() {
    try { await apiRequest("PATCH", "/api/user/intent", { intent: "explore" }); } catch { /* non-fatal */ }
    navigate("/");
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-lg mx-auto px-4 py-12 space-y-8">

        {/* Checkmark + headline */}
        <div className="text-center space-y-4">
          <div className="flex justify-center">
            <div className="w-20 h-20 rounded-full bg-cyan-600/10 flex items-center justify-center">
              <CheckCircle2 className="h-10 w-10 text-cyan-400" />
            </div>
          </div>
          <div className="space-y-1">
            <h1 className="text-2xl font-bold">
              {isPartial ? "Session Saved" : "Session Complete"}
            </h1>
            {participantCode && (
              <p className="text-muted-foreground text-sm">
                Thank you,{" "}
                <Badge variant="outline" className="border-primary/40 text-primary font-mono">
                  {participantCode}
                </Badge>
              </p>
            )}
            {isPartial && (
              <p className="text-xs text-amber-400">
                Session ended early — partial data saved.
              </p>
            )}
          </div>
        </div>

        {/* Status badges */}
        <div className="flex justify-center gap-3">
          <Badge variant="outline" className={stressDone ? "border-cyan-500/50 text-cyan-400" : "border-muted text-muted-foreground"}>
            <Zap className="h-3 w-3 mr-1" />Stress {stressDone ? "done" : "pending"}
          </Badge>
          <Badge variant="outline" className={foodDone ? "border-cyan-500/50 text-cyan-400" : "border-muted text-muted-foreground"}>
            <Brain className="h-3 w-3 mr-1" />Food {foodDone ? "done" : "pending"}
          </Badge>
        </div>

        {/* Stress arc chart + interpretation */}
        {hasStressData && !isPartial && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Your stress arc</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <StressArcChart pre={preStress} peak={peakStress} post={postStress} />
              <p className="text-sm text-muted-foreground leading-relaxed">
                {interpretStress(preStress, postStress)}
              </p>
            </CardContent>
          </Card>
        )}

        {/* Next action card */}
        {bothDone ? (
          <Card className="border-cyan-500/20 bg-cyan-600/5">
            <CardContent className="pt-6 space-y-4 text-center">
              <FlaskConical className="h-8 w-8 text-cyan-400 mx-auto" />
              <p className="font-semibold">Both sessions complete!</p>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Your data is contributing to brain science. Thank you for participating
                in the AntarAI pilot study.
              </p>
              <Button className="w-full" onClick={exploreApp}>
                <Compass className="mr-2 h-4 w-4" />
                Explore the full app
                <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardContent className="pt-6 space-y-4">
              <p className="text-sm text-muted-foreground leading-relaxed">
                {!stressDone
                  ? "Next up: the Stress Session (~30 min). You can start now or return later."
                  : "Next up: the Food Session (~30 min). You can start now or return later."}
              </p>
              <Button className="w-full" size="lg" onClick={goNextSession}>
                {!stressDone ? <Zap className="mr-2 h-4 w-4" /> : <Brain className="mr-2 h-4 w-4" />}
                Start {!stressDone ? "Stress" : "Food"} Session
                <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Footer */}
        <div className="text-center text-xs text-muted-foreground space-y-1 pt-2">
          <p>Your data is anonymous and used exclusively for academic research.</p>
          <p>Participant code: <span className="font-mono">{participantCode || "—"}</span></p>
        </div>
      </div>
    </div>
  );
}
