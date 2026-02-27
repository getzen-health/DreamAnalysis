import { useEffect } from "react";
import { useLocation, useSearch } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, ChevronRight, Brain, Zap, FlaskConical } from "lucide-react";

export default function StudyComplete() {
  const [, navigate] = useLocation();
  const search = useSearch();

  const params = new URLSearchParams(search);
  const participantCode = params.get("code") ?? "";
  const done = params.get("done") ?? "";

  // ── Persist completed session to localStorage ──────────────────────────────

  useEffect(() => {
    if (!participantCode) return;
    if (done === "stress") {
      localStorage.setItem(`study_completed_stress_${participantCode}`, "true");
    } else if (done === "food") {
      localStorage.setItem(`study_completed_food_${participantCode}`, "true");
    }
  }, [participantCode, done]);

  // ── Derive completion state ────────────────────────────────────────────────

  const stressDone =
    done === "stress" ||
    localStorage.getItem(`study_completed_stress_${participantCode}`) === "true";
  const foodDone =
    done === "food" ||
    localStorage.getItem(`study_completed_food_${participantCode}`) === "true";
  const bothDone = stressDone && foodDone;

  // ── Navigation targets ─────────────────────────────────────────────────────

  function goFood() {
    navigate(`/study/session/food?code=${encodeURIComponent(participantCode)}`);
  }

  function goStress() {
    navigate(`/study/session/stress?code=${encodeURIComponent(participantCode)}`);
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-lg mx-auto px-4 py-16 space-y-8">

        {/* Checkmark + headline */}
        <div className="text-center space-y-4">
          <div className="flex justify-center">
            <div className="w-20 h-20 rounded-full bg-green-500/10 flex items-center justify-center">
              <CheckCircle2 className="h-10 w-10 text-green-400" />
            </div>
          </div>
          <div className="space-y-1">
            <h1 className="text-2xl font-bold">Session Complete</h1>
            {participantCode && (
              <p className="text-muted-foreground text-sm">
                Thank you,{" "}
                <Badge variant="outline" className="border-primary/40 text-primary font-mono">
                  {participantCode}
                </Badge>
              </p>
            )}
          </div>
        </div>

        {/* Status badges */}
        <div className="flex justify-center gap-3">
          <Badge
            variant="outline"
            className={
              stressDone
                ? "border-green-500/50 text-green-400"
                : "border-muted text-muted-foreground"
            }
          >
            <Zap className="h-3 w-3 mr-1" />
            Stress {stressDone ? "done" : "pending"}
          </Badge>
          <Badge
            variant="outline"
            className={
              foodDone
                ? "border-green-500/50 text-green-400"
                : "border-muted text-muted-foreground"
            }
          >
            <Brain className="h-3 w-3 mr-1" />
            Food {foodDone ? "done" : "pending"}
          </Badge>
        </div>

        {/* Context card */}
        {bothDone ? (
          <Card className="border-green-500/20 bg-green-500/5">
            <CardContent className="pt-6 space-y-4 text-center">
              <FlaskConical className="h-8 w-8 text-green-400 mx-auto" />
              <p className="font-semibold">You have completed both sessions.</p>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Your data is helping advance brain science. Thank you for your contribution
                to the Neural Dream Workshop pilot study.
              </p>
            </CardContent>
          </Card>
        ) : done === "stress" ? (
          <Card>
            <CardContent className="pt-6 space-y-4">
              <p className="text-sm text-muted-foreground leading-relaxed">
                You have completed the Stress Session. Next step: complete the Food Session.
              </p>
              <p className="text-xs text-muted-foreground">
                You can do the food session now or come back later. Use the same
                participant code.
              </p>
              <Button className="w-full" size="lg" onClick={goFood}>
                <Brain className="mr-2 h-4 w-4" />
                Start Food Session
                <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            </CardContent>
          </Card>
        ) : done === "food" ? (
          <Card>
            <CardContent className="pt-6 space-y-4">
              <p className="text-sm text-muted-foreground leading-relaxed">
                You have completed the Food Session. Next step: complete the Stress Session.
              </p>
              <p className="text-xs text-muted-foreground">
                You can do the stress session now or come back later. Use the same
                participant code.
              </p>
              <Button className="w-full" size="lg" onClick={goStress}>
                <Zap className="mr-2 h-4 w-4" />
                Start Stress Session
                <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardContent className="pt-6 space-y-4">
              <p className="text-sm text-muted-foreground">
                Your session has been recorded. Check with your study coordinator for next steps.
              </p>
            </CardContent>
          </Card>
        )}

        {/* Science note */}
        <div className="text-center text-xs text-muted-foreground space-y-1 pt-2">
          <p>Your data is anonymous and used exclusively for academic research.</p>
          <p>Participant code: <span className="font-mono">{participantCode || "—"}</span></p>
        </div>
      </div>
    </div>
  );
}
