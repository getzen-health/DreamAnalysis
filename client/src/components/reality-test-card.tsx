/**
 * RealityTestCard — daytime lucid dreaming habit prompt.
 *
 * Ask "Am I dreaming?" and log the response. Shows today's count + streak.
 * Backed by POST /api/reality-test and GET /api/reality-test/:userId.
 */

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Eye, Flame, CheckCircle2 } from "lucide-react";
import { getParticipantId } from "@/lib/participant";
import { apiRequest } from "@/lib/queryClient";

const USER_ID = getParticipantId();

const CHECKS = [
  { label: "Look at your hands", sub: "Count your fingers — wrong count = dreaming" },
  { label: "Read text twice", sub: "In dreams, text changes between readings" },
  { label: "Push finger through palm", sub: "It passes through in dreams" },
];

type Result = "dreaming" | "awake" | "unsure";

interface Stats {
  todayCount: number;
  streak: number;
  totalCount: number;
}

export function RealityTestCard() {
  const qc = useQueryClient();
  const [step, setStep] = useState<"idle" | "checking" | "done">("idle");
  const [checkIdx, setCheckIdx] = useState(0);
  const [result, setResult] = useState<Result | null>(null);

  const { data: stats } = useQuery<Stats>({
    queryKey: ["/api/reality-test", USER_ID],
    queryFn: async () => {
      const res = await apiRequest("GET", `/api/reality-test/${USER_ID}`);
      if (!res.ok) return { todayCount: 0, streak: 0, totalCount: 0 };
      return res.json();
    },
    staleTime: 60 * 1000,
  });

  const { mutate: logTest, isPending } = useMutation({
    mutationFn: async (r: Result) =>
      apiRequest("POST", "/api/reality-test", { userId: USER_ID, result: r }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["/api/reality-test", USER_ID] });
      setStep("done");
    },
  });

  const startCheck = () => {
    setCheckIdx(0);
    setResult(null);
    setStep("checking");
  };

  const nextCheck = () => {
    if (checkIdx < CHECKS.length - 1) {
      setCheckIdx((i) => i + 1);
    } else {
      setStep("done");
    }
  };

  const handleResult = (r: Result) => {
    setResult(r);
    logTest(r);
  };

  const todayCount = stats?.todayCount ?? 0;
  const streak = stats?.streak ?? 0;
  const target = 10;

  return (
    <Card className="rounded-[14px] bg-card border border-border p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Eye className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-medium">Reality Check</h3>
        </div>
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          {streak > 0 && (
            <span className="flex items-center gap-1">
              <Flame className="h-3.5 w-3.5 text-orange-400" />
              {streak}d streak
            </span>
          )}
          <span>
            {todayCount}/{target} today
          </span>
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 rounded-full bg-muted">
        <div
          className="h-1.5 rounded-full bg-primary transition-all"
          style={{ width: `${Math.min(100, (todayCount / target) * 100)}%` }}
        />
      </div>

      {/* States */}
      {step === "idle" && (
        <div className="space-y-3">
          <p className="text-xs text-muted-foreground leading-relaxed">
            10+ reality checks per day significantly increases lucid dream frequency (Baird et al. 2019).
          </p>
          <button
            onClick={startCheck}
            className="w-full py-2.5 rounded-lg border border-primary/30 bg-primary/5 text-sm font-medium text-primary hover:bg-primary/10 transition-colors"
          >
            Am I dreaming right now?
          </button>
        </div>
      )}

      {step === "checking" && (
        <div className="space-y-3">
          <div className="rounded-lg bg-muted/30 p-3 space-y-1">
            <p className="text-sm font-medium">{CHECKS[checkIdx].label}</p>
            <p className="text-xs text-muted-foreground">{CHECKS[checkIdx].sub}</p>
          </div>
          {checkIdx < CHECKS.length - 1 ? (
            <button
              onClick={nextCheck}
              className="w-full py-2 rounded-lg border border-border text-xs text-muted-foreground hover:border-muted-foreground transition-colors"
            >
              Next check →
            </button>
          ) : (
            <div className="space-y-2">
              <p className="text-xs text-center text-muted-foreground">What's your conclusion?</p>
              <div className="flex gap-2">
                {(["awake", "unsure", "dreaming"] as Result[]).map((r) => (
                  <button
                    key={r}
                    onClick={() => handleResult(r)}
                    disabled={isPending}
                    className={`flex-1 py-2 rounded-lg border text-xs font-medium transition-colors ${
                      r === "dreaming"
                        ? "border-violet-500/50 text-violet-400 hover:bg-violet-500/10"
                        : r === "unsure"
                        ? "border-amber-500/50 text-amber-400 hover:bg-amber-500/10"
                        : "border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/10"
                    }`}
                  >
                    {r.charAt(0).toUpperCase() + r.slice(1)}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {step === "done" && (
        <div className="flex items-center gap-3">
          <CheckCircle2 className="h-5 w-5 text-emerald-400 shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-medium">
              {result === "dreaming" ? "You're dreaming! 🎉" : result === "unsure" ? "Stay curious — check again soon" : "Awake and aware"}
            </p>
            <p className="text-xs text-muted-foreground">{todayCount} checks today</p>
          </div>
          <button
            onClick={startCheck}
            className="text-xs text-primary hover:underline"
          >
            Again
          </button>
        </div>
      )}
    </Card>
  );
}
