import { useState } from "react";
import { useLocation } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2, FlaskConical, Brain } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";

export default function IntentSelect() {
  const [, navigate] = useLocation();
  const [loading, setLoading] = useState<"study" | "explore" | null>(null);

  async function choose(intent: "study" | "explore") {
    setLoading(intent);
    try {
      await apiRequest("PATCH", "/api/user/intent", { intent });
    } catch (_) {
      // non-fatal — intent will be re-asked next login if save fails
    }
    navigate(intent === "study" ? "/study" : "/");
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4">
      <div className="max-w-xl w-full space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold">Welcome to AntarAI</h1>
          <p className="text-muted-foreground">What brings you here today?</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Card
            className="cursor-pointer border-2 hover:border-primary transition-colors"
            onClick={() => !loading && choose("study")}
          >
            <CardContent className="pt-8 pb-8 text-center space-y-4">
              <FlaskConical className="w-10 h-10 mx-auto text-primary" />
              <div>
                <p className="font-semibold text-lg">Join the Study</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Participate in EEG research sessions. ~30 min total.
                </p>
              </div>
              {loading === "study" && <Loader2 className="w-4 h-4 mx-auto animate-spin" />}
            </CardContent>
          </Card>

          <Card
            className="cursor-pointer border-2 hover:border-primary transition-colors"
            onClick={() => !loading && choose("explore")}
          >
            <CardContent className="pt-8 pb-8 text-center space-y-4">
              <Brain className="w-10 h-10 mx-auto text-violet-400" />
              <div>
                <p className="font-semibold text-lg">Explore the App</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Try the full dashboard, emotion lab, dream journal, and more.
                </p>
              </div>
              {loading === "explore" && <Loader2 className="w-4 h-4 mx-auto animate-spin" />}
            </CardContent>
          </Card>
        </div>

        <p className="text-center text-xs text-muted-foreground">
          You can switch later — study participants can explore after completing both sessions.
        </p>
      </div>
    </div>
  );
}
