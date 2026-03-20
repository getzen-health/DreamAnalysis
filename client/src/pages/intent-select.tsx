import { useState } from "react";
import { useLocation } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2, Brain } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";

export default function IntentSelect() {
  const [, navigate] = useLocation();
  const [loading, setLoading] = useState<"explore" | null>(null);

  async function choose() {
    setLoading("explore");
    try {
      await apiRequest("PATCH", "/api/user/intent", { intent: "explore" });
    } catch (_) {
      // non-fatal
    }
    navigate("/");
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4">
      <div className="max-w-xl w-full space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold">Welcome to AntarAI</h1>
          <p className="text-muted-foreground">Your personal wellness companion</p>
        </div>

        <Card
          className="cursor-pointer border-2 hover:border-primary transition-colors"
          onClick={() => !loading && choose()}
        >
          <CardContent className="pt-8 pb-8 text-center space-y-4">
            <Brain className="w-10 h-10 mx-auto text-violet-400" />
            <div>
              <p className="font-semibold text-lg">Get Started</p>
              <p className="text-sm text-muted-foreground mt-1">
                Explore the full dashboard — emotions, health, brain monitoring, and more.
              </p>
            </div>
            {loading && <Loader2 className="w-4 h-4 mx-auto animate-spin" />}
          </CardContent>
        </Card>

        <p className="text-center text-xs text-muted-foreground">
          Powered by AI and neuroscience.
        </p>
      </div>
    </div>
  );
}
