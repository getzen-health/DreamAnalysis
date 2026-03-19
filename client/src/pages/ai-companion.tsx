import { AICompanion } from "@/components/ai-companion";
import { getParticipantId } from "@/lib/participant";
import { useLocation } from "wouter";
import { ArrowLeft } from "lucide-react";

export default function AICompanionPage() {
  const userId = getParticipantId();
  const [, navigate] = useLocation();

  return (
    <main className="p-4 md:p-6 pb-24 space-y-6">
      <button
        onClick={() => navigate("/")}
        className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors mb-2"
        aria-label="Back to dashboard"
      >
        <ArrowLeft size={20} />
        <span className="text-sm font-medium">Back</span>
      </button>
      <AICompanion userId={userId} />
    </main>
  );
}
