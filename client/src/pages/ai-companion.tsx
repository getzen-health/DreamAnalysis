import { AICompanion } from "@/components/ai-companion";
import { getParticipantId } from "@/lib/participant";

export default function AICompanionPage() {
  const userId = getParticipantId();

  return (
    <main className="h-[calc(100vh-56px)] overflow-hidden">
      <AICompanion userId={userId} />
    </main>
  );
}
