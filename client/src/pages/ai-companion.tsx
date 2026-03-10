import { AICompanion } from "@/components/ai-companion";
import { getParticipantId } from "@/lib/participant";

export default function AICompanionPage() {
  const userId = getParticipantId();

  return (
    <main className="p-4 md:p-6 space-y-6">
      <AICompanion userId={userId} />
    </main>
  );
}
