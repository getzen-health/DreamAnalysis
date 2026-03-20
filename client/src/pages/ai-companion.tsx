import { AICompanion } from "@/components/ai-companion";
import { getParticipantId } from "@/lib/participant";

export default function AICompanionPage() {
  const userId = getParticipantId();

  return (
    <div style={{
      position: "fixed",
      top: 0,
      left: 0,
      right: 0,
      bottom: 56, // leave space for bottom tab bar
      zIndex: 30,
      background: "var(--background)",
      overflow: "hidden",
    }}>
      <AICompanion userId={userId} />
    </div>
  );
}
