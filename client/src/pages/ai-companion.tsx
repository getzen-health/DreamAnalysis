import { AICompanion } from "@/components/ai-companion";

export default function AICompanionPage() {
  const userId = "demo-user";

  return (
    <main className="p-4 md:p-6 space-y-6">
      <AICompanion userId={userId} />
    </main>
  );
}
