import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Loader2, CheckCircle2 } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

interface IrtWorkflowCardProps {
  userId: string;
  originalDreamText: string;
}

type Step = "read" | "rewrite" | "rehearse" | "done";

export function IrtWorkflowCard({ userId, originalDreamText }: IrtWorkflowCardProps) {
  const { toast } = useToast();
  const [step, setStep] = useState<Step>("read");
  const [rewrittenEnding, setRewrittenEnding] = useState("");
  const [rehearsalNote, setRehearsalNote] = useState("");
  const [saving, setSaving] = useState(false);

  async function handleSave() {
    if (!rewrittenEnding.trim()) return;
    setSaving(true);
    try {
      await apiRequest("POST", "/api/irt-session", {
        userId,
        originalDreamText,
        rewrittenEnding: rewrittenEnding.trim(),
        rehearsalNote: rehearsalNote.trim() || null,
      });
      setStep("done");
    } catch {
      toast({ title: "Save failed", description: "Could not save IRT session. Try again.", variant: "destructive" });
    } finally {
      setSaving(false);
    }
  }

  return (
    <Card className="border border-amber-500/40 bg-amber-500/5">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold text-amber-400 flex items-center gap-2">
          {step === "done" ? (
            <><CheckCircle2 className="w-4 h-4" /> IRT Session Saved</>
          ) : (
            "Image Rehearsal Therapy (IRT)"
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        {step === "done" && (
          <p className="text-muted-foreground">
            Great work. Mentally revisit your rewritten ending before sleep tonight — that's the rehearsal that makes IRT effective.
          </p>
        )}

        {step === "read" && (
          <>
            <p className="text-muted-foreground text-xs">
              <span className="font-semibold text-amber-400">Step 1 of 3 — Read your nightmare.</span>{" "}
              IRT works by rewriting the nightmare's ending. First, read it through calmly.
            </p>
            <div className="bg-background/50 rounded-md p-3 text-muted-foreground whitespace-pre-wrap text-xs max-h-40 overflow-y-auto">
              {originalDreamText}
            </div>
            <Button size="sm" variant="outline" className="w-full border-amber-500/40 text-amber-300 hover:bg-amber-500/10" onClick={() => setStep("rewrite")}>
              I've read it — rewrite the ending
            </Button>
          </>
        )}

        {step === "rewrite" && (
          <>
            <p className="text-muted-foreground text-xs">
              <span className="font-semibold text-amber-400">Step 2 of 3 — Rewrite the ending.</span>{" "}
              Change the nightmare however you like — make it neutral, pleasant, or give yourself control. Be specific.
            </p>
            <Textarea
              className="min-h-[100px] text-sm bg-background/50 resize-none"
              placeholder="In my rewritten version, the dream ends with..."
              value={rewrittenEnding}
              onChange={e => setRewrittenEnding(e.target.value)}
            />
            <Button
              size="sm"
              variant="outline"
              className="w-full border-amber-500/40 text-amber-300 hover:bg-amber-500/10"
              disabled={!rewrittenEnding.trim()}
              onClick={() => setStep("rehearse")}
            >
              Continue to rehearsal
            </Button>
          </>
        )}

        {step === "rehearse" && (
          <>
            <p className="text-muted-foreground text-xs">
              <span className="font-semibold text-amber-400">Step 3 of 3 — Rehearse.</span>{" "}
              Close your eyes and mentally play through the rewritten ending slowly, two or three times. When ready, save the session.
            </p>
            <Textarea
              className="min-h-[70px] text-sm bg-background/50 resize-none"
              placeholder="Optional: note anything about the rehearsal..."
              value={rehearsalNote}
              onChange={e => setRehearsalNote(e.target.value)}
            />
            <Button
              size="sm"
              className="w-full bg-amber-500/80 hover:bg-amber-500 text-black font-semibold"
              disabled={saving}
              onClick={handleSave}
            >
              {saving ? <><Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> Saving…</> : "Save IRT session"}
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  );
}
