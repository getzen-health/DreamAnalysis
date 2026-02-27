import { useState } from "react";
import { useLocation } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AlertCircle, ChevronRight, Shield, Loader2 } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

const CONSENT_TEXT = `INFORMED CONSENT — Neural Dream Workshop Pilot Study

Principal Investigator: Neural Dream Workshop Research Team
Version: 1.0

PURPOSE
This study examines how EEG (electroencephalogram) brain wave patterns correlate with
everyday stress and food-related emotional states. Your participation will contribute to
an academic research paper.

WHAT DATA IS COLLECTED
During each session, we record:
  - EEG brain wave patterns (alpha, beta, theta, delta, gamma band powers) via Muse 2 headband
  - Self-report survey responses (numeric ratings only)
  - Session metadata (block type, timestamps)

We do NOT collect your name, email address, or any directly identifying information.

HOW YOUR DATA IS USED
All data is used exclusively for academic research purposes. Your data may be included
in a published paper presenting aggregated, anonymous findings. No individual-level
data will be published. The dataset may be shared with other researchers under a data
use agreement that requires the same anonymization standards.

ANONYMIZATION
You are assigned a unique participant code (e.g. P001) before the study begins.
All data stored in our database is linked to this code only. The mapping between
codes and real identities is never stored.

RISKS AND BENEFITS
There are no known risks from EEG recording with the Muse 2 headband. The device
does not deliver any electrical signal to your brain. Participation may contribute
to scientific understanding of brain-body relationships, which is the primary benefit.

VOLUNTARY PARTICIPATION
Your participation is entirely voluntary. You may choose not to answer any question or
withdraw from the study at any time without penalty or consequence. If you withdraw
mid-session, any data already submitted for that session will be deleted upon request.

RIGHT TO WITHDRAW
To withdraw, stop the session at any time. To request deletion of your data, contact
the research team with your participant code. All data linked to that code will be
permanently removed from our database within 7 days.

CONTACT
For questions about this study, contact the research team through the application.

By checking the box below you confirm that you have read and understood the above,
that you are 18 years of age or older, and that you agree to participate voluntarily.`;

const PARTICIPANT_CODE_REGEX = /^P\d{3}$/;

export default function StudyConsent() {
  const [, navigate] = useLocation();
  const { toast } = useToast();

  const [agreed, setAgreed] = useState(false);
  const [code, setCode] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [codeError, setCodeError] = useState("");

  const isCodeValid = PARTICIPANT_CODE_REGEX.test(code.trim());
  const canSubmit = agreed && isCodeValid;

  function handleCodeChange(value: string) {
    setCode(value);
    if (value.length > 0 && !PARTICIPANT_CODE_REGEX.test(value.trim())) {
      setCodeError("Code must be P followed by exactly 3 digits (e.g. P001)");
    } else {
      setCodeError("");
    }
  }

  async function handleSubmit() {
    if (!canSubmit) return;
    setIsSubmitting(true);

    const timestamp = new Date().toISOString();
    const consentSnapshot = `Participant agreed to terms on ${timestamp}`;

    try {
      await apiRequest("POST", "/api/study/consent", {
        participant_code: code.trim(),
        consent_text: consentSnapshot,
        age: 0,
        diet_type: "omnivore",
        has_apple_watch: false,
      });

      navigate(`/study/profile?code=${encodeURIComponent(code.trim())}`);
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "Submission failed — please try again";
      toast({ title: "Could not submit consent", description: msg, variant: "destructive" });
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-2xl mx-auto px-4 py-10 space-y-6">

        {/* Header */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            <h1 className="text-2xl font-bold">Informed Consent</h1>
          </div>
          <p className="text-sm text-muted-foreground">
            Please read the following carefully before continuing.
          </p>
        </div>

        {/* Consent text */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Study Consent Form</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-64 pr-4">
              <pre className="text-xs text-muted-foreground leading-relaxed whitespace-pre-wrap font-sans">
                {CONSENT_TEXT}
              </pre>
            </ScrollArea>
          </CardContent>
        </Card>

        {/* Checkbox */}
        <Card>
          <CardContent className="pt-5">
            <div className="flex items-start gap-3">
              <Checkbox
                id="consent-agree"
                checked={agreed}
                onCheckedChange={(checked) => setAgreed(!!checked)}
                className="mt-0.5"
              />
              <Label
                htmlFor="consent-agree"
                className="text-sm leading-relaxed cursor-pointer"
              >
                I have read the above and agree to participate voluntarily. I confirm
                I am 18 years of age or older.
              </Label>
            </div>
          </CardContent>
        </Card>

        {/* Participant code */}
        <Card>
          <CardContent className="pt-5 space-y-3">
            <div className="space-y-1">
              <Label htmlFor="participant-code" className="text-sm font-medium">
                Participant code
              </Label>
              <p className="text-xs text-muted-foreground">
                You should have received a code from the study coordinator (e.g. P001).
              </p>
            </div>
            <Input
              id="participant-code"
              type="text"
              placeholder="e.g. P001"
              value={code}
              onChange={(e) => handleCodeChange(e.target.value)}
              maxLength={4}
              className="font-mono tracking-widest w-36"
            />
            {codeError && (
              <div className="flex items-start gap-2">
                <AlertCircle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />
                <p className="text-xs text-destructive">{codeError}</p>
              </div>
            )}
            {isCodeValid && (
              <Badge
                variant="outline"
                className="border-green-500/50 text-green-400 text-xs w-fit"
              >
                Valid code
              </Badge>
            )}
          </CardContent>
        </Card>

        {/* Submit */}
        <Button
          className="w-full"
          size="lg"
          disabled={!canSubmit || isSubmitting}
          onClick={handleSubmit}
        >
          {isSubmitting ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Submitting…
            </>
          ) : (
            <>
              Continue
              <ChevronRight className="ml-2 h-4 w-4" />
            </>
          )}
        </Button>

        {!canSubmit && (agreed || code.length > 0) && (
          <div className="flex items-start gap-2">
            <AlertCircle className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
            <p className="text-xs text-muted-foreground">
              Check the agreement box and enter a valid participant code to continue.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
