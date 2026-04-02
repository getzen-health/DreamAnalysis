import { useState, useEffect, useCallback } from "react";
import { useLocation } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AlertCircle, ChevronRight, Shield, Loader2 } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { sbGetSetting, sbSaveSetting } from "../../lib/supabase-store";

const CONSENT_TEXT = `INFORMED CONSENT — AntarAI Pilot Study

Principal Investigator: AntarAI Research Team
Version: 1.0

PURPOSE
This study examines how EEG (electroencephalogram) brain wave patterns correlate with
everyday stress and food-related emotional states. Your participation will contribute to
an academic research paper.

WHAT DATA IS COLLECTED
During each session, we record:
  - EEG brain wave patterns (alpha, beta, theta, delta, gamma band powers) via EEG headband
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
There are no known risks from EEG recording with the headband. The device
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

const STUDY_CODE_KEY = "ndw_study_code";

function generateCode(): string {
  const n = Math.floor(Math.random() * 9000) + 1000; // 1000–9999
  return `P${n}`;
}

function getStoredCode(): string | null {
  const stored = sbGetSetting(STUDY_CODE_KEY);
  if (stored && /^P\d{4}$/.test(stored)) return stored;
  return null;
}

async function checkCodeAvailable(code: string): Promise<boolean> {
  const resp = await apiRequest("GET", `/api/study/check-code?code=${encodeURIComponent(code)}`);
  if (!resp.ok) return false;
  const data = await resp.json() as { available: boolean };
  return data.available;
}

async function getOrCreateCode(): Promise<string> {
  const stored = getStoredCode();
  if (stored) return stored;

  const MAX_ATTEMPTS = 5;
  for (let i = 0; i < MAX_ATTEMPTS; i++) {
    const candidate = generateCode();
    const available = await checkCodeAvailable(candidate);
    if (available) {
      sbSaveSetting(STUDY_CODE_KEY, candidate);
      return candidate;
    }
  }
  // Fallback: use last generated code even if check failed
  const fallback = generateCode();
  sbSaveSetting(STUDY_CODE_KEY, fallback);
  return fallback;
}

export default function StudyConsent() {
  const [, navigate] = useLocation();
  const { toast } = useToast();

  const [agreed, setAgreed] = useState(false);
  const [code, setCode] = useState<string | null>(getStoredCode);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const initCode = useCallback(async () => {
    if (code) return;
    const newCode = await getOrCreateCode();
    setCode(newCode);
  }, [code]);

  useEffect(() => {
    void initCode();
  }, [initCode]);

  const canSubmit = agreed && code !== null;

  async function handleSubmit() {
    if (!canSubmit) return;
    setIsSubmitting(true);

    const timestamp = new Date().toISOString();
    const consentSnapshot = `Participant agreed to terms on ${timestamp}`;

    try {
      await apiRequest("POST", "/api/study/consent", {
        participant_code: code,
        consent_text: consentSnapshot,
      });

      navigate(`/study/profile?code=${encodeURIComponent(code)}`);
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

        {/* Auto-generated participant code */}
        <Card className="border-primary/30 bg-primary/5">
          <CardContent className="pt-5 space-y-3">
            <div className="space-y-1">
              <Label className="text-sm font-medium">Your anonymous participant code</Label>
              <p className="text-xs text-muted-foreground">
                This code is saved to this device. As long as you use the same browser,
                you'll get the same code automatically. Screenshot or note it down as a backup.
              </p>
            </div>
            <div className="flex items-center gap-3">
              {code ? (
                <>
                  <span className="font-mono text-3xl font-bold tracking-widest text-primary select-all">
                    {code}
                  </span>
                  <Badge
                    variant="outline"
                    className="border-cyan-500/50 text-cyan-400 text-xs"
                  >
                    Saved to device
                  </Badge>
                </>
              ) : (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span className="text-sm">Generating code…</span>
                </div>
              )}
            </div>
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
              Accept &amp; Continue
              <ChevronRight className="ml-2 h-4 w-4" />
            </>
          )}
        </Button>

        {!agreed && (
          <div className="flex items-start gap-2">
            <AlertCircle className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
            <p className="text-xs text-muted-foreground">
              Check the agreement box above to continue.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
