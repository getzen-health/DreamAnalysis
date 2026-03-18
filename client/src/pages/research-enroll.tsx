import { useState } from "react";
import { useLocation } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import {
  CheckCircle2,
  AlertCircle,
  ChevronRight,
  Brain,
  Shield,
  Sparkles,
  Star,
} from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { getParticipantId, saveStudyCode, getStudyCode } from "@/lib/participant";

const USER_ID = getParticipantId();

// ─── Main component ───────────────────────────────────────────────────────────

export default function ResearchEnroll() {
  const [, navigate] = useLocation();
  const { toast } = useToast();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [agreed, setAgreed] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [enrolled, setEnrolled] = useState(false);
  const [betaCode, setBetaCode] = useState<string | null>(null);

  // If already enrolled, show the "you're in" state immediately
  const existingCode = getStudyCode();
  const alreadyEnrolled = !!existingCode;

  const canSubmit = name.trim().length >= 2 && email.trim().includes("@") && agreed;

  const handleSignUp = async () => {
    setIsSubmitting(true);
    try {
      const res = await apiRequest("POST", "/api/study/enroll", {
        userId: USER_ID,
        studyId: "svapnastra-beta-v1",
        consentVersion: "beta-1.0",
        overnightEegConsent: false,
        preferredMorningTime: "07:00",
        preferredDaytimeTime: "10:00",
        preferredEveningTime: "21:00",
        consentFullName: name.trim(),
        consentInitials: {},
        email: email.trim(),
      });
      const data = await res.json();
      const code = data.studyCode ?? data.betaCode ?? "BETA";
      saveStudyCode(code);
      setBetaCode(code);
      setEnrolled(true);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Sign-up failed — please try again";
      toast({ title: "Sign-up failed", description: msg, variant: "destructive" });
    } finally {
      setIsSubmitting(false);
    }
  };

  // ── Already enrolled view ─────────────────────────────────────────────────
  if (alreadyEnrolled && !enrolled) {
    return (
      <div className="max-w-lg mx-auto py-8 px-4">
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2 mb-1">
              <Star className="w-5 h-5 text-violet-400" />
              <CardTitle className="text-lg">AntarAI Beta</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="text-center space-y-4 py-4">
              <div className="w-16 h-16 rounded-full bg-cyan-600/20 flex items-center justify-center mx-auto">
                <CheckCircle2 className="w-8 h-8 text-cyan-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold">You're in the beta — thank you</h2>
                <p className="text-sm text-muted-foreground mt-1">
                  Your contributions are helping improve the models.
                </p>
              </div>
              <Badge className="bg-violet-600 text-white text-sm px-3 py-1">Beta Member</Badge>
            </div>

            <Card className="border-violet-500/30 bg-violet-500/5">
              <CardContent className="pt-4 pb-4 space-y-3">
                <p className="text-sm font-medium text-center">Your contributions</p>
                <div className="grid grid-cols-2 gap-3 text-center">
                  <div className="space-y-0.5">
                    <p className="text-2xl font-bold text-violet-300">—</p>
                    <p className="text-xs text-muted-foreground">Sessions contributed</p>
                  </div>
                  <div className="space-y-0.5">
                    <p className="text-2xl font-bold text-violet-300">—</p>
                    <p className="text-xs text-muted-foreground">EEG patterns shared</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Button className="w-full" onClick={() => navigate("/")}>
              Go to Dashboard
              <ChevronRight className="w-4 h-4 ml-1" />
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // ── Post-signup confirmation view ──────────────────────────────────────────
  if (enrolled) {
    return (
      <div className="max-w-lg mx-auto py-8 px-4">
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2 mb-1">
              <Star className="w-5 h-5 text-violet-400" />
              <CardTitle className="text-lg">AntarAI Beta</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="text-center space-y-4 py-4">
              <div className="w-16 h-16 rounded-full bg-cyan-600/20 flex items-center justify-center mx-auto">
                <CheckCircle2 className="w-8 h-8 text-cyan-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold">You're in!</h2>
                <p className="text-sm text-muted-foreground mt-1">
                  Welcome to the AntarAI Beta. Your early access is now active.
                </p>
              </div>
              <Badge className="bg-violet-600 text-white text-sm px-3 py-1">Beta Member</Badge>
            </div>

            <Card className="border-violet-500/30 bg-violet-500/5">
              <CardContent className="pt-4 pb-4 space-y-2">
                <p className="text-xs text-muted-foreground text-center uppercase tracking-widest">
                  Your beta code
                </p>
                <p className="text-3xl font-mono font-bold tracking-[0.2em] text-violet-300 text-center">
                  {betaCode ?? "BETA"}
                </p>
              </CardContent>
            </Card>

            <div className="bg-muted/40 rounded-lg p-4 space-y-2 text-sm text-muted-foreground">
              <p className="font-medium text-foreground">What happens next:</p>
              <ul className="space-y-1 list-disc list-inside">
                <li>New features will appear in your dashboard as they ship</li>
                <li>Your anonymized EEG patterns improve the models automatically</li>
                <li>You can delete your data anytime from Settings</li>
              </ul>
            </div>

            <Button className="w-full" onClick={() => navigate("/")}>
              Go to Dashboard
              <ChevronRight className="w-4 h-4 ml-1" />
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // ── Sign-up form (default view) ────────────────────────────────────────────
  return (
    <div className="max-w-lg mx-auto py-8 px-4">
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2 mb-1">
            <Sparkles className="w-5 h-5 text-violet-400" />
            <CardTitle className="text-lg">Join the Beta</CardTitle>
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Hero */}
          <div className="text-center space-y-2">
            <h2 className="text-2xl font-bold">Join the AntarAI Beta</h2>
            <p className="text-sm text-muted-foreground">
              Help improve the models and get early access to new features.
            </p>
          </div>

          {/* What you contribute */}
          <Card className="border-violet-500/30 bg-violet-500/5">
            <CardContent className="pt-4 pb-4 space-y-3">
              <p className="text-sm font-medium flex items-center gap-2">
                <Brain className="w-4 h-4 text-violet-400" />
                What you contribute
              </p>
              <ul className="text-sm text-muted-foreground space-y-1.5">
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-violet-400 mt-0.5 shrink-0" />
                  Anonymized EEG patterns — never raw data
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-violet-400 mt-0.5 shrink-0" />
                  Session metadata (duration, device type)
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-violet-400 mt-0.5 shrink-0" />
                  Feedback ratings on model outputs
                </li>
              </ul>
            </CardContent>
          </Card>

          {/* What you get */}
          <Card className="border-cyan-500/30 bg-cyan-600/5">
            <CardContent className="pt-4 pb-4 space-y-3">
              <p className="text-sm font-medium flex items-center gap-2">
                <Star className="w-4 h-4 text-cyan-400" />
                What you get
              </p>
              <ul className="text-sm text-muted-foreground space-y-1.5">
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400 mt-0.5 shrink-0" />
                  Earlier access to new features before public release
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400 mt-0.5 shrink-0" />
                  A "Beta" badge on your profile
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400 mt-0.5 shrink-0" />
                  Model accuracy improvements personalized to you over time
                </li>
              </ul>
            </CardContent>
          </Card>

          {/* Privacy note */}
          <div className="flex items-start gap-3 rounded-lg bg-muted/40 p-3">
            <Shield className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
            <p className="text-xs text-muted-foreground leading-relaxed">
              All data is anonymized before leaving your device. You can delete your data
              anytime from Settings.
            </p>
          </div>

          {/* Form */}
          <div className="space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="beta-name" className="text-sm">Name</Label>
              <Input
                id="beta-name"
                type="text"
                placeholder="Your name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                autoComplete="name"
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="beta-email" className="text-sm">Email</Label>
              <Input
                id="beta-email"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
              />
            </div>

            <div className="flex items-start gap-3">
              <Checkbox
                id="beta-agree"
                checked={agreed}
                onCheckedChange={(checked) => setAgreed(!!checked)}
                className="mt-0.5"
              />
              <Label htmlFor="beta-agree" className="text-sm leading-relaxed cursor-pointer">
                I agree to share anonymized EEG data to help improve the models
              </Label>
            </div>

            {!canSubmit && name.length > 0 && (
              <div className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
                <p className="text-xs text-muted-foreground">
                  Fill in your name, a valid email, and check the box to continue.
                </p>
              </div>
            )}

            <Button
              className="w-full bg-violet-600 hover:bg-violet-700"
              onClick={handleSignUp}
              disabled={!canSubmit || isSubmitting}
            >
              {isSubmitting ? "Signing up…" : "Join the Beta"}
              <ChevronRight className="w-4 h-4 ml-1" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
