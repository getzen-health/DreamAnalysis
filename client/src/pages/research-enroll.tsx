import { useState, useRef, useEffect } from "react";
import { useLocation } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  FlaskConical,
  CheckCircle2,
  AlertCircle,
  Copy,
  Check,
  ChevronRight,
  ChevronLeft,
  Moon,
  Brain,
  Sunset,
  DollarSign,
  Calendar,
  Shield,
} from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

const USER_ID = "default";
const CONSENT_VERSION = "2.0";

// ─── Consent sections ────────────────────────────────────────────────────────

const CONSENT_SECTIONS = [
  {
    id: "purpose",
    title: "1. Study Purpose and Overview",
    content:
      "This 30-day study investigates whether emotions experienced during the day predict the content of dreams that night — the Continuity Hypothesis of Dreaming. You will use your own Muse 2 EEG headset and the NeuralDreamWorkshop app to complete three brief daily tasks over 30 consecutive days. The research is conducted by independent researcher Lakshmi Sravya Vedantham and has been reviewed by an independent Institutional Review Board (IRB).",
  },
  {
    id: "tasks",
    title: "2. What You Will Do",
    content:
      "Each day you will complete three brief tasks: (Morning, 5–10 min) Record your dream in the app within minutes of waking — if you don't remember, select 'No recall.' (Daytime, 15–20 min) Between 9 AM and 1 PM, wear your Muse 2 for a 10-minute EEG session and answer mood questions. (Evening, 8–10 min) Complete a brief questionnaire about your day, meals, and emotional state. Additionally, you will attend a 60-minute onboarding video call at the start and a 20-minute debrief call at the end.",
  },
  {
    id: "risks",
    title: "3. Potential Risks",
    content:
      "Risks are minimal. Wearing the Muse 2 consumer EEG headset poses no known physical risk — it uses no electrical stimulation and produces no radiation. Reflecting on dreams or emotional experiences may occasionally feel mildly uncomfortable. You may skip or stop any entry at any time. If the morning mood check indicates you are experiencing distress, the app will display mental health resources.",
  },
  {
    id: "benefits",
    title: "4. Potential Benefits",
    content:
      "There are no guaranteed direct benefits to you personally. As a thank-you, you will receive a personalized summary of your EEG patterns and mood trends at the end of the study. Your participation may contribute to scientific understanding of the relationship between daily emotional states and dream content, which has potential applications for mental health and sleep research.",
  },
  {
    id: "compensation",
    title: "5. Compensation",
    content:
      "This study is unpaid. There is no monetary compensation for participation. At the end of the study, you will receive a personalized summary report of your EEG patterns, mood trends, and food–dream correlations collected over the 30 days.",
  },
  {
    id: "privacy",
    title: "6. Privacy and Data Security",
    content:
      "Your data is stored only under a 6-character pseudonymous study code (e.g., 'NX4T82') — never your real name. All data is transmitted over encrypted connections (TLS 1.3) to a SOC 2 Type II certified database. Dream journal text is processed server-side by the OpenAI GPT API to extract an emotional score; no identifying information is sent to OpenAI, and OpenAI's terms prohibit using API submissions for model training. Your name and email are stored only in a separate encrypted file on the researcher's device and are never in the research database.",
  },
  {
    id: "voluntary",
    title: "7. Voluntary Participation and Withdrawal",
    content:
      "Participation is entirely voluntary. You may withdraw at any time, for any reason, without penalty. To withdraw, use the 'Withdraw from Study' option on your research dashboard or email the researcher. Data collected before withdrawal will be retained for research unless you request deletion within 30 days of withdrawal — requests are honored within 7 days.",
  },
  {
    id: "contact",
    title: "8. Questions and Contact",
    content:
      "If you have questions about this study, contact Principal Investigator Lakshmi Sravya Vedantham at lakshmisravya.vedantham@gmail.com. For questions about your rights as a research participant — including complaints about how the study is being conducted — contact the approving IRB (to be confirmed at submission). Participation implies consent to these terms. A PDF copy of this consent form will be emailed to you upon enrollment.",
  },
];

// ─── Time options ─────────────────────────────────────────────────────────────

function timeOptions(startHour: number, endHour: number) {
  const opts: { value: string; label: string }[] = [];
  for (let h = startHour; h <= endHour; h++) {
    for (const m of [0, 30]) {
      if (h === endHour && m > 0) break;
      const hh = String(h).padStart(2, "0");
      const mm = String(m).padStart(2, "0");
      const value = `${hh}:${mm}`;
      const ampm = h < 12 ? "AM" : "PM";
      const displayH = h > 12 ? h - 12 : h === 0 ? 12 : h;
      const label = `${displayH}:${mm} ${ampm}`;
      opts.push({ value, label });
    }
  }
  return opts;
}

const morningTimes = timeOptions(4, 11);
const daytimeTimes = timeOptions(9, 13);
const eveningTimes = timeOptions(18, 23);

// ─── Step indicator ───────────────────────────────────────────────────────────

const STEP_LABELS = ["Overview", "Eligibility", "Consent", "Overnight EEG", "Preferences", "Confirmation"];

function StepIndicator({ step }: { step: number }) {
  return (
    <div className="mb-6">
      <div className="flex justify-between text-xs text-muted-foreground mb-2">
        <span>Step {step + 1} of {STEP_LABELS.length}</span>
        <span>{STEP_LABELS[step]}</span>
      </div>
      <Progress value={((step + 1) / STEP_LABELS.length) * 100} className="h-1.5" />
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function ResearchEnroll() {
  const [, navigate] = useLocation();
  const { toast } = useToast();

  const [step, setStep] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [studyCode, setStudyCode] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Step 2 — eligibility
  const [eligibility, setEligibility] = useState({
    regularSleep: false,
    noSleepDisorder: false,
    noEpilepsy: false,
    noMedications: false,
    ownsMuse2: false,
  });

  // Step 3 — consent sections
  const [sectionStates, setSectionStates] = useState<
    Record<string, { initial: string; read: boolean }>
  >(Object.fromEntries(CONSENT_SECTIONS.map((s) => [s.id, { initial: "", read: false }])));
  const [hasScrolledConsent, setHasScrolledConsent] = useState(false);
  const consentRef = useRef<HTMLDivElement>(null);

  // Step 4 — overnight EEG
  const [overnightConsent, setOvernightConsent] = useState<"yes" | "no" | "">("");

  // Step 5 — preferences
  const [morningTime, setMorningTime] = useState("07:00");
  const [daytimeTime, setDaytimeTime] = useState("10:00");
  const [eveningTime, setEveningTime] = useState("21:00");

  // ── Scroll detection for consent ──────────────────────────────────────────
  const handleConsentScroll = () => {
    const el = consentRef.current;
    if (!el) return;
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
    if (atBottom) setHasScrolledConsent(true);
  };

  useEffect(() => {
    setHasScrolledConsent(false);
  }, []);

  // ── Derived state ─────────────────────────────────────────────────────────
  const allEligible = Object.values(eligibility).every(Boolean);

  const allSectionsComplete = CONSENT_SECTIONS.every(
    (s) => sectionStates[s.id].initial.trim().length > 0 && sectionStates[s.id].read
  );
  const consentCanContinue = hasScrolledConsent && allSectionsComplete;

  // ── Navigation guards ────────────────────────────────────────────────────
  const canAdvance = () => {
    if (step === 1) return allEligible;
    if (step === 2) return consentCanContinue;
    if (step === 3) return overnightConsent !== "";
    return true;
  };

  const goNext = () => setStep((s) => s + 1);
  const goBack = () => setStep((s) => s - 1);

  // ── Enroll API call (happens on step 5 → 6) ──────────────────────────────
  const handleEnroll = async () => {
    setIsSubmitting(true);
    try {
      const res = await apiRequest("POST", "/api/study/enroll", {
        userId: USER_ID,
        studyId: "emotional-day-night-v1",
        consentVersion: CONSENT_VERSION,
        overnightEegConsent: overnightConsent === "yes",
        preferredMorningTime: morningTime,
        preferredDaytimeTime: daytimeTime,
        preferredEveningTime: eveningTime,
      });
      const data = await res.json();
      setStudyCode(data.studyCode);
      setStep(5);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Enrollment failed";
      toast({ title: "Enrollment failed", description: msg, variant: "destructive" });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCopy = () => {
    if (studyCode) {
      navigator.clipboard.writeText(studyCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Step renders
  // ─────────────────────────────────────────────────────────────────────────

  // ── Step 0: Overview ──────────────────────────────────────────────────────
  const renderOverview = () => (
    <div className="space-y-6">
      <div className="text-center space-y-2">
        <div className="flex justify-center">
          <FlaskConical className="w-12 h-12 text-violet-400" />
        </div>
        <h2 className="text-2xl font-bold">The Emotional Day-Night Cycle</h2>
        <p className="text-muted-foreground text-sm">30-Day EEG + Dream Study</p>
      </div>

      <Card className="border-violet-500/30 bg-violet-500/5">
        <CardContent className="pt-5 space-y-4">
          <p className="text-sm leading-relaxed">
            Do your daytime emotions show up in your dreams that night? We're using
            real EEG data from your Muse 2 to find out — in the real world, not a
            sleep lab.
          </p>
          <div className="grid grid-cols-1 gap-3">
            {[
              { icon: Calendar, label: "30 consecutive days", sub: "~25–35 min/day" },
              { icon: DollarSign, label: "Unpaid volunteer study", sub: "Contribute to dream science" },
              { icon: Brain, label: "Requires: Muse 2 EEG headset", sub: "Bring your own device" },
              { icon: Shield, label: "IRB-reviewed research", sub: "Pseudonymized data" },
            ].map(({ icon: Icon, label, sub }) => (
              <div key={label} className="flex items-center gap-3 text-sm">
                <Icon className="w-4 h-4 text-violet-400 shrink-0" />
                <div>
                  <span className="font-medium">{label}</span>
                  <span className="text-muted-foreground ml-2 text-xs">{sub}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="bg-muted/40 rounded-lg p-4 space-y-2">
        <p className="text-sm font-medium">Each day, three brief tasks:</p>
        <div className="space-y-2 text-sm text-muted-foreground">
          <div className="flex items-start gap-2">
            <Moon className="w-4 h-4 mt-0.5 text-blue-400 shrink-0" />
            <span><strong className="text-foreground">Morning (5 min)</strong> — Record your dream right when you wake up</span>
          </div>
          <div className="flex items-start gap-2">
            <Brain className="w-4 h-4 mt-0.5 text-violet-400 shrink-0" />
            <span><strong className="text-foreground">Daytime (15 min)</strong> — 10-min Muse 2 EEG session + mood check</span>
          </div>
          <div className="flex items-start gap-2">
            <Sunset className="w-4 h-4 mt-0.5 text-orange-400 shrink-0" />
            <span><strong className="text-foreground">Evening (8 min)</strong> — Brief questions about your day and what you ate</span>
          </div>
        </div>
      </div>
    </div>
  );

  // ── Step 1: Eligibility ───────────────────────────────────────────────────
  const renderEligibility = () => {
    const items: { key: keyof typeof eligibility; label: string }[] = [
      { key: "regularSleep", label: "I sleep at roughly the same time each night (regular schedule)" },
      { key: "noSleepDisorder", label: "I do NOT have a diagnosed sleep disorder (insomnia, sleep apnea, or narcolepsy)" },
      { key: "noEpilepsy", label: "I do NOT have epilepsy or a history of seizures" },
      { key: "noMedications", label: "I am NOT currently taking anticonvulsants or antipsychotic medications" },
      { key: "ownsMuse2", label: "I own a Muse 2 EEG headset and it is working" },
    ];

    return (
      <div className="space-y-5">
        <div>
          <h2 className="text-xl font-bold mb-1">Eligibility Check</h2>
          <p className="text-sm text-muted-foreground">
            Please confirm all of the following apply to you. All five must be true to participate.
          </p>
        </div>

        <div className="space-y-4">
          {items.map(({ key, label }) => (
            <div key={key} className="flex items-start gap-3">
              <Checkbox
                id={key}
                checked={eligibility[key]}
                onCheckedChange={(checked) =>
                  setEligibility((e) => ({ ...e, [key]: !!checked }))
                }
                className="mt-0.5"
              />
              <Label htmlFor={key} className="text-sm leading-relaxed cursor-pointer">
                {label}
              </Label>
            </div>
          ))}
        </div>

        {Object.values(eligibility).some(Boolean) && !allEligible && (
          <Card className="border-destructive/40 bg-destructive/10">
            <CardContent className="pt-4 pb-4">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-destructive mt-0.5 shrink-0" />
                <p className="text-sm text-destructive">
                  All five criteria must be confirmed to participate. This study requires specific
                  conditions for scientific validity and participant safety.
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {allEligible && (
          <Card className="border-green-500/40 bg-green-500/10">
            <CardContent className="pt-4 pb-4">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" />
                <p className="text-sm text-green-400 font-medium">
                  You're eligible! Continue to review the consent form.
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    );
  };

  // ── Step 2: Consent form ──────────────────────────────────────────────────
  const renderConsent = () => (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-bold mb-1">Informed Consent</h2>
        <p className="text-sm text-muted-foreground">
          Read each section carefully. Initial each section after reading and check the box.
          Scroll to the bottom to unlock the Continue button.
        </p>
      </div>

      <div className="text-xs text-muted-foreground flex gap-2 items-center">
        <Badge variant="outline" className="text-xs">v{CONSENT_VERSION}</Badge>
        <span>Consent Form — The Emotional Day-Night Cycle Study</span>
      </div>

      <div
        ref={consentRef}
        onScroll={handleConsentScroll}
        className="h-[340px] overflow-y-auto border border-border rounded-lg p-4 space-y-6 text-sm"
      >
        {CONSENT_SECTIONS.map((section) => (
          <div key={section.id} className="space-y-3 pb-4 border-b border-border last:border-0">
            <h3 className="font-semibold">{section.title}</h3>
            <p className="text-muted-foreground leading-relaxed">{section.content}</p>
            <div className="flex items-center gap-4 pt-1">
              <div className="flex items-center gap-2">
                <Label htmlFor={`initial-${section.id}`} className="text-xs shrink-0">
                  Initials:
                </Label>
                <Input
                  id={`initial-${section.id}`}
                  maxLength={3}
                  placeholder="e.g. LSV"
                  className="w-20 h-7 text-xs uppercase"
                  value={sectionStates[section.id].initial}
                  onChange={(e) =>
                    setSectionStates((prev) => ({
                      ...prev,
                      [section.id]: {
                        ...prev[section.id],
                        initial: e.target.value.toUpperCase(),
                      },
                    }))
                  }
                />
              </div>
              <div className="flex items-center gap-2">
                <Checkbox
                  id={`read-${section.id}`}
                  checked={sectionStates[section.id].read}
                  onCheckedChange={(checked) =>
                    setSectionStates((prev) => ({
                      ...prev,
                      [section.id]: { ...prev[section.id], read: !!checked },
                    }))
                  }
                />
                <Label htmlFor={`read-${section.id}`} className="text-xs cursor-pointer">
                  I have read this section
                </Label>
              </div>
            </div>
          </div>
        ))}

        <div className="py-4 text-center text-xs text-muted-foreground">
          — End of consent form —
          <br />
          By continuing, you confirm you have read and understood all sections above.
          <br />
          A PDF copy will be emailed to you upon enrollment.
        </div>
      </div>

      {!hasScrolledConsent && (
        <p className="text-xs text-muted-foreground text-center">
          ↕ Scroll to the bottom of the consent form to continue
        </p>
      )}
      {hasScrolledConsent && !allSectionsComplete && (
        <p className="text-xs text-amber-400 text-center">
          Please initial and check all sections above
        </p>
      )}
    </div>
  );

  // ── Step 3: Overnight EEG ─────────────────────────────────────────────────
  const renderOvernightEeg = () => (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-bold mb-1">Optional: Overnight EEG</h2>
        <Badge variant="outline" className="text-xs mb-3">Optional — does not affect eligibility</Badge>
        <p className="text-sm text-muted-foreground leading-relaxed">
          In addition to the required daytime EEG session, you may optionally wear your Muse 2
          while you sleep. This allows us to detect REM sleep periods and correlate them with your
          dream reports.
        </p>
      </div>

      <Card className="bg-amber-500/10 border-amber-500/30">
        <CardContent className="pt-4 pb-4 space-y-2">
          <p className="text-sm font-medium">Overnight EEG details:</p>
          <ul className="text-sm text-muted-foreground space-y-1 list-disc list-inside">
            <li>Wear the Muse 2 headset during sleep (same sensor, no extra equipment)</li>
            <li>No electrical stimulation — purely passive recording</li>
            <li>You may remove it at any time during the night</li>
            <li>Optional — you may remove the headset at any time during the night</li>
          </ul>
        </CardContent>
      </Card>

      <div className="space-y-3">
        <p className="text-sm font-medium">Do you consent to the optional overnight EEG?</p>
        <RadioGroup
          value={overnightConsent}
          onValueChange={(v) => setOvernightConsent(v as "yes" | "no")}
          className="space-y-3"
        >
          <div className="flex items-start gap-3 p-3 rounded-lg border border-border hover:bg-muted/30 cursor-pointer">
            <RadioGroupItem value="yes" id="overnight-yes" className="mt-0.5" />
            <Label htmlFor="overnight-yes" className="cursor-pointer space-y-0.5">
              <div className="font-medium text-sm">Yes, I consent to the optional overnight EEG</div>
              <div className="text-xs text-muted-foreground">I may skip individual nights with no penalty</div>
            </Label>
          </div>
          <div className="flex items-start gap-3 p-3 rounded-lg border border-border hover:bg-muted/30 cursor-pointer">
            <RadioGroupItem value="no" id="overnight-no" className="mt-0.5" />
            <Label htmlFor="overnight-no" className="cursor-pointer space-y-0.5">
              <div className="font-medium text-sm">No thanks — daytime sessions only</div>
              <div className="text-xs text-muted-foreground">Daytime sessions only, full participation</div>
            </Label>
          </div>
        </RadioGroup>
      </div>
    </div>
  );

  // ── Step 4: Preferences ───────────────────────────────────────────────────
  const renderPreferences = () => (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-bold mb-1">Set Your Daily Schedule</h2>
        <p className="text-sm text-muted-foreground">
          Choose times you can reliably protect each day for 30 days. The app will send
          gentle reminders at these times.
        </p>
      </div>

      <div className="space-y-5">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Moon className="w-4 h-4 text-blue-400" />
            <Label className="text-sm font-medium">Morning dream entry</Label>
          </div>
          <p className="text-xs text-muted-foreground pl-6">
            This should be your alarm time — record the moment you wake up.
          </p>
          <Select value={morningTime} onValueChange={setMorningTime}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {morningTimes.map(({ value, label }) => (
                <SelectItem key={value} value={value}>{label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-violet-400" />
            <Label className="text-sm font-medium">Daytime EEG session</Label>
          </div>
          <p className="text-xs text-muted-foreground pl-6">
            Must be between 9 AM – 1 PM. Pick a time you can sit quietly for 20 minutes.
          </p>
          <Select value={daytimeTime} onValueChange={setDaytimeTime}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {daytimeTimes.map(({ value, label }) => (
                <SelectItem key={value} value={value}>{label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Sunset className="w-4 h-4 text-orange-400" />
            <Label className="text-sm font-medium">Evening check-in</Label>
          </div>
          <p className="text-xs text-muted-foreground pl-6">
            Complete within 2 hours of your usual bedtime.
          </p>
          <Select value={eveningTime} onValueChange={setEveningTime}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {eveningTimes.map(({ value, label }) => (
                <SelectItem key={value} value={value}>{label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="bg-muted/40 rounded-lg p-3 text-xs text-muted-foreground">
        You can adjust these times later from your research dashboard.
      </div>
    </div>
  );

  // ── Step 5: Confirmation ──────────────────────────────────────────────────
  const renderConfirmation = () => (
    <div className="space-y-6 text-center">
      <div className="flex flex-col items-center gap-3">
        <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center">
          <CheckCircle2 className="w-8 h-8 text-green-400" />
        </div>
        <h2 className="text-2xl font-bold">You're enrolled!</h2>
        <p className="text-sm text-muted-foreground max-w-sm">
          Welcome to the study. Your anonymous research identifier is shown below —
          save it somewhere safe.
        </p>
      </div>

      <Card className="border-violet-500/30 bg-violet-500/5 mx-auto max-w-xs">
        <CardContent className="pt-5 pb-5 space-y-3">
          <p className="text-xs text-muted-foreground uppercase tracking-widest">Your Study Code</p>
          <div className="text-4xl font-mono font-bold tracking-[0.25em] text-violet-300">
            {studyCode ?? "------"}
          </div>
          <Button
            variant="outline"
            size="sm"
            className="gap-2"
            onClick={handleCopy}
          >
            {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            {copied ? "Copied!" : "Copy code"}
          </Button>
          <p className="text-xs text-muted-foreground">
            This is your anonymous identifier. It is never linked to your name in the research database.
          </p>
        </CardContent>
      </Card>

      <div className="bg-muted/40 rounded-lg p-4 text-left space-y-2">
        <p className="text-sm font-medium">What happens next:</p>
        <ol className="text-sm text-muted-foreground space-y-1.5 list-decimal list-inside">
          <li>The researcher will email you a onboarding call invitation within 3 days</li>
          <li>Your study starts the morning after your onboarding call</li>
          <li>
            <strong className="text-foreground">First thing tomorrow morning</strong> — before
            checking anything else — open the app and record your dream
          </li>
        </ol>
      </div>

      <div className="text-xs text-muted-foreground">
        A PDF copy of your consent form will be emailed to you shortly.
        <br />
        Questions? Email lakshmisravya.vedantham@gmail.com
      </div>

      <Button className="w-full" onClick={() => navigate("/research")}>
        Go to Research Dashboard
        <ChevronRight className="w-4 h-4 ml-1" />
      </Button>
    </div>
  );

  // ─── Render ───────────────────────────────────────────────────────────────

  const stepRenderers = [
    renderOverview,
    renderEligibility,
    renderConsent,
    renderOvernightEeg,
    renderPreferences,
    renderConfirmation,
  ];

  const isLastInputStep = step === 4; // step before confirmation
  const isFinalStep = step === 5;

  return (
    <div className="max-w-lg mx-auto py-8 px-4">
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2 mb-1">
            <FlaskConical className="w-5 h-5 text-violet-400" />
            <CardTitle className="text-lg">Join the Study</CardTitle>
          </div>
          {!isFinalStep && <StepIndicator step={step} />}
        </CardHeader>

        <CardContent className="space-y-6">
          {stepRenderers[step]?.()}

          {!isFinalStep && (
            <div className="flex justify-between pt-2">
              <Button
                variant="ghost"
                onClick={goBack}
                disabled={step === 0}
                className="gap-1"
              >
                <ChevronLeft className="w-4 h-4" />
                Back
              </Button>

              {isLastInputStep ? (
                <Button
                  onClick={handleEnroll}
                  disabled={isSubmitting}
                  className="gap-1 bg-violet-600 hover:bg-violet-700"
                >
                  {isSubmitting ? "Enrolling…" : "Confirm & Enroll"}
                  <ChevronRight className="w-4 h-4" />
                </Button>
              ) : (
                <Button
                  onClick={goNext}
                  disabled={!canAdvance()}
                  className="gap-1"
                >
                  Continue
                  <ChevronRight className="w-4 h-4" />
                </Button>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
