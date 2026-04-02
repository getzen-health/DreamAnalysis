import { getParticipantId } from "@/lib/participant";
import { useState, type Dispatch, type SetStateAction } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Moon, CheckCircle2, Loader2, Activity } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

const USER_ID = getParticipantId();

const VALENCE_LABELS: Record<number, string> = { 1:"😞", 3:"😟", 5:"😐", 7:"🙂", 9:"😊" };
const STRESS_LABELS:  Record<number, string> = { 1:"😌", 3:"🙂", 5:"😐", 7:"😰", 9:"😱" };

function closestLabel(val: number, labels: Record<number, string>) {
  const keys = Object.keys(labels).map(Number);
  const nearest = keys.reduce((a, b) => Math.abs(b - val) < Math.abs(a - val) ? b : a);
  return labels[nearest];
}

function RatingSlider({ label, value, onChange, labels }: {
  label: string; value: number; onChange: (v: number) => void;
  labels: Record<number, string>;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium">{label}</p>
        <span className="text-xl">{closestLabel(value, labels)}</span>
      </div>
      <Slider min={1} max={9} step={1} value={[value]} onValueChange={([v]) => onChange(v)} />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>1</span><span>{value}</span><span>9</span>
      </div>
    </div>
  );
}

const EXERCISE_OPTIONS = [
  { val: "none",     label: "None" },
  { val: "light",    label: "Light"     },
  { val: "moderate", label: "Moderate"  },
  { val: "vigorous", label: "Vigorous"  },
];

const CRAVING_TYPES = ["sweet", "salty", "fatty", "carbs", "caffeine", "alcohol"];

type IngestEntry = {
  name: string;
  dosage: string;
  timeTaken: string;
};

function emptyIngestEntry(): IngestEntry {
  return { name: "", dosage: "", timeTaken: "" };
}

function IngestSection({
  title,
  description,
  entries,
  setEntries,
}: {
  title: string;
  description: string;
  entries: IngestEntry[];
  setEntries: Dispatch<SetStateAction<IngestEntry[]>>;
}) {
  const updateEntry = (index: number, field: keyof IngestEntry, value: string) => {
    setEntries((prev) => prev.map((entry, i) => (
      i === index ? { ...entry, [field]: value } : entry
    )));
  };

  const addEntry = () => setEntries((prev) => [...prev, emptyIngestEntry()]);
  const removeEntry = (index: number) => setEntries((prev) => prev.filter((_, i) => i !== index));

  return (
    <Card>
      <CardContent className="pt-5 space-y-4">
        <div>
          <p className="text-sm font-medium">{title}</p>
          <p className="text-xs text-muted-foreground mt-1">{description}</p>
        </div>

        <div className="space-y-3">
          {entries.map((entry, index) => (
            <div key={`${title}-${index}`} className="rounded-lg border border-border/60 p-3 space-y-3">
              <div className="grid gap-3 md:grid-cols-3">
                <Input
                  value={entry.name}
                  onChange={(e) => updateEntry(index, "name", e.target.value)}
                  placeholder="Name"
                />
                <Input
                  value={entry.dosage}
                  onChange={(e) => updateEntry(index, "dosage", e.target.value)}
                  placeholder="Dosage"
                />
                <Input
                  value={entry.timeTaken}
                  onChange={(e) => updateEntry(index, "timeTaken", e.target.value)}
                  placeholder="Time taken"
                />
              </div>
              {entries.length > 1 && (
                <Button variant="ghost" size="sm" onClick={() => removeEntry(index)}>
                  Remove
                </Button>
              )}
            </div>
          ))}
        </div>

        <Button variant="outline" size="sm" onClick={addEntry}>
          Add {title.slice(0, -1)}
        </Button>
      </CardContent>
    </Card>
  );
}

export default function ResearchEvening() {
  const { toast } = useToast();
  const [, navigate] = useLocation();

  const [dayValence,     setDayValence]     = useState(5);
  const [dayArousal,     _setDayArousal]     = useState(5);
  const [stressNow,      setStressNow]      = useState(5);
  const [peakIntensity,  setPeakIntensity]  = useState(5);
  const [peakDirection,  setPeakDirection]  = useState<"positive" | "negative" | null>(null);
  const [exercise,       setExercise]       = useState<string | null>(null);
  const [alcohol,        setAlcohol]        = useState(0);
  const [supplements,    setSupplements]    = useState<IngestEntry[]>([emptyIngestEntry()]);
  const [cravings,       setCravings]       = useState(false);
  const [cravingTypes,   setCravingTypes]   = useState<string[]>([]);
  const [medications,    setMedications]    = useState<IngestEntry[]>([emptyIngestEntry()]);
  const [readyForSleep,  setReadyForSleep]  = useState<boolean | null>(null);
  const [submitting,     setSubmitting]     = useState(false);
  const [done,           setDone]           = useState(false);
  const [validDay,       setValidDay]       = useState(false);

  const { data: status } = useQuery({
    queryKey: ["/api/study/status", USER_ID],
    queryFn: async () => {
      const res = await apiRequest("GET", `/api/study/status/${USER_ID}`);
      if (!res.ok) throw new Error("Failed");
      return res.json() as Promise<{ enrolled: boolean; todaySession: { eveningCompleted: boolean } | null }>;
    },
  });

  const alreadyDone = status?.todaySession?.eveningCompleted === true;

  const toggleCravingType = (t: string) =>
    setCravingTypes(prev => prev.includes(t) ? prev.filter(x => x !== t) : [...prev, t]);

  const cleanEntries = (entries: IngestEntry[]) =>
    entries.filter((entry) => entry.name.trim() || entry.dosage.trim() || entry.timeTaken.trim());

  const handleSubmit = async () => {
    if (!peakDirection) {
      toast({ title: "One more thing", description: "Was your peak emotion positive or negative?", variant: "destructive" });
      return;
    }
    if (!exercise) {
      toast({ title: "One more thing", description: "What was your exercise level today?", variant: "destructive" });
      return;
    }
    if (readyForSleep === null) {
      toast({ title: "One more thing", description: "Are you ready for sleep?", variant: "destructive" });
      return;
    }
    setSubmitting(true);
    try {
      const supplementEntries = cleanEntries(supplements);
      const medicationEntries = cleanEntries(medications);
      const result = await apiRequest("POST", "/api/study/evening", {
        userId: USER_ID,
        dayValence, dayArousal,
        peakEmotionIntensity: peakIntensity,
        peakEmotionDirection: peakDirection,
        exerciseLevel: exercise,
        alcoholDrinks: alcohol,
        supplementsTaken: supplementEntries,
        cravingsToday: cravings,
        cravingTypes: cravings ? cravingTypes : [],
        medicationsTaken: medicationEntries.length > 0,
        medicationsDetails: medicationEntries,
        stressRightNow: stressNow,
        readyForSleep,
      });
      const data = await result.json();
      setValidDay(data.validDay ?? false);
      setDone(true);
    } catch {
      toast({ title: "Submit failed", description: "Try again in a moment.", variant: "destructive" });
    } finally {
      setSubmitting(false);
    }
  };

  // ── Already done / success ───────────────────────────────────────────────
  if (alreadyDone || done) {
    return (
      <div className="max-w-lg mx-auto py-16 px-4 text-center space-y-4">
        <CheckCircle2 className="w-12 h-12 text-cyan-400 mx-auto" />
        <h2 className="text-xl font-bold">Day complete {validDay ? "✓ Valid day!" : ""}</h2>
        {validDay && (
          <p className="text-sm text-cyan-400 font-medium">Valid day counted · morning, daytime & evening all done</p>
        )}
        <p className="text-sm text-muted-foreground">Good night. Fill in your dream journal tomorrow morning.</p>
        <Button variant="outline" onClick={() => navigate("/research")}>Back to Study Hub</Button>
      </div>
    );
  }

  if (status && !status.enrolled) {
    return (
      <div className="max-w-lg mx-auto py-16 px-4 text-center space-y-4">
        <p className="text-muted-foreground">You're not enrolled in the study.</p>
        <Button onClick={() => navigate("/research/enroll")}>Enroll now</Button>
      </div>
    );
  }

  // ── Form ────────────────────────────────────────────────────────────────────
  return (
    <div className="max-w-lg mx-auto py-6 px-4 space-y-5">

      {/* Header */}
      <div className="flex items-center gap-3">
        <Moon className="w-6 h-6 text-indigo-400" />
        <div>
          <h1 className="text-xl font-bold">Evening Analysis</h1>
          <p className="text-xs text-muted-foreground">Before bed · takes ~3 min</p>
        </div>
      </div>

      {/* Day mood */}
      <Card>
        <CardContent className="pt-5 space-y-5">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            How was your day overall?
          </p>
          <RatingSlider label="Day mood"    value={dayValence} onChange={setDayValence} labels={VALENCE_LABELS} />
          <RatingSlider label="Stress now"  value={stressNow}  onChange={setStressNow}  labels={STRESS_LABELS}  />
        </CardContent>
      </Card>

      {/* Peak emotion */}
      <Card>
        <CardContent className="pt-5 space-y-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Strongest emotion today
          </p>
          <RatingSlider label="Intensity" value={peakIntensity} onChange={setPeakIntensity} labels={{ 1:"barely", 5:"noticeable", 9:"overwhelming" }} />
          <div className="flex gap-3">
            {(["positive", "negative"] as const).map(dir => (
              <button
                key={dir}
                onClick={() => setPeakDirection(dir)}
                className={`flex-1 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                  peakDirection === dir
                    ? dir === "positive"
                      ? "border-cyan-500 bg-cyan-600/15 text-cyan-400"
                      : "border-rose-500 bg-rose-500/15 text-rose-400"
                    : "border-border hover:border-muted-foreground/40"
                }`}
              >
                {dir === "positive" ? "😊 Positive" : "😞 Negative"}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Exercise */}
      <Card>
        <CardContent className="pt-5 space-y-3">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <p className="text-sm font-medium">Exercise today</p>
          </div>
          <div className="grid grid-cols-4 gap-2">
            {EXERCISE_OPTIONS.map(({ val, label }) => (
              <button
                key={val}
                onClick={() => setExercise(val)}
                className={`py-2 rounded-lg border text-xs font-medium transition-all ${
                  exercise === val
                    ? "border-cyan-500 bg-cyan-600/15 text-cyan-400"
                    : "border-border hover:border-muted-foreground/40"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Alcohol */}
      <Card>
        <CardContent className="pt-5 space-y-3">
          <p className="text-sm font-medium">Alcoholic drinks today</p>
          <div className="flex items-center gap-4">
            <button onClick={() => setAlcohol(a => Math.max(0, a - 1))}
              className="w-9 h-9 rounded-full border border-border flex items-center justify-center text-lg hover:bg-muted/50 transition-colors">−</button>
            <span className="text-2xl font-bold w-6 text-center">{alcohol}</span>
            <button onClick={() => setAlcohol(a => Math.min(10, a + 1))}
              className="w-9 h-9 rounded-full border border-border flex items-center justify-center text-lg hover:bg-muted/50 transition-colors">+</button>
          </div>
        </CardContent>
      </Card>

      <IngestSection
        title="Supplements"
        description="Log any vitamins, supplements, powders, or nootropics taken today with name, dosage, and time."
        entries={supplements}
        setEntries={setSupplements}
      />

      {/* Cravings */}
      <Card>
        <CardContent className="pt-5 space-y-3">
          <p className="text-sm font-medium">Any cravings today?</p>
          <div className="flex gap-3">
            {([["Yes", true], ["No", false]] as [string, boolean][]).map(([label, val]) => (
              <button key={label} onClick={() => setCravings(val)}
                className={`flex-1 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                  cravings === val
                    ? "border-amber-500 bg-amber-500/15 text-amber-400"
                    : "border-border hover:border-muted-foreground/40"
                }`}>{label}</button>
            ))}
          </div>
          {cravings && (
            <div className="flex flex-wrap gap-2 pt-1">
              {CRAVING_TYPES.map(t => (
                <button key={t} onClick={() => toggleCravingType(t)}
                  className={`px-3 py-1.5 rounded-full border text-xs font-medium transition-all ${
                    cravingTypes.includes(t)
                      ? "border-amber-500 bg-amber-500/15 text-amber-400"
                      : "border-border hover:border-muted-foreground/40"
                  }`}>{t}</button>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <IngestSection
        title="Medications"
        description="Log any medications taken today with name, dosage, and time."
        entries={medications}
        setEntries={setMedications}
      />

      {/* Ready for sleep */}
      <Card>
        <CardContent className="pt-5 space-y-3">
          <p className="text-sm font-medium">Ready for sleep?</p>
          <div className="flex gap-3">
            {([["Yes 😴", true], ["Not yet", false]] as [string, boolean][]).map(([label, val]) => (
              <button key={label} onClick={() => setReadyForSleep(val)}
                className={`flex-1 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                  readyForSleep === val
                    ? "border-indigo-500 bg-indigo-500/15 text-indigo-400"
                    : "border-border hover:border-muted-foreground/40"
                }`}>{label}</button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Button className="w-full bg-indigo-700 hover:bg-indigo-800" onClick={handleSubmit} disabled={submitting}>
        {submitting ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : null}
        Close the day
      </Button>
    </div>
  );
}
