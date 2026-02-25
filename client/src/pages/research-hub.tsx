import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  FlaskConical,
  Moon,
  Brain,
  Sunset,
  CheckCircle2,
  Camera,
  ChevronRight,
  Loader2,
  AlertCircle,
  TrendingUp,
} from "lucide-react";

const USER_ID = "default";

// ── Types ─────────────────────────────────────────────────────────────────────

interface StudyStatus {
  enrolled: boolean;
  studyCode: string;
  dayNumber: number;
  completedDays: number;
  targetDays: number;
  compensationEarned: number;
  todaySession: {
    morningCompleted: boolean;
    daytimeCompleted: boolean;
    eveningCompleted: boolean;
  } | null;
}

interface CorrelationDay {
  dayNumber: number;
  sessionDate: string;
  validDay: boolean;
  morning: {
    dreamValence: number | null;
    noRecall: boolean;
    nightmareFlag: string | null;
    dreamSnippet: string | null;
    welfareScore: number | null;
  } | null;
  daytime: {
    samValence: number | null;
    samStress: number | null;
    faa: number | null;
  } | null;
  foods: {
    id: string;
    summary: string | null;
    mealType: string | null;
    totalCalories: number | null;
    dominantMacro: string | null;
    glycemicImpact: string | null;
    aiMoodImpact: string | null;
    aiDreamRelevance: string | null;
    loggedAt: string;
  }[];
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const MACRO_COLOR: Record<string, string> = {
  carbs: "text-amber-400",
  protein: "text-blue-400",
  fat: "text-rose-400",
  balanced: "text-green-400",
};

const GI_BADGE: Record<string, string> = {
  low: "border-green-500/40 text-green-400 bg-green-500/10",
  medium: "border-amber-500/40 text-amber-400 bg-amber-500/10",
  high: "border-rose-500/40 text-rose-400 bg-rose-500/10",
};

function valenceEmoji(v: number | null) {
  if (v === null) return "—";
  if (v <= 3) return "😞";
  if (v <= 5) return "😐";
  if (v <= 7) return "🙂";
  return "😊";
}

function macroLabel(m: string | null) {
  if (!m) return "";
  return m.charAt(0).toUpperCase() + m.slice(1);
}

// ── Task card ─────────────────────────────────────────────────────────────────

function TaskCard({
  icon: Icon,
  label,
  hint,
  done,
  path,
  iconColor,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  hint: string;
  done: boolean;
  path: string;
  iconColor: string;
}) {
  const [, navigate] = useLocation();
  return (
    <button
      onClick={() => !done && navigate(path)}
      disabled={done}
      className={`flex-1 rounded-xl border p-3 text-left transition-all ${
        done
          ? "border-green-500/30 bg-green-500/5 opacity-80 cursor-default"
          : "border-border hover:border-primary/40 hover:bg-muted/30 cursor-pointer"
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <Icon className={`w-4 h-4 ${done ? "text-green-400" : iconColor}`} />
        {done && <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />}
      </div>
      <p className="text-xs font-semibold">{label}</p>
      <p className="text-xs text-muted-foreground mt-0.5">
        {done ? "Done ✓" : hint}
      </p>
    </button>
  );
}

// ── Correlation row ───────────────────────────────────────────────────────────

function CorrelationRow({ day }: { day: CorrelationDay }) {
  const hasFoods = day.foods.length > 0;
  const hasDaytime = !!day.daytime;
  const hasMorning = !!day.morning;

  if (!hasFoods && !hasDaytime && !hasMorning) return null;

  return (
    <div className="space-y-2 pb-4 border-b border-border/40 last:border-0">
      {/* Day label */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
          Day {day.dayNumber}
        </span>
        {day.validDay && (
          <Badge variant="outline" className="text-[10px] border-green-500/30 text-green-400 bg-green-500/10 py-0 px-1.5">
            valid
          </Badge>
        )}
      </div>

      {/* Food → Mood arrow chain */}
      <div className="flex items-stretch gap-1.5 text-xs">
        {/* Food block */}
        <div className="flex-1 min-w-0 rounded-lg bg-amber-500/10 border border-amber-500/20 p-2 space-y-0.5">
          <p className="font-medium text-amber-400">🍽️ Food</p>
          {hasFoods ? (
            <>
              <p className="text-muted-foreground truncate">{day.foods[0].summary ?? "Meal logged"}</p>
              {day.foods[0].totalCalories && (
                <p className="text-muted-foreground/70">
                  {day.foods[0].totalCalories} cal
                  {day.foods[0].dominantMacro && (
                    <span className={` · ${MACRO_COLOR[day.foods[0].dominantMacro] ?? ""}`}>
                      {macroLabel(day.foods[0].dominantMacro)}
                    </span>
                  )}
                </p>
              )}
              {day.foods.length > 1 && (
                <p className="text-muted-foreground/60">+{day.foods.length - 1} more meal{day.foods.length > 2 ? "s" : ""}</p>
              )}
            </>
          ) : (
            <p className="text-muted-foreground/60 italic">No meals logged</p>
          )}
        </div>

        {/* Arrow */}
        <div className="flex items-center text-muted-foreground/40 text-base">→</div>

        {/* EEG mood block */}
        <div className="flex-1 min-w-0 rounded-lg bg-violet-500/10 border border-violet-500/20 p-2 space-y-0.5">
          <p className="font-medium text-violet-400">🧠 Mood</p>
          {hasDaytime ? (
            <>
              <p className="text-muted-foreground">
                {valenceEmoji(day.daytime!.samValence)} {day.daytime!.samValence ?? "—"}/9
              </p>
              {day.daytime!.samStress !== null && (
                <p className="text-muted-foreground/70">
                  Stress: {day.daytime!.samStress}/9
                </p>
              )}
            </>
          ) : (
            <p className="text-muted-foreground/60 italic">No EEG session</p>
          )}
        </div>

        {/* Arrow */}
        <div className="flex items-center text-muted-foreground/40 text-base">→</div>

        {/* Dream block */}
        <div className="flex-1 min-w-0 rounded-lg bg-blue-500/10 border border-blue-500/20 p-2 space-y-0.5">
          <p className="font-medium text-blue-400">🌙 Dream</p>
          {hasMorning ? (
            day.morning!.noRecall ? (
              <p className="text-muted-foreground/60 italic">No recall</p>
            ) : (
              <>
                <p className="text-muted-foreground">
                  {valenceEmoji(day.morning!.dreamValence)} {day.morning!.dreamValence ?? "—"}/9
                </p>
                {day.morning!.dreamSnippet && (
                  <p className="text-muted-foreground/70 line-clamp-2 leading-tight">
                    "{day.morning!.dreamSnippet}"
                  </p>
                )}
              </>
            )
          ) : (
            <p className="text-muted-foreground/60 italic">Not yet recorded</p>
          )}
        </div>
      </div>

      {/* AI insight (if food logged + daytime EEG done) */}
      {hasFoods && hasDaytime && day.foods[0].aiMoodImpact && (
        <div className="flex items-start gap-2 rounded-lg bg-muted/30 p-2.5 text-xs">
          <TrendingUp className="w-3.5 h-3.5 text-muted-foreground shrink-0 mt-0.5" />
          <p className="text-muted-foreground leading-relaxed">{day.foods[0].aiMoodImpact}</p>
        </div>
      )}
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────

export default function ResearchHub() {
  const [, navigate] = useLocation();

  const { data: status, isLoading } = useQuery<StudyStatus>({
    queryKey: ["/api/study/status", USER_ID],
    queryFn: async () => {
      const res = await fetch(`/api/study/status/${USER_ID}`, { credentials: "include" });
      if (!res.ok) throw new Error("Failed");
      return res.json();
    },
  });

  const { data: correlation } = useQuery<CorrelationDay[]>({
    queryKey: ["/api/research/correlation", USER_ID],
    queryFn: async () => {
      const res = await fetch(`/api/research/correlation/${USER_ID}`, { credentials: "include" });
      if (!res.ok) return [];
      return res.json();
    },
    enabled: status?.enrolled === true,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  // ── Not enrolled ────────────────────────────────────────────────────────────
  if (!status?.enrolled) {
    return (
      <div className="max-w-lg mx-auto py-10 px-4 space-y-6">
        <div className="text-center space-y-3">
          <FlaskConical className="w-12 h-12 text-violet-400 mx-auto" />
          <h1 className="text-2xl font-bold">The Emotional Day-Night Cycle</h1>
          <p className="text-sm text-muted-foreground">
            30-day EEG + dream study — track how what you eat and feel during the day
            shows up in your dreams that night.
          </p>
        </div>

        <Card className="border-violet-500/30 bg-violet-500/5">
          <CardContent className="pt-5 space-y-3">
            {[
              { emoji: "💰", text: "Up to $150 + $25 bonus — $5 per valid day" },
              { emoji: "📅", text: "30 days · ~25 min/day · all from home" },
              { emoji: "🧠", text: "Requires your own Muse 2 EEG headset" },
              { emoji: "🍽️", text: "Photo your meals — AI tracks food → mood → dreams" },
            ].map(({ emoji, text }) => (
              <div key={text} className="flex items-center gap-3 text-sm">
                <span>{emoji}</span>
                <span className="text-muted-foreground">{text}</span>
              </div>
            ))}
          </CardContent>
        </Card>

        <Button className="w-full bg-violet-600 hover:bg-violet-700 gap-2" onClick={() => navigate("/research/enroll")}>
          Join the Study
          <ChevronRight className="w-4 h-4" />
        </Button>
      </div>
    );
  }

  // ── Enrolled ────────────────────────────────────────────────────────────────
  const { dayNumber, completedDays, targetDays, compensationEarned, todaySession, studyCode } = status;
  const progressPct = Math.round((completedDays / targetDays) * 100);

  const morningDone = todaySession?.morningCompleted ?? false;
  const daytimeDone = todaySession?.daytimeCompleted ?? false;
  const eveningDone = todaySession?.eveningCompleted ?? false;

  const correlationDays = (correlation ?? []).filter(
    d => d.foods.length > 0 || d.daytime || d.morning
  );

  return (
    <div className="max-w-lg mx-auto py-6 px-4 space-y-5">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="space-y-3">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2">
              <FlaskConical className="w-5 h-5 text-violet-400" />
              <h1 className="text-lg font-bold">Research Study</h1>
              <Badge variant="outline" className="text-[10px]">{studyCode}</Badge>
            </div>
            <p className="text-xs text-muted-foreground mt-0.5">
              Day {dayNumber} of {targetDays} · ${compensationEarned} earned
            </p>
          </div>
          <div className="text-right">
            <p className="text-2xl font-bold text-violet-400">{progressPct}%</p>
            <p className="text-xs text-muted-foreground">complete</p>
          </div>
        </div>
        <Progress value={progressPct} className="h-1.5" />
      </div>

      {/* ── Today's tasks ───────────────────────────────────────────────────── */}
      <Card>
        <CardContent className="pt-4 pb-4 space-y-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Today</p>
          <div className="flex gap-2">
            <TaskCard
              icon={Moon}
              label="Morning"
              hint="On waking"
              done={morningDone}
              path="/research/morning"
              iconColor="text-blue-400"
            />
            <TaskCard
              icon={Brain}
              label="Daytime"
              hint="9 AM – 1 PM"
              done={daytimeDone}
              path="/research/daytime"
              iconColor="text-violet-400"
            />
            <TaskCard
              icon={Sunset}
              label="Evening"
              hint="Before bed"
              done={eveningDone}
              path="/research/evening"
              iconColor="text-orange-400"
            />
          </div>
        </CardContent>
      </Card>

      {/* ── Log a meal ──────────────────────────────────────────────────────── */}
      <button
        onClick={() => navigate("/food-log")}
        className="w-full flex items-center gap-3 rounded-xl border border-amber-500/30 bg-amber-500/5 hover:bg-amber-500/10 p-4 transition-colors text-left"
      >
        <div className="w-9 h-9 rounded-lg bg-amber-500/20 flex items-center justify-center shrink-0">
          <Camera className="w-5 h-5 text-amber-400" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium">Log a meal</p>
          <p className="text-xs text-muted-foreground">Photo your food → AI tracks food → mood → dream</p>
        </div>
        <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />
      </button>

      {/* ── Food → Mood → Dream correlation log ─────────────────────────────── */}
      {correlationDays.length > 0 ? (
        <Card>
          <CardContent className="pt-4 pb-4 space-y-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Recent days — food · mood · dream
            </p>
            {correlationDays.map(day => (
              <CorrelationRow key={day.dayNumber} day={day} />
            ))}
          </CardContent>
        </Card>
      ) : (
        <Card className="border-dashed border-border/50">
          <CardContent className="py-8 text-center space-y-2">
            <AlertCircle className="w-8 h-8 text-muted-foreground/40 mx-auto" />
            <p className="text-sm text-muted-foreground">
              No data yet — complete your first day and log a meal to see the food → mood → dream pattern here.
            </p>
          </CardContent>
        </Card>
      )}

      {/* ── Withdraw link ───────────────────────────────────────────────────── */}
      <div className="text-center pt-2">
        <button
          onClick={() => navigate("/research/withdraw")}
          className="text-xs text-muted-foreground/50 hover:text-muted-foreground underline underline-offset-2 transition-colors"
        >
          Withdraw from study
        </button>
      </div>
    </div>
  );
}
