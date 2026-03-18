import { useState, useMemo, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { resolveUrl, apiRequest } from "@/lib/queryClient";
import { getParticipantId } from "@/lib/participant";
import { hapticSuccess } from "@/lib/haptics";
import { useVoiceData } from "@/hooks/use-voice-data";

// ── Types ─────────────────────────────────────────────────────────────────────

interface FoodItem {
  name: string;
  portion: string;
  calories: number;
  carbs_g: number;
  protein_g: number;
  fat_g: number;
}

interface FoodLog {
  id: string;
  loggedAt: string;
  mealType: string | null;
  summary: string | null;
  totalCalories: number | null;
  dominantMacro: string | null;
  foodItems: FoodItem[] | null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const MEAL_ICONS: Record<string, string> = {
  breakfast: "🌅",
  lunch: "☀️",
  dinner: "🌙",
  snack: "🍎",
};

function isToday(iso: string): boolean {
  return new Date(iso).toDateString() === new Date().toDateString();
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function getMealLabel(mealType: string | null): string {
  if (!mealType) return "Meal";
  return mealType.charAt(0).toUpperCase() + mealType.slice(1);
}

function autoMealType(): string {
  const h = new Date().getHours();
  if (h >= 5 && h < 10) return "breakfast";
  if (h >= 11 && h < 15) return "lunch";
  if (h >= 17 && h < 22) return "dinner";
  return "snack";
}

function getCravingFromVoice(voice: { stress_index?: number; valence?: number } | null): { text: string; label: string } {
  if (!voice) return { text: "balanced — you're eating from hunger, not emotion", label: "Balanced" };
  const stress = voice.stress_index ?? 0;
  const valence = voice.valence ?? 0;
  if (stress > 0.6) return { text: "stress eating — your body seeks comfort food", label: "Stress" };
  if (valence > 0.3) return { text: "mindful eating — you're calm and present", label: "Mindful" };
  if (valence < -0.2) return { text: "comfort seeking — emotional eating tendency", label: "Comfort" };
  return { text: "balanced — you're eating from hunger, not emotion", label: "Balanced" };
}

// ── Calorie Ring ──────────────────────────────────────────────────────────────

const CAL_GOAL = 2000;
const PROTEIN_GOAL = 50;
const CARBS_GOAL = 275;
const FAT_GOAL = 78;

function CalorieRing({ calories }: { calories: number }) {
  const r = 58;
  const stroke = 8;
  const circ = 2 * Math.PI * r;
  const pct = Math.min(calories / CAL_GOAL, 1);
  const dash = circ * pct;
  const size = 140;
  const cx = size / 2;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginBottom: 20 }}>
      <div style={{ position: "relative", width: size, height: size }}>
        <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
          {/* Background track */}
          <circle
            cx={cx}
            cy={cx}
            r={r}
            fill="none"
            stroke="var(--border)"
            strokeWidth={stroke}
          />
          {/* Fill arc */}
          <circle
            cx={cx}
            cy={cx}
            r={r}
            fill="none"
            stroke="url(#calGrad)"
            strokeWidth={stroke}
            strokeDasharray={`${dash} ${circ}`}
            strokeLinecap="round"
          />
          <defs>
            <linearGradient id="calGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#d4a017" />
              <stop offset="100%" stopColor="#ea580c" />
            </linearGradient>
          </defs>
        </svg>
        {/* Center text */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <span style={{ fontSize: 28, fontWeight: 700, color: "var(--foreground)", lineHeight: 1 }}>
            {calories}
          </span>
          <span style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 2 }}>of {CAL_GOAL} kcal</span>
        </div>
      </div>
    </div>
  );
}

// ── Macro Progress Bar ────────────────────────────────────────────────────────

function MacroBar({ value, goal, color }: { value: number; goal: number; color: string }) {
  const pct = Math.min((value / goal) * 100, 100);
  return (
    <div
      style={{
        height: 3,
        background: "var(--border)",
        borderRadius: 2,
        marginTop: 6,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: `${pct}%`,
          height: "100%",
          background: color,
          borderRadius: 2,
          transition: "width 0.4s ease",
        }}
      />
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

// ── Expandable Meal Card (Appediet-style) ─────────────────────────────────

function MealCard({ log, isLast }: { log: FoodLog; isLast: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const items = log.foodItems ?? [];
  const totalP = items.reduce((s, f) => s + (f.protein_g ?? 0), 0);
  const totalC = items.reduce((s, f) => s + (f.carbs_g ?? 0), 0);
  const totalF = items.reduce((s, f) => s + (f.fat_g ?? 0), 0);

  // AI insight for this meal
  const insight = (() => {
    const cal = log.totalCalories ?? 0;
    if (cal > 800) return "🔥 High-calorie meal — balance with lighter options later";
    if (cal < 200) return "🥗 Light meal — you may need a snack soon";
    if (totalP > 25) return "💪 Great protein intake — supports muscle and mood";
    if (totalC > 60) return "⚡ Carb-heavy — expect an energy boost, then a dip";
    return "✅ Balanced meal — good nutrient distribution";
  })();

  return (
    <div style={{ borderBottom: isLast ? "none" : "1px solid var(--border)" }}>
      {/* Main row — tappable to expand */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: "flex", alignItems: "center", padding: "12px 14px", cursor: "pointer",
        }}
      >
        <span style={{ fontSize: 18, marginRight: 10 }}>
          {MEAL_ICONS[log.mealType ?? "snack"] ?? "🍽️"}
        </span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 13, fontWeight: 500, color: "var(--foreground)",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {log.summary ?? "Meal"}
          </div>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>
            {formatTime(log.loggedAt)} · {getMealLabel(log.mealType)}
          </div>
        </div>
        {log.totalCalories != null && (
          <div style={{ fontSize: 13, fontWeight: 600, color: "#e8b94a", flexShrink: 0, marginRight: 6 }}>
            {log.totalCalories} kcal
          </div>
        )}
        <span style={{ color: "var(--muted-foreground)", fontSize: 14, transition: "transform 0.2s", transform: expanded ? "rotate(90deg)" : "none" }}>›</span>
      </div>

      {/* Expanded: per-item breakdown + AI insight */}
      {expanded && (
        <div style={{ padding: "0 14px 12px 42px" }}>
          {/* Per-item list */}
          {items.length > 0 && (
            <div style={{ marginBottom: 8 }}>
              {items.map((item, i) => (
                <div key={i} style={{
                  display: "flex", justifyContent: "space-between", alignItems: "baseline",
                  fontSize: 11, color: "var(--muted-foreground)", padding: "3px 0",
                }}>
                  <span style={{ color: "var(--foreground)", fontWeight: 500 }}>{item.name}</span>
                  <span style={{ fontSize: 10, flexShrink: 0, marginLeft: 8 }}>
                    {Math.round(item.calories)} kcal · {item.portion}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Macro breakdown */}
          <div style={{
            display: "flex", gap: 12, fontSize: 10, color: "var(--muted-foreground)", marginBottom: 8,
          }}>
            <span><span style={{ color: "#7ba7d9" }}>P</span> {Math.round(totalP)}g</span>
            <span><span style={{ color: "#e8b94a" }}>C</span> {Math.round(totalC)}g</span>
            <span><span style={{ color: "#e87676" }}>F</span> {Math.round(totalF)}g</span>
          </div>

          {/* AI insight */}
          <div style={{
            fontSize: 11, color: "var(--muted-foreground)", fontStyle: "italic",
            padding: "6px 10px", background: "var(--muted)", borderRadius: 8,
          }}>
            {insight}
          </div>
        </div>
      )}
    </div>
  );
}

export default function Nutrition() {
  const userId = getParticipantId();
  const qc = useQueryClient();
  const [captureMode, setCaptureMode] = useState<"none" | "camera" | "text">("none");
  const [mealText, setMealText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const { data: logs } = useQuery<FoodLog[]>({
    queryKey: ["/api/food/logs", userId],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/food/logs/${userId}`));
      if (!res.ok) return [];
      return res.json();
    },
  });

  const todayLogs = useMemo(() => {
    if (!logs) return [];
    return logs.filter((l) => isToday(l.loggedAt)).sort(
      (a, b) => new Date(b.loggedAt).getTime() - new Date(a.loggedAt).getTime()
    );
  }, [logs]);

  const totalCalories = useMemo(
    () => todayLogs.reduce((s, l) => s + (l.totalCalories ?? 0), 0),
    [todayLogs]
  );

  const { totalProtein, totalCarbs, totalFat } = useMemo(() => {
    let p = 0, c = 0, f = 0;
    for (const l of todayLogs) {
      for (const fi of l.foodItems ?? []) {
        p += fi.protein_g ?? 0;
        c += fi.carbs_g ?? 0;
        f += fi.fat_g ?? 0;
      }
    }
    return { totalProtein: p, totalCarbs: c, totalFat: f };
  }, [todayLogs]);

  const voiceData = useVoiceData();
  const craving = useMemo(() => getCravingFromVoice(voiceData), [voiceData]);

  function handleAnalyzed() {
    // Delay slightly so the DB write completes before refetch
    setTimeout(() => {
      qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
    }, 600);
  }

  return (
    <main
      style={{
        background: "var(--background)",
        minHeight: "100vh",
        padding: 16,
        paddingBottom: 100,
        color: "var(--foreground)",
        fontFamily: "Inter, system-ui, sans-serif",
      }}
    >
      {/* Header */}
      <h1 style={{ fontSize: 18, fontWeight: 600, marginBottom: 20, marginTop: 4 }}>
        Nutrition
      </h1>

      {/* Calorie Ring */}
      <CalorieRing calories={totalCalories} />

      {/* Macro Cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: 8,
          marginBottom: 16,
        }}
      >
        {/* Protein */}
        <div
          style={{
            background: "var(--card)",
            borderRadius: 12,
            border: "1px solid var(--border)",
            padding: "12px 8px",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 700, color: "#3b82f6" }}>
            {Math.round(totalProtein)}g
          </div>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>Protein</div>
          <MacroBar value={totalProtein} goal={PROTEIN_GOAL} color="#3b82f6" />
        </div>

        {/* Carbs */}
        <div
          style={{
            background: "var(--card)",
            borderRadius: 12,
            border: "1px solid var(--border)",
            padding: "12px 8px",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 700, color: "#d4a017" }}>
            {Math.round(totalCarbs)}g
          </div>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>Carbs</div>
          <MacroBar value={totalCarbs} goal={CARBS_GOAL} color="#d4a017" />
        </div>

        {/* Fat */}
        <div
          style={{
            background: "var(--card)",
            borderRadius: 12,
            border: "1px solid var(--border)",
            padding: "12px 8px",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 700, color: "#e879a8" }}>
            {Math.round(totalFat)}g
          </div>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>Fat</div>
          <MacroBar value={totalFat} goal={FAT_GOAL} color="#e879a8" />
        </div>
      </div>

      {/* Craving Analysis Card */}
      <div
        style={{
          background: "var(--card)",
          border: "1px solid #2d2418",
          borderRadius: 14,
          padding: 14,
          marginBottom: 16,
        }}
      >
        <div
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: "#d4a017",
            marginBottom: 8,
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <span>🧠</span>
          <span>Craving Analysis</span>
        </div>
        <p style={{ fontSize: 13, color: "var(--foreground)", lineHeight: 1.5, margin: 0 }}>
          Right now you show signs of <strong style={{ color: "var(--foreground)" }}>{craving.text}</strong>.
          Track your meals to see how your emotional state shapes your eating patterns.
        </p>
      </div>

      {/* Hidden file inputs */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        style={{ display: "none" }}
        onChange={async (e) => {
          const file = e.target.files?.[0];
          if (!file) return;
          setIsAnalyzing(true);
          setAnalysisError(null);
          setCaptureMode("camera");
          try {
            const reader = new FileReader();
            const base64 = await new Promise<string>((resolve, reject) => {
              reader.onload = () => {
                const result = reader.result as string;
                resolve(result.split(",")[1]); // strip data:image/...;base64,
              };
              reader.onerror = reject;
              reader.readAsDataURL(file);
            });
            const res = await apiRequest("POST", "/api/food/analyze", {
              userId,
              mealType: autoMealType(),
              imageBase64: base64,
            });
            await res.json();
            hapticSuccess();
            try { localStorage.setItem("ndw_meal_logged", "true"); } catch {}
            await new Promise(r => setTimeout(r, 500));
            qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
            setCaptureMode("none");
          } catch (err) {
            setAnalysisError(err instanceof Error ? err.message : "Analysis failed");
          } finally {
            setIsAnalyzing(false);
            if (cameraInputRef.current) cameraInputRef.current.value = "";
          }
        }}
      />
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={async (e) => {
          const file = e.target.files?.[0];
          if (!file) return;
          setIsAnalyzing(true);
          setAnalysisError(null);
          setCaptureMode("camera");
          try {
            const reader = new FileReader();
            const base64 = await new Promise<string>((resolve, reject) => {
              reader.onload = () => {
                const result = reader.result as string;
                resolve(result.split(",")[1]);
              };
              reader.onerror = reject;
              reader.readAsDataURL(file);
            });
            const res = await apiRequest("POST", "/api/food/analyze", {
              userId,
              mealType: autoMealType(),
              imageBase64: base64,
            });
            await res.json();
            hapticSuccess();
            await new Promise(r => setTimeout(r, 500));
            qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
            setCaptureMode("none");
          } catch (err) {
            setAnalysisError(err instanceof Error ? err.message : "Analysis failed");
          } finally {
            setIsAnalyzing(false);
            if (fileInputRef.current) fileInputRef.current.value = "";
          }
        }}
      />

      {/* Mindful Eating Prompt — appears when emotional eating detected */}
      {captureMode === "none" && voiceData && (voiceData.stress_index ?? 0) > 0.4 && (
        <div style={{
          background: "var(--card)", border: "1px solid var(--border)",
          borderRadius: 14, padding: 14, marginBottom: 12,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <span style={{ fontSize: 16 }}>🧘</span>
            <span style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)" }}>Before you eat...</span>
          </div>
          <p style={{ fontSize: 12, color: "var(--muted-foreground)", margin: 0, lineHeight: 1.5 }}>
            Your voice analysis shows elevated stress. Take a breath and ask yourself:
            <strong style={{ color: "var(--foreground)" }}> Am I eating because I'm hungry, or because I'm feeling {voiceData.emotion ?? "stressed"}?</strong>
          </p>
          <p style={{ fontSize: 10, color: "var(--muted-foreground)", margin: "6px 0 0 0", fontStyle: "italic" }}>
            No judgment — just awareness. Log your meal either way.
          </p>
        </div>
      )}

      {/* Action Buttons — Appediet-style: Scan (primary) + Describe + Barcode */}
      {captureMode === "none" && (
        <div style={{ marginBottom: 14 }}>
          {/* Primary: Camera scan — large button */}
          <button
            onClick={() => cameraInputRef.current?.click()}
            style={{
              width: "100%", background: "linear-gradient(135deg, #e8b94a, #d4940a)",
              color: "#13111a", borderRadius: 14, padding: 14, fontSize: 14, fontWeight: 700,
              border: "none", cursor: "pointer", marginBottom: 8,
              display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
            }}
          >
            📷 Scan Your Meal
          </button>
          {/* Secondary row: Describe + Barcode */}
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={() => setCaptureMode("text")}
              style={{
                flex: 1, background: "var(--card)", color: "var(--foreground)", borderRadius: 12,
                padding: 10, fontSize: 12, fontWeight: 500, border: "1px solid var(--border)", cursor: "pointer",
              }}
            >
              ✍ Describe Meal
            </button>
            <button
              onClick={() => setCaptureMode("text")}
              style={{
                flex: 1, background: "var(--card)", color: "var(--foreground)", borderRadius: 12,
                padding: 10, fontSize: 12, fontWeight: 500, border: "1px solid var(--border)", cursor: "pointer",
              }}
            >
              📊 Enter Barcode
            </button>
          </div>
        </div>
      )}

      {/* Analyzing state */}
      {isAnalyzing && (
        <div style={{
          background: "var(--card)", borderRadius: 14, border: "1px solid var(--border)",
          padding: 20, marginBottom: 14, textAlign: "center",
        }}>
          <div style={{ width: 28, height: 28, border: "3px solid #d4a017", borderTopColor: "transparent",
            borderRadius: "50%", margin: "0 auto 8px", animation: "spin 0.8s linear infinite" }} />
          <p style={{ fontSize: 13, color: "var(--foreground)", margin: 0 }}>Analyzing your meal...</p>
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
      )}

      {/* Error */}
      {analysisError && (
        <div style={{
          background: "var(--card)", borderRadius: 14, border: "1px solid #2d1f18",
          padding: 14, marginBottom: 14,
        }}>
          <p style={{ fontSize: 12, color: "#e879a8", margin: 0 }}>{analysisError}</p>
          <button onClick={() => { setAnalysisError(null); setCaptureMode("none"); }}
            style={{ marginTop: 8, fontSize: 12, color: "var(--muted-foreground)", background: "none", border: "none", cursor: "pointer" }}>
            Try again
          </button>
        </div>
      )}

      {/* Describe mode — text input */}
      {captureMode === "text" && (
        <div style={{
          background: "var(--card)", borderRadius: 14, border: "1px solid var(--border)",
          padding: 14, marginBottom: 14,
        }}>
          <p style={{ fontSize: 12, color: "var(--muted-foreground)", margin: "0 0 8px 0" }}>
            What did you eat? Be specific for better accuracy.
          </p>
          <textarea
            value={mealText}
            onChange={(e) => setMealText(e.target.value)}
            placeholder="e.g. rice bowl with grilled chicken, steamed vegetables, and soy sauce"
            style={{
              width: "100%", minHeight: 80, background: "var(--background)", color: "var(--foreground)",
              border: "1px solid var(--border)", borderRadius: 10, padding: 12, fontSize: 13,
              resize: "none", outline: "none", fontFamily: "inherit",
            }}
          />
          <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
            <button
              onClick={() => { setCaptureMode("none"); setMealText(""); }}
              style={{
                flex: 1, background: "transparent", color: "var(--muted-foreground)", borderRadius: 10,
                padding: 10, fontSize: 13, border: "1px solid var(--border)", cursor: "pointer",
              }}
            >
              Cancel
            </button>
            <button
              disabled={!mealText.trim() || isAnalyzing}
              onClick={async () => {
                if (!mealText.trim()) return;
                setIsAnalyzing(true);
                setAnalysisError(null);
                try {
                  const res = await apiRequest("POST", "/api/food/analyze", {
                    userId,
                    mealType: autoMealType(),
                    textDescription: mealText.trim(),
                  });
                  await res.json();
                  hapticSuccess();
                  setMealText("");
                  await new Promise(r => setTimeout(r, 500));
                  qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
                  setCaptureMode("none");
                } catch (err) {
                  setAnalysisError(err instanceof Error ? err.message : "Analysis failed");
                } finally {
                  setIsAnalyzing(false);
                }
              }}
              style={{
                flex: 1, background: mealText.trim() ? "#d4a017" : "var(--muted)",
                color: mealText.trim() ? "#0a0e17" : "var(--muted-foreground)", borderRadius: 10,
                padding: 10, fontSize: 13, fontWeight: 600, border: "none", cursor: "pointer",
              }}
            >
              {isAnalyzing ? "Analyzing..." : "Log Meal"}
            </button>
          </div>
        </div>
      )}

      {/* Today's Meals */}
      <div>
        <div
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: "var(--muted-foreground)",
            textTransform: "uppercase",
            letterSpacing: "0.5px",
            marginBottom: 8,
          }}
        >
          Today's Meals
        </div>

        <div
          style={{
            background: "var(--card)",
            borderRadius: 14,
            border: "1px solid var(--border)",
            overflow: "hidden",
          }}
        >
          {todayLogs.length === 0 ? (
            <div
              style={{
                padding: "24px 16px",
                textAlign: "center",
                fontSize: 13,
                color: "var(--muted-foreground)",
              }}
            >
              Log your first meal to start tracking
            </div>
          ) : (
            todayLogs.map((log, idx) => (
              <MealCard key={log.id} log={log} isLast={idx === todayLogs.length - 1} />
            ))
          )}
        </div>
      </div>
    </main>
  );
}
