import { useState, useMemo, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { resolveUrl, apiRequest } from "@/lib/queryClient";
import { getParticipantId } from "@/lib/participant";

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

function getCravingAnalysis(): { text: string; label: string } {
  try {
    const raw = localStorage.getItem("ndw_last_emotion");
    if (!raw) return { text: "balanced — you're eating from hunger, not emotion", label: "Balanced" };
    const emotion = JSON.parse(raw) as { stress_index?: number; valence?: number };
    const stress = emotion.stress_index ?? 0;
    const valence = emotion.valence ?? 0;
    if (stress > 0.6) {
      return { text: "stress eating — your body seeks comfort food", label: "Stress" };
    }
    if (valence > 0.3) {
      return { text: "mindful eating — you're calm and present", label: "Mindful" };
    }
    if (valence < -0.2) {
      return { text: "comfort seeking — emotional eating tendency", label: "Comfort" };
    }
    return { text: "balanced — you're eating from hunger, not emotion", label: "Balanced" };
  } catch {
    return { text: "balanced — you're eating from hunger, not emotion", label: "Balanced" };
  }
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
            stroke="#1f2937"
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
              <stop offset="0%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#f97316" />
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
          <span style={{ fontSize: 28, fontWeight: 700, color: "#e8e0d4", lineHeight: 1 }}>
            {calories}
          </span>
          <span style={{ fontSize: 11, color: "#8b8578", marginTop: 2 }}>of {CAL_GOAL} kcal</span>
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
        background: "#1f2937",
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

  const craving = useMemo(() => getCravingAnalysis(), []);

  function handleAnalyzed() {
    // Delay slightly so the DB write completes before refetch
    setTimeout(() => {
      qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
    }, 600);
  }

  return (
    <main
      style={{
        background: "#0a0e17",
        minHeight: "100vh",
        padding: 16,
        paddingBottom: 100,
        color: "#e8e0d4",
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
            background: "#111827",
            borderRadius: 12,
            border: "1px solid #1f2937",
            padding: "12px 8px",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 700, color: "#60a5fa" }}>
            {Math.round(totalProtein)}g
          </div>
          <div style={{ fontSize: 10, color: "#8b8578", marginTop: 2 }}>Protein</div>
          <MacroBar value={totalProtein} goal={PROTEIN_GOAL} color="#60a5fa" />
        </div>

        {/* Carbs */}
        <div
          style={{
            background: "#111827",
            borderRadius: 12,
            border: "1px solid #1f2937",
            padding: "12px 8px",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 700, color: "#f59e0b" }}>
            {Math.round(totalCarbs)}g
          </div>
          <div style={{ fontSize: 10, color: "#8b8578", marginTop: 2 }}>Carbs</div>
          <MacroBar value={totalCarbs} goal={CARBS_GOAL} color="#f59e0b" />
        </div>

        {/* Fat */}
        <div
          style={{
            background: "#111827",
            borderRadius: 12,
            border: "1px solid #1f2937",
            padding: "12px 8px",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 700, color: "#f87171" }}>
            {Math.round(totalFat)}g
          </div>
          <div style={{ fontSize: 10, color: "#8b8578", marginTop: 2 }}>Fat</div>
          <MacroBar value={totalFat} goal={FAT_GOAL} color="#f87171" />
        </div>
      </div>

      {/* Craving Analysis Card */}
      <div
        style={{
          background: "linear-gradient(135deg, #1a1410, #111827)",
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
            color: "#f59e0b",
            marginBottom: 8,
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <span>🧠</span>
          <span>Craving Analysis</span>
        </div>
        <p style={{ fontSize: 13, color: "#d1cdc4", lineHeight: 1.5, margin: 0 }}>
          Right now you show signs of <strong style={{ color: "#e8e0d4" }}>{craving.text}</strong>.
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
            await new Promise(r => setTimeout(r, 500));
            qc.invalidateQueries({ queryKey: [resolveUrl(`/api/food/logs/${userId}`)] });
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
            await new Promise(r => setTimeout(r, 500));
            qc.invalidateQueries({ queryKey: [resolveUrl(`/api/food/logs/${userId}`)] });
            setCaptureMode("none");
          } catch (err) {
            setAnalysisError(err instanceof Error ? err.message : "Analysis failed");
          } finally {
            setIsAnalyzing(false);
            if (fileInputRef.current) fileInputRef.current.value = "";
          }
        }}
      />

      {/* Action Buttons */}
      {captureMode === "none" && (
        <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
          <button
            onClick={() => cameraInputRef.current?.click()}
            style={{
              flex: 1, background: "#f59e0b", color: "#0a0e17", borderRadius: 12,
              padding: 12, fontSize: 13, fontWeight: 600, border: "none", cursor: "pointer",
            }}
          >
            📷 Capture Meal
          </button>
          <button
            onClick={() => setCaptureMode("text")}
            style={{
              flex: 1, background: "#111827", color: "#e8e0d4", borderRadius: 12,
              padding: 12, fontSize: 13, fontWeight: 500, border: "1px solid #1f2937", cursor: "pointer",
            }}
          >
            ✍ Describe
          </button>
        </div>
      )}

      {/* Analyzing state */}
      {isAnalyzing && (
        <div style={{
          background: "#111827", borderRadius: 14, border: "1px solid #1f2937",
          padding: 20, marginBottom: 14, textAlign: "center",
        }}>
          <div style={{ width: 28, height: 28, border: "3px solid #f59e0b", borderTopColor: "transparent",
            borderRadius: "50%", margin: "0 auto 8px", animation: "spin 0.8s linear infinite" }} />
          <p style={{ fontSize: 13, color: "#e8e0d4", margin: 0 }}>Analyzing your meal...</p>
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
      )}

      {/* Error */}
      {analysisError && (
        <div style={{
          background: "#1f1210", borderRadius: 14, border: "1px solid #2d1f18",
          padding: 14, marginBottom: 14,
        }}>
          <p style={{ fontSize: 12, color: "#f87171", margin: 0 }}>{analysisError}</p>
          <button onClick={() => { setAnalysisError(null); setCaptureMode("none"); }}
            style={{ marginTop: 8, fontSize: 12, color: "#8b8578", background: "none", border: "none", cursor: "pointer" }}>
            Try again
          </button>
        </div>
      )}

      {/* Describe mode — text input */}
      {captureMode === "text" && (
        <div style={{
          background: "#111827", borderRadius: 14, border: "1px solid #1f2937",
          padding: 14, marginBottom: 14,
        }}>
          <p style={{ fontSize: 12, color: "#8b8578", margin: "0 0 8px 0" }}>
            What did you eat? Be specific for better accuracy.
          </p>
          <textarea
            value={mealText}
            onChange={(e) => setMealText(e.target.value)}
            placeholder="e.g. rice bowl with grilled chicken, steamed vegetables, and soy sauce"
            style={{
              width: "100%", minHeight: 80, background: "#0a0e17", color: "#e8e0d4",
              border: "1px solid #1f2937", borderRadius: 10, padding: 12, fontSize: 13,
              resize: "none", outline: "none", fontFamily: "inherit",
            }}
          />
          <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
            <button
              onClick={() => { setCaptureMode("none"); setMealText(""); }}
              style={{
                flex: 1, background: "transparent", color: "#8b8578", borderRadius: 10,
                padding: 10, fontSize: 13, border: "1px solid #1f2937", cursor: "pointer",
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
                  setMealText("");
                  await new Promise(r => setTimeout(r, 500));
                  qc.invalidateQueries({ queryKey: [resolveUrl(`/api/food/logs/${userId}`)] });
                  setCaptureMode("none");
                } catch (err) {
                  setAnalysisError(err instanceof Error ? err.message : "Analysis failed");
                } finally {
                  setIsAnalyzing(false);
                }
              }}
              style={{
                flex: 1, background: mealText.trim() ? "#f59e0b" : "#374151",
                color: mealText.trim() ? "#0a0e17" : "#6b7280", borderRadius: 10,
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
            color: "#8b8578",
            textTransform: "uppercase",
            letterSpacing: "0.5px",
            marginBottom: 8,
          }}
        >
          Today's Meals
        </div>

        <div
          style={{
            background: "#111827",
            borderRadius: 14,
            border: "1px solid #1f2937",
            overflow: "hidden",
          }}
        >
          {todayLogs.length === 0 ? (
            <div
              style={{
                padding: "24px 16px",
                textAlign: "center",
                fontSize: 13,
                color: "#8b8578",
              }}
            >
              Log your first meal to start tracking
            </div>
          ) : (
            todayLogs.map((log, idx) => (
              <div
                key={log.id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  padding: "12px 14px",
                  borderBottom:
                    idx < todayLogs.length - 1 ? "1px solid #1f2937" : "none",
                }}
              >
                {/* Meal icon */}
                <span style={{ fontSize: 18, marginRight: 10 }}>
                  {MEAL_ICONS[log.mealType ?? "snack"] ?? "🍽️"}
                </span>

                {/* Name + time */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontSize: 13,
                      fontWeight: 500,
                      color: "#e8e0d4",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {log.summary ?? "Meal"}
                  </div>
                  <div style={{ fontSize: 10, color: "#8b8578", marginTop: 2 }}>
                    {formatTime(log.loggedAt)} · {getMealLabel(log.mealType)}
                  </div>
                </div>

                {/* Calories */}
                {log.totalCalories != null && (
                  <div
                    style={{
                      fontSize: 13,
                      fontWeight: 600,
                      color: "#f59e0b",
                      flexShrink: 0,
                    }}
                  >
                    {log.totalCalories} kcal
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </main>
  );
}
