# App Reliability + Full Model Display — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all navigation crashes, food tab state bugs, and device restore issues; then surface all 16 ML model outputs across every page; then add Apple Health connect UI.

**Architecture:** Patch-and-extend — each task is fully independent and commits separately. No rewrites. Every change is the minimum needed to make the feature work.

**Tech Stack:** React 18 + TypeScript, wouter, TanStack Query, shadcn/ui, Tailwind CSS, FastAPI ML backend (port 8080), Express (port 4000).

---

## Task 1: Fix ErrorBoundary — navigation no longer crashes whole app

**Files:**
- Modify: `client/src/App.tsx` (lines 114–116)

**Context:** `AppRoutes()` has a single `<ErrorBoundary>` wrapping all routes. When any page crashes, `hasError=true` stays set forever — every page shows "Something went wrong" until full reload. Adding `key={location}` makes React unmount+remount the ErrorBoundary on every navigation, resetting error state automatically.

**Step 1: Open `client/src/App.tsx` and find `AppRoutes`**

It starts at line 114:
```tsx
function AppRoutes() {
  return (
    <ErrorBoundary>
```

**Step 2: Add `useLocation` import and `key` prop**

Replace this:
```tsx
function AppRoutes() {
  return (
    <ErrorBoundary>
```

With this:
```tsx
function AppRoutes() {
  const [location] = useLocation();
  return (
    <ErrorBoundary key={location}>
```

`useLocation` is already imported from `wouter` at the top of the file (line 2).

**Step 3: Verify in browser**

1. Navigate to any page
2. Intentionally break something by going to `/device-setup` with ML backend off
3. Navigate to `/emotions`
4. Emotions page should show normally — NOT "Something went wrong"

**Step 4: Commit**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add client/src/App.tsx
git commit -m "fix: isolate ErrorBoundary per route — navigation no longer crashes whole app"
```

---

## Task 2: Fix food log — switching meal tabs no longer shows wrong meal's data

**Files:**
- Modify: `client/src/pages/food-log.tsx`

**Context:** The food log fetches ALL logs for the user with query key `["/api/food/logs", USER_ID]`. The `filterType` state controls which meal tab is active. The bug is that when switching tabs, React Query serves the cached result immediately (all meals), and the filter applied in the JSX either isn't filtering `mealType` correctly or `mealType` on the log records doesn't match the filter values.

**Step 1: Find the history query in `food-log.tsx` (around line 146)**

```tsx
const { data: history } = useQuery<FoodLog[]>({
  queryKey: ["/api/food/logs", USER_ID],
  queryFn: async () => {
    const res = await fetch(`/api/food/logs/${USER_ID}`, { credentials: "include" });
    if (!res.ok) return [];
    return res.json();
  },
});
```

**Step 2: Find where `filterType` filters the history list**

Search for `filterType` in the JSX render section. It will look something like:
```tsx
{history?.filter(item => filterType === "all" || item.mealType === filterType).map(...)}
```

**Step 3: Fix the filter to also clear the `analysis` state when switching tabs**

Add a `useEffect` that resets `analysis` when `filterType` changes (so logged result from breakfast doesn't linger when you switch to lunch):

Find the state declarations near line 133–137:
```tsx
const [filterType, setFilterType] = useState<FilterType>("all");
const [isAnalyzing, setIsAnalyzing] = useState(false);
const [analysis, setAnalysis] = useState<FoodAnalysis | null>(null);
```

Add this effect AFTER those declarations:
```tsx
// Clear result display when switching meal tabs
useEffect(() => {
  setAnalysis(null);
}, [filterType]);
```

**Step 4: Ensure the history filter matches the mealType field**

The `FoodLog` type has `mealType: string | null`. The `FilterType` values are `"all" | "breakfast" | "lunch" | "dinner" | "snack"`. Find where history is filtered in the JSX and make sure the comparison is correct:

```tsx
const filteredHistory = (history ?? []).filter(
  (item) => filterType === "all" || item.mealType === filterType
);
```

Replace any existing filter logic with this. Then use `filteredHistory` instead of `history` in the `.map()` call.

**Step 5: Commit**
```bash
git add client/src/pages/food-log.tsx
git commit -m "fix: clear food analysis result on tab switch, filter history by mealType"
```

---

## Task 3: Fix device state — streaming state restores after page reload

**Files:**
- Modify: `client/src/hooks/use-device.tsx` (onMount useEffect, around line 429)
- Verify: `client/.env` has `VITE_ML_API_URL=http://localhost:8080`

**Context:** On page reload, `DeviceProvider` calls `getDeviceStatus()` to check if the ML backend's device is still streaming. If the call fails (wrong URL, timeout), device state stays "disconnected." We add a single delayed retry and ensure the ML URL is always read from localStorage/env.

**Step 1: Verify `client/.env` exists with correct value**
```bash
cat /Users/sravyalu/NeuralDreamWorkshop/client/.env
```
Should contain: `VITE_ML_API_URL=http://localhost:8080`

If not, create it:
```bash
echo "VITE_ML_API_URL=http://localhost:8080" > /Users/sravyalu/NeuralDreamWorkshop/client/.env
```

**Step 2: Find the onMount effect in `use-device.tsx` (around line 429)**

```tsx
useEffect(() => {
  let cancelled = false;
  (async () => {
    try {
      const status = await getDeviceStatus();
      if (cancelled) return;
      setBrainflowAvailable(status.brainflow_available ?? false);
      setDevicesLoaded(true);
      if (status.streaming) {
        ...openWebSocket();
        startSession("general").catch(() => {});
      } else if (status.connected) {
        ...
      }
    } catch {
      // ML service not available
    }
  })();
  return () => { cancelled = true; };
}, [openWebSocket]);
```

**Step 3: Add retry after 3 seconds if first call fails**

Replace the entire `useEffect` block with:
```tsx
useEffect(() => {
  let cancelled = false;

  const tryRestore = async () => {
    try {
      const status = await getDeviceStatus();
      if (cancelled) return;
      setBrainflowAvailable(status.brainflow_available ?? false);
      setDevicesLoaded(true);
      if (status.streaming) {
        setDeviceStatus(status);
        setSelectedDevice(status.device_type);
        setState("streaming");
        isStreamingRef.current = true;
        reconnectRef.current = 0;
        openWebSocket();
        startSession("general").catch(() => {});
      } else if (status.connected) {
        setDeviceStatus(status);
        setSelectedDevice(status.device_type);
        setState("connected");
      }
    } catch {
      // First attempt failed — retry once after 3s (handles race with ML backend startup)
      if (!cancelled) {
        setTimeout(async () => {
          if (cancelled) return;
          try {
            const status = await getDeviceStatus();
            if (cancelled) return;
            setBrainflowAvailable(status.brainflow_available ?? false);
            setDevicesLoaded(true);
            if (status.streaming) {
              setDeviceStatus(status);
              setSelectedDevice(status.device_type);
              setState("streaming");
              isStreamingRef.current = true;
              reconnectRef.current = 0;
              openWebSocket();
            }
          } catch {
            // Still unreachable — leave state as disconnected
          }
        }, 3000);
      }
    }
  };

  tryRestore();
  return () => { cancelled = true; };
}, [openWebSocket]);
```

**Step 4: Commit**
```bash
git add client/src/hooks/use-device.tsx client/.env
git commit -m "fix: retry device status restore on reload, ensure ML URL reads from env"
```

---

## Task 4: Dashboard — show capability badges + last session snapshot always

**Files:**
- Modify: `client/src/pages/dashboard.tsx`

**Context:** When not streaming, the dashboard shows only a "Connect device" banner and empty cards. Add: (a) a row of 16 capability badges always visible so user knows what's being tracked, (b) last session snapshot cards showing real data from the most recent session.

**Step 1: Find the return statement in `dashboard.tsx` (line 426)**

The current structure:
```tsx
return (
  <main className="p-6 space-y-6 max-w-6xl">
    {/* 1. Connection Banner */}
    {!isStreaming && (
      <Link href="/device-setup">...
```

**Step 2: Add capability badges row AFTER the connection banner (after line ~439)**

Insert this block after the `{!isStreaming && ...}` connection banner block:

```tsx
{/* Capability Badges — always visible */}
<div className="flex flex-wrap gap-1.5">
  {[
    "Sleep Staging", "Emotion", "Flow State", "Creativity",
    "Memory", "Drowsiness", "Cognitive Load", "Attention",
    "Stress", "Lucid Dream", "Meditation", "Anomaly",
    "Artifact", "Denoising", "Online Learning", "Dream Detect"
  ].map((cap) => (
    <span
      key={cap}
      className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-primary/8 text-primary/70 border border-primary/15"
    >
      {cap}
    </span>
  ))}
</div>
```

**Step 3: Add "Last Session" snapshot below capability badges when NOT streaming**

Find the variable `lastSession` (already computed around line 386). Add this block after the capability badges and before the existing `{(lastInsight || sessionsWithData.length > 0) && ...}` section:

```tsx
{/* Last Session Snapshot — shown when not live */}
{!isStreaming && lastSession && lastSession.summary && (
  <Card className="glass-card p-4">
    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
      Last Session · {relativeDay(lastSession.start_time ?? 0)}
    </p>
    <div className="grid grid-cols-3 gap-3">
      {[
        { label: "Focus", value: Math.round((lastSession.summary.avg_focus ?? 0) * 100), color: "text-primary" },
        { label: "Stress", value: Math.round((lastSession.summary.avg_stress ?? 0) * 100), color: "text-destructive" },
        { label: "Flow", value: Math.round((lastSession.summary.avg_flow ?? 0) * 100), color: "text-success" },
      ].map(({ label, value, color }) => (
        <div key={label} className="text-center">
          <p className={`text-2xl font-bold font-mono ${color}`}>{value}%</p>
          <p className="text-[10px] text-muted-foreground mt-0.5">{label}</p>
        </div>
      ))}
    </div>
    {lastSession.summary.dominant_emotion && (
      <p className="text-xs text-muted-foreground mt-3 text-center">
        Dominant state: <span className="text-foreground font-medium capitalize">{lastSession.summary.dominant_emotion}</span>
        {" · "}{Math.round((lastSession.summary.duration_sec ?? 0) / 60)}m session
      </p>
    )}
  </Card>
)}
```

**Step 4: Verify `SessionSummary` type has `avg_stress` and `avg_flow`**

Check `client/src/lib/ml-api.ts` for the `SessionSummary` type. It should have these fields. If not, add them to the interface.

**Step 5: Commit**
```bash
git add client/src/pages/dashboard.tsx
git commit -m "feat: dashboard always shows capability badges + last session snapshot when offline"
```

---

## Task 5: Emotion Lab — all 6 probabilities + creativity + valence/arousal

**Files:**
- Modify: `client/src/pages/emotion-lab.tsx`

**Context:** Currently shows top emotion + 3 bars (stress/focus/relaxation). Missing: all 6 emotion probability bars, creativity score, valence/arousal 2D indicator.

**Step 1: Find where the bars are rendered (around line 183)**

```tsx
{/* Bars */}
<div className="space-y-3">
  <Bar label="Stress"      value={stress}      color="hsl(0,72%,55%)" />
  <Bar label="Focus"       value={focus}       color="hsl(152,60%,48%)" />
  <Bar label="Relaxation"  value={relaxation}  color="hsl(217,91%,60%)" />
</div>
```

**Step 2: Add emotion probability section after the main emotion display**

After the bars block (closing `</div>`) and before the closing of the live emotion block, add:

```tsx
{/* All 6 emotion probabilities */}
{emotions?.probabilities && (
  <div>
    <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-2">
      Emotion breakdown
    </p>
    <div className="space-y-1.5">
      {(["happy","sad","angry","fear","surprise","neutral"] as const).map((e) => {
        const prob = (emotions.probabilities?.[e] ?? 0) * 100;
        const isTop = e === emotion;
        return (
          <div key={e} className="flex items-center gap-2">
            <span className="text-xs w-16 text-muted-foreground capitalize shrink-0">{e}</span>
            <div className="flex-1 h-1.5 rounded-full bg-muted/40 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${prob}%`,
                  background: isTop ? "hsl(152,60%,48%)" : "hsl(220,12%,40%)"
                }}
              />
            </div>
            <span className="text-[10px] text-muted-foreground w-8 text-right shrink-0">
              {Math.round(prob)}%
            </span>
          </div>
        );
      })}
    </div>
  </div>
)}

{/* Creativity + Valence/Arousal row */}
{(() => {
  const creativity = (analysis as Record<string,any>)?.creativity;
  const creativityScore = Math.round((creativity?.creativity_score ?? 0) * 100);
  const arousal = Math.round((emotions?.arousal ?? 0) * 100);
  const valenceNum = emotions?.valence ?? 0;
  const valencePct = Math.round(((valenceNum + 1) / 2) * 100); // -1..1 → 0..100
  return (
    <div className="grid grid-cols-2 gap-3 pt-1">
      <div className="p-3 rounded-xl bg-muted/30 border border-border/20 text-center">
        <p className="text-lg font-bold font-mono text-accent">{creativityScore}%</p>
        <p className="text-[10px] text-muted-foreground mt-0.5">Creativity</p>
      </div>
      <div className="p-3 rounded-xl bg-muted/30 border border-border/20 text-center">
        <p className="text-lg font-bold font-mono text-secondary">{arousal}%</p>
        <p className="text-[10px] text-muted-foreground mt-0.5">Arousal</p>
      </div>
    </div>
  );
})()}
```

**Step 3: Verify `emotions?.probabilities` is typed**

In `client/src/hooks/use-device.tsx`, the `emotions` field in `EEGStreamFrame` has:
```tsx
probabilities?: Record<string, number>;
```
This is already there (line 48). Good.

**Step 4: Commit**
```bash
git add client/src/pages/emotion-lab.tsx
git commit -m "feat: emotion lab shows all 6 probabilities, creativity score, arousal"
```

---

## Task 6: Inner Energy — chakras always visible (not blank when offline)

**Files:**
- Modify: `client/src/pages/inner-energy.tsx` (line 87)

**Context:** Chakra activations are set to `0` when `!isStreaming` (line 87: `activation: isStreaming ? activations[i] : 0`). This makes all chakras show empty rings when not streaming. Fix: show baseline values (e.g., 15%) when offline, so the UI always looks alive and the user understands what each chakra represents.

**Step 1: Find line 87 in `inner-energy.tsx`**

```tsx
activation: isStreaming ? activations[i] : 0,
```

**Step 2: Replace with baseline fallback**

```tsx
activation: isStreaming ? activations[i] : 15,  // 15% baseline so rings are never empty
```

**Step 3: Find the other `isStreaming ? X : 0` patterns and update them**

Around lines 94, 98–100, 106–108:
```tsx
const meditationPercent = isStreaming ? Math.round(meditationScore * 100) : 0;
const consciousnessRaw = isStreaming ? (...) : 0;
const thirdEyeActivation = isStreaming ? (...) : 0;
```

Change each `isStreaming ? X : 0` to `isStreaming ? X : 10` for meditation and consciousness (so circles show a sliver rather than nothing).

**Step 4: Update the guidance text for offline state**

Line 121:
```tsx
setGuidance("Connect your Muse 2 to begin reading your energy centers from live EEG data.");
```

Change to:
```tsx
setGuidance("These are your 7 energy centers mapped to EEG brainwave frequencies. Connect your device to see live readings.");
```

**Step 5: Commit**
```bash
git add client/src/pages/inner-energy.tsx
git commit -m "feat: inner energy shows baseline chakra values when offline, not blank rings"
```

---

## Task 7: Biofeedback — show EEG context before starting exercise

**Files:**
- Modify: `client/src/pages/biofeedback.tsx`

**Context:** The page jumps straight to exercise selection with no context about the user's current state. Add a "Your brain right now" card at the top that shows current stress, focus, and a recommendation.

**Step 1: Find where `useDevice` is used in `biofeedback.tsx` (line ~7)**

```tsx
import { useDevice } from "@/hooks/use-device";
```

**Step 2: Find the destructuring of `useDevice()` (will be in the component body)**

It will look like:
```tsx
const { latestFrame, state: deviceState } = useDevice();
```

**Step 3: Add computed values for the context card (after the useDevice destructuring)**

```tsx
const isStreaming = deviceState === "streaming";
const emotions = latestFrame?.analysis?.emotions;
const stressPct = Math.round((emotions?.stress_index ?? 0) * 100);
const focusPct  = Math.round((emotions?.focus_index  ?? 0) * 100);

function getBreathingRecommendation(stress: number, focus: number): string {
  if (stress > 65) return "Your stress is elevated — 4-7-8 breathing drops heart rate within 30 seconds.";
  if (focus < 30)  return "Low focus detected — Coherence breathing at 5.5 breaths/min maximises mental clarity.";
  if (stress < 20 && focus > 60) return "You're already in a good state — Box breathing will sharpen your focus further.";
  return "Coherence breathing syncs your heart and brain for optimal performance.";
}
const recommendation = getBreathingRecommendation(stressPct, focusPct);
```

**Step 4: Add the context card at the TOP of the return JSX (before exercise selection)**

Find the opening `<main>` or `<div>` tag of the return statement and add this as the first child:

```tsx
{/* Brain state context */}
{isStreaming && (
  <Card className="p-4 border-border/30 mb-4">
    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
      Your brain right now
    </p>
    <div className="flex items-center gap-6 mb-3">
      <div className="text-center">
        <p className={`text-2xl font-bold font-mono ${stressPct > 65 ? "text-destructive" : stressPct > 35 ? "text-warning" : "text-success"}`}>
          {stressPct}%
        </p>
        <p className="text-[10px] text-muted-foreground">Stress</p>
      </div>
      <div className="text-center">
        <p className={`text-2xl font-bold font-mono ${focusPct > 60 ? "text-success" : "text-muted-foreground"}`}>
          {focusPct}%
        </p>
        <p className="text-[10px] text-muted-foreground">Focus</p>
      </div>
    </div>
    <p className="text-xs text-muted-foreground leading-relaxed">{recommendation}</p>
  </Card>
)}
```

**Step 5: Commit**
```bash
git add client/src/pages/biofeedback.tsx
git commit -m "feat: biofeedback shows live EEG context (stress/focus + recommendation) before exercise"
```

---

## Task 8: Settings — add Device Status card + Health Integrations card

**Files:**
- Modify: `client/src/pages/settings.tsx`

**Context:** No visible place to see device connection status or connect Apple Health. Add two new cards at the top of Settings.

**Step 1: Add `useDevice` import to `settings.tsx`**

At the top of the file, add:
```tsx
import { useDevice } from "@/hooks/use-device";
import { Link } from "wouter";
```

**Step 2: Add device state inside the component**

After `const { toast } = useToast();`, add:
```tsx
const { state: deviceState, selectedDevice } = useDevice();
const isStreaming = deviceState === "streaming";
const isConnected = deviceState === "connected" || deviceState === "streaming";
```

**Step 3: Add Device Status card at the top of the Settings return JSX**

Find the opening `<main>` or top-level div of the return and add FIRST:

```tsx
{/* Device Status */}
<Card className="p-5">
  <div className="flex items-center justify-between mb-3">
    <p className="text-sm font-semibold">Connected Device</p>
    <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${
      isStreaming ? "bg-success/10 text-success" :
      isConnected ? "bg-primary/10 text-primary" :
      "bg-muted text-muted-foreground"
    }`}>
      {isStreaming ? "Streaming" : isConnected ? "Connected" : "Not connected"}
    </span>
  </div>
  {isConnected ? (
    <p className="text-sm text-muted-foreground">
      Device: <span className="text-foreground font-medium capitalize">{selectedDevice ?? "Unknown"}</span>
    </p>
  ) : (
    <p className="text-sm text-muted-foreground mb-3">No EEG device connected.</p>
  )}
  <Link href="/device-setup">
    <button className="mt-3 text-xs text-primary hover:text-primary/80 transition-colors">
      {isConnected ? "Change device →" : "Connect device →"}
    </button>
  </Link>
</Card>

{/* Health Integrations */}
<Card className="p-5">
  <p className="text-sm font-semibold mb-4">Health Integrations</p>
  <div className="space-y-4">
    {/* Apple Health */}
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-red-500/10 flex items-center justify-center">
          <Apple className="h-5 w-5 text-red-400" />
        </div>
        <div>
          <p className="text-sm font-medium">Apple Health</p>
          <p className="text-xs text-muted-foreground">Sync sleep, HRV, stress & focus</p>
        </div>
      </div>
      <Button
        size="sm"
        variant="outline"
        onClick={async () => {
          try {
            const res = await fetch("/api/health/export-to-healthkit", {
              method: "POST",
              credentials: "include",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ user_id: userId }),
            });
            if (res.ok) {
              toast({ title: "Apple Health", description: "Data exported to Apple Health." });
            } else {
              toast({ title: "Not available", description: "Apple Health requires iOS or macOS.", variant: "destructive" });
            }
          } catch {
            toast({ title: "Not available", description: "Apple Health requires iOS or macOS.", variant: "destructive" });
          }
        }}
      >
        Connect
      </Button>
    </div>

    {/* Google Fit */}
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-blue-500/10 flex items-center justify-center">
          <Smartphone className="h-5 w-5 text-blue-400" />
        </div>
        <div>
          <p className="text-sm font-medium">Google Fit</p>
          <p className="text-xs text-muted-foreground">Sync activity, sleep & wellness</p>
        </div>
      </div>
      <Button
        size="sm"
        variant="outline"
        onClick={() => {
          toast({ title: "Google Fit", description: "Google Fit integration coming soon." });
        }}
      >
        Connect
      </Button>
    </div>
  </div>
</Card>
```

Note: `Apple`, `Smartphone` icons are already imported at line 27 in settings.tsx.

**Step 4: Commit**
```bash
git add client/src/pages/settings.tsx
git commit -m "feat: settings shows device status card + Apple Health / Google Fit connect cards"
```

---

## Task 9: Brain Monitor — 16-model output grid

**Files:**
- Modify: `client/src/pages/brain-monitor.tsx`

**Context:** Brain monitor has waveforms and band powers. Need to confirm all 16 model outputs are visible and add a compact grid card.

**Step 1: Read the current brain-monitor.tsx to find where model data is shown**
```bash
grep -n "creativity\|flow_state\|drowsiness\|cognitive_load\|attention\|meditation\|lucid" \
  /Users/sravyalu/NeuralDreamWorkshop/client/src/pages/brain-monitor.tsx | head -20
```

**Step 2: Find the analysis object destructuring**

It will look like:
```tsx
const analysis = latestFrame?.analysis;
```

**Step 3: Add a "16 Models Live" compact grid card**

Find a good insertion point (after band powers section, before or after signal quality). Add:

```tsx
{/* 16 Models — compact grid */}
{isStreaming && (
  <Card className="glass-card p-4">
    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
      All Models Live
    </p>
    <div className="grid grid-cols-2 gap-x-4 gap-y-2.5">
      {[
        { label: "Emotion",       value: analysis?.emotions?.emotion ?? "—",                              isText: true  },
        { label: "Stress",        value: Math.round((analysis?.emotions?.stress_index ?? 0) * 100) + "%", isText: false },
        { label: "Focus",         value: Math.round((analysis?.emotions?.focus_index ?? 0) * 100) + "%",  isText: false },
        { label: "Flow",          value: Math.round(((analysis as any)?.flow_state?.flow_score ?? 0) * 100) + "%", isText: false },
        { label: "Creativity",    value: Math.round(((analysis as any)?.creativity?.creativity_score ?? 0) * 100) + "%", isText: false },
        { label: "Meditation",    value: analysis?.meditation?.depth ?? "—",                              isText: true  },
        { label: "Sleep Stage",   value: analysis?.sleep_staging?.stage ?? "—",                           isText: true  },
        { label: "Drowsiness",    value: analysis?.drowsiness?.state ?? "—",                              isText: true  },
        { label: "Attention",     value: analysis?.attention?.state ?? "—",                               isText: true  },
        { label: "Cognitive Load",value: analysis?.cognitive_load?.level ?? "—",                         isText: true  },
        { label: "Dream Detect",  value: (analysis?.dream_detection?.is_dreaming ? "Dreaming" : "Awake"), isText: true  },
        { label: "Lucid Dream",   value: analysis?.lucid_dream?.state ?? "—",                            isText: true  },
        { label: "Memory",        value: ((analysis as any)?.memory_encoding?.state ?? "—"),              isText: true  },
        { label: "Relaxation",    value: Math.round((analysis?.emotions?.relaxation_index ?? 0) * 100) + "%", isText: false },
        { label: "Valence",       value: ((analysis?.emotions?.valence ?? 0) > 0 ? "Positive" : "Negative"), isText: true },
        { label: "Arousal",       value: Math.round((analysis?.emotions?.arousal ?? 0) * 100) + "%",     isText: false },
      ].map(({ label, value }) => (
        <div key={label} className="flex items-center justify-between py-0.5 border-b border-border/10">
          <span className="text-[10px] text-muted-foreground">{label}</span>
          <span className="text-[10px] font-mono text-foreground capitalize">{String(value)}</span>
        </div>
      ))}
    </div>
  </Card>
)}
```

**Step 4: Commit**
```bash
git add client/src/pages/brain-monitor.tsx
git commit -m "feat: brain monitor shows all 16 model outputs in compact live grid"
```

---

## Task 10: Push everything and verify

**Step 1: Run TypeScript check**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop/client
npx tsc --noEmit 2>&1 | head -40
```

Fix any type errors before pushing.

**Step 2: Check the running app**

- Navigate between pages: no "Something went wrong"
- Go to food log → switch breakfast/lunch/dinner tabs: analysis result clears
- Go to inner energy: chakra rings show ~15% baseline values
- Connect Synthetic device via device-setup
- Go to emotion lab: see all 6 probability bars
- Go to brain monitor: see "All Models Live" grid
- Go to biofeedback: see stress/focus card before exercises
- Go to settings: see "Connected Device" card + Apple Health / Google Fit cards

**Step 3: Push to GitHub**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git push
```

---

## Success Criteria

- [ ] Navigating between any two pages never shows "Something went wrong"
- [ ] Switching food tabs clears the analysis result from the previous tab
- [ ] Inner Energy page shows non-zero chakra rings when offline
- [ ] Emotion Lab shows all 6 emotion probability bars when streaming
- [ ] Brain Monitor shows 16-model live grid when streaming
- [ ] Biofeedback shows stress/focus context card when streaming
- [ ] Settings has "Connected Device" status + Apple Health + Google Fit cards
- [ ] Dashboard shows capability badges + last session snapshot when offline
- [ ] All TypeScript checks pass (`npx tsc --noEmit`)
- [ ] `git push` succeeds
