# Automatic Study Flow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the manual study flow with a fully automatic pipeline — intent selection at login, Muse auto-start/stop triggering sessions, 30s EEG checkpoints to DB, and a live admin dashboard.

**Architecture:** Post-login intent screen routes users to study or explore. Study sessions open a DB record the moment Muse pairs, auto-save EEG every 30s via a new PATCH checkpoint endpoint, and auto-close when Muse disconnects. No manual CSV export needed during the study.

**Tech Stack:** React 18 + TypeScript, Wouter routing, TanStack Query, Drizzle ORM, Neon PostgreSQL, shadcn/ui, Recharts, useDevice hook (Muse 2 BLE)

**Design doc:** `docs/plans/2026-02-27-automatic-study-flow-design.md`

---

## Context: Key Files

| File | Role |
|---|---|
| `shared/schema.ts` | Drizzle ORM schema — source of truth for DB tables |
| `server/routes.ts` | Express API routes (local dev) |
| `api/[...path].ts` | Vercel catch-all serverless handler (production) |
| `client/src/App.tsx` | All client routes + providers |
| `client/src/hooks/use-auth.tsx` | Auth state, login/register, post-login redirect |
| `client/src/hooks/use-device.tsx` | Muse 2 connection, `latestFrame`, device status |
| `client/src/pages/study/` | All study pages |
| `client/src/pages/study/StudyAdmin.tsx` | Researcher admin dashboard |

---

### Task 1: Schema — add `intent` to users + checkpoint columns to pilot_sessions

**Files:**
- Modify: `shared/schema.ts`

**Step 1: Add columns to schema**

In `shared/schema.ts`, find the `users` table and add one column:
```typescript
intent: varchar("intent", { length: 10 }),  // 'study' | 'explore' | null
```

Find the `pilotSessions` table and add three columns:
```typescript
partial:       boolean("partial").default(false),
phaseLog:      jsonb("phase_log"),
checkpointAt:  timestamp("checkpoint_at"),
```

**Step 2: Push schema to DB**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
set -a && source .env.local && set +a && npx drizzle-kit push
```

Expected: `[✓] Changes applied` — 4 new columns added across 2 tables.

**Step 3: Typecheck**

```bash
npx tsc --noEmit 2>&1 | head -20
```

Expected: no errors.

**Step 4: Commit**

```bash
git add shared/schema.ts
git commit -m "feat: add intent to users, checkpoint columns to pilot_sessions"
```

---

### Task 2: New API endpoint — PATCH /api/study/session/:id/checkpoint

**Files:**
- Modify: `server/routes.ts` (lines after the existing `/api/study/session/complete` route)
- Modify: `api/[...path].ts`

**Step 1: Add route to server/routes.ts**

After the `POST /api/study/session/complete` block, add:

```typescript
// PATCH /api/study/session/:id/checkpoint
app.patch("/api/study/session/:id/checkpoint", async (req, res) => {
  try {
    const sessionId = Number(req.params.id);
    if (isNaN(sessionId)) return res.status(400).json({ error: "invalid session id" });
    const { pre_eeg_json, post_eeg_json, eeg_features_json, intervention_triggered, partial, phase_log } = req.body;
    await db.update(pilotSessions)
      .set({
        ...(pre_eeg_json !== undefined   && { preEegJson: pre_eeg_json }),
        ...(post_eeg_json !== undefined  && { postEegJson: post_eeg_json }),
        ...(eeg_features_json !== undefined && { eegFeaturesJson: eeg_features_json }),
        ...(intervention_triggered !== undefined && { interventionTriggered: !!intervention_triggered }),
        ...(partial !== undefined        && { partial: !!partial }),
        ...(phase_log !== undefined      && { phaseLog: phase_log }),
        checkpointAt: new Date(),
      })
      .where(eq(pilotSessions.id, sessionId));
    return res.json({ success: true });
  } catch (err) {
    console.error("PATCH /api/study/session/:id/checkpoint error:", err);
    return res.status(500).json({ error: "checkpoint failed" });
  }
});
```

Also add `GET /api/user/intent` and `PATCH /api/user/intent` after the settings routes:

```typescript
// GET /api/user/intent
app.get("/api/user/intent", async (req, res) => {
  const userId = (req.session as { userId?: string }).userId;
  if (!userId) return res.status(401).json({ error: "Unauthorized" });
  const [u] = await db.select({ intent: users.intent }).from(users).where(eq(users.id, Number(userId)));
  return res.json({ intent: u?.intent ?? null });
});

// PATCH /api/user/intent
app.patch("/api/user/intent", async (req, res) => {
  const userId = (req.session as { userId?: string }).userId;
  if (!userId) return res.status(401).json({ error: "Unauthorized" });
  const { intent } = req.body;
  if (!["study", "explore"].includes(intent)) return res.status(400).json({ error: "invalid intent" });
  await db.update(users).set({ intent }).where(eq(users.id, Number(userId)));
  return res.json({ success: true, intent });
});
```

**Step 2: Add handlers to api/[...path].ts**

Add these two handler functions before the `// ── Main router` comment:

```typescript
async function pilotSessionCheckpoint(req: VercelRequest, res: VercelResponse, sessionId: number) {
  if (req.method !== 'PATCH') return methodNotAllowed(res, ['PATCH']);
  const { pre_eeg_json, post_eeg_json, eeg_features_json, intervention_triggered, partial, phase_log } = req.body;
  const db = getDb();
  await db.update(schema.pilotSessions)
    .set({
      ...(pre_eeg_json !== undefined   && { preEegJson: pre_eeg_json }),
      ...(post_eeg_json !== undefined  && { postEegJson: post_eeg_json }),
      ...(eeg_features_json !== undefined && { eegFeaturesJson: eeg_features_json }),
      ...(intervention_triggered !== undefined && { interventionTriggered: !!intervention_triggered }),
      ...(partial !== undefined        && { partial: !!partial }),
      ...(phase_log !== undefined      && { phaseLog: phase_log }),
      checkpointAt: new Date(),
    })
    .where(eq(schema.pilotSessions.id, sessionId));
  return success(res, { success: true });
}

async function userIntentGet(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const authResult = requireAuth(req);
  if (!authResult) return unauthorized(res);
  const db = getDb();
  const [u] = await db.select({ intent: schema.users.intent }).from(schema.users).where(eq(schema.users.id, authResult.userId));
  return success(res, { intent: u?.intent ?? null });
}

async function userIntentPatch(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'PATCH') return methodNotAllowed(res, ['PATCH']);
  const authResult = requireAuth(req);
  if (!authResult) return unauthorized(res);
  const { intent } = req.body;
  if (!['study', 'explore'].includes(intent)) return badRequest(res, 'invalid intent');
  const db = getDb();
  await db.update(schema.users).set({ intent }).where(eq(schema.users.id, authResult.userId));
  return success(res, { success: true, intent });
}
```

Add routing cases to the main handler:

```typescript
// Inside if (s0 === 'study'):
if (s1 === 'session' && segs[2] && segs[3] === 'checkpoint') {
  return await pilotSessionCheckpoint(req, res, Number(segs[2]));
}

// New top-level block:
if (s0 === 'user') {
  if (s1 === 'intent' && req.method === 'GET')   return await userIntentGet(req, res);
  if (s1 === 'intent' && req.method === 'PATCH')  return await userIntentPatch(req, res);
}
```

**Step 3: Typecheck**

```bash
npx tsc --noEmit 2>&1 | head -20
```

Expected: no errors. If `schema.users.intent` is not found, you forgot Task 1 — go back and re-push schema.

**Step 4: Test locally**

With server on port 4000:
```bash
# Start session first
curl -s -X POST http://localhost:4000/api/study/session/start \
  -H "Content-Type: application/json" \
  -d '{"participant_code":"P001","block_type":"stress"}'
# → {"session_id": N}

# Checkpoint it
curl -s -X PATCH http://localhost:4000/api/study/session/N/checkpoint \
  -H "Content-Type: application/json" \
  -d '{"eeg_features_json":{"alpha":0.4,"beta":0.3,"stress_level":0.6}}'
# → {"success": true}
```

**Step 5: Commit**

```bash
git add server/routes.ts 'api/[...path].ts'
git commit -m "feat: add checkpoint endpoint + user intent API"
```

---

### Task 3: Onboarding page — intent selection screen

**Files:**
- Create: `client/src/pages/onboarding.tsx`

**Step 1: Create the page**

```typescript
import { useState } from "react";
import { useLocation } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Brain, FlaskConical, Loader2 } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";

export default function Onboarding() {
  const [, navigate] = useLocation();
  const [loading, setLoading] = useState<"study" | "explore" | null>(null);

  async function choose(intent: "study" | "explore") {
    setLoading(intent);
    try {
      await apiRequest("PATCH", "/api/user/intent", { intent });
    } catch (_) {
      // non-fatal — intent will be re-asked next login if it fails
    }
    navigate(intent === "study" ? "/study" : "/");
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4">
      <div className="max-w-xl w-full space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold">Welcome to Neural Dream Workshop</h1>
          <p className="text-muted-foreground">What brings you here today?</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Card
            className="cursor-pointer border-2 hover:border-primary transition-colors"
            onClick={() => choose("study")}
          >
            <CardContent className="pt-8 pb-8 text-center space-y-4">
              <FlaskConical className="w-10 h-10 mx-auto text-primary" />
              <div>
                <p className="font-semibold text-lg">Join the Study</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Participate in EEG research sessions. ~30 min total.
                </p>
              </div>
              {loading === "study" && <Loader2 className="w-4 h-4 mx-auto animate-spin" />}
            </CardContent>
          </Card>

          <Card
            className="cursor-pointer border-2 hover:border-primary transition-colors"
            onClick={() => choose("explore")}
          >
            <CardContent className="pt-8 pb-8 text-center space-y-4">
              <Brain className="w-10 h-10 mx-auto text-violet-400" />
              <div>
                <p className="font-semibold text-lg">Explore the App</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Try the full dashboard, emotion lab, dream journal, and more.
                </p>
              </div>
              {loading === "explore" && <Loader2 className="w-4 h-4 mx-auto animate-spin" />}
            </CardContent>
          </Card>
        </div>

        <p className="text-center text-xs text-muted-foreground">
          You can switch later — study participants can explore after completing their sessions.
        </p>
      </div>
    </div>
  );
}
```

**Step 2: Add route in App.tsx**

Add lazy import near the other study imports:
```typescript
const Onboarding = lazy(() => import("@/pages/onboarding"));
```

Add the route (no ProtectedRoute needed but user must be logged in — add it):
```typescript
<Route path="/onboarding">
  <ProtectedRoute><Onboarding /></ProtectedRoute>
</Route>
```

**Step 3: Typecheck**

```bash
npx tsc --noEmit 2>&1 | head -20
```

**Step 4: Commit**

```bash
git add client/src/pages/onboarding.tsx client/src/App.tsx
git commit -m "feat: add onboarding intent-selection page"
```

---

### Task 4: Auth redirect — route by intent after login

**Files:**
- Modify: `client/src/hooks/use-auth.tsx`

**Step 1: Add intent fetch after login**

Inside `use-auth.tsx`, after a successful login/register that sets the user, add a check for intent:

Find where `login()` and `register()` navigate after success. Replace the hardcoded `navigate("/")` with:

```typescript
// After successful login/register, determine where to send the user
async function redirectAfterAuth() {
  try {
    const res = await fetch("/api/user/intent", { credentials: "include" });
    const data = await res.json();
    if (data.intent === "study") {
      setLocation("/study");
    } else if (data.intent === "explore") {
      setLocation("/");
    } else {
      setLocation("/onboarding"); // first time — no intent set yet
    }
  } catch {
    setLocation("/onboarding");
  }
}
```

Call `redirectAfterAuth()` instead of `setLocation("/")` in both `login()` and `register()` success handlers.

**Step 2: Typecheck**

```bash
npx tsc --noEmit 2>&1 | head -20
```

**Step 3: Manual test**

1. Log out, log back in — should land on `/onboarding`
2. Choose "Explore" — should go to `/`
3. Log out, log back in again — should go straight to `/` (intent saved)

**Step 4: Commit**

```bash
git add client/src/hooks/use-auth.tsx
git commit -m "feat: route to onboarding or intent-based destination after login"
```

---

### Task 5: Unified StudySession page — Muse auto-start + phase engine

**Files:**
- Create: `client/src/pages/study/StudySession.tsx`
- Modify: `client/src/App.tsx` (add route `/study/session`)

**Step 1: Create StudySession.tsx**

This replaces `StudySessionStress.tsx` and `StudySessionFood.tsx` with one smart page.

Core structure:
```typescript
import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation, useSearch } from "wouter";
import { useDevice } from "@/hooks/use-device";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
// ... shadcn/ui imports

type BlockType = "stress" | "food";
type Phase = "block-pick" | "muse-pair" | "baseline" | "task" | "intervention" | "recovery" | "survey" | "done";

export default function StudySession() {
  const [, navigate] = useLocation();
  const search = useSearch();
  const params = new URLSearchParams(search);
  const participantCode = params.get("code") ?? localStorage.getItem("ndw_study_code") ?? "";
  const { toast } = useToast();

  const { status: deviceStatus, connect, disconnect, latestFrame } = useDevice();

  const [blockType, setBlockType] = useState<BlockType | null>(null);
  const [phase, setPhase] = useState<Phase>("block-pick");
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [showStress, setShowStress] = useState(false);
  const [interventionTriggered, setInterventionTriggered] = useState(false);
  const [surveyAnswers, setSurveyAnswers] = useState<Record<string, number>>({});

  // EEG buffer
  const eegBuffer = useRef<{ phase: string; frame: typeof latestFrame; ts: number }[]>([]);
  const phaseLog = useRef<{ phase: string; at: number }[]>([]);
  const checkpointTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  // ... (see Steps 2-6 for the logic)
}
```

**Step 2: Block picker UI + session start on Muse pair**

When block is selected → show Muse pair screen.
When `deviceStatus` changes to `"connected"` → call session start API:

```typescript
useEffect(() => {
  if (deviceStatus === "connected" && blockType && !sessionId && phase === "muse-pair") {
    startSession();
  }
}, [deviceStatus, blockType, phase]);

async function startSession() {
  try {
    const res = await apiRequest("POST", "/api/study/session/start", {
      participant_code: participantCode,
      block_type: blockType,
    });
    const data = await res.json();
    setSessionId(data.session_id);
    setPhase("baseline");
    logPhase("baseline");
    startCheckpointLoop(data.session_id);
  } catch (err) {
    toast({ title: "Could not start session", variant: "destructive" });
  }
}
```

**Step 3: EEG buffer — collect latestFrame per phase**

```typescript
useEffect(() => {
  if (!latestFrame || !sessionId) return;
  eegBuffer.current.push({ phase: phase, frame: latestFrame, ts: Date.now() });

  // Auto-trigger intervention during task phase
  if (phase === "task" && !interventionTriggered) {
    const stressLevel = (latestFrame as any)?.emotions?.stress_index ?? 0;
    if (stressLevel > 0.65) {
      setInterventionTriggered(true);
      setPhase("intervention");
      logPhase("intervention");
    }
  }
}, [latestFrame]);

function logPhase(p: string) {
  phaseLog.current.push({ phase: p, at: Date.now() });
}
```

**Step 4: 30s checkpoint loop**

```typescript
function startCheckpointLoop(sid: number) {
  checkpointTimer.current = setInterval(() => checkpoint(sid, false), 30_000);
}

async function checkpoint(sid: number, isFinal: boolean) {
  const frames = eegBuffer.current;
  const byPhase = (p: string) => frames.filter(f => f.phase === p).map(f => f.frame);

  const avgBands = (fs: typeof latestFrame[]) => {
    if (!fs.length) return null;
    const keys = ["alpha", "beta", "theta", "delta", "gamma"];
    const result: Record<string, number> = {};
    for (const k of keys) {
      result[k] = fs.reduce((s, f) => s + ((f as any)?.emotions?.band_powers?.[k] ?? 0), 0) / fs.length;
    }
    return result;
  };

  await apiRequest("PATCH", `/api/study/session/${sid}/checkpoint`, {
    pre_eeg_json:        avgBands(byPhase("baseline")),
    post_eeg_json:       avgBands(byPhase("recovery")),
    eeg_features_json:   { frame_count: frames.length, last_stress: (latestFrame as any)?.emotions?.stress_index },
    intervention_triggered: interventionTriggered,
    phase_log:           phaseLog.current,
    ...(isFinal && { partial: false }),
  }).catch(() => {}); // non-fatal
}
```

**Step 5: Muse disconnect handler**

```typescript
useEffect(() => {
  if (deviceStatus === "disconnected" && sessionId && phase !== "survey" && phase !== "done") {
    // Unexpected disconnect
    if (checkpointTimer.current) clearInterval(checkpointTimer.current);
    checkpoint(sessionId, false); // save what we have
    setPhase("muse-pair"); // show reconnect overlay
    toast({ title: "Muse disconnected", description: "Reconnect to continue or save & exit." });
  }
}, [deviceStatus]);

async function saveAndExit() {
  if (sessionId) {
    await checkpoint(sessionId, false);
    await apiRequest("PATCH", `/api/study/session/${sessionId}/checkpoint`, { partial: true });
  }
  navigate(`/study/complete?code=${participantCode}&partial=true`);
}
```

**Step 6: Phase timer engine**

Phase durations (seconds):
- `baseline`: 300 (5 min)
- `task`: 900 (15 min)
- `intervention`: 180 (3 min box breathing)
- `recovery`: 300 (5 min)

```typescript
const PHASE_DURATIONS: Record<string, number> = {
  baseline: 300, task: 900, intervention: 180, recovery: 300
};

useEffect(() => {
  const dur = PHASE_DURATIONS[phase];
  if (!dur) return;
  const t = setTimeout(() => advancePhase(), dur * 1000);
  return () => clearTimeout(t);
}, [phase]);

function advancePhase() {
  const order: Phase[] = ["baseline", "task", "intervention", "recovery", "survey"];
  const idx = order.indexOf(phase as Phase);
  if (idx === -1 || idx >= order.length - 1) return;
  const next = order[idx + 1];
  setPhase(next);
  logPhase(next);
  if (next === "survey" && sessionId && checkpointTimer.current) {
    clearInterval(checkpointTimer.current);
    checkpoint(sessionId, true);
  }
}
```

**Step 7: Survey submit → navigate to complete**

```typescript
async function submitSurvey() {
  if (!sessionId) return;
  await apiRequest("POST", "/api/study/session/complete", {
    session_id: sessionId,
    survey_json: surveyAnswers,
    intervention_triggered: interventionTriggered,
  });
  navigate(`/study/complete?code=${participantCode}&done=${blockType}`);
}
```

**Step 8: Add route to App.tsx**

```typescript
const StudySession = lazy(() => import("@/pages/study/StudySession"));

// In routes:
<Route path="/study/session">
  <StudySession />
</Route>
```

**Step 9: Typecheck**

```bash
npx tsc --noEmit 2>&1 | head -40
```

Fix any type errors. Common ones: `latestFrame` is typed as `EEGStreamFrame | null` — cast appropriately, or use optional chaining.

**Step 10: Commit**

```bash
git add client/src/pages/study/StudySession.tsx client/src/App.tsx
git commit -m "feat: unified StudySession page with auto EEG capture and 30s checkpoints"
```

---

### Task 6: Update StudyProfile → navigate to unified session page

**Files:**
- Modify: `client/src/pages/study/StudyProfile.tsx`

**Step 1: Change navigation target**

Find the two `navigate(...)` calls that go to `/study/session/stress` and `/study/session/food`. Replace both with:

```typescript
navigate(`/study/session?code=${participantCode}&block=${sessionType}`);
```

Also in `StudySession.tsx` Task 5, read block from URL if pre-selected:
```typescript
const preBlock = params.get("block") as BlockType | null;
const [blockType, setBlockType] = useState<BlockType | null>(preBlock);
const [phase, setPhase] = useState<Phase>(preBlock ? "muse-pair" : "block-pick");
```

**Step 2: Typecheck + commit**

```bash
npx tsc --noEmit 2>&1 | head -20
git add client/src/pages/study/StudyProfile.tsx client/src/pages/study/StudySession.tsx
git commit -m "feat: StudyProfile navigates to unified session page"
```

---

### Task 7: Post-session summary — stress arc chart + explore CTA

**Files:**
- Modify: `client/src/pages/study/StudyComplete.tsx`

**Step 1: Add stress arc chart**

The checkpoint saves phase-tagged EEG to `eegFeaturesJson`. On the complete page, fetch the session data and show a simple Recharts line chart of stress over phases.

Add to `StudyComplete.tsx`:
```typescript
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

// After session completes, fetch session data:
const { data: sessionData } = useQuery({
  queryKey: ["/api/study/session/latest", participantCode, blockType],
  queryFn: async () => {
    const res = await fetch(`/api/study/admin/sessions`, { credentials: "include" });
    // Filter to latest session for this participant — or add a dedicated endpoint
    return res.json();
  },
  enabled: !!participantCode,
});

// Render:
const stressPoints = [
  { phase: "Baseline", stress: 0.3 },   // from preEegJson
  { phase: "Task",     stress: 0.72 },  // from eegFeaturesJson peak
  { phase: "Recovery", stress: 0.41 },  // from postEegJson
];

<ResponsiveContainer width="100%" height={120}>
  <LineChart data={stressPoints}>
    <XAxis dataKey="phase" tick={{ fontSize: 11 }} />
    <YAxis domain={[0, 1]} hide />
    <Tooltip />
    <Line type="monotone" dataKey="stress" stroke="#f59e0b" strokeWidth={2} dot />
  </LineChart>
</ResponsiveContainer>
```

**Step 2: Interpretation sentence**

```typescript
function interpretStress(pre: number, post: number): string {
  const pct = Math.round(Math.abs(pre - post) / pre * 100);
  if (post < pre) return `Your stress dropped ${pct}% after the intervention.`;
  if (post > pre) return `Stress increased ${pct}% — this happens, the data is still valuable.`;
  return "Stress stayed stable throughout the session.";
}
```

**Step 3: "Explore the full app" button when both sessions complete**

```typescript
const bothDone = completedSessions.includes("stress") && completedSessions.includes("food");

{bothDone && (
  <Button variant="outline" onClick={() => {
    apiRequest("PATCH", "/api/user/intent", { intent: "explore" });
    navigate("/");
  }}>
    Explore the full app →
  </Button>
)}
```

**Step 4: Typecheck + commit**

```bash
npx tsc --noEmit 2>&1 | head -20
git add client/src/pages/study/StudyComplete.tsx
git commit -m "feat: stress arc chart, interpretation, and explore CTA on study complete"
```

---

### Task 8: Admin dashboard — live refresh + status badges + sparklines

**Files:**
- Modify: `client/src/pages/study/StudyAdmin.tsx`

**Step 1: Auto-refresh every 60s**

```typescript
const { data: sessions, refetch: refetchSessions } = useQuery({ ... });

useEffect(() => {
  const t = setInterval(() => { refetchParticipants(); refetchSessions(); }, 60_000);
  return () => clearInterval(t);
}, []);
```

**Step 2: Session status badge**

```typescript
function sessionStatus(session: StudySession): "recording" | "complete" | "partial" {
  if (session.partial) return "partial";
  if (session.surveyJson) return "complete";
  return "recording";
}

const STATUS_COLOR = {
  recording: "border-blue-500/50 text-blue-400",
  complete:  "border-green-500/50 text-green-400",
  partial:   "border-amber-500/50 text-amber-400",
};

<Badge variant="outline" className={STATUS_COLOR[sessionStatus(s)]}>
  {sessionStatus(s)}
</Badge>
```

**Step 3: Inline stress sparkline per row**

```typescript
import { Sparklines, SparklinesLine } from "react-sparklines";
// OR use a simple inline bar: no extra dependency

// Simple inline stress bar (no extra package):
function StressBar({ preEeg, postEeg }: { preEeg: any; postEeg: any }) {
  const pre  = preEeg?.beta  ?? 0;
  const post = postEeg?.beta ?? 0;
  return (
    <div className="flex gap-1 items-center text-xs text-muted-foreground">
      <div className="w-12 h-1.5 bg-muted rounded-full overflow-hidden">
        <div className="h-full bg-amber-400 rounded-full" style={{ width: `${Math.min(100, pre * 200)}%` }} />
      </div>
      →
      <div className="w-12 h-1.5 bg-muted rounded-full overflow-hidden">
        <div className="h-full bg-green-400 rounded-full" style={{ width: `${Math.min(100, post * 200)}%` }} />
      </div>
    </div>
  );
}
```

**Step 4: Typecheck + commit**

```bash
npx tsc --noEmit 2>&1 | head -20
git add client/src/pages/study/StudyAdmin.tsx
git commit -m "feat: admin dashboard live refresh, status badges, stress sparklines"
```

---

### Task 9: Push to GitHub + verify Vercel deploy

**Step 1: Push**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git push
```

**Step 2: Verify Vercel**

```bash
gh api repos/LakshmiSravyaVedantham/DreamAnalysis/deployments \
  --jq '.[0] | {sha: .sha[0:8], environment, created_at}'
```

Expected: latest SHA matches HEAD, environment = "Production".

**Step 3: Manual smoke test on production**

1. Go to production URL → login → should see `/onboarding`
2. Click "Join the Study" → should reach `/study`
3. Consent → Profile → Session → pick stress block → Muse pair screen
4. (Without Muse) verify UI shows correctly at each step
5. Go to `/study/admin` (logged in) → verify sessions table loads

---

## Done Criteria

- [ ] Login routes to `/onboarding` for new users, intent-based for returning
- [ ] Study session DB record opens the moment Muse pairs (no manual start)
- [ ] EEG checkpoints auto-save every 30s
- [ ] Muse disconnect triggers reconnect overlay + auto-save
- [ ] Post-session summary shows stress arc + interpretation
- [ ] Both-complete screen has "Explore the full app" button
- [ ] Admin dashboard auto-refreshes with status badges
- [ ] All `npx tsc --noEmit` checks pass
- [ ] Deployed to Vercel from main branch
