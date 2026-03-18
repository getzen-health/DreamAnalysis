# Bevel-Grade Health Pipeline: Phase 0 + Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate NeuralDreamWorkshop from Neon to Supabase, then build the event-driven health data pipeline with adapter pattern, ingestion layer, aggregation triggers, and score engine skeleton.

**Architecture:** Full Supabase migration (DB + Auth + Storage + Edge Functions + Realtime). Pipeline uses PG triggers for fast aggregation, Database Webhooks to invoke score computation Edge Functions, and Supabase Realtime for live dashboard updates. Adapter interfaces (EventBus, ScoreCache, JobQueue) enable future Redis swap.

**Tech Stack:** Supabase (PostgreSQL, Auth, Edge Functions/Deno, Realtime, Storage), Drizzle ORM, React 18, TypeScript, Vercel Cron, Zod validation.

**Spec:** `docs/superpowers/specs/2026-03-17-bevel-pipeline-design.md`

---

## File Structure

### Files to Modify
| File | Change |
|------|--------|
| `server/db.ts` | Replace Neon Pool with Supabase postgres.js connection |
| `shared/schema.ts` | Add 15 new tables, update user_id types to uuid |
| `package.json` | Remove `@neondatabase/serverless`, add `@supabase/supabase-js`, `@supabase/ssr` |
| `server/routes.ts` | Replace bcrypt auth with Supabase Auth, remove express-session |
| `client/src/lib/health-sync.ts` | Update POST target from ML backend to Supabase Edge Function |
| `vercel.json` | Add `crons` config for daily trend detection |
| `.env` | Add `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY` |
| `.env.example` | Add Supabase env var templates |

### Files to Create
| File | Purpose |
|------|---------|
| `lib/supabase.ts` | Shared Supabase client (server + client) |
| `lib/supabase-admin.ts` | Service-role Supabase client (server-side only) |
| `client/src/lib/supabase-browser.ts` | Browser-side Supabase client with auth |
| `client/src/hooks/use-auth.ts` | Supabase Auth hook (replaces custom auth) |
| `lib/adapters/event-bus.ts` | EventBus interface + PgNotifyEventBus |
| `lib/adapters/cache.ts` | ScoreCache interface + PgTableCache |
| `lib/adapters/job-queue.ts` | JobQueue interface + DirectCallQueue |
| `supabase/migrations/001_new_tables.sql` | New pipeline tables (health_samples enhanced, daily_aggregates, user_baselines, user_scores, score_history, trend_alerts, device_connections, body_metrics, exercises, workouts, workout_sets, workout_templates, exercise_history, habits, habit_logs, cycle_tracking, mood_logs) |
| `supabase/migrations/002_triggers.sql` | PG triggers for aggregation on health_samples INSERT |
| `supabase/migrations/003_rls.sql` | Row-Level Security policies for all tables |
| `supabase/functions/ingest-health-data/index.ts` | Edge Function: validate, normalize, dedup, INSERT |
| `supabase/functions/compute-scores/index.ts` | Edge Function: read aggregates, compute 6 scores, UPSERT |
| `supabase/functions/daily-trends/index.ts` | Edge Function: trend detection, anomaly alerting |
| `tests/adapters/event-bus.test.ts` | Tests for EventBus adapter |
| `tests/adapters/cache.test.ts` | Tests for ScoreCache adapter |
| `tests/pipeline/ingest.test.ts` | Tests for ingestion validation/normalization |
| `tests/pipeline/scores.test.ts` | Tests for score computation logic |

---

## Chunk 1: Supabase Project Setup + Database Migration

### Task 1: Create Supabase Project and Configure Environment

**Files:**
- Modify: `.env`
- Modify: `.env.example`
- Create: `lib/supabase.ts`
- Create: `lib/supabase-admin.ts`

- [ ] **Step 1: Create Supabase project**

Go to https://supabase.com/dashboard → New Project. Save:
- Project URL → `SUPABASE_URL`
- Anon public key → `SUPABASE_ANON_KEY`
- Service role key → `SUPABASE_SERVICE_ROLE_KEY`
- Database password → `SUPABASE_DB_PASSWORD`
- Connection string → `DATABASE_URL` (replace Neon URL)

- [ ] **Step 2: Update .env with Supabase credentials**

Add to `.env`:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...
SUPABASE_DB_PASSWORD=your-db-password
DATABASE_URL=postgresql://postgres.[project-ref]:[password]@aws-0-us-east-1.pooler.supabase.com:6543/postgres
```

- [ ] **Step 3: Update .env.example**

Add template entries (no real values):
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
DATABASE_URL=postgresql://postgres.[ref]:[password]@aws-0-us-east-1.pooler.supabase.com:6543/postgres
```

- [ ] **Step 4: Install Supabase packages**

Run: `npm install @supabase/supabase-js @supabase/ssr`
Run: `npm uninstall @neondatabase/serverless`

- [ ] **Step 5: Create lib/supabase.ts (shared client)**

```typescript
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.SUPABASE_URL!
const supabaseAnonKey = process.env.SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnonKey)
```

- [ ] **Step 6: Create lib/supabase-admin.ts (service role)**

```typescript
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

export const supabaseAdmin = createClient(supabaseUrl, supabaseServiceKey)
```

- [ ] **Step 7: Commit**

```bash
git add .env.example lib/supabase.ts lib/supabase-admin.ts package.json package-lock.json
git commit -m "chore: add Supabase client setup, remove Neon dependency"
```

---

### Task 2: Migrate Database Connection (Drizzle ORM)

**Files:**
- Modify: `server/db.ts`
- Modify: `package.json`

- [ ] **Step 1: Install Drizzle Supabase adapter**

Run: `npm install postgres`
(Drizzle supports `postgres` (postgres.js) which works with Supabase's connection pooler)

- [ ] **Step 2: Replace Neon connection in server/db.ts**

Replace the entire file:
```typescript
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "@shared/schema";

const connectionString = process.env.DATABASE_URL!;
const client = postgres(connectionString, { prepare: false });
export const db = drizzle(client, { schema });
```

Note: `prepare: false` is required for Supabase's connection pooler (pgBouncer in transaction mode).

- [ ] **Step 3: Verify the app starts without errors**

Run: `npm run dev`
Expected: App starts on port 4000, no database connection errors.

- [ ] **Step 4: Commit**

```bash
git add server/db.ts package.json package-lock.json
git commit -m "feat: migrate database connection from Neon to Supabase"
```

---

### Task 3: Export Existing Schema to Supabase

**Files:**
- Modify: `shared/schema.ts` (minimal — just verify compatibility)

- [ ] **Step 1: Push existing Drizzle schema to Supabase**

Run: `npx drizzle-kit push`
Expected: All 23 tables created in Supabase PostgreSQL. Watch for any errors related to type mismatches.

- [ ] **Step 2: Verify tables exist in Supabase Dashboard**

Go to Supabase Dashboard → Table Editor. Confirm all 23 tables are present:
users, health_metrics, dream_analysis, dream_symbols, emotion_readings, ai_chats, user_settings, eeg_sessions, push_subscriptions, brain_readings, health_samples, datadog_error_log, study_participants, study_sessions, study_morning_entries, study_daytime_entries, study_evening_entries, rate_limit_entries, password_reset_tokens, food_logs, meal_history, user_readings, pilot_participants, pilot_sessions.

- [ ] **Step 3: Commit (if any schema.ts changes were needed)**

```bash
git add shared/schema.ts
git commit -m "chore: verify schema compatibility with Supabase PostgreSQL"
```

---

### Task 4: Migrate Auth to Supabase Auth

**Files:**
- Create: `client/src/lib/supabase-browser.ts`
- Create: `client/src/hooks/use-auth.ts`
- Modify: `server/routes.ts` (auth routes section)
- Modify: `client/src/App.tsx` (auth provider wrapping)

- [ ] **Step 1: Create browser-side Supabase client**

Create `client/src/lib/supabase-browser.ts`:
```typescript
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

export const supabase = createClient(supabaseUrl, supabaseAnonKey)
```

- [ ] **Step 2: Add VITE_ env vars to .env**

```
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=eyJ...
```

- [ ] **Step 3: Create use-auth.ts hook**

Create `client/src/hooks/use-auth.ts`:
```typescript
import { useState, useEffect } from 'react'
import { supabase } from '@/lib/supabase-browser'
import type { User, Session } from '@supabase/supabase-js'

export function useAuth() {
  const [user, setUser] = useState<User | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session)
      setUser(session?.user ?? null)
      setLoading(false)
    })

    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setSession(session)
        setUser(session?.user ?? null)
      }
    )

    return () => subscription.unsubscribe()
  }, [])

  const signUp = async (email: string, password: string) => {
    const { data, error } = await supabase.auth.signUp({ email, password })
    if (error) throw error
    return data
  }

  const signIn = async (email: string, password: string) => {
    const { data, error } = await supabase.auth.signInWithPassword({ email, password })
    if (error) throw error
    return data
  }

  const signOut = async () => {
    const { error } = await supabase.auth.signOut()
    if (error) throw error
  }

  return { user, session, loading, signUp, signIn, signOut }
}
```

- [ ] **Step 4: Update Express auth routes to use Supabase Auth verification**

In `server/routes.ts`, replace the JWT/bcrypt auth middleware with Supabase token verification. The Express server now validates Supabase JWTs instead of managing its own sessions:

```typescript
import { supabaseAdmin } from '../lib/supabase-admin'

async function requireAuth(req: Request, res: Response, next: NextFunction) {
  const token = req.headers.authorization?.replace('Bearer ', '')
  if (!token) return res.status(401).json({ error: 'No token provided' })

  const { data: { user }, error } = await supabaseAdmin.auth.getUser(token)
  if (error || !user) return res.status(401).json({ error: 'Invalid token' })

  req.user = user
  next()
}
```

- [ ] **Step 5: Test auth flow end-to-end**

1. Start app: `npm run dev`
2. Try registering a new user
3. Try logging in
4. Try accessing a protected route
Expected: All auth operations work via Supabase Auth.

- [ ] **Step 6: Commit**

```bash
git add client/src/lib/supabase-browser.ts client/src/hooks/use-auth.ts server/routes.ts .env.example
git commit -m "feat: migrate auth from bcrypt/JWT to Supabase Auth"
```

---

## Chunk 2: New Pipeline Tables + Triggers

### Task 5: Create New Pipeline Tables via Drizzle Schema

**Files:**
- Modify: `shared/schema.ts`

- [ ] **Step 1: Add pipeline tables to schema.ts**

Append to `shared/schema.ts`:

```typescript
// ============ PIPELINE TABLES ============

export const dailyAggregates = pgTable("daily_aggregates", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  date: date("date").notNull(),
  metric: text("metric").notNull(),
  avgValue: numeric("avg_value"),
  minValue: numeric("min_value"),
  maxValue: numeric("max_value"),
  sumValue: numeric("sum_value"),
  sampleCount: integer("sample_count").default(0),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => [
  uniqueIndex("uq_daily_agg").on(table.userId, table.date, table.metric),
]);

export const userBaselines = pgTable("user_baselines", {
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  metric: text("metric").notNull(),
  baselineAvg: numeric("baseline_avg"),
  baselineStddev: numeric("baseline_stddev"),
  sampleCount: integer("sample_count").default(0),
  lastUpdated: timestamp("last_updated").defaultNow(),
}, (table) => [
  primaryKey({ columns: [table.userId, table.metric] }),
]);

export const userScores = pgTable("user_scores", {
  userId: uuid("user_id").primaryKey().references(() => users.id, { onDelete: "cascade" }),
  recoveryScore: numeric("recovery_score"),
  sleepScore: numeric("sleep_score"),
  strainScore: numeric("strain_score"),
  stressScore: numeric("stress_score"),
  nutritionScore: numeric("nutrition_score"),
  energyBank: numeric("energy_bank"),
  recoveryInputs: jsonb("recovery_inputs"),
  sleepInputs: jsonb("sleep_inputs"),
  strainInputs: jsonb("strain_inputs"),
  stressInputs: jsonb("stress_inputs"),
  nutritionInputs: jsonb("nutrition_inputs"),
  computedAt: timestamp("computed_at").defaultNow(),
});

export const scoreHistory = pgTable("score_history", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  date: date("date").notNull(),
  recoveryScore: numeric("recovery_score"),
  sleepScore: numeric("sleep_score"),
  strainScore: numeric("strain_score"),
  stressScore: numeric("stress_score"),
  nutritionScore: numeric("nutrition_score"),
  energyBank: numeric("energy_bank"),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => [
  uniqueIndex("uq_score_history").on(table.userId, table.date),
]);

export const trendAlerts = pgTable("trend_alerts", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  alertType: text("alert_type").notNull(),
  metric: text("metric").notNull(),
  severity: text("severity").notNull(),
  message: text("message").notNull(),
  value: numeric("value"),
  baseline: numeric("baseline"),
  acknowledged: boolean("acknowledged").default(false),
  createdAt: timestamp("created_at").defaultNow(),
});

export const deviceConnections = pgTable("device_connections", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  provider: text("provider").notNull(),
  accessToken: text("access_token").notNull(),
  refreshToken: text("refresh_token"),
  tokenExpiresAt: timestamp("token_expires_at"),
  scopes: text("scopes").array(),
  lastSyncAt: timestamp("last_sync_at"),
  syncStatus: text("sync_status").default("active"),
  errorMessage: text("error_message"),
  connectedAt: timestamp("connected_at").defaultNow(),
}, (table) => [
  uniqueIndex("uq_device_conn").on(table.userId, table.provider),
]);

// ============ EXERCISE & BODY TABLES ============

export const exercises = pgTable("exercises", {
  id: uuid("id").defaultRandom().primaryKey(),
  name: text("name").notNull(),
  category: text("category").notNull(),
  muscleGroups: text("muscle_groups").array().notNull(),
  equipment: text("equipment"),
  instructions: text("instructions"),
  videoUrl: text("video_url"),
  isCustom: boolean("is_custom").default(false),
  createdBy: uuid("created_by").references(() => users.id),
});

export const workouts = pgTable("workouts", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  name: text("name"),
  workoutType: text("workout_type").notNull(),
  startedAt: timestamp("started_at").notNull(),
  endedAt: timestamp("ended_at"),
  durationMin: numeric("duration_min"),
  totalStrain: numeric("total_strain"),
  avgHr: numeric("avg_hr"),
  maxHr: numeric("max_hr"),
  caloriesBurned: numeric("calories_burned"),
  hrZones: jsonb("hr_zones"),
  hrRecovery: numeric("hr_recovery"),
  source: text("source").notNull(),
  eegSessionId: text("eeg_session_id"),
  notes: text("notes"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const workoutSets = pgTable("workout_sets", {
  id: uuid("id").defaultRandom().primaryKey(),
  workoutId: uuid("workout_id").references(() => workouts.id, { onDelete: "cascade" }),
  exerciseId: uuid("exercise_id").references(() => exercises.id),
  setNumber: integer("set_number").notNull(),
  setType: text("set_type").default("normal"),
  reps: integer("reps"),
  weightKg: numeric("weight_kg"),
  durationSec: integer("duration_sec"),
  restSec: integer("rest_sec"),
  rpe: numeric("rpe"),
  completed: boolean("completed").default(true),
  createdAt: timestamp("created_at").defaultNow(),
});

export const workoutTemplates = pgTable("workout_templates", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  name: text("name").notNull(),
  description: text("description"),
  exercises: jsonb("exercises").notNull(),
  isAiGenerated: boolean("is_ai_generated").default(false),
  timesUsed: integer("times_used").default(0),
  createdAt: timestamp("created_at").defaultNow(),
});

export const bodyMetrics = pgTable("body_metrics", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  weightKg: numeric("weight_kg"),
  bodyFatPct: numeric("body_fat_pct"),
  leanMassKg: numeric("lean_mass_kg"),
  bmi: numeric("bmi"),
  heightCm: numeric("height_cm"),
  source: text("source").notNull(),
  recordedAt: timestamp("recorded_at").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const exerciseHistory = pgTable("exercise_history", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  exerciseId: uuid("exercise_id").references(() => exercises.id),
  date: date("date").notNull(),
  bestWeightKg: numeric("best_weight_kg"),
  bestReps: integer("best_reps"),
  estimated1rm: numeric("estimated_1rm"),
  totalVolume: numeric("total_volume"),
}, (table) => [
  uniqueIndex("uq_exercise_history").on(table.userId, table.exerciseId, table.date),
]);

// ============ LIFESTYLE TABLES ============

export const habits = pgTable("habits", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  name: text("name").notNull(),
  category: text("category"),
  icon: text("icon"),
  targetValue: numeric("target_value"),
  unit: text("unit"),
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at").defaultNow(),
});

export const habitLogs = pgTable("habit_logs", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  habitId: uuid("habit_id").references(() => habits.id, { onDelete: "cascade" }),
  value: numeric("value").notNull(),
  note: text("note"),
  loggedAt: timestamp("logged_at").defaultNow(),
});

export const cycleTracking = pgTable("cycle_tracking", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  date: date("date").notNull(),
  flowLevel: text("flow_level"),
  symptoms: text("symptoms").array(),
  phase: text("phase"),
  contraception: text("contraception"),
  basalTemp: numeric("basal_temp"),
  notes: text("notes"),
}, (table) => [
  uniqueIndex("uq_cycle").on(table.userId, table.date),
]);

export const moodLogs = pgTable("mood_logs", {
  id: uuid("id").defaultRandom().primaryKey(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  moodScore: numeric("mood_score").notNull(),
  energyLevel: numeric("energy_level"),
  notes: text("notes"),
  loggedAt: timestamp("logged_at").defaultNow(),
});
```

- [ ] **Step 2: Push new tables to Supabase**

Run: `npx drizzle-kit push`
Expected: 15 new tables created. No errors.

- [ ] **Step 3: Verify in Supabase Dashboard**

Check Table Editor for: daily_aggregates, user_baselines, user_scores, score_history, trend_alerts, device_connections, exercises, workouts, workout_sets, workout_templates, body_metrics, exercise_history, habits, habit_logs, cycle_tracking, mood_logs.

- [ ] **Step 4: Commit**

```bash
git add shared/schema.ts
git commit -m "feat: add 16 new pipeline + exercise + lifestyle tables"
```

---

### Task 6: Create PG Triggers for Aggregation

**Files:**
- Create: `supabase/migrations/002_triggers.sql`

- [ ] **Step 1: Write aggregation trigger SQL**

Create `supabase/migrations/002_triggers.sql`:

```sql
-- Trigger: On health_samples INSERT, update daily_aggregates
CREATE OR REPLACE FUNCTION update_daily_aggregates()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO daily_aggregates (id, user_id, date, metric, avg_value, min_value, max_value, sum_value, sample_count, updated_at)
  VALUES (
    gen_random_uuid(),
    NEW.user_id,
    DATE(NEW.recorded_at),
    NEW.metric,
    NEW.value,
    NEW.value,
    NEW.value,
    NEW.value,
    1,
    NOW()
  )
  ON CONFLICT (user_id, date, metric) DO UPDATE SET
    avg_value = (daily_aggregates.avg_value * daily_aggregates.sample_count + NEW.value) / (daily_aggregates.sample_count + 1),
    min_value = LEAST(daily_aggregates.min_value, NEW.value),
    max_value = GREATEST(daily_aggregates.max_value, NEW.value),
    sum_value = daily_aggregates.sum_value + NEW.value,
    sample_count = daily_aggregates.sample_count + 1,
    updated_at = NOW();

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_health_samples_aggregate
  AFTER INSERT ON health_samples
  FOR EACH ROW
  EXECUTE FUNCTION update_daily_aggregates();

-- Trigger: On health_samples INSERT, update user_baselines (14-day rolling)
CREATE OR REPLACE FUNCTION update_user_baselines()
RETURNS TRIGGER AS $$
DECLARE
  _avg numeric;
  _stddev numeric;
  _count integer;
BEGIN
  SELECT AVG(avg_value), STDDEV(avg_value), COUNT(*)
  INTO _avg, _stddev, _count
  FROM daily_aggregates
  WHERE user_id = NEW.user_id
    AND metric = NEW.metric
    AND date >= CURRENT_DATE - INTERVAL '14 days';

  INSERT INTO user_baselines (user_id, metric, baseline_avg, baseline_stddev, sample_count, last_updated)
  VALUES (NEW.user_id, NEW.metric, _avg, COALESCE(_stddev, 0), _count, NOW())
  ON CONFLICT (user_id, metric) DO UPDATE SET
    baseline_avg = _avg,
    baseline_stddev = COALESCE(_stddev, 0),
    sample_count = _count,
    last_updated = NOW();

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_health_samples_baselines
  AFTER INSERT ON health_samples
  FOR EACH ROW
  EXECUTE FUNCTION update_user_baselines();

-- Simple threshold alerts
CREATE OR REPLACE FUNCTION check_health_thresholds()
RETURNS TRIGGER AS $$
BEGIN
  -- Heart rate too high
  IF NEW.metric = 'heart_rate' AND NEW.value > 120 THEN
    INSERT INTO trend_alerts (id, user_id, alert_type, metric, severity, message, value, created_at)
    VALUES (gen_random_uuid(), NEW.user_id, 'threshold', 'heart_rate', 'warning',
            'Heart rate above 120 bpm: ' || NEW.value || ' bpm', NEW.value, NOW());
  END IF;

  -- SpO2 too low
  IF NEW.metric = 'spo2' AND NEW.value < 94 THEN
    INSERT INTO trend_alerts (id, user_id, alert_type, metric, severity, message, value, created_at)
    VALUES (gen_random_uuid(), NEW.user_id, 'threshold', 'spo2', 'critical',
            'Blood oxygen below 94%: ' || NEW.value || '%', NEW.value, NOW());
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_health_thresholds
  AFTER INSERT ON health_samples
  FOR EACH ROW
  EXECUTE FUNCTION check_health_thresholds();
```

- [ ] **Step 2: Apply migration to Supabase**

Run via Supabase Dashboard → SQL Editor, paste and execute. Or:
Run: `psql "$DATABASE_URL" -f supabase/migrations/002_triggers.sql`

- [ ] **Step 3: Test trigger by inserting a sample row**

Via Supabase SQL Editor:
```sql
INSERT INTO health_samples (id, user_id, source, metric, value, unit, recorded_at)
VALUES (gen_random_uuid(), '<your-user-id>', 'manual', 'heart_rate', 75, 'bpm', NOW());

-- Verify daily_aggregates was created:
SELECT * FROM daily_aggregates WHERE metric = 'heart_rate';

-- Verify user_baselines was created:
SELECT * FROM user_baselines WHERE metric = 'heart_rate';
```

Expected: Both tables have entries for the inserted metric.

- [ ] **Step 4: Commit**

```bash
git add supabase/migrations/002_triggers.sql
git commit -m "feat: add PG triggers for health data aggregation and threshold alerts"
```

---

### Task 7: Create RLS Policies

**Files:**
- Create: `supabase/migrations/003_rls.sql`

- [ ] **Step 1: Write RLS policies**

Create `supabase/migrations/003_rls.sql`:

```sql
-- Enable RLS on all user-owned tables
ALTER TABLE health_samples ENABLE ROW LEVEL SECURITY;
ALTER TABLE daily_aggregates ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_baselines ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE score_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE trend_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE device_connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE exercises ENABLE ROW LEVEL SECURITY;
ALTER TABLE workouts ENABLE ROW LEVEL SECURITY;
ALTER TABLE workout_sets ENABLE ROW LEVEL SECURITY;
ALTER TABLE workout_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE body_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE exercise_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE habits ENABLE ROW LEVEL SECURITY;
ALTER TABLE habit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE cycle_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE mood_logs ENABLE ROW LEVEL SECURITY;

-- Default policy: users access only their own data
CREATE POLICY "own_data" ON health_samples FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "own_data" ON daily_aggregates FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "own_data" ON user_baselines FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "own_data" ON user_scores FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "own_data" ON score_history FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "own_data" ON trend_alerts FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "own_data" ON device_connections FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "own_data" ON workouts FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "own_data" ON workout_sets FOR ALL USING (auth.uid() = (SELECT user_id FROM workouts WHERE id = workout_id));
CREATE POLICY "own_data" ON workout_templates FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "own_data" ON body_metrics FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "own_data" ON exercise_history FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "own_data" ON habits FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "own_data" ON habit_logs FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "own_data" ON cycle_tracking FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "own_data" ON mood_logs FOR ALL USING (auth.uid() = user_id);

-- Exercises: shared library readable by all, custom exercises owned
CREATE POLICY "read_all" ON exercises FOR SELECT USING (true);
CREATE POLICY "manage_custom" ON exercises FOR INSERT WITH CHECK (is_custom = true AND created_by = auth.uid());
CREATE POLICY "update_custom" ON exercises FOR UPDATE USING (is_custom = true AND created_by = auth.uid());
CREATE POLICY "delete_custom" ON exercises FOR DELETE USING (is_custom = true AND created_by = auth.uid());

-- Enable Realtime on user_scores
ALTER PUBLICATION supabase_realtime ADD TABLE user_scores;
```

- [ ] **Step 2: Apply via Supabase SQL Editor**

Run: `psql "$DATABASE_URL" -f supabase/migrations/003_rls.sql`

- [ ] **Step 3: Commit**

```bash
git add supabase/migrations/003_rls.sql
git commit -m "feat: add Row-Level Security policies for all pipeline tables"
```

---

## Chunk 3: Adapter Interfaces + Ingestion Edge Function

### Task 8: Create Pipeline Adapter Interfaces

**Files:**
- Create: `lib/adapters/event-bus.ts`
- Create: `lib/adapters/cache.ts`
- Create: `lib/adapters/job-queue.ts`

- [ ] **Step 1: Create EventBus adapter**

Create `lib/adapters/event-bus.ts`:
```typescript
export interface EventBus {
  publish(channel: string, data: unknown): Promise<void>
  subscribe(channel: string, handler: (data: unknown) => void): void
}

// PostgreSQL NOTIFY-based implementation
export class PgNotifyEventBus implements EventBus {
  constructor(private db: any) {}

  async publish(channel: string, data: unknown): Promise<void> {
    await this.db.execute(
      `SELECT pg_notify($1, $2)`,
      [channel, JSON.stringify(data)]
    )
  }

  subscribe(channel: string, handler: (data: unknown) => void): void {
    // In PG implementation, subscription happens via Supabase Realtime
    // or Database Webhooks — not direct LISTEN (Edge Functions are stateless)
    console.log(`[PgNotifyEventBus] Subscribe to ${channel} — use Supabase Realtime or Database Webhooks`)
  }
}

// Future: Redis Streams implementation
// export class RedisStreamEventBus implements EventBus { ... }
```

- [ ] **Step 2: Create ScoreCache adapter**

Create `lib/adapters/cache.ts`:
```typescript
export interface UserScores {
  recoveryScore: number | null
  sleepScore: number | null
  strainScore: number | null
  stressScore: number | null
  nutritionScore: number | null
  energyBank: number | null
  computedAt: string | null
}

export interface ScoreCache {
  get(userId: string): Promise<UserScores | null>
  set(userId: string, scores: UserScores): Promise<void>
  invalidate(userId: string): Promise<void>
}

// PostgreSQL table-based cache (reads user_scores table directly)
export class PgTableCache implements ScoreCache {
  constructor(private supabase: any) {}

  async get(userId: string): Promise<UserScores | null> {
    const { data, error } = await this.supabase
      .from('user_scores')
      .select('*')
      .eq('user_id', userId)
      .single()
    if (error || !data) return null
    return {
      recoveryScore: data.recovery_score,
      sleepScore: data.sleep_score,
      strainScore: data.strain_score,
      stressScore: data.stress_score,
      nutritionScore: data.nutrition_score,
      energyBank: data.energy_bank,
      computedAt: data.computed_at,
    }
  }

  async set(userId: string, scores: UserScores): Promise<void> {
    await this.supabase.from('user_scores').upsert({
      user_id: userId,
      recovery_score: scores.recoveryScore,
      sleep_score: scores.sleepScore,
      strain_score: scores.strainScore,
      stress_score: scores.stressScore,
      nutrition_score: scores.nutritionScore,
      energy_bank: scores.energyBank,
      computed_at: new Date().toISOString(),
    })
  }

  async invalidate(userId: string): Promise<void> {
    // No-op for PG cache — data is always fresh from table
  }
}

// Future: Redis implementation
// export class RedisCache implements ScoreCache { ... }
```

- [ ] **Step 3: Create JobQueue adapter**

Create `lib/adapters/job-queue.ts`:
```typescript
export interface JobOpts {
  delay?: number
  priority?: number
  retries?: number
}

export interface JobQueue {
  enqueue(job: string, payload: unknown, opts?: JobOpts): Promise<void>
  process(job: string, handler: (payload: unknown) => Promise<void>): void
}

// Direct Edge Function invocation (synchronous call)
export class DirectCallQueue implements JobQueue {
  constructor(private supabaseUrl: string, private serviceKey: string) {}

  async enqueue(job: string, payload: unknown, _opts?: JobOpts): Promise<void> {
    const response = await fetch(
      `${this.supabaseUrl}/functions/v1/${job}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.serviceKey}`,
        },
        body: JSON.stringify(payload),
      }
    )
    if (!response.ok) {
      console.error(`[DirectCallQueue] ${job} failed:`, await response.text())
    }
  }

  process(_job: string, _handler: (payload: unknown) => Promise<void>): void {
    // No-op for direct call — Edge Functions handle their own processing
  }
}

// Future: BullMQ implementation
// export class BullMQQueue implements JobQueue { ... }
```

- [ ] **Step 4: Commit**

```bash
git add lib/adapters/
git commit -m "feat: add pipeline adapter interfaces (EventBus, ScoreCache, JobQueue)"
```

---

### Task 9: Create Ingestion Edge Function

**Files:**
- Create: `supabase/functions/ingest-health-data/index.ts`

- [ ] **Step 1: Initialize Supabase Edge Functions locally**

Run: `npx supabase init` (if not already done)
Run: `npx supabase functions new ingest-health-data`

- [ ] **Step 2: Write the ingestion Edge Function**

Create `supabase/functions/ingest-health-data/index.ts`:
```typescript
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { serve } from 'https://deno.land/std@0.177.0/http/server.ts'

interface HealthSample {
  source: string
  metric: string
  value: number
  unit: string
  recorded_at: string
  metadata?: Record<string, unknown>
}

interface IngestRequest {
  user_id: string
  samples: HealthSample[]
}

const VALID_SOURCES = ['apple_health', 'google_fit', 'oura', 'whoop', 'garmin', 'cgm', 'eeg', 'manual']
const VALID_METRICS = [
  'heart_rate', 'hrv_rmssd', 'resting_hr', 'respiratory_rate', 'spo2',
  'skin_temp', 'sleep_deep_min', 'sleep_rem_min', 'sleep_light_min',
  'sleep_awake_min', 'sleep_efficiency', 'steps', 'active_calories',
  'weight_kg', 'body_fat_pct', 'lean_mass_kg', 'vo2_max',
  'workout_strain', 'glucose_mgdl', 'basal_calories', 'exercise_minutes',
]

serve(async (req: Request) => {
  if (req.method !== 'POST') {
    return new Response('Method not allowed', { status: 405 })
  }

  const authHeader = req.headers.get('Authorization')
  if (!authHeader) {
    return new Response('Unauthorized', { status: 401 })
  }

  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,
  )

  const body: IngestRequest = await req.json()

  // Validate
  if (!body.user_id || !body.samples?.length) {
    return new Response(JSON.stringify({ error: 'user_id and samples required' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  // Normalize and validate each sample
  const validSamples = body.samples.filter(s => {
    if (!VALID_SOURCES.includes(s.source)) return false
    if (!VALID_METRICS.includes(s.metric)) return false
    if (typeof s.value !== 'number' || isNaN(s.value)) return false
    if (!s.recorded_at) return false
    return true
  }).map(s => ({
    user_id: body.user_id,
    source: s.source,
    metric: s.metric,
    value: s.value,
    unit: s.unit || 'unknown',
    metadata: s.metadata || null,
    recorded_at: s.recorded_at,
    ingested_at: new Date().toISOString(),
  }))

  if (validSamples.length === 0) {
    return new Response(JSON.stringify({ error: 'No valid samples', accepted: 0 }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  // Idempotent insert (ON CONFLICT DO NOTHING)
  const { data, error } = await supabase
    .from('health_samples')
    .upsert(validSamples, {
      onConflict: 'user_id,source,metric,recorded_at',
      ignoreDuplicates: true,
    })

  if (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  // PG triggers handle aggregation automatically on INSERT.
  // Call compute-scores via direct invocation (DirectCallQueue pattern).
  const scoreResponse = await fetch(
    `${Deno.env.get('SUPABASE_URL')}/functions/v1/compute-scores`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')}`,
      },
      body: JSON.stringify({
        user_id: body.user_id,
        metrics_changed: [...new Set(validSamples.map(s => s.metric))],
      }),
    }
  )

  return new Response(JSON.stringify({
    accepted: validSamples.length,
    rejected: body.samples.length - validSamples.length,
    scores_updated: scoreResponse.ok,
  }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
})
```

- [ ] **Step 3: Test locally**

Run: `npx supabase functions serve ingest-health-data`
Test: `curl -X POST http://localhost:54321/functions/v1/ingest-health-data -H "Authorization: Bearer $SUPABASE_ANON_KEY" -H "Content-Type: application/json" -d '{"user_id":"test-uuid","samples":[{"source":"manual","metric":"heart_rate","value":72,"unit":"bpm","recorded_at":"2026-03-17T12:00:00Z"}]}'`

Expected: `{"accepted":1,"rejected":0,"scores_updated":true}`

- [ ] **Step 4: Commit**

```bash
git add supabase/functions/ingest-health-data/
git commit -m "feat: add ingest-health-data Edge Function with validation and dedup"
```

---

### Task 10: Create Score Computation Edge Function (Skeleton)

**Files:**
- Create: `supabase/functions/compute-scores/index.ts`

- [ ] **Step 1: Write the compute-scores skeleton**

Create `supabase/functions/compute-scores/index.ts`:
```typescript
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { serve } from 'https://deno.land/std@0.177.0/http/server.ts'

interface ScoreRequest {
  user_id: string
  metrics_changed: string[]
}

// Score engine stubs — will be fully implemented in Phase 4
function computeRecoveryScore(aggregates: any[], baselines: any[]): number | null {
  // TODO: Phase 4 — HRV, RHR, sleep, temp, resp, SpO2 vs 14-day baseline
  return null
}

function computeSleepScore(aggregates: any[], baselines: any[]): number | null {
  // TODO: Phase 4 — time asleep, stages, HR dip, efficiency
  return null
}

function computeStrainScore(aggregates: any[], baselines: any[]): number | null {
  // TODO: Phase 4 — TRIMP, HR zones, active + passive
  return null
}

function computeStressScore(aggregates: any[], baselines: any[]): number | null {
  // TODO: Phase 4 — HRV trend, HR elevation, resp, temp, EEG
  return null
}

function computeNutritionScore(aggregates: any[], baselines: any[]): number | null {
  // TODO: Phase 4 — AHEI food quality + glucose
  return null
}

function computeEnergyBank(scores: Record<string, number | null>): number | null {
  // TODO: Phase 4 — composite of recovery, sleep, strain, stress, nutrition
  return null
}

serve(async (req: Request) => {
  if (req.method !== 'POST') {
    return new Response('Method not allowed', { status: 405 })
  }

  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,
  )

  const body: ScoreRequest = await req.json()
  if (!body.user_id) {
    return new Response(JSON.stringify({ error: 'user_id required' }), { status: 400 })
  }

  // Read 14-day aggregates
  const { data: aggregates } = await supabase
    .from('daily_aggregates')
    .select('*')
    .eq('user_id', body.user_id)
    .gte('date', new Date(Date.now() - 14 * 86400000).toISOString().split('T')[0])
    .order('date', { ascending: false })

  // Read baselines
  const { data: baselines } = await supabase
    .from('user_baselines')
    .select('*')
    .eq('user_id', body.user_id)

  // Compute all scores
  const scores = {
    recovery_score: computeRecoveryScore(aggregates || [], baselines || []),
    sleep_score: computeSleepScore(aggregates || [], baselines || []),
    strain_score: computeStrainScore(aggregates || [], baselines || []),
    stress_score: computeStressScore(aggregates || [], baselines || []),
    nutrition_score: computeNutritionScore(aggregates || [], baselines || []),
    energy_bank: computeEnergyBank({
      recovery: null, sleep: null, strain: null, stress: null, nutrition: null,
    }),
    computed_at: new Date().toISOString(),
  }

  // Upsert user_scores (triggers Supabase Realtime push to dashboard)
  const { error } = await supabase.from('user_scores').upsert({
    user_id: body.user_id,
    ...scores,
  })

  if (error) {
    return new Response(JSON.stringify({ error: error.message }), { status: 500 })
  }

  // Record in score_history (one row per day)
  await supabase.from('score_history').upsert({
    user_id: body.user_id,
    date: new Date().toISOString().split('T')[0],
    ...scores,
  }, { onConflict: 'user_id,date' })

  return new Response(JSON.stringify({ scores, status: 'skeleton' }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
})
```

- [ ] **Step 2: Commit**

```bash
git add supabase/functions/compute-scores/
git commit -m "feat: add compute-scores Edge Function skeleton (score engines pending Phase 4)"
```

---

### Task 11: Create Daily Trends Edge Function + Vercel Cron

**Files:**
- Create: `supabase/functions/daily-trends/index.ts`
- Modify: `vercel.json`

- [ ] **Step 1: Write daily-trends Edge Function**

Create `supabase/functions/daily-trends/index.ts`:
```typescript
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { serve } from 'https://deno.land/std@0.177.0/http/server.ts'

serve(async (req: Request) => {
  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,
  )

  // Get all users with recent health data
  const { data: activeUsers } = await supabase
    .from('daily_aggregates')
    .select('user_id')
    .gte('date', new Date(Date.now() - 7 * 86400000).toISOString().split('T')[0])
    .limit(1000)

  const userIds = [...new Set((activeUsers || []).map(u => u.user_id))]
  const alerts: any[] = []

  for (const userId of userIds) {
    // Weight trend: >2% change in 7 days
    const { data: weightData } = await supabase
      .from('body_metrics')
      .select('weight_kg, recorded_at')
      .eq('user_id', userId)
      .order('recorded_at', { ascending: false })
      .limit(14)

    if (weightData && weightData.length >= 2) {
      const latest = Number(weightData[0].weight_kg)
      const weekAgo = weightData.find(w =>
        new Date(w.recorded_at) <= new Date(Date.now() - 6 * 86400000)
      )
      if (weekAgo) {
        const change = Math.abs((latest - Number(weekAgo.weight_kg)) / Number(weekAgo.weight_kg))
        if (change > 0.02) {
          alerts.push({
            user_id: userId,
            alert_type: 'rapid_change',
            metric: 'weight',
            severity: change > 0.05 ? 'critical' : 'warning',
            message: `Weight changed ${(change * 100).toFixed(1)}% in the past week`,
            value: latest,
            baseline: Number(weekAgo.weight_kg),
          })
        }
      }
    }

    // HRV declining trend
    const { data: hrvData } = await supabase
      .from('daily_aggregates')
      .select('avg_value, date')
      .eq('user_id', userId)
      .eq('metric', 'hrv_rmssd')
      .gte('date', new Date(Date.now() - 7 * 86400000).toISOString().split('T')[0])
      .order('date', { ascending: true })

    if (hrvData && hrvData.length >= 5) {
      const values = hrvData.map(d => Number(d.avg_value))
      const firstHalf = values.slice(0, Math.floor(values.length / 2))
      const secondHalf = values.slice(Math.floor(values.length / 2))
      const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length
      const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length

      if (secondAvg < firstAvg * 0.85) {
        alerts.push({
          user_id: userId,
          alert_type: 'trend',
          metric: 'hrv_rmssd',
          severity: 'warning',
          message: `HRV declining: avg dropped from ${firstAvg.toFixed(0)}ms to ${secondAvg.toFixed(0)}ms over 7 days`,
          value: secondAvg,
          baseline: firstAvg,
        })
      }
    }

    // Data retention: delete health_samples older than 90 days
    await supabase
      .from('health_samples')
      .delete()
      .eq('user_id', userId)
      .lt('ingested_at', new Date(Date.now() - 90 * 86400000).toISOString())
  }

  // Insert all alerts
  if (alerts.length > 0) {
    await supabase.from('trend_alerts').insert(alerts)
  }

  return new Response(JSON.stringify({
    users_processed: userIds.length,
    alerts_created: alerts.length,
  }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
})
```

- [ ] **Step 2: Add Vercel Cron to trigger daily trends**

Add to `vercel.json`:
```json
{
  "crons": [
    {
      "path": "/api/trigger-daily-trends",
      "schedule": "0 0 * * *"
    }
  ]
}
```

- [ ] **Step 3: Create the Vercel API route that calls the Edge Function**

Create `api/trigger-daily-trends.ts`:
```typescript
import type { VercelRequest, VercelResponse } from '@vercel/node'

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const response = await fetch(
    `${process.env.SUPABASE_URL}/functions/v1/daily-trends`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.SUPABASE_SERVICE_ROLE_KEY}`,
      },
      body: JSON.stringify({}),
    }
  )
  const data = await response.json()
  return res.status(200).json(data)
}
```

- [ ] **Step 4: Commit**

```bash
git add supabase/functions/daily-trends/ vercel.json api/trigger-daily-trends.ts
git commit -m "feat: add daily trend detection Edge Function + Vercel Cron trigger"
```

---

## Chunk 4: Frontend Realtime Integration + Health Sync Update

### Task 12: Connect Dashboard to Supabase Realtime for Live Scores

**Files:**
- Create: `client/src/hooks/use-scores.ts`

- [ ] **Step 1: Create use-scores hook with Realtime subscription**

Create `client/src/hooks/use-scores.ts`:
```typescript
import { useState, useEffect } from 'react'
import { supabase } from '@/lib/supabase-browser'

export interface UserScores {
  recoveryScore: number | null
  sleepScore: number | null
  strainScore: number | null
  stressScore: number | null
  nutritionScore: number | null
  energyBank: number | null
  computedAt: string | null
}

export function useScores(userId: string | undefined) {
  const [scores, setScores] = useState<UserScores | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!userId) return

    // Initial fetch
    supabase
      .from('user_scores')
      .select('*')
      .eq('user_id', userId)
      .single()
      .then(({ data }) => {
        if (data) {
          setScores({
            recoveryScore: data.recovery_score,
            sleepScore: data.sleep_score,
            strainScore: data.strain_score,
            stressScore: data.stress_score,
            nutritionScore: data.nutrition_score,
            energyBank: data.energy_bank,
            computedAt: data.computed_at,
          })
        }
        setLoading(false)
      })

    // Subscribe to real-time updates
    const channel = supabase
      .channel('user-scores')
      .on(
        'postgres_changes',
        {
          event: 'UPDATE',
          schema: 'public',
          table: 'user_scores',
          filter: `user_id=eq.${userId}`,
        },
        (payload) => {
          const d = payload.new as any
          setScores({
            recoveryScore: d.recovery_score,
            sleepScore: d.sleep_score,
            strainScore: d.strain_score,
            stressScore: d.stress_score,
            nutritionScore: d.nutrition_score,
            energyBank: d.energy_bank,
            computedAt: d.computed_at,
          })
        }
      )
      .subscribe()

    return () => {
      supabase.removeChannel(channel)
    }
  }, [userId])

  return { scores, loading }
}
```

- [ ] **Step 2: Commit**

```bash
git add client/src/hooks/use-scores.ts
git commit -m "feat: add useScores hook with Supabase Realtime subscription"
```

---

### Task 13: Update Health Sync to Post to Supabase Edge Function

**Files:**
- Modify: `client/src/lib/health-sync.ts`

- [ ] **Step 1: Update pullAppleHealth to call ingest-health-data Edge Function**

In `client/src/lib/health-sync.ts`, update the sync function to POST samples to the Supabase Edge Function instead of the ML backend `/biometrics/update` endpoint:

Add import at top:
```typescript
import { supabase } from '@/lib/supabase-browser'
```

Replace the POST to ML backend with:
```typescript
async function syncToSupabase(userId: string, samples: HealthSample[]) {
  const { data: { session } } = await supabase.auth.getSession()
  if (!session) return

  await fetch(`${import.meta.env.VITE_SUPABASE_URL}/functions/v1/ingest-health-data`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${session.access_token}`,
    },
    body: JSON.stringify({ user_id: userId, samples }),
  })
}
```

Note: Keep the existing ML backend POST as well for EEG biometric context during live sessions. The Supabase path is for persistent health pipeline storage.

- [ ] **Step 2: Commit**

```bash
git add client/src/lib/health-sync.ts
git commit -m "feat: update health sync to post to Supabase ingestion pipeline"
```

---

### Task 14: Deploy Edge Functions + Final Verification

- [ ] **Step 1: Deploy all Edge Functions to Supabase**

Run: `npx supabase functions deploy ingest-health-data`
Run: `npx supabase functions deploy compute-scores`
Run: `npx supabase functions deploy daily-trends`

- [ ] **Step 2: Set Edge Function secrets**

Run: `npx supabase secrets set SUPABASE_URL=<your-url> SUPABASE_SERVICE_ROLE_KEY=<your-key>`

- [ ] **Step 3: Set up Database Webhook for score computation**

In Supabase Dashboard → Database → Webhooks:
- Name: `score-computation-trigger`
- Table: `daily_aggregates`
- Events: `UPDATE`
- URL: `https://<project>.supabase.co/functions/v1/compute-scores`
- Headers: `Authorization: Bearer <service_role_key>`

- [ ] **Step 4: End-to-end test**

1. Start app: `npm run dev`
2. Log in
3. Manually insert a health sample via the app or curl
4. Check Supabase: `health_samples` has row, `daily_aggregates` updated, `user_scores` updated
5. Check browser: dashboard should receive Realtime update (scores will be null since engines are skeleton)

- [ ] **Step 5: Deploy to Vercel**

Run: `vercel --prod`
Verify: Vercel cron is configured, daily-trends will fire at midnight.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat: complete Phase 0 + Phase 1 — Supabase migration + pipeline infrastructure"
```

---

## Summary

After completing this plan, you have:

1. **Supabase fully set up** — DB, Auth, Edge Functions, Realtime, Storage
2. **All 23 existing tables migrated** from Neon to Supabase
3. **16 new tables** for pipeline, exercise, body, lifestyle tracking
4. **3 PG triggers** firing on every health data INSERT (aggregation, baselines, threshold alerts)
5. **3 Edge Functions** deployed (ingest-health-data, compute-scores skeleton, daily-trends)
6. **Adapter interfaces** ready for Redis swap (EventBus, ScoreCache, JobQueue)
7. **Supabase Realtime** pushing score updates to dashboard via useScores hook
8. **RLS policies** on all tables
9. **Vercel Cron** triggering daily trend detection
10. **Health sync** posting to Supabase pipeline

**Next plans to write:**
- Phase 2: Exercise & Weight Tracking (Strength Builder UI, workout logging, body metrics, 1RM)
- Phase 3: Wearable Integrations (Whoop, Oura, Garmin, CGM OAuth flows + adapters)
- Phase 4: Score Engines (Recovery, Sleep, Strain, Stress, Nutrition, Energy Bank algorithms)
- Phase 5: Remaining Features (Habits, Cycle, Smart Alarm, Mood, Sharing)
