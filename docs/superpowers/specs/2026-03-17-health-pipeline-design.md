# NeuralDreamWorkshop: Premium Health Pipeline Design Spec

**Date:** 2026-03-17
**Status:** Approved
**Scope:** Full Supabase migration + event-driven health pipeline + all competitor features + EEG integration

---

## 1. Overview

Transform NeuralDreamWorkshop from an EEG-focused emotion lab into a complete health platform that matches every competitor feature (Recovery/Sleep/Strain/Stress/Nutrition scores, Energy Bank, exercise tracking, weight/body composition, habit journal, cycle tracking, smart alarm, cardio load, wearable integrations) while retaining our EEG moat (16 ML models, dream analysis, flow detection, voice biomarkers).

### Architecture Decision

- **Frontend:** Vercel (React, free)
- **Database:** Supabase PostgreSQL (replaces Neon, free tier 500MB)
- **Auth:** Supabase Auth (replaces bcrypt + express-session)
- **Storage:** Supabase Storage (food photos, dream images, voice recordings, 1GB free)
- **API:** Supabase Edge Functions (score engines, health ingestion, 500K invocations/mo free)
- **Real-time:** Supabase Realtime (Postgres LISTEN/NOTIFY for live score updates)
- **Event Bus:** PgNotifyEventBus adapter (swap to Redis Streams later)
- **Cache:** PgTableCache adapter (swap to Redis later)
- **Job Queue:** DirectCallQueue adapter (swap to BullMQ later)
- **ML Backend:** Local FastAPI (16 models, EEG stream, BrainFlow — stays on Mac)
- **Cron:** Vercel Cron (`@daily` on Hobby plan) — triggers trend detection Edge Functions
- **Monthly cost:** $0 (requires data retention policy — see Section 2.3)

### Storage Budget (Critical Constraint)

Supabase free tier = 500MB. Estimated usage per active user:
- health_samples: ~50-100MB/day if continuous HR (1 row/sec from Whoop)
- **Mitigation:** Raw health_samples retained for 90 days, then archived to Supabase Storage as compressed JSON. Daily aggregates retained indefinitely (tiny). pg_cron replacement (Vercel Cron) runs `DELETE FROM health_samples WHERE ingested_at < now() - interval '90 days'` nightly.
- **Budget for 1 user:** ~15MB/day after dedup (15-min polling, not per-second). 90 days = ~1.3GB → exceeds free tier. Must either: (a) downsample to hourly aggregates after 30 days, or (b) upgrade to Supabase Pro ($25/mo) when data exceeds 400MB.
- **Monitoring:** Edge Function checks storage usage weekly, alerts at 80% (400MB).

### Redis Migration Path

All pipeline abstractions use adapter interfaces. When Redis is needed:
1. Add Upstash Redis (serverless, free tier 10K cmd/day)
2. Swap `PgNotifyEventBus` → `RedisStreamEventBus`
3. Swap `PgTableCache` → `RedisCache`
4. Swap `DirectCallQueue` → `BullMQQueue`
5. Zero changes to score engines, API endpoints, or frontend

If cost becomes an issue, fall back to PostgreSQL adapters with one config change.

---

## 2. Pipeline Architecture

### 2.1 Data Flow

```
SOURCES
  Muse 2/S, OpenBCI, Emotiv, NeuroSky, BrainBit (any BrainFlow EEG device)
  Apple HealthKit (iOS Capacitor)
  Google Health Connect (Android Capacitor)
  Whoop Band (OAuth2)
  Oura Ring (OAuth2)
  Garmin (OAuth1 + push)
  CGM — Dexcom / Libre (OAuth2)
  Manual Input (weight, meals, habits)

INGESTION LAYER
  Each source → Edge Function: ingest-health-data
    - Validates schema (Zod)
    - Normalizes units (kg/lbs, C/F configurable)
    - Deduplicates (source + metric + timestamp unique)
    - Tags source provider
  → INSERT into health_samples ON CONFLICT (user_id, source, metric, recorded_at) DO NOTHING
    (idempotent — webhook retries and duplicate syncs are safe)
  → EventBus.publish("health.ingested", { userId, metrics[] })

AGGREGATION LAYER
  PG Trigger on health_samples INSERT (PL/pgSQL, runs in-database):
    - UPSERT daily_aggregates (running avg, min, max, count per metric)
    - UPSERT user_baselines (14-day rolling average per metric)
    - Simple threshold checks (HR > 120, SpO2 < 94) → INSERT trend_alerts
    Note: aggregation runs as PL/pgSQL inside the trigger — no Edge Function needed.

SCORE ENGINE LAYER
  Supabase Database Webhook on daily_aggregates UPDATE
    → POSTs to Edge Function: compute-scores (HTTP callback, not pg_notify)
    - Reads daily_aggregates + user_baselines (14-day window)
    - Runs 6 score engines:
      1. Recovery (HRV + RHR + temp + resp + SpO2 + sleep)
      2. Sleep (stages + HR dip + efficiency + debt)
      3. Strain (TRIMP + HR zones + active + passive)
      4. Stress (HRV + EEG stress index + wearable data)
      5. Nutrition (AHEI + macros + glucose response)
      6. Energy Bank (composite of above 5)
  → UPSERT user_scores
  → Cache.set("user:{id}:scores", scores)

TREND DETECTION LAYER
  Vercel Cron (@daily) → calls Edge Function: daily-trends
    - 7-day moving averages for all metrics
    - Rate-of-change analysis (weight ±2%/week, HRV declining)
    - Anomaly detection (z-score > 2 from baseline)
    - Sleep debt accumulation
    - Cardio load status (ATL vs CTL ratio)
  → INSERT trend_alerts if anomaly found
  → Push notification if critical

DELIVERY LAYER
  Supabase Realtime watches user_scores table
    - Any UPDATE → pushed to all connected clients
    - Dashboard auto-updates scores, charts, Energy Bank
  Push Notifications for alerts + smart alarm
```

### 2.2 Adapter Interfaces

```typescript
// adapters/event-bus.ts
interface EventBus {
  publish(channel: string, data: unknown): Promise<void>
  subscribe(channel: string, handler: (data: unknown) => void): void
}
// Now: PgNotifyEventBus (pg_notify + LISTEN)
// Later: RedisStreamEventBus (XADD + XREAD)

// adapters/cache.ts
interface ScoreCache {
  get(userId: string): Promise<UserScores | null>
  set(userId: string, scores: UserScores, ttl?: number): Promise<void>
  invalidate(userId: string): Promise<void>
}
// Now: PgTableCache (user_scores table direct read)
// Later: RedisCache (HSET/HGETALL)

// adapters/job-queue.ts
interface JobQueue {
  enqueue(job: string, payload: unknown, opts?: JobOpts): Promise<void>
  process(job: string, handler: (payload: unknown) => Promise<void>): void
}
// Now: DirectCallQueue (sync Edge Function invocation)
// Later: BullMQQueue (Redis-backed with retries)
```

---

## 3. Database Schema

### 3.1 Existing Tables (Migrated from Neon to Supabase)

All 16 existing Drizzle ORM tables migrate to Supabase PostgreSQL. Same schema, new connection string. `users` table replaced by Supabase Auth (auth.users).

Tables: health_metrics, emotion_readings, brain_readings, health_samples, dream_analysis, dream_symbols, eeg_sessions, ai_chats, user_settings, push_subscriptions, food_logs, meal_history, user_readings, study_participants, study_sessions, study_morning/daytime/evening_entries, password_reset_tokens, datadog_error_log.

### 3.2 New Tables

#### health_samples (enhanced, time-series primary)
```sql
CREATE TABLE health_samples (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  source text NOT NULL, -- apple_health | google_fit | oura | whoop | garmin | cgm | eeg | manual
  metric text NOT NULL, -- heart_rate | hrv_rmssd | resting_hr | respiratory_rate | spo2 | skin_temp |
                        -- sleep_deep_min | sleep_rem_min | sleep_light_min | sleep_awake_min |
                        -- sleep_efficiency | steps | active_calories | weight_kg | body_fat_pct |
                        -- lean_mass_kg | vo2_max | workout_strain | glucose_mgdl | ...
  value numeric NOT NULL,
  unit text NOT NULL,
  metadata jsonb,
  recorded_at timestamptz NOT NULL,
  ingested_at timestamptz DEFAULT now(),
  UNIQUE(user_id, source, metric, recorded_at)
);
CREATE INDEX idx_health_samples_user_metric ON health_samples(user_id, metric, recorded_at DESC);
```

#### daily_aggregates
```sql
CREATE TABLE daily_aggregates (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  date date NOT NULL,
  metric text NOT NULL,
  avg_value numeric,
  min_value numeric,
  max_value numeric,
  sum_value numeric,
  sample_count integer DEFAULT 0,
  updated_at timestamptz DEFAULT now(),
  UNIQUE(user_id, date, metric)
);
CREATE INDEX idx_daily_agg_user_date ON daily_aggregates(user_id, date DESC);
```

#### user_baselines (14-day rolling)
```sql
CREATE TABLE user_baselines (
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  metric text NOT NULL,
  baseline_avg numeric,
  baseline_stddev numeric,
  sample_count integer DEFAULT 0,
  last_updated timestamptz DEFAULT now(),
  PRIMARY KEY(user_id, metric)
);
```

#### user_scores (Realtime-enabled)
```sql
CREATE TABLE user_scores (
  user_id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  recovery_score numeric, -- 0-100
  sleep_score numeric,    -- 0-100
  strain_score numeric,   -- 0-100+ (logarithmic)
  stress_score numeric,   -- 0-100
  nutrition_score numeric, -- 1-100
  energy_bank numeric,    -- 0-100
  recovery_inputs jsonb,
  sleep_inputs jsonb,
  strain_inputs jsonb,
  stress_inputs jsonb,
  nutrition_inputs jsonb,
  computed_at timestamptz DEFAULT now()
);
-- Enable Supabase Realtime on this table
```

#### score_history
```sql
CREATE TABLE score_history (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  date date NOT NULL,
  recovery_score numeric,
  sleep_score numeric,
  strain_score numeric,
  stress_score numeric,
  nutrition_score numeric,
  energy_bank numeric,
  created_at timestamptz DEFAULT now(),
  UNIQUE(user_id, date)
);
```

#### trend_alerts
```sql
CREATE TABLE trend_alerts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  alert_type text NOT NULL, -- anomaly | threshold | trend | rapid_change
  metric text NOT NULL,
  severity text NOT NULL,   -- info | warning | critical
  message text NOT NULL,
  value numeric,
  baseline numeric,
  acknowledged boolean DEFAULT false,
  created_at timestamptz DEFAULT now()
);
CREATE INDEX idx_alerts_user ON trend_alerts(user_id, created_at DESC);
```

#### device_connections
```sql
CREATE TABLE device_connections (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  provider text NOT NULL, -- whoop | oura | garmin | dexcom | libre | strava
  access_token text NOT NULL, -- encrypted via Supabase Vault (pgsodium)
  refresh_token text,         -- encrypted via Supabase Vault (pgsodium)
  token_expires_at timestamptz,
  scopes text[],
  last_sync_at timestamptz,
  sync_status text DEFAULT 'active', -- active | error | disconnected
  error_message text,
  connected_at timestamptz DEFAULT now(),
  UNIQUE(user_id, provider)
);
```

#### exercises (library)
```sql
CREATE TABLE exercises (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  category text NOT NULL, -- strength | cardio | flexibility | hiit
  muscle_groups text[] NOT NULL,
  equipment text, -- barbell | dumbbell | cable | machine | bodyweight | band | kettlebell
  instructions text,
  video_url text,
  is_custom boolean DEFAULT false,
  created_by uuid REFERENCES auth.users(id)
);
```

#### workouts
```sql
CREATE TABLE workouts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  name text,
  workout_type text NOT NULL, -- strength | cardio | flexibility | hiit | mixed
  started_at timestamptz NOT NULL,
  ended_at timestamptz,
  duration_min numeric,
  total_strain numeric,
  avg_hr numeric,
  max_hr numeric,
  calories_burned numeric,
  hr_zones jsonb, -- {z1_min, z2_min, z3_min, z4_min, z5_min}
  hr_recovery numeric, -- bpm drop at +2min
  source text NOT NULL, -- manual | apple_watch | garmin | whoop | oura
  eeg_session_id uuid REFERENCES eeg_sessions(id),
  notes text,
  created_at timestamptz DEFAULT now()
);
CREATE INDEX idx_workouts_user ON workouts(user_id, started_at DESC);
```

#### workout_sets
```sql
CREATE TABLE workout_sets (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  workout_id uuid REFERENCES workouts(id) ON DELETE CASCADE,
  exercise_id uuid REFERENCES exercises(id),
  set_number integer NOT NULL,
  set_type text DEFAULT 'normal', -- normal | warmup | dropset | superset | failure | timed
  reps integer,
  weight_kg numeric,
  duration_sec integer,
  rest_sec integer,
  rpe numeric, -- 1-10
  completed boolean DEFAULT true,
  created_at timestamptz DEFAULT now()
);
```

#### workout_templates
```sql
CREATE TABLE workout_templates (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  name text NOT NULL,
  description text,
  exercises jsonb NOT NULL, -- [{exercise_id, sets, reps, weight, rest}]
  is_ai_generated boolean DEFAULT false,
  times_used integer DEFAULT 0,
  created_at timestamptz DEFAULT now()
);
```

#### body_metrics
```sql
CREATE TABLE body_metrics (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  weight_kg numeric,
  body_fat_pct numeric,
  lean_mass_kg numeric,
  bmi numeric,
  height_cm numeric,
  source text NOT NULL, -- apple_health | google_fit | manual | smart_scale
  recorded_at timestamptz NOT NULL,
  created_at timestamptz DEFAULT now()
);
CREATE INDEX idx_body_user ON body_metrics(user_id, recorded_at DESC);
```

#### exercise_history (1RM tracking)
```sql
CREATE TABLE exercise_history (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  exercise_id uuid REFERENCES exercises(id),
  date date NOT NULL,
  best_weight_kg numeric,
  best_reps integer,
  estimated_1rm numeric, -- Epley: weight × (1 + reps/30)
  total_volume numeric,  -- sum(sets × reps × weight)
  UNIQUE(user_id, exercise_id, date)
);
```

#### habits
```sql
CREATE TABLE habits (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  name text NOT NULL,
  category text, -- hydration | sunlight | screen_time | caffeine | custom
  icon text,
  target_value numeric,
  unit text,
  is_active boolean DEFAULT true,
  created_at timestamptz DEFAULT now()
);
```

#### habit_logs
```sql
CREATE TABLE habit_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  habit_id uuid REFERENCES habits(id) ON DELETE CASCADE,
  value numeric NOT NULL,
  note text,
  logged_at timestamptz DEFAULT now()
);
CREATE INDEX idx_habit_logs_user ON habit_logs(user_id, logged_at DESC);
```

#### cycle_tracking
```sql
CREATE TABLE cycle_tracking (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  date date NOT NULL,
  flow_level text, -- none | light | medium | heavy
  symptoms text[], -- cramps, bloating, headache, mood_swings, fatigue, ...
  phase text, -- period | follicular | ovulatory | luteal (computed)
  contraception text,
  basal_temp numeric,
  notes text,
  UNIQUE(user_id, date)
);
```

#### mood_logs
```sql
CREATE TABLE mood_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  mood_score numeric NOT NULL, -- 1-10 slider
  energy_level numeric, -- 1-10
  notes text,
  logged_at timestamptz DEFAULT now()
);
```

---

## 4. Wearable Integrations

### 4.1 Device Adapter Interface

```typescript
interface WearableAdapter {
  name: string // "whoop" | "oura" | "garmin" | "dexcom" | "libre"
  authenticate(userId: string): Promise<OAuthTokens>
  sync(userId: string, since: Date): Promise<HealthSample[]>
  getCapabilities(): MetricType[]
  webhookHandler?(req: Request): Promise<void> // push-based devices
  refreshToken?(userId: string): Promise<OAuthTokens>
}
```

All adapters normalize to `HealthSample` format → same `ingest-health-data` Edge Function → same pipeline.

### 4.2 Supported Devices

| Device | Auth | Sync Method | Unique Data |
|--------|------|-------------|-------------|
| **Whoop** | OAuth2 | 15-min poll + webhooks | Continuous HR, strain, recovery |
| **Oura Ring** | OAuth2 | 15-min poll + webhooks | Readiness score, temp deviation, 2yr history |
| **Garmin** | OAuth1 | Push API (Garmin pushes to webhook) | Body Battery, stress level, GPS, VO2 max |
| **Apple HealthKit** | Capacitor plugin | 15-min on-device poll | Weight from smart scales, all Apple Watch data |
| **Google Health Connect** | Capacitor plugin | 15-min on-device poll | Android wearable data |
| **Dexcom CGM** | OAuth2 | 5-min poll | Real-time glucose mg/dL |
| **Libre CGM** | OAuth2 | 15-min poll | Glucose readings |
| **Any EEG** | BrainFlow (local BLE/USB) | 256Hz WebSocket stream | 4-16ch EEG, emotions, sleep, dreams, flow |

### 4.3 EEG Device Abstraction

```typescript
interface EEGDeviceAdapter {
  name: string // "muse_2" | "openbci_cyton" | "emotiv_epoc" | ...
  channelCount: number
  samplingRate: number
  electrodePositions: string[] // ["TP9","AF7","AF8","TP10"] for Muse
  connect(): Promise<void>
  startStream(): AsyncGenerator<EEGFrame>
  stop(): Promise<void>
}
```

BrainFlow handles hardware abstraction. The adapter normalizes channel count and electrode positions so ML models receive consistent input. Higher channel-count devices (OpenBCI 16ch) get better accuracy from models that can use extra channels; 4ch models work on all devices.

### 4.4 Data Source Priority

When multiple devices report the same metric:
1. **User preference** — user picks primary device per metric in Settings
2. **Highest fidelity** — if no preference, use device with most data points
3. **Recency** — if tied, use most recent reading
4. All sources stored in health_samples regardless; priority affects only score computation

---

## 5. Score Engines

### 5.1 Recovery Score (0-100%)

"How ready is your body to perform today?"

**Inputs (weighted, vs 14-day personal baseline):**
- HRV vs baseline: 25%
- Resting HR vs baseline: 20%
- Sleep Score (previous night): 25%
- Skin Temperature deviation: 10%
- Respiratory Rate vs baseline: 10%
- SpO2 vs baseline: 10%

**EEG Bonus:** Evening EEG stress index adjusts recovery ±5%.

**Calibration:** 14-day learning period to establish personal baselines. Each metric scored as percentage of personal baseline (above = boost, below = penalty).

### 5.2 Sleep Score (0-100%)

"How well did you sleep last night?"

**Inputs:**
- Time Asleep vs Goal: 30%
- Sleep Stage Balance (deep/REM/light proportions): 25%
- Heart Rate Dip during sleep: 20%
- Sleep Efficiency (% time asleep vs in bed): 25%

**Sleep Needed** = Goal + Strain Adjustment + Sleep Debt (7-day weighted, distributed over 4 days) + Efficiency Buffer.

**EEG Bonus:** If EEG worn overnight, real sleep staging (92.98% accuracy) replaces wearable estimates. Dream detection + lucidity score added.

### 5.3 Strain Score (0-100+, logarithmic)

"How much stress has your body absorbed today?"

**Components:**
- Active Strain: from logged workouts (TRIMP)
- Passive Strain: steps, movement, non-workout HR elevation

**Algorithm:**
```
TRIMP = duration(min) × avg_HR_ratio × e^(1.92 × avg_HR_ratio)
HR_ratio = (exerciseHR - restHR) / (maxHR - restHR)
strain = 14.3 × ln(1 + TRIMP)  // k=14.3 maps to ~0-21 scale (Whoop-equivalent)
                                // Gender factor: 1.92 (male) or 1.67 (female) in TRIMP exponent
```

**Target Strain:** Personalized daily recommendation based on recovery score + historical pattern.

**EEG Bonus:** Cognitive strain from focus/concentration sessions adds to total strain.

**Edge Function timeout handling:** Score computation split into 6 independent Edge Function invocations (one per score) called in parallel via `Promise.all()`. Each completes well within the 2-second wall-clock limit. Only changed scores are recomputed (incremental — if only sleep data changed, only Sleep Score and Energy Bank recompute).

### 5.4 Stress Score (0-100)

"How stressed is your body right now?"

**Inputs:**
- HRV trend (declining = stress): 30%
- HR elevation above resting: 25%
- Respiratory rate elevation: 15%
- Skin temp deviation: 10%
- **EEG (when active): 20%** — high-beta/beta ratio, frontal asymmetry, theta suppression

This is our biggest differentiator. Whoop/competitor guess stress from HR. We measure it from the brain directly.

### 5.5 Nutrition Score (1-100)

"How well are you eating?"

**Part 1 — Food Quality (AHEI):**
- Positive: vegetables, fruits, whole grains, healthy oils, nuts/legumes, omega-3
- Negative: red meat, processed meat, alcohol, excess sugar, excess sodium
- Diminishing returns prevent single food groups from dominating
- Based on Harvard's Alternate Healthy Eating Index

**Part 2 — Glucose Impact (if CGM connected):**
- Per-meal glucose response: time to peak, peak height, return to baseline
- Lower spike = higher score

**EEG Bonus:** Food-mood correlation analysis ("meals high in processed sugar correlate with 23% lower focus scores 2 hours later").

### 5.6 Energy Bank (0-100)

"Your body's battery level right now."

```
energy = recovery × (1 - strain/max_strain) × sleep_factor × stress_penalty
```

- Starts at Recovery Score each morning
- Depletes with strain throughout day
- Stress accelerates depletion
- Good nutrition slows depletion
- Recharges during sleep

**EEG Bonus:** Flow states boost energy bank. High cognitive load without flow drains it faster.

### 5.7 Cardio Load (TRIMP-based)

- **Acute Training Load (ATL):** 7-day exponentially weighted TRIMP
- **Chronic Training Load (CTL):** 42-day exponentially weighted TRIMP
- **Training Stress Balance (TSB):** CTL - ATL

**7 Cardio Statuses:**
- TSB > 15: Detraining
- TSB 5-15: Maintaining
- TSB -5 to 5: Productive
- TSB -15 to -5: Peaking
- TSB < -15: Fatigued / Overtraining

### 5.8 Heart Rate Recovery

HRR = peak HR (zone 4+) - HR at +2 minutes post-exercise.

Benchmarks: >50 bpm Excellent, 40-50 Good, 30-40 Average, <30 Below Average.

---

## 6. Exercise & Workout Tracking

### 6.1 Strength Builder
- 700+ exercise library with muscle groups, equipment, instructions
- Set types: normal, superset, dropset, warmup, cooldown, timed, failure
- Weight defaults to last-used weight per exercise
- Rest timers configurable per exercise or per set
- 1RM estimation via Epley formula: weight × (1 + reps/30)
- Workout templates: save, reuse, AI-generated
- Progression history per exercise (weight/rep over time)
- Muscular strain visualization (radar chart of muscle groups)

### 6.2 Passive Tracking
- Apple Watch workouts auto-imported with HR data
- Garmin/Whoop/Oura workout sync via OAuth
- Steps, active energy, exercise minutes from wearables
- HR zones: time in each of 5 zones per workout

### 6.3 EEG + Exercise Integration
- Link workout to EEG session via `eeg_session_id`
- Flow state detection during exercise
- Focus quality scoring (mind-muscle connection)
- Post-workout brain recovery tracking
- Exercise-emotion correlation analysis
- Optimal workout timing recommendations

---

## 7. Weight & Body Composition

### 7.1 Data Sources
- Apple HealthKit (body mass, body fat %, lean mass, BMI, height)
- Google Health Connect (weight, body fat, height)
- Smart scales via HealthKit (Withings, Eufy, Renpho)
- Manual input
- Oura (if synced to Apple Health)

### 7.2 Trend Detection
- 7-day moving average (smooths daily fluctuations)
- Rate of change alerts: >2% change/week or >5%/month
- Goal progress: on track / falling behind / ahead
- Correlation: weight change vs sleep, strain, nutrition scores

---

## 8. Additional Features

### 8.1 Habit Journal
- Pre-built: hydration, sunlight, screen time, caffeine
- Custom user-defined habits with targets
- Mood slider (1-10 scale)
- Morning prompts
- Habit-score correlations ("hydration improves recovery by 12%")
- Streak tracking

### 8.2 Cycle Tracking
- Flow and symptom logging (20+ symptoms)
- Phase prediction: period, follicular, ovulatory, luteal
- Temperature shift for ovulatory detection
- Recovery/HRV/stress trends per cycle phase
- Contraception logging

### 8.3 Smart Alarm
- **Sleep Needed Alarm:** wakes when sleep goal reached
- **Smart Alarm:** lightest sleep phase within window
- **Regular Alarm:** fixed time
- **No Alarm:** tracks time to fall asleep
- Uses EEG sleep staging if worn (most accurate), falls back to wearable motion data

---

## 9. Security: Row-Level Security Policies

Every table with `user_id` gets this default RLS policy:
```sql
ALTER TABLE <table> ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can only access their own data"
  ON <table> FOR ALL USING (auth.uid() = user_id);
```

**Exceptions:**
- `exercises` (shared library): `SELECT` allowed for all authenticated users. `INSERT/UPDATE/DELETE` only where `is_custom = true AND created_by = auth.uid()`.
- `device_connections`: Strictest RLS — `SELECT/UPDATE/DELETE` only by owner. Token columns additionally encrypted via Supabase Vault (pgsodium). Edge Functions use service_role key to decrypt tokens during sync.
- `trend_alerts`: Users can SELECT and UPDATE (acknowledge) their own alerts only.

**OAuth Token Encryption:**
Tokens in `device_connections` are encrypted using Supabase Vault (`pgsodium` extension, available on free tier). Edge Functions decrypt using `vault.decrypted_secrets` view with service_role key. Application code never sees raw tokens — only the Edge Function sync workers.

---

## 10. Migration Plan (Neon → Supabase)

### 10.1 Database Migration
1. Create Supabase project
2. Export Neon schema via pg_dump
3. Import to Supabase PostgreSQL
4. **User ID type migration:** Existing schema uses `varchar("id")` with UUID values. Supabase Auth uses native `uuid`. Steps:
   a. Verify all existing IDs are valid UUIDs (they are — generated by `gen_random_uuid()`)
   b. ALTER all `user_id` columns from `varchar` to `uuid` using `ALTER COLUMN user_id TYPE uuid USING user_id::uuid`
   c. Create user entries in `auth.users` matching existing user IDs
   d. Update all FK references to point to `auth.users(id)`
   e. Drop old `users` table after verification
5. **health_samples migration:** Existing table uses `real("value")` and no UNIQUE constraint. Steps:
   a. `ALTER COLUMN value TYPE numeric`
   b. Deduplicate any existing rows that would violate the new constraint
   c. `ADD CONSTRAINT ... UNIQUE(user_id, source, metric, recorded_at)`
   d. Add `unit` column as `text NOT NULL DEFAULT 'unknown'` (backfill existing rows)
6. Update Drizzle ORM connection string (DATABASE_URL) and schema types
7. Verify all existing queries work
8. Export/import existing data if any

### 10.2 Auth Migration
1. Set up Supabase Auth (email/password + OAuth providers)
2. Update frontend auth hooks to use @supabase/supabase-js
3. Migrate existing bcrypt password hashes (Supabase supports bcrypt import)
4. Remove express-session + connect-pg-simple
5. Enable RLS on all tables, apply policies from Section 9

### 10.3 API Migration
1. Move simple CRUD routes to Supabase client direct access (with RLS)
2. Move complex business logic to Edge Functions
3. Keep Express locally for ML proxy + GPT integration
4. Update frontend API client to use Supabase client + Edge Function calls

---

## 11. Implementation Phases

### Phase 0: Supabase Migration (Foundation)
- Create Supabase project
- Migrate schema + data from Neon
- Migrate auth to Supabase Auth
- Update frontend to use Supabase client
- Verify all existing features work

### Phase 1: Pipeline Infrastructure
- Create new tables (health_samples, daily_aggregates, user_baselines, user_scores, score_history, trend_alerts)
- Implement adapter interfaces (EventBus, ScoreCache, JobQueue)
- PG triggers for aggregation on health_samples INSERT
- Edge Function: ingest-health-data (validation, normalization, dedup)
- Edge Function: compute-scores (skeleton — wired to pipeline, scores computed later)
- Supabase Realtime on user_scores table
- pg_cron for daily trend detection

### Phase 2: Exercise & Weight Tracking
- Create tables (exercises, workouts, workout_sets, workout_templates, body_metrics, exercise_history)
- Seed 700+ exercise library
- Workout logging UI (strength builder)
- Weight/body composition input + HealthKit sync
- HR zones computation
- 1RM estimation + progression history
- Workout → health_samples pipeline integration (strain)

### Phase 3: Wearable Integrations
- device_connections table + OAuth flow UI
- Whoop adapter (OAuth2 + sync + webhooks)
- Oura adapter (OAuth2 + sync + webhooks)
- Garmin adapter (OAuth1 + push API)
- CGM adapter (Dexcom OAuth2)
- EEG device abstraction (BrainFlow multi-device)
- Enhanced Apple HealthKit sync (weight, body fat, VO2 max)

### Phase 4: Score Engines
- Recovery Score engine
- Sleep Score engine (+ Sleep Needed + Sleep Debt)
- Strain Score engine (TRIMP + logarithmic scaling)
- Stress Score engine (HRV + HR + EEG fusion)
- Nutrition Score engine (AHEI + glucose)
- Energy Bank engine (composite)
- Cardio Load (ATL/CTL/TSB)
- Heart Rate Recovery tracking
- Score history recording + trend charts

### Phase 5: Remaining Features
- Habit journal (habits + habit_logs tables, UI, streak tracking)
- Cycle tracking (cycle_tracking table, phase prediction, symptom logging)
- Smart alarm (4 types, EEG-enhanced)
- Mood logger (mood_logs table, slider UI)
- Trend alerts UI (acknowledge, history)
- Push notifications for anomalies
- Settings page (device priority, units, sleep goal, HR zones config)

---

## 12. What Makes This Different From competitor

Every score has an **EEG bonus** that competitor, Whoop, and Oura can never match:

| Feature | Them (HR-only) | Us (HR + EEG) |
|---------|---------------|---------------|
| Stress | "HR is elevated" (could be caffeine) | Neural high-beta + HR + HRV (actual stress) |
| Sleep | "You didn't move" (could be lying awake) | Real EEG staging at 92.98% accuracy |
| Recovery | Proxy from autonomic nervous system | Brain recovery + body recovery |
| Flow | Not measured | Direct detection during work/exercise |
| Dreams | Not measured | Detection + lucidity scoring |
| Emotions | Not measured | 6-class classification + valence/arousal |
| Food impact | Calorie counting | Food-mood-dream correlation |

Plus: voice biomarkers, 132 ML API routes, 30-day research study protocol, dream symbol tracking, AI wellness companion (GPT-5).
