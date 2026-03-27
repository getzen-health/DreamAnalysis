# InsightEngine — Design Spec
**Date:** 2026-03-27
**Project:** NeuralDreamWorkshop
**Status:** Approved by Sravya — Rev 2 (post spec-review)

---

## Problem

Current NDW coaching is rule-based and reactive: hard-coded thresholds ("stress > 65 → suggest box breathing") compare against population averages, not the user's own baseline. Oura, Whoop, and Garmin all do the same thing — none have EEG. The result: generic advice that doesn't account for *your* patterns, and a 6-class emotion ceiling that doesn't match how humans actually feel.

**Goal:** Build an insight engine that (a) compares every signal against *your personal baseline*, (b) discovers non-obvious causal patterns in your own data, (c) delivers interventions at three urgency tiers, and (d) lets you extend the emotion taxonomy with your own labels — grounded in EEG fingerprints.

---

## What This Replaces / Extends

| Existing | What Changes |
|----------|-------------|
| `brain-coach-card.tsx` rule engine | Replaced by `InsightEngine.getRealTimeInsights()` |
| `insights.tsx` rule-based cards | Replaced by `InsightEngine.getStoredInsights()` |
| `intervention-engine.ts` | Absorbed into `InterventionLibrary` class |
| `recovery-interventions.tsx` | Becomes a renderer for `InsightEngine` output |
| 6-class emotion ML output | Extended by `EmotionTaxonomy` (user labels + EEG fingerprints) |

Everything is backward-compatible — existing components receive `InsightEngine` output through the same prop interfaces.

---

## Architecture

```
Raw signals (EEG frame + health sync + user emotion label)
        │
        ▼
  BaselineStore                     client/src/lib/insight-engine/baseline-store.ts
  - rolling 7-day z-score per metric
  - metrics: stress, focus, valence, arousal, HRV, sleep, steps, energy
  - persists to localStorage("ndw_baseline_map") only
  - keys are normalized to 0-1 range before storage (see scale note below)
        │
        ▼
  PatternDiscovery                  client/src/lib/insight-engine/pattern-discovery.ts
  - 5 statistical passes (see below)
  - runs as background worker on data update
  - discovered patterns stored in Supabase user_patterns table (upsert on userId+passType)
        │
        ▼
  DeviationDetector                 client/src/lib/insight-engine/deviation-detector.ts
  - inputs: current reading + BaselineStore
  - outputs: DeviationEvent (see below)
  - duration tracking: stored in localStorage("ndw_deviation_timers") keyed by metric
        │
        ├──▶ RealTimeTier           (EEG streaming → instant banner, cooldown 15min)
        ├──▶ PushTier               (deviation sustained >20min → browser push notification)
        └──▶ MorningBriefingTier    (first open after 6AM → LLM synthesis of last 24h)

  EmotionTaxonomy                   client/src/lib/insight-engine/emotion-taxonomy.ts
  - runs parallel, independent of insight tiers
  - Layer 1: ML base (valence + arousal → circumplex position)
  - Layer 2: 64-item preset vocabulary (4 quadrants × 16)
  - Layer 3: personal vocabulary (user labels + live EEG fingerprints)
  - all Supabase writes gated by getSupabaseIfAllowed() (privacy mode safe)
```

All computation runs **client-side** except the morning briefing LLM call (Claude API via Express `/api/morning-briefing`). New database tables: `user_patterns` and `emotion_fingerprints`.

---

## Module 1: BaselineStore

**File:** `client/src/lib/insight-engine/baseline-store.ts`

Tracks a rolling 7-day window of readings per metric. Computes mean and standard deviation per 2-hour time bucket (12 buckets × 8 metrics = 96 baseline cells).

```typescript
interface BaselineCell {
  mean: number;      // all values normalized to 0-1 range
  std: number;
  sampleCount: number;
  lastUpdated: string; // ISO timestamp
}

// Key: `${metric}_${hourBucket}` e.g. "stress_14" = stress readings 14:00-16:00
type BaselineMap = Record<string, BaselineCell>;
```

**Scale normalization (critical):** All metrics are stored and compared on a 0–1 scale.
- `stress`, `focus`, `arousal`, `energy`: already 0–1, no transform
- `valence`: stored as `(rawValence + 1) / 2` (maps −1…+1 → 0…1)
- `hrv`: stored as `Math.min(hrv / 120, 1)` (caps at 120ms as 1.0)
- `sleep`: score / 100
- `steps`: `Math.min(steps / 15000, 1)` (caps at 15k as 1.0)

**Z-score computation:**
```
zScore = (normalizedValue - cell.mean) / Math.max(cell.std, 0.01)
```

Minimum `sampleCount` before a cell is used: **7**. Cells with fewer samples fall back to population defaults (all on 0–1 scale):

| Metric | Default mean | Default std | Source |
|--------|-------------|-------------|--------|
| stress | 0.40 | 0.15 | NDW user population estimate |
| focus | 0.55 | 0.18 | NDW user population estimate |
| valence (normalized) | 0.55 | 0.20 | Slightly positive default (Russell 1980) |
| arousal (normalized) | 0.50 | 0.18 | Neutral default |
| hrv (normalized) | 0.42 | 0.15 | 50ms raw ≈ 0.42 normalized |
| sleep | 0.65 | 0.15 | Population sleep score estimate |
| steps (normalized) | 0.35 | 0.20 | ~5k steps/day ≈ 0.35 of 15k cap |
| energy | 0.50 | 0.18 | Neutral default (already 0–1, no transform) |

**Persistence:** `localStorage("ndw_baseline_map")` only. Not written to Supabase. The baseline is a derived aggregate — it can be recomputed from raw readings if needed. Storing it in `alertThresholds` JSONB would conflict with existing production usage of that column.

**Privacy mode:** BaselineStore reads/writes localStorage only — no Supabase calls, so privacy mode has no effect on this module.

---

## Module 2: PatternDiscovery

**File:** `client/src/lib/insight-engine/pattern-discovery.ts`

Five correlation passes. Each has a **minimum data threshold** — no pattern is surfaced without sufficient evidence.

### Pass 1 — Time-of-Day Buckets
- Groups all historical readings into 2-hour buckets
- Computes bucket mean + SD for stress, focus, valence
- Fires insight when current reading deviates >1.5 SD from that bucket's baseline
- Minimum: 7+ readings in the target bucket
- Output: `"Your focus at 2PM is usually 0.71. Right now it's 0.38. This dip is consistent across 80% of your afternoons."`

### Pass 2 — Food→Emotion Lag
- For each food log entry, checks metric change at T+60, T+90, T+120, T+180 minutes
- Uses Pearson r — only surfaces if |r| > 0.45
- Minimum: 10+ food+emotion paired data points
- **Graceful degradation:** When `food_logs.carbs` / `dominantMacro` is null (food photo AI not used), degrades to presence/absence correlation: "Eating within X hours predicts a stress change in you." Macro classification only claimed when `carbs` field is non-null.
- Lag with highest |r| wins; reported with that lag time
- Output (with macros): `"High-carb meals predict a stress spike in you ~90 minutes later."`
- Output (without macros): `"Eating in the 2 hours before 2PM correlates with your stress pattern."`

### Pass 3 — Sleep Debt Cascade
- Poor sleep night (score <60 OR hours <6): track downstream focus (next 24h), stress (next 12h), valence (same day)
- Builds user-specific sleep-debt delta signature
- Minimum: 5+ poor-sleep nights with next-day data
- Output: `"You slept 5.2h last night. On past short-sleep days, your valence drops 0.4 points by 3PM. It's 2:45PM — that window is now."`

### Pass 4 — HRV→Valence Coupling
- Correlates morning HRV against afternoon valence (12PM–6PM window)
- Pearson r threshold: 0.4
- Minimum: 14+ mornings with both HRV and afternoon valence data
- HRV sourced from `health_samples` table where `metric = 'hrv_sdnn'`; pass skipped if fewer than 14 HRV readings available
- Output: `"Your morning HRV was 28ms (your low range). On days like this, your mood tends to dip after 4PM. Plan light work after 4."`

### Pass 5 — Weekly Rhythm
- Groups readings by day-of-week; computes mean stress/valence per day
- Fires if any day shows >1.3x weekday baseline stress
- Minimum: 3+ occurrences of the target day with data
- Output: `"Sundays show elevated stress in your data — 1.8x your weekday baseline. This is a pattern, not a bad day."`

**Discovery confidence** is always shown: `"Pattern found across 12 of your last 18 similar days."`

**Caching:** Results cached for 6 hours in localStorage(`ndw_pattern_cache`). Re-run only when new data arrives or cache expires. Results also written to Supabase `user_patterns` (upsert by `userId + passType` — see schema).

**Privacy gate:** `user_patterns` Supabase writes are gated by `getSupabaseIfAllowed()`. Pattern computation still runs locally; only remote persistence is blocked in privacy mode.

---

## Module 3: EmotionTaxonomy

**File:** `client/src/lib/insight-engine/emotion-taxonomy.ts`

### Layer 1 — ML Base
EEG valence (−1→+1) + arousal (0→1) places the state in Russell's circumplex. The 6-class ML label becomes a *starting suggestion*, not a final answer.

### Layer 2 — Preset Vocabulary (64 emotions, 4 quadrants × 16)
```
High Arousal + Positive (16):
  excited, inspired, euphoric, motivated, awe, flow, electric, energized,
  playful, confident, fierce, bold, grateful, proud, radiant, joyful

High Arousal + Negative (16):
  overwhelmed, anxious, scattered, wired, dread, rage, panicked, frantic,
  irritated, restless, tense, on-edge, desperate, trapped, chaotic, alarmed

Low Arousal + Positive (16):
  calm, content, grateful, nostalgic, tender, serene, fulfilled, peaceful,
  cozy, reflective, safe, gentle, dreamy, open, grounded, clear

Low Arousal + Negative (16):
  sad, hollow, numb, melancholy, grief, drained, detached, empty,
  defeated, hopeless, foggy, withdrawn, invisible, flat, resigned, heavy
```
Multiple selections allowed — emotions blend. Picker shows user's personal vocabulary first.

### Layer 3 — Personal Vocabulary
**Adding a new emotion:**
1. User speaks/types label OR selects from Layer 2
2. App captures **live EEG snapshot from the current session** (not from historical Supabase reads):
   `{ valence, arousal, stress_index, focus_index, alpha_power, beta_power, theta_power, frontal_asymmetry }`
3. If no EEG session is active, fingerprint is created from valence+arousal only (partial fingerprint — band powers set to null, recognition disabled until 3 full fingerprints exist)
4. Stored as `EmotionFingerprint` in Supabase `emotion_fingerprints` (upsert on `userId + label`)

**Future recognition:**
When live EEG reading falls within Euclidean distance threshold of a stored fingerprint (using only the non-null fields in the centroid) → app suggests the personal label: `"You seem scattered — is that right?"`. User confirms (sharpens centroid via running average) or corrects (centroid shifts toward correction).

**Fingerprint structure:**
```typescript
interface EEGSnapshot {
  valence: number;              // always present
  arousal: number;              // always present
  stress_index: number | null;
  focus_index: number | null;
  alpha_power: number | null;   // null if not in active EEG session
  beta_power: number | null;
  theta_power: number | null;
  frontal_asymmetry: number | null;
}

interface EmotionFingerprint {
  id: string;
  userId: string;
  label: string;           // "scattered", "wired but tired", etc.
  quadrant: "ha_pos" | "ha_neg" | "la_pos" | "la_neg";
  centroid: EEGSnapshot;   // running average of all confirmed readings
  sampleCount: number;
  lastSeen: string;        // ISO timestamp
  isPersonal: boolean;     // true = user-created, false = preset confirmed
}
```

Minimum 3 confirmed readings (sampleCount ≥ 3) before a fingerprint triggers recognition suggestions.

**Quadrant assignment boundaries** (used for both Layer 2 display and fingerprint storage):
- High vs Low Arousal: normalized arousal ≥ 0.5 → High; < 0.5 → Low
- Positive vs Negative Valence: normalized valence ≥ 0.5 → Positive; < 0.5 → Negative
  (normalized valence 0.5 = raw valence 0.0 = neutral — slightly negative raw → Negative quadrant)

**Privacy gate:** All `emotion_fingerprints` Supabase writes gated by `getSupabaseIfAllowed()`. Personal vocabulary falls back to localStorage(`ndw_emotion_fingerprints`) when privacy mode is active.

---

## Module 4: DeviationDetector

**File:** `client/src/lib/insight-engine/deviation-detector.ts`

Inputs: current reading from EEG/health sync + BaselineStore snapshot.

```typescript
// Canonical metric keys — used as BaselineStore cell keys, deviation timer keys, and intervention trigger keys.
// Note: HRV Supabase column is "hrv_sdnn" but the canonical key here is always "hrv".
type DeviationMetric = "stress" | "focus" | "valence" | "arousal" | "hrv" | "sleep" | "steps" | "energy";

interface DeviationEvent {
  metric: DeviationMetric;
  currentValue: number;        // normalized 0-1
  baselineMean: number;        // normalized 0-1
  zScore: number;
  direction: "high" | "low";
  durationMinutes: number;     // how long the deviation has been sustained
  baselineQuality: number;     // 0-1, based on baseline sampleCount (distinct from pattern correlationStrength)
  relatedPattern?: {
    passType: string;
    correlationStrength: number; // Pearson r, distinct from baselineQuality
    summary: string;
  };
}
```

**Duration tracking:** Deviation start timestamps stored in `localStorage("ndw_deviation_timers")` as `Record<metric, { startedAt: ISO, zScore: number }>`. On each reading:
- If |zScore| > 1.5 and no timer exists → start timer
- If |zScore| ≤ 1.0 → clear timer (recovery)
- `durationMinutes = (Date.now() - startedAt) / 60000`
- This persists across page refreshes and tab background/foreground cycles (browser suspension freezes JS but localStorage is durable)

Deviation thresholds:
- Real-time tier: |zScore| > 1.5, durationMinutes > 2
- Push tier: |zScore| > 1.5, durationMinutes > 20
- Morning briefing: uses 24h aggregate, no z-score threshold

---

## Module 5: Three-Tier Delivery

### Tier 1 — Real-Time Banner
**Component:** `client/src/components/insight-banner.tsx`
**Location:** Renders on `/brain-monitor` page as a bottom-slide dismissable banner
**Cooldown:** 15 minutes between banners (stored in localStorage, persists across refreshes)

```
┌─────────────────────────────────────────────────────┐
│ ⚡ You shifted into scattered 8 min ago — 2x your   │
│    normal beta/theta ratio.  [Box breathing →]  ✕   │
└─────────────────────────────────────────────────────┘
```

Format: `{personal emotion label if fingerprint matched, else metric name} + deviation context + CTA button + dismiss`

### Tier 2 — Sustained Push Notification
**Trigger:** DeviationEvent with `durationMinutes > 20`
**Delivery:** Web Push API (existing VAPID setup in `server/routes.ts`)
**Content rules:**
- If prior intervention data exists for this user + metric: "Last time: [intervention] brought you back. Try it now →"
- If no prior data: "[Scientific basis, 1 line]. Try it now →"

### Tier 3 — Morning Briefing
**Component:** `client/src/components/morning-briefing-card.tsx`
**Trigger:** First app open after 6AM local time; generated once per calendar date
**API:** `POST /api/morning-briefing` → Express → Claude API (`claude-haiku-4-5-20251001` for cost)
**Rate limit:** Express route enforces 1 request per userId per calendar day using the existing `rate_limit_entries` table in `shared/schema.ts`. Key: `morning_briefing:${userId}:${YYYY-MM-DD}`. On-hit: return 429 with the cached briefing date. This works correctly on Vercel serverless because state is DB-backed, not in-memory.

**Cache:** Stored in `localStorage("ndw_morning_briefing_${userId}")` — user-scoped to prevent cross-user bleed on shared devices. Value: `{ date: "YYYY-MM-DD", content: BriefingResponse }`. The `date` field is always UTC (`new Date().toISOString().slice(0,10)`). On app open: if `date === today's UTC date`, render from cache without calling API. If stale, call API and update cache. Client and server must use the same UTC date string for the cache key and rate limit key to stay consistent.

**Request body:**
```typescript
interface BriefingRequest {
  sleepData: {
    totalHours: number | null;
    deepHours: number | null;       // null if health platform not connected
    remHours: number | null;        // null if health platform not connected
    efficiency: number | null;      // null if health platform not connected
    dataAvailability: "full" | "total_only" | "none";
  };
  morningHrv: number | null;        // null if no health platform
  hrvRange: { min: number; max: number } | null;
  emotionSummary: {
    readingCount: number;
    avgStress: number;
    avgFocus: number;
    avgValence: number;              // normalized 0-1
    dominantLabel: string;
    dominantMinutes: number;
  };
  patternSummaries: string[];        // plain-text summaries from PatternDiscovery (empty array if none)
  yesterdaySummary: string;          // plain-text: "{dominant emotion} for {N}min, avg stress {s}, avg focus {f}. {N} interventions completed."
                                     // Generated client-side from localStorage emotion history filtered to previous calendar day.
                                     // Empty string if no prior-day data exists.
}
```

**Prompt:** The Claude prompt is conditional on `sleepData.dataAvailability`:
- `"full"`: includes all sleep stage details
- `"total_only"`: "Sleep duration: {hours}h (stage data unavailable — health platform not connected)"
- `"none"`: omits sleep section entirely; prompt notes "no sleep data available today"

**Response:**
```typescript
interface BriefingResponse {
  stateSummary: string;          // 3 sentences
  actions: [string, string, string]; // exactly 3
  forecast: { label: string; probability: number; reason: string };
}
```
**Structured output:** The Claude API call uses `response_format: { type: "json_object" }` (JSON mode) with the prompt instructing it to return valid JSON matching `BriefingResponse`. Server parses with `JSON.parse()` and validates that `actions.length === 3`. On parse failure or validation error: return a 500 with a hardcoded fallback briefing derived from the raw metrics (no LLM), so the user always gets a morning card.
```

---

## Module 6: InterventionLibrary

**File:** `client/src/lib/insight-engine/intervention-library.ts`

Each intervention: trigger conditions, duration bucket, description, deeplink, effectiveness score.

### 2-Minute Resets
| Trigger | Intervention | Deeplink |
|---------|-------------|----------|
| High beta >1.5x baseline | 4-4-4-4 box breathing | `/biofeedback` |
| Scattered (high arousal + negative valence) | Cold water on wrists + 5 slow breaths | Inline |
| Low focus + high alpha suppression | Shake body for 60 seconds | Inline |
| Valence dip + low arousal | Send one message to someone you like | Inline |

### 5-Minute Resets
| Trigger | Intervention | Deeplink |
|---------|-------------|----------|
| Stress sustained >20min | Guided coherent breathing (5s in / 5s out) | `/biofeedback` |
| Brain fog + recovered body | Open-focus meditation — defocus eyes | `/neurofeedback` |
| Overwhelmed emotion label | Brain dump — voice note or text | `/ai-companion` |
| Low mood + good energy | Walk outside, no phone | Inline |
| Wired but tired | 4-7-8 breathing + progressive muscle relaxation | `/biofeedback` |

### 20-Minute Planned Actions (morning briefing only)
| Trigger | Intervention | Deeplink |
|---------|-------------|----------|
| Low HRV + poor sleep | Easy movement only — no intense training | `/health-analytics` |
| Sleep debt cascade predicted | 20-min nap window flagged (1-3PM) | Inline |
| Low REM + low valence forecast | Journaling to process residue | `/dreams` |
| Consecutive high-stress days | Social connection — schedule something | Inline |
| Weekly rhythm stress day | Front-load creative work before 11AM | `/neurofeedback` |

**Effectiveness tracking persistence:**
When a CTA is tapped, record to `localStorage("ndw_intervention_pending")`:
```typescript
Record<interventionId, { tappedAt: ISO; metric: string; baselineZScore: number }>
```
On each reading update, check if any pending intervention is ≥25 minutes old. Read current z-score; if recovered (|zScore| reduced by >0.5), mark effective and append to `localStorage("ndw_intervention_results")`. Pending entries older than 2 hours without recovery are marked ineffective and cleared. Results summary (count, effectiveness rate per intervention) computed on-demand from `ndw_intervention_results`.

---

## InsightEngine Public API

**File:** `client/src/lib/insight-engine/index.ts`

```typescript
export class InsightEngine {
  /** Feed a new EEG/health reading into the engine */
  ingest(reading: NormalizedReading): void;

  /** Returns DeviationEvents for real-time banner (called on each EEG frame) */
  getRealTimeInsights(): DeviationEvent[];

  /** Returns stored insights from pattern discovery (called on page load) */
  getStoredInsights(): StoredInsight[];

  /** Returns the morning briefing if generated today, else null */
  getMorningBriefing(): BriefingResponse | null;

  /** Triggers morning briefing generation (calls /api/morning-briefing) */
  generateMorningBriefing(): Promise<BriefingResponse>;

  /** Adds a user emotion label with optional live EEG snapshot */
  labelEmotion(label: string, eegSnapshot?: EEGSnapshot): Promise<EmotionFingerprint>;

  /** Returns all fingerprints for the current user */
  getFingerprints(): EmotionFingerprint[];

  /** Records that an intervention CTA was tapped */
  recordInterventionTap(interventionId: string, metric: string): void;
}

interface NormalizedReading {
  stress: number;       // 0-1
  focus: number;        // 0-1
  valence: number;      // 0-1 (normalized via (raw+1)/2)
  arousal: number;      // 0-1
  energy?: number;      // 0-1
  hrv?: number;         // raw ms — BaselineStore normalizes internally
  sleep?: number;       // raw score 0-100 — BaselineStore normalizes internally
  steps?: number;       // raw step count — BaselineStore normalizes internally
  source: "eeg" | "health" | "voice";
  timestamp: string;
}

// Shared union — use this exact type for both StoredInsight.category and user_patterns.passType
// to guarantee upsert key alignment. Never use raw string literals in either location.
type PassType = "time_bucket" | "food_lag" | "sleep_cascade" | "hrv_valence" | "weekly_rhythm";

interface StoredInsight {
  id: string;
  category: PassType;
  priority: "high" | "medium" | "low";
  headline: string;
  context: string;
  action: string;
  actionHref: string;
  correlationStrength: number; // Pearson r from PatternDiscovery (same field as user_patterns.correlationStrength)
  discoveredAt: string;
}
```

---

## New Database Schema

Two new tables (Supabase migrations via `drizzle-kit push`):

```typescript
// user_patterns — discovered correlations per user
export const userPatterns = pgTable("user_patterns", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  passType: text("pass_type").notNull(), // "time_bucket" | "food_lag" | "sleep_cascade" | "hrv_valence" | "weekly_rhythm"
  patternData: jsonb("pattern_data").notNull(),
  correlationStrength: real("correlation_strength").notNull(),
  sampleCount: integer("sample_count").notNull(),
  lastComputed: timestamp("last_computed").defaultNow().notNull(),
  isActive: boolean("is_active").default(true),
}, (table) => [
  uniqueIndex("user_patterns_user_pass_idx").on(table.userId, table.passType),
]);

// emotion_fingerprints — personal emotion vocabulary + EEG signatures
export const emotionFingerprints = pgTable("emotion_fingerprints", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  label: text("label").notNull(),
  quadrant: text("quadrant").notNull(), // "ha_pos" | "ha_neg" | "la_pos" | "la_neg"
  centroid: jsonb("centroid").notNull(), // EEGSnapshot (band powers may be null)
  sampleCount: integer("sample_count").notNull().default(0),
  lastSeen: timestamp("last_seen").defaultNow(),
  isPersonal: boolean("is_personal").default(false),
}, (table) => [
  uniqueIndex("emotion_fingerprints_user_label_idx").on(table.userId, table.label),
]);
```

**Upsert pattern for both tables:**
Client-side writes use the Supabase JS client (returned by `getSupabaseIfAllowed()`):
```typescript
await supabase.from("user_patterns").upsert(data, { onConflict: "user_id,pass_type" });
await supabase.from("emotion_fingerprints").upsert(data, { onConflict: "user_id,label" });
```
The `onConflict` value is a comma-separated string of column names matching the unique index. This is Supabase JS syntax — **not** Drizzle's `.onConflictDoUpdate()` which is server-only. Never use plain `.insert()` for these tables.

---

## New API Endpoint

```
POST /api/morning-briefing
Auth: required (getAuthUserId)
Rate limit: 1 call per userId per UTC calendar date (DB-backed via rate_limit_entries; key: morning_briefing:${userId}:${YYYY-MM-DD-UTC})
Body: BriefingRequest (see above)
Response: BriefingResponse (see above)
Model: claude-haiku-4-5-20251001 (fast + cheap; briefing is ~300 tokens)
```

---

## Files Created / Modified

| File | Action | Purpose |
|------|--------|---------|
| `client/src/lib/insight-engine/baseline-store.ts` | Create | Rolling 7-day z-score baseline |
| `client/src/lib/insight-engine/pattern-discovery.ts` | Create | 5 statistical passes |
| `client/src/lib/insight-engine/deviation-detector.ts` | Create | Deviation events + duration tracking |
| `client/src/lib/insight-engine/emotion-taxonomy.ts` | Create | Open emotion vocabulary + fingerprints |
| `client/src/lib/insight-engine/intervention-library.ts` | Create | Timed interventions + effectiveness tracking |
| `client/src/lib/insight-engine/index.ts` | Create | InsightEngine class + public API |
| `client/src/components/insight-banner.tsx` | Create | Real-time tier UI |
| `client/src/components/morning-briefing-card.tsx` | Create | Morning briefing UI |
| `client/src/components/emotion-picker.tsx` | Create | 64-item picker + personal vocabulary |
| `client/src/pages/insights.tsx` | Modify | Replace rule engine with InsightEngine |
| `client/src/components/brain-coach-card.tsx` | Modify | Feed from InsightEngine output |
| `server/routes.ts` | Modify | Add POST /api/morning-briefing with rate limit |
| `shared/schema.ts` | Modify | Add user_patterns + emotion_fingerprints tables |

---

## What No Other App Can Replicate

1. **Personal baseline per time-bucket** — not population averages; your 2PM is compared to *your* 2PM history
2. **Food→emotion lag discovery** — requires food logs + EEG data simultaneously
3. **EEG-fingerprinted emotion vocabulary** — personal labels anchored in physiology, not just self-report
4. **Brain forecast** — predicts afternoon mood/focus dip from morning EEG + HRV; not possible without brain wave data
5. **Intervention effectiveness tracking** — the app learns which resets actually move your EEG, not just which feel good
