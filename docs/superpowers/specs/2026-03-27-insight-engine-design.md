# InsightEngine — Design Spec
**Date:** 2026-03-27
**Project:** NeuralDreamWorkshop
**Status:** Approved by Sravya

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
  - persists to localStorage + Supabase emotion_history
        │
        ▼
  PatternDiscovery                  client/src/lib/insight-engine/pattern-discovery.ts
  - 5 statistical passes (see below)
  - runs as background worker on data update
  - discovered patterns stored in Supabase user_patterns table
        │
        ▼
  DeviationDetector                 client/src/lib/insight-engine/deviation-detector.ts
  - inputs: current reading + BaselineStore
  - outputs: { metric, zScore, direction, duration, confidence }
        │
        ├──▶ RealTimeTier           (EEG streaming → instant banner, cooldown 15min)
        ├──▶ PushTier               (deviation sustained >20min → browser push notification)
        └──▶ MorningBriefingTier    (first open after 6AM → LLM synthesis of last 24h)

  EmotionTaxonomy                   client/src/lib/insight-engine/emotion-taxonomy.ts
  - runs parallel, independent of insight tiers
  - Layer 1: ML base (valence + arousal → circumplex position)
  - Layer 2: 60-item preset vocabulary (4 quadrants)
  - Layer 3: personal vocabulary (user labels + EEG fingerprints)
```

All computation runs **client-side** except the morning briefing LLM call (Claude API via Express `/api/morning-briefing`). No new database tables except `user_patterns` and `emotion_fingerprints`.

---

## Module 1: BaselineStore

**File:** `client/src/lib/insight-engine/baseline-store.ts`

Tracks a rolling 7-day window of readings per metric. Computes mean and standard deviation per 2-hour time bucket (12 buckets × 8 metrics = 96 baseline cells).

```typescript
interface BaselineCell {
  mean: number;
  std: number;
  sampleCount: number;
  lastUpdated: string; // ISO timestamp
}

// Key: `${metric}_${hourBucket}` e.g. "stress_14" = stress readings 14:00-16:00
type BaselineMap = Record<string, BaselineCell>;
```

**Z-score computation:**
```
zScore = (currentValue - cell.mean) / Math.max(cell.std, 0.01)
```

Minimum `sampleCount` before a cell is used: **7** (prevents false deviations from sparse data). Cells with fewer samples fall back to population defaults (stress: mean=0.4, std=0.15; focus: mean=0.55, std=0.18; valence: mean=0.1, std=0.35).

Persists: `localStorage("ndw_baseline_map")` + Supabase `user_settings.alertThresholds` JSONB column (reuses existing column, no migration).

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
- Lag with highest |r| wins; reported with that lag time
- Output: `"High-carb meals predict a stress spike in you ~90 minutes later. You had pasta at 12:30. Watch for it around 2PM."`

### Pass 3 — Sleep Debt Cascade
- Poor sleep night (score <60 OR hours <6): track downstream focus (next 24h), stress (next 12h), valence (same day)
- Builds user-specific sleep-debt delta signature
- Minimum: 5+ poor-sleep nights with next-day data
- Output: `"You slept 5.2h last night. On past short-sleep days, your valence drops 0.4 points by 3PM. It's 2:45PM — that window is now."`

### Pass 4 — HRV→Valence Coupling
- Correlates morning HRV against afternoon valence (12PM-6PM window)
- Pearson r threshold: 0.4
- Minimum: 14+ mornings with both HRV and afternoon valence data
- Output: `"Your morning HRV was 28ms (your low range). On days like this, your mood tends to dip after 4PM. Plan light work after 4."`

### Pass 5 — Weekly Rhythm
- Groups readings by day-of-week; computes mean stress/valence per day
- Fires if any day shows >1.3x weekday baseline stress
- Minimum: 3+ occurrences of the target day with data
- Output: `"Sundays show elevated stress in your data — 1.8x your weekday baseline. This is a pattern, not a bad day."`

**Discovery confidence** is always shown: `"Pattern found across 12 of your last 18 similar days."` Results cached for 6 hours (re-run only when new data arrives). Stored in Supabase `user_patterns` table as JSONB.

---

## Module 3: EmotionTaxonomy

**File:** `client/src/lib/insight-engine/emotion-taxonomy.ts`

### Layer 1 — ML Base
EEG valence (−1→+1) + arousal (0→1) places the state in Russell's circumplex. The 6-class ML label becomes a *starting suggestion*, not a final answer.

### Layer 2 — Preset Vocabulary (60 emotions, 4 quadrants)
```
High Arousal + Positive:  excited, inspired, euphoric, motivated, awe, flow,
                          electric, energized, playful, confident, fierce, bold

High Arousal + Negative:  overwhelmed, anxious, scattered, wired, dread, rage,
                          panicked, frantic, irritated, restless, tense, on-edge

Low Arousal + Positive:   calm, content, grateful, nostalgic, tender, serene,
                          fulfilled, peaceful, cozy, reflective, safe, gentle

Low Arousal + Negative:   sad, hollow, numb, melancholy, grief, drained,
                          detached, empty, defeated, hopeless, foggy, withdrawn
```
Multiple selections allowed — emotions blend. Picker shows user's personal vocabulary first.

### Layer 3 — Personal Vocabulary
**Adding a new emotion:**
1. User speaks/types label OR selects from Layer 2
2. App captures current EEG snapshot: `{ valence, arousal, stress_index, focus_index, alpha_power, beta_power, theta_power, frontal_asymmetry }`
3. Stored as `EmotionFingerprint` in Supabase `emotion_fingerprints` table

**Future recognition:**
When live EEG reading falls within Euclidean distance threshold of a stored fingerprint → app suggests the personal label: `"You seem scattered — is that right?"`. User confirms (sharpens fingerprint) or corrects (updates fingerprint center).

**Fingerprint structure:**
```typescript
interface EmotionFingerprint {
  id: string;
  userId: string;
  label: string;           // "scattered", "wired but tired", etc.
  quadrant: Quadrant;      // inferred from valence + arousal
  centroid: EEGSnapshot;   // average of all confirmed readings
  sampleCount: number;
  lastSeen: string;
  isPersonal: boolean;     // true = user-created, false = preset confirmed
}
```

Minimum 3 confirmed readings before a fingerprint is used for recognition (avoids false suggestions from one data point).

---

## Module 4: DeviationDetector

**File:** `client/src/lib/insight-engine/deviation-detector.ts`

Inputs: current reading from EEG/health sync + BaselineStore snapshot.

```typescript
interface DeviationEvent {
  metric: string;           // "stress" | "focus" | "valence" | "arousal" | "hrv"
  currentValue: number;
  baselineMean: number;
  zScore: number;
  direction: "high" | "low";
  durationMinutes: number;  // how long the deviation has been sustained
  confidence: number;       // 0-1, based on baseline sample count
  relatedPattern?: DiscoveredPattern; // from PatternDiscovery if applicable
}
```

Deviation thresholds:
- Real-time tier: |zScore| > 1.5, sustained >2 minutes
- Push tier: |zScore| > 1.5, sustained >20 minutes
- Morning briefing: uses 24h aggregate, no z-score threshold

---

## Module 5: Three-Tier Delivery

### Tier 1 — Real-Time Banner
**Component:** `client/src/components/insight-banner.tsx`
**Location:** Renders on `/brain-monitor` page as a bottom-slide dismissable banner
**Cooldown:** 15 minutes between banners (prevents spam during long sessions)

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
- If prior intervention data exists: "Last time: [intervention] brought you back. Try it now →"
- If no prior data: "[Scientific basis, 1 line]. Try it now →"

### Tier 3 — Morning Briefing
**Component:** `client/src/components/morning-briefing-card.tsx`
**Trigger:** First app open after 6AM local time, generated once per calendar day
**API:** `POST /api/morning-briefing` → Express → Claude API

**Prompt structure sent to Claude:**
```
System: You are a personal brain-body coach. Be specific, warm, brief.
        Never use filler phrases. Reference the user's own data directly.

User data (last 24h):
- Sleep: {hours}h, {deepHours}h deep, {remHours}h REM, efficiency {pct}%
- Morning HRV: {hrv}ms (user range: {min}-{max}ms)
- Emotion sessions: {count} readings, avg stress {s}, focus {f}, valence {v}
- Dominant state: {label} for {minutes}min
- Discovered patterns active today: {patternSummaries}
- Yesterday afternoon: {summary}

Generate:
1. A 3-sentence brain-body state summary (specific, no generics)
2. Exactly 3 prioritized actions for today (timed, linked to NDW features)
3. One "brain forecast": a specific probability + reason (e.g. "74% chance of afternoon focus dip based on low HRV + short REM")
```

**Response rendered as:** Full-screen card with dismiss. Stored in localStorage so it persists without re-calling Claude on re-open same day.

---

## Module 6: InterventionLibrary

**File:** `client/src/lib/insight-engine/intervention-library.ts`

Each intervention has: trigger conditions, duration bucket (2min / 5min / 20min), description, deeplink, and an effectiveness score (computed from post-intervention metric recovery).

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

**Effectiveness tracking:**
After each intervention CTA is tapped, `InterventionLibrary` records a pending check. 30 minutes later it reads the relevant metric from BaselineStore. If the metric recovered toward baseline (z-score reduced by >0.5), the intervention is marked effective for this user. Interventions with consistently low effectiveness (<30% recovery rate over 5+ uses) are deprioritized in future suggestions.

---

## New Database Schema

Two new tables (Supabase migrations via `drizzle-kit push`):

```typescript
// user_patterns — discovered correlations per user
export const userPatterns = pgTable("user_patterns", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  passType: text("pass_type").notNull(), // "time_bucket" | "food_lag" | "sleep_cascade" | "hrv_valence" | "weekly_rhythm"
  patternData: jsonb("pattern_data").notNull(), // correlation result
  confidence: real("confidence").notNull(),
  sampleCount: integer("sample_count").notNull(),
  lastComputed: timestamp("last_computed").defaultNow().notNull(),
  isActive: boolean("is_active").default(true),
});

// emotion_fingerprints — personal emotion vocabulary + EEG signatures
export const emotionFingerprints = pgTable("emotion_fingerprints", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  label: text("label").notNull(),
  quadrant: text("quadrant").notNull(), // "ha_pos" | "ha_neg" | "la_pos" | "la_neg"
  centroid: jsonb("centroid").notNull(), // EEGSnapshot average
  sampleCount: integer("sample_count").notNull().default(0),
  lastSeen: timestamp("last_seen").defaultNow(),
  isPersonal: boolean("is_personal").default(false),
}, (table) => [
  index("emotion_fingerprints_user_idx").on(table.userId),
]);
```

No other schema changes. `user_patterns` and `emotion_fingerprints` are new. All other data reuses existing tables.

---

## New API Endpoint

```
POST /api/morning-briefing
Auth: required (getAuthUserId)
Body: { userId, sleepData, emotionSummary, patternSummaries, yesterdaySummary }
Response: { headline, stateSummary, actions: string[], forecast: { label, probability, reason } }
Cache: stored in user_settings JSONB, one entry per calendar date, re-generated if stale
```

---

## Files Created / Modified

| File | Action | Purpose |
|------|--------|---------|
| `client/src/lib/insight-engine/baseline-store.ts` | Create | Rolling 7-day z-score baseline |
| `client/src/lib/insight-engine/pattern-discovery.ts` | Create | 5 statistical passes |
| `client/src/lib/insight-engine/deviation-detector.ts` | Create | Deviation events from baseline |
| `client/src/lib/insight-engine/emotion-taxonomy.ts` | Create | Open emotion vocabulary + fingerprints |
| `client/src/lib/insight-engine/intervention-library.ts` | Create | Timed interventions + effectiveness tracking |
| `client/src/lib/insight-engine/index.ts` | Create | Unified `InsightEngine` export |
| `client/src/components/insight-banner.tsx` | Create | Real-time tier UI |
| `client/src/components/morning-briefing-card.tsx` | Create | Morning briefing UI |
| `client/src/components/emotion-picker.tsx` | Create | 60-item picker + personal vocabulary |
| `client/src/pages/insights.tsx` | Modify | Replace rule engine with InsightEngine |
| `client/src/components/brain-coach-card.tsx` | Modify | Feed from InsightEngine output |
| `server/routes.ts` | Modify | Add POST /api/morning-briefing |
| `shared/schema.ts` | Modify | Add user_patterns + emotion_fingerprints tables |

---

## What No Other App Can Replicate

1. **Personal baseline per time-bucket** — not population averages; your 2PM is compared to *your* 2PM history
2. **Food→emotion lag discovery** — requires both food logs and EEG data simultaneously
3. **EEG-fingerprinted emotion vocabulary** — personal labels anchored in physiology, not just self-report
4. **Brain forecast** — predicts afternoon mood/focus dip from morning EEG + HRV, not possible without brain wave data
5. **Intervention effectiveness tracking** — the app learns which resets actually move your EEG, not just which feel good
