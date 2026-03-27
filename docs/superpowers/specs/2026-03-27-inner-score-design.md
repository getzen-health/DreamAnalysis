# Inner Score — Design Spec

**Date:** 2026-03-27
**Status:** Approved
**Scope:** Single composite "Inner Score" (0-100) shown as the hero metric on the Today page. Adapts formula based on available data sources (voice-only, health+voice, EEG+health+voice).

## Overview

Replace the 3 score circles (Recovery/Sleep/Strain) on the Today page with a single "Inner Score" — one number users check every morning. The score adapts its computation to whatever data sources the user has connected, from voice-only (minimum) through full EEG.

Differentiator vs Oura Readiness / WHOOP Recovery: Inner Score fuses brain/emotion signals (EEG or voice) with health data. No competitor does this.

## Three-Tier Adaptive Computation

### Tier 1 — Voice Only (minimum viable)

| Factor | Weight | Source |
|--------|--------|--------|
| Stress inverse | 40% | Voice emotion analysis |
| Valence (mood) | 40% | Voice emotion analysis |
| Self-reported energy | 20% | Manual mood log or voice arousal |

Confidence label: *"Based on how you sound today"*

### Tier 2 — Health + Voice

| Factor | Weight | Source |
|--------|--------|--------|
| Sleep quality | 35% | Health Connect / Oura |
| HRV trend | 20% | Health Connect / Oura |
| Stress inverse | 20% | Voice emotion |
| Valence | 15% | Voice emotion |
| Activity level | 10% | Steps / calories |

Confidence label: *"Based on your sleep, body, and mood"*

### Tier 3 — EEG + Health + Voice

| Factor | Weight | Source |
|--------|--------|--------|
| Sleep quality | 30% | Health Connect / Oura |
| Brain health | 25% | BrainHealthScore 5-domain model |
| HRV trend | 15% | Health Connect / Oura |
| Stress inverse | 15% | EEG stress index |
| Valence | 10% | EEG frontal alpha asymmetry |
| Activity | 5% | Steps / calories |

Confidence label: *"Based on your brain, body, and mood"*

### Tier Auto-Detection

The system selects the highest available tier automatically:
- If EEG session data exists from today or yesterday → Tier 3
- If Health Connect / Oura data exists → Tier 2
- If only voice check-in or manual mood → Tier 1
- If no data at all → "Building" state (no score computed)

## Migration / Deprecation

Inner Score **replaces** the existing Emotional Fitness Score (EFS) system. The `emotionalFitnessScores` table in `shared/schema.ts` is deprecated — no new data will be written to it. The `efs-hero-score.tsx` component is deprecated and will not be used directly; the building animation pattern is adapted into the new `inner-score-card.tsx`.

The existing `ScoreCircle` components (Recovery/Sleep/Strain) on the Today page are removed and replaced by the single Inner Score hero gauge. The sub-scores become factor bars inside the tap-to-expand breakdown.

No data migration is needed — the `inner_scores` table starts fresh. Old EFS data remains in the DB but is not read by any active UI.

## Partial Data Within a Tier

Each tier has **required** and **optional** factors:

**Tier 1 — Voice Only:**
- Required: at least one of stress or valence (from voice check-in or manual mood)
- Optional: self-reported energy
- If energy missing: redistribute its 20% equally to stress and valence (each becomes 50%)

**Tier 2 — Health + Voice:**
- Required: sleep quality AND at least one of stress/valence
- Optional: HRV trend, activity
- If optional factors missing: redistribute their weight proportionally among present factors
- If sleep quality missing: fall back to Tier 1

**Tier 3 — EEG + Health + Voice:**
- Required: brain health score AND sleep quality
- Optional: HRV trend, stress, valence, activity
- If brain health missing: fall back to Tier 2
- If sleep quality missing: fall back to Tier 2

**Redistribution rule:** When optional factors are absent, their weight is distributed proportionally among the remaining factors. Example: Tier 2 without HRV (20%) → sleep becomes 35/80×100=43.75%, stress 25%, valence 18.75%, activity 12.5%.

## Tier Detection Rules

- Tier detection uses the user's **local date** (midnight to midnight based on browser timezone)
- "Today or yesterday" for EEG means: the current local calendar day or the one before it
- Data staleness: health data older than 48 hours is ignored for tier detection
- Tier is recomputed on each score calculation, not cached separately

## Score Recomputation

The score is recomputed (not just cached for 24h) when:
- A new voice check-in is recorded
- New health data syncs from Health Connect / Oura
- A new EEG session completes
- The user manually requests a refresh (pull-to-refresh on Today page)

Between triggers, the cached score is served. Cache TTL is 4 hours maximum — after that, recompute even without a trigger.

## Normalization Rules

**Self-reported energy** (Tier 1):
- Mood log scale 1-5 → multiply by 20 → 0-100
- Mood log scale 1-10 → multiply by 10 → 0-100
- Voice arousal 0.0-1.0 → multiply by 100 → 0-100
- If both exist: voice arousal takes priority (physiological > self-report)
- If neither exists: use 50 as neutral default

**Stress inverse:** `(1 - stress_index) * 100` where stress_index is 0.0-1.0

**Valence:** `(valence + 1) / 2 * 100` where valence is -1.0 to +1.0 → maps to 0-100

## Error & Edge Cases

**API error responses:**
- No data (building state): `{ "score": null, "state": "building", "cta": "Do a voice check-in to get your Inner Score" }`
- User not found: `404 { "error": "User not found" }`
- Computation failure: `500 { "error": "Score computation failed" }`

**Trend computation:**
- Delta = `today.score - yesterday.score` (null if no yesterday score)
- Sparkline: 7-element array, null for missing days, component renders gaps gracefully
- Fewer than 7 days of history: pad leading positions with null

## Accessibility

- SVG gauge has `aria-label="Inner Score: 72 out of 100, Good"`
- Factor bars have `role="progressbar"` with `aria-valuenow`, `aria-valuemin`, `aria-valuemax`
- Tap-to-expand is keyboard accessible (Enter/Space to toggle)
- Score range labels ("Thriving"/"Good"/"Steady"/"Low") provide color-blind-safe indication
- All colors use CSS variables, never hardcoded hex in components

## Gauge Animation

- On mount: arc animates from 0 to score value over 1.2s with `cubic-bezier(0.34, 1.56, 0.64, 1)` (spring overshoot)
- On score change: arc transitions smoothly over 0.8s
- Building state: pulsing opacity animation (0.4 → 1.0, 2s infinite)

## UI Design

### Today Page Layout Change

Remove the 3 score circles (Recovery/Sleep/Strain) from the top of today.tsx. Replace with:

1. **Inner Score hero gauge** — 220px diameter SVG arc (270-degree sweep), emerald gradient stroke using `var(--primary)`. Score number centered (52px bold, `font-family: var(--font-sans)`), "Inner Score" label below (12px muted).

2. **Tier confidence label** — below the gauge, 10px muted text showing data source tier.

3. **Trend indicator** — 7-day sparkline (small, 56x20px) plus delta vs yesterday ("+5" or "-3").

4. **Tap to expand** — spring animation reveals factor breakdown.

### Score Color Mapping

| Range | Color | Label |
|-------|-------|-------|
| 80-100 | `var(--success)` / emerald green | Thriving |
| 60-79 | `var(--primary)` / teal | Good |
| 40-59 | `var(--warning)` / amber | Steady |
| 0-39 | `var(--destructive)` / red | Low |

### Building State (No Data)

When no data sources are available:
- Arc gauge renders with animated pulse, "—" instead of number
- CTA text: "Do a voice check-in to get your Inner Score"
- Reuse `efs-hero-score.tsx` building animation pattern

### Factor Breakdown (Tap to Expand)

On tap, slides down with Framer Motion spring animation:

1. **AI narrative** (one line) — e.g., "Good sleep is carrying you today, but elevated stress is holding you back."
2. **Factor bars** — 4-5 horizontal bars depending on tier:
   - Each bar: label (12px), progress bar (6px height, rounded), percentage (12px mono)
   - Color: green (>70), amber (40-70), red (<40)
   - Factors shown depend on tier (voice-only shows fewer bars)

### Design System Compliance

All colors from the premium design system CSS variables. No hardcoded hex values in components. Clean flat cards (`rounded-[14px] bg-card border border-border`). Public Sans font. Emerald accent throughout.

## Data Persistence

### New Table: `inner_scores`

```sql
CREATE TABLE inner_scores (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT NOT NULL,
  score INTEGER NOT NULL CHECK (score >= 0 AND score <= 100),
  tier TEXT NOT NULL CHECK (tier IN ('voice', 'health_voice', 'eeg_health_voice')),
  factors JSONB NOT NULL DEFAULT '{}',
  narrative TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_inner_scores_user_date ON inner_scores (user_id, created_at DESC);
```

Score computed once per day at first app open, cached for 24 hours.

### Factors JSON Shape

```json
{
  "sleep_quality": 85,
  "stress_inverse": 58,
  "valence": 71,
  "hrv_trend": 64,
  "activity": 72,
  "brain_health": 68
}
```

Only factors relevant to the tier are included.

## API Endpoints

### `GET /api/inner-score/:userId`

Returns today's Inner Score (computes if not cached):

```json
{
  "score": 72,
  "label": "Good",
  "color": "#2DD4BF",
  "tier": "health_voice",
  "confidence": "Based on your sleep, body, and mood",
  "factors": { "sleep_quality": 85, "stress_inverse": 58, "valence": 71, "hrv_trend": 64, "activity": 72 },
  "narrative": "Good sleep is carrying you today, but elevated stress is holding you back.",
  "delta": 5,
  "trend": [65, 68, 72, 70, 67, 71, 72]
}
```

### `GET /api/inner-score/:userId/history?days=30`

Returns daily score history for trend display:

```json
{
  "scores": [
    { "date": "2026-03-27", "score": 72, "tier": "health_voice" },
    { "date": "2026-03-26", "score": 67, "tier": "voice" }
  ]
}
```

## File Structure

| File | Type | Purpose |
|------|------|---------|
| `client/src/lib/inner-score.ts` | Lib | Score computation (3-tier formula, tier detection, factor calculation) |
| `client/src/components/inner-score-card.tsx` | Component | Hero gauge + tap-to-expand breakdown + building state |
| `client/src/test/components/inner-score-card.test.tsx` | Test | Component render tests |
| `client/src/test/lib/inner-score.test.ts` | Test | Computation unit tests (all 3 tiers, edge cases) |
| `api/[...path].ts` | Modify | Add inner-score routes to Vercel catch-all |
| `server/routes.ts` | Modify | Add inner-score Express routes |
| `shared/schema.ts` | Modify | Add inner_scores table |
| `client/src/pages/today.tsx` | Modify | Replace 3 score circles with Inner Score hero |

## Narrative Generation

The AI narrative is template-based (no LLM call needed for v1):

- Highest factor is the "carrier": "Good sleep is carrying you today"
- Lowest factor is the "drag": "but elevated stress is holding you back"
- If all factors within 10 points: "You're well-balanced across the board today"
- If score delta > +10: "Strong improvement from yesterday"
- If score delta < -10: "Dip from yesterday — check what changed"

## Testing Strategy

- Unit tests for score computation: all 3 tiers, missing data handling, boundary values (0, 50, 100), tier auto-detection
- Component tests: renders score, building state, tap to expand, factor bars, narrative text, color mapping
- API tests: cache behavior, history endpoint, missing user
