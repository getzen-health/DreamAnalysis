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

## UI Design

### Today Page Layout Change

Remove the 3 score circles (Recovery/Sleep/Strain) from the top of today.tsx. Replace with:

1. **Inner Score hero gauge** — 160px diameter SVG arc (270-degree sweep), emerald gradient stroke matching the premium design system. Score number centered (42px bold Public Sans), "Inner Score" label below (12px muted).

2. **Tier confidence label** — below the gauge, 10px muted text showing data source tier.

3. **Trend indicator** — 7-day sparkline (small, 56x20px) plus delta vs yesterday ("+5" or "-3").

4. **Tap to expand** — spring animation reveals factor breakdown.

### Score Color Mapping

| Range | Color | Label |
|-------|-------|-------|
| 80-100 | Emerald green (`#34D399`) | Thriving |
| 60-79 | Teal (`#2DD4BF`) | Good |
| 40-59 | Amber (`#F59E0B`) | Steady |
| 0-39 | Red (`#F87171`) | Low |

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
