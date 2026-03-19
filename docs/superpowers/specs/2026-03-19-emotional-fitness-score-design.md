# Emotional Fitness Score (EFS) — Design Spec

**Date:** 2026-03-19
**Status:** Approved
**Author:** Sravya + Claude

## Problem

NeuralDreamWorkshop has 210 ML models, 5+ scoring systems, and deep emotional identity features (genome, twin, granularity, calibration) — all backend-only with no user-facing experience. Users see moment-to-moment emotion readings and daily readiness, but have no lasting measure of their emotional health. There is no "holy shit" moment that makes someone screenshot the app and share it.

Neural age (#447) works for brain metrics. Emotions need their own equivalent — not an age, but a measurable fitness score backed by real biometric data.

## Solution

**Emotional Fitness Score (EFS):** A single 0-100 composite score measuring overall emotional health, computed daily from 5 vital signs. Each vital sign is independently trackable. A daily insight engine surfaces the most surprising personal discovery.

The analogy: **VO2 Max for emotional health.** One number everyone understands, backed by real science, with drill-down into what drives it.

## The Five Emotional Vital Signs

| Vital Sign | What It Measures | Source | Scoring Logic |
|---|---|---|---|
| **Resilience** | How fast you bounce back from negative emotional states | `emotional_genome.py` — `recovery_speed` trait (autocorrelation decay rate). **New code needed:** also compute event-level recovery from `emotionReadings` table — detect valence dips below -0.3, measure time until valence returns to user's rolling baseline. | `score = clamp(recovery_speed_normalized * 60 + event_recovery_score * 40, 0, 100)`. `event_recovery_score`: map median recovery time: ≤5 min → 100, 10 min → 75, 20 min → 50, 30+ min → 25. If no dip events in 14 days, use genome trait alone. 14-day rolling window. |
| **Regulation** | How effectively you manage emotional intensity | `emotion_regulation.py` — per-session `regulation_score` (0-100) and `regulation_success` (boolean). **New code needed:** aggregate endpoint that queries last 14 days of regulation events. | `score = mean(regulation_scores)` from last 14 days, weighted by recency (exponential decay, half-life = 7 days). If < 3 regulation events, use fallback: stress recovery rate from `emotionReadings` (how often high-stress episodes resolve within 15 min). |
| **Awareness** | How accurately you perceive your own emotions | `emotionCalibration` table — `awarenessScore` (0-100). Already computed and stored. | Direct read: `score = awarenessScore`. If multiple calibration records exist, use the most recent. If no calibration record exists, vital is marked `unavailable`. |
| **Range** | How many distinct emotions you can differentiate | `emotional_granularity.py` — returns `granularity_score` (0-1) with sub-scores: `label_diversity`, `taxonomy_depth`, `distribution_evenness`, `icc`. | `score = round(granularity_score * 100)`. Uses the existing composite score which already blends diversity, depth, evenness, and ICC. If < 5 granularity episodes in 14 days, vital is marked `unavailable`. |
| **Stability** | Baseline emotional consistency, resistance to external disruption | `emotional_genome.py` — `emotional_stability` trait (0-1, computed from valence variance). | `score = round(emotional_stability * 100)`. Uses the genome trait directly. The genome computes stability over all available data, not a fixed window — this is intentional (stability is a trait, not a state). If genome not yet computed (< 7 days data), vital is marked `unavailable`. |

### Data Persistence for Vitals

**Important:** `EmotionTrajectoryTracker` and `EmotionRegulationTrainer` are in-memory only with no persistence across server restarts. The EFS compute endpoint must NOT rely on in-memory state. Instead:

- **Resilience**: Query `emotionReadings` table (persisted) for valence time series. Detect negative dips and measure recovery. Supplement with `emotional_genome.recovery_speed` trait.
- **Regulation**: Query `emotionReadings` for stress→calm transitions as a proxy if no explicit regulation session data exists. If regulation sessions are tracked in-memory only, add a new `regulationEvents` table (userId, sessionDate, regulationScore, regulationSuccess, computedAt).
- **Range, Stability**: Already use genome/granularity endpoints which persist their own state.
- **Awareness**: Already persisted in `emotionCalibration` table.

### Partial Vital Data Handling

Not all vitals will be available for every user. Rules:

| Available Vitals | Behavior |
|---|---|
| 5 of 5 | Full EFS score, no caveats |
| 3-4 of 5 | Compute EFS using available vitals only. Re-normalize weights to sum to 1.0. Show "Partial — N of 5 vitals active" badge. Show unavailable vitals as locked cards with "Complete a calibration session to unlock Awareness" guidance. |
| 1-2 of 5 | No composite EFS score. Show individual vitals that are available. Show "Keep tracking to unlock your Emotional Fitness Score" message. |
| 0 of 5 | Show onboarding: "Do your first voice check-in to start building your Emotional Fitness profile." |

## Composite Score Formula

```
EFS = (Resilience × 0.25) + (Regulation × 0.20) + (Awareness × 0.25) + (Range × 0.15) + (Stability × 0.15)
```

**Weight rationale:**
- Awareness (0.25): Most surprising insight for users. Suppressor/amplifier discovery is the "holy shit" moment.
- Resilience (0.25): Most actionable. Users can actively train this. Improvement is motivating.
- Regulation (0.20): Core emotional competency. Directly tied to mental health outcomes.
- Range (0.15): Differentiator — no other app measures emotional granularity.
- Stability (0.15): Important but less actionable in the short term.

## Color Thresholds

| Score Range | Color | Label |
|---|---|---|
| 70-100 | Green | Strong |
| 40-69 | Amber | Developing |
| 0-39 | Red | Needs Attention |

## Minimum Data Requirements

| Data Available | Behavior |
|---|---|
| < 3 days of check-ins | No score. Show "Building your profile" progress bar with percentage. |
| 3-7 days | Compute with "Early estimate" badge. Wider confidence interval displayed. |
| 7+ days | Full confidence score. No badge. |

**Definition of "check-in":** Any data-producing interaction that creates an `emotionReadings` row: voice check-in, EEG session, or manual mood log. A "day of check-ins" means at least one such event on that calendar date (UTC). The minimum 3-day requirement ensures at least 3 distinct calendar days with data, not 3 events on the same day.

## API Design

### Endpoint: `GET /emotional-fitness/{user_id}`

GET is correct — this is a read-heavy operation that returns cached daily scores. Use `?force=true` query param to trigger recomputation.

**Orchestration flow:**
1. Check `emotionalFitnessScores` table for today's cached score (by userId + date). If exists and `force` is not set, return cached response.
2. Query `emotionReadings` table for last 14 days of valence data → compute Resilience (detect dips, measure recovery time, blend with genome recovery_speed)
3. Query regulation events (from `regulationEvents` table if exists, else infer from `emotionReadings` stress→calm transitions) → compute Regulation
4. Query latest `emotionCalibration` record → pull Awareness (`awarenessScore`)
5. Call emotional granularity endpoint → get Range (`granularity_score * 100`)
6. Call emotional genome endpoint → get Stability (`emotional_stability * 100`)
7. Mark unavailable vitals (insufficient data per the rules above)
8. Compute weighted EFS from available vitals (re-normalize weights if < 5 vitals)
9. Compute trend: query score from 30 days ago in `emotionalFitnessScores`. If no 30-day-ago score, try 14 days. If no prior score at all, trend is `null`.
10. Generate daily insight (highest-priority available, respecting rotation rule)
11. Upsert into `emotionalFitnessScores` table (unique on userId + date)
12. Return full response

**Caching:** Score is computed on first request of the day and cached. "Day" is determined by UTC date — simpler than local timezone, avoids needing user timezone storage. The score represents a rolling 14-day window, so UTC vs local date difference (at most ±1 day) is negligible. Force recompute via `?force=true`.

### Error / Insufficient Data Response

When user has insufficient data (< 3 days of any check-in activity):

```json
{
  "score": null,
  "confidence": "building",
  "progress": {
    "daysTracked": 1,
    "daysRequired": 3,
    "percentage": 33,
    "message": "Keep tracking for 2 more days to unlock your Emotional Fitness Score"
  },
  "vitals": {
    "resilience": { "score": null, "status": "unavailable", "unlockHint": "Track for 3+ days" },
    "regulation": { "score": null, "status": "unavailable", "unlockHint": "Track for 3+ days" },
    "awareness": { "score": null, "status": "unavailable", "unlockHint": "Complete a calibration session" },
    "range": { "score": null, "status": "unavailable", "unlockHint": "Do 5+ check-ins" },
    "stability": { "score": null, "status": "unavailable", "unlockHint": "Track for 7+ days" }
  },
  "dailyInsight": null,
  "computedAt": null
}

### Response Schema

```json
{
  "score": 87,
  "color": "green",
  "label": "Strong",
  "confidence": "full",
  "trend": {
    "direction": "up",
    "delta": 12,
    "period": "30d"
  },
  "vitals": {
    "resilience": {
      "score": 82,
      "history": [{"date": "2026-03-12", "score": 78}, ...],
      "insight": "You recover from stress in 12 min, down from 28 min last month"
    },
    "regulation": {
      "score": 91,
      "history": [...],
      "insight": "Your regulation success rate is 88% this week"
    },
    "awareness": {
      "score": 64,
      "history": [...],
      "insight": "You're a Suppressor — you rate yourself calmer than your brain shows"
    },
    "range": {
      "score": 78,
      "history": [...],
      "insight": "You differentiated 18 distinct emotions this week"
    },
    "stability": {
      "score": 85,
      "history": [...],
      "insight": "Your emotional baseline has been consistent for 10 days"
    }
  },
  "dailyInsight": {
    "text": "You're a Suppressor — you rate yourself calmer than your brain shows.",
    "type": "awareness_gap",
    "actionNudge": "Try naming your emotion out loud during your next check-in"
  },
  "computedAt": "2026-03-19T08:00:00Z"
}
```

**Response field notes:**
- `color` and `label`: Derived at response time from score thresholds, not stored in DB.
- `trend`: Computed at response time by querying prior scores in `emotionalFitnessScores` table. Period is always 30 days. If no score exists 30 days ago, try 14 days. If no prior score at all, `trend` is `null`.
- `history` arrays: Return last 14 days by default. The UI timeline (30/60/90 day toggles) makes separate queries with `?days=30|60|90` param.
- `actionNudge`: Generated deterministically from `dailyInsightType` using a static lookup table (one nudge per insight type). Not stored — regenerated on each response from the stored type.

## Database Schema

### New table: `emotionalFitnessScores`

| Column | Type | Description |
|---|---|---|
| id | uuid | Primary key |
| userId | uuid | Foreign key to users |
| date | date | Score date (one per day per user) |
| score | integer | Composite EFS (0-100) |
| resilience | integer | Resilience vital (0-100) |
| regulation | integer | Regulation vital (0-100) |
| awareness | integer | Awareness vital (0-100) |
| range | integer | Range vital (0-100) |
| stability | integer | Stability vital (0-100) |
| dailyInsightText | text | Generated insight text |
| dailyInsightType | text | Insight category (awareness_gap, improvement, range_expansion, pattern, milestone, transformation) |
| confidence | text | full, early_estimate |
| computedAt | timestamp | When the score was computed |

**Unique constraint:** `(userId, date)` — one score per user per day.

**Implementation note:** This table must be added to `shared/schema.ts` using Drizzle ORM (matching the existing schema pattern) and migrated via `drizzle-kit push`. The Markdown table above defines the logical schema; the actual implementation is Drizzle TypeScript.

## Insight Engine

### Priority-ranked insight categories

| Priority | Type | Trigger Condition | Example |
|---|---|---|---|
| 1 | `awareness_gap` | awarenessScore < 50 OR reporterType = suppressor/amplifier | "You're a Suppressor — your brain was stressed 3 times this week but you reported feeling fine" |
| 2 | `improvement` | Any vital up 10+ points in 30 days | "Your resilience jumped 15 points — you now recover from stress in 12 min vs 28 min last month" |
| 3 | `range_expansion` | Range score increased week-over-week | "You differentiated 22 distinct emotions this week — that's your personal best" |
| 4 | `pattern` | Day-of-week or time-of-day pattern detected in any vital | "Your regulation drops every Thursday evening — work stress carrying over?" |
| 5 | `milestone` | Sustained score above threshold for 30 days | "30-day streak: your EFS has been above 70 for a full month" |
| 6 | `transformation` | Large delta from first-ever EFS computation | "You started at 43. You're now 81. That's a transformation." |

### Insight rules
- Never compare to other users. Always compare to YOUR past self.
- One insight per day. Pick the highest-priority available.
- Each insight ends with an actionable nudge (from static lookup by type).
- Rotation: prefer not to show the same type two days in a row. If the highest-priority type was shown yesterday, skip to priority 2. But if ONLY one type triggers (e.g., `awareness_gap` every day for a chronic Suppressor), show it anyway rather than showing nothing. The rotation rule is soft, not hard.
- `improvement` threshold (10+ points in 30 days) is absolute, not percentage-based. This is intentional — a 10-point jump is meaningful at any level on the 0-100 scale.

## UI Design

### Page: `/emotional-fitness`

**Layout (top to bottom):**

#### 1. Hero — The Score
- Large circular arc gauge (270-degree, like Readiness Score but larger)
- Score number centered: large font, bold
- Label below arc: "Emotional Fitness"
- Color fills arc based on score (green/amber/red)
- Trend badge chip: "↑ 12 pts this month" (or ↓ or →)
- Subtitle: "Updated daily from your brain, voice, and self-reports"

#### 2. Five Vital Signs — Card Grid
- 2-column grid (last card spans full width if 5 items, or scrollable row)
- Each card contains:
  - Icon + vital name (e.g., shield icon + "Resilience")
  - Score: "82/100" with color indicator
  - Mini sparkline (14-day, Recharts LineChart)
  - One-line insight text
- Tap card → expands to detail view with full 30/60/90 day trend chart + explanation of what the vital measures + tips to improve

#### 3. Daily Insight Banner
- Highlighted card with accent border
- Insight text (1-2 sentences)
- Action nudge in muted text below
- Share icon in top-right corner
- This is the screenshot-worthy element

#### 4. History Timeline
- Recharts LineChart showing EFS over 30/60/90 days (toggle buttons)
- Tap any data point to see that day's 5 vital signs breakdown in a tooltip or bottom sheet

#### 5. Share Button
- Fixed at bottom or in header
- Generates PNG card via canvas (same pattern as Weekly Brain Summary):
  - Dark background with app branding
  - Large EFS number
  - 5 small horizontal bars for each vital
  - Daily insight text
  - "Tracked by NeuralDreamWorkshop" watermark
- Share sheet: Instagram Stories, iMessage, copy image

### Components to build

| Component | File | Purpose |
|---|---|---|
| EmotionalFitnessPage | `client/src/pages/emotional-fitness.tsx` | Main page |
| EFSHeroScore | `client/src/components/efs-hero-score.tsx` | Arc gauge with score |
| EFSVitalCard | `client/src/components/efs-vital-card.tsx` | Individual vital sign card |
| EFSInsightBanner | `client/src/components/efs-insight-banner.tsx` | Daily insight display |
| EFSHistoryChart | `client/src/components/efs-history-chart.tsx` | Timeline chart |
| EFSShareCard | `client/src/components/efs-share-card.tsx` | PNG export for sharing |
| EFSMiniCard | `client/src/components/efs-mini-card.tsx` | Dashboard widget |

### Vital Card Detail View Content (static per vital)

Each vital card, when tapped, expands to show a detail view. Content is static strings per vital:

| Vital | Explanation | Tips to Improve |
|---|---|---|
| Resilience | "How quickly your emotions return to baseline after a negative event. Measured from your valence dips over the past 14 days." | "Practice 4-7-8 breathing during stressful moments. Regular sleep improves recovery speed." |
| Regulation | "How effectively you manage emotional intensity when it spikes. Based on your stress-to-calm transitions." | "Try the biofeedback exercises in the app. Even 2 minutes of guided breathing strengthens regulation." |
| Awareness | "How accurately you perceive your own emotions. Compares what you report vs what your brain and voice actually show." | "During check-ins, pause before answering. Notice body sensations first, then name the emotion." |
| Range | "How many distinct emotions you can identify and differentiate. Higher range = better emotion regulation." | "When you feel 'bad', try to be more specific: frustrated? disappointed? anxious? lonely? The distinction matters." |
| Stability | "How consistent your emotional baseline is day to day. Higher stability means less emotional volatility." | "Consistent sleep schedule, regular meals, and daily routines all contribute to emotional stability." |

### EFS Mini Card Spec

The `EFSMiniCard` component for the dashboard:
- Height: same as other dashboard cards (compact)
- Content: EFS score number (large), trend arrow (↑/↓/→) with delta, "Emotional Fitness" label
- Tap navigates to `/emotional-fitness`
- If score is null (building), show progress ring with percentage instead

### Share Card Reference

The PNG share card uses the same canvas-to-PNG pattern as `client/src/pages/weekly-summary.tsx` (the Weekly Brain Summary export). Reference that file for the implementation approach (Canvas 2D API, `toDataURL()`, share sheet).

## Integration Points

| Location | What Shows | Purpose |
|---|---|---|
| Dashboard home | EFS mini-card (score + trend arrow) | Daily visibility |
| Daily Brain Report | EFS alongside readiness score | Morning context |
| Weekly Summary PNG | EFS trend line added | Shareable artifact |
| Push notification | Weekly: "Your EFS is 81, up 4 from last week" | Re-engagement |

### Relationship to existing scores
- **Readiness Score** = "How ready are you TODAY?" → daily snapshot, morning-focused
- **EFS** = "How emotionally fit are you OVERALL?" → rolling 14-day window, identity-level
- Readiness is weather. EFS is climate. They complement, not compete.

## Data Dependencies

| Dependency | Status | Required For | New Code Needed |
|---|---|---|---|
| `emotionReadings` table | Running, persisted | Resilience (valence dip detection) | Yes — recovery time computation from stored readings |
| `emotional_genome.py` recovery_speed + stability traits | Running | Resilience, Stability | No — existing traits, just call the endpoint |
| `emotion_regulation.py` | Running (in-memory only) | Regulation | Yes — either persist regulation events to new `regulationEvents` table, or infer from `emotionReadings` stress transitions |
| `emotionCalibration` table + compute endpoint | Running, persisted | Awareness | No — direct read of `awarenessScore` |
| `emotional_granularity.py` | Running | Range | No — use existing `granularity_score` |
| Voice check-ins OR EEG sessions | Running | All (data input) | No |
| `emotionalFitnessScores` table | **New** | EFS storage + history | Yes — Drizzle schema + migration |

**No new ML models needed.** EFS requires new orchestration code (the compute endpoint), a new DB table, and recovery-time computation logic — but no new model training.

## What Makes This Category-Defining

1. **No competitor measures emotional awareness accuracy.** "You're a Suppressor" is an insight no other app can generate.
2. **No competitor tracks emotional granularity.** Telling someone they differentiated 22 emotions vs 8 is a novel concept.
3. **Backed by real biometrics, not quizzes.** Unlike MBTI or enneagram, this is computed from brain/voice/body data.
4. **Always compares to YOUR past self.** Privacy-first, non-competitive, growth-oriented.
5. **The share card is the growth engine.** A screenshot of "Emotional Fitness: 87" with 5 vitals is inherently intriguing to anyone who sees it.

## Out of Scope (for v1)

- Benchmarking against other users
- Recommendations engine (beyond insight nudges)
- Therapist-facing EFS view (future: integrate with FHIR bridge #414)
- EFS-based content personalization (future: feed into recommendation engine)
- Apple Health / Google Fit export of EFS
