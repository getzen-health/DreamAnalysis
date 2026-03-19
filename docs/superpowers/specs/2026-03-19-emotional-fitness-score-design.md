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

| Vital Sign | What It Measures | Source (Already Built) | Scoring Logic |
|---|---|---|---|
| **Resilience** | How fast you bounce back from negative emotional states | `emotion_trajectory.py` — recovery speed from valence dips | Time-to-baseline after negative dips. Faster recovery = higher score. Measured over 14-day rolling window. |
| **Regulation** | How effectively you manage emotional intensity | `emotion_regulation.py` — LPP proxy + frontal theta | Frequency and success rate of regulation attempts. Higher success rate = higher score. |
| **Awareness** | How accurately you perceive your own emotions | `emotionCalibration` table — `awarenessScore` (0-100) | Gap between self-reported and measured valence/arousal. Smaller gap = higher score. Already computed. |
| **Range** | How many distinct emotions you can differentiate | `emotional_granularity.py` — 27-emotion VAD mapping | Count of distinct emotions differentiated in past 14 days. More granularity = higher score. |
| **Stability** | Baseline emotional consistency, resistance to external disruption | `emotional_genome.py` — stability trait | Variance of baseline emotional state over 7 days. Lower variance = higher score. |

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

## API Design

### Endpoint: `POST /emotional-fitness/compute/{user_id}`

**Orchestration flow:**
1. Fetch last 14 days of emotion trajectory → compute Resilience (0-100)
2. Fetch last 14 days of regulation events → compute Regulation (0-100)
3. Fetch latest `emotionCalibration` → pull Awareness (`awarenessScore`, already 0-100)
4. Fetch last 14 days of granularity episodes → compute Range (0-100)
5. Fetch emotional genome stability trait → compute Stability (0-100)
6. Weighted blend → EFS composite score
7. Generate daily insight (highest-priority available)
8. Store in `emotionalFitnessScores` table
9. Return full response

**Caching:** Compute once daily (first request after midnight local time). Cache until next day. Force recompute available via `?force=true` query param.

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
- Each insight ends with an actionable nudge.
- Insights rotate — don't show the same type two days in a row.

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

## Data Dependencies (all already built)

| Dependency | Status | Required For |
|---|---|---|
| Emotion trajectory tracking | Running | Resilience |
| emotion_regulation.py | Running | Regulation |
| emotionCalibration table + compute endpoint | Running | Awareness |
| emotional_granularity.py | Running | Range |
| emotional_genome.py | Running | Stability |
| Voice check-ins OR EEG sessions | Running | All (data input) |

**No new ML models needed.** EFS is purely an orchestration + UI layer on existing infrastructure.

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
