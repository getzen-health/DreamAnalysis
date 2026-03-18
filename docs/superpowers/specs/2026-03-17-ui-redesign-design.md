# AntarAI Mobile UI Redesign — Premium

**Date:** 2026-03-17
**Status:** Approved
**Scope:** Full mobile UI redesign — 5-tab structure, Premium card layout, mobile-only

## Overview

Redesign AntarAI from a cluttered 42-page web app into a clean, score-first mobile app with 5 bottom tabs. Inspired by competitor Health's card-based layout. Mobile-only (Capacitor Android/iOS). Vercel serves API backend only.

## Tab Structure

```
[ Today ]  [ Discover ]  [ 🎙 Mic ]  [ Nutrition ]  [ You ]
```

The mic button is elevated, circular, primary-colored — the single voice check-in entry point.

## Design System

### Colors
- Background: `#0a0e17` (near-black)
- Card background: `#111827` (dark gray)
- Card border: `#1f2937` (subtle)
- Text primary: `#e8e0d4` (warm white)
- Text secondary: `#8b8578` (muted)
- Accent green: `#2dd4a0` → `#059669` (readiness, positive)
- Accent amber: `#f59e0b` → `#f97316` (nutrition, warnings)
- Accent purple: `#a78bfa` (sleep)
- Accent blue: `#60a5fa` (focus)
- Accent red: `#f87171` (stress high, destructive)
- AI insight card: gradient `#0f1f1a` → `#111827` with green border `#1f3a2e`

### Typography
- Font: Inter (system-ui fallback)
- Hero scores: 28-42px, weight 700
- Card titles: 13-14px, weight 600
- Subtext: 10-11px, color `#8b8578`
- Section labels: 11px, uppercase, letter-spacing 0.5px

### Cards
- Border radius: 14px (standard), 18px (hero/featured)
- Border: 1px solid `#1f2937`
- Padding: 14-16px
- No shadows (flat design, borders only)
- Spacing between cards: 10-14px

### Score Visualizations
- Circle scores: SVG arc, 270° sweep, gradient stroke, big number center
- Progress bars: 6px height, rounded, `#1f2937` track
- Mini stat cards: 3-column grid, number + label + optional progress

## Tab 1: Today

### Layout (top to bottom)
1. **Header**: Date + greeting (left), profile avatar circle (right)
2. **Readiness Score Circle**: 160px SVG, score 0-100, teal gradient, subtitle "You're feeling good today"
3. **3 Mini Score Cards**: Mood (emotion label + confidence), Stress (% + level), Focus (% + level) — 3-column grid
4. **AI Insight Card**: Gradient background, sparkle icon, 1-2 sentence personalized insight from voice/health data
5. **Sleep Card**: Duration + quality %, sleep stage bar (deep/light/REM/awake color segments)
6. **Health Metrics**: 2-column grid — Heart Rate (bpm + status), Steps (count + % of goal)
7. **Nutrition Summary**: Calorie progress bar with "X / 2,000 kcal"

### Data Sources
- Readiness score: composite of stress (inverted), focus, sleep quality, valence
- Mood/Stress/Focus: from voice check-in (`localStorage ndw_last_emotion`) or Express `/api/brain/today-totals`
- Sleep: from Health Connect or `/api/health-metrics`
- HR/Steps: from Health Connect
- Nutrition: from Express `/api/food/logs`
- AI Insight: generated from emotion + health data context

### No Voice Input Here
Voice check-in was moved to the center mic tab. Today shows stats only.

## Tab 2: Discover

### Layout (top to bottom)
1. **Header**: "Discover" + "Explore your mind and body"
2. **Featured Card**: Emotion Trends — gradient background, mini sparkline SVG, "7-day mood journey"
3. **2x2 Card Grid**: Inner Energy, AI Companion, Brain Monitor, Dreams — each with emoji icon (28px) + title + subtitle
4. **2x1 Card Row**: Neurofeedback, Sleep Session
5. **Horizontal Scroll "More"**: Insights, Weekly Summary, Sleep Stories, CBT-i — pill-shaped chips

### Cards Link To
Each card navigates to its existing sub-page (inner-energy, ai-companion, brain-monitor, dreams, neurofeedback, sleep-session, insights, weekly-summary, sleep-stories, cbti). Sub-pages get the mobile back button from app-layout.

## Tab 3: Mic (Center Button)

- Circular button (44x44px), elevated -5px above tab bar
- Background: teal gradient
- Tap opens bottom sheet modal with `VoiceCheckinCard`
- After completion: stores result in `localStorage`, saves to Express DB, invalidates all query keys
- Modal dismissible by tapping backdrop

Already implemented in bottom-tabs.tsx.

## Tab 4: Nutrition

### Layout (top to bottom)
1. **Calorie Ring**: 140px SVG, amber gradient, "1,240 of 2,000 kcal"
2. **Macro Cards**: 3-column — Protein (blue), Carbs (amber), Fat (red) — each with grams + progress bar
3. **Craving Analysis Card**: AI-derived from voice emotion — "mindful eating", "stress eating", etc. with explanation
4. **Action Buttons**: "Capture Meal" (primary amber) + "Describe" (outline) — side by side
5. **Today's Meals**: List cards — emoji + name + time + meal type + calories

### Data Sources
- Calories/macros: Express `/api/food/logs/{userId}` aggregated for today
- Craving analysis: from `food_emotion_predictor.py` via ML backend, using voice emotion state
- Meal history: Express `/api/food/logs/{userId}` + `/api/meal-history/{userId}`

## Tab 5: You

### Layout (top to bottom)
1. **Profile Header**: Avatar circle (initials, gradient), name, "Member since" date
2. **Stats Row**: 2-column — Day Streak (fire emoji) + Sessions Total
3. **Activity Section** (grouped list): Session History, Personal Records, Weekly Summary
4. **Connected Section** (grouped list): Google Health Connect (status badge), Muse 2 EEG (status badge) — platform-specific (Android shows Google, iOS shows Apple)
5. **Settings Section** (grouped list): Appearance (dark/light), Notifications, Export Data, Privacy & Data, Help & Feedback
6. **Sign Out Button**: Full-width, outline, red text

### Grouped List Style
- Section label: 11px uppercase, muted color
- Items: background `#111827`, rounded 14px container, items separated by 1px border
- Each item: emoji (18px) + title (14px) + optional right badge + chevron `›`

## Pages to Remove / Consolidate

### Remove entirely (functionality absorbed into tabs)
- `dashboard.tsx` → replaced by Today tab
- `emotion-lab.tsx` → emotion data shown in Today; trends in Discover
- `health-analytics.tsx` → health data shown in Today
- `food-log.tsx` → replaced by Nutrition tab
- `food-emotion.tsx` → craving analysis moved into Nutrition tab
- `settings.tsx` → replaced by You tab
- `daily-brain-report.tsx` → AI insight in Today
- `formal-benchmarks-dashboard.tsx` → dev-only, remove from nav
- `architecture-guide.tsx` → dev-only, remove from nav
- `landing.tsx` → not needed for mobile app
- `intent-select.tsx` → absorbed into onboarding

### Keep as sub-pages (accessible from Discover)
- `inner-energy.tsx`
- `ai-companion.tsx`
- `brain-monitor.tsx`
- `dream-journal.tsx` + `dream-patterns.tsx`
- `neurofeedback.tsx` + `biofeedback.tsx`
- `sleep-session.tsx`
- `insights.tsx`
- `weekly-brain-summary.tsx`
- `session-history.tsx`
- `personal-records.tsx`
- `supplements.tsx`
- `emotional-intelligence.tsx`

### Keep as auth/onboarding flow
- `auth.tsx`, `forgot-password.tsx`, `reset-password.tsx`
- `onboarding.tsx` / `onboarding-new.tsx`
- `welcome-intro.tsx`
- `calibration.tsx`
- `device-setup.tsx`
- `privacy-policy.tsx`

### Keep as research flow (hidden from main nav)
- `research-*.tsx` pages
- `study/*.tsx` pages

## Layout Changes

### Remove
- Sidebar component (`sidebar.tsx`) — mobile doesn't need it
- Desktop header (hidden md:block) — already hidden on mobile
- `NeuralBackground` component — heavy, wastes battery
- All `md:` responsive breakpoints — design for phone width only

### Keep
- Bottom tabs (updated with new 5-tab structure)
- Mobile back button header (app-layout.tsx)
- Pull-to-refresh
- Toast notifications
- Intervention banner

## New Files to Create

| File | Purpose |
|------|---------|
| `pages/today.tsx` | New Today tab (replaces dashboard) |
| `pages/discover.tsx` | New Discover tab (card grid) |
| `pages/nutrition.tsx` | New Nutrition tab (replaces food-log) |
| `pages/you.tsx` | New You tab (replaces settings) |

## Data Flow

```
Voice Check-in (Mic tab)
    ├── localStorage: ndw_last_emotion
    ├── Express: POST /api/emotion-readings/batch
    └── Invalidate: all query keys
         │
         ├── Today tab reads: emotion state, stress, focus
         ├── Discover > Emotion Trends reads: /api/brain/history
         ├── Nutrition > Craving reads: food-emotion predictor
         └── You > Session History reads: /api/sessions

Health Connect (Capacitor plugin)
    └── Today tab reads: HR, HRV, steps, sleep
         └── Also shown in You > Connected section

Food Capture (Nutrition tab)
    ├── Express: POST /api/food/analyze (GPT-5 vision)
    └── Express: GET /api/food/logs (history)
         └── Today tab reads: nutrition summary
```

## Implementation Order

1. Create new page files (today.tsx, discover.tsx, nutrition.tsx, you.tsx)
2. Update App.tsx routes
3. Update bottom-tabs.tsx with new tab structure
4. Update app-layout.tsx (remove sidebar, neural background, desktop header)
5. Wire data sources into new pages
6. Test on device
7. Remove old pages that were fully replaced
8. Build APK
