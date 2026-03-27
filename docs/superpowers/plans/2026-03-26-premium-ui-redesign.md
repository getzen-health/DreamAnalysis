# Premium UI Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a premium UX (strength builder, habit streaks, social sharing, recovery interventions, energy timeline, UI polish) while preserving NDW's EEG/voice/emotion differentiators.

**Architecture:** Frontend-heavy — backend already has exercises/workouts/sets/habits/streaks endpoints (113 API routes, 40+ tables). Plan focuses on building rich UI pages that wire to existing API, plus new sharing/intervention components.

**Tech Stack:** React 18 + TypeScript, Tailwind + shadcn/ui, Framer Motion, Recharts, Canvas 2D (sharing cards), wouter routing, TanStack Query

---

## Phase 1: Strength Builder UI (Backend exists — wire it up)

### Task 1: Exercise Library Browser

**Files:**
- Create: `client/src/pages/exercise-library.tsx`
- Create: `client/src/components/exercise-card.tsx`
- Modify: `client/src/App.tsx` (add route)

- [ ] **Step 1: Create ExerciseCard component**

```tsx
// client/src/components/exercise-card.tsx
// Card showing exercise name, muscle groups as badges, equipment icon
// Uses existing premium-card class + shadcn Badge
// Props: exercise (from GET /api/exercises), onSelect callback
```

- [ ] **Step 2: Create ExerciseLibrary page**

```tsx
// client/src/pages/exercise-library.tsx
// Search bar + category filter (strength, cardio, flexibility, compound, isolation)
// Muscle group filter chips (chest, back, legs, shoulders, arms, core)
// Grid of ExerciseCards fetched from GET /api/exercises
// TanStack Query with staleTime: 5 * 60_000
// Tap card → navigates to exercise detail or adds to workout
```

- [ ] **Step 3: Add route in App.tsx**

Add lazy route: `<Route path="/exercises"><ExerciseLibrary /></Route>`

- [ ] **Step 4: Commit**

```bash
git add client/src/pages/exercise-library.tsx client/src/components/exercise-card.tsx client/src/App.tsx
git commit -m "feat: add exercise library browser with search and muscle group filters"
```

### Task 2: Active Workout Tracker

**Files:**
- Create: `client/src/pages/active-workout.tsx`
- Create: `client/src/components/workout-set-row.tsx`
- Create: `client/src/components/rest-timer.tsx`
- Create: `client/src/hooks/use-workout-session.ts`
- Modify: `client/src/App.tsx` (add route)

- [ ] **Step 1: Create useWorkoutSession hook**

```tsx
// client/src/hooks/use-workout-session.ts
// State: currentWorkout (exercises + sets), elapsedTime, restTimer
// Actions: startWorkout, addExercise, logSet(exerciseId, reps, weight, rpe), finishWorkout
// On finish: POST /api/workouts with all sets via POST /api/workouts/:id/sets
// Persists in-progress workout to localStorage (survives app crash)
```

- [ ] **Step 2: Create RestTimer component**

```tsx
// client/src/components/rest-timer.tsx
// Circular countdown timer (60s/90s/120s/custom)
// Auto-starts after logging a set
// Haptic feedback on completion (Capacitor Haptics)
// Premium style: large number in center, progress ring around it
```

- [ ] **Step 3: Create WorkoutSetRow component**

```tsx
// client/src/components/workout-set-row.tsx
// Row: set#, weight input, reps input, RPE selector (6-10), checkmark button
// Shows previous session's weight/reps as ghost text placeholder
// Swipe to delete set
```

- [ ] **Step 4: Create ActiveWorkout page**

```tsx
// client/src/pages/active-workout.tsx
// Header: workout timer (elapsed), finish button
// Exercise list: each exercise shows name + sets logged
// "Add Exercise" button → opens exercise library as sheet
// "Add Set" button per exercise → new WorkoutSetRow
// Rest timer appears between sets
// On finish: summary screen with total volume, duration, PRs hit
```

- [ ] **Step 5: Add route, commit**

```bash
git commit -m "feat: add active workout tracker with set logging, rest timer, and session persistence"
```

### Task 3: Workout History & Personal Records

**Files:**
- Modify: `client/src/pages/workout.tsx` (upgrade from HealthKit-only to full history)
- Create: `client/src/pages/exercise-detail.tsx`
- Create: `client/src/components/pr-badge.tsx`

- [ ] **Step 1: Upgrade workout.tsx**

```tsx
// Replace HealthKit-only display with:
// - "Start Workout" FAB button → navigates to /active-workout
// - Recent workouts from GET /api/workouts/:userId (not just HealthKit)
// - Each workout card: type, duration, total volume (kg), exercises count, PRs hit
// - Tap → workout detail with all sets
```

- [ ] **Step 2: Create ExerciseDetail page**

```tsx
// client/src/pages/exercise-detail.tsx
// Route: /exercises/:id
// Shows: exercise name, muscle groups, instructions, equipment
// History: chart of weight progression over time (Recharts LineChart)
// Personal records: 1RM, max volume, max reps
// "Add to Workout" button
```

- [ ] **Step 3: Create PR badge component**

```tsx
// client/src/components/pr-badge.tsx
// Gold badge with trophy icon for new personal records
// Animated entrance (scale + glow)
// Shows on workout summary when a PR is hit
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: upgrade workout page with full history, exercise detail, and PR badges"
```

### Task 4: Workout Templates

**Files:**
- Create: `client/src/pages/workout-templates.tsx`
- Create: `client/src/components/template-card.tsx`

- [ ] **Step 1: Create WorkoutTemplates page**

```tsx
// Route: /workout-templates
// "My Templates" — GET /api/workout-templates/:userId
// Each template: name, exercise count, estimated duration, muscle groups
// Tap → starts active workout pre-loaded with template exercises
// "Create Template" → build from scratch or save current workout as template
// Prebuilt templates: Push/Pull/Legs, Upper/Lower, Full Body, 5x5
```

- [ ] **Step 2: Commit**

```bash
git commit -m "feat: add workout templates with prebuilt routines and custom template creation"
```

---

## Phase 2: Habit Streak Visuals & Analytics

### Task 5: Habit Streak Heatmap

**Files:**
- Create: `client/src/components/habit-heatmap.tsx`
- Create: `client/src/components/habit-streak-card.tsx`
- Modify: `client/src/pages/habits.tsx` (add heatmap + streak section)

- [ ] **Step 1: Create HabitHeatmap component**

```tsx
// client/src/components/habit-heatmap.tsx
// GitHub-style contribution graph for last 90 days
// Color intensity = completion percentage (0=gray, 25%=light green, 100%=dark green)
// Day cells in 7-row grid (Mon-Sun), month labels on top
// Tap a day → shows which habits were completed
// Data from GET /api/habit-logs/:userId
```

- [ ] **Step 2: Create HabitStreakCard component**

```tsx
// client/src/components/habit-streak-card.tsx
// Shows: current streak (days), longest streak, completion rate (%)
// Fire emoji for active streaks, trophy for record-breaking
// Animated counter for streak number
// Data from GET /api/habit-logs/:userId/streaks
```

- [ ] **Step 3: Integrate into habits page**

```tsx
// Modify habits.tsx:
// Add "Streaks" section at top with HabitStreakCard per habit
// Add "Activity" section with HabitHeatmap (all habits combined)
// Add "Analytics" tab: completion rate chart, best/worst days, habit correlations
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add habit streak heatmap and streak cards with analytics"
```

---

## Phase 3: Social Sharing Cards

### Task 6: Shareable Score Cards

**Files:**
- Create: `client/src/components/share-card-generator.tsx`
- Create: `client/src/components/share-panel.tsx`
- Modify: `client/src/pages/scores-dashboard.tsx` (add share button)
- Modify: `client/src/pages/workout.tsx` (add share after workout)

- [ ] **Step 1: Create ShareCardGenerator**

```tsx
// client/src/components/share-card-generator.tsx
// Canvas 2D PNG generator (extends efs-share-card.tsx pattern)
// Templates:
//   1. DailyOverview — recovery + sleep + strain + stress scores, date
//   2. WorkoutSummary — exercises, total volume, PRs, duration
//   3. WeeklySummary — 7-day score trends, best day, insights
//   4. BrainReport — emotion distribution, focus hours, brain age
//   5. HabitStreak — streak count, heatmap snapshot, completion %
// All: gradient background, scores with color coding, NDW branding, date
// Output: 1080x1920 PNG (Instagram Stories) or 1080x1080 (square)
```

- [ ] **Step 2: Create SharePanel**

```tsx
// client/src/components/share-panel.tsx
// Bottom sheet (shadcn Sheet) with:
// - Preview of share card
// - "Copy Image" button (clipboard API)
// - "Save to Photos" button (Capacitor Filesystem)
// - "Share" button (native share via Capacitor Share plugin)
// - Template selector (daily/workout/weekly/brain/habit)
```

- [ ] **Step 3: Wire share buttons into pages**

```tsx
// scores-dashboard.tsx: share icon in header → opens SharePanel with DailyOverview
// workout.tsx: share button on workout summary → WorkoutSummary template
// habits.tsx: share button → HabitStreak template
// brain-monitor.tsx: share button → BrainReport template
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add social sharing cards with 5 templates and native share integration"
```

---

## Phase 4: Recovery Interventions

### Task 7: Smart Recovery Suggestions

**Files:**
- Create: `client/src/components/recovery-interventions.tsx`
- Create: `client/src/lib/intervention-engine.ts`
- Modify: `client/src/pages/today.tsx` (show interventions when recovery is low)

- [ ] **Step 1: Create intervention engine**

```tsx
// client/src/lib/intervention-engine.ts
// Pure function: getInterventions(scores, healthData) → Intervention[]
// Rules:
//   recovery < 40 → "Rest day recommended", "Prioritize sleep tonight"
//   stress > 70 → "Try 5-min box breathing", "Take a walk outside"
//   sleep < 6h → "Go to bed 30 min earlier tonight"
//   strain > 80 + recovery < 50 → "You're overtraining — active recovery only"
//   hrv dropping 3 days → "Your HRV is trending down — check sleep and stress"
//   focus < 30 → "Try a 10-min meditation to reset", "Reduce screen time"
// Each intervention: { title, description, icon, action?, priority }
```

- [ ] **Step 2: Create RecoveryInterventions component**

```tsx
// client/src/components/recovery-interventions.tsx
// Horizontal scrollable card list (premium style)
// Each card: icon + title + short description + optional "Start" CTA
// CTAs: "Start Breathing" → navigates to neurofeedback, "Log Sleep" → sleep page
// Only shows when there are actionable interventions (not when everything is green)
```

- [ ] **Step 3: Wire into Today page**

```tsx
// today.tsx: add RecoveryInterventions below readiness score
// Only renders when recovery < 60 OR stress > 60 OR sleep < 6h
// Animated entrance (fade + slide up)
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add smart recovery interventions based on scores and health data"
```

---

## Phase 5: Predicted Energy Timeline

### Task 8: Peak Focus Hour Forecast

**Files:**
- Create: `client/src/components/energy-timeline.tsx`
- Create: `client/src/lib/energy-predictor.ts`
- Modify: `client/src/pages/today.tsx` (add timeline)

- [ ] **Step 1: Create energy predictor**

```tsx
// client/src/lib/energy-predictor.ts
// Uses circadian profile (GET /api/brain/circadian-profile/:userId)
// + sleep data (hours slept, wake time)
// + current scores (recovery, stress)
// Outputs: hourly energy forecast for remaining day
// Each hour: { hour: 9, energy: 0.82, label: "Peak Focus", isBest: true }
// Algorithm: circadian acrophase ± adjustment for sleep debt and stress
```

- [ ] **Step 2: Create EnergyTimeline component**

```tsx
// client/src/components/energy-timeline.tsx
// Horizontal bar chart: hours on X axis, energy level on Y (color-coded)
// Green = peak (>70%), amber = moderate (40-70%), gray = low (<40%)
// Current hour highlighted with pulse animation
// "Best window" badge on peak hours
// "Schedule deep work here" tooltip on peak
```

- [ ] **Step 3: Wire into Today page**

```tsx
// today.tsx: add EnergyTimeline below interventions
// "Your Energy Today" section header
// Shows forecast from current hour to end of day
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add predicted energy timeline with peak focus hour forecast"
```

---

## Phase 6: Premium UI Polish

### Task 9: Card Animations & Micro-interactions

**Files:**
- Modify: `client/src/lib/animations.ts` (add new variants)
- Modify: `client/src/index.css` (refine card styles)
- Modify: `client/src/pages/today.tsx` (apply stagger animations)
- Modify: `client/src/pages/scores-dashboard.tsx` (apply card press effects)

- [ ] **Step 1: Add premium animation variants**

```tsx
// animations.ts additions:
// premiumCardVariants: scale(0.98) on tap, 0.2s spring
// premiumStagger: 0.06s delay, fadeInUp from 12px
// premiumScoreReveal: countUp animation for numbers
// premiumPageTransition: slide from right with spring
```

- [ ] **Step 2: Apply to Today page**

```tsx
// Wrap score cards in motion.div with premiumStagger
// Add press effect to all tappable cards (whileTap={{ scale: 0.98 }})
// AnimatedNumber for all score values
// Smooth section transitions
```

- [ ] **Step 3: Apply to Scores Dashboard**

```tsx
// Score cards: premiumCardVariants with stagger
// Energy battery: animated fill on mount
// Trend arrows: animated entrance
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add premium card animations and micro-interactions across dashboard"
```

### Task 10: Learn More Contextual Overlays

**Files:**
- Create: `client/src/components/learn-more-overlay.tsx`
- Modify: various pages (add ? icons that trigger overlays)

- [ ] **Step 1: Create LearnMoreOverlay component**

```tsx
// client/src/components/learn-more-overlay.tsx
// Bottom sheet (shadcn Sheet) with:
// - Title (e.g., "What is Recovery Score?")
// - Description (2-3 sentences)
// - "How it's calculated" expandable section
// - "Tips to improve" bullet list
// - Close button
// Content keyed by metric name (recovery, strain, sleep, stress, etc.)
```

- [ ] **Step 2: Add to score cards**

```tsx
// ScoreCard: add small ? icon in top-right corner
// Tap → opens LearnMoreOverlay for that metric
// Non-intrusive — doesn't clutter main UI
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: add contextual Learn More overlays for all health metrics"
```

---

## Execution Priority

| Phase | Feature | Impact | Effort | Order |
|-------|---------|--------|--------|-------|
| 1 | Strength Builder UI | High | Large | 1st (parallel) |
| 2 | Habit Streak Visuals | High | Medium | 1st (parallel) |
| 3 | Social Sharing Cards | High | Medium | 1st (parallel) |
| 4 | Recovery Interventions | Medium | Small | 2nd |
| 5 | Energy Timeline | Medium | Medium | 2nd |
| 6 | UI Polish | High | Small | 3rd (touches everything) |

**Phase 1 tasks (1-4) can run in parallel** — they touch different files.
**Phase 2 tasks (5-6) can run in parallel** — independent features.
**Phase 3 tasks (7-8) can run in parallel** — different pages.
**Phase 4 tasks (9-10) run last** — they modify files from earlier phases.
