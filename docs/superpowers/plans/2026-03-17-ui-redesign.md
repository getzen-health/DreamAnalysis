# Premium UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign AntarAI into a clean 5-tab mobile app (Today | Discover | Mic | Nutrition | You) with Premium card layouts.

**Architecture:** Replace 42-page cluttered web app with 4 new focused pages + updated bottom tabs + simplified app layout. Keep all existing sub-pages accessible from Discover. Mobile-only — remove sidebar, desktop header, neural background.

**Tech Stack:** React 18, TypeScript, Tailwind CSS, wouter routing, TanStack Query, Capacitor

---

## Chunk 1: Design System + Layout Foundation

### Task 1: Update CSS variables to competitor color palette

**Files:**
- Modify: `client/src/index.css`

- [ ] Update dark theme CSS variables to match competitor:
  - `--background`: `hsl(225 40% 4%)` (maps to #0a0e17)
  - `--card`: `hsl(220 30% 9%)` (maps to #111827)
  - `--border`: `hsl(220 25% 14%)` (maps to #1f2937)
  - `--foreground`: `hsl(35 20% 90%)` (maps to #e8e0d4)
  - `--muted-foreground`: `hsl(35 8% 50%)` (maps to #8b8578)
- [ ] Commit: `style: update CSS variables to competitor dark palette`

### Task 2: Strip desktop layout from app-layout.tsx

**Files:**
- Modify: `client/src/layouts/app-layout.tsx`

- [ ] Remove `Sidebar` import and `<Sidebar />` component
- [ ] Remove `NeuralBackground` import and `<NeuralBackground />` component
- [ ] Remove the desktop-only header (`hidden md:block` section)
- [ ] Remove `md:ml-56` from main element
- [ ] Keep: mobile back button header, pull-to-refresh, bottom tabs, intervention banner, offline banner
- [ ] Commit: `refactor: strip desktop UI from app layout — mobile only`

### Task 3: Update bottom-tabs.tsx with new 5-tab structure

**Files:**
- Modify: `client/src/components/bottom-tabs.tsx`

- [ ] Update tabs array: Today (Sun icon, "/"), Discover (Compass icon, "/discover"), Nutrition (Apple icon, "/nutrition"), You (CircleUser icon, "/you")
- [ ] Keep center mic button as-is (already implemented)
- [ ] Update aliases: Today aliases=[], Discover aliases=["/inner-energy","/ai-companion","/brain-monitor","/dreams","/neurofeedback"], Nutrition aliases=["/food","/food-log","/food-emotion"], You aliases=["/settings","/sessions","/records"]
- [ ] Commit: `feat: update bottom tabs to 5-tab competitor layout`

---

## Chunk 2: Today Page

### Task 4: Create today.tsx page

**Files:**
- Create: `client/src/pages/today.tsx`

- [ ] Build the Today page with these sections (top to bottom):
  1. **Header**: date + greeting (left), profile avatar initial circle (right)
  2. **Readiness Score Circle**: 160px SVG arc, 0-100, teal gradient — computed from (100 - stress*100)*0.3 + focus*100*0.3 + sleepQuality*0.2 + (valence+1)*50*0.2
  3. **3 Mini Score Cards**: grid-cols-3 — Mood (emotion label + confidence%), Stress (% + Low/Med/High), Focus (% + level)
  4. **AI Insight Card**: gradient bg #0f1f1a→#111827, green border, sparkle icon, 1-2 sentence insight
  5. **Sleep Card**: duration + quality%, sleep stage color bar
  6. **Health Metrics**: grid-cols-2 — HR (bpm + status), Steps (count + % goal)
  7. **Nutrition Summary**: calorie progress bar

- [ ] Data sources:
  - Voice emotion: read from `localStorage ndw_last_emotion`
  - Today totals: fetch `/api/brain/today-totals/{userId}`
  - Health: from `useHealthSync()` hook
  - Nutrition: fetch `/api/food/logs/{userId}` filtered to today
  - Sleep: from health metrics or `/api/health-metrics/{userId}`

- [ ] All inline styles use competitor colors (#0a0e17 bg, #111827 cards, #1f2937 borders)
- [ ] No voice input — stats only
- [ ] Commit: `feat: create premium Today page`

---

## Chunk 3: Discover + Nutrition Pages

### Task 5: Create discover.tsx page

**Files:**
- Create: `client/src/pages/discover.tsx`

- [ ] Build with sections:
  1. **Header**: "Discover" + subtitle
  2. **Featured Card**: Emotion Trends — gradient bg, mini sparkline SVG, links to `/emotions` or `/insights`
  3. **2x2 Card Grid**: Inner Energy (🧘), AI Companion (🤖), Brain Monitor (🧠), Dreams (🌃) — each navigates via `useLocation`
  4. **2x1 Row**: Neurofeedback (🎯), Sleep Session (😴)
  5. **Horizontal scroll "More"**: Insights, Weekly Summary, Sleep Stories, CBT-i — pill chips

- [ ] Each card: emoji 28px + title 13px/600 + subtitle 10px/#8b8578
- [ ] Card background: #111827, border: 1px solid #1f2937, border-radius: 14px, padding: 16px
- [ ] Commit: `feat: create premium Discover page`

### Task 6: Create nutrition.tsx page

**Files:**
- Create: `client/src/pages/nutrition.tsx`

- [ ] Build with sections:
  1. **Calorie Ring**: 140px SVG, amber gradient, "X of 2,000 kcal"
  2. **Macro Cards**: grid-cols-3 — Protein (#60a5fa), Carbs (#f59e0b), Fat (#f87171) — grams + progress bar
  3. **Craving Analysis Card**: amber gradient border, brain emoji, insight from food-emotion predictor
  4. **Action Buttons**: "Capture Meal" (amber solid) + "Describe" (outline) — flex row
  5. **Today's Meals**: list cards — meal emoji + name + time/type + calories

- [ ] Import and use `FoodCapture` component for camera/barcode flow
- [ ] Fetch food logs from `/api/food/logs/{userId}`, aggregate today's totals
- [ ] Craving: read voice emotion from localStorage, derive craving type
- [ ] Commit: `feat: create premium Nutrition page`

---

## Chunk 4: You Page + Routing

### Task 7: Create you.tsx page

**Files:**
- Create: `client/src/pages/you.tsx`

- [ ] Build with sections:
  1. **Profile Header**: centered avatar circle (72px, gradient, initial), name, "Member since" date
  2. **Stats Row**: grid-cols-2 — Streak (🔥 + count), Sessions (count)
  3. **Activity Section**: grouped list — Session History, Personal Records, Weekly Summary — each navigates
  4. **Connected Section**: grouped list — Google Health Connect / Apple Health (platform-specific), Muse 2 EEG — with status badges
  5. **Settings Section**: grouped list — Appearance (dark/light toggle), Notifications, Export Data, Privacy & Data, Help & Feedback
  6. **Sign Out Button**: full width, outline, red text

- [ ] Grouped list style: section label (11px uppercase #8b8578), items in #111827 rounded-[14px] container, 1px borders between, emoji + title + optional badge + chevron
- [ ] Platform detection via Capacitor.getPlatform() for health section
- [ ] Commit: `feat: create premium You page`

### Task 8: Update App.tsx routes

**Files:**
- Modify: `client/src/App.tsx`

- [ ] Add imports for Today, Discover, Nutrition, You pages
- [ ] Update routes:
  - `/` → `<Today />`
  - `/discover` → `<Discover />`
  - `/nutrition` → `<Nutrition />`
  - `/you` → `<You />`
- [ ] Keep all existing sub-page routes (inner-energy, ai-companion, brain-monitor, etc.)
- [ ] Remove routes for replaced pages: `/emotions` redirect to `/`, `/trends` redirect to `/discover`, `/food` and `/food-log` redirect to `/nutrition`, `/settings` redirect to `/you`
- [ ] Commit: `feat: update routes for 5-tab competitor layout`

---

## Chunk 5: Cleanup + Build

### Task 9: Update tests

**Files:**
- Modify: `client/src/test/pages/dashboard.test.tsx` → test today.tsx
- Modify: `client/src/test/pages/emotion-lab.test.tsx` → remove or redirect
- Modify: `client/src/test/components/bottom-tabs.test.tsx` → update tab labels

- [ ] Update dashboard test to render Today component instead
- [ ] Fix any broken test references
- [ ] Run `npm test -- --run` — all must pass
- [ ] Commit: `test: update tests for new page structure`

### Task 10: Build APK and push

- [ ] Run `npm test -- --run` — verify all pass
- [ ] Run `npm run build`
- [ ] Run `npx cap sync android`
- [ ] Run `JAVA_HOME="..." ./gradlew assembleDebug`
- [ ] Commit APK: `git add -f android/app/build/outputs/apk/debug/app-debug.apk`
- [ ] Push to GitHub
- [ ] Upload APK to v1.1.0-beta release
- [ ] Commit: `chore: rebuild APK with competitor UI redesign`
