# Neural Dream Workshop — Product Vision

> Read this at the start of every session.
> Every feature you build should move one of these needles.
> If it doesn't, don't build it.

---

## The One Sentence

**Your mental readiness score, every morning. From your voice, your health data, and what you consume. EEG is optional.**

That is the wedge. Oura, Whoop, Garmin, and Apple all explain the body first.
This product explains the mind first, with a low-friction voice entry point and
optional EEG depth for users who want lab-grade measurements.

The external story should be:

> Know your mind. No headband required.

---

## Why Someone Opens This App Tomorrow Morning

They don't open it because they want to "track emotions."
Nobody wakes up thinking that.

They open it because they want to know:

> *"What should I do today, and when?"*

That question already exists in their head every morning.
This app should answer it.

The user should be able to get that answer without being blocked on hardware.
Voice and health data are the default acquisition path. EEG is the upgrade path.

---

## The North Star Feature — Daily Brain Report

Every morning, one screen. No scrolling. No charts to interpret.

```
Good morning

Last night
  Deep sleep    2h 14m
  REM           1h 02m
  Dreams        2 episodes detected

Today's forecast
  Peak focus    9:30am – 12:00pm   ← protect this time
  Likely slump  2:30pm –  3:30pm
  Stress risk   moderate

Yesterday's insight
  Focus was 23% higher after your 11am walk.
  Stress spiked 40% during your 1pm call.
  → Try a 5-min walk before afternoon meetings.

Recommended now
  [ Start coherence breathing — 4 min ]
```

This is the product. Everything else is infrastructure to produce this screen.

Default input stack:
- Voice check-in
- Health data
- Sleep/recovery context
- Consumption logs

Premium depth stack:
- Live EEG
- Sleep staging
- Dream detection
- Neurofeedback

---

## The Moment That Makes Users Believe

Every product has one moment that converts a skeptic into a believer.
Ours is this:

**The user does a 10-second voice check-in in the morning, gets a mental
readiness score that matches how the day actually feels, then uses a short
breathing reset and sees the state shift.**

That moment matters because it happens before hardware commitment. The user
gets value on day one. EEG then becomes a depth feature, not a prerequisite.

The strongest product sequence is:
1. Voice predicts state
2. Health context explains why
3. Intervention changes trajectory
4. EEG validates and deepens the model for advanced users

---

## The Retention Loop

```
Measure → Insight → Action → Visible result → Measure again
```

Without this loop there is no product. Just a dashboard with numbers.

| Step | What it looks like in the app |
|------|-------------------------------|
| Measure | Voice check-in, health sync, optional EEG session |
| Insight | "Your readiness drops every day at 2:30pm" |
| Action | "Try a 10-min walk at 2pm tomorrow" |
| Visible result | Stress drops after breathing or a short reset block |
| Measure again | Next day: "Yesterday's walk kept readiness up until 4pm" |

The user who has gone around this loop once will come back tomorrow.
The user who has gone around it five times will never leave.

---

## What Brings Users Back Daily

| Mechanic | Execution |
|----------|-----------|
| Daily anticipation | "What did my brain do last night?" Same pull as step count |
| Prediction that comes true | App says peak focus 10am. User feels sharp at 10am. Hooked. |
| Personal records | "Longest focus streak: 47 min — beat it?" |
| Cause and effect | "4 meditations this week → baseline stress down 18%" |
| Weekly brain summary | Shareable card. Social loop. Brings new users. |

---

## The Three Product Lanes — Pick One

### Lane 1 — Personal Performance
- **Target:** Knowledge workers, students, founders, anyone optimising their work
- **Positioning:** "Understand your mental rhythm the way athletes understand physical recovery"
- **Comps:** Oura Ring, Whoop, Apple Health readiness layers
- **Hook:** Schedule your day around your actual mental energy, not guesswork

### Lane 2 — Meditation & Wellness
- **Target:** 40M meditators in the US who already have Muse or similar
- **Positioning:** "See your meditation actually working in your brain"
- **Hook:** "Your 10-min morning meditation gave you 2 extra hours of low-stress today"
- **Differentiator from Muse app:** They measure during session. We measure all day and show the downstream effect.

### Lane 3 — Corporate / Research
- **Target:** Universities, clinics, corporate wellness programs, therapists
- **Positioning:** Group-level brain insights, session recordings, exportable data
- **Why it's faster:** They already have budget. Some already have hardware. Voice-first onboarding lowers deployment friction.
- **Fastest path to first evidence.** Start here if unsure.

---

## Positioning Rules

When writing public copy, keep these distinctions explicit:

- Primary promise: mental readiness from voice + health + consumption context
- Secondary promise: optional EEG for live brain-state depth
- Internal truth: the platform is multimodal, but the first-time user should not feel blocked by hardware
- Storefront rule: lead with voice-first value, not with sensors, models, or lab language

---

## Honest Assessment — February 2026

Read this before starting any new feature. Be honest about where things are.

```
Core ML / Signal pipeline    █████████░  98%
  Mastoid reref, DASM/RASM, FAA, FMT, 4-sec epochs, BaselineCalibrator all done.
  Food-Emotion module complete (6 states, 4 biomarkers, dietary guidance).
  Mega LGBM now 74.21% CV (9 datasets: DEAP+DREAMER+GAMEEMO+DENS+FACED+SEED-IV+EEG-ER+STEW+Muse-Sub, 163 534 samples).
  PPO RL agent live: adaptive threshold on every /neurofeedback/evaluate call.
    67% reward rate in flow zone (target 40–75%). 21 models total.
  PersonalModelAdapter wired into /analyze-eeg with personal_override blending. ✅
  Parallel ML inference via ThreadPoolExecutor (/analyze-eeg + /simulate-eeg). ✅
  Local ONNX inference: emotion_classifier_model.onnx (2.2 MB) + JS heuristics for sleep/dream. ✅
  TSception CNN (69.00% CV) now active in emotion classifier fallback chain (after DEAP, before heuristics). ✅
  RunningNormalizer: per-user rolling z-score normalizer in eeg_processor.py, 150-frame buffer (~5 min). ✅
  Voice Emotion Fallback (2026-03-04): VoiceEmotionModel (emotion2vec+/iic/emotion2vec_plus_base + LightGBM), ✅
    6-class output, /voice-watch/analyze + /voice-watch/cache + /voice-watch/latest/{user_id} endpoints.
    70-80% accuracy in no-EEG mode. EEG+Voice fusion at 85-90% (70/30 blend).
  Missing: personalization fine-tuning after 5 sessions.

Backend API                  █████████░  96%
  87 endpoints. 18 modular route files (routes.py split done). WebSocket exists.
  Per-user state isolation fixed. Food-emotion + simulation support added.
  RL training endpoint runs in isolated subprocess (no GIL/OpenMP deadlock). ✅
  Parallel inference (ThreadPoolExecutor) on hot paths. ✅
  Prod deployment: Render free tier (neural-dream-ml.onrender.com). ✅
  Spotify OAuth + play + status + disconnect endpoints added. ✅
  Just-in-time push notification trigger (POST /api/notifications/brain-state-trigger). ✅
  Yesterday's Insights endpoint (GET /api/brain/yesterday-insights/:userId). ✅

Frontend                     ██████████  97%
  25 pages exist. All core user flows built and working.
  Daily Brain Report (/brain-report): sleep summary, forecast, yesterday's insight,
    weekly 7-day avg card, recommended action.
  Sleep session mode (/sleep-session): dim-screen mode, tap-to-peek, real stats from DB.
  Weekly brain summary (/weekly-summary): this week vs last, trend arrows, PNG export. ✅
  Intervention library Evidence tab: personal before/after stress bars + science citations. ✅
  Privacy policy (/privacy): 6-section policy, no auth guard. ✅
  Baseline calibration onboarding (/onboarding): fullscreen 3-phase guided calibration.
  Device pairing wizard: Connect Device → /device-setup → /onboarding. ✅
  Dashboard: personal records (peak focus/flow/longest session + "beat it" indicator + gamification). ✅
  Biofeedback: 7 exercises + Music tab (SpotifyConnect auto-play) + Evidence tab. ✅
  Research beta signup (/research/enroll): IRB language removed, beta program framing.
  Vite bundle splitting + React.lazy() on 14 pages (faster load). ✅
  Session history 24-hour timeline strip (Today view — green/orange/cyan session blocks). ✅
  Push notification service worker + Settings subscribe UI (morning reminder to /brain-report). ✅
  401 Vitest tests across 43 files; all passing. ✅
  Voice Emotion Fallback UI (2026-03-04): useVoiceEmotion hook (7s MediaRecorder + backend), ✅
    Emotion Lab amber panel shown when no EEG, Dashboard voice emotion card, Brain Monitor
    signal source badge (EEG/Voice/Health/EEG+Voice). Intervention engine triggers on voice
    emotion (arousal >= 0.7 or valence <= -0.3).

Mobile (Capacitor)           █████████░  88%
  Capacitor 8.1.0 installed. capacitor.config.ts created. ✅
  Safe area insets: viewport-fit=cover, env(safe-area-inset-*) CSS vars. ✅
  iOS HIG touch targets: 44px min height on all interactive elements. ✅
  Haptic feedback: @capacitor/haptics; light/medium/heavy/success/warning/error. ✅
  Local ML inference: ONNX + JS heuristics, no server needed. ✅
  Offline mode: IndexedDB queue (dreams + EEG sessions + health metrics), auto-sync. ✅
  Spotify: OAuth 2.0, auto-play calm/focus on music intervention. ✅
  BLE Muse 2/S: @capacitor-community/bluetooth-le, GATT packet decoder, ring buffers, FAA/stress. ✅
  Apple HealthKit: HR, HRV, respiratory rate, SpO2, sleep stages, steps, calories → /biometrics/update. ✅
  Google Health Connect: steps, HR, calories, mindfulness via capacitor-health. ✅
  Native push (APNs/FCM): @capacitor/push-notifications, token registration, firebase-admin optional. ✅
  Background EEG sleep: Screen Wake Lock (web) + BackgroundRunner (native) + ongoing notification. ✅
  App Store listing: docs/app-store-listing.md — full copy, privacy strings, Info.plist config. ✅
  Missing: cap add ios/android (needs Xcode + JDK), home screen widget (WidgetKit), Siri Shortcuts.

Product thinking             █████████░  90%
  Full measure → insight → action → result loop now in place.
  Daily Brain Report answers "what should I do today, and when?" ✅
  Yesterday's Insight card identifies patterns from previous day's data. ✅
  Personal records give reason to beat previous bests. ✅
  Weekly summary card gives shareable social artifact. ✅
  Intervention library shows evidence + personal before/after. ✅
  Spotify auto-plays calm music when stress is high — closes the loop. ✅
  Missing: App Store submission, pilot study data.

Retention mechanics          ████████░░  82%
  Daily pull: Daily Brain Report gives morning pull. ✅
  Personal records: peak focus/flow/longest session + "beat it" live indicator. ✅
  Weekly brain summary: 7-day avg, PNG export, shareable. ✅
  Cause and effect: biofeedback before/after + brain report correlations. ✅
  Haptic breathing: tactile feedback makes breathing sessions feel real. ✅
  Spotify auto-play: music starts automatically when stress crosses threshold. ✅
  Missing: App Store distribution, pilot user base.

Infrastructure               █████████░  93%
  Frontend on Vercel (dream-analysis.vercel.app).
  ML backend on Render free tier (neural-dream-ml.onrender.com). ✅
  Per-user isolation: fixed. No monitoring. No auth enforcement.
  Bundle optimized: Vite vendor splitting + lazy loading. ✅
  Offline: IndexedDB queue + auto-sync on reconnect. ✅
  PWA: manifest, service worker, installable. ✅
  Cold-start handled gracefully: animated MLWarmupScreen overlay during Render spin-up;
    keep-alive ping every 14 min prevents sleep; mlFetch 3-retry backoff (1s/3s/9s) + 30s timeout. ✅
  ML status dot in sidebar (green/amber/red) with latency tooltip and Reconnect button. ✅
  SimulationModeBanner on emotion-lab + brain-monitor when ML unreachable. ✅
  .env.example + vercel.json env block updated; render.yaml CORS confirmed. ✅
  Vercel API fully working — all /api/* routes functional. ✅
    Fixed: ESM .js extension resolution, lazy-loaded heavy packages (drizzle/openai/schema)
    to prevent cold-start crash, explicit /api/:path* rewrite for catch-all routing.
  Missing: cap add ios/android, App Store, pilot data.
```

**The gap is not technical. The gap is narrative.**
Numbers need to become insights.
Insights need to become actions.
Actions need to produce visible results.
That loop is the entire product.

---

## What Is Broken Right Now (Do Not Ship Without Fixing)

### ~~1. Per-user state isolation~~ ✅ Fixed
~~`_EpochBuffer`, `BaselineCalibrator`, EMA smoothing — all module-level
singletons. User A's brain state bleeds into User B's readings.~~
Fixed: both are now per-user dicts keyed by `user_id`. Thread-safe lazy init.

### ~~2. Baseline calibration has no UX~~ ✅ Fixed
~~The API (`/calibration/baseline/add-frame`) is built and tested.
The `/calibration` page exists but there is no guided 2-min onboarding screen
that walks a new user through the eyes-closed resting baseline.~~
Fixed: `/onboarding` fullscreen 3-phase guided calibration screen built. Dashboard
shows a calibration banner until `BaselineCalibrator` is ready. Device pairing
wizard routes through `/device-setup` → `/onboarding` automatically.

### ~~3. Signal quality is not visible~~ ✅ Fixed
~~The headset might be seated wrong. The app shows emotion readings anyway.~~
Fixed: SQI banners show in emotion-lab. Readings blocked below 40% SQI.
HSI status visible before any reading is displayed.

### ~~4. `epoch_ready` flag is ignored~~ ✅ Fixed
~~The API returns `epoch_ready: false` for the first 4 seconds.~~
Fixed: buffering progress bar shown; emotion predictions blocked until
`epoch_ready: true`.

### ~~5. ML backend is not deployed~~ ✅ Fixed
~~Every demo requires running `uvicorn` locally.~~
Fixed: deployed to Render free tier (neural-dream-ml.onrender.com).

### ~~6. Neurofeedback uses a static, fixed difficulty threshold~~ ✅ Fixed
~~The protocol threshold never changes. Too easy → boredom. Too hard → disengagement.
Neither produces learning.~~
Fixed: PPO RL agent fires on every `/neurofeedback/evaluate` call. Reads 8-dim
session state (score history, reward rate, streak, band ratio, trend, volatility),
samples action (easier / hold / harder), adjusts threshold ±0.05 immediately.
67% live reward rate — in the target flow zone (40–75%). Model: `rl_nf_agent.pt`.

### ~~7. Device pairing UX missing~~ ✅ Fixed
~~No guided flow to connect Muse 2. User must know to go to Settings and
connect manually. No signal quality check before first session starts.~~
Fixed: "Connect Device" in sidebar links to `/device-setup`. Dashboard banner
also links there. `/device-setup` routes into `/onboarding` for guided calibration.

### ~~8. ML backend cold start causes blank screen~~ ✅ Fixed
~~Render free tier spins down after 15 min of inactivity. First request after idle
takes 30–60 seconds — no feedback to the user, app appears broken.~~
Fixed: `MLWarmupScreen` full-screen animated overlay shown during cold start.
`useMLConnection` state machine (idle → connecting → warming → ready | error) manages
the connection lifecycle. Keep-alive ping every 14 min prevents spin-down.
`mlFetch` retries 3× with exponential backoff (1s/3s/9s) + 30s AbortController timeout.
`SimulationModeBanner` on emotion-lab and brain-monitor pages when ML is unreachable.

### ~~9. EEG signal drift not corrected within session~~ ✅ Fixed
~~Band-power features drift as electrodes warm up and impedance shifts over a session.
Population-average normalization from training data doesn't account for within-session drift.~~
Fixed: `RunningNormalizer` per-user rolling z-score normalizer in `ml/processing/eeg_processor.py`,
buffer of 150 frames (~5 min), wired into `_predict_mega_lgbm()`. Thread-safe, resets per user.
Continuously corrects within-session non-stationarity without requiring a full baseline recalibration.

### ~~10. TSception not wired into inference~~ ✅ Fixed
~~TSception architecture (best for 4-ch asymmetry EEG) was trained and saved but never
integrated into the live inference chain.~~
Fixed: TSception CNN (69.00% CV, 19 800 training epochs) inserted into emotion classifier
fallback chain after DEAP LGBM models, before feature heuristics. Activates when epoch
buffer has ≥1024 samples (4 seconds at 256 Hz). Model file: `tsception_emotion.pt`.

### ~~11. Vercel API functions crashing (FUNCTION_INVOCATION_FAILED)~~ ✅ Fixed
~~All `/api/*` routes on dream-analysis.vercel.app returned `FUNCTION_INVOCATION_FAILED`.
The serverless functions were crashing at cold-start before any request could be handled.~~
Fixed three root causes:
1. `"type": "module"` in `package.json` makes all files ESM — local relative imports
   in `api/[...path].ts` and `api/_lib/auth.ts` needed explicit `.js` extensions.
2. Heavy packages (`drizzle-orm/neon-http`, `openai`, `shared/schema`) were imported
   at module level, crashing on cold start. Moved to lazy `loadModules()` with dynamic
   `import()` called inside the request handler.
3. Multi-segment paths (`/api/auth/register`, `/api/health-metrics/1`) returned Vercel
   404 because specific handler stubs in `api/auth/`, `api/dreams/`, etc. registered
   Vercel routes but weren't deployed (Hobby plan 12-function limit). Deleted all 21
   stub files and added explicit `{ source: "/api/:path*", destination: "/api/[...path]" }`
   rewrite in `vercel.json`.
All endpoints verified working: ping, auth, dreams, health-metrics, emotions, ai-chat.

### ~~12. Page tests have stale copy assertions~~ ✅ Fixed
~~~20 Vitest tests fail due to copy changes in Daily Brain Report, settings, and study pages.~~
Fixed: all 43 test files, 401 tests passing. Updated assertions in bottom-tabs, onboarding,
insights, health-analytics, and session-history to match current component copy.

---

## The Build Order That Actually Matters

### Phase 0 — Make it not embarrassing ✅ COMPLETE
- [x] Deploy ML backend to Render (neural-dream-ml.onrender.com)
- [x] Fix per-user state isolation (epoch buffer + baseline cal now per-user dicts)
- [x] Restore all hidden pages: Inner Energy/Chakra, Brain Connectivity, Dream Patterns,
      Health Analytics, Insights — all back in routing and sidebar
- [x] Show signal quality / HSI before first reading (SQI banners in emotion-lab, blocked <40%, warned 40-60%)
- [x] Show "calibrating…" until `epoch_ready: true` (buffering progress bar + blocks on emotionReady)
- [x] Show confidence on emotion label ("likely relaxed — 68%") (confidence badge in emotion wheel card)

### Phase 1 — Create the aha moment ✅ COMPLETE
- [x] Real-time biofeedback screen during breathing exercise
      (/biofeedback — 7 exercises, expanding circle, live stress chart,
      before/after comparison, works with or without Muse)
- [x] Food & Cravings page (/food — 6 EEG-based eating states, dietary guidance, simulation mode)
- [x] DREAMER + GAMEEMO integration + device-aware gamma masking (69.25% CV)
- [x] Baseline calibration onboarding screen (`/onboarding` — fullscreen 3-phase guided calibration)
- [x] Dashboard calibration banner — shows until BaselineCalibrator is ready
- [x] Device pairing wizard — Connect Device in sidebar → `/device-setup` → `/onboarding`
- [x] Parallel ML inference — `ThreadPoolExecutor` for `/analyze-eeg` + `/simulate-eeg`
- [x] Vite vendor bundle splitting + React.lazy() across 14 pages

### Phase 2 — Create a reason to come back ✅ COMPLETE
- [x] Session history with timeline view
- [x] "Yesterday's insight" card — pattern detection from previous day's health data (`/brain-report`)
- [x] Intervention library — 7 evidence-based exercises with before/after (Physio Sigh, Cyclic Sigh, Power Breath added)
- [x] Personal records — peak focus/flow/longest session + live "beat it" indicator on dashboard

### Phase 3 — Daily pull ✅ COMPLETE
- [x] Daily Brain Report screen (`/brain-report` — morning summary, sleep/dreams/forecast/recommended action)
- [x] Sleep session mode (`/sleep-session` — idle/recording/summary state machine, live staging, dream detection, sleep score)
- [ ] Pattern engine: correlate time-of-day, activities, mental states
- [x] Morning push notification — SW + subscribe UI done; server VAPID trigger pending

### Phase 4 — Growth 🔄 IN PROGRESS
- [x] Weekly brain summary card — 7-day avg stress/focus/sleep with copy button (`/brain-report`)
- [ ] User-correctable labels → feeds personalization
- [ ] Per-user model fine-tuning after 5 sessions
- [ ] Export data (CSV, Apple Health sync)

---

## The Hardware Problem

Muse 2 costs $250. That gates 99% of potential users.

**Watch:** Consumer EEG headbands under $100 are entering market in 2025–2026.
Build the software now. Ride the hardware cost curve down later.
**The moat is the software and the personal data patterns, not the device.**

In the meantime:
- Target users who already own Muse (large community — r/muse has 15K members)
- Build for Muse first, design architecture to swap in any BrainFlow device
- Consider OpenBCI Ganglion ($200, 4-channel, research grade) as alternative

---

## Positioning That Actually Works

| Don't say | Say instead |
|-----------|-------------|
| "Emotion tracking app" | "Personal mental performance tracker" |
| "Brain monitoring" | "Understand your brain's daily rhythm" |
| "EEG analysis" | "Like a fitness tracker, but for your mind" |
| "Stress detection" | "Know when to push and when to recover" |
| "Valence: -0.3" | "Your brain is in recovery mode right now" |

**Reframe everything from measurement to meaning.**

---

## What Success Looks Like in 6 Months

A user opens the app every morning without being reminded.
They make one scheduling decision per week based on their peak focus window.
They have done the breathing biofeedback exercise at least once and felt it work.
They have shared their weekly brain summary at least once.
They have been using it for 30 days and their personalized model accuracy
is above 70%.

That user will pay $15/month and tell three people.
That is the product.

---

## Questions to Ask Before Building Anything

1. Does this move a user through the loop (measure → insight → action → result)?
2. Does this help produce the Daily Brain Report?
3. Does this make the aha moment (live biofeedback) more powerful?
4. Does this give a reason to open the app tomorrow morning?

If the answer to all four is no — don't build it yet.

---

## Regulatory Positioning (March 2026)

**Current classification**: General Wellness device — FDA-exempt under January 2026 guidance.

**What this means for marketing:**
- ✅ "Track your emotional patterns and brain rhythms"
- ✅ "Understand how sleep, food, and supplements affect your focus"
- ✅ "Research-backed methodology for emotional awareness"
- ❌ "Diagnose," "treat," "screen for," or "prevent" any condition
- ❌ "Clinically validated" (requires IRB study + peer review)
- ❌ "FDA approved" or "medical grade"

**Path to DTx (optional, 24–36 months):**
1. Publish peer-reviewed paper on emotion detection accuracy
2. Complete 10-person feasibility pilot with IRB oversight
3. Run small RCT (n=30–50) targeting "stress awareness + breathing intervention"
4. File De Novo submission for "emotion awareness device with behavioral intervention"

Full strategy in `docs/regulatory-strategy.md`.
