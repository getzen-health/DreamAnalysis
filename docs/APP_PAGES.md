# AntarAI -- App Pages Reference

All pages in the current application, organized by category with route status.

---

## Active Pages

### Bottom Tab Pages (Main Navigation)

| # | Page | Route | File | Status | Description |
|---|------|-------|------|--------|-------------|
| 1 | **Today** | `/` | `today.tsx` | Active | Daily overview -- wellness gauge, mood/stress/focus scores, weather, cycle phase context |
| 2 | **Discover** | `/discover` | `discover.tsx` | Active | Emotions graph, emotion timeline, mood insights, navigation to features |
| 3 | **Nutrition** | `/nutrition` | `nutrition.tsx` | Active | Food logging, meal history, vitamins, supplements, GLP-1 tracker, food quality score |
| 4 | **AI Chat** | `/ai-companion` | `ai-companion.tsx` | Active | AI wellness companion chat |
| 5 | **You** | `/you` | `you.tsx` | Active | Profile, streaks, achievements link, connected devices, settings links |

---

### Brain & EEG Pages

| # | Page | Route | File | Status | Description |
|---|------|-------|------|--------|-------------|
| 6 | **Brain Monitor** | `/brain-monitor` | `brain-tabs.tsx` | Active | Live EEG waveforms, band powers, brain age, ML model scores, EEG music |
| 7 | **Brain Connectivity** | `/brain-connectivity` | `brain-connectivity.tsx` | Active | Brain region connectivity analysis |
| 8 | **Neurofeedback** | `/neurofeedback` | `neurofeedback.tsx` | Active | Neurofeedback training with cognitive reappraisal prompts |
| 9 | **Biofeedback** | `/biofeedback` | `biofeedback.tsx` | Active | Meditation, flow, creativity -- guided biofeedback sessions |
| 10 | **Calibration** | `/calibration` | `calibration.tsx` | Active | EEG baseline calibration (2-min resting state) |
| 11 | **Device Setup** | `/device-setup` | `device-setup.tsx` | Active | Muse 2 / Muse S / Synthetic device connection |
| 12 | **Deep Work** | `/deep-work` | `deep-work.tsx` | Active | Pomodoro timer with EEG-enhanced focus tracking |

---

### Health & Wellness Pages

| # | Page | Route | File | Status | Description |
|---|------|-------|------|--------|-------------|
| 13 | **Health** | `/health` | `health.tsx` | Active | Health sync status, Withings/Health Connect data, body metrics tabs |
| 14 | **Health Analytics** | `/health-analytics` | `health-analytics.tsx` | Active | Valence/arousal/stress/focus charts, composite scores |
| 15 | **Wellness** | `/wellness` | `wellness.tsx` | Active | Mood logging, menstrual cycle tracking, energy tracking |
| 16 | **Sleep** | `/sleep` | `sleep.tsx` | Active | Sleep tracking and analysis |
| 17 | **Sleep Session** | `/sleep-session` | `sleep-session.tsx` | Active | Active sleep recording session |
| 18 | **Sleep Music** | `/sleep-music` | sleep-stories component | Active | Sleep stories and calming sounds |
| 19 | **CBTI** | `/cbti` | `cbti-module.tsx` | Active | Cognitive Behavioral Therapy for Insomnia module |
| 20 | **Heart Rate** | `/heart-rate` | `heart-rate.tsx` | Active | Heart rate trends and history |
| 21 | **Steps** | `/steps` | `steps.tsx` | Active | Step count tracking |
| 22 | **Body Metrics** | `/body-metrics` | `body-metrics.tsx` | Active | Weight, body fat, body composition |
| 23 | **Workout** | `/workout` | `workout.tsx` | Active | Exercise and workout tracking |
| 24 | **Inner Energy** | `/inner-energy` | `inner-energy.tsx` | Active | Spiritual energy and chakra visualization |

---

### Trends & Analytics Pages

| # | Page | Route | File | Status | Description |
|---|------|-------|------|--------|-------------|
| 25 | **Stress Trends** | `/stress` | `stress-trends.tsx` | Active | Stress history over time |
| 26 | **Focus Trends** | `/focus` | `focus-trends.tsx` | Active | Focus history over time |
| 27 | **Insights** | `/insights` | `insights.tsx` | Active | AI-generated wellness insights |
| 28 | **Scores Dashboard** | `/scores` | `scores-dashboard.tsx` | Active | All wellness scores in one view |
| 29 | **Daily Brain Report** | `/brain-report` | `daily-brain-report.tsx` | Active | Daily summary of brain activity |
| 30 | **Weekly Summary** | `/weekly-summary` | `weekly-brain-summary.tsx` | Active | Weekly brain and wellness summary |

---

### Dreams & Journaling

| # | Page | Route | File | Status | Description |
|---|------|-------|------|--------|-------------|
| 31 | **Dream Journal** | `/dreams` | `dream-journal.tsx` | Active | Record and analyze dreams with AI |
| 32 | **Food-Emotion** | `/food-emotion` | `food-emotion.tsx` | Active | Correlation between food and emotions |

---

### Special Features

| # | Page | Route | File | Status | Description |
|---|------|-------|------|--------|-------------|
| 33 | **Achievements** | `/achievements` | `achievements.tsx` | Active | Badges, tiers (bronze/silver/gold), progress tracking |
| 34 | **Community** | `/community` | `community.tsx` | Active | Anonymous mood sharing, daily challenges, streaks leaderboard |
| 35 | **Pain Tracker** | `/pain-tracker` | `pain-tracker.tsx` | Active | Pain/migraine logging with EEG theta tracking |
| 36 | **tPBM Session** | `/tpbm` | `tpbm-session.tsx` | Active | Transcranial photobiomodulation session tracking |
| 37 | **Quick Session** | `/quick-session` | `quick-session.tsx` | Active | 5-minute voice + breathing + meditation flow |
| 38 | **Couples Meditation** | `/couples-meditation` | `couples-meditation.tsx` | Active | Dual-device meditation session |
| 39 | **Emotional Intelligence** | `/emotional-intelligence` | `emotional-intelligence.tsx` | Active | EQ training and exercises |
| 40 | **Emotional Fitness** | `/emotional-fitness` | `emotional-fitness.tsx` | Active | Emotion regulation workouts |

---

### User & Settings Pages

| # | Page | Route | File | Status | Description |
|---|------|-------|------|--------|-------------|
| 41 | **Settings** | `/settings` | `settings.tsx` | Active | App preferences, ML backend URL, privacy mode, wellness disclaimer |
| 42 | **Connected Assets** | `/connected-assets` | `connected-assets.tsx` | Active | Device connections (Health Connect, Muse, Oura, WHOOP, Garmin) |
| 43 | **Consent Settings** | `/consent-settings` | `consent-settings.tsx` | Active | Per-modality biometric consent toggles |
| 44 | **Notifications** | `/notifications` | `notifications.tsx` | Active | Notification center |
| 45 | **Export** | `/export` | `export.tsx` | Active | Data export and download |
| 46 | **Help & Feedback** | `/help` | `help.tsx` | Active | Quick start guide, FAQ, feedback form, contact |
| 47 | **Privacy Policy** | `/privacy` | `privacy-policy.tsx` | Active | Full privacy policy with EU AI Act notice (public, no auth) |
| 48 | **Session History** | `/sessions` | `session-history.tsx` | Active | Past EEG/voice session records |
| 49 | **Personal Records** | `/records` | `personal-records.tsx` | Active | Personal bests and milestones |
| 50 | **Supplements** | `/supplements` | `supplements.tsx` | Active | Supplement tracking (standalone page) |
| 51 | **Habits** | `/habits` | `habits.tsx` | Active | Habit tracking |

---

### Research & Study Pages

| # | Page | Route | File | Status | Description |
|---|------|-------|------|--------|-------------|
| 52 | **Research Hub** | `/research` | `research-hub.tsx` | Active | Research study participation |
| 53 | **Research Enroll** | `/research/enroll` | `research-enroll.tsx` | Active | Study enrollment |
| 54 | **Research Morning** | `/research/morning` | `research-morning.tsx` | Active | Morning research session |
| 55 | **Research Daytime** | `/research/daytime` | `research-daytime.tsx` | Active | Daytime research session |
| 56 | **Research Evening** | `/research/evening` | `research-evening.tsx` | Active | Evening research session |
| 57 | **Study Landing** | `/study` | `StudyLanding.tsx` | Active | Study landing page (public, no auth) |
| 58 | **Study Consent** | `/study/consent` | `StudyConsent.tsx` | Active | Study consent form (public, no auth) |
| 59 | **Study Profile** | `/study/profile` | `StudyProfile.tsx` | Active | Study participant profile (public, no auth) |
| 60 | **Study Session** | `/study/session` | `StudySession.tsx` | Active | Active study session (public, no auth) |
| 61 | **Study Stress** | `/study/session/stress` | `StudySessionStress.tsx` | Active | Stress assessment in study (public, no auth) |
| 62 | **Study Food** | `/study/session/food` | `StudySessionFood.tsx` | Active | Food logging in study (public, no auth) |
| 63 | **Study Complete** | `/study/complete` | `StudyComplete.tsx` | Active | Study completion screen (public, no auth) |
| 64 | **Study Admin** | `/study/admin` | `StudyAdmin.tsx` | Active | Study admin dashboard (public, no auth) |

---

### Auth & Onboarding Pages

| # | Page | Route | File | Status | Description |
|---|------|-------|------|--------|-------------|
| 65 | **Welcome** | `/welcome` | `landing.tsx` | Active | Welcome/landing screen (public, no auth) |
| 66 | **Auth** | `/auth` | `auth.tsx` | Active | Login / register (public, no auth) |
| 67 | **Forgot Password** | `/forgot-password` | `forgot-password.tsx` | Active | Password reset request (public, no auth) |
| 68 | **Reset Password** | `/reset-password` | `reset-password.tsx` | Active | Password reset form (public, no auth) |
| 69 | **Onboarding** | `/onboarding` | `onboarding.tsx` | Active | 4-screen onboarding flow (public, no auth) |
| 70 | **Intent Select** | `/intent` | `intent-select.tsx` | Active | Choose: study participant vs explore app |

---

### Developer Pages

| # | Page | Route | File | Status | Description |
|---|------|-------|------|--------|-------------|
| 71 | **Architecture Guide** | `/architecture-guide` | `architecture-guide.tsx` | Active | System architecture documentation (public, no auth) |
| 72 | **Benchmarks** | `/benchmarks` | `formal-benchmarks-dashboard.tsx` | Active | ML model accuracy benchmarks |

---

## Hidden / Redirected Pages

These routes exist in `App.tsx` but redirect to other pages via `RedirectTo`. The original pages were removed.

| Old Route | Redirects To | Status | Reason |
|-----------|-------------|--------|--------|
| `/emotions` | `/brain-monitor` | Hidden | Emotion Lab removed -- functionality merged into Brain Monitor |
| `/mood` | `/brain-monitor` | Hidden | Mood Trends removed -- functionality merged into Brain Monitor |
| `/journal` | `/brain-monitor` | Hidden | Journal alias removed -- functionality merged into Brain Monitor |
| `/onboarding-new` | `/onboarding` | Hidden | Legacy route -- unified into single onboarding flow |
| `/welcome-intro` | `/onboarding` | Hidden | Legacy route -- unified into single onboarding flow |

---

## Route Aliases (Same Component, Different URL)

These routes render the same component as another active route. They exist for URL convenience or backward compatibility.

| Alias Route | Points To | Component | Status |
|------------|-----------|-----------|--------|
| `/food` | `/nutrition` | `nutrition.tsx` | Alias |
| `/food-log` | `/nutrition` | `nutrition.tsx` | Alias |
| `/trends` | `/health-analytics` | `health-analytics.tsx` | Alias |

---

## Summary

| Category | Count |
|----------|-------|
| Active pages | 72 |
| Hidden/redirected routes | 5 |
| Route aliases | 3 |
| **Total routes in App.tsx** | **80** |

*Last updated: 2026-03-23*
