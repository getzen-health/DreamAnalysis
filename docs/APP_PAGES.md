# AntarAI — App Pages Reference

All active pages in the current application (v3.1.0).

---

## Bottom Tab Pages (Main Navigation)

| # | Page | Route | File | Description |
|---|------|-------|------|-------------|
| 1 | **Today** | `/` | `today.tsx` | Daily overview — wellness gauge, mood/stress/focus scores, weather, cycle phase context |
| 2 | **Discover** | `/discover` | `discover.tsx` | Emotions graph, emotion timeline, mood insights, navigation to features |
| 3 | **Nutrition** | `/nutrition` | `nutrition.tsx` | Food logging, meal history, vitamins, supplements, GLP-1 tracker, food quality score |
| 4 | **AI Chat** | `/ai-companion` | `ai-companion.tsx` | AI wellness companion chat |
| 5 | **You** | `/you` | `you.tsx` | Profile, streaks, achievements link, connected devices, settings links |

---

## Brain & EEG Pages

| # | Page | Route | File | Description |
|---|------|-------|------|-------------|
| 6 | **Brain Monitor** | `/brain-monitor` | `brain-tabs.tsx` | Live EEG waveforms, band powers, brain age, ML model scores, EEG music |
| 7 | **Brain Connectivity** | `/brain-connectivity` | `brain-connectivity.tsx` | Brain region connectivity analysis |
| 8 | **Neurofeedback** | `/neurofeedback` | `neurofeedback.tsx` | Neurofeedback training with cognitive reappraisal prompts |
| 9 | **Biofeedback** | `/biofeedback` | `biofeedback.tsx` | Meditation, flow, creativity — guided biofeedback sessions |
| 10 | **Calibration** | `/calibration` | `calibration.tsx` | EEG baseline calibration (2-min resting state) |
| 11 | **Device Setup** | `/device-setup` | `device-setup.tsx` | Muse 2 / Muse S / Synthetic device connection |
| 12 | **Deep Work** | `/deep-work` | `deep-work.tsx` | Pomodoro timer with EEG-enhanced focus tracking |

---

## Health & Wellness Pages

| # | Page | Route | File | Description |
|---|------|-------|------|-------------|
| 13 | **Health** | `/health` | `health.tsx` | Health sync status, Withings/Health Connect data, body metrics tabs |
| 14 | **Health Analytics** | `/health-analytics` | `health-analytics.tsx` | Valence/arousal/stress/focus charts, composite scores |
| 15 | **Wellness** | `/wellness` | `wellness.tsx` | Mood logging, menstrual cycle tracking, energy tracking |
| 16 | **Sleep** | `/sleep` | `sleep.tsx` | Sleep tracking and analysis |
| 17 | **Sleep Session** | `/sleep-session` | `sleep-session.tsx` | Active sleep recording session |
| 18 | **Sleep Music** | `/sleep-music` | sleep-stories component | Sleep stories and calming sounds |
| 19 | **CBTI** | `/cbti` | `cbti-module.tsx` | Cognitive Behavioral Therapy for Insomnia module |
| 20 | **Heart Rate** | `/heart-rate` | `heart-rate.tsx` | Heart rate trends and history |
| 21 | **Steps** | `/steps` | `steps.tsx` | Step count tracking |
| 22 | **Body Metrics** | `/body-metrics` | `body-metrics.tsx` | Weight, body fat, body composition |
| 23 | **Workout** | `/workout` | `workout.tsx` | Exercise and workout tracking |
| 24 | **Inner Energy** | `/inner-energy` | `inner-energy.tsx` | Spiritual energy and chakra visualization |

---

## Trends & Analytics Pages

| # | Page | Route | File | Description |
|---|------|-------|------|-------------|
| 25 | **Stress Trends** | `/stress` | `stress-trends.tsx` | Stress history over time |
| 26 | **Focus Trends** | `/focus` | `focus-trends.tsx` | Focus history over time |
| 27 | **Trends** | `/trends` | `health-analytics.tsx` | Alias for Health Analytics |
| 28 | **Insights** | `/insights` | `insights.tsx` | AI-generated wellness insights |
| 29 | **Scores Dashboard** | `/scores` | `scores-dashboard.tsx` | All wellness scores in one view |
| 30 | **Daily Brain Report** | `/brain-report` | `daily-brain-report.tsx` | Daily summary of brain activity |
| 31 | **Weekly Summary** | `/weekly-summary` | `weekly-brain-summary.tsx` | Weekly brain and wellness summary |

---

## Dreams & Journaling

| # | Page | Route | File | Description |
|---|------|-------|------|-------------|
| 32 | **Dream Journal** | `/dreams` | `dream-journal.tsx` | Record and analyze dreams with AI |
| 33 | **Food-Emotion** | `/food-emotion` | `food-emotion.tsx` | Correlation between food and emotions |

---

## Special Features

| # | Page | Route | File | Description |
|---|------|-------|------|-------------|
| 34 | **Achievements** | `/achievements` | `achievements.tsx` | Badges, tiers (bronze/silver/gold), progress tracking |
| 35 | **Community** | `/community` | `community.tsx` | Anonymous mood sharing, daily challenges, streaks leaderboard |
| 36 | **Pain Tracker** | `/pain-tracker` | `pain-tracker.tsx` | Pain/migraine logging with EEG theta tracking |
| 37 | **tPBM Session** | `/tpbm` | `tpbm-session.tsx` | Transcranial photobiomodulation session tracking |
| 38 | **Quick Session** | `/quick-session` | `quick-session.tsx` | 5-minute voice + breathing + meditation flow |
| 39 | **Couples Meditation** | `/couples-meditation` | `couples-meditation.tsx` | Dual-device meditation session |
| 40 | **Emotional Intelligence** | `/emotional-intelligence` | `emotional-intelligence.tsx` | EQ training and exercises |
| 41 | **Emotional Fitness** | `/emotional-fitness` | `emotional-fitness.tsx` | Emotion regulation workouts |

---

## User & Settings Pages

| # | Page | Route | File | Description |
|---|------|-------|------|-------------|
| 42 | **Settings** | `/settings` | `settings.tsx` | App preferences, ML backend URL, privacy mode, wellness disclaimer |
| 43 | **Connected Assets** | `/connected-assets` | `connected-assets.tsx` | Device connections (Health Connect, Muse, Oura, WHOOP, Garmin) |
| 44 | **Consent Settings** | `/consent-settings` | `consent-settings.tsx` | Per-modality biometric consent toggles |
| 45 | **Notifications** | `/notifications` | `notifications.tsx` | Notification center |
| 46 | **Export** | `/export` | `export.tsx` | Data export and download |
| 47 | **Help & Feedback** | `/help` | `help.tsx` | Quick start guide, FAQ, feedback form, contact |
| 48 | **Privacy Policy** | `/privacy` | `privacy-policy.tsx` | Full privacy policy with EU AI Act notice |
| 49 | **Session History** | `/sessions` | `session-history.tsx` | Past EEG/voice session records |
| 50 | **Personal Records** | `/records` | `personal-records.tsx` | Personal bests and milestones |
| 51 | **Supplements** | `/supplements` | `supplements.tsx` | Supplement tracking (standalone page) |
| 52 | **Habits** | `/habits` | `habits.tsx` | Habit tracking |

---

## Research & Study Pages

| # | Page | Route | File | Description |
|---|------|-------|------|-------------|
| 53 | **Research Hub** | `/research` | `research-hub.tsx` | Research study participation |
| 54 | **Research Enroll** | `/research/enroll` | `research-enroll.tsx` | Study enrollment |
| 55 | **Research Morning** | `/research/morning` | `research-morning.tsx` | Morning research session |
| 56 | **Research Daytime** | `/research/daytime` | `research-daytime.tsx` | Daytime research session |
| 57 | **Research Evening** | `/research/evening` | `research-evening.tsx` | Evening research session |
| 58 | **Study Landing** | `/study` | `StudyLanding.tsx` | Study landing page |
| 59 | **Study Consent** | `/study/consent` | `StudyConsent.tsx` | Study consent form |
| 60 | **Study Profile** | `/study/profile` | `StudyProfile.tsx` | Study participant profile |
| 61 | **Study Session** | `/study/session` | `StudySession.tsx` | Active study session |
| 62 | **Study Stress** | `/study/session/stress` | `StudySessionStress.tsx` | Stress assessment in study |
| 63 | **Study Food** | `/study/session/food` | `StudySessionFood.tsx` | Food logging in study |
| 64 | **Study Complete** | `/study/complete` | `StudyComplete.tsx` | Study completion screen |
| 65 | **Study Admin** | `/study/admin` | `StudyAdmin.tsx` | Study admin dashboard |

---

## Auth & Onboarding Pages

| # | Page | Route | File | Description |
|---|------|-------|------|-------------|
| 66 | **Welcome** | `/welcome` | `landing.tsx` | Welcome/landing screen |
| 67 | **Auth** | `/auth` | `auth.tsx` | Login / register |
| 68 | **Forgot Password** | `/forgot-password` | `forgot-password.tsx` | Password reset request |
| 69 | **Reset Password** | `/reset-password` | `reset-password.tsx` | Password reset form |
| 70 | **Onboarding** | `/onboarding` | `onboarding.tsx` | 4-screen onboarding flow |
| 71 | **Intent Select** | `/intent` | `intent-select.tsx` | Choose: study participant vs explore app |

---

## Developer Pages

| # | Page | Route | File | Description |
|---|------|-------|------|-------------|
| 72 | **Architecture Guide** | `/architecture-guide` | `architecture-guide.tsx` | System architecture documentation |
| 73 | **Benchmarks** | `/benchmarks` | `formal-benchmarks-dashboard.tsx` | ML model accuracy benchmarks |

---

## Redirects (Removed Pages)

These routes redirect to other pages — the original pages were removed.

| Old Route | Redirects To | Reason |
|-----------|-------------|--------|
| `/emotions` | `/brain-monitor` | Emotion Lab removed |
| `/mood` | `/brain-monitor` | Mood Trends removed |
| `/journal` | `/brain-monitor` | Journal alias removed |
| `/onboarding-new` | `/onboarding` | Legacy redirect |
| `/welcome-intro` | `/onboarding` | Legacy redirect |
| `/food` | `/nutrition` | Alias |
| `/food-log` | `/nutrition` | Alias |

---

## Route Aliases (Same Page, Different URL)

| Alias Route | Points To | Page |
|------------|-----------|------|
| `/food` | Nutrition | `nutrition.tsx` |
| `/food-log` | Nutrition | `nutrition.tsx` |
| `/trends` | Health Analytics | `health-analytics.tsx` |

---

**Total: 73 active pages** (excluding redirects and aliases)

*Last updated: 2026-03-23*
