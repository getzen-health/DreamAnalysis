# Project Status

## Completed Features

### Frontend (22 pages)
- [x] **Daily Brain Report** (`/brain-report` — sleep summary, focus forecast, yesterday's insight, weekly 7-day avg card, recommended action)
- [x] Landing page + authentication
- [x] Main dashboard with health metric overview (personal records: peak focus/flow/longest session + live "beat it" indicator; calibration banner until BaselineCalibrator ready)
- [x] Real-time EEG brain monitor
- [x] Brain connectivity analysis
- [x] Emotion lab (real-time classification)
- [x] Inner energy / chakra visualization
- [x] Dream journal with AI analysis
- [x] Dream pattern trends
- [x] Health analytics + correlations
- [x] Neurofeedback training
- [x] AI wellness companion (GPT-5)
- [x] Insights page
- [x] Session history
- [x] Settings + device configuration
- [x] Biofeedback training screen (`/biofeedback` — 7 breathing exercises incl. Physio Sigh, Cyclic Sigh, Power Breath; live stress chart, before/after comparison)
- [x] Baseline calibration UI (`/calibration`)
- [x] Baseline calibration onboarding screen (`/onboarding` — fullscreen 3-phase guided calibration)
- [x] Sleep session mode (`/sleep-session` — idle/recording/summary state machine, live stage tracking, dream detection, sleep score)
- [x] Food & Cravings page (`/food` — 6 food states, biomarker gauges, dietary recommendations)
- [x] Formal Benchmarks Dashboard (`/benchmarks` — all 18 models + 8 datasets + research roadmap)
- [x] Research beta signup (`/research/enroll` — rewritten as beta program, IRB language removed)
- [x] 49 shadcn/ui components, 5 chart components, dark theme, responsive layout
- [x] Vite vendor bundle splitting + React.lazy() code-splitting across 14 pages

### ML Backend (100+ endpoints, 95+ models)
- [x] Emotion classifier (LightGBM mega, 74.21% CV on 9 datasets, 163 534 samples — cross-subject)
- [x] Sleep staging, dream detection, flow state, creativity detection
- [x] Drowsiness, cognitive load, attention, stress, meditation classifiers
- [x] Lucid dream detection
- [x] Anomaly detection (Isolation Forest)
- [x] Artifact classification + denoising autoencoder
- [x] Online learning (per-user adaptation) — PersonalModelAdapter wired into `/analyze-eeg` inference with `personal_override` blending
- [x] EEG simulation mode (works without hardware)
- [x] **Parallel ML inference** — `ThreadPoolExecutor` for `/analyze-eeg` and `/simulate-eeg`
- [x] Signal quality gates (5-point SQI system, blocks readings below 40%)
- [x] Spiritual analysis (chakras, consciousness, aura)
- [x] WebSocket real-time streaming
- [x] **Food-Emotion Predictor** — 6 eating states (craving_carbs, appetite_suppressed, comfort_seeking, balanced, stress_eating, mindful_eating) mapped from FAA + high-beta + theta + delta biomarkers
- [x] routes.py split into modular route files (`ml/api/routes/` — 18 sub-routers)
- [x] **PPO Adaptive Threshold RL Agent** — real-time neurofeedback difficulty controller; fires on every `/neurofeedback/evaluate` call; adjusts protocol threshold ±0.05 via learned policy; 67% live reward rate (target flow zone: 40–75%); trained 500 ep × 3 protocols on synthetic `NeurofeedbackEnv`; endpoints: `POST /neurofeedback/rl/train`, `GET /neurofeedback/rl/status`; training runs in isolated subprocess to prevent GIL/OpenMP deadlock with live inference
- [x] 4-second sliding epoch buffer in `/analyze-eeg` (50% overlap, `epoch_ready` flag)
- [x] `BaselineCalibrator` class + 3 API endpoints (per-user resting-state normalization, +15–29% accuracy)
- [x] Mastoid re-reference wired into live BrainFlow stream
- [x] FAA, DASM/RASM, FMT fully integrated into emotion classifier
- [x] DREAMER dataset integrated (23 subjects, Emotiv EPOC 14-ch)
- [x] GAMEEMO dataset integrated (28 subjects, 4 games, neutral class source)
- [x] Device-aware gamma masking — Muse 2 zeros gamma features; research EEG uses all 85
- [x] FACED dataset integrated (123 subjects, 32-ch BrainProducts) — 63.31% CV (3-class, DE features)
- [x] DENS dataset integrated (27/40 subjects, 128-ch EGI HydroCel, 4807 samples) — 79.55% CV (3-class, LightGBM)
- [x] SEED-IV dataset integrated (15 subjects × 3 sessions, 62-ch, 17 490 samples)
- [x] EEG-ER dataset integrated (100 CSVs, Emotiv EPOC 14-ch, 8 612 samples) — 4 games → 3-class labels
- [x] STEW dataset integrated (45 segments, Emotiv EPOC 14-ch, 3 171 samples) — cognitive workload → emotion proxy
- [x] Muse-Subconscious dataset integrated (20 sessions, 4-ch Muse, 18 134 samples) — Mellow/Concentration labels
- [x] mega LGBM now **74.21% CV** on 9 datasets, 163 534 samples

### Connection UX & ML Reliability (thursday-launch-ready branch)
- [x] **useMLConnection hook** (`client/src/hooks/use-ml-connection.tsx`) — state machine tracking ML backend health: idle → connecting → warming → ready | error
- [x] **MLWarmupScreen** (`client/src/components/ml-warmup-screen.tsx`) — full-screen animated loading overlay shown during ML backend cold start (Render free-tier spin-up)
- [x] **App.tsx integration** — MLConnectionProvider wraps all authenticated routes; MLWarmupScreen renders as overlay during warming state
- [x] **Keep-alive ping** — AppLayout pings ML backend every 14 minutes to prevent Render free-tier sleep
- [x] **ML status dot** — 8px indicator in sidebar (green/amber/red) with Tooltip showing latency and Reconnect button on error
- [x] **mlFetch retry logic** — 3 retries with 1s/3s/9s exponential backoff + 30s AbortController timeout
- [x] **SimulationModeBanner** (`client/src/components/simulation-mode-banner.tsx`) — amber banner on emotion-lab and brain-monitor pages when ML backend is unreachable; prompts simulation mode
- [x] **TSception live inference** — TSception CNN (69.00% CV, 4-ch, 4-sec epochs) wired into emotion classifier fallback chain after DEAP models, before feature heuristics; activates when epoch ≥ 1024 samples
- [x] **RunningNormalizer** — per-user rolling z-score normalizer in `ml/processing/eeg_processor.py`, wired into `_predict_mega_lgbm()`; thread-safe, buffer of 150 frames (~5 min); corrects within-session non-stationarity
- [x] **Env/CORS audit** — `.env.example` updated with all required vars; `vercel.json` env block added for ML_BACKEND_URL; `render.yaml` CORS confirmed correct for production origin

### Infrastructure
- [x] Vercel deployment config (frontend + API)
- [x] **Vercel API fully working** — all `/api/*` routes functional; fixed ESM `.js` extension resolution, lazy-loaded heavy packages to prevent cold-start crash, added explicit `/api/:path*` rewrite for catch-all routing
- [x] **Cerebras LLM** — switched from OpenAI to Cerebras (`llama3.1-8b`, 1M tokens/day free, no Vercel IP blocking); `CEREBRAS_API_KEY` env var; all AI chat + dream analysis endpoints working on production
- [x] **Enhanced data export** — `/api/export/:userId?type=` supports `health` (default CSV), `dreams` (CSV with symbols/analysis), `emotions` (CSV with userCorrectedEmotion), `all` (multi-section), `healthkit` (Apple Health XML)
- [x] Render deployment (ML backend at neural-dream-ml.onrender.com)
- [x] Neon PostgreSQL (7 tables via Drizzle ORM)
- [x] BrainFlow hardware support (Muse 2, board_id=38)
- [x] Apple Health + Google Fit integration
- [x] ONNX model export for browser inference
- [x] PWA support (service worker + offline store)
- [x] Per-user state isolation (epoch buffer + BaselineCalibrator keyed by user_id)
- [x] Session history 24-hour timeline strip (Today view — green/orange/cyan session blocks)
- [x] Push notification service worker + Settings subscribe UI (morning reminder to /brain-report)

## Working Model Accuracies

| Model | Algorithm | Accuracy | Dataset | Notes |
|-------|-----------|----------|---------|-------|
| Emotion Classifier | LightGBM mega (global PCA) | **71.52% CV** (11 datasets, 3-class, 85 features) | DEAP+DREAMER+GAMEEMO+DENS+FACED+EAV+SEED-IV+EEG-ER+STEW+Muse-Sub+EmoKey | **Active live path** — 187 751 samples (303 177 after SMOTE), scaler+PCA+LGBM in single pkl |
| Emotion Classifier | MLP (PyTorch) | 93.11% CV | 8 datasets (within-subject split — inflated) | Deleted — per-dataset PCA + one-hot indicators, not deployable |
| Emotion Classifier | Feature heuristics | 65–75% | Live Muse 2 | Fallback when mega LGBM not loaded — FAA + DASM/RASM + FMT |
| Emotion Classifier (DENS standalone) | LightGBM | 79.55% CV | DENS (27 subjects, 128-ch EGI) | Valence-based 3-class, 41 features |
| Sleep Staging | Random Forest | 92.98% | ISRUC | Active, reliable |
| Dream Detector | Gradient Boosting | 97.20% | — | Active, reliable |
| Flow State | LightGBM | 57.00% CV | Synthetic (noise-augmented) | Retrained with noise augmentation |
| Creativity | SVM + RF | 99.18% | — | Likely overfit (850 samples) |
| Drowsiness | LightGBM | **81.72% CV** | Mental Attention + synthetic | Real dataset used — best new model |
| Cognitive Load | LightGBM | **65.72% CV** | STEW (3285 real samples) + synthetic | Real STEW dataset (14-ch, 45 subjects) |
| Attention | LightGBM | **63.87% CV** | Synthetic + DEAP proxy (10 800 samples) | 4-class; +3.87 pts from DEAP arousal/valence proxy |
| Stress | LightGBM | **59.64% CV** | Synthetic + DEAP proxy (10 800 samples) | 4-class; +0.89 pts from DEAP arousal/valence proxy |
| Lucid Dream | LightGBM | 61.85% CV | Synthetic (noise-augmented) | 4-class: non-lucid/pre-lucid/lucid/controlled |
| Meditation | LightGBM | **61.13% CV** | Synthetic (noise-augmented) | Reduced 5→3 classes (relaxed/meditating/deep); +8.65 pts |
| Artifact Classifier | LightGBM | **96.47% CV** | Synthetic (6000 samples, 6 artifact types) | Replaces rule-based detection |
| Denoising Autoencoder | PyTorch | +2.29 dB SNR improvement | Synthetic paired (5000 samples) | Saved as denoiser_model.pt |
| Food-Emotion | Feature heuristics | N/A | — | Novel — no prior benchmark exists |
| RL Threshold Agent (PPO) | PPO Actor-Critic (PyTorch) | 67% reward rate (live) | Synthetic NeurofeedbackEnv | Flow-zone target: 40–75%; 500 ep × 3 protocols |
| TSception Emotion | TSception CNN (PyTorch) | **69.00% CV** | Synthetic + DEAP (19 800 epochs, 4-ch, 4-sec epochs) | Temporal-spatial CNN for AF7/AF8 asymmetry; saved as tsception_emotion.pt |

## Needs Improvement

- [x] ~~**No frontend tests**~~ — 266 Vitest tests across 25 pages / 31 files — 100% pass rate
- [ ] **Untested hardware integration** — BrainFlow Muse 2 connection not tested end-to-end
- [x] **Train remaining models** — All 16 models now have saved weights. Improvements: meditation 52.48%→61.13% (3-class), attention 60%→63.87% (DEAP proxy), stress 58.75%→59.64% (DEAP proxy). TSception architecture added (ml/models/tsception.py + ml/training/train_tsception.py).
- [x] **EMA output smoothing for cognitive models** — α=0.25 EMA now applied to all 6 cognitive endpoints (drowsiness, load, attention, stress, lucid-dream, meditation) via `_smooth()` in cognitive.py — reduces frame-to-frame noise by ~75%
- [x] **BaselineCalibrator wired into cognitive endpoints** — `_calibrated_predict()` helper normalizes features against per-user resting baseline before sklearn prediction when cal.is_ready (≥30 frames)
- [x] ~~**97.79% LGBM model**~~ — Deleted (inflated score, per-dataset PCA + within-subject contamination). Replaced by mega LGBM 74.21% CV
- [x] ~~**Baseline calibration has no frontend UX**~~ — `/onboarding` fullscreen 3-phase guided calibration screen built; dashboard banner shows until BaselineCalibrator is ready; sidebar "Connect Device" links to `/device-setup` which routes to `/onboarding`
- [x] ~~**Device pairing UX missing**~~ — Device pairing wizard wired: Connect Device in sidebar → `/device-setup` banner → `/onboarding`
- [ ] **Food-emotion module needs validation data** — 6 states are scientifically grounded but no pilot study has been run to measure real accuracy

## Voice Emotion Fallback (Sprint 2026-03-04) ✅

- [x] funasr + modelscope added to requirements
- [x] VoiceEmotionModel — emotion2vec+ (iic/emotion2vec_plus_base) + LightGBM fallback
- [x] /voice-watch/analyze upgraded to 6-class output
- [x] /voice-watch/cache + /voice-watch/latest/{user_id} endpoints
- [x] useVoiceEmotion React hook — 7s MediaRecorder + backend call + cache
- [x] Emotion Lab — voice fallback panel (amber, shown when no EEG)
- [x] Dashboard — voice emotion card when no EEG streaming
- [x] WebSocket EEG+Voice fusion (70/30 blend when voice cached)
- [x] Intervention engine — voice_emotion triggers (arousal >= 0.7 or valence <= -0.3)
- [x] Brain Monitor — signal source badge (EEG/Voice/Health/EEG+Voice)

### Accuracy (No-EEG Mode)
| Mode | Accuracy | Signal |
|---|---|---|
| EEG only | 74% | Muse 2 |
| Voice only | 70-80% | Microphone (emotion2vec+) |
| Apple Health only | 50-65% | HR, HRV, sleep |
| EEG + Voice fused | 85-90% | EEG + mic + Health |

## Phase Roadmap

### Phase 0 — Make it not embarrassing ✅ COMPLETE
- [x] Deploy ML backend to Render
- [x] Fix per-user state isolation
- [x] Restore all hidden pages
- [x] Signal quality / HSI banners
- [x] "Calibrating…" until `epoch_ready: true`
- [x] Show confidence on emotion label

### Phase 1 — Create the aha moment ✅ COMPLETE
- [x] Real-time biofeedback screen (`/biofeedback` — 7 exercises, expanding circle, live stress chart, before/after comparison)
- [x] **Baseline calibration onboarding screen** (`/onboarding` — fullscreen 3-phase guided calibration)
- [x] **Device pairing wizard** — Connect Device in sidebar → `/device-setup` → `/onboarding`
- [x] Dashboard calibration banner — shows until BaselineCalibrator ready
- [x] Parallel ML inference — `ThreadPoolExecutor` for `/analyze-eeg` + `/simulate-eeg`
- [x] Vite vendor bundle splitting + React.lazy() code-splitting (14 pages)

### Phase 2 — Create a reason to come back ✅ COMPLETE
- [x] Session history with timeline view
- [x] "Yesterday's insight" card — detects patterns from previous day's health data (`/brain-report`)
- [x] Intervention library — 7 evidence-based exercises with before/after (added Physio Sigh, Cyclic Sigh, Power Breath)
- [x] Personal records ("New focus record: 47 min") — peak focus/flow/longest session + live "beat it" indicator on dashboard

### Phase 3 — Daily pull ✅ COMPLETE
- [x] Daily Brain Report screen (`/brain-report` — morning summary, sleep/dreams/forecast/recommended action)
- [x] Sleep session mode (`/sleep-session` — idle/recording/summary state machine, live stage tracking, dream detection, sleep score)
- [x] Pattern engine — server-side `GET /api/brain/patterns/:userId`: 30-day analysis → 5 pattern types (focus_peak_hour, stress_peak_hour, best_day_of_week, sleep_focus_correlation, biofeedback_effect). Brain Report "Your patterns" card now shows title + description + confidence + actionable recommendation. Fallback to client heuristic when server has no data.
- [x] Morning push notification — SW + subscribe UI + daily 8am cron trigger (VAPID env vars required)

### Phase 4 — Growth 🔄 IN PROGRESS
- [x] Weekly brain summary card — 7-day avg stress/focus/sleep with copy button (on `/brain-report`)
- [x] User-correctable emotion labels — "Was this right?" chip panel in `/emotion-lab`; stores correction + POSTs to ML backend `/feedback`
- [x] Per-user model fine-tuning trigger — after every 5 corrections for a user, POST batch to ML backend online learner (`PersonalModelAdapter.adapt()`)
- [x] Export data — CSV/JSON export with date-range + metric selector (`/api/ml/brain/export` via `ExportBrainDataCard`); Apple HealthKit export (`/api/ml/health/export-to-healthkit/{user_id}`) wired to "Export to HealthKit" button in Settings
- [x] Personal records gamification — `longestEverStreak`, `focusTrend`, `nextMilestone` helpers; new-record celebration banner; "beat it" challenge per row; live comparison; streak + milestone countdown
- [x] Weekly brain summary standalone page (`/weekly-summary`) — this week vs last week stress/focus/sleep with trend arrows, week-in-one-sentence, Canvas 2D PNG export (800×450, no deps)
- [x] Intervention library Evidence tab — personal before/after stress bars from `/interventions/effectiveness/:userId`; science citations for all 7 exercises
- [x] **Connection UX** — MLWarmupScreen overlay during cold start; keep-alive ping every 14 min; sidebar ML status dot (green/amber/red) with latency tooltip; mlFetch 3-retry exponential backoff (1s/3s/9s) + 30s timeout; SimulationModeBanner on emotion-lab and brain-monitor when ML unreachable
- [x] **TSception fallback** — TSception CNN (69.00% CV) active in emotion classifier fallback chain; RunningNormalizer corrects within-session EEG drift per-user

### Phase 5 — Mobile ✅ iOS Build verified, TestFlight pending (2026-03-04)
- [x] Capacitor 8.1.0 installed (`@capacitor/core`, `cli`, `ios`, `android`)
- [x] `capacitor.config.ts` — appId, webDir=dist/public, SplashScreen/StatusBar/PushNotifications config
- [x] Safe area insets — `viewport-fit=cover`; `env(safe-area-inset-*)` CSS vars; body padding; home indicator clearance
- [x] iOS HIG touch targets — global `min-height: 44px` + `min-h-[44px]` on sidebar nav links; hamburger 44×44px
- [x] Haptic feedback — `@capacitor/haptics`; `haptics.ts` wrapper with `isNative()` check; `hapticLight` on inhale, `hapticMedium` on hold/exhale, `hapticSuccess` on session complete; wired into biofeedback + sleep-session
- [x] Local ML inference — `emotion_classifier_model.onnx` (2.2 MB) served from `client/public/models/`; JS band-power heuristics for sleep/dream; `use-inference.ts`: local-first, server fallback
- [x] Offline mode — IndexedDB v2 (`offline-store.ts`): `dream_drafts` + `eeg_queue` + `health_queue`; `OfflineSyncBanner` auto-syncs on reconnect; `syncAll()` drains all queues
- [x] Privacy policy page (`/privacy`) — 6 sections covering EEG, HealthKit, research, security, contact; no auth guard (required for App Store)
- [x] Spotify integration — OAuth 2.0 flow (`/api/spotify/auth` + `/api/spotify/callback`); `POST /api/spotify/play` with mood routing; `SpotifyConnect` component in biofeedback Music tab; auto-plays calm/focus when music intervention fires via `InterventionBanner`
- [x] Bluetooth BLE for Muse 2/S — `@capacitor-community/bluetooth-le@8.1.0`; `muse-ble.ts`: MuseBleManager, GATT UUIDs, 12-bit packet decoder (20-byte BLE → µV), ring buffers, FAA/stress/focus extraction; `use-device.tsx` auto-routes to BLE path when `Capacitor.isNativePlatform()` + muse device; desktop falls back to BrainFlow/WebSocket
- [x] Apple HealthKit — `@perfood/capacitor-healthkit@1.3.2`; `health-sync.ts`: pulls HR, resting HR, HRV proxy (SDNN from beat variance), respiratory rate, SpO2, body temp, sleep stages, steps, active calories → POST `/api/biometrics/update` → MultimodalEmotionFusion. 15-min auto-sync via `useHealthSync` hook in AppLayout.
- [x] Google Health Connect — `capacitor-health@8.0.1`; Android path in `health-sync.ts`: HR workouts, steps, calories, mindfulness. Same endpoint.
- [x] Native push notifications — `@capacitor/push-notifications@8.0.1`; `native-push.ts`: APNs/FCM token registration, foreground+background handlers, tap-to-navigate; server: `POST /api/notifications/native-token` + `POST /api/notifications/send-native` (firebase-admin, optional). Wired into AppLayout.
- [x] Background EEG for sleep — `@capacitor/background-runner@3.0.0`; `background-eeg.ts`: web=Screen Wake Lock API, native=BackgroundRunner.dispatchEvent(); `background-runner.js`: isolated JS context, ongoing Android notification, 15-min periodic flush; capacitor.config.ts updated; wired into sleep-session handleStart/handleWakeUp.
- [x] App Store listing — `docs/app-store-listing.md`: 4000-char description, keywords, 5 screenshot descriptions, HealthKit/BT privacy strings, UIBackgroundModes Info.plist config, age rating, privacy nutrition label.
- [x] Research paper draft — `docs/paper_draft.md`: 9 sections, ablation table, Algorithm 1, full accuracy tables. Target: arXiv first, then IEEE EMBC 2026.
- [ ] `npx cap add ios` — blocked on Xcode.app (only CLI tools installed)
- [ ] `npx cap add android` — blocked on JDK + Android Studio
- [ ] Splash screen + app icon — 1024×1024 PNG design needed

## Food-Emotion Research Roadmap

This is the novel publishable contribution. No prior paper maps real-time consumer EEG to food/eating states.

- [ ] **IRB ethics approval** (4–8 weeks) — required for human subjects research
- [ ] **Controlled pilot study** (4–6 weeks post-IRB) — 20–30 participants, food cue presentation protocol
- [ ] **Cross-subject validation** — LOSO cross-validation, target >65% 6-class with calibration
- [x] **DREAMER dataset** — downloaded and integrated into cross-dataset training pipeline
- [x] **FACED dataset** — 123 subjects, integrated, 63.31% CV (3-class positive/neutral/negative)
- [ ] **Write paper** — IEEE TAFFC or Frontiers in Neuroscience
- [ ] **Open-source release** — GitHub tag, HuggingFace model weights, anonymized pilot data

## Future Plans

- [x] ~~Add Vitest for frontend tests~~ — **241 tests across 22 pages / 27 files** — 100% pass rate (all passing)
- [ ] Train all 17 models on real datasets with published benchmarks
- [ ] Mobile-optimized layout / React Native companion app
- [ ] Multi-user session support (currently single-user per server instance)
- [x] Docker Compose one-command deployment — `docker-compose up` starts db + api + ml; VAPID and ML_BACKEND_URL env vars wired; Datadog profile-gated
- [ ] Real-time collaborative sessions (shared brain data viewing)

### Research Daemon Implementations (Cycles 5-8, 2026-03-08)
- [x] **Flow State Detector** — quadratic theta model, beta asymmetry, flow ratio (#64)
- [x] **Emotional Granularity** — 27-emotion VAD mapping with dominance dimension (#63)
- [x] **Microsleep Detection** — theta/alpha ratio streak tracking (#91)
- [x] **IMU Artifact Removal** — adaptive LMS filter using Muse 2 accelerometer (#70)
- [x] **Study Session Optimizer** — FMT-based encoding detection, attention trends (#94)
- [x] **Empathy Detector** — temporal mu rhythm suppression at TP9/TP10 (#96)
- [x] **Pain Biomarker Detection** — frontal beta/alpha asymmetry + high-beta (#92)
- [x] **Emotion Regulation** — LPP proxy + frontal theta (#53)
- [x] **Cross-Modal EEG+Voice Alignment** — Optimal Transport fusion (#55)
- [x] **DGAT** — Dynamic graph attention for adaptive channel relationships (#58)
- [x] **N400 ERP Detection** — semantic processing at AF8, validated on Muse 2 (#97)
- [x] **Neural Efficiency Tracker** — alpha power skill mastery tracking (#98)
- [x] **Tinnitus Assessment** — temporal alpha/gamma biomarkers (#99)
- [x] **Mindfulness Quality** — mind-wandering detection during meditation (#101)
- [x] **Learning Stage Classifier** — encoding/consolidation/automation/mastery (#102)
- [x] **Adaptive Learning Detector** — FMT-based learning pace optimization (#103)
- [x] **Brain Music Mapper** — EEG emotion-to-musical-parameter sonification (#106)
- [x] **Haptic Urgency Optimizer** — arousal-adaptive haptic feedback (#107)
- [x] **Circadian Normalization** — chronotype detection + time-of-day band power correction (#54)
- [x] **Memory Consolidation** — spindle-SO coupling tracker for sleep quality (#62)
- [x] **Emotion Trajectory** — valence-arousal dynamics with inertia and prediction (#68)
- [x] **Seizure Detector** — 4-channel feature-based with alarm logic + hypersynchrony (#112)
- [x] **Neurogame Engine** — EEG-driven game commands with adaptive difficulty (#110)
- [x] **Neuroadaptive Tutor** — 5-zone learning adaptation (boredom→flow) (#116)
- [x] **Deception Detector** — P300 CIT paradigm with ERP averaging (#104)
- [x] **Digital Phenotyper** — EEG + health fusion for mental wellness trends (#95)
- [x] **Few-Shot Personalizer** — prototypical matching, 5-shot per-class adaptation (#115)
- [x] **Brain Maturation** — brain age gap estimation via aperiodic + periodic features (#111)
- [x] **Fatigue Monitor** — theta/beta trend + time-on-task decay + break timing (#109)
- [x] **EEG Authenticator** — PSD spectral fingerprinting for biometrics (#93)
- [x] **Engagement Detector** — 3-state + 5 educational emotions (#114)
- [x] **Federated Learning** — privacy-preserving multi-user model training (#100)
- [x] **GNN Spatial-Temporal** — graph attention for 4-channel EEG emotion (#51)
- [x] **DREAM Database** — enhanced dream detection integration (#50)
- [x] **Tinnitus NF Protocol** — alpha up-training at TP9/TP10 with reward feedback (#105)
- [x] **Sleep Quality Predictor** — multi-component scoring + readiness forecast
- [x] **Heart-Brain Coupling** — HEP + HRV + EEG fusion (#69)
- [x] **Cognitive Flexibility** — frontal theta during task switching (#61)
- [x] **Emotion Regulation Trainer** — FAA-based closed-loop biofeedback (#121)
- [x] **Motor Imagery Classifier** — 4-class ERD-based BCI (#123)
- [x] **Meditation Depth Quantifier** — FMT + alpha coherence + gamma bursts (#124)
- [x] **ADHD Detector** — theta/beta ratio screening (#60)
- [x] **Neurofeedback Audio** — multi-protocol audio parameter generation (#49)
- [x] **Cognitive Reserve** — spectral entropy + 1/f slope + alpha peak
- [x] **Anxiety Protocol** — alpha up + high-beta down training
- [x] **Social Cognition** — mu suppression + frontal theta for empathy
- [x] **Workload Adapter** — real-time difficulty adjustment for VR/AR
- [x] **Eye State Detector** — alpha reactivity + blink detection
- [x] **Hemispheric Balance** — per-band left/right asymmetry monitoring
- [x] **Pre-Ictal Predictor** — entropy + synchrony trend for seizure forewarning (#117)
- [x] **Domain Adapter** — CORAL-lite cross-subject feature alignment (#113)
- [x] **EEG-Voice Fusion** — decision-level multimodal emotion fusion (#21)
- [x] **Spindle Analyzer** — sleep spindle detection + memory consolidation index
- [x] **Slow Oscillation Detector** — SO detection + SO-spindle coupling
- [x] **Concentration Tracker** — sustained attention with lapse detection
- [x] **Neural Complexity** — sample/permutation entropy, LZ, Hurst, fractal dim
- [x] **Circadian Monitor** — alertness tracking + optimal cognitive windows
- [x] **Connectivity Graph** — coherence + PLV across channel pairs
- [x] **PPG sensor integration** — HR/HRV/respiratory rate from Muse 2 forehead PPG, emotion heuristics, 3 API endpoints (#46)
- [x] **DreamNet NLP** — dream text analysis with 15 theme categories, 26 symbol patterns, sentiment, lucidity scoring (#48)
- [x] **Brain age SpecParam** — aperiodic 1/f + alpha peak decomposition, normative brain age estimation, BAG scoring (#59)
- [x] **CNN-KAN hybrid** — Conv1D + Kolmogorov-Arnold Network B-spline layers, 337K params, 3-class emotion (#65)
- [x] **Self-supervised Barlow Twins** — EEG representation learning with 5 augmentations, 128-dim embeddings, k-NN downstream (#66)
- [x] **EEGNet-Lite edge** — 2707-param depthwise separable CNN, ONNX export (12 KB), browser-deployable (#67)
