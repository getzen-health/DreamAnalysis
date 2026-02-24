# Project Status

## Completed Features

### Frontend (19 pages)
- [x] Landing page + authentication
- [x] Main dashboard with health metric overview
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
- [x] Biofeedback training screen (`/biofeedback` — 4 breathing exercises, live stress chart, before/after comparison)
- [x] Baseline calibration UI (`/calibration`)
- [x] Food & Cravings page (`/food` — 6 food states, biomarker gauges, dietary recommendations)
- [x] Formal Benchmarks Dashboard (`/benchmarks` — all 16 models + 8 datasets + research roadmap)
- [x] 49 shadcn/ui components, 5 chart components, dark theme, responsive layout

### ML Backend (79 endpoints, 17 models)
- [x] Emotion classifier (LightGBM, 97.79% on DEAP+SEED+GAMEEMO — within-subject)
- [x] Sleep staging, dream detection, flow state, creativity detection
- [x] Drowsiness, cognitive load, attention, stress, meditation classifiers
- [x] Lucid dream detection
- [x] Anomaly detection (Isolation Forest)
- [x] Artifact classification + denoising autoencoder
- [x] Online learning (per-user adaptation)
- [x] EEG simulation mode (works without hardware)
- [x] Signal quality gates (5-point SQI system, blocks readings below 40%)
- [x] Spiritual analysis (chakras, consciousness, aura)
- [x] WebSocket real-time streaming
- [x] **Food-Emotion Predictor** — 6 eating states (craving_carbs, appetite_suppressed, comfort_seeking, balanced, stress_eating, mindful_eating) mapped from FAA + high-beta + theta + delta biomarkers
- [x] routes.py split into modular route files (`ml/api/routes/` — 18 sub-routers)
- [x] 4-second sliding epoch buffer in `/analyze-eeg` (50% overlap, `epoch_ready` flag)
- [x] `BaselineCalibrator` class + 3 API endpoints (per-user resting-state normalization, +15–29% accuracy)
- [x] Mastoid re-reference wired into live BrainFlow stream
- [x] FAA, DASM/RASM, FMT fully integrated into emotion classifier
- [x] DREAMER dataset integrated (23 subjects, Emotiv EPOC 14-ch)
- [x] GAMEEMO dataset integrated (28 subjects, 4 games, neutral class source)
- [x] Device-aware gamma masking — Muse 2 zeros gamma features; research EEG uses all 85

### Infrastructure
- [x] Vercel deployment config (frontend + Express)
- [x] Render deployment (ML backend at neural-dream-ml.onrender.com)
- [x] Neon PostgreSQL (7 tables via Drizzle ORM)
- [x] BrainFlow hardware support (Muse 2, board_id=38)
- [x] Apple Health + Google Fit integration
- [x] ONNX model export for browser inference
- [x] PWA support (service worker + offline store)
- [x] Per-user state isolation (epoch buffer + BaselineCalibrator keyed by user_id)

## Working Model Accuracies

| Model | Algorithm | Accuracy | Dataset | Notes |
|-------|-----------|----------|---------|-------|
| Emotion Classifier | LightGBM (mega) | 97.79% | DEAP + SEED + GAMEEMO | Within-subject — not loaded in live path |
| Emotion Classifier | XGBoost | ~95% | DEAP + SEED + GAMEEMO | Within-subject — not loaded |
| Emotion Classifier | MLP (PyTorch) | 93.11% | DEAP + SEED + GAMEEMO | Within-subject — not loaded |
| Emotion Classifier | Feature heuristics | 65–75% | Live Muse 2 | **This is the live path** — FAA + DASM/RASM + FMT |
| Emotion Classifier | GBM cross-subject | 69.25% CV (DEAP+DREAMER+GAMEEMO, 3-class, 85 features) | DEAP + DREAMER + GAMEEMO | Formal cross-subject benchmark |
| Sleep Staging | Random Forest | 92.98% | ISRUC | Active, reliable |
| Dream Detector | Gradient Boosting | 97.20% | — | Active, reliable |
| Flow State | MLP | 62.86% | — | Active, marginal |
| Creativity | SVM + RF | 99.18% | — | Likely overfit (850 samples) |
| Food-Emotion | Feature heuristics | N/A | — | Novel — no prior benchmark exists |

## Needs Improvement

- [ ] **No frontend tests** — No Vitest or Jest configured. All 19 pages are untested
- [ ] **Untested hardware integration** — BrainFlow Muse 2 connection not tested end-to-end
- [ ] **Train remaining 15 models** — Only emotion classifier has formal training pipeline. Others use heuristic fallbacks
- [ ] **97.79% LGBM model not integrated** — Exists in `ml/models/` but requires PCA transform + 3→6 class mapping to plug in
- [ ] **Baseline calibration has no frontend UX** — API (`/calibration/baseline/add-frame`) exists but there is no guided 2-min onboarding screen
- [ ] **Device pairing UX missing** — No guided flow to connect Muse 2 with signal quality check before session starts
- [ ] **Food-emotion module needs validation data** — 6 states are scientifically grounded but no pilot study has been run to measure real accuracy

## Phase Roadmap

### Phase 0 — Make it not embarrassing ✅ COMPLETE
- [x] Deploy ML backend to Render
- [x] Fix per-user state isolation
- [x] Restore all hidden pages
- [x] Signal quality / HSI banners
- [x] "Calibrating…" until `epoch_ready: true`
- [x] Show confidence on emotion label

### Phase 1 — Create the aha moment 🔄 IN PROGRESS
- [x] Real-time biofeedback screen (`/biofeedback` — breathing exercises with live stress chart)
- [ ] **Baseline calibration onboarding screen** — 2-min eyes-closed guided session before first reading
- [ ] **Device pairing flow** — guided connect + HSI signal quality check before session starts

### Phase 2 — Create a reason to come back ⏳ PENDING
- [ ] Session history with timeline view
- [ ] "Yesterday's insight" card (one surprising pattern from last session)
- [ ] Intervention library (5–10 evidence-based exercises with before/after)
- [ ] Personal records ("New focus record: 47 min")

### Phase 3 — Daily pull ⏳ PENDING
- [ ] Daily Brain Report screen (the North Star — one-screen morning summary)
- [ ] Sleep session mode (overnight recording with dream detection)
- [ ] Pattern engine: correlate time-of-day, activities, mental states
- [ ] Morning push notification with yesterday's summary

### Phase 4 — Growth ⏳ PENDING
- [ ] Weekly brain summary card (shareable)
- [ ] User-correctable labels → feeds personalization
- [ ] Per-user model fine-tuning after 5 sessions
- [ ] Export data (CSV, Apple Health sync)

## Food-Emotion Research Roadmap

This is the novel publishable contribution. No prior paper maps real-time consumer EEG to food/eating states.

- [ ] **IRB ethics approval** (4–8 weeks) — required for human subjects research
- [ ] **Controlled pilot study** (4–6 weeks post-IRB) — 20–30 participants, food cue presentation protocol
- [ ] **Cross-subject validation** — LOSO cross-validation, target >65% 6-class with calibration
- [x] **DREAMER dataset** — downloaded and integrated into cross-dataset training pipeline
- [ ] **FACED dataset** — create Synapse.org account, 123 subjects, 9 emotions
- [ ] **Write paper** — IEEE TAFFC or Frontiers in Neuroscience
- [ ] **Open-source release** — GitHub tag, HuggingFace model weights, anonymized pilot data

## Future Plans

- [ ] Add Vitest for frontend component + integration tests
- [ ] Train all 17 models on real datasets with published benchmarks
- [ ] Mobile-optimized layout / React Native companion app
- [ ] Multi-user session support (currently single-user per server instance)
- [ ] Docker Compose setup for one-command deployment (frontend + backend + DB)
- [ ] Real-time collaborative sessions (shared brain data viewing)
