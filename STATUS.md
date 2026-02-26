# Project Status

## Completed Features

### Frontend (20 pages)
- [x] **Daily Brain Report** (`/brain-report` — sleep summary, focus forecast, yesterday's insight, recommended action)
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
- [x] Formal Benchmarks Dashboard (`/benchmarks` — all 18 models + 8 datasets + research roadmap)
- [x] 49 shadcn/ui components, 5 chart components, dark theme, responsive layout

### ML Backend (82 endpoints, 18 models)
- [x] Emotion classifier (LightGBM mega, 74.21% CV on 9 datasets, 163 534 samples — cross-subject)
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
| Emotion Classifier | LightGBM mega (global PCA) | **74.21% CV** (9 datasets, 3-class, 85 features) | DEAP+DREAMER+GAMEEMO+DENS+FACED+SEED-IV+EEG-ER+STEW+Muse-Sub | **Active live path** — 163 534 samples, scaler+PCA+LGBM in single pkl |
| Emotion Classifier | MLP (PyTorch) | 93.11% CV | 8 datasets (within-subject split — inflated) | Deleted — per-dataset PCA + one-hot indicators, not deployable |
| Emotion Classifier | Feature heuristics | 65–75% | Live Muse 2 | Fallback when mega LGBM not loaded — FAA + DASM/RASM + FMT |
| Emotion Classifier (DENS standalone) | LightGBM | 79.55% CV | DENS (27 subjects, 128-ch EGI) | Valence-based 3-class, 41 features |
| Sleep Staging | Random Forest | 92.98% | ISRUC | Active, reliable |
| Dream Detector | Gradient Boosting | 97.20% | — | Active, reliable |
| Flow State | MLP | 62.86% | — | Active, marginal |
| Creativity | SVM + RF | 99.18% | — | Likely overfit (850 samples) |
| Food-Emotion | Feature heuristics | N/A | — | Novel — no prior benchmark exists |
| RL Threshold Agent (PPO) | PPO Actor-Critic (PyTorch) | 67% reward rate (live) | Synthetic NeurofeedbackEnv | Flow-zone target: 40–75%; 500 ep × 3 protocols |

## Needs Improvement

- [ ] **No frontend tests** — No Vitest or Jest configured. All 19 pages are untested
- [ ] **Untested hardware integration** — BrainFlow Muse 2 connection not tested end-to-end
- [ ] **Train remaining 15 models** — Only emotion classifier has formal training pipeline. Others use heuristic fallbacks
- [x] ~~**97.79% LGBM model**~~ — Deleted (inflated score, per-dataset PCA + within-subject contamination). Replaced by mega LGBM 74.21% CV
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
- [x] **FACED dataset** — 123 subjects, integrated, 63.31% CV (3-class positive/neutral/negative)
- [ ] **Write paper** — IEEE TAFFC or Frontiers in Neuroscience
- [ ] **Open-source release** — GitHub tag, HuggingFace model weights, anonymized pilot data

## Future Plans

- [ ] Add Vitest for frontend component + integration tests
- [ ] Train all 17 models on real datasets with published benchmarks
- [ ] Mobile-optimized layout / React Native companion app
- [ ] Multi-user session support (currently single-user per server instance)
- [ ] Docker Compose setup for one-command deployment (frontend + backend + DB)
- [ ] Real-time collaborative sessions (shared brain data viewing)
