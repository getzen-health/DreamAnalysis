# Neural Dream Workshop

A brain-computer interface (BCI) web application that reads EEG signals from a Muse 2 headband and uses 16 machine-learning models to classify emotions, detect dreams, stage sleep, measure focus, and more — all visualized in a real-time React dashboard.

## Architecture

```
Browser (React + TypeScript)
    |
    +-- REST --> Express.js (:5000) --> PostgreSQL (Neon)
    |               |
    |               +-- /api/dream-analysis, /api/ai-chat (GPT-5)
    |
    +-- REST + WebSocket --> FastAPI (:8000)
                               |
                               +-- 16 ML models (LightGBM, ONNX, PyTorch)
                               +-- EEG signal processing pipeline
                               +-- BrainFlow (Muse 2 hardware)
```

## Project Structure

```
NeuralDreamWorkshop/
|
+-- .github/workflows/
|   +-- ci.yml                          # CI pipeline (lint, test, build, deploy)
|
+-- client/                             # React 18 + TypeScript frontend
|   +-- src/
|   |   +-- pages/                      # 16 route pages
|   |   |   +-- dashboard.tsx               Main dashboard
|   |   |   +-- brain-monitor.tsx           Real-time EEG visualization
|   |   |   +-- emotion-lab.tsx             Emotion analysis lab
|   |   |   +-- dream-journal.tsx           Dream recording & analysis
|   |   |   +-- dream-patterns.tsx          Dream pattern recognition
|   |   |   +-- neurofeedback.tsx           Neurofeedback training
|   |   |   +-- brain-connectivity.tsx      Brain region connectivity
|   |   |   +-- health-analytics.tsx        Health metrics dashboard
|   |   |   +-- inner-energy.tsx            Chakra & spiritual analysis
|   |   |   +-- session-history.tsx         Past session browser
|   |   |   +-- insights.tsx                Weekly insights & trends
|   |   |   +-- ai-companion.tsx            AI chat companion
|   |   |   +-- settings.tsx                App settings
|   |   |   +-- auth.tsx                    Login / register
|   |   |   +-- landing.tsx                 Landing page
|   |   |   +-- not-found.tsx               404 page
|   |   |
|   |   +-- components/                 # Reusable UI components
|   |   |   +-- charts/                     5 chart components (EEG, sleep, mood, etc.)
|   |   |   +-- ui/                         49 shadcn/ui primitives
|   |   |   +-- ai-analysis.tsx             AI-powered analysis panel
|   |   |   +-- ai-companion.tsx            Chat interface
|   |   |   +-- brain-bands.tsx             Frequency band visualizer
|   |   |   +-- calibration-wizard.tsx      User calibration flow
|   |   |   +-- device-connection.tsx       Muse 2 pairing UI
|   |   |   +-- emotion-wheel.tsx           Valence-arousal wheel
|   |   |   +-- neural-background.tsx       Animated background
|   |   |   +-- neural-network.tsx          Network visualization
|   |   |   +-- session-controls.tsx        Start/stop session
|   |   |   +-- signal-quality-badge.tsx    Signal quality indicator
|   |   |   +-- sidebar.tsx                 App navigation
|   |   |   +-- voice-recorder.tsx          Voice input
|   |   |
|   |   +-- hooks/                       # 7 custom hooks
|   |   |   +-- use-auth.tsx                Auth state management
|   |   |   +-- use-device.tsx              BCI device connection
|   |   |   +-- use-inference.ts            ML model inference
|   |   |   +-- use-metrics.tsx             Health metrics
|   |   |   +-- use-mobile.tsx              Mobile responsiveness
|   |   |   +-- use-theme.tsx               Dark/light theme
|   |   |   +-- use-toast.ts               Toast notifications
|   |   |
|   |   +-- lib/                         # Utility libraries
|   |   |   +-- ml-api.ts                   FastAPI ML client
|   |   |   +-- ml-local.ts                 Client-side ONNX inference
|   |   |   +-- eeg-features.ts             JS feature extraction
|   |   |   +-- data-simulation.ts          Demo data generator
|   |   |   +-- queryClient.ts              TanStack Query config
|   |   |   +-- openai.ts                   OpenAI client
|   |   |   +-- offline-store.ts            Offline data persistence
|   |   |   +-- utils.ts                    Shared utilities
|   |   |
|   |   +-- layouts/
|   |   |   +-- app-layout.tsx              Main layout wrapper
|   |   |
|   |   +-- App.tsx                      # Router (wouter) + all routes
|   |   +-- main.tsx                     # Entry point
|   |   +-- index.css                    # Tailwind imports
|   |
|   +-- public/
|       +-- manifest.json                # PWA manifest
|       +-- sw.js                        # Service worker
|
+-- server/                              # Express.js middleware
|   +-- index.ts                         # Entry point (:5000)
|   +-- routes.ts                        # 10 REST endpoints
|   +-- storage.ts                       # Drizzle ORM data layer
|   +-- vite.ts                          # Vite dev server integration
|
+-- shared/
|   +-- schema.ts                        # 7 Drizzle tables + Zod validators
|
+-- api/                                 # Vercel serverless functions
|   +-- _lib/                            # Shared helpers (auth, db, openai)
|   +-- auth/                            # login, logout, me, register
|   +-- ai-chat/                         # AI conversation endpoints
|   +-- dream-analysis/                  # Dream analysis CRUD
|   +-- dreams/                          # Dream journal (list, create, analytics, generate-image)
|   +-- emotions/                        # Emotion history + record
|   +-- health-metrics/                  # Health data CRUD
|   +-- insights/                        # Weekly insights
|   +-- export/                          # Data export
|   +-- notifications/                   # Push subscription
|   +-- settings/                        # User settings
|   +-- analyze-mood.ts                  # Mood analysis
|
+-- ml/                                  # Python ML backend
|   +-- main.py                          # FastAPI entry point (:8000)
|   +-- pytest.ini                       # Test config
|   +-- requirements.txt                 # Python dependencies
|   +-- ruff.toml                        # Linter config
|   |
|   +-- api/
|   |   +-- routes.py                    # 76 REST endpoints (2K lines)
|   |   +-- websocket.py                 # Real-time EEG streaming
|   |
|   +-- models/                          # 16 ML model classes
|   |   +-- emotion_classifier.py            6 emotions, LightGBM, 97.79% accuracy
|   |   +-- sleep_staging.py                 Wake / N1 / N2 / N3 / REM classification
|   |   +-- dream_detector.py                Dream state detection during sleep
|   |   +-- flow_state_detector.py           Flow state scoring (0-1)
|   |   +-- creativity_detector.py           Creative thinking + memory encoding
|   |   +-- drowsiness_detector.py           Sleepiness from theta power
|   |   +-- cognitive_load_estimator.py      Mental workload estimation
|   |   +-- attention_classifier.py          Attention level from beta/theta
|   |   +-- stress_detector.py               Stress from beta asymmetry
|   |   +-- lucid_dream_detector.py          Gamma bursts during REM
|   |   +-- meditation_classifier.py         Meditation depth from alpha coherence
|   |   +-- anomaly_detector.py              Unusual EEG patterns (Isolation Forest)
|   |   +-- artifact_classifier.py           Eye blink / muscle / electrode artifacts
|   |   +-- denoising_autoencoder.py         Signal cleaning (PyTorch autoencoder)
|   |   +-- online_learner.py                Per-user model adaptation
|   |   +-- saved/                           Trained weights (.onnx, .pkl, .pt)
|   |
|   +-- processing/                      # Signal processing pipeline
|   |   +-- eeg_processor.py                 Feature extraction (17 features)
|   |   +-- artifact_detector.py             Artifact detection & removal
|   |   +-- signal_quality.py                Signal quality scoring
|   |   +-- calibration.py                   Per-user baseline calibration
|   |   +-- confidence_calibration.py        Model confidence calibration
|   |   +-- state_transitions.py             Brain state transition engine
|   |   +-- connectivity.py                  Brain region connectivity
|   |   +-- emotion_shift_detector.py        Pre-conscious emotion detection
|   |   +-- noise_augmentation.py            Training data augmentation
|   |   +-- spiritual_energy.py              Chakra / aura / consciousness
|   |   +-- user_feedback.py                 Personalized model tuning
|   |
|   +-- simulation/
|   |   +-- eeg_simulator.py                 Synthetic EEG generation
|   |
|   +-- training/                        # Model training scripts
|   |   +-- mega_trainer.py                  Multi-algorithm trainer
|   |   +-- data_loaders.py                  8 EEG dataset loaders
|   |   +-- train_emotion.py                 Emotion classifier training
|   |   +-- train_sleep.py                   Sleep staging training
|   |   +-- train_dream.py                   Dream detector training
|   |   +-- benchmark.py                     Model benchmarking
|   |   +-- (+ 13 more experiment scripts)
|   |
|   +-- health/                          # Wearable health integrations
|   |   +-- apple_health.py                  Apple Health import/export
|   |   +-- google_fit.py                    Google Fit import
|   |   +-- correlation_engine.py            Brain-body correlations
|   |
|   +-- hardware/
|   |   +-- brainflow_manager.py             Muse 2 device manager (BrainFlow)
|   |
|   +-- storage/
|   |   +-- session_recorder.py              Session data persistence
|   |   +-- session_analytics.py             Trend analysis & comparisons
|   |
|   +-- neurofeedback/
|   |   +-- protocol_engine.py               Neurofeedback protocols
|   |
|   +-- tools/                           # CLI utilities
|   |   +-- demo_full_pipeline.py            End-to-end pipeline demo
|   |   +-- import_apple_health.py           Apple Health XML importer
|   |   +-- live_brain_session.py            Live session runner
|   |
|   +-- tests/                           # pytest test suite
|   |   +-- conftest.py                      Shared fixtures (sample EEG signals)
|   |   +-- test_models.py                   All 6 core model tests
|   |   +-- test_processing.py               Signal processing pipeline tests
|   |   +-- test_api.py                      FastAPI endpoint tests
|   |   +-- test_accuracy_pipeline.py        Accuracy module tests
|   |
|   +-- benchmarks/                      # Training results (JSON)
|   +-- data/                            # EEG datasets (gitignored)
|
+-- docs/
|   +-- architecture.html                # Interactive architecture diagram
|
+-- CLAUDE.md                            # Project context for AI agents
+-- agent.md                             # Agent conventions & patterns
+-- VERCEL_DEPLOYMENT.md                 # Vercel deployment guide
+-- vercel.json                          # Vercel config
+-- package.json                         # Node dependencies
+-- tsconfig.json                        # TypeScript config
+-- vite.config.ts                       # Vite bundler config
+-- tailwind.config.ts                   # Tailwind CSS config
+-- drizzle.config.ts                    # Drizzle ORM config
+-- components.json                      # shadcn/ui config
```

## Quick Start

```bash
# Frontend + Express middleware (port 5000)
npm install
npm run dev

# ML backend (port 8000)
cd ml
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Environment Variables

| Variable | Used By | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | Express | Neon PostgreSQL connection string |
| `OPENAI_API_KEY` | Express | GPT-5 for dream analysis + AI chat |
| `SESSION_SECRET` | Express | Express session encryption |
| `JWT_SECRET` | Vercel API | JWT token signing (serverless) |

## The 16 ML Models

| Model | Accuracy | What It Does |
|-------|----------|-------------|
| Emotion Classifier | 97.79% | 6 emotions + valence/arousal (LightGBM) |
| Sleep Staging | - | Wake / N1 / N2 / N3 / REM classification |
| Dream Detector | - | Detects dreaming during sleep |
| Flow State | - | Measures "in the zone" (0-1 score) |
| Creativity | - | Creative thinking from alpha/theta ratios |
| Memory Encoding | - | Predicts memory formation strength |
| Drowsiness | - | Sleepiness from theta power increases |
| Cognitive Load | - | Mental workload (low/medium/high) |
| Attention | - | Attention level from beta/theta ratios |
| Stress | - | Stress from beta asymmetry |
| Lucid Dream | - | Gamma bursts during REM |
| Meditation | - | Meditation depth from alpha coherence |
| Anomaly | - | Unusual EEG patterns (Isolation Forest) |
| Artifact Classifier | - | Eye blink / muscle / electrode artifacts |
| Denoising Autoencoder | - | Cleans noisy EEG (PyTorch) |
| Online Learner | - | Adapts to individual users over time |

## CI/CD Pipeline

Single GitHub Actions workflow (`.github/workflows/ci.yml`) with 4 jobs:

```
ci.yml
  |
  +-- frontend          Lint (tsc) + Build (vite)
  |
  +-- ml                Ruff lint + pytest (4 test files) + coverage + inference benchmark
  |
  +-- dependency-audit  npm audit + pip-audit (non-blocking)
  |
  +-- deploy            Vercel production deploy (only on push to main, after frontend + ml pass)
```

- PRs get lint + test + build checks
- Deploys are gated: broken code cannot reach production
- ML tests use proper pytest with coverage (`--cov`), not inline scripts

## Testing

```bash
# ML tests (from ml/ directory)
cd ml && pytest tests/ -v --cov=. --cov-report=term-missing

# TypeScript type checking
npx tsc --noEmit

# Full frontend build
npm run build
```

## Deployment

- **Frontend + Express**: Vercel (see `vercel.json` and `VERCEL_DEPLOYMENT.md`)
- **ML Backend**: Docker or standalone (`uvicorn main:app`)
- **Database**: Neon PostgreSQL (`drizzle-kit push` for migrations)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Tailwind CSS, shadcn/ui, wouter, TanStack Query, Recharts |
| Middleware | Express.js, Passport, Drizzle ORM, JWT |
| ML Backend | FastAPI, scikit-learn, LightGBM, PyTorch, ONNX Runtime, BrainFlow |
| Database | PostgreSQL (Neon serverless) |
| Hosting | Vercel (frontend + API), self-hosted (ML backend) |
| CI/CD | GitHub Actions |

## License

MIT
