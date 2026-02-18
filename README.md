# Neural Dream Workshop

A brain-computer interface (BCI) web application that reads EEG signals from a Muse 2 headband and uses 16 machine-learning models to classify emotions, detect dreams, stage sleep, measure focus, and more — all visualized in a real-time React dashboard.

---

## Architecture

```mermaid
graph TB
    subgraph Browser["Browser"]
        React["React 18 + TypeScript<br/>16 pages, shadcn/ui, Tailwind"]
    end

    subgraph Express["Express.js :5000"]
        Auth["Passport Auth"]
        DB["Drizzle ORM"]
        AI["GPT-5 Chat"]
    end

    subgraph FastAPI["FastAPI :8000"]
        Models["16 ML Models<br/>LightGBM, ONNX, PyTorch"]
        Pipeline["EEG Signal Pipeline<br/>17-feature extraction"]
        WS["WebSocket<br/>Real-time streaming"]
    end

    subgraph Infra["Infrastructure"]
        Neon["PostgreSQL (Neon)"]
        Vercel["Vercel (Hosting)"]
        Muse["Muse 2 (BrainFlow)"]
    end

    React -->|REST| Express
    React -->|REST + WS| FastAPI
    Express --> Neon
    Express --> AI
    FastAPI --> Models
    FastAPI --> Pipeline
    Muse -->|EEG signals| FastAPI
    Vercel -.->|deploys| React
    Vercel -.->|serverless| Express
```

---

## Project Structure

```mermaid
graph LR
    subgraph Root["NeuralDreamWorkshop"]
        direction TB

        subgraph FE["client/ — React Frontend"]
            pages["pages/ (16 routes)<br/>dashboard, brain-monitor, emotion-lab,<br/>dream-journal, neurofeedback, insights..."]
            comps["components/<br/>15 business + 5 charts + 49 shadcn/ui"]
            hooks["hooks/ (7)<br/>auth, device, inference, metrics..."]
            libs["lib/ (8)<br/>ml-api, queryClient, eeg-features..."]
        end

        subgraph MW["server/ — Express Middleware"]
            routes["routes.ts — 10 endpoints"]
            storage["storage.ts — Drizzle ORM"]
            schema["shared/schema.ts — 7 tables"]
        end

        subgraph ML["ml/ — Python ML Backend"]
            api["api/routes.py — 76 endpoints"]
            models["models/ — 16 classifiers"]
            proc["processing/ — 11 signal modules"]
            train["training/ — data loaders + trainers"]
            health["health/ — Apple Health + Google Fit"]
            hw["hardware/ — BrainFlow (Muse 2)"]
            tests["tests/ — 4 pytest files"]
        end

        subgraph API["api/ — Vercel Serverless"]
            sls["auth, dreams, emotions,<br/>health-metrics, ai-chat,<br/>insights, export, settings"]
        end

        subgraph CI["CI/CD"]
            gha[".github/workflows/ci.yml"]
        end
    end
```

---

## Detailed File Tree

```
NeuralDreamWorkshop/
│
├── .github/workflows/
│   └── ci.yml                              CI pipeline (lint, test, build, deploy)
│
├── client/src/                             REACT FRONTEND
│   ├── pages/                              16 route pages
│   │   ├── dashboard.tsx                     Main dashboard
│   │   ├── brain-monitor.tsx                 Real-time EEG visualization
│   │   ├── emotion-lab.tsx                   Emotion analysis lab
│   │   ├── dream-journal.tsx                 Dream recording & analysis
│   │   ├── dream-patterns.tsx                Dream pattern recognition
│   │   ├── neurofeedback.tsx                 Neurofeedback training
│   │   ├── brain-connectivity.tsx            Brain region connectivity
│   │   ├── health-analytics.tsx              Health metrics dashboard
│   │   ├── inner-energy.tsx                  Chakra & spiritual analysis
│   │   ├── session-history.tsx               Past session browser
│   │   ├── insights.tsx                      Weekly insights & trends
│   │   ├── ai-companion.tsx                  AI chat companion
│   │   ├── settings.tsx                      App settings
│   │   ├── auth.tsx                          Login / register
│   │   ├── landing.tsx                       Landing page
│   │   └── not-found.tsx                     404 page
│   ├── components/
│   │   ├── charts/                           5 charts (EEG, sleep, mood, spectrogram, connectivity)
│   │   ├── ui/                               49 shadcn/ui primitives
│   │   ├── brain-bands.tsx                   Frequency band visualizer
│   │   ├── calibration-wizard.tsx            User calibration flow
│   │   ├── device-connection.tsx             Muse 2 pairing UI
│   │   ├── emotion-wheel.tsx                 Valence-arousal wheel
│   │   ├── neural-network.tsx                Network visualization
│   │   ├── session-controls.tsx              Start/stop session
│   │   ├── signal-quality-badge.tsx          Signal quality indicator
│   │   └── sidebar.tsx                       App navigation
│   ├── hooks/                              7 custom hooks
│   │   ├── use-auth.tsx                      Auth state
│   │   ├── use-device.tsx                    BCI device connection
│   │   ├── use-inference.ts                  ML model inference
│   │   ├── use-metrics.tsx                   Health metrics
│   │   └── use-theme.tsx                     Dark/light theme
│   ├── lib/                                8 utility modules
│   │   ├── ml-api.ts                         FastAPI ML client
│   │   ├── ml-local.ts                       Client-side ONNX inference
│   │   ├── eeg-features.ts                   JS feature extraction
│   │   ├── queryClient.ts                    TanStack Query config
│   │   └── offline-store.ts                  Offline data persistence
│   ├── App.tsx                             Router (wouter) + all routes
│   └── main.tsx                            Entry point
│
├── server/                                 EXPRESS MIDDLEWARE
│   ├── index.ts                              Entry point (:5000)
│   ├── routes.ts                             10 REST endpoints
│   ├── storage.ts                            Drizzle ORM data layer
│   └── vite.ts                               Vite dev server integration
│
├── shared/
│   └── schema.ts                             7 Drizzle tables + Zod validators
│
├── api/                                    VERCEL SERVERLESS FUNCTIONS
│   ├── _lib/                                 Shared helpers (auth, db, openai)
│   ├── auth/                                 login, logout, me, register
│   ├── ai-chat/                              AI conversation endpoints
│   ├── dream-analysis/                       Dream analysis CRUD
│   ├── dreams/                               list, create, analytics, generate-image
│   ├── emotions/                             Emotion history + record
│   ├── health-metrics/                       Health data CRUD
│   ├── insights/                             Weekly insights
│   ├── export/                               Data export
│   ├── notifications/                        Push subscription
│   └── settings/                             User settings
│
├── ml/                                     PYTHON ML BACKEND
│   ├── main.py                               FastAPI entry point (:8000)
│   ├── api/
│   │   ├── routes.py                         76 REST endpoints (2K lines)
│   │   └── websocket.py                      Real-time EEG streaming
│   ├── models/                             16 ML model classes
│   │   ├── emotion_classifier.py             6 emotions, LightGBM, 97.79%
│   │   ├── sleep_staging.py                  Wake / N1 / N2 / N3 / REM
│   │   ├── dream_detector.py                 Dream state detection
│   │   ├── flow_state_detector.py            Flow scoring (0–1)
│   │   ├── creativity_detector.py            Creative thinking + memory encoding
│   │   ├── drowsiness_detector.py            Sleepiness detection
│   │   ├── cognitive_load_estimator.py       Mental workload
│   │   ├── attention_classifier.py           Attention level
│   │   ├── stress_detector.py                Stress detection
│   │   ├── lucid_dream_detector.py           Lucid dreaming
│   │   ├── meditation_classifier.py          Meditation depth
│   │   ├── anomaly_detector.py               Unusual EEG (Isolation Forest)
│   │   ├── artifact_classifier.py            Artifact classification
│   │   ├── denoising_autoencoder.py          Signal cleaning (PyTorch)
│   │   ├── online_learner.py                 Per-user adaptation
│   │   └── saved/                            Weights (.onnx, .pkl, .pt)
│   ├── processing/                         11 signal processing modules
│   │   ├── eeg_processor.py                  Feature extraction (17 features)
│   │   ├── artifact_detector.py              Artifact detection & removal
│   │   ├── signal_quality.py                 Signal quality scoring
│   │   ├── calibration.py                    Per-user baseline calibration
│   │   ├── confidence_calibration.py         Model confidence calibration
│   │   ├── state_transitions.py              Brain state engine
│   │   ├── connectivity.py                   Brain region connectivity
│   │   ├── emotion_shift_detector.py         Pre-conscious emotion detection
│   │   ├── noise_augmentation.py             Training data augmentation
│   │   ├── spiritual_energy.py               Chakra / aura / consciousness
│   │   └── user_feedback.py                  Personalized model tuning
│   ├── simulation/
│   │   └── eeg_simulator.py                  Synthetic EEG generation
│   ├── training/                           Model training
│   │   ├── mega_trainer.py                   Multi-algorithm trainer
│   │   ├── data_loaders.py                   8 EEG dataset loaders
│   │   ├── train_emotion.py                  Emotion classifier
│   │   ├── train_sleep.py                    Sleep staging
│   │   ├── train_dream.py                    Dream detector
│   │   └── benchmark.py                      Model benchmarking
│   ├── health/                             Wearable integrations
│   │   ├── apple_health.py                   Apple Health import/export
│   │   ├── google_fit.py                     Google Fit import
│   │   └── correlation_engine.py             Brain-body correlations
│   ├── hardware/
│   │   └── brainflow_manager.py              Muse 2 device manager
│   ├── storage/
│   │   ├── session_recorder.py               Session persistence
│   │   └── session_analytics.py              Trends & comparisons
│   ├── neurofeedback/
│   │   └── protocol_engine.py                Neurofeedback protocols
│   ├── tools/                              CLI utilities
│   │   ├── demo_full_pipeline.py             E2E pipeline demo
│   │   ├── import_apple_health.py            Apple Health XML importer
│   │   └── live_brain_session.py             Live session runner
│   ├── tests/                              pytest test suite
│   │   ├── conftest.py                       Shared fixtures
│   │   ├── test_models.py                    6 core model tests
│   │   ├── test_processing.py                Signal pipeline tests
│   │   ├── test_api.py                       FastAPI endpoint tests
│   │   └── test_accuracy_pipeline.py         Accuracy module tests
│   ├── benchmarks/                           Training results (JSON)
│   └── data/                                 EEG datasets (gitignored)
│
├── docs/
│   └── architecture.html                     Interactive architecture diagram
│
├── vercel.json                             Vercel config
├── package.json                            Node dependencies
├── tsconfig.json                           TypeScript config
├── vite.config.ts                          Vite bundler config
├── tailwind.config.ts                      Tailwind CSS config
├── drizzle.config.ts                       Drizzle ORM config
└── components.json                         shadcn/ui config
```

---

## Data Flow

```mermaid
graph LR
    A["Muse 2<br/>Headband"] -->|raw EEG| B["BrainFlow<br/>Driver"]
    B -->|256 Hz signal| C["Artifact<br/>Detection"]
    C -->|clean signal| D["Bandpass<br/>Filter"]
    D -->|filtered| E["Feature<br/>Extraction<br/>(17 features)"]
    E -->|feature vector| F["16 ML<br/>Models"]
    F -->|predictions| G["State<br/>Engine"]
    G -->|smoothed states| H["React<br/>Dashboard"]

    style A fill:#e1bee7,color:#000
    style F fill:#bbdefb,color:#000
    style H fill:#c8e6c9,color:#000
```

---

## CI/CD Pipeline

```mermaid
graph TD
    trigger["Push to main / PR"]
    trigger --> frontend
    trigger --> ml
    trigger --> audit

    subgraph frontend["Frontend"]
        f1["npm ci"] --> f2["tsc --noEmit"] --> f3["vite build"]
    end

    subgraph ml["ML Backend"]
        m1["pip install"] --> m2["ruff check ."] --> m3["pytest + coverage"] --> m4["Inference benchmark"]
    end

    subgraph audit["Dependency Audit"]
        a1["npm audit"]
        a2["pip-audit"]
    end

    frontend --> deploy
    ml --> deploy

    deploy{"Deploy to Vercel<br/>(main branch only)"}

    style deploy fill:#fff9c4,color:#000
    style frontend fill:#c8e6c9,color:#000
    style ml fill:#bbdefb,color:#000
    style audit fill:#f5f5f5,color:#000
```

---

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

```mermaid
graph TD
    EEG["🧠 Raw EEG Signal<br/>256 Hz, 17 features"]

    EEG --> PRE

    subgraph PRE["Signal Pre-Processing"]
        direction LR
        ART["Artifact Classifier<br/><i>Rule-based + trainable</i><br/>Eye blink / muscle / electrode"]
        DEN["Denoising Autoencoder<br/><i>PyTorch autoencoder</i><br/>Reconstruction MSE"]
        SQ["Signal Quality<br/><i>Multi-metric scorer</i><br/>Usability gating"]
    end

    PRE --> FEAT["Feature Extraction<br/>5 band powers + 3 Hjorth params<br/>+ entropy + ratios + asymmetry<br/><b>= 17 features</b>"]

    FEAT --> TRAINED
    FEAT --> SLEEP
    FEAT --> FOCUS
    FEAT --> SPECIAL

    subgraph TRAINED["Trained Model — LightGBM"]
        EMO["Emotion Classifier<br/><b>97.79% accuracy</b><br/>6 emotions + valence/arousal<br/>Trained on 8 datasets"]
    end

    subgraph SLEEP["Sleep & Dream Models — Feature-Based"]
        direction LR
        SS["Sleep Staging<br/><i>Delta/theta/alpha ratios</i><br/>Wake · N1 · N2 · N3 · REM"]
        DD["Dream Detector<br/><i>Theta + REM signatures</i><br/>Dreaming detection"]
        LD["Lucid Dream<br/><i>Gamma bursts in REM</i><br/>Lucid state detection"]
        DROW["Drowsiness<br/><i>Theta increase + alpha drop</i><br/>Sleepiness detection"]
    end

    subgraph FOCUS["Cognition & Emotion Models — Feature-Based"]
        direction LR
        FS["Flow State<br/><i>Alpha/theta + beta stability</i><br/>0–1 flow score"]
        CR["Creativity<br/><i>Alpha/theta + gamma</i><br/>Creative thinking"]
        ME["Memory Encoding<br/><i>Theta + gamma coupling</i><br/>Formation strength"]
        CL["Cognitive Load<br/><i>Beta/alpha + frontal theta</i><br/>Low / Medium / High"]
        AT["Attention<br/><i>Beta/theta ratio</i><br/>Attention level"]
        ST["Stress<br/><i>Beta asymmetry</i><br/>Stress detection"]
        MED["Meditation<br/><i>Alpha coherence + theta</i><br/>Meditation depth"]
    end

    subgraph SPECIAL["Adaptive Models"]
        direction LR
        AN["Anomaly Detector<br/><i>Isolation Forest</i><br/>Unsupervised outlier flagging"]
        OL["Online Learner<br/><i>Per-user SGD</i><br/>Adapts to individual users"]
    end

    TRAINED --> OUT
    SLEEP --> OUT
    FOCUS --> OUT
    SPECIAL --> OUT

    OUT["State Engine<br/>Smoothing · Transition validation<br/>Coherence checks"]
    OUT --> DASH["📊 React Dashboard<br/>Real-time visualization"]

    style EEG fill:#e1bee7,stroke:#7b1fa2,color:#000
    style FEAT fill:#fff3e0,stroke:#e65100,color:#000
    style TRAINED fill:#c8e6c9,stroke:#2e7d32,color:#000
    style EMO fill:#a5d6a7,stroke:#2e7d32,color:#000
    style SLEEP fill:#bbdefb,stroke:#1565c0,color:#000
    style FOCUS fill:#b3e5fc,stroke:#0277bd,color:#000
    style SPECIAL fill:#f5f5f5,stroke:#616161,color:#000
    style PRE fill:#fce4ec,stroke:#c62828,color:#000
    style OUT fill:#fff9c4,stroke:#f9a825,color:#000
    style DASH fill:#c8e6c9,stroke:#2e7d32,color:#000
```

> **Why only 1 model has accuracy?** The Emotion Classifier is the only model trained on labeled datasets (SEED, GAMEEMO, Brainwave, etc.). The other 15 models use peer-reviewed EEG neuroscience heuristics — band power ratios, spectral entropy, Hjorth parameters — because no large labeled datasets exist for subjective states like "flow" or "creativity". They work without trained weights by applying established biomarker rules.

## Testing

```bash
# ML tests with coverage
cd ml && pytest tests/ -v --cov=. --cov-report=term-missing

# TypeScript type checking
npx tsc --noEmit

# Full frontend build
npm run build
```

## Deployment

- **Frontend + Express** — Vercel (see `VERCEL_DEPLOYMENT.md`)
- **ML Backend** — Docker or standalone (`uvicorn main:app`)
- **Database** — Neon PostgreSQL (`drizzle-kit push` for migrations)

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
