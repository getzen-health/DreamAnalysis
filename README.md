# AntarAI

A multimodal AI health platform that fuses EEG brain data, voice analysis, and health device sync (Withings, Oura, WHOOP, Garmin) to track emotions, stress, focus, sleep, nutrition, and wellness -- with 16 ML models running on-device and in the cloud.

**72 pages | 16 ML models | 3 data sources | Supabase backend | Capacitor mobile app**

---

## System Architecture

```mermaid
graph TB
    subgraph Mobile["Mobile App (React 18 + Capacitor)"]
        UI[72 Pages / shadcn-ui]
        Hooks[Custom Hooks<br/>auth, device, fusion, consent]
        Cache[localStorage + Sync Queue]
    end

    subgraph Data["Data Sources"]
        EEG[Muse 2 / Muse S<br/>4-ch EEG via BLE]
        Voice[Voice Mic<br/>30-sec recordings]
        HC[Health Connect<br/>Apple HealthKit]
        Wearables[Withings / Oura<br/>WHOOP / Garmin]
    end

    subgraph Supabase["Supabase"]
        Auth[Auth - JWT + RLS]
        DB[(PostgreSQL<br/>12 tables)]
        Edge[Edge Functions<br/>score compute, health ingest]
    end

    subgraph ML["ML Backend (FastAPI :8080)"]
        Models[16 ML Models<br/>EEGNet, LightGBM, PyTorch]
        Pipeline[EEG Signal Processing<br/>12 modules]
        VoiceML[Voice Biomarkers<br/>eGeMAPS + emotion2vec]
        API[76+ REST Endpoints<br/>+ WebSocket]
    end

    Hosting[Vercel - Frontend Hosting]

    EEG -->|BLE 256 Hz| UI
    Voice -->|MediaRecorder| UI
    HC -->|Capacitor Plugin| Hooks
    Wearables -->|OAuth API| Edge

    UI --> Hooks
    Hooks -->|REST + WS| API
    Hooks -->|REST| Auth
    Cache -->|Offline Sync| DB

    API --> Models
    API --> Pipeline
    API --> VoiceML

    Edge --> DB
    Auth --> DB
    Mobile -->|Deploy| Hosting
```

---

## Data Fusion Architecture

```mermaid
graph LR
    subgraph Inputs
        EEG["EEG (Muse 2/S)<br/>256 Hz, 4 channels"]
        Voice["Voice (Mic)<br/>30-sec sample"]
        Health["Health Sync<br/>HR, Sleep, Steps"]
    end

    subgraph Fusion["Data Fusion Bus"]
        direction TB
        Weights["Source Weights<br/>EEG 50% | Voice 35% | Health 15%"]
        Stale["Stale Discount<br/>>5 min = 50% weight reduction"]
        Adjust["Circadian + Cycle Phase<br/>Adjustment"]
    end

    subgraph State["Unified State"]
        Stress[Stress Index]
        Focus[Focus Index]
        Mood[Mood Score]
        Valence[Valence -1..+1]
        Arousal[Arousal 0..1]
        Emotion[Emotion Label]
    end

    subgraph Pages["UI Pages"]
        Today[Today]
        Discover[Discover]
        HealthP[Health Analytics]
        Scores[Scores Dashboard]
    end

    EEG --> Weights
    Voice --> Weights
    Health --> Weights
    Weights --> Stale --> Adjust
    Adjust --> Stress & Focus & Mood & Valence & Arousal & Emotion
    Stress --> Today & Discover & HealthP & Scores
    Focus --> Today & Discover & HealthP & Scores
    Mood --> Today & Discover & HealthP & Scores
```

---

## EEG Signal Processing Pipeline

```mermaid
graph TD
    Raw["Raw Muse 2 EEG<br/>256 Hz, 4 channels, uV"]
    Reref["Mastoid Re-reference<br/>AF7/AF8 re-ref to avg(TP9, TP10)"]
    BP["Bandpass Filter<br/>Butterworth 1-50 Hz, order 5"]
    Notch["Notch Filters<br/>50 Hz (EU) + 60 Hz (US)"]
    Artifact["Artifact Detection<br/>75 uV threshold, kurtosis > 10"]
    Buffer["Epoch Buffer<br/>4-sec sliding window, 50% overlap"]

    subgraph Features["Feature Extraction"]
        Band["Band Powers<br/>delta, theta, alpha, beta, gamma"]
        FAA["Frontal Alpha Asymmetry<br/>ln(AF8) - ln(AF7)"]
        DASM["DASM / RASM<br/>10 asymmetry features"]
        FMT["Frontal Midline Theta<br/>ACC/mPFC activity"]
        Hjorth["Hjorth Parameters<br/>activity, mobility, complexity"]
        Entropy["Spectral + Differential Entropy"]
    end

    subgraph Models["16 ML Models"]
        Emotion["Emotion Classifier<br/>EEGNet 4-ch, 85% CV"]
        Sleep["Sleep Staging<br/>92.98% accuracy"]
        Dream["Dream Detector<br/>82-88% est."]
        FlowM["Flow State<br/>62.86%"]
        StressM["Stress Detector<br/>4 levels"]
        Others["Attention, Meditation,<br/>Drowsiness, Cognitive Load,<br/>Lucid Dream, Brain Age,<br/>Anomaly, Artifact,<br/>Denoising, Online Learner"]
    end

    StateEngine["State Engine<br/>valence, arousal, stress,<br/>focus, emotion label"]
    UIOut["UI Components<br/>charts, gauges, scores"]

    Raw --> Reref --> BP --> Notch --> Artifact --> Buffer
    Buffer --> Band & FAA & DASM & FMT & Hjorth & Entropy
    Band --> Emotion & Sleep & Dream & FlowM & StressM & Others
    FAA --> Emotion
    DASM --> Emotion
    FMT --> Emotion
    Hjorth --> Emotion & Others
    Entropy --> Others
    Emotion & Sleep & Dream & FlowM & StressM & Others --> StateEngine --> UIOut
```

---

## Database Schema (Supabase)

```mermaid
erDiagram
    mood_logs {
        uuid id PK
        text user_id
        int mood
        int energy
        text notes
        timestamptz created_at
    }

    voice_history {
        uuid id PK
        text user_id
        text emotion
        numeric stress
        numeric focus
        numeric valence
        numeric arousal
        timestamptz created_at
    }

    emotion_history {
        uuid id PK
        text user_id
        numeric stress
        numeric focus
        numeric mood
        text source
        timestamptz created_at
    }

    food_logs {
        uuid id PK
        text user_id
        text summary
        numeric calories
        numeric protein
        numeric carbs
        numeric fat
        numeric food_quality_score
        timestamptz created_at
    }

    cycle_data {
        uuid id PK
        text user_id
        date last_period_start
        int cycle_length
        int period_length
        jsonb logged_days
        timestamptz updated_at
    }

    brain_age {
        uuid id PK
        text user_id
        numeric estimated_age
        numeric actual_age
        numeric gap
        timestamptz created_at
    }

    glp1_injections {
        uuid id PK
        text user_id
        text medication
        numeric dose
        timestamptz injected_at
    }

    supplements {
        uuid id PK
        text user_id
        text name
        text dosage
        boolean taken
        timestamptz created_at
    }

    notifications {
        uuid id PK
        text user_id
        text type
        text title
        text body
        boolean read
        timestamptz created_at
    }

    user_settings {
        uuid id PK
        text user_id
        text key
        text value
        timestamptz updated_at
    }

    generic_store {
        uuid id PK
        text user_id
        text key
        jsonb value
        timestamptz updated_at
    }

    user_feedback {
        uuid id PK
        text user_id
        text predicted_emotion
        text corrected_emotion
        text source
        real confidence
        jsonb features
        text session_id
        timestamptz created_at
    }
```

All tables have Row-Level Security (RLS) with `auth.uid()` policies for per-user data isolation.

---

## Mobile App Page Hierarchy

```mermaid
graph TD
    subgraph BottomTabs["Bottom Tab Bar"]
        T1["Today /"]
        T2["Discover /discover"]
        T3["Voice Mic<br/>(center button)"]
        T4["AI Chat /ai-companion"]
        T5["You /you"]
    end

    subgraph TodaySub["Today Tab"]
        T1 --> BrainReport["/brain-report"]
        T1 --> WeeklySummary["/weekly-summary"]
        T1 --> Scores["/scores"]
        T1 --> QuickSession["/quick-session"]
    end

    subgraph DiscoverSub["Discover Tab"]
        T2 --> Brain["/brain-monitor"]
        T2 --> Sleep["/sleep"]
        T2 --> Health["/health"]
        T2 --> Nutrition["/nutrition"]
        T2 --> Wellness["/wellness"]
        Brain --> BrainConn["/brain-connectivity"]
        Brain --> Neurofeedback["/neurofeedback"]
        Brain --> Biofeedback["/biofeedback"]
        Brain --> DeepWork["/deep-work"]
        Brain --> Calibration["/calibration"]
        Sleep --> SleepSession["/sleep-session"]
        Sleep --> SleepMusic["/sleep-music"]
        Sleep --> CBTI["/cbti"]
        Health --> HealthAnalytics["/health-analytics"]
        Health --> HeartRate["/heart-rate"]
        Health --> Steps["/steps"]
        Health --> BodyMetrics["/body-metrics"]
        Health --> Workout["/workout"]
        T2 --> Stress["/stress"]
        T2 --> Focus["/focus"]
        T2 --> Dreams["/dreams"]
        T2 --> InnerEnergy["/inner-energy"]
        T2 --> FoodEmotion["/food-emotion"]
        T2 --> Insights["/insights"]
        T2 --> Habits["/habits"]
    end

    subgraph AIChatSub["AI Chat Tab"]
        T4 --> CouplesMed["/couples-meditation"]
    end

    subgraph YouSub["You Tab"]
        T5 --> Settings["/settings"]
        T5 --> Sessions["/sessions"]
        T5 --> Records["/records"]
        T5 --> Achievements["/achievements"]
        T5 --> ConnectedAssets["/connected-assets"]
        T5 --> Export["/export"]
        T5 --> Help["/help"]
        T5 --> ConsentSettings["/consent-settings"]
        T5 --> Notifications["/notifications"]
        T5 --> DeviceSetup["/device-setup"]
        T5 --> Supplements["/supplements"]
    end
```

---

## Project Structure

```
AntarAI/
├── client/                     # React 18 + TypeScript frontend
│   ├── src/pages/              # 72 route pages
│   ├── src/components/         # UI components + charts
│   ├── src/hooks/              # React hooks (auth, device, fusion, consent)
│   ├── src/lib/                # Utilities (supabase-store, data-fusion, ml-api,
│   │                           #   health-sync, eeg-compression, i18n, chronotype,
│   │                           #   adaptive-sampling, weather, posthog, etc.)
│   ├── src/locales/            # i18n translations (en, hi, te)
│   └── src/test/               # 129 test files, 1700+ tests (vitest)
│
├── server/                     # Express.js middleware
│   ├── routes.ts               # REST API endpoints
│   └── storage.ts              # Drizzle ORM
│
├── ml/                         # Python ML backend
│   ├── models/                 # 16 ML model classes + saved weights
│   ├── processing/             # EEG signal pipeline (12 modules)
│   ├── training/               # Training scripts + data loaders
│   ├── api/                    # FastAPI routes + auth + CORS + rate limiting
│   ├── benchmarks/             # Model accuracy results (JSON)
│   └── tests/                  # 6400+ pytest tests
│
├── android/                    # Capacitor Android project
├── ios/                        # Capacitor iOS project
├── supabase/                   # Database
│   ├── migrations/             # SQL migrations (12 tables)
│   └── functions/              # Edge Functions (score compute, health ingest)
│
├── scripts/                    # Build tools
│   ├── build-custom-ort.sh     # Custom ONNX WASM build
│   ├── quantize-models.py      # INT8 model quantization
│   └── enable-timescaledb.sql  # EEG time-series optimization
│
├── store-listing/              # Google Play Store assets
├── docs/                       # Documentation
│   ├── APP_PAGES.md            # All 72 pages reference
│   ├── BUSINESS_ROADMAP.md     # Business strategy roadmap
│   └── COMPLETE_SCIENTIFIC_GUIDE.md  # 40KB EEG science reference
│
├── CLAUDE.md                   # AI assistant instructions
└── README.md                   # This file
```

---

## Key Pages (72 total)

See [docs/APP_PAGES.md](docs/APP_PAGES.md) for the full reference.

| Tab | Pages | Key Features |
|-----|-------|-------------|
| **Today** | 1 page | Wellness gauge, mood/stress/focus scores, weather context, cycle phase |
| **Discover** | 1 page | Emotions graph (stress/focus/mood trends), feature navigation |
| **Nutrition** | 1 page | Food logging, GLP-1 tracker, vitamins, meal history, quality score |
| **AI Chat** | 1 page | AI wellness companion with safeguards |
| **You** | 1 page | Profile, streaks, achievements link, connected devices |
| **Brain** | 7 pages | EEG monitor, neurofeedback, biofeedback, deep work, connectivity |
| **Health** | 12 pages | Health sync, analytics, sleep, workout, body metrics, wellness |
| **Settings** | 11 pages | Consent, privacy, export, help, notifications, connected assets |
| **Research** | 13 pages | Study sessions, enrollment, admin |

---

## The 16 ML Models

| Model | Type | Accuracy | Input |
|-------|------|----------|-------|
| Emotion Classifier | EEGNet 4-ch | **85.00% CV** | EEG |
| Sleep Staging | GradientBoosting | **92.98%** | EEG |
| Dream Detector | GradientBoosting | **82-88% est.** | EEG |
| Flow State | Feature-based | **62.86%** (binary ~70%) | EEG |
| Creativity | EXPERIMENTAL | **~60% real** | EEG |
| Stress Detector | Feature-based | 4 levels | EEG |
| Attention | Feature-based | Beta/theta ratio | EEG |
| Meditation | Feature-based | Engagement + stability | EEG |
| Drowsiness | Feature-based | Theta + alpha | EEG |
| Cognitive Load | Feature-based | 3 levels | EEG |
| Lucid Dream | Feature-based | Gamma in REM | EEG |
| Brain Age | Heuristic | Alpha peak regression | EEG |
| Anomaly | Isolation Forest | Unsupervised | EEG |
| Artifact | Rule-based | Blink/muscle/electrode | EEG |
| Denoising | PyTorch autoencoder | Signal reconstruction | EEG |
| Online Learner | Per-user SGD | Adapts over time | EEG |

Additional ML capabilities:
- **Voice biomarkers**: eGeMAPS features (jitter, shimmer, HNR, MFCC)
- **emotion2vec wrapper**: 300M param model (lazy-loaded from HuggingFace)
- **EEGPT wrapper**: 10M param EEG transformer (requires fine-tuning)
- **YASA sleep staging**: Advanced spindle + slow oscillation detection

---

## Security & Compliance

- **API auth**: X-API-Key middleware on all ML endpoints
- **CORS**: Explicit origin whitelist (no wildcard)
- **Rate limiting**: 100 req/min/IP sliding window
- **Path traversal**: `sanitize_id()` on all file-path endpoints
- **RLS**: Per-user data isolation on all Supabase tables
- **HIPAA notifications**: `sanitizeNotificationText()` strips all PHI
- **Biometric consent**: Per-modality toggles (EEG, voice, health, nutrition, location)
- **Privacy mode**: All-local processing, zero cloud sync
- **EU AI Act**: Notice in privacy policy (Annex III high-risk classification)
- **Google Play**: Health app declaration + FDA/wellness disclaimer
- **Regulatory**: Full compliance constants in `regulatory-compliance.ts`

---

## Quick Start

```bash
# Frontend + Express middleware (port 4000)
npm install
npm run dev

# ML backend (port 8080) -- use start.sh
cd ml && ./start.sh

# Android APK
npx cap sync android
# Open Android Studio -> Build -> Build APK
```

## Environment Variables

| Variable | Used By | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | Express | Supabase PostgreSQL connection |
| `SUPABASE_URL` | Client | Supabase project URL |
| `SUPABASE_ANON_KEY` | Client | Supabase anonymous key |
| `VITE_SUPABASE_URL` | Vite | Supabase URL (client build) |
| `VITE_SUPABASE_ANON_KEY` | Vite | Supabase key (client build) |
| `OPENAI_API_KEY` | Express | GPT-5 for dream analysis + AI chat |
| `SESSION_SECRET` | Express | Express session encryption |
| `ML_API_KEY` | ML backend | API key for FastAPI auth |
| `VITE_ML_API_URL` | Client | ML backend URL |
| `VITE_POSTHOG_KEY` | Client | PostHog analytics (optional) |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Tailwind CSS, shadcn/ui, wouter, TanStack Query, Recharts, Framer Motion |
| Mobile | Capacitor (Android + iOS), BLE, Health Connect, HealthKit |
| Database | Supabase PostgreSQL + Auth + Edge Functions + Storage |
| ML Backend | FastAPI, scikit-learn, LightGBM, PyTorch, ONNX Runtime, BrainFlow |
| Data Fusion | Custom event bus (EEG 50% + Voice 35% + Health 15%) |
| Offline | localStorage cache + Supabase sync queue |
| Analytics | PostHog (consent-gated) |
| CI/CD | GitHub Actions |
| Hosting | Vercel (frontend), Railway (ML backend) |

## Testing

```bash
# Frontend -- 1700+ tests
npx vitest run

# ML -- 6400+ tests
cd ml && pytest tests/ -v

# Full suite
npm run test && cd ml && pytest
```

## License

MIT
