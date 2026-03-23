# AntarAI

A multimodal AI health platform that fuses EEG brain data, voice analysis, and health device sync (Withings, Oura, WHOOP, Garmin) to track emotions, stress, focus, sleep, nutrition, and wellness — with 16 ML models running on-device and in the cloud.

**73 pages | 16 ML models | 3 data sources | Supabase backend | Capacitor mobile app**

---

## Architecture

```
                          ┌─────────────────────────────────────┐
                          │         AntarAI Mobile App          │
                          │    React 18 + TypeScript + Capacitor │
                          │         73 pages, shadcn/ui          │
                          └──────────┬──────────┬───────────────┘
                                     │          │
                    ┌────────────────┘          └────────────────┐
                    ▼                                            ▼
          ┌─────────────────┐                        ┌──────────────────┐
          │   Supabase       │                        │   FastAPI ML     │
          │   PostgreSQL     │                        │   Backend :8080  │
          │                  │                        │                  │
          │ • 9 data tables  │                        │ • 16 ML models   │
          │ • user_settings  │                        │ • EEG processing │
          │ • generic_store  │                        │ • Voice analysis │
          │ • Auth (JWT)     │                        │ • 76+ endpoints  │
          │ • RLS per user   │                        │ • API key auth   │
          │ • Edge Functions │                        │ • Rate limiting  │
          └─────────────────┘                        └────────┬─────────┘
                                                              │
                                                     ┌────────┴────────┐
                                                     │   Data Sources   │
                                                     │                  │
                                                     │ • Muse 2/S (BLE)│
                                                     │ • Voice (mic)    │
                                                     │ • Health Connect │
                                                     │ • Apple HealthKit│
                                                     │ • Withings       │
                                                     │ • Oura / WHOOP   │
                                                     └─────────────────┘
```

---

## Data Flow — Three Input Sources

```
┌──────────┐     ┌──────────┐     ┌───────────────┐
│  EEG     │     │  Voice   │     │ Health Sync   │
│ Muse 2/S │     │  Mic     │     │ Withings/Oura │
│ 256 Hz   │     │ 30 sec   │     │ HR/Sleep/Steps│
└────┬─────┘     └────┬─────┘     └──────┬────────┘
     │                │                   │
     ▼                ▼                   ▼
┌────────────────────────────────────────────────┐
│              Data Fusion Bus                    │
│  EEG: 50% weight | Voice: 35% | Health: 15%   │
│  Stale readings (>5 min) discounted 50%        │
│  + Circadian adjustment + Cycle phase adjust   │
└────────────────────────┬───────────────────────┘
                         │
                         ▼
              ┌─────────────────┐
              │  Unified State   │
              │ stress, focus,   │
              │ mood, valence,   │
              │ arousal, emotion │
              └────────┬────────┘
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
         Today     Discover    Health
         Page       Page       Page
```

---

## Project Structure

```
AntarAI/
├── client/                     # React 18 + TypeScript frontend
│   ├── src/pages/              # 73 route pages
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
│   │   ├── emotion_classifier.py      # 85% CV (EEGNet 4-ch, active path)
│   │   ├── sleep_staging.py           # 92.98% (ISRUC dataset)
│   │   ├── dream_detector.py          # 82-88% est. (real data pipeline)
│   │   ├── flow_state_detector.py     # 62.86% (binary ~70-75%)
│   │   ├── creativity_detector.py     # EXPERIMENTAL (overfit, ~60% real)
│   │   ├── stress_detector.py         # 4 stress levels
│   │   ├── attention_classifier.py    # Beta/theta ratio
│   │   ├── meditation_classifier.py   # Engagement/stability (validated)
│   │   ├── drowsiness_detector.py     # Theta + alpha slowing
│   │   ├── cognitive_load_estimator.py # 3 workload levels
│   │   ├── lucid_dream_detector.py    # Gamma bursts in REM
│   │   ├── brain_age_estimator.py     # Alpha peak regression
│   │   ├── anomaly_detector.py        # Isolation Forest
│   │   ├── artifact_classifier.py     # Blink/muscle/electrode
│   │   ├── denoising_autoencoder.py   # PyTorch signal cleaner
│   │   └── online_learner.py          # Per-user adaptation
│   ├── processing/             # EEG signal pipeline (12 modules)
│   ├── training/               # Training scripts + data loaders
│   ├── api/                    # FastAPI routes + auth + CORS + rate limiting
│   ├── benchmarks/             # Model accuracy results (JSON)
│   └── tests/                  # 6400+ pytest tests
│
├── android/                    # Capacitor Android project
├── ios/                        # Capacitor iOS project
├── supabase/                   # Database
│   ├── migrations/             # 6 SQL migrations
│   │   ├── 004_app_data.sql    # 9 biometric tables
│   │   ├── 005_fix_rls.sql     # auth.uid() RLS policies
│   │   └── 006_user_settings.sql # settings + generic store
│   └── functions/              # Edge Functions (score compute, health ingest)
│
├── scripts/                    # Build tools
│   ├── build-custom-ort.sh     # Custom ONNX WASM build
│   ├── quantize-models.py      # INT8 model quantization
│   ├── enable-timescaledb.sql  # EEG time-series optimization
│   └── generate_readme.py      # Auto-generate README from registry
│
├── store-listing/              # Google Play Store assets
├── docs/                       # Documentation
│   ├── APP_PAGES.md            # All 73 pages reference
│   ├── BUSINESS_ROADMAP.md     # Business strategy roadmap
│   └── COMPLETE_SCIENTIFIC_GUIDE.md  # 40KB EEG science reference
│
├── CLAUDE.md                   # AI assistant instructions
└── README.md                   # This file
```

---

## Key Pages (73 total)

See [docs/APP_PAGES.md](docs/APP_PAGES.md) for the full reference.

| Tab | Pages | Key Features |
|-----|-------|-------------|
| **Today** | 1 page | Wellness gauge, mood/stress/focus scores, weather context, cycle phase |
| **Discover** | 1 page | Emotions graph (stress/focus/mood trends), feature navigation |
| **Nutrition** | 1 page | Food logging, GLP-1 tracker, vitamins, meal history, quality score |
| **AI Chat** | 1 page | GPT-5 wellness companion with safeguards |
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

## Database (Supabase)

| Table | Purpose |
|-------|---------|
| `mood_logs` | Mood + energy tracking |
| `voice_history` | Voice analysis results |
| `emotion_history` | Emotion readings over time |
| `food_logs` | Nutrition tracking |
| `cycle_data` | Menstrual cycle tracking |
| `brain_age` | Brain age readings |
| `glp1_injections` | GLP-1 medication tracking |
| `supplements` | Supplement tracking |
| `notifications` | HIPAA-safe notifications |
| `user_settings` | App preferences (key-value) |
| `generic_store` | JSON blob storage |

All tables have Row-Level Security (RLS) with `auth.uid()` initPlan pattern.

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

# ML backend (port 8080) — use start.sh
cd ml && ./start.sh

# Android APK
npx cap sync android
# Open Android Studio → Build → Build APK
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
# Frontend — 1700+ tests
npx vitest run

# ML — 6400+ tests
cd ml && pytest tests/ -v

# Full suite
npm run test && cd ml && pytest
```

## License

MIT
