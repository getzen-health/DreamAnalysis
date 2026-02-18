# Neural Dream Workshop

A brain-computer interface (BCI) web application that reads EEG signals from a Muse 2 headband and uses 16 machine-learning models to classify emotions, detect dreams, stage sleep, measure focus, and more — all visualized in a real-time React dashboard.

Built as a full-stack system: React frontend, Express.js middleware, FastAPI ML backend, PostgreSQL database.

## Quick Start

```bash
# Frontend + Express middleware (port 5000)
npm install && npm run dev

# ML backend (port 8000)
cd ml && pip install -r requirements.txt && uvicorn main:app --reload --port 8000
```

## Architecture

```
Browser (React)
    │
    ├── REST ──▶ Express.js (:5000) ──▶ PostgreSQL (Neon)
    │               │
    │               └── /api/dream-analysis, /api/ai-chat (GPT-5)
    │
    └── REST + WS ──▶ FastAPI (:8000)
                        │
                        ├── 16 ML models (LightGBM, ONNX, PyTorch)
                        ├── EEG signal processing pipeline
                        └── BrainFlow (Muse 2 hardware)
```

## Directory Map

| Folder | What It Is |
|--------|-----------|
| `client/` | React 18 + TypeScript frontend (17 pages, shadcn/ui, Tailwind) |
| `server/` | Express.js middleware — auth, DB, AI chat, data export |
| `shared/` | Drizzle ORM schema shared between client and server |
| `ml/` | Python ML backend — 16 models, 76 API endpoints, signal processing |
| `ml/models/` | Model classes + saved weights (ONNX, pkl, joblib) |
| `ml/processing/` | EEG signal processing pipeline (11 modules) |
| `ml/training/` | Training scripts + data loaders for 8 EEG datasets |
| `ml/health/` | Apple Health + Google Fit integration |
| `ml/hardware/` | BrainFlow EEG device manager (Muse 2) |
| `ml/api/` | FastAPI routes (2K-line routes.py) + WebSocket |
| `api/` | Vercel serverless function stubs |
| `docs/` | Scientific guide (40KB reference on EEG + ML) |
| `attached_assets/` | Screenshots and generated dream images |

## Key Conventions

- **Routing**: [wouter](https://github.com/molefrog/wouter) (not react-router)
- **UI Components**: [shadcn/ui](https://ui.shadcn.com/) in `client/src/components/ui/`
- **Server State**: TanStack Query (no Redux)
- **Styling**: Tailwind CSS, dark theme by default
- **Charts**: Recharts (main) + Chart.js (some pages)
- **ML Model Loading**: Auto-discovery with fallback chain: ONNX → pkl → feature-based
- **Import Aliases**: `@/` maps to `client/src/`
- **EEG Standard**: 256 Hz sampling rate, 17-feature vectors, numpy arrays

## Environment Variables

| Variable | Used By | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | Express | Neon PostgreSQL connection string |
| `OPENAI_API_KEY` | Express | GPT-5 for dream analysis + AI chat |
| `SESSION_SECRET` | Express | Express session encryption |
| `PORT` | Express | Server port (default 5000) |

## Deployment

- **Frontend + Express**: Vercel (see `vercel.json`)
- **ML Backend**: Docker or standalone (`uvicorn main:app`)
- **Database**: Neon PostgreSQL (`drizzle-kit push` for migrations)

## Key Files

| File | Why It Matters |
|------|---------------|
| `client/src/App.tsx` | All 17 routes defined here |
| `server/routes.ts` | Express API (10 endpoints: health, dreams, chat, settings, export) |
| `ml/api/routes.py` | FastAPI ML API (76 endpoints, 2K lines — read the README) |
| `shared/schema.ts` | Database schema (7 tables: users, health, dreams, emotions, chats, settings, push) |
| `ml/main.py` | FastAPI app entry point |
| `docs/COMPLETE_SCIENTIFIC_GUIDE.md` | 40KB scientific reference on EEG signal processing + ML |
