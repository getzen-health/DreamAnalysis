# Agent Guide — Neural Dream Workshop

## Project Structure

```
NeuralDreamWorkshop/
├── client/src/         # React 18 + TypeScript (17 pages)
│   ├── pages/          # 17 route pages
│   ├── components/     # 15 business + 5 charts + 49 shadcn/ui
│   ├── hooks/          # 7 custom hooks (auth, device, inference, etc.)
│   ├── lib/            # 8 utilities (API clients, query, simulation)
│   └── layouts/        # AppLayout wrapper
├── server/             # Express.js middleware
│   ├── index.ts        # Entry point (:5000)
│   ├── routes.ts       # 10 REST endpoints (DB + OpenAI)
│   ├── storage.ts      # Drizzle ORM data layer
│   └── vite.ts         # Dev server integration
├── shared/schema.ts    # 7 Drizzle tables + Zod validators
├── ml/                 # Python ML backend
│   ├── main.py         # FastAPI entry (:8000)
│   ├── api/routes.py   # 76 endpoints (2017 lines)
│   ├── models/         # 15 model classes + saved weights
│   ├── processing/     # 11 signal processing modules
│   ├── training/       # Training scripts + data loaders
│   ├── health/         # Apple Health + Google Fit
│   └── hardware/       # BrainFlow (Muse 2)
└── docs/               # Scientific reference guide
```

## Architecture

Three-tier system:

1. **React frontend** — 17 pages, wouter routing, TanStack Query for server state
2. **Express middleware** — Auth (Passport), DB (Drizzle/Neon), AI chat (GPT-5)
3. **FastAPI ML backend** — 16 models, EEG processing, WebSocket streaming

Frontend talks to Express for CRUD + AI, and directly to FastAPI for ML inference.

## Critical Patterns

### Frontend
- **Router**: wouter `<Switch>` in `App.tsx`, all routes listed there
- **Hooks**: `use-auth`, `use-device`, `use-inference`, `use-metrics`, `use-mobile`, `use-theme`, `use-toast`
- **API calls**: `lib/ml-api.ts` for ML backend, `lib/queryClient.ts` for Express
- **Import aliases**: `@/` → `client/src/`
- **Styling**: Tailwind only, dark theme default, no CSS modules

### Backend (Express)
- `server/routes.ts` — registers all Express routes, returns HTTP server
- `server/storage.ts` — Drizzle ORM CRUD (7 tables defined in `shared/schema.ts`)
- `server/vite.ts` — Vite dev server middleware (dev only)

### Backend (ML)
- `ml/api/routes.py` — all 76 endpoints in one file (line-number map in `ml/api/README.md`)
- Model pattern: class with `__init__(model_path)` → auto-discover ONNX/pkl → `predict(features)` → dict
- Processing pipeline: raw EEG → artifact detection → bandpass filter → feature extraction → model inference
- 17-feature vectors: 5 band powers + 3 Hjorth params + entropy + band ratios + asymmetry

## Common Pitfalls

1. **routes.py is 2K lines** — Don't add endpoints randomly. Check the category sections.
2. **`.gitignore` excludes ML data** — `ml/data/`, all `.joblib`/`.pkl`/`.pt` files are gitignored.
3. **Replit plugins guarded** — `@replit/vite-plugin-*` in package.json only loads on Replit.
4. **Dual charting libs** — Recharts (main) + Chart.js (legacy pages). Prefer Recharts for new work.
5. **Large model files** — `emotion_classifier_rf.joblib` is 3.1GB. Gitignored.
6. **No frontend tests** — No Vitest/Jest configured yet.

## Testing

```bash
# ML tests
cd ml && pytest tests/

# TypeScript type checking
npx tsc --noEmit
```

## When Adding Features

### New Page
1. Create `client/src/pages/my-page.tsx` (kebab-case)
2. Add route in `client/src/App.tsx` wrapped with `<AppLayout>`
3. Add sidebar link in `client/src/components/sidebar.tsx`

### New ML Model
1. Create class in `ml/models/my_model.py` with `predict(features) → dict`
2. Save weights to `ml/models/saved/my_model.pkl`
3. Initialize in `ml/api/routes.py` with `_find_model("my_model")`
4. Add endpoint in `ml/api/routes.py`
5. Add client call in `client/src/lib/ml-api.ts`

### New DB Table
1. Add table in `shared/schema.ts`
2. Create insert schema + types
3. Add CRUD methods in `server/storage.ts`
4. Add routes in `server/routes.ts`
5. Run `npx drizzle-kit push`
