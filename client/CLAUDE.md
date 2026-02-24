# Client — React Frontend

The browser-facing half of Neural Dream Workshop. A single-page app with 17 pages for real-time EEG visualization, emotion analysis, dream journaling, neurofeedback training, and health analytics.

## Tech Stack

| Dependency | Purpose |
|-----------|---------|
| React 18 | UI framework |
| TypeScript 5.6 | Type safety |
| Vite 5 | Build tool + dev server |
| wouter | Client-side routing (lightweight, not react-router) |
| TanStack Query | Server state management (caching, refetching) |
| shadcn/ui + Radix | 49 pre-built accessible UI components |
| Tailwind CSS 3 | Utility-first styling, dark theme default |
| Recharts | Primary charting library |
| Chart.js | Secondary charts (some legacy pages) |
| Framer Motion | Animations |
| Lucide React | Icon library |
| onnxruntime-web | Client-side ML inference (ONNX models) |
| react-hook-form + zod | Form validation |

## Folder Structure

| Path | What | Files |
|------|------|-------|
| `src/pages/` | Route pages (one per URL) | 17 |
| `src/components/` | Business components | 15 |
| `src/components/charts/` | Chart visualizations | 5 |
| `src/components/ui/` | shadcn/ui primitives (don't edit) | 49 |
| `src/hooks/` | Custom React hooks | 7 |
| `src/lib/` | Utilities + API clients | 8 |
| `src/layouts/` | Page layout wrappers | 1 |

## Data Flow

```
Pages (UI)
  └─▶ Hooks (use-auth, use-device, use-metrics, use-inference)
        └─▶ API Clients
              ├─▶ lib/queryClient.ts ──▶ Express (:5000) ──▶ PostgreSQL
              └─▶ lib/ml-api.ts ──▶ FastAPI (:8000) ──▶ ML Models
```

## How to Add a New Page

1. Create `src/pages/my-page.tsx` — default export, kebab-case filename
2. Add route in `App.tsx`: `<Route path="/my-page"><AppLayout><MyPage /></AppLayout></Route>`
3. Add sidebar nav link in `src/components/sidebar.tsx`
4. If it needs ML data, add API call in `src/lib/ml-api.ts`

## The 17 Pages

| Route | Component | Description |
|-------|-----------|-------------|
| `/welcome` | `landing.tsx` | Welcome screen, no auth required |
| `/auth` | `auth.tsx` | Login / register |
| `/` | `dashboard.tsx` | Main dashboard with health metrics |
| `/emotions` | `emotion-lab.tsx` | Real-time emotion classification |
| `/inner-energy` | `inner-energy.tsx` | Spiritual energy + chakra visualization |
| `/brain-monitor` | `brain-monitor.tsx` | Live EEG waveform display |
| `/brain-connectivity` | `brain-connectivity.tsx` | Brain region connectivity analysis |
| `/dreams` | `dream-journal.tsx` | Dream recording + AI analysis |
| `/dream-patterns` | `dream-patterns.tsx` | Dream pattern trends |
| `/health-analytics` | `health-analytics.tsx` | Health metric correlations |
| `/neurofeedback` | `neurofeedback.tsx` | Neurofeedback training protocols |
| `/ai-companion` | `ai-companion.tsx` | AI wellness chat (GPT-5) |
| `/insights` | `insights.tsx` | AI-generated wellness insights |
| `/sessions` | `session-history.tsx` | Past EEG session records |
| `/settings` | `settings.tsx` | User preferences + device config |
| `404` | `not-found.tsx` | Not found page |

## Key Hooks

| Hook | Purpose |
|------|---------|
| `use-auth` | Authentication state, login/logout, user context |
| `use-device` | EEG device connection status, connect/disconnect |
| `use-inference` | Client-side ONNX model inference |
| `use-metrics` | Health metric data fetching |
| `use-mobile` | Responsive breakpoint detection |
| `use-theme` | Light/dark theme toggle |
| `use-toast` | Toast notification system |

## Post-Task Checklist (MANDATORY after every completed task)

After finishing any task, always do these in order — no exceptions:

1. **Push to GitHub**: `git push`
2. **Deploy to Vercel**: Vercel auto-deploys on push to `main`. If not auto-connected,
   run `vercel --prod` from the project root.
3. **Update `STATUS.md`**: Mark completed items [x], update model accuracies, add new
   endpoints or pages built.
4. **Update `PRODUCT.md`**: Update the "Honest Assessment" percentages and "What Is Broken"
   section if anything was fixed.
5. **Update benchmark dashboard** (`client/src/pages/formal-benchmarks-dashboard.tsx`):
   Update model accuracy numbers, dataset statuses, and publishing-plan checkboxes to
   reflect the latest results.

## Git Commit Rules

- **NEVER add `Co-Authored-By: Claude` (or any Claude/AI co-author line) to commit messages.** Claude must not appear as a contributor in the git history.
