# Pages

Each file is one route in the app. Routes are defined in `App.tsx`. Authenticated pages are wrapped with `<AppLayout>`.

## All 17 Pages

| Route | File | Description |
|-------|------|-------------|
| `/welcome` | `landing.tsx` | Welcome screen, no auth required |
| `/auth` | `auth.tsx` | Login / register forms |
| `/` | `dashboard.tsx` | Main dashboard — health metrics, neural activity overview |
| `/emotions` | `emotion-lab.tsx` | Real-time emotion classification from EEG |
| `/inner-energy` | `inner-energy.tsx` | Chakra + spiritual energy visualization |
| `/brain-monitor` | `brain-monitor.tsx` | Live EEG waveform display + band powers |
| `/brain-connectivity` | `brain-connectivity.tsx` | Brain region connectivity analysis |
| `/dreams` | `dream-journal.tsx` | Dream text/voice recording + AI analysis |
| `/dream-patterns` | `dream-patterns.tsx` | Dream symbol trends + recurring patterns |
| `/health-analytics` | `health-analytics.tsx` | Health metric correlations + trends |
| `/neurofeedback` | `neurofeedback.tsx` | Neurofeedback training protocols |
| `/ai-companion` | `ai-companion.tsx` | AI wellness chat powered by GPT-5 |
| `/insights` | `insights.tsx` | AI-generated wellness insights |
| `/sessions` | `session-history.tsx` | Past EEG session records |
| `/settings` | `settings.tsx` | User preferences, device config, themes |
| `404` | `not-found.tsx` | Not found fallback |

## How to Add a New Page

1. Create `my-page.tsx` here (kebab-case, default export)
2. Import in `App.tsx` and add a `<Route>` wrapped in `<AppLayout>`
3. Add a sidebar link in `../components/sidebar.tsx`
4. If it needs ML data, add the API call in `../lib/ml-api.ts`
