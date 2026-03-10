# Pages

Each file is one route in the app. Routes are defined in `App.tsx`. Authenticated pages are wrapped with `<AppLayout>`.

## Modality Tiers

| Tier | Description |
|------|-------------|
| `health` | Works with voice + watch only; no EEG required |
| `multimodal` | EEG enhances but is not required; voice/health fallback exists |
| `eeg` | Requires or strongly prefers Muse 2 streaming |

## Page Inventory

### Public (no auth)

| Route | File | Notes |
|-------|------|-------|
| `/welcome` | `landing.tsx` | Entry point → /onboarding-new |
| `/auth` | `auth.tsx` | Login / register |
| `/onboarding-new` | `onboarding-new.tsx` | Voice-first onboarding path |

### Voice + Watch First (health + multimodal)

| Route | File | Tier | Description |
|-------|------|------|-------------|
| `/` | `dashboard.tsx` | health | Dashboard — voice check-in, streak, emotion landscape |
| `/brain-report` | `daily-brain-report.tsx` | multimodal | Today's report — voice + sleep + EEG fallback |
| `/weekly-summary` | `weekly-brain-summary.tsx` | multimodal | Weekly emotion/focus trends |
| `/insights` | `insights.tsx` | multimodal | AI narrative — voice fallback when no EEG |

### Mind & Recovery

| Route | File | Tier | Description |
|-------|------|------|-------------|
| `/emotional-intelligence` | `emotional-intelligence.tsx` | health | EI dashboard — composite score from voice + history |
| `/emotions` | `emotion-lab.tsx` | multimodal | Real-time emotion — EEG primary, voice fallback |
| `/health-analytics` | `health-analytics.tsx` | multimodal | Health metric correlations — voice fallback |
| `/sessions` | `session-history.tsx` | multimodal | Past session records |

### Health & Life

| Route | File | Tier | Description |
|-------|------|------|-------------|
| `/food` | `food-log.tsx` | health | Food + mood journaling |
| `/supplements` | `supplements.tsx` | health | Supplement + medication tracker |
| `/dreams` | `dream-journal.tsx` | multimodal | Dream journaling — manual-first, EEG auto-detection optional |
| `/dream-patterns` | `dream-patterns.tsx` | multimodal | Dream pattern trends |
| `/sleep-session` | `sleep-session.tsx` | multimodal | Sleep + HRV session log |
| `/biofeedback` | `biofeedback.tsx` | health | Guided breathwork — voice stress baseline, EEG optional |

### Support

| Route | File | Tier | Description |
|-------|------|------|-------------|
| `/ai-companion` | `ai-companion.tsx` | health | AI wellness chat |
| `/records` | `personal-records.tsx` | health | Personal bests + achievements |

### Add EEG Later

These routes remain available in the app shell, but only as an optional hardware layer below the main voice + watch flow.

| Route | File | Tier | Description |
|-------|------|------|-------------|
| `/brain-monitor` | `brain-monitor.tsx` | eeg | Live EEG waveforms — voice fallback panel |
| `/brain-connectivity` | `brain-connectivity.tsx` | eeg | Brain region connectivity |
| `/inner-energy` | `inner-energy.tsx` | multimodal | Chakra visualization — voice-derived fallback |
| `/neurofeedback` | `neurofeedback.tsx` | eeg | Neurofeedback training protocols |
| `/calibration` | `calibration.tsx` | eeg | EEG baseline calibration |
| `/device-setup` | `device-setup.tsx` | eeg | Muse 2 hardware setup |

### Hidden From Default Navigation

These routes still exist, but they are intentionally hidden from the default app shell because they are research-only or internal.

| Route | File | Tier | Description |
|-------|------|------|-------------|
| `/food-emotion` | `food-emotion.tsx` | health | Food-emotion research surface |

### Auth & Utility (no auth)

| Route | File | Notes |
|-------|------|-------|
| `/forgot-password` | `forgot-password.tsx` | Password reset request |
| `/reset-password` | `reset-password.tsx` | Password reset confirmation |
| `/privacy-policy` | `privacy-policy.tsx` | Privacy policy page |
| `/intent` | `intent-select.tsx` | Onboarding intent selection |
| `*` | `not-found.tsx` | 404 fallback |

### Research Study Flow

| Route | File | Description |
|-------|------|-------------|
| `/research` | `research-evening.tsx` | My Day — evening journal |
| `/research/morning` | `research-morning.tsx` | Morning dream + mood journal |
| `/research/daytime` | `research-daytime.tsx` | Daytime check-in |
| `/research/hub` | `research-hub.tsx` | Study participant hub |
| `/research/enroll` | `research-enroll.tsx` | Study enrollment |

### Dev / Admin Tools

| Route | File | Description |
|-------|------|-------------|
| `/architecture-guide` | `architecture-guide.tsx` | Internal project guide |
| `/benchmarks` | `formal-benchmarks-dashboard.tsx` | Model accuracy benchmarks |
| `/settings` | `settings.tsx` | User preferences + device config |

## How to Add a New Page

1. Create `my-page.tsx` here (kebab-case, default export)
2. Import in `App.tsx` and add a `<Route>` wrapped in `<AppLayout>`
3. Add a sidebar link in `../components/sidebar.tsx` (place in correct tier section)
4. If it needs ML data, add the API call in `../lib/ml-api.ts`
5. Every page must work at tier `health` or be explicitly labeled `eeg`
