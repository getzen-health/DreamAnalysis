# NeuralDreamWorkshop — Thursday Launch Design

**Date:** 2026-03-03
**Deadline:** Thursday 2026-03-06
**Author:** Sravya Vedantham
**Goal:** App works end-to-end on localhost and Vercel production by Thursday

---

## Problem Statement

Three failure modes block the app from "working":

1. **Cold start (P0):** Render free tier sleeps after 15 min of inactivity. ML backend takes 30-60 seconds to wake up. Users land on blank/error screen with no feedback.
2. **No resilience (P1):** A single mlFetch failure throws immediately. No retry, no fallback, no graceful degradation.
3. **Deployment gap (P1):** `VITE_ML_API_URL` may not be set in Vercel dashboard. CORS origins in render.yaml may not match production Vercel URL. App works locally but fails in production.
4. **Model gap (P2):** TSception (69% CV) is trained but not wired into the live inference chain. Feature heuristics (65%) are used as fallback instead.

---

## Approach Selected: Full Connection Resilience + Model Improvement

Three options were considered:
- **A (Minimal):** Loading screen only. Rejected — still breaks on retry failure and doesn't fix production env.
- **B (Full — SELECTED):** Complete solution covering all four failure modes. Achievable by Thursday.
- **C (Paid infra):** Upgrade Render tier to avoid sleep. Rejected — costs money, overkill for prototype.

---

## Architecture

### Connection Lifecycle

```
User opens app
    │
    ├── Is user authenticated?
    │       │
    │       NO → render /welcome, /auth normally (no ML needed)
    │       │
    │       YES → MLConnectionProvider mounts
    │               │
    │               └── ping /health every 5s
    │                       │
    │                       ├── 200 OK → status='ready' → show app
    │                       ├── pending (0-40s) → status='warming' → show MLWarmupScreen
    │                       └── 3 failures → status='error' → show SimulationModeBanner
```

### Keep-Alive Loop (prevents future cold starts)

```
AppLayout mounts
    │
    └── setInterval(14 min):
            if (tab visible AND user authenticated):
                navigator.sendBeacon(ML_HEALTH_URL)
```

### ML Inference Fallback Chain (Backend)

```
POST /api/analyze-eeg
    │
    ├── mega_lgbm loaded AND CV >= 74%? → use it (primary)
    ├── TSception loaded AND epoch >= 4s? → use it (fallback, 69%)
    └── else → _predict_features heuristics (last resort, 65%)
```

### Frontend mlFetch Retry Logic

```
mlFetch(endpoint)
    │
    ├── attempt 1 → network/5xx error? → wait 1s
    ├── attempt 2 → network/5xx error? → wait 3s
    ├── attempt 3 → network/5xx error? → wait 9s
    └── attempt 4 → throw (caller handles)

4xx errors → throw immediately (no retry)
All calls: 30s AbortController timeout
```

---

## Components

### New Frontend Files

| File | Purpose |
|------|---------|
| `client/src/hooks/use-ml-connection.ts` | Connection state machine, health ping loop |
| `client/src/components/ml-warmup-screen.tsx` | Full-screen warm-up animation (35s progress bar) |
| `client/src/components/simulation-mode-banner.tsx` | Amber banner shown when ML unreachable |

### Modified Frontend Files

| File | Change |
|------|--------|
| `client/src/App.tsx` | Wrap authenticated routes with `MLConnectionProvider` |
| `client/src/layouts/app-layout.tsx` | Add keep-alive ping every 14 min |
| `client/src/components/sidebar.tsx` | Add 8px status dot (green/amber/red) + tooltip |
| `client/src/lib/ml-api.ts` | Add retry wrapper + 30s timeout to mlFetch |
| `client/src/pages/emotion-lab.tsx` | Render SimulationModeBanner when status='error' |
| `client/src/pages/brain-monitor.tsx` | Render SimulationModeBanner when status='error' |

### New Backend Files

| File | Purpose |
|------|---------|
| `ml/processing/eeg_processor.py` | Add `RunningNormalizer` class (session drift correction) |

### Modified Backend Files

| File | Change |
|------|--------|
| `ml/models/emotion_classifier.py` | Wire TSception into fallback chain |

### Configuration Files

| File | Change |
|------|--------|
| `.env.example` | Add all required vars with comments |
| `render.yaml` | Verify/fix CORS_ORIGINS |
| `vercel.json` | Document `VITE_ML_API_URL` requirement |

---

## Data Flow: MLWarmupScreen

```
useMLConnection()
    ├── status: 'connecting' | 'warming' | 'ready' | 'error'
    ├── warmupProgress: 0-100 (increments ~3/sec over 35s)
    ├── latencyMs: number | null
    └── retryCount: number

MLWarmupScreen:
    - shows when status = 'connecting' | 'warming'
    - full-screen fixed overlay, z-50
    - animated Brain icon (Lucide, pulsing CSS)
    - progress bar: width = warmupProgress%
    - messages rotate every 8s:
        1. "Initializing neural engines..."
        2. "Loading EEG models..."
        3. "Calibrating signal pipeline..."
        4. "Almost ready..."
    - elapsed timer in seconds
    - after 40s: "Continue in Simulation Mode" button
    - on status='ready': smooth fade-out (opacity transition)
```

---

## Model Improvement: TSception Integration

TSception is already trained (`ml/models/saved/tsception_emotion.pt`, `ml/models/tsception.py`). It achieves 69% CV — better than the feature heuristic baseline (65%).

**Wire-in condition:** TSception runs when:
1. mega_lgbm is not loaded (or CV < 60%)
2. Input epoch has >= 1024 samples (4 seconds × 256 Hz) — TSception requires minimum window
3. `tsception_emotion.pt` exists at expected path

**Response:** `model_type: 'tsception'` in emotion result (frontend already handles this field).

---

## Model Improvement: RunningNormalizer

Addresses EEG non-stationarity (signal drift within a session) — identified by SJTU SEED team and UESTC FACED paper as the largest fixable accuracy problem (-10 to -20 pts).

**Mechanism:** Circular buffer of last 150 feature vectors per user_id (≈5 min at 2s hop). Normalize incoming features against rolling mean/std. Falls back to raw features when buffer < 30 samples.

**Wire-in:** `EmotionClassifier._predict_mega_lgbm()` calls `RunningNormalizer.normalize(features, user_id)` when `user_id` is in the request context.

---

## Error Handling

| Failure | User Experience |
|---------|----------------|
| Backend sleeping (cold start) | MLWarmupScreen shows with 35s progress animation |
| Backend never wakes (permanent error) | SimulationModeBanner shown after 40s, app continues in simulation |
| Single API call fails | mlFetch retries 3x automatically (transparent to user) |
| All retries fail | Component-level error boundary catches, shows "Try again" |
| Vercel env var missing | ml-api.ts falls back to `http://localhost:8000` (dev only) |

---

## Priority Order (Thursday Execution)

1. `useMLConnection` hook (foundation)
2. `MLWarmupScreen` component
3. Integrate into `App.tsx`
4. Keep-alive in `AppLayout`
5. Status dot in sidebar
6. Retry logic in `mlFetch`
7. Simulation mode banner
8. TSception wiring (backend)
9. `RunningNormalizer` (backend)
10. Env/CORS audit
11. STATUS.md + PRODUCT.md update

Stories 1-7 = 1 day. Stories 8-9 = half day. Story 10 = 2 hours. Story 11 = 30 min.

**Total: 2 full days. Thursday deadline is achievable.**

---

## Success Criteria

- [ ] User opens app → sees animated loading screen (never a blank/error page)
- [ ] Loading screen disappears in ≤ 60 seconds and app is usable
- [ ] Keep-alive prevents backend from sleeping during active sessions
- [ ] Single ML call failure is invisible to user (auto-retry)
- [ ] Status dot in sidebar always shows true backend state
- [ ] App works the same on `localhost:5000` and `dream-analysis.vercel.app`
- [ ] TSception appears in `/benchmarks` as "Active (fallback)"
- [ ] RunningNormalizer visible improvement in live emotion readings after 5 min
