# App Reliability + Full Model Display Design
**Date:** 2026-03-02

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all navigation crashes and state bugs, surface all 16 ML model outputs across every page, and add full two-way Apple Health / Google Fit integration.

**Approach:** Patch & Extend — fix crashes first (fast, isolated), then enrich model data display page by page, then wire health integrations.

**Stack:** React 18 + TypeScript, Vite, wouter, TanStack Query, shadcn/ui, Tailwind, FastAPI ML backend (port 8080), PostgreSQL (Neon).

---

## Section 1 — Bug Fixes

### 1a. Navigation crash ("Something went wrong" on every tab)

**Root cause:** Single `ErrorBoundary` in `App.tsx` (line 116) wraps all routes. When any page crashes, `hasError=true` persists across navigation — every page shows the error screen until full reload.

**Fix:** Add `key={location}` to the `ErrorBoundary` so React re-mounts it (resetting `hasError`) whenever the route changes.

```tsx
// client/src/App.tsx — AppRoutes()
function AppRoutes() {
  const [location] = useLocation();
  return (
    <ErrorBoundary key={location}>   // ← this one addition
      <Suspense fallback={<PageLoader />}>
        <Switch>
          ...
```

**Files:** `client/src/App.tsx`
**Risk:** Zero — this is the standard React pattern for route-level error isolation.

---

### 1b. Food tab — morning data shows in lunch view

**Root cause:** The food log query does not include `mealPeriod` in its TanStack Query key. When you switch tabs, the cached result from the previous period is served immediately before the new fetch completes.

**Fix:** Add the active `mealPeriod` to the query key so each tab gets its own cache slot:

```tsx
// client/src/pages/food-log.tsx
useQuery({
  queryKey: ["food-logs", userId, mealPeriod],  // ← add mealPeriod here
  queryFn: () => fetchFoodLogs(userId, mealPeriod),
  ...
})
```

**Files:** `client/src/pages/food-log.tsx`

---

### 1c. Device state lost after page reload

**Root cause:** `DeviceProvider` onMount calls `getDeviceStatus()` at startup. If localStorage has ML URL set to `http://localhost:8080` but the call races or fails silently, device state stays "disconnected" even though the backend is streaming.

**Fix:** Add a 2-second retry in the onMount effect when the first status call fails, and also fix the `ML_API_URL_DEFAULT` in `ml-api.ts` to match the actual port (`8080`) via the `VITE_ML_API_URL` env var (already in `client/.env`).

**Files:** `client/src/hooks/use-device.tsx`, `client/.env`

---

## Section 2 — Model Data Display

### 2a. Dashboard — always show something

**Problem:** All dashboard live data is gated on `isStreaming`. With no device connected, the page shows only a "Connect your Muse 2" banner + empty cards.

**Fix:**
- Add "Last Session Snapshot" row: pull last session's stress/focus/emotion/creativity from `listSessions()` and show as static metric cards even when offline.
- Add "What's being tracked" capability badges row: 16 small pills (Sleep, Emotion, Flow, Creativity, Spiritual, Lucid Dream, Stress, Focus, Cognitive Load, Attention, Drowsiness, Meditation, Memory, Artifact, Denoising, Online Learning). Always visible.

**Files:** `client/src/pages/dashboard.tsx`

---

### 2b. Brain Monitor — waves + all 16 model outputs visible

**Problem:** EEG waveform canvas and band powers exist but may not animate if `latestFrame` throttle is too slow (1500ms). The "16 Models" panel exists but some outputs are missing.

**Fix:**
- Lower waveform update throttle to 300ms (raw signals via `eeg-signals` custom event already bypass throttle — this is fine).
- Add "All 16 Models" grid card: shows every model output in a 4×4 grid with color-coded values. Each cell: model name, current value, confidence bar.
- Ensure creativity score, memory encoding, cognitive load, drowsiness all appear.

**Files:** `client/src/pages/brain-monitor.tsx`

---

### 2c. Emotion Lab — full breakdown

**Problem:** Shows top emotion and stress/focus/relaxation bars. Missing: all 6 emotion probabilities, creativity score, valence/arousal quadrant.

**Fix:**
- Add horizontal probability bars for all 6 emotions (happy, sad, angry, fear, surprise, neutral).
- Add creativity score badge pulled from `latestFrame.analysis.creativity`.
- Add valence/arousal 2D chart: a quadrant (calm/excited × negative/positive) with a dot showing current state.

**Files:** `client/src/pages/emotion-lab.tsx`

---

### 2d. Inner Energy / Spiritual — show even at baseline

**Problem:** 7 chakras render nothing unless streaming. Meditation depth and consciousness level are hidden.

**Fix:**
- Render all 7 chakras with band-power values always visible (show 0 / baseline state when not streaming — not blank).
- Meditation depth and kundalini/prana values always shown; update live when streaming.
- Remove `isStreaming` guard from the main chakra render; keep it only for the "live" badge.

**Files:** `client/src/pages/inner-energy.tsx`

---

### 2e. Biofeedback — add EEG context before + during exercise

**Problem:** Jumps straight to breathing exercise with no context about why you need it or whether it's working.

**Fix:**
- Add a "Before you start" card at the top: current stress level (from last frame or last session), alpha/beta ratio, and a generated sentence ("Your stress is elevated — this session will help lower it").
- During breathing: add a small live alpha/beta ratio sparkline showing whether your brain is calming down.
- After session: show before vs. after stress comparison.

**Files:** `client/src/pages/biofeedback.tsx`

---

### 2f. Settings — Device + Health connect hub

**Problem:** ML backend URL is buried in a card. No visible Apple Health or Google Fit connect button.

**Fix:**
- Add "Connected Devices" card at the top of Settings: shows connected device (or "None"), connect button linking to `/device-setup`, and device status.
- Add "Health Integrations" card below: Apple Health (HealthKit) connect button, Google Fit connect button, both with "Connected / Not connected" status.

**Files:** `client/src/pages/settings.tsx`

---

## Section 3 — Apple Health Two-Way Sync

### 3a. Settings connect flow

New "Health Integrations" card in Settings with:
- "Connect Apple Health" button → calls `POST /api/health/export-to-healthkit` or triggers native HealthKit permission flow
- "Connect Google Fit" button → Google OAuth flow
- Status: "Connected — last synced X min ago" or "Not connected"

### 3b. Read from Apple Health

Pull these metrics and show on dashboard:
- Sleep: bedtime, wake time, duration
- Heart rate + HRV (resting)
- Steps, activity

Show as "Health Context" card on Dashboard: "Last night: 7h12m sleep · HRV 52ms"
Wire into Brain-Health Insights correlation engine (already exists at `/api/health/insights`).

### 3c. Write to Apple Health

After each EEG session ends, `POST /api/health/export-to-healthkit` with:
- Mindful session minutes
- Stress index as custom quantity
- Focus index
- Flow state episode flag

### 3d. Backend

The ML backend already has `ml/health/` module and `/api/health/export-to-healthkit` endpoint. Wire the Settings button to call it. Add `/api/health/connect` endpoint that returns current connection status and triggers the HealthKit permissions dialog (on iOS/macOS via the native layer).

**Files:** `client/src/pages/settings.tsx`, `ml/api/routes.py` (health section), `client/src/lib/ml-api.ts`

---

## Implementation Order

| Phase | What | Time estimate | Priority |
|-------|------|--------------|---------|
| 1 | Fix ErrorBoundary navigation crash | 5 min | CRITICAL |
| 2 | Fix food tab mealPeriod query key | 15 min | HIGH |
| 3 | Fix device state restore on reload | 20 min | HIGH |
| 4 | Dashboard: last session snapshot + capability badges | 30 min | HIGH |
| 5 | Brain Monitor: 16-model grid + wave throttle | 30 min | HIGH |
| 6 | Emotion Lab: all probabilities + valence/arousal quadrant + creativity | 30 min | HIGH |
| 7 | Inner Energy: always-visible chakras | 20 min | MEDIUM |
| 8 | Biofeedback: EEG context before/during/after | 30 min | MEDIUM |
| 9 | Settings: device status card + health integrations card | 30 min | MEDIUM |
| 10 | Apple Health / Google Fit backend wiring | 45 min | MEDIUM |

---

## Success Criteria

- Navigating between any two pages never shows "Something went wrong"
- Switching food tabs (morning → lunch) never shows the wrong meal's data
- Connecting Synthetic device → Brain Monitor shows live alpha/beta/gamma waveforms
- Emotion Lab shows all 6 emotion probability bars + valence/arousal quadrant
- Inner Energy / Spiritual renders chakra values always (not blank)
- Dashboard shows last session data even when no device connected
- Settings shows device connection status + Apple Health / Google Fit connect buttons
- Apple Health read: last night's sleep + HRV appears on dashboard
- Apple Health write: after session, focus/stress metrics pushed to Health app
