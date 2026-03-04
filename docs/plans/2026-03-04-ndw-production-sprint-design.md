# NDW Production Sprint — Design Doc
**Date:** 2026-03-04
**Branch:** `ralph/ndw-production-sprint`
**Stories:** 14

---

## 1. Forgot Password (US-001–004)

**Stack:** Drizzle ORM schema + Express routes + Resend email + React pages.

No email library exists in the project. Adding `resend` (smallest API, one function call). Token is a 32-byte hex string stored in a new `passwordResetTokens` table, expires in 1 hour, single-use (marked `usedAt` on redemption). Endpoint never reveals whether email exists (prevents enumeration). Frontend: "Forgot password?" link → `/forgot-password` (email input) → `/reset-password?token=` (new password).

**Requires:** `RESEND_API_KEY` env var in Vercel dashboard + local `.env`.

---

## 2. Research Session 20/20 Fix (US-005–006)

**Current:** Baseline 5 min + Task 15 min + Breathing 3 min + Recovery 5 min = 28 min, mandatory breathing for everyone.
**New:** Task 20 min flat. Mid-session check-in: "Are you feeling stressed?" → Yes = 3-min box breathing inserted → No = skip. Same logic for both stress and food sessions.

---

## 3. WebSocket Stability (US-007–008)

**Root cause:** `MAX_RECONNECT_ATTEMPTS = 5` — after 5 drops the app silently gives up.
**Fix:** Remove attempt cap; reconnect forever while device is selected. Add 30s JSON ping (`{type:"ping"}`) to prevent server-side idle timeout. Add visible amber "Reconnecting… (attempt N)" banner on streaming pages.

---

## 4. Remove Hardcoded Values (US-009)

All EEG-derived metrics show `--` or `0` when `latestFrame` is null. No simulation baselines. Food emotion percentages (currently hardcoded 17%) wired to `latestFrame.analysis.emotions.probabilities`.

---

## 5. Food Fixes (US-010–012)

- **Tab context:** Each tab (Breakfast/Lunch/Dinner/Snack) has its own description prompt, switching on `activeTab`.
- **Craving graph:** Recharts bar chart of 7-day food state distribution from DB (restored).
- **Live EEG wiring:** Food emotion % → derived from live `latestFrame` emotion probabilities using stress/relaxation/valence thresholds.

---

## 6. iOS Build (US-013–014)

`npm run build → npx cap sync ios → Xcode archive → TestFlight`. Config already has correct bundle ID (`com.neuraldreamworkshop.app`), BLE permissions, and iOS 14+ target. Sravya does the Xcode archive step manually. Build instructions added to README.

---

## Story Execution Order

| Priority | Story | Dependency |
|----------|-------|------------|
| 1 | DB schema — reset tokens | — |
| 2 | API — forgot-password endpoint | US-001 |
| 3 | API — reset-password endpoint | US-001 |
| 4 | UI — forgot/reset pages | US-002, US-003 |
| 5 | Research stress session 20 min | — |
| 6 | Research food session 20 min | US-005 pattern |
| 7 | WS reconnect fix | — |
| 8 | Reconnecting banner | US-007 |
| 9 | Remove hardcoded values | — |
| 10 | Food tab context | — |
| 11 | Food craving graph | — |
| 12 | Food emotion live EEG | US-009 |
| 13 | iOS cap sync verify | — |
| 14 | README + STATUS update | US-013 |
