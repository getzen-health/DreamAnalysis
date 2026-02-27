# Automatic Study Flow Design
Date: 2026-02-27

## Problem

The current study flow requires manual steps at every stage: participant codes are pre-assigned, sessions are started by hand, EEG data is polled manually, and data export requires a CSV download. Nothing is automatic.

## Goal

A fully automatic pipeline where:
- Participants choose their intent at login (study vs explore)
- The Muse 2 pairing triggers the session — no "start" button
- EEG data flows straight to the database every 30 seconds
- Muse disconnect closes the session automatically
- The researcher sees live data in the admin dashboard — no CSV needed mid-study

---

## Section 1: Login & Intent Flow

After login or register, first-time users land on `/onboarding` — a single screen asking "What brings you here?" with two cards:

- **Join the Study** → `/study/consent`
- **Explore the App** → `/` (Dashboard)

Their choice is saved to the user account as `intent: "study" | "explore"`. Returning users skip this screen:
- Study intent → `/study` (next session or completion screen)
- Explore intent → `/` dashboard

After completing both study sessions, a participant sees an **"Explore the full app"** button, allowing them to switch to the explore path permanently.

---

## Section 2: Automatic Study Session Flow

```
/study/consent    → code auto-assigned, saved to localStorage (already built)
      ↓
/study/profile    → age, diet, Apple Watch (one-time, pre-fills on return)
      ↓
/study/session    → single smart page for both block types
      │
      ├─ Step 1: Pick block     → Stress or Food (skips completed blocks)
      ├─ Step 2: Pair Muse      → one tap Bluetooth; DB record created immediately
      ├─ Step 3: Session runs   → EEG streams automatically by phase
      ├─ Step 4: Auto-close     → Muse disconnect OR "End Session" tap
      └─ Step 5: /study/complete → summary + next session CTA
```

The survey (5 questions, ~30 seconds) is the only manual step.

---

## Section 3: Automatic EEG → Database Pipeline

```
Muse 2 (Bluetooth)
      ↓
useDevice hook  →  latestFrame fires every ~1.5s
      ↓
EEG Session Buffer (in-memory, tagged by phase)
      ↓  every 30s        ↓  on disconnect     ↓  on survey complete
  Checkpoint save      Final save              Full save
      ↓
PATCH /api/study/session/:id/checkpoint  (new endpoint)
      ↓
Neon PostgreSQL — pilot_sessions table
```

**Fields written automatically:**

| Field | Trigger | Content |
|---|---|---|
| `preEegJson` | End of baseline phase | Avg alpha/beta/theta/delta/gamma |
| `postEegJson` | End of recovery phase | Same, post-intervention |
| `eegFeaturesJson` | Every 30s checkpoint | Full rolling EEG feature vector |
| `interventionTriggered` | Stress > 0.65 detected | Boolean, auto-set |
| `surveyJson` | Survey submission | 5-question responses |
| `partial` | Muse drops, can't reconnect | Boolean flag |
| `checkpoint_at` | Every 30s | Timestamp of last auto-save |

---

## Section 4: Participant Experience During Session

- Stress bar is **hidden by default** (prevents anxiety feedback loop)
- Participant can tap "Show my stress level" to expand it — choice persists for session
- Intervention triggers automatically at stress > 0.65 regardless of visibility
- Post-session: full stress arc chart shown (baseline → peak → post-intervention)
- Muse disconnect mid-session: reconnect overlay shown; partial data checkpointed

**Post-session summary:**
- Stress arc line chart
- One-sentence interpretation ("Your stress dropped 34% after breathing")
- Next session CTA or completion screen with "Explore the full app" option

---

## Section 5: Schema Changes & Admin Dashboard

**New columns on `pilot_sessions`:**
```sql
partial        boolean   DEFAULT false
phase_log      jsonb
checkpoint_at  timestamp
```

**New API endpoint:**
```
PATCH /api/study/session/:id/checkpoint
  body: { preEegJson?, postEegJson?, eegFeaturesJson?, interventionTriggered? }
  → upserts fields, sets checkpoint_at = now()
```

**Admin dashboard upgrades:**
- Auto-refresh every 60s
- Session status badges: `recording` / `complete` / `partial`
- Inline stress sparkline per participant row
- Participant progress: which blocks done, which pending

---

## User Account Schema Addition

```sql
ALTER TABLE users ADD COLUMN intent varchar(10) DEFAULT NULL;
-- values: 'study' | 'explore' | null (not yet chosen)
```

---

## Implementation Order

1. Schema migration — add `intent` to users, add `partial`/`phase_log`/`checkpoint_at` to pilot_sessions
2. New API endpoint — `PATCH /api/study/session/:id/checkpoint`
3. Onboarding page — `/onboarding` with intent selection, redirect logic in App.tsx
4. Auth redirect — post-login routes to `/onboarding` (first time) or intent-based route (returning)
5. Unified session page — `/study/session` replaces stress + food separate pages, handles Muse auto-start/stop
6. EEG buffer + checkpoint loop — 30s auto-save using `useDevice` latestFrame
7. Muse disconnect handler — auto-close session, save partial flag if needed
8. Post-session summary — stress arc chart, one-sentence interpretation
9. Admin dashboard upgrades — live refresh, status badges, sparklines
10. "Explore the full app" button on study complete screen
