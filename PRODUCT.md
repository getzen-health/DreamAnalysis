# Neural Dream Workshop — Product Vision

> Read this at the start of every session.
> Every feature you build should move one of these needles.
> If it doesn't, don't build it.

---

## The One Sentence

**Oura Ring tells you your physical readiness. This tells you your mental readiness.**

That is the gap in the market. Nobody has solved daily cognitive performance
tracking for normal people. Whoop, Oura, Garmin all measure body. This
measures brain. That is the wedge.

---

## Why Someone Opens This App Tomorrow Morning

They don't open it because they want to "track emotions."
Nobody wakes up thinking that.

They open it because they want to know:

> *"What should I do today, and when?"*

That question already exists in their head every morning.
This app should answer it.

---

## The North Star Feature — Daily Brain Report

Every morning, one screen. No scrolling. No charts to interpret.

```
Good morning

Last night
  Deep sleep    2h 14m
  REM           1h 02m
  Dreams        2 episodes detected

Today's forecast
  Peak focus    9:30am – 12:00pm   ← protect this time
  Likely slump  2:30pm –  3:30pm
  Stress risk   moderate

Yesterday's insight
  Focus was 23% higher after your 11am walk.
  Stress spiked 40% during your 1pm call.
  → Try a 5-min walk before afternoon meetings.

Recommended now
  [ Start coherence breathing — 4 min ]
```

This is the product. Everything else is infrastructure to produce this screen.

---

## The Moment That Makes Users Believe

Every product has one moment that converts a skeptic into a believer.
Ours is this:

**The user is stressed. The live reading shows it. They do a 4-7-8
breathing exercise for 3 minutes. They watch the stress line drop
in real time on screen.**

That moment — seeing their own brain respond to something they chose to do —
is viscerally powerful. It is unlike anything else on their phone.

Every competitor shows a score *after* a session ends.
We show the brain change *during* the action.
That is the differentiator. Build this first.

---

## The Retention Loop

```
Measure → Insight → Action → Visible result → Measure again
```

Without this loop there is no product. Just a dashboard with numbers.

| Step | What it looks like in the app |
|------|-------------------------------|
| Measure | Wear headset. Session recorded automatically. |
| Insight | "Your focus crashes every day at 2:30pm" |
| Action | "Try a 10-min walk at 2pm tomorrow" |
| Visible result | Stress line drops live during the walk |
| Measure again | Next day: "Yesterday's walk kept focus up until 4pm" |

The user who has gone around this loop once will come back tomorrow.
The user who has gone around it five times will never leave.

---

## What Brings Users Back Daily

| Mechanic | Execution |
|----------|-----------|
| Daily anticipation | "What did my brain do last night?" Same pull as step count |
| Prediction that comes true | App says peak focus 10am. User feels sharp at 10am. Hooked. |
| Personal records | "Longest focus streak: 47 min — beat it?" |
| Cause and effect | "4 meditations this week → baseline stress down 18%" |
| Weekly brain summary | Shareable card. Social loop. Brings new users. |

---

## The Three Product Lanes — Pick One

### Lane 1 — Personal Performance (B2C, $15/month)
- **Target:** Knowledge workers, students, founders, anyone optimising their work
- **Positioning:** "Understand your cognitive rhythm the way elite athletes understand their physical rhythm"
- **Comps:** Oura Ring, Whoop — but for brain
- **Hook:** Schedule your day around your actual mental energy, not guesswork

### Lane 2 — Meditation & Wellness (B2C, $10/month)
- **Target:** 40M meditators in the US who already have Muse or similar
- **Positioning:** "See your meditation actually working in your brain"
- **Hook:** "Your 10-min morning meditation gave you 2 extra hours of low-stress today"
- **Differentiator from Muse app:** They measure during session. We measure all day and show the downstream effect.

### Lane 3 — Corporate / Research (B2B, $500–2000/month per seat)
- **Target:** Universities, clinics, corporate wellness programs, therapists
- **Positioning:** Group-level brain insights, session recordings, exportable data
- **Why it's faster:** They already have budget. They already have hardware. No consumer acquisition.
- **Fastest path to first dollar.** Start here if unsure.

---

## Honest Assessment — February 2026

Read this before starting any new feature. Be honest about where things are.

```
Core ML / Signal pipeline    ████████░░  80%
  Good science. Mastoid reref, DASM/RASM, FAA, 4-sec epochs all done.
  Missing: personalization. Without baseline calibration the model
  is 45% accurate — barely above chance for 6 classes.

Backend API                  ████████░░  80%
  76 endpoints. Complete signal pipeline. WebSocket exists.
  Missing: per-user state isolation (crashes under 2+ users).
           prod deployment (ML backend is still local-only).

Frontend                     █████░░░░░  50%
  17 pages exist. Pages show numbers.
  Missing: the narrative. Numbers without story are not a product.
           Device pairing UX. Baseline calibration UX.
           The intervention biofeedback screen.
           Daily brain report screen.

Product thinking             ███░░░░░░░  30%
  The loop (measure → insight → action → result) does not exist yet.
  No reason to open the app tomorrow morning.
  No "aha moment" in onboarding.

Retention mechanics          ██░░░░░░░░  20%
  No daily pull. No streaks. No personal records.
  No predictions. No weekly summary.

Infrastructure               ████░░░░░░  40%
  Frontend on Vercel. ML backend not deployed.
  No per-user isolation. No monitoring. No auth enforcement.
```

**The gap is not technical. The gap is narrative.**
Numbers need to become insights.
Insights need to become actions.
Actions need to produce visible results.
That loop is the entire product.

---

## What Is Broken Right Now (Do Not Ship Without Fixing)

### 1. Per-user state isolation
`_EpochBuffer`, `BaselineCalibrator`, EMA smoothing — all module-level
singletons. User A's brain state bleeds into User B's readings.
Will silently produce wrong results the moment two users connect.

### 2. Baseline calibration has no UX
The API (`/calibration/baseline/add-frame`) is built and tested.
There is zero UI for it. This is the single biggest accuracy improvement
available (+15–29%) and nobody can access it.
**Every session should start with a 2-min eyes-closed baseline screen.**

### 3. Signal quality is not visible
The headset might be seated wrong. The app shows emotion readings anyway.
Users will blame the product when the problem is electrode contact.
Show HSI indicators loudly before any reading is displayed.

### 4. `epoch_ready` flag is ignored
The API returns `epoch_ready: false` for the first 4 seconds of every
session. The frontend ignores this and shows numbers immediately.
Show "calibrating…" until the buffer is full.

### 5. ML backend is not deployed
Every demo requires running `uvicorn` locally. Not a product.

---

## The Build Order That Actually Matters

### Phase 0 — Make it not embarrassing (1–2 weeks)
- [ ] Deploy ML backend to Railway or Fly.io
- [ ] Fix per-user state isolation
- [ ] Show signal quality / HSI before first reading
- [ ] Show "calibrating…" until `epoch_ready: true`
- [ ] Show confidence on emotion label ("likely relaxed — 68%")

### Phase 1 — Create the aha moment (2–3 weeks)
- [ ] Real-time biofeedback screen during breathing exercise
      (user watches stress line drop live while breathing)
- [ ] Baseline calibration screen (2-min eyes-closed session at start)
- [ ] Device pairing flow with signal quality check

### Phase 2 — Create a reason to come back (3–4 weeks)
- [ ] Session history with timeline view
- [ ] "Yesterday's insight" card (one surprising pattern from last session)
- [ ] Intervention library (5–10 evidence-based exercises with before/after)
- [ ] Personal records ("New focus record: 47 min")

### Phase 3 — Daily pull (4–6 weeks)
- [ ] Daily Brain Report screen (the North Star)
- [ ] Sleep session mode (overnight recording with dream detection)
- [ ] Pattern engine: correlate time-of-day, activities, mental states
- [ ] Morning push notification with yesterday's summary

### Phase 4 — Growth (ongoing)
- [ ] Weekly brain summary card (shareable)
- [ ] User-correctable labels → feeds personalization
- [ ] Per-user model fine-tuning after 5 sessions
- [ ] Export data (CSV, Apple Health sync)

---

## The Hardware Problem

Muse 2 costs $250. That gates 99% of potential users.

**Watch:** Consumer EEG headbands under $100 are entering market in 2025–2026.
Build the software now. Ride the hardware cost curve down later.
**The moat is the software and the personal data patterns, not the device.**

In the meantime:
- Target users who already own Muse (large community — r/muse has 15K members)
- Build for Muse first, design architecture to swap in any BrainFlow device
- Consider OpenBCI Ganglion ($200, 4-channel, research grade) as alternative

---

## Positioning That Actually Works

| Don't say | Say instead |
|-----------|-------------|
| "Emotion tracking app" | "Personal mental performance tracker" |
| "Brain monitoring" | "Understand your brain's daily rhythm" |
| "EEG analysis" | "Like a fitness tracker, but for your mind" |
| "Stress detection" | "Know when to push and when to recover" |
| "Valence: -0.3" | "Your brain is in recovery mode right now" |

**Reframe everything from measurement to meaning.**

---

## What Success Looks Like in 6 Months

A user opens the app every morning without being reminded.
They make one scheduling decision per week based on their peak focus window.
They have done the breathing biofeedback exercise at least once and felt it work.
They have shared their weekly brain summary at least once.
They have been using it for 30 days and their personalized model accuracy
is above 70%.

That user will pay $15/month and tell three people.
That is the product.

---

## Questions to Ask Before Building Anything

1. Does this move a user through the loop (measure → insight → action → result)?
2. Does this help produce the Daily Brain Report?
3. Does this make the aha moment (live biofeedback) more powerful?
4. Does this give a reason to open the app tomorrow morning?

If the answer to all four is no — don't build it yet.
