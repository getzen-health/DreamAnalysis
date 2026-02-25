# RESEARCH ENROLLMENT MODULE — TECHNICAL SPEC
## NeuralDreamWorkshop — Participant "Bring Your Own Device" Research Platform

**Version:** 1.0
**Date:** February 25, 2026
**Status:** Ready to build (do while waiting for IRB approval)

---

## OVERVIEW

Add a self-contained "Research Mode" to NeuralDreamWorkshop that lets Muse 2
owners enroll in the 30-day study, complete daily protocol sessions via the app,
and have their data flow automatically into a pseudonymized research database —
with zero device cost and zero researcher logistics.

**What gets built:** 6 new pages, 8 new API endpoints, 5 new DB tables, and a
sidebar section. Nothing in the existing app is removed or broken.

---

## 1. DATABASE CHANGES (shared/schema.ts)

Add 5 new tables. All use the existing `varchar` PK pattern and Drizzle ORM.

### Table 1: `studyParticipants`

```typescript
export const studyParticipants = pgTable("study_participants", {
  id:                  varchar("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
  userId:              varchar("user_id").notNull().references(() => users.id, { onDelete: "cascade" }),
  studyId:             text("study_id").notNull(),           // "emotional-day-night-v1"
  studyCode:           varchar("study_code", { length: 6 }).notNull().unique(), // "NX4T82"
  enrolledAt:          timestamp("enrolled_at").defaultNow(),
  consentVersion:      text("consent_version").notNull(),    // "2.0"
  consentSignedAt:     timestamp("consent_signed_at").notNull(),
  overnightEegConsent: boolean("overnight_eeg_consent").default(false),
  status:              text("status").default("active"),      // "active" | "completed" | "withdrawn"
  targetDays:          integer("target_days").default(30),
  completedDays:       integer("completed_days").default(0),
  startDate:           timestamp("start_date").defaultNow(),
  withdrawnAt:         timestamp("withdrawn_at"),
  preferredMorningTime:   text("preferred_morning_time"),    // "07:00"
  preferredDaytimeTime:   text("preferred_daytime_time"),    // "10:00"
  preferredEveningTime:   text("preferred_evening_time"),    // "21:00"
});
```

### Table 2: `studySessions` (one row per participant per day)

```typescript
export const studySessions = pgTable("study_sessions", {
  id:               varchar("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
  participantId:    varchar("participant_id").notNull().references(() => studyParticipants.id, { onDelete: "cascade" }),
  studyCode:        varchar("study_code", { length: 6 }).notNull(),
  dayNumber:        integer("day_number").notNull(),          // 1–30
  sessionDate:      timestamp("session_date").notNull(),      // date only (store as midnight UTC)
  morningCompleted: boolean("morning_completed").default(false),
  daytimeCompleted: boolean("daytime_completed").default(false),
  eveningCompleted: boolean("evening_completed").default(false),
  validDay:         boolean("valid_day").default(false),      // true if ≥ 2 of 3 completed
  createdAt:        timestamp("created_at").defaultNow(),
}, (t) => ({
  uniqueDayIdx: uniqueIndex("study_session_day_idx").on(t.participantId, t.dayNumber),
}));
```

### Table 3: `studyMorningEntries`

```typescript
export const studyMorningEntries = pgTable("study_morning_entries", {
  id:                varchar("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
  sessionId:         varchar("session_id").notNull().references(() => studySessions.id, { onDelete: "cascade" }),
  studyCode:         varchar("study_code", { length: 6 }).notNull(),
  dreamText:         text("dream_text"),                      // null if noRecall
  noRecall:          boolean("no_recall").default(false),
  dreamValence:      integer("dream_valence"),                // SAM 1–9
  dreamArousal:      integer("dream_arousal"),                // SAM 1–9
  nightmareFlag:     text("nightmare_flag"),                  // "yes" | "no" | "unsure"
  sleepQuality:      integer("sleep_quality"),                // 1–9
  sleepHours:        real("sleep_hours"),
  minutesFromWaking: integer("minutes_from_waking"),          // data quality metric
  currentMoodRating: integer("current_mood_rating"),          // welfare check 1–9
  submittedAt:       timestamp("submitted_at").defaultNow(),
});
```

### Table 4: `studyDaytimeEntries`

```typescript
export const studyDaytimeEntries = pgTable("study_daytime_entries", {
  id:                   varchar("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
  sessionId:            varchar("session_id").notNull().references(() => studySessions.id, { onDelete: "cascade" }),
  studyCode:            varchar("study_code", { length: 6 }).notNull(),
  eegFeatures:          jsonb("eeg_features"),                // 85-dim feature vector
  faa:                  real("faa"),                          // frontal alpha asymmetry
  highBeta:             real("high_beta"),
  fmt:                  real("fmt"),                          // frontal midline theta
  sqiMean:              real("sqi_mean"),                     // signal quality index
  eegDurationSec:       integer("eeg_duration_sec"),
  samValence:           integer("sam_valence"),               // 1–9
  samArousal:           integer("sam_arousal"),               // 1–9
  samStress:            integer("sam_stress"),                // 1–9
  panasItems:           jsonb("panas_items"),                 // {pa: number, na: number}
  sleepHoursReported:   real("sleep_hours_reported"),
  caffeineServings:     integer("caffeine_servings"),
  significantEventYN:   boolean("significant_event_yn"),
  submittedAt:          timestamp("submitted_at").defaultNow(),
});
```

### Table 5: `studyEveningEntries`

```typescript
export const studyEveningEntries = pgTable("study_evening_entries", {
  id:                     varchar("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
  sessionId:              varchar("session_id").notNull().references(() => studySessions.id, { onDelete: "cascade" }),
  studyCode:              varchar("study_code", { length: 6 }).notNull(),
  dayValence:             integer("day_valence"),             // 1–9
  dayArousal:             integer("day_arousal"),             // 1–9
  peakEmotionIntensity:   integer("peak_emotion_intensity"),  // 1–9
  peakEmotionDirection:   text("peak_emotion_direction"),     // "positive" | "negative"
  meals:                  jsonb("meals"),                     // [{description, motivation, fullness, mindfulness}]
  emotionalEatingDay:     text("emotional_eating_day"),       // "yes" | "no" | "unsure"
  cravingsToday:          boolean("cravings_today"),
  cravingTypes:           jsonb("craving_types"),             // ["sweet", "salty", ...]
  exerciseLevel:          text("exercise_level"),             // "none"|"light"|"moderate"|"vigorous"
  alcoholDrinks:          integer("alcohol_drinks"),
  medicationsTaken:       boolean("medications_taken"),
  stressRightNow:         integer("stress_right_now"),        // 1–9
  readyForSleep:          boolean("ready_for_sleep"),
  submittedAt:            timestamp("submitted_at").defaultNow(),
});
```

### Migration Command

After adding tables to schema.ts, run:
```bash
npm run db:push
# or: npx drizzle-kit push
```

---

## 2. NEW API ENDPOINTS (server/routes.ts)

Add these 8 routes after the existing endpoints.

---

### POST `/api/study/enroll`

```typescript
app.post("/api/study/enroll", async (req, res) => {
  const { userId, studyId, consentVersion, overnightEegConsent,
          preferredMorningTime, preferredDaytimeTime, preferredEveningTime } = req.body;

  // Generate unique 6-char study code
  const generateCode = () => {
    const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"; // no O,0,I,1 (confusing)
    return Array.from({length: 6}, () => chars[Math.floor(Math.random() * chars.length)]).join("");
  };
  let studyCode = generateCode();
  // Ensure uniqueness (retry on collision)
  while (await db.select().from(studyParticipants).where(eq(studyParticipants.studyCode, studyCode)).limit(1)) {
    studyCode = generateCode();
  }

  const [participant] = await db.insert(studyParticipants).values({
    userId, studyId, studyCode, consentVersion,
    consentSignedAt: new Date(),
    overnightEegConsent: overnightEegConsent ?? false,
    preferredMorningTime, preferredDaytimeTime, preferredEveningTime,
  }).returning();

  res.json({ studyCode: participant.studyCode, enrolledAt: participant.enrolledAt });
});
```

---

### GET `/api/study/status/:userId`

Returns current enrollment state — used by the research dashboard to know what
to show and which sessions are complete today.

```typescript
app.get("/api/study/status/:userId", async (req, res) => {
  const participant = await db.select().from(studyParticipants)
    .where(and(
      eq(studyParticipants.userId, req.params.userId),
      eq(studyParticipants.status, "active")
    )).limit(1);

  if (!participant[0]) return res.json({ enrolled: false });

  const p = participant[0];
  const today = await db.select().from(studySessions)
    .where(and(
      eq(studySessions.participantId, p.id),
      eq(studySessions.dayNumber, p.completedDays + 1)  // today's day
    )).limit(1);

  res.json({
    enrolled: true,
    studyCode: p.studyCode,
    dayNumber: p.completedDays + 1,
    completedDays: p.completedDays,
    targetDays: p.targetDays,
    compensationEarned: p.completedDays * 5,
    todaySession: today[0] ?? null,
    preferredTimes: {
      morning: p.preferredMorningTime,
      daytime: p.preferredDaytimeTime,
      evening: p.preferredEveningTime,
    }
  });
});
```

---

### POST `/api/study/morning`

```typescript
app.post("/api/study/morning", async (req, res) => {
  const { userId, dreamText, noRecall, dreamValence, dreamArousal,
          nightmareFlag, sleepQuality, sleepHours, minutesFromWaking,
          currentMoodRating } = req.body;

  const participant = await getActiveParticipant(userId); // helper
  const session = await getOrCreateTodaySession(participant);

  await db.insert(studyMorningEntries).values({
    sessionId: session.id, studyCode: participant.studyCode,
    dreamText: noRecall ? null : dreamText,
    noRecall, dreamValence, dreamArousal, nightmareFlag,
    sleepQuality, sleepHours, minutesFromWaking, currentMoodRating,
  });

  await db.update(studySessions)
    .set({ morningCompleted: true })
    .where(eq(studySessions.id, session.id));

  await checkAndMarkValidDay(session.id);

  res.json({ success: true, dayNumber: session.dayNumber });
});
```

---

### POST `/api/study/daytime`

```typescript
app.post("/api/study/daytime", async (req, res) => {
  const { userId, eegFeatures, faa, highBeta, fmt, sqiMean, eegDurationSec,
          samValence, samArousal, samStress, panasItems,
          sleepHoursReported, caffeineServings, significantEventYN } = req.body;

  const participant = await getActiveParticipant(userId);
  const session = await getOrCreateTodaySession(participant);

  await db.insert(studyDaytimeEntries).values({
    sessionId: session.id, studyCode: participant.studyCode,
    eegFeatures, faa, highBeta, fmt, sqiMean, eegDurationSec,
    samValence, samArousal, samStress, panasItems,
    sleepHoursReported, caffeineServings, significantEventYN,
  });

  await db.update(studySessions)
    .set({ daytimeCompleted: true })
    .where(eq(studySessions.id, session.id));

  await checkAndMarkValidDay(session.id);

  res.json({ success: true, dayNumber: session.dayNumber });
});
```

---

### POST `/api/study/evening`

```typescript
app.post("/api/study/evening", async (req, res) => {
  const { userId, dayValence, dayArousal, peakEmotionIntensity,
          peakEmotionDirection, meals, emotionalEatingDay,
          cravingsToday, cravingTypes, exerciseLevel, alcoholDrinks,
          medicationsTaken, stressRightNow, readyForSleep } = req.body;

  const participant = await getActiveParticipant(userId);
  const session = await getOrCreateTodaySession(participant);

  await db.insert(studyEveningEntries).values({
    sessionId: session.id, studyCode: participant.studyCode,
    dayValence, dayArousal, peakEmotionIntensity, peakEmotionDirection,
    meals, emotionalEatingDay, cravingsToday, cravingTypes,
    exerciseLevel, alcoholDrinks, medicationsTaken, stressRightNow, readyForSleep,
  });

  await db.update(studySessions)
    .set({ eveningCompleted: true })
    .where(eq(studySessions.id, session.id));

  const isValid = await checkAndMarkValidDay(session.id);

  // If this made it a valid day, increment completedDays
  if (isValid) {
    await db.update(studyParticipants)
      .set({ completedDays: sql`${studyParticipants.completedDays} + 1` })
      .where(eq(studyParticipants.id, participant.id));
  }

  res.json({
    success: true,
    validDay: isValid,
    compensationEarned: (participant.completedDays + (isValid ? 1 : 0)) * 5,
  });
});
```

---

### GET `/api/study/history/:userId`

```typescript
app.get("/api/study/history/:userId", async (req, res) => {
  const participant = await getActiveParticipant(req.params.userId);
  if (!participant) return res.json([]);

  const sessions = await db.select().from(studySessions)
    .where(eq(studySessions.participantId, participant.id))
    .orderBy(asc(studySessions.dayNumber));

  res.json(sessions);
});
```

---

### POST `/api/study/withdraw`

```typescript
app.post("/api/study/withdraw", async (req, res) => {
  const { userId } = req.body;
  const participant = await getActiveParticipant(userId);

  await db.update(studyParticipants)
    .set({ status: "withdrawn", withdrawnAt: new Date() })
    .where(eq(studyParticipants.id, participant.id));

  res.json({
    daysCompleted: participant.completedDays,
    compensationEarned: participant.completedDays * 5,
    message: "You have been withdrawn from the study. Thank you for participating."
  });
});
```

---

### Helper functions (add above routes)

```typescript
async function getActiveParticipant(userId: string) {
  const [p] = await db.select().from(studyParticipants)
    .where(and(
      eq(studyParticipants.userId, userId),
      eq(studyParticipants.status, "active")
    )).limit(1);
  return p;
}

async function getOrCreateTodaySession(participant: typeof studyParticipants.$inferSelect) {
  const todayStart = new Date();
  todayStart.setHours(0, 0, 0, 0);

  const [existing] = await db.select().from(studySessions)
    .where(and(
      eq(studySessions.participantId, participant.id),
      gte(studySessions.sessionDate, todayStart)
    )).limit(1);

  if (existing) return existing;

  const [created] = await db.insert(studySessions).values({
    participantId: participant.id,
    studyCode: participant.studyCode,
    dayNumber: participant.completedDays + 1,
    sessionDate: todayStart,
  }).returning();

  return created;
}

async function checkAndMarkValidDay(sessionId: string) {
  const [session] = await db.select().from(studySessions)
    .where(eq(studySessions.id, sessionId));
  const completedCount = [session.morningCompleted, session.daytimeCompleted, session.eveningCompleted]
    .filter(Boolean).length;
  const isValid = completedCount >= 2;
  if (isValid) {
    await db.update(studySessions).set({ validDay: true }).where(eq(studySessions.id, sessionId));
  }
  return isValid;
}
```

---

## 3. NEW REACT PAGES

### Page 1: `/research` — Research Hub

**File:** `client/src/pages/research-hub.tsx`

This is the entry point. Shows the study card if not enrolled, or the participant
dashboard if enrolled.

```
┌─────────────────────────────────────────────────────┐
│  🔬 Research Studies                                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  THE EMOTIONAL DAY-NIGHT CYCLE              │   │
│  │  30-Day EEG + Dream Study                   │   │
│  │                                             │   │
│  │  💰 Up to $150 + $25 bonus                  │   │
│  │  📅 30 days · ~30 min/day                   │   │
│  │  🧠 Requires: Muse 2 EEG headset            │   │
│  │  👥 Enrolling now · 12 spots remaining      │   │
│  │                                             │   │
│  │  Do your daytime emotions predict what      │   │
│  │  you dream about at night? Help us find     │   │
│  │  out — using your own Muse 2.               │   │
│  │                                             │   │
│  │         [ Join This Study → ]               │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Already enrolled? Your dashboard is below ↓       │
└─────────────────────────────────────────────────────┘
```

**Logic:**
- `useQuery` → `GET /api/study/status/:userId`
- If `enrolled: false` → show study card with "Join" button → navigate to `/research/enroll`
- If `enrolled: true` → show participant dashboard (inline, same page)

---

### Page 2: `/research/enroll` — Consent + Enrollment Flow

**File:** `client/src/pages/research-enroll.tsx`

Multi-step wizard. Use `useState(step)` to track progress. No page navigation
between steps — all on one page, step controls shown/hidden.

```
Step 1: Study Overview     (what, how long, compensation)
Step 2: Eligibility Check  (5 quick yes/no questions)
Step 3: Consent Form       (scrollable full consent, initials per section)
Step 4: Overnight EEG      (optional consent — two radio buttons)
Step 5: Preferences        (pick morning/daytime/evening times)
Step 6: Confirmation       (study code shown, copy button, what happens next)
```

**Step 2 — Eligibility (inline mini-screen, 5 questions):**
```
✓ I sleep at roughly the same time each night
✓ I do NOT have a diagnosed sleep disorder (insomnia, apnea, narcolepsy)
✓ I do NOT have epilepsy or a seizure history
✓ I am NOT currently taking anticonvulsants or antipsychotic medications
✓ I own a Muse 2 EEG headset
```
If any is unchecked → show ineligible message, stop.

**Step 3 — Consent:**
- Render the consent form text (hardcode v2.0 content as a constant)
- Each section has an initial field (text input, max 3 chars)
- "I have read this section" checkbox per block
- Full scroll required before "Continue" button activates
- Collect: `consentVersion: "2.0"`, `consentSignedAt: new Date()`

**Step 6 — Confirmation:**
```
┌────────────────────────────────────────┐
│  🎉 You're enrolled!                  │
│                                        │
│  Your Study Code: NX4T82              │
│                  [ Copy ]             │
│                                        │
│  Save this code — it's your           │
│  anonymous research identifier.       │
│                                        │
│  Your study starts tomorrow.          │
│  First thing tomorrow morning:        │
│  open the app before checking         │
│  anything else and record your        │
│  dream.                               │
│                                        │
│     [ Go to Research Dashboard ]      │
└────────────────────────────────────────┘
```

**On submit:** `POST /api/study/enroll` → store returned `studyCode` in local state → show Step 6.

---

### Page 3: `/research/morning` — Morning Dream Entry

**File:** `client/src/pages/research-morning.tsx`

Simple, fast, minimal — designed to be used within 5 minutes of waking.
Dark background. Large text. No navigation clutter.

```
┌──────────────────────────────────────────┐
│  🌙 Day 12 — Morning Entry              │
│  Record this before doing anything else  │
├──────────────────────────────────────────┤
│                                          │
│  Did you remember a dream?              │
│  ○ Yes    ○ No recall                   │
│                                          │
│  [If yes — dream text area:]            │
│  ┌────────────────────────────────────┐ │
│  │ What do you remember? Even just a  │ │
│  │ feeling or a single image...       │ │
│  │                                    │ │
│  └────────────────────────────────────┘ │
│                                          │
│  Dream emotional tone:                  │
│  😞 1 ── 2 ── 3 ── 4 ── 5 ── 6 ── 7 ── 8 ── 9 😊 │
│                                          │
│  Was it a nightmare?                    │
│  ○ Yes   ○ No   ○ Not sure             │
│                                          │
│  Sleep quality last night:             │
│  😴 1 ─────────────────── 9 ✨         │
│                                          │
│  Hours slept (approx): [   ] hrs       │
│                                          │
│  How are you feeling right now? (1–9)  │
│  [welfare check — shown subtly]         │
│                                          │
│       [ Submit Morning Entry ✓ ]        │
└──────────────────────────────────────────┘
```

**On submit:**
- `POST /api/study/morning`
- If `currentMoodRating ≤ 2` → show mental health resources card with option to continue or take a break
- Show: "Morning entry saved ✓ — see you this afternoon for your EEG session."

---

### Page 4: `/research/daytime` — Daytime EEG + Mood

**File:** `client/src/pages/research-daytime.tsx`

Reuses the existing EEG recording infrastructure already in `/emotions`.

```
┌──────────────────────────────────────────┐
│  🧠 Day 12 — Daytime Session            │
│  Best time: 9 AM – 1 PM                │
├──────────────────────────────────────────┤
│                                          │
│  STEP 1: EEG Recording                 │
│  ┌──────────────────────────────────┐  │
│  │  [Existing EEG session widget]   │  │
│  │  Put on Muse 2 → Start          │  │
│  │  ● 2 min eyes closed            │  │
│  │  ● 2 min eyes open (fixation)   │  │
│  │  SQI: ████░░ 72%                │  │
│  └──────────────────────────────────┘  │
│                                          │
│  STEP 2: How do you feel right now?    │
│                                          │
│  Mood:    😞 ─────────── 😊            │
│  Energy:  😴 ─────────── ⚡            │
│  Stress:  😌 ─────────── 😤            │
│                                          │
│  STEP 3: Quick mood check (10 items)   │
│  Rate each word: Not at all → Extremely │
│  Active □□□□□  Distressed □□□□□       │
│  [5 positive + 5 negative PANAS items] │
│                                          │
│  Hours slept: [  ]  Coffees today: [  ]│
│  Anything significant happen today? ○Y ○N│
│                                          │
│       [ Submit Daytime Entry ✓ ]        │
└──────────────────────────────────────────┘
```

**EEG data capture:** At end of EEG session, extract `faa`, `highBeta`, `fmt`,
`sqiMean` from the ML backend response (already returned by `/api/ml/analyze-eeg`).
Pass these fields along with the full `eegFeatures` vector in the POST body.

---

### Page 5: `/research/evening` — Evening Questionnaire

**File:** `client/src/pages/research-evening.tsx`

```
┌──────────────────────────────────────────┐
│  🌆 Day 12 — Evening Check-in           │
│  Complete within 2 hours of sleep       │
├──────────────────────────────────────────┤
│                                          │
│  Overall day mood:                      │
│  Mood:   😞 ─────────── 😊             │
│  Energy: 😴 ─────────── ⚡             │
│                                          │
│  What was the most emotional moment     │
│  of your day?                           │
│  Intensity: 1 ────────── 9             │
│  Direction: ○ Positive  ○ Negative     │
│                                          │
│  ─── Eating Today ──────────────────── │
│                                          │
│  Meal 1: [briefly describe]            │
│  Before eating: ○ Physically hungry    │
│                 ○ Emotional need       │
│                 ○ Habit / routine      │
│  How was the eating? ○ Mindful         │
│                      ○ Rushed          │
│                      ○ Emotional       │
│  [ + Add another meal ]                │
│                                          │
│  Overall: Did today feel like an        │
│  emotional eating day?  ○ Yes ○ No ○ ? │
│                                          │
│  Any cravings? ○ No  ○ Yes →           │
│  [Sweet] [Salty] [Fatty] [Comfort]     │
│                                          │
│  ─── Context ──────────────────────── │
│  Exercise: ○ None ○ Light ○ Mod ○ Vig │
│  Alcohol: [  ] drinks                  │
│  Medications today: ○ Yes ○ No         │
│                                          │
│  Stress right now: 1 ─────────── 9    │
│  Ready to sleep?  ○ Yes  ○ Not yet    │
│                                          │
│       [ Submit Evening Entry ✓ ]        │
└──────────────────────────────────────────┘
```

**On submit:** Show compensation update:
```
Day 12 complete ✓ — Valid day earned!
Compensation so far: $60 of $150
```

---

### Page 6: `/research/dashboard` (embedded in `/research` when enrolled)

Shown inline on the Research Hub page once enrolled.

```
┌──────────────────────────────────────────────────┐
│  Your Study — Day 12 of 30                      │
│  Compensation: $60 earned · $90 remaining       │
├──────────────────────────────────────────────────┤
│                                                  │
│  TODAY                                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ MORNING │  │ DAYTIME │  │ EVENING │         │
│  │   ✓     │  │  Open   │  │  Later  │         │
│  │ Done    │  │ [Start] │  │         │         │
│  └─────────┘  └─────────┘  └─────────┘         │
│                                                  │
│  30-Day Calendar                                │
│  ┌──────────────────────────────────────────┐  │
│  │ 1✓ 2✓ 3✓ 4✓ 5✓ 6✓ 7✓ 8✓ 9✓ 10✓      │  │
│  │ 11✓ 12◐ 13  14  15  16  17  18  19  20  │  │
│  │ 21  22  23  24  25  26  27  28  29  30  │  │
│  └──────────────────────────────────────────┘  │
│  ✓ = valid day  ◐ = in progress  □ = upcoming  │
│                                                  │
│  Last 7 days — your mood trend:                 │
│  [small sparkline chart — SAM valence over 7d] │
│                                                  │
│  [ Withdraw from Study ]  (small, grey link)   │
└──────────────────────────────────────────────────┘
```

---

## 4. SIDEBAR UPDATE (client/src/components/sidebar.tsx)

Add a new section above "Settings":

```typescript
// Add to navSections array, before the settings section:
{
  title: "Research",
  items: [
    {
      href: "/research",
      icon: FlaskConical,     // from lucide-react
      label: "Research Study",
    }
  ]
}
```

Import `FlaskConical` from lucide-react (already installed).

---

## 5. ROUTE (client/src/App.tsx)

Add these routes inside the router, alongside existing routes:

```typescript
import ResearchHub   from "@/pages/research-hub";
import ResearchEnroll from "@/pages/research-enroll";
import ResearchMorning from "@/pages/research-morning";
import ResearchDaytime from "@/pages/research-daytime";
import ResearchEvening from "@/pages/research-evening";

// Inside <Router>:
<Route path="/research"><AppLayout><ResearchHub /></AppLayout></Route>
<Route path="/research/enroll"><AppLayout><ResearchEnroll /></AppLayout></Route>
<Route path="/research/morning"><AppLayout><ResearchMorning /></AppLayout></Route>
<Route path="/research/daytime"><AppLayout><ResearchDaytime /></AppLayout></Route>
<Route path="/research/evening"><AppLayout><ResearchEvening /></AppLayout></Route>
```

---

## 6. BUILD ORDER (do in this sequence)

| Step | What | Why first |
|---|---|---|
| 1 | Add 5 tables to schema.ts + `npm run db:push` | Everything else depends on it |
| 2 | Add 8 API endpoints to server/routes.ts | Pages need real endpoints |
| 3 | Build `/research/enroll` page | Enrollment generates study codes |
| 4 | Build `/research/morning` page | Simplest page, tests DB write |
| 5 | Build `/research/evening` page | No EEG dependency |
| 6 | Build `/research/daytime` page | Needs EEG wiring, do last |
| 7 | Build `/research` hub + dashboard | Needs all others working |
| 8 | Add sidebar item + routes | Final wiring |

---

## 7. WHAT NOT TO BUILD YET

Keep scope tight. These are explicitly out of scope for now:

- Push notifications (browser push API is complex — use reminder screen in-app for now)
- Researcher analytics dashboard (you can query the DB directly for now)
- Automated compensation payment (handle manually via gift cards)
- Multi-study support (one study for now)
- Data export for IRB (build when you have data to export)

---

## 8. TOTAL SCOPE ESTIMATE

| Component | New files | Changes to existing |
|---|---|---|
| DB tables | schema.ts +5 tables | schema.ts |
| API endpoints | routes.ts +8 endpoints | routes.ts |
| React pages | 5 new `.tsx` files | App.tsx, sidebar.tsx |
| **Total** | **5 new files** | **3 existing files** |

This is a self-contained feature addition. Nothing existing breaks.
