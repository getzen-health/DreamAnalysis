# PRD: Neural Dream Workshop — Research Study Mode

## Introduction

Add a `/study` flow to the existing NeuralDreamWorkshop application to conduct a 2-week human pilot study (N=20-30). Participants complete stress and food-emotion sessions using the Muse 2 headband. All data is stored anonymously and exported for paper analysis. This is the foundation for a peer-reviewed publication targeting Frontiers in Human Neuroscience / arXiv by March 21, 2026.

---

## Goals

- Allow 20-30 anonymous participants to complete EEG study sessions via a web link
- Collect EEG features + self-report surveys for two study blocks (stress at work, food-emotion)
- Store all data in existing Neon PostgreSQL database
- Give Sravya an admin dashboard to monitor participation and export data as CSV
- Be fully deployed on existing Vercel infrastructure — no new hosting needed

---

## User Stories

### US-001: Add study_participants and study_sessions tables
**Description:** As a developer, I need database tables to store participant consent and session data.

**Acceptance Criteria:**
- [ ] Add `study_participants` table: id, participant_code (varchar, unique), age (int), diet_type (varchar), has_apple_watch (bool), consent_text (text), consent_timestamp (timestamp), created_at
- [ ] Add `study_sessions` table: id, participant_code (varchar), block_type (varchar: stress|food|sleep), eeg_features_json (jsonb), pre_eeg_json (jsonb), post_eeg_json (jsonb), survey_json (jsonb), intervention_triggered (bool), created_at
- [ ] Drizzle migration runs successfully
- [ ] Typecheck passes

### US-002: Add study API routes
**Description:** As a developer, I need API endpoints to handle consent, participant registration, and session data submission.

**Acceptance Criteria:**
- [ ] `POST /api/study/consent` — saves participant consent record, returns participant_code
- [ ] `POST /api/study/session/start` — creates pending session record, returns session_id
- [ ] `POST /api/study/session/complete` — saves eeg_features_json, survey_json, marks session complete
- [ ] `GET /api/study/admin/participants` — returns all participants (auth-gated)
- [ ] `GET /api/study/admin/sessions` — returns all sessions with participant codes (auth-gated)
- [ ] `GET /api/study/admin/export-csv` — returns CSV file of all session data (auth-gated)
- [ ] All routes return proper error responses (400/401/500)
- [ ] Typecheck passes

### US-003: Build /study landing page
**Description:** As a potential participant, I want to understand the study and join it from a single page.

**Acceptance Criteria:**
- [ ] Page at `/study` shows: study title, what it is, what participants do, time required (25-40 min), that it's anonymous
- [ ] "Join the Study" button navigates to `/study/consent`
- [ ] Page is mobile-responsive (participants may be on phones)
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-004: Build /study/consent page
**Description:** As a participant, I want to read and digitally sign the consent form before starting.

**Acceptance Criteria:**
- [ ] Page shows full informed consent text (what data is collected, how it's used, voluntary participation, right to withdraw)
- [ ] Checkbox: "I have read and agree to participate voluntarily"
- [ ] Text input: participant enters their assigned code (e.g. P001)
- [ ] "Continue" button disabled until checkbox checked and code entered
- [ ] On submit: POST to `/api/study/consent`, store consent with timestamp
- [ ] On success: navigate to `/study/profile`
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-005: Build /study/profile page
**Description:** As a participant, I want to enter basic demographic info before starting sessions.

**Acceptance Criteria:**
- [ ] Fields: age (number input), diet type (dropdown: omnivore/vegetarian/vegan/other), do you own an Apple Watch (yes/no toggle)
- [ ] "Start Session" button navigates to `/study/session/stress` (Block A first)
- [ ] Profile data saved to `study_participants` record
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-006: Build /study/session/stress page (Block A)
**Description:** As a participant, I want to complete the stress session: baseline EEG, work task, intervention, post-EEG, survey.

**Acceptance Criteria:**
- [ ] Step 1: "Baseline" — 5 min timer, instructional text ("sit still, eyes closed"), captures EEG features from existing `/api/analyze-eeg` endpoint, stores as `pre_eeg_json`
- [ ] Step 2: "Work Task" — 15 min timer, text "Continue your normal work", live stress indicator visible (uses existing stress classification)
- [ ] Step 3: "Intervention" — auto-triggers when stress level > 0.65 OR at 15 min mark; shows existing breathing exercise component (Physio Sigh or Box Breathing); captures intervention_triggered = true
- [ ] Step 4: "Post-session" — 5 min timer, captures post-intervention EEG as `post_eeg_json`
- [ ] Step 5: Survey — "Did you feel stressed during the session? (1-10)", "Did the breathing exercise help? (1-10)", "How do you feel now? (1-10)" — stores as `survey_json`
- [ ] On complete: POST to `/api/study/session/complete`, navigate to `/study/complete`
- [ ] Progress indicator showing which step participant is on
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-007: Build /study/session/food page (Block B)
**Description:** As a participant, I want to complete the food-emotion session: pre-meal EEG, eat, post-meal EEG, survey.

**Acceptance Criteria:**
- [ ] Step 1: Pre-meal survey — "How hungry are you? (1-10)", "What is your current mood? (1-10)"
- [ ] Step 2: "Pre-meal baseline" — 5 min timer with Muse 2 on, captures EEG as `pre_eeg_json`
- [ ] Step 3: "Eat your meal" — instruction screen "Remove headband and eat your normal meal. Return in 15-20 minutes." — countdown timer (user sets own end time)
- [ ] Step 4: "Post-meal EEG" — 10 min timer, Muse 2 back on, captures EEG as `post_eeg_json`
- [ ] Step 5: Post-meal survey — "What did you eat? (text)", "How healthy was it? (1-10)", "Energy level now? (1-10)", "Mood now? (1-10)", "Do you feel satisfied? (1-10)"
- [ ] On complete: POST to `/api/study/session/complete`, navigate to `/study/complete`
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-008: Build /study/complete page
**Description:** As a participant, I want confirmation that my session was saved and know what to do next.

**Acceptance Criteria:**
- [ ] Shows "Session complete — thank you!" confirmation
- [ ] Shows which blocks are done vs. remaining (stress done → food next, or both done)
- [ ] If Block B not yet done: "Book your food session" button → `/study/session/food`
- [ ] If both done: "You're done! Your data helps advance brain science." message
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-009: Build /study/admin dashboard
**Description:** As Sravya, I want to see all participants and session data, and export it as CSV.

**Acceptance Criteria:**
- [ ] Route guarded: only accessible when logged in as admin (existing auth)
- [ ] Table showing: participant_code, age, diet_type, has_apple_watch, sessions_completed, last_session_date
- [ ] Clickable row expands to show session details (block type, survey scores, intervention triggered)
- [ ] "Export CSV" button downloads all session data as a single CSV file with columns: participant_code, block_type, pre_alpha, pre_beta, pre_theta, post_alpha, post_beta, post_theta, intervention_triggered, survey scores
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

---

## Functional Requirements

- FR-1: All participant data keyed to anonymous participant_code (no names stored)
- FR-2: Consent timestamp stored with consent text snapshot (legal record)
- FR-3: EEG features stored as JSON (17 features: 5 band powers, Hjorth params, entropy, ratios, asymmetry)
- FR-4: Study pages accessible via direct URL share (no login required for participants)
- FR-5: Admin dashboard requires existing auth
- FR-6: CSV export includes pre/post EEG band powers + all survey numeric fields
- FR-7: Stress intervention triggers at stress_score > 0.65 OR at 15-minute mark, whichever comes first
- FR-8: All timers show countdown with visual progress indicator
- FR-9: Pages must work on mobile (participants may use phones)

---

## Non-Goals

- No email collection or account creation for participants
- No real-time data sync between participants (offline is fine)
- No automatic Muse 2 pairing in study flow (participants pair separately via existing /device-setup)
- No statistical analysis in the app (done externally in Python)
- No IRB integration
- No payment or incentive system

---

## Technical Considerations

- Extend existing Express server (`server/routes.ts`) for new API routes
- Extend existing Drizzle schema (`shared/schema.ts`) for new tables
- New pages live in `client/src/pages/study/`
- Reuse existing EEG analysis hooks and breathing exercise components
- Admin route uses existing `requireAuth` middleware
- CSV export uses `json2csv` or manual CSV string construction

---

## Success Metrics

- 20-30 participants complete at least Block A by March 14, 2026
- CSV export downloads cleanly with all required columns
- Zero data loss between session completion and DB storage
- Paper submitted to arXiv by March 21, 2026
