# DATA SECURITY AND PRIVACY PROTECTION PLAN
## Consumer-Grade EEG as a Biomarker for Emotional State and Daily Eating Behavior

**Version:** 1.3
**Date:** February 25, 2026
**IRB Protocol Number:** [TO BE ASSIGNED]
**Principal Investigator:** Lakshmi Sravya Vedantham

---

## 1. OVERVIEW

This document describes the specific technical and procedural measures used to protect participant data throughout the study. It supplements the brief data security summary in Section 7 of the Study Protocol (01_STUDY_PROTOCOL.md) and reflects the 30-day longitudinal design described in COMBINED_STUDY_PROTOCOL_DRAFT.md. Data is collected via the NeuralDreamWorkshop app across three daily touchpoints (morning dream journal, daytime EEG session, evening eating/mood questionnaire); see Section 6.2 for in-app consent and automated transmission details.

All data handling follows:
- **45 CFR 46** (Common Rule for human subjects protection)
- **NIST SP 800-171** (protecting Controlled Unclassified Information)
- **GDPR-equivalent principles** (even for US participants, applied as best-practice standard)

---

## 2. DATA INVENTORY

### 2.1 What Is Collected

| Data Element | Format | Collected When | Identifiable? | Retention |
|---|---|---|---|---|
| EEG feature vectors | 85 floats per 4-sec epoch | Daytime session (daily) | No | 7 years |
| FAA, high-beta, FMT, SQI | Derived floats | Daytime session (daily) | No | 7 years |
| SAM scores (valence + arousal) | 2 integers (1–9) | Daytime session (daily) | No | 7 years |
| PANAS scores | 10 integers (1–5) | Daytime session (daily) | No | 7 years |
| Dream journal entries (free text) | Text | Morning session (daily) | **Yes — personal** | 7 years |
| Dream affect ratings (1–9), nightmare flag | Integers + boolean | Morning session (daily) | No | 7 years |
| Welfare check scores | Integer (1–9) | Morning session (daily) | Pseudonymized | 7 years |
| Evening questionnaire: food, emotion, stress | Text + integers | Evening session (daily) | Pseudonymized | 7 years |
| Sleep hours, caffeine, exercise, alcohol | Integers | Evening session (daily) | No | 7 years |
| Optional overnight EEG features | 85 floats per epoch | Overnight (participant-selected nights) | No | 7 years |
| Demographic data | Age, sleep schedule | At enrollment | Pseudonymized | 7 years |
| In-app consent record | Timestamps + booleans | At enrollment | Pseudonymized | 7 years |
| Screening survey responses | Multiple choice | Pre-enrollment | Destroyed after enrollment decision | — |
| Study ID ↔ Name linkage file | Text (ID + real name + email) | At enrollment | Identifiable | 7 years, then destroyed |
| Consent form email (PDF copy) | PDF | At enrollment | Identifiable | Stored by participant only — not retained by researcher app |

### 2.2 What Is NOT Collected

- **Raw EEG voltage traces** — NOT stored. The NeuralDreamWorkshop application processes signal in real-time and stores only pre-computed 85-dimensional feature vectors per 4-second epoch.
- **Audio recordings of sessions** — NOT made.
- **Video recordings of sessions** — NOT made.
- **IP addresses or device identifiers** — NOT logged by the research application. Note: Standard web server access logs may briefly capture IP addresses as part of HTTPS handshaking; these logs are purged on a rolling 7-day cycle and are never associated with participant study IDs or research data.
- **Photographs or biometric images** — NOT collected.
- **Personal identifiers embedded in free-text responses** — The structured questionnaire may elicit free-text responses that incidentally contain names of people or places. The researcher will review and redact any such identifiers during data entry before storage in the research database. Participants are instructed during the session not to include names of specific individuals in their responses.

---

## 3. DE-IDENTIFICATION PROTOCOL

### 3.1 Study ID Assignment

At enrollment (before any data is collected), each participant is assigned a unique study ID:
- **Format:** 6-character alphanumeric string (e.g., "NX4T82")
- **Generation:** Randomly generated using a cryptographically secure random number generator (Python `secrets.token_hex(3).upper()`)
- **Uniqueness:** Checked against existing IDs to prevent duplicates

### 3.2 The Linkage File

The only place where a participant's real identity is linked to their study ID is the **Master Linkage File**:

```
Master Linkage File (linkage.encrypted):
    NX4T82 | Jane Doe | jane.doe@email.com | enrolled 2026-03-01
    PQ7R21 | John Smith | john.smith@email.com | enrolled 2026-03-05
```

**Access controls:**
- Stored ONLY on the PI's local encrypted device (macOS FileVault / Windows BitLocker)
- Backed up ONLY to an end-to-end encrypted cloud service (Tresorit or ProtonDrive)
- Protected by a strong passphrase (20+ characters, not reused anywhere)
- Never stored in the cloud without encryption
- Never emailed or transmitted unencrypted
- The linkage file itself is encrypted with AES-256 using a dedicated passphrase stored in a password manager (1Password or Bitwarden)

**All research data files** use only the 6-character study ID — never the participant's real name, email, or other identifiers.

### 3.3 Data Flow Diagram

```
Participant (30 days, 3 touchpoints/day)
        │
        ├── MORNING: dream journal, affect ratings, welfare check
        │       POST /api/study/morning
        │
        ├── DAYTIME: 10-min EEG session, PANAS, mood ratings
        │       EEG processed ON DEVICE → 85-feature vector
        │       POST /api/study/daytime
        │
        └── EVENING: eating behavior, mood summary, sleep notes
                POST /api/study/evening

All transmissions: HTTPS TLS 1.3 only
        │
        ↓
Express.js Server
        ├── Validates session token → userId
        ├── Resolves userId → studyCode (6-char pseudonymous code)
        └── All database writes use studyCode only
        │
        ↓
Encrypted PostgreSQL DB (Neon, SOC 2 Type II)
        ├── study_participants   → studyCode, consent timestamps
        ├── study_sessions       → studyCode, date, completion flags
        ├── study_morning_entries → studyCode, dream text, affect
        ├── study_daytime_entries → studyCode, EEG features, PANAS
        └── study_evening_entries → studyCode, eating data, mood
        │
        Never associated with real name or contact info
        Real identity stored ONLY in encrypted Master Linkage File
        (separate encrypted file, never in research DB)
```

---

## 4. DATA STORAGE SYSTEMS

### 4.1 Primary Database (Research Data)

**System:** Neon PostgreSQL (https://neon.tech)
- **Certification:** SOC 2 Type II certified
- **Data residency:** Confirm that the selected Neon region stores data exclusively within the United States (e.g., AWS us-east-1 or us-west-2) prior to study initiation. Document the confirmed region in the study records.
- **Encryption at rest:** AES-256 (managed by Neon)
- **Encryption in transit:** TLS 1.3
- **Access control:** Single service account with minimum-necessary permissions
- **Authentication:** Strong password + IP allowlisting (only the PI's known IPs can connect). When the PI operates from a new IP address (e.g., travel), the allowlist will be temporarily updated and reverted upon return, or a VPN with a fixed egress IP will be used.
- **Backups:** Automated daily backups, retained for 30 days, encrypted

**Database schema:** See Appendix A for table definitions.

### 4.2 Master Linkage File

**Primary copy:** PI's laptop
- FileVault (macOS) or BitLocker (Windows) full-disk encryption enabled
- Strong login password required
- File itself encrypted with Veracrypt or 7-Zip AES-256

**Backup copy:** Tresorit or ProtonDrive (end-to-end encrypted)
- E2E encryption means the cloud provider cannot read the file
- Access requires the PI's E2E encryption credentials (not just cloud account password)

### 4.3 Paper Consent Forms

- Stored in a **locked, fireproof drawer** at the PI's home/office
- Access restricted to PI only
- Never photographed and stored digitally without encryption
- Destroyed by cross-cut shredding after the 7-year retention period

### 4.4 Screening Survey Responses

- Collected via Tally or Typeform (HIPAA-compliant plan)
- Eligibility determination only — not linked to study ID
- **Permanently deleted within 30 days** after enrollment decision is made for each respondent

---

## 5. ACCESS CONTROL

| Person | What They Can Access | How Access Is Granted |
|--------|---------------------|----------------------|
| Principal Investigator | All data | Account credentials |
| No one else | — | No other access granted |

This is a single-PI study. No research assistants, students, or collaborators have access to identifiable data. If this changes, the IRB must be notified and this plan updated.

---

## 6. TRANSMISSION SECURITY

### 6.1 General Transmission Controls

- All API connections to the Neon database use **TLS 1.3**
- All video call sessions (for remote onboarding/offboarding) use **end-to-end encrypted platforms** (Zoom with E2EE enabled, or Signal for video)
- No participant data is ever sent via unencrypted email
- No participant data is ever stored in standard Google Drive, Dropbox, or iCloud without encryption
- If sharing data with collaborators in the future, it will be de-identified first and transmitted via encrypted file transfer (SFTP or Tresorit Send)

### 6.2 In-App Consent Flow and Automated Data Transmission

The study uses the **NeuralDreamWorkshop mobile/web application** as the primary data collection platform. Participants complete all daily study tasks (morning dream journal, daytime EEG session, evening questionnaire) through the app. This section describes the security controls specific to this in-app, self-administered protocol.

#### 6.2.1 In-App Consent Process

At enrollment, participants complete a **6-step digital consent wizard** within the app before any study data is collected:

1. **Study Overview** — plain-language summary of the 30-day study
2. **Eligibility Confirmation** — participant confirms all inclusion criteria
3. **Full Consent Form** — scrollable text of the complete informed consent document, with mandatory scroll-to-bottom before proceeding
4. **Consent Initials** — 10 individual checkbox initials (mirroring the paper consent form items)
5. **Optional Overnight EEG** — separate optional consent with its own initial
6. **Preferences** — participant selects daily session times and notification preferences

**Consent record stored:** Upon completion of the wizard, the following are recorded in the database:
- `consentedAt` timestamp (ISO 8601, UTC)
- `consentVersion` (e.g., "2.0") — links to the specific consent document version presented
- `overnightEegConsent` (boolean)
- `preferredMorningTime`, `preferredDaytimeTime`, `preferredEveningTime`

The consent record is stored under the participant's 6-character pseudonymous study code — never their name. It is not possible to enroll in the study (i.e., no study data endpoints will accept submissions) until the consent wizard is completed and the `consentedAt` timestamp is recorded.

**A signed PDF copy** of the consent form, pre-populated with the participant's initials and timestamp, is emailed to the participant at the email address they provided during eligibility screening. This email is sent from the researcher's account (not stored in the app database) and is the only point at which the participant's email address touches the study data pipeline. After sending, the email is not retained by the app.

#### 6.2.2 Automated Daily Data Submission

Each of the three daily touchpoints submits data to the encrypted Neon PostgreSQL database via the app's authenticated REST API. The submission flow is:

```
Participant's Device (app)
        │
        │ HTTPS (TLS 1.3) — POST /api/study/morning
        │                    POST /api/study/daytime
        │                    POST /api/study/evening
        ↓
Express.js Server (authenticated session token required)
        │
        ├── Validates session token → resolves to userId
        ├── Calls getActiveParticipant(userId) → resolves to studyCode
        ├── Calls getOrCreateTodaySession(participant) → idempotent
        └── Inserts data under studyCode only (not userId or real name)
        │
        ↓
Neon PostgreSQL (SOC 2 Type II, AES-256 at rest, TLS 1.3 in transit)
```

**What is transmitted per submission:**

| Endpoint | Data Sent | Sensitive? |
|---|---|---|
| `POST /api/study/morning` | Dream text (free text), SAM valence/arousal (1–9), nightmare flag, welfare check score | **Yes — dream text is personal** |
| `POST /api/study/daytime` | EEG feature vector (85 floats), FAA, high-beta, FMT, SQI, PANAS scores (10 ints), mood/stress/energy ratings | No |
| `POST /api/study/evening` | Meal descriptions (free text or categorical), emotional eating ratings, craving types (categorical), exercise level, alcohol, medications | Mildly sensitive |

**No raw EEG voltage is transmitted.** EEG processing runs entirely on-device (within the app via the FastAPI ML backend). Only processed 85-dimensional feature vectors are sent to the database — equivalent to a step counter sending "10,000 steps" rather than a recording of every muscle movement.

#### 6.2.3 Dream Journal Text — Additional Controls

Dream journal text is the most sensitive data in the study. Additional controls specific to this field:

- **Transmission**: Dream text is transmitted immediately on morning submission over TLS 1.3. It is never cached in browser local storage or device storage after upload.
- **Storage**: Stored in the `study_morning_entries` table under `studyCode` only, in the encrypted Neon database. Never stored in a separate file, spreadsheet, or email.
- **AI processing**: The app passes dream text to an AI language model (OpenAI GPT API) to extract a single numerical emotional valence score (−1.0 to +1.0). This call is made server-side (not from the participant's device). The text is transmitted to OpenAI's servers for processing; no participant identifiers are included in the request, and OpenAI's API terms prohibit use of API-submitted content for model training. Only the extracted score is stored in the analysis tables; the raw dream text is stored separately in the `dreamText` column under the pseudonymous study code.
- **Researcher access**: The PI may read dream text entries only for data quality checks (e.g., verifying the morning flow is working correctly). Dream text is never shared with any third party in identifiable or pseudonymized form.
- **Public dataset**: Dream text will NOT be included in any public open dataset. Only the extracted numerical valence score will be shared if a de-identified dataset is published.
- **Participant skip right**: Participants may skip any dream entry at any time by tapping "Skip — I don't want to record this one." This creates a session row with `dreamSkipped = true` and no text content. Skipping has no consequence on compensation.

#### 6.2.4 Welfare Check and Safety Triggers

The morning submission includes a mandatory welfare check question: *"How are you feeling right now?"* (1–9 scale). The app implements two automatic safety triggers server-side:

- **Single-session trigger**: If welfare score ≤ 2, the app immediately presents mental health resources and offers to notify the researcher.
- **Consecutive-day trigger**: If welfare score ≤ 2 on three consecutive mornings, the app prompts the participant to reach out for support.

These triggers are implemented in the `/api/study/morning` endpoint's response payload. No automatic alert is sent to the researcher without participant action — participant privacy is preserved unless they choose to initiate contact.

---

## 7. DEVICE AND ENDPOINT SECURITY

### PI's Research Computer

- **Full-disk encryption:** FileVault (macOS) or BitLocker (Windows) — mandatory
- **Operating system:** Up to date with latest security patches
- **Screen lock:** Automatic lock after 5 minutes of inactivity, strong password required
- **Firewall:** Enabled
- **Antivirus/antimalware:** macOS XProtect or Windows Defender (enabled)
- **VPN:** Used when on public Wi-Fi for any database access

### NeuralDreamWorkshop Application (Software)

- **Authentication:** All API endpoints require authenticated session tokens
- **HTTPS only:** No HTTP connections accepted
- **Environment variables:** API keys and database credentials stored as environment variables, never hardcoded or committed to git
- **No analytics SDKs:** No third-party analytics (e.g., Google Analytics) that could log session behavior

---

## 8. INCIDENT RESPONSE (DATA BREACH PROTOCOL)

In the event of a suspected or confirmed data breach:

### Step 1 — Contain (within 1 hour of discovery)
- Immediately revoke compromised credentials and rotate all API keys
- Disconnect the affected system from the network if applicable
- Preserve logs for forensic analysis (do not delete)

### Step 2 — Assess (within 24 hours)
- Determine what data was accessed or exposed
- Determine whether any identifiable data was involved
- Determine the number of participants potentially affected

### Step 3 — Notify — Participants (within 72 hours)
If identifiable data may have been exposed, notify affected participants at the email address they provided at enrollment:

> *"We are writing to inform you of an incident that may have affected the security of data you provided as part of our research study. [Description of what happened]. The data potentially involved was [description]. We have taken the following steps: [steps taken]. If you have questions or concerns, please contact lakshmisravya.vedantham@gmail.com."*

### Step 4 — Notify — IRB (within 5 business days)
Submit an Unanticipated Problem Report to the approving IRB per their reporting procedures.

### Step 5 — Document
Document all aspects of the incident, assessment, notifications, and remediation actions in a permanent incident log.

---

## 9. DATA RETENTION AND DESTRUCTION

| Data Type | Retention Period | Destruction Method |
|---|---|---|
| EEG features + questionnaire responses (pseudonymized) | 7 years from study completion | Secure database deletion (NIST 800-88 compliant) |
| SAM / PANAS scores (pseudonymized) | 7 years from study completion | Secure database deletion |
| Master Linkage File | 7 years from study completion | Secure file deletion + shredding |
| Paper consent forms | 7 years from study completion | Cross-cut shredding |
| Screening survey responses | 30 days after enrollment decision | Permanent deletion from platform |

**"Secure deletion"** means using a platform-appropriate secure-erase method — not simply moving to the recycle bin:
- **macOS (APFS filesystem):** Use `rm -P` for HFS+ volumes, or for APFS (which uses copy-on-write and makes traditional overwrite ineffective), store sensitive files in an encrypted disk image (`.dmg`) and delete the entire encrypted container. Full-disk FileVault encryption provides equivalent protection for APFS volumes.
- **Linux:** Use `shred -u` to overwrite and remove the file.
- **Database records:** Use the database provider's secure delete API or issue a `DELETE` SQL statement followed by `VACUUM` to remove dead tuples from PostgreSQL storage pages.

**7 years** is the standard retention period recommended by most IRBs and FDA regulations for human subjects research records.

---

## 10. FUTURE DATA SHARING PLAN

The study protocol notes that de-identified EEG feature data may be shared publicly as an open dataset to support future research. If this occurs:

1. **IRB notification or amendment** will be submitted before sharing (some IRBs require amendment approval for new data uses, even if de-identified).
2. **Verification of de-identification:** Confirm that the dataset contains no study IDs, no timestamps that could identify individuals, and no unique combinations of demographic data that could re-identify participants (e.g., a 64-year-old who sleeps exactly 3.5 hours per night is potentially identifiable).
3. **Publication format:** NumPy `.npz` or CSV files with only feature columns and label columns. No contact information, no study IDs, no session dates.
4. **License:** Creative Commons Attribution 4.0 (CC BY 4.0), requiring citation of the published paper.
5. **Repository:** Zenodo, OSF, or PhysioNet (standard academic open-data repositories with DOI assignment).

---

## APPENDIX A: DATABASE SCHEMA (PSEUDOCODE)

```sql
-- All tables use study_code (6-char string) as the only identifier.
-- No names, emails, or contact information stored in the database.
-- Full TypeScript/Drizzle schema defined in docs/RESEARCH_MODULE_SPEC.md.

TABLE study_participants (
    id                    SERIAL PRIMARY KEY,
    user_id               INTEGER NOT NULL,           -- internal app user ID (not study_code)
    study_code            CHAR(6) UNIQUE NOT NULL,    -- e.g., "NX4T82" — only field linking to user
    status                VARCHAR DEFAULT 'active',   -- 'active' | 'withdrawn' | 'completed'
    consented_at          TIMESTAMP NOT NULL,
    consent_version       VARCHAR NOT NULL,           -- e.g., "2.0"
    overnight_eeg_consent BOOLEAN DEFAULT FALSE,
    enrolled_date         DATE NOT NULL,
    completed_days        INTEGER DEFAULT 0,
    preferred_morning_time  TIME,
    preferred_daytime_time  TIME,
    preferred_evening_time  TIME
);

TABLE study_sessions (
    id              SERIAL PRIMARY KEY,
    participant_id  INTEGER REFERENCES study_participants(id),
    session_date    DATE NOT NULL,
    morning_done    BOOLEAN DEFAULT FALSE,
    daytime_done    BOOLEAN DEFAULT FALSE,
    evening_done    BOOLEAN DEFAULT FALSE,
    valid_day       BOOLEAN DEFAULT FALSE,   -- TRUE if ≥2 of 3 touchpoints completed
    UNIQUE(participant_id, session_date)
);

TABLE study_morning_entries (
    id              SERIAL PRIMARY KEY,
    session_id      INTEGER REFERENCES study_sessions(id),
    dream_text      TEXT,                    -- free text; may be null if skipped
    dream_skipped   BOOLEAN DEFAULT FALSE,
    dream_valence   FLOAT,                   -- AI-extracted score: -1.0 to +1.0
    sam_valence     INTEGER,                 -- 1–9
    sam_arousal     INTEGER,                 -- 1–9
    nightmare_flag  BOOLEAN,
    welfare_score   INTEGER,                 -- 1–9 morning wellbeing check
    submitted_at    TIMESTAMP NOT NULL
);

TABLE study_daytime_entries (
    id              SERIAL PRIMARY KEY,
    session_id      INTEGER REFERENCES study_sessions(id),
    eeg_features    JSONB,                   -- 85-dim feature vector
    faa             FLOAT,                   -- frontal alpha asymmetry
    high_beta       FLOAT,                   -- high-beta (stress) power
    fmt             FLOAT,                   -- frontal midline theta
    sqi_mean        FLOAT,                   -- signal quality index
    panas_scores    JSONB,                   -- 10 item scores
    mood_rating     INTEGER,                 -- 1–9
    energy_rating   INTEGER,                 -- 1–9
    stress_rating   INTEGER,                 -- 1–9
    sleep_hours     FLOAT,
    caffeine_mg     INTEGER,
    submitted_at    TIMESTAMP NOT NULL
);

TABLE study_evening_entries (
    id                    SERIAL PRIMARY KEY,
    session_id            INTEGER REFERENCES study_sessions(id),
    meals                 JSONB,             -- categorical meal descriptions
    emotional_eating_day  INTEGER,           -- 1–9
    craving_types         TEXT[],            -- categorical
    exercise_level        INTEGER,           -- 0–3
    alcohol_units         FLOAT,
    medications           TEXT,              -- free text
    day_valence           INTEGER,           -- 1–9 overall day mood
    significant_event     TEXT,              -- one-sentence description
    event_intensity       INTEGER,           -- 1–9
    submitted_at          TIMESTAMP NOT NULL
);
```

---

## APPENDIX B: PARTICIPANT CONTACT (FOR BREACH NOTIFICATION)

The PI maintains a separate encrypted file (not the research database) with:

```
study_id | email (for breach notification only)
```

This file is:
- Stored only on the PI's encrypted device
- Never in the research database
- Destroyed at the end of the retention period together with the Master Linkage File
