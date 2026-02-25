# DATA SECURITY AND PRIVACY PROTECTION PLAN
## Consumer-Grade EEG as a Biomarker for Emotional State and Daily Eating Behavior

**Version:** 1.0
**Date:** February 25, 2026
**IRB Protocol Number:** [TO BE ASSIGNED]
**Principal Investigator:** [YOUR NAME]

---

## 1. OVERVIEW

This document describes the specific technical and procedural measures used to protect participant data throughout the study. It supplements the brief data security summary in Section 7 of the Study Protocol (01_STUDY_PROTOCOL.md).

All data handling follows:
- **45 CFR 46** (Common Rule for human subjects protection)
- **NIST SP 800-171** (protecting Controlled Unclassified Information)
- **GDPR-equivalent principles** (even for US participants, applied as best-practice standard)

---

## 2. DATA INVENTORY

### 2.1 What Is Collected

| Data Element | Format | Collected When | Identifiable? | Retention |
|---|---|---|---|---|
| EEG feature vectors | 85 floats per 4-sec epoch | During session (real-time) | No | 7 years |
| SAM scores (valence + arousal) | 2 integers (1–9) | End of each session | No | 7 years |
| PANAS scores | 20 integers (1–5) | End of each session | No | 7 years |
| Questionnaire responses (Sections A–C) | Text + integers | During session | Pseudonymized | 7 years |
| Demographic data | Age, sex, sleep, exercise, caffeine | During session | Pseudonymized | 7 years |
| Screening survey responses | Multiple choice | Pre-enrollment | Destroyed after enrollment decision | |
| Study ID ↔ Name linkage file | Text (ID + real name + email) | At enrollment | Identifiable | 7 years, then destroyed |
| Consent form (signed) | PDF or paper | At Session 1 | Identifiable | 7 years, then shredded |

### 2.2 What Is NOT Collected

- **Raw EEG voltage traces** — NOT stored. The NeuralDreamWorkshop application processes signal in real-time and stores only pre-computed 85-dimensional feature vectors per 4-second epoch.
- **Audio recordings of sessions** — NOT made.
- **Video recordings of sessions** — NOT made.
- **IP addresses or device identifiers** — NOT logged.
- **Photographs or biometric images** — NOT collected.

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
    NX4T82 | Jane Doe | jane.doe@email.com | 555-0100 | enrolled 2026-03-01
    PQ7R21 | John Smith | ... | ...
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
Participant → [Session] → NeuralDreamWorkshop App
                                    │
                   ┌────────────────┼────────────────────┐
                   ↓                ↓                    ↓
          EEG Features       Questionnaire          SAM/PANAS
          (85 floats)        Responses              Scores
          stored as          stored as text         stored as
          study_id +         study_id +             study_id +
          timestamp          pseudonymized text     integer arrays
                   └────────────────┼────────────────────┘
                                    ↓
                         Encrypted PostgreSQL DB
                         (Neon, SOC 2 Type II)
                         Accessed only via
                         authenticated API
                                    │
                         Never associated with
                         real name or contact info
```

---

## 4. DATA STORAGE SYSTEMS

### 4.1 Primary Database (Research Data)

**System:** Neon PostgreSQL (https://neon.tech)
- **Certification:** SOC 2 Type II certified
- **Encryption at rest:** AES-256 (managed by Neon)
- **Encryption in transit:** TLS 1.3
- **Access control:** Single service account with minimum-necessary permissions
- **Authentication:** Strong password + IP allowlisting (only the PI's known IPs can connect)
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

- Collected via REDCap (if available) or an encrypted Google Forms alternative
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

- All API connections to the Neon database use **TLS 1.3**
- All video call sessions (for remote participants) use **end-to-end encrypted platforms** (Zoom with E2EE enabled, or Signal for video)
- No participant data is ever sent via unencrypted email
- No participant data is ever stored in standard Google Drive, Dropbox, or iCloud without encryption
- If sharing data with collaborators in the future, it will be de-identified first and transmitted via encrypted file transfer (SFTP or Tresorit Send)

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

> *"We are writing to inform you of an incident that may have affected the security of data you provided as part of our research study. [Description of what happened]. The data potentially involved was [description]. We have taken the following steps: [steps taken]. If you have questions or concerns, please contact [PI email/phone]."*

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

**"Secure deletion"** means overwriting the file with zeros or using a dedicated secure-erase tool (macOS Secure Empty Trash equivalent, or `shred` on Linux), not just moving it to the recycle bin.

**7 years** is the standard retention period recommended by most IRBs and FDA regulations for human subjects research records.

---

## 10. FUTURE DATA SHARING PLAN

The study protocol notes that de-identified EEG feature data may be shared publicly as an open dataset to support future research. If this occurs:

1. **IRB notification or amendment** will be submitted before sharing (some IRBs require amendment approval for new data uses, even if de-identified).
2. **Verification of de-identification:** Confirm that the dataset contains no study IDs, no timestamps that could identify individuals, and no unique combinations of demographic data that could re-identify participants (e.g., a 64-year-old female who sleeps exactly 3.5 hours and exercises vigorously every day is potentially identifiable).
3. **Publication format:** NumPy `.npz` or CSV files with only feature columns and label columns. No contact information, no study IDs, no session dates.
4. **License:** Creative Commons Attribution 4.0 (CC BY 4.0), requiring citation of the published paper.
5. **Repository:** Zenodo, OSF, or PhysioNet (standard academic open-data repositories with DOI assignment).

---

## APPENDIX A: DATABASE SCHEMA (PSEUDOCODE)

```sql
-- All tables use study_id (6-char string) as the only identifier.
-- No names, emails, or contact information stored in the database.

TABLE eeg_sessions (
    id              SERIAL PRIMARY KEY,
    study_id        CHAR(6) NOT NULL,     -- e.g., "NX4T82"
    session_num     INTEGER NOT NULL,      -- 1 or 2
    session_date    DATE NOT NULL,         -- date only, no timestamp
    duration_min    FLOAT,
    sqI_mean        FLOAT                  -- signal quality index
);

TABLE eeg_epochs (
    id              SERIAL PRIMARY KEY,
    session_id      INTEGER REFERENCES eeg_sessions(id),
    epoch_index     INTEGER NOT NULL,
    features        FLOAT[85] NOT NULL,    -- 85-dim feature vector
    artifact_flag   BOOLEAN DEFAULT FALSE
);

TABLE questionnaire_responses (
    id              SERIAL PRIMARY KEY,
    session_id      INTEGER REFERENCES eeg_sessions(id),
    section         CHAR(1) NOT NULL,      -- 'A', 'B', or 'C'
    question_key    VARCHAR(50) NOT NULL,
    response_text   TEXT,
    response_num    FLOAT                  -- numeric rating if applicable
);

TABLE validated_scales (
    id              SERIAL PRIMARY KEY,
    session_id      INTEGER REFERENCES eeg_sessions(id),
    sam_valence     INTEGER NOT NULL,      -- 1–9
    sam_arousal     INTEGER NOT NULL,      -- 1–9
    panas_pa        FLOAT NOT NULL,        -- Positive Affect subscale
    panas_na        FLOAT NOT NULL         -- Negative Affect subscale
);

TABLE demographics (
    id              SERIAL PRIMARY KEY,
    study_id        CHAR(6) UNIQUE NOT NULL,
    age             INTEGER,
    sex             VARCHAR(20),
    enrollment_date DATE
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
