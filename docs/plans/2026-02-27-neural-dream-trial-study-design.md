# Design Doc: Neural Dream Workshop — Human Trial Study Mode

**Date:** 2026-02-27
**Author:** Sravya Vedantham (Independent Researcher)
**Status:** Approved — proceed to implementation

---

## Overview

Add a `/study` research mode to the existing NeuralDreamWorkshop app to conduct a 2-week human pilot study (N=20-30 participants). The study collects EEG data during stress and food-emotion scenarios, feeding into a peer-reviewed paper targeting arXiv + Frontiers in Human Neuroscience (mid-March 2026 deadline).

**Core thesis of the paper:**
> *Consumer health technology ignores the emotional layer. Real-time EEG monitoring can detect stress, food-emotional states, and sleep quality — and deliver personalized interventions. This is a gap no existing product (Apple Health, Fitbit, Oura) addresses.*

---

## Study Protocol

### Block A — Stress at Work (25 minutes)
1. Muse 2 on → 5 min resting baseline (eyes closed)
2. 15 min real work task (participant does their actual work)
3. App auto-detects stress spike → fires guided breathing intervention
4. 5 min post-intervention EEG capture
5. Post-session survey: "Did you feel stressed? Did the intervention help? (1-10)"

### Block B — Food + Emotional State (40 minutes)
1. Muse 2 on → 5 min pre-meal baseline (hungry/pre-eating state EEG)
2. Muse 2 off → participant eats their normal meal (uncontrolled — that's data)
3. Muse 2 back on 5-10 min after eating → 10 min post-meal EEG
4. Survey: what did they eat, healthy rating (1-10), energy before/after (1-10), mood before/after (1-10)
5. Analysis: pre vs. post EEG delta patterns correlated with food quality self-report

### Block C — Sleep + Dream Recall (optional, async, no Muse needed)
- Import Apple Health REM duration + sleep efficiency
- Morning survey: dream recall (1-10), mood (1-10), energy (1-10)
- Uses existing NeuralDreamWorkshop dream journal feature

---

## Privacy Design

- No names stored — each participant gets an anonymous code (P001, P002...)
- Sravya assigns codes offline, participants enter code to start
- All DB records keyed to participant code only
- Digital consent form stored with timestamp
- Paper language: "N=20-30 anonymized participants, written informed consent obtained"

---

## App Architecture

### New Pages (7)

```
/study                    → Study overview + Join CTA
/study/consent            → Digital consent form → stored in DB
/study/profile            → Participant code + age + diet type + Apple Watch y/n
/study/session/stress     → Block A full flow
/study/session/food       → Block B full flow
/study/session/sleep      → Block C: Apple Health import + morning survey
/study/admin              → Admin-only: all participants, sessions, CSV export
```

### New DB Tables (2)

**study_participants**
```
id, participant_code (P001...), age, diet_type, has_apple_watch,
consent_text, consent_timestamp, created_at
```

**study_sessions**
```
id, participant_code, block_type (stress|food|sleep),
eeg_features_json, survey_json, intervention_triggered (bool),
pre_session_eeg_json, post_session_eeg_json, created_at
```

### New API Routes
```
POST /api/study/consent
POST /api/study/participant
POST /api/study/session/start
POST /api/study/session/complete
GET  /api/study/admin/participants
GET  /api/study/admin/sessions
GET  /api/study/admin/export-csv
```

### Admin Access
- Route guarded by existing auth (Sravya's account only)
- CSV export downloads all session data for Python/pandas analysis

---

## Paper Strategy

| Venue | Odds | Notes |
|---|---|---|
| arXiv | 100% | Submit system description + model benchmarks immediately |
| Frontiers in Human Neuroscience | 50% | Rolling submission, pilot studies accepted |
| JMIR Mental Health | 45% | Digital health focus, small studies ok |
| IEEE EMBC 2026 | 65% | Technology/systems track, lenient on IRB |
| ACM CHI Late Breaking Work | 55% | Written consent sufficient |

**Strategy:** arXiv first (system paper, this week). Full pilot study paper to Frontiers/JMIR after trials complete.

**For Green Card (EB-1A/O-1):** arXiv + one peer-reviewed publication + citations = strong evidence of original contribution.

---

## Timeline

```
Feb 27 - Mar 1:   Build /study mode (agents, 3 days)
Mar 1:            Go live — send link to 20-30 participants
Mar 1 - Mar 14:   Run trials (Block A + Block B per participant)
Mar 14 - Mar 17:  Data analysis (Python/pandas, paired t-tests)
Mar 17 - Mar 20:  Write paper
Mar 21:           Submit to arXiv + Frontiers in Human Neuroscience
```

---

## Paper Claims (Realistic, Based on Pilot Data)

1. **Stress detection:** EEG beta/alpha ratio correlates with self-reported stress (Pearson r > 0.5 expected)
2. **Intervention effectiveness:** Guided breathing reduced stress markers X% within 3 minutes (paired t-test)
3. **Food-emotion:** Post-meal EEG shows distinct patterns correlated with food quality self-report
4. **System feasibility:** NeuralDreamWorkshop runs reliably in consumer setting (N=20-30, no adverse events)

---

## What This Is NOT Claiming

- Not a clinical diagnostic tool
- Not making medical claims
- Not replacing professional mental health treatment
- Not generalizable beyond pilot study (acknowledged in paper limitations)
