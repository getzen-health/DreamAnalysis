# STUDY PROTOCOL
## Consumer-Grade EEG as a Biomarker for Emotional State and Daily Eating Behavior: A Validation Study

**Version:** 1.0
**Date:** February 25, 2026
**IRB Submission Type:** Expedited Review (45 CFR 46.110, Categories 6 & 7)
**Principal Investigator:** [YOUR NAME], Independent Researcher
**Contact:** [YOUR EMAIL] | [YOUR PHONE]

---

## 1. BACKGROUND AND RATIONALE

### 1.1 Scientific Background

Electroencephalography (EEG) is a non-invasive neuroimaging technique that measures
electrical activity produced by the brain via electrodes placed on the scalp. EEG is
well-established in clinical and research settings for measuring cognitive and emotional
states. Consumer-grade EEG headsets — particularly the Muse 2 (InteraXon Inc.) —
have made EEG accessible outside laboratory environments, enabling real-world
naturalistic data collection.

Frontal Alpha Asymmetry (FAA), the difference in alpha-band (8–12 Hz) power between
right and left frontal cortices, is the most validated EEG marker of emotional valence,
with 30+ years of replication (Davidson, 1992; Coan & Allen, 2004). High-beta power
(20–30 Hz) correlates reliably with stress and anxiety (Giannakakis et al., 2019).
Theta activity (4–8 Hz) in the frontal midline correlates with working memory load and
self-regulatory behavior (Klimesch, 1999).

Despite extensive research linking these EEG biomarkers to emotion, no published study
has validated the use of real-time consumer-grade EEG to predict daily eating behavior
and food-related emotional states. Research on the neural correlates of hunger, craving,
and satiety has been conducted exclusively in laboratory settings using fMRI or
research-grade EEG with controlled food stimuli (Killgore & Yurgelun-Todd, 2005;
Frank et al., 2010). The present study is the first to test whether consumer EEG can
detect food-emotion states under naturalistic, ecologically valid conditions.

### 1.2 Specific Aims

**Aim 1:** Validate the accuracy of consumer-grade EEG (Muse 2) emotion classification
against participants' own self-reported emotional states collected via a structured
end-of-day questionnaire.

**Aim 2:** Examine whether EEG-derived biomarkers (FAA, theta, high-beta, delta)
correlate with self-reported food consumption patterns and predicted eating states
(craving, satiety, stress eating, mindful eating).

**Aim 3:** Assess the improvement in classification accuracy achieved through individual
baseline calibration versus population-average thresholds.

### 1.3 Hypothesis

We hypothesize that:
(H1) EEG-derived emotion classifications will show significant agreement with
self-reported emotional states (target: >65% cross-subject accuracy, Cohen's κ > 0.40).

(H2) EEG biomarker patterns measured immediately before or during meals will
significantly differ from patterns measured at least two hours after a meal, consistent
with hunger vs. satiety states.

(H3) Personalized baseline calibration (+15–29% accuracy expected) will significantly
improve individual-level classification compared to population-average thresholds.

---

## 2. STUDY DESIGN

**Design:** Cross-sectional, repeated-measures observational study.
**Setting:** Participants' natural environment (home or office) via remote participation,
OR at a designated collection site. No laboratory induction of emotions or food stimuli.
**Sessions per participant:** 2 sessions, 7 days apart (to assess test-retest reliability).
**Session duration:** Approximately 45–60 minutes each.

### 2.1 Session Timeline

**Session 1 (45–60 min):**
| Time | Activity |
|------|----------|
| 0–5 min | Consent review, device fitting, headset quality check |
| 5–7 min | 2-minute eyes-closed resting baseline (mandatory for calibration) |
| 7–10 min | 2-minute eyes-open resting baseline (fixation cross) |
| 10–40 min | Structured questionnaire while EEG records continuously |
| 40–50 min | Validated emotion scales (SAM, PANAS) |
| 50–55 min | Debrief, compensation |

**Session 2 (45 min, 7 days later):**
Identical to Session 1. Provides test-retest data on EEG biomarker reliability.

### 2.2 Questionnaire Content

The structured interview covers the prior 24 hours and is administered verbally by the
researcher (or via app interface) while EEG is recording:

**Section A — Daily Emotional Experience:**
- "How would you rate your overall mood today?" (1–9 SAM valence scale)
- "How energized or calm did you feel?" (1–9 SAM arousal scale)
- "Did anything particularly stressful happen?" (open-ended, then 1–9 scale)
- "Did you feel anxious at any point today?" (yes/no, then 1–9 scale)
- "How would you describe your focus levels today?" (1–9 scale)
- Three significant events from the day with associated emotion ratings

**Section B — Food Consumption:**
- "Walk me through everything you ate and drank in the last 24 hours."
- "Before each meal or snack: were you physically hungry, emotionally driven, or habit?"
  (3-choice: Physical hunger / Emotional need / Habit/social)
- "After your last main meal, how did you feel?" (1–9 fullness scale)
- "Did you experience any cravings today? What kind?" (free-text + category: sweet,
  salty, fatty, comfort food, or none)
- "Would you characterize your eating today as mindful, rushed, or emotional?" (3-choice)

**Section C — Context Factors (Covariates):**
- Hours of sleep last night (numeric)
- Physical exercise today (none / light / moderate / vigorous)
- Caffeine intake (cups)
- Alcohol intake (standard drinks)
- Menstrual cycle phase (if applicable; optional)
- Any medication that may affect EEG (antihistamines, stimulants, etc.)

### 2.3 EEG Recording

**Hardware:** Muse 2 EEG headband (InteraXon Inc., Toronto, Canada)
- 4 channels: TP9 (left temporal), AF7 (left frontal), AF8 (right frontal), TP10 (right temporal)
- Sampling rate: 256 Hz
- Wireless: Bluetooth LE

**Software:** NeuralDreamWorkshop application (custom FastAPI + React stack)
- Signal processing: 1–50 Hz Butterworth bandpass (zero-phase), 60 Hz notch filter
- Mastoid re-referencing: AF7/AF8 referenced to (TP9+TP10)/2
- Artifact rejection: epochs with amplitude > 100 µV or kurtosis > 10 are flagged and excluded
- Feature extraction: 85-dimensional per-epoch feature vector (5 frequency bands × 4 channels × 4 statistics + 5 DASM asymmetry features)
- Signal quality index (SQI) displayed live; sessions paused if SQI < 40%

**EEG Features Extracted:**
- Band powers: delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–12 Hz), beta (12–30 Hz), high-beta (20–30 Hz)
- Frontal Alpha Asymmetry (FAA): ln(AF8_alpha) − ln(AF7_alpha)
- Differential Asymmetry (DASM) across all 5 bands
- Frontal Midline Theta (FMT)
- Hjorth parameters (activity, mobility, complexity)
- Spectral entropy and differential entropy per band

**What is NOT recorded:** Raw EEG voltage traces are NOT stored. Only pre-processed
85-dimensional feature vectors per 4-second epoch are retained.

---

## 3. PARTICIPANTS

### 3.1 Sample Size

**Target N = 50 participants** (allowing for up to 10% dropout, effective N ≥ 45).

**Power analysis:** Using G*Power 3.1, for a paired t-test (Session 1 vs. Session 2
FAA reliability), with effect size d = 0.50 (medium, conservative for EEG), α = 0.05,
power = 0.80, required N = 34. We recruit 50 to account for attrition and to power
the cross-subject classification analysis (leave-one-subject-out cross-validation).

### 3.2 Inclusion Criteria

- Age 18–65 years
- Fluent English speaker (for questionnaire comprehension)
- No self-reported neurological or psychiatric diagnosis (see exclusion criteria)
- Willing and able to wear EEG headset for approximately 45 minutes
- Owns or has access to a smartphone or computer (for remote sessions, if applicable)
- Able to provide informed consent

### 3.3 Exclusion Criteria

- Current diagnosis of epilepsy, seizure disorder, or history of seizures
- Current diagnosis of a neurological condition affecting EEG signal (e.g., Parkinson's disease, traumatic brain injury, dementia)
- Active diagnosis of a psychotic disorder (e.g., schizophrenia) — EEG signals may be atypical
- Current use of medications with known strong EEG effects (benzodiazepines, antipsychotics, anticonvulsants) — these significantly alter band powers and would confound results
- Scalp conditions, open wounds, or skin conditions on the forehead or temporal region that would prevent safe headset use
- Pregnancy (excluded due to known hormonal effects on EEG band powers and unknown effects of sustained wireless Bluetooth proximity to the abdomen — erring on the side of caution)
- Participation in another EEG research study in the past 30 days

### 3.4 Recruitment

Participants will be recruited via:
- Flyers posted at local gyms, coffee shops, community centers, and college campuses
- Online postings (Nextdoor, Craigslist, local Facebook groups, Reddit r/[local city])
- Word-of-mouth referrals from enrolled participants

Enrollment will be first-come, first-served until N = 50 is reached.

---

## 4. PROCEDURES

### 4.1 Screening

Interested individuals will complete a brief online screening survey (REDCap or Google
Forms) covering inclusion/exclusion criteria. Eligible candidates will be contacted
within 3 business days to schedule their first session.

### 4.2 Informed Consent

Prior to any study procedures, the researcher will review the consent form verbally
with the participant, answer all questions, and obtain written (or e-signature) consent.
Participants will be given a copy of the signed consent form.

Participants will be reminded that participation is voluntary and they may withdraw at
any time without penalty.

### 4.3 EEG Setup and Quality Check

The researcher will assist the participant in fitting the Muse 2 headset. The app
displays a real-time Signal Quality Index (SQI) for each electrode. The researcher
will ensure SQI ≥ 60% on all four channels before proceeding. If adequate signal
cannot be obtained within 10 minutes, the session will be rescheduled.

### 4.4 Baseline Resting State

Participants will close their eyes and rest quietly for 2 minutes (eyes-closed baseline)
followed by 2 minutes of eyes-open resting with a fixation cross on screen. This
establishes the individual's personalized neurophysiological baseline used to normalize
all subsequent EEG readings.

### 4.5 EEG + Questionnaire Session

The researcher administers the structured questionnaire (Sections A, B, C as described
in Section 2.2) while the EEG headset records continuously. Participants answer
naturally; they are not asked to induce emotions or change their behavior.
Questionnaire responses are recorded by the researcher in a paper log or secure
digital form.

### 4.6 Validated Emotion Scales

After the questionnaire, participants complete two standardized scales:
- **Self-Assessment Manikin (SAM):** 9-point pictographic scales for valence and arousal. Takes ~2 minutes.
- **PANAS (Positive and Negative Affect Schedule):** 20-item validated affect measure. Takes ~5 minutes.

These serve as gold-standard ground-truth labels for EEG validation (Aim 1).

### 4.7 Debrief and Compensation

The researcher will briefly explain the study's purpose, answer any questions, and
provide compensation. Participants will be invited to return in 7 days for Session 2.

---

## 5. RISKS AND BENEFITS

### 5.1 Potential Risks

**Physical risks:** Minimal. The Muse 2 is a commercially available consumer device
that has been used in thousands of consumer and research contexts. It uses passive
dry electrodes with no electrical current delivered to the participant. The most
common discomfort reported is mild scalp pressure from the headset frame. No adverse
events related to Muse 2 use have been reported in peer-reviewed literature.

**Psychological risks:** The questionnaire asks about daily emotional experiences
including stress and anxiety. While unlikely to cause distress, some participants may
find reflection on a difficult day mildly uncomfortable. Participants will be reminded
they may skip any question. A referral list for free/low-cost mental health resources
will be provided to all participants.

**Privacy risks:** The study collects EEG data and personal daily experiences including
food habits. These are sensitive data. See Section 7 (Data Security) for protections.

**Classification accuracy risks:** Participants will not be given real-time emotion
feedback during the session to avoid influencing questionnaire responses.

### 5.2 Benefits

**Direct benefits:** Participants receive no direct medical or therapeutic benefit.
Some participants may find reflecting on their emotional and dietary patterns personally
informative.

**Societal benefits:** This study will be the first to validate consumer-grade EEG as
a biomarker for naturalistic eating behavior, with potential applications in:
- Personalized nutrition and wellness technology
- Stress-eating intervention apps
- Low-cost neurobiological assessment tools for public health research

### 5.3 Risk-Benefit Assessment

Risks are minimal and the same as, or less than, those ordinarily encountered in daily
life. The potential societal benefit of validating a novel, accessible neurobiological
measurement tool is significant. The risk-benefit ratio is favorable for conducting
this research.

---

## 6. COMPENSATION

Participants will receive **$25 per session** ($50 total for both sessions) as a
gift card (Amazon or equivalent) provided at the end of each session. Compensation
is not contingent on completing both sessions; participants who complete only Session 1
will receive $25. This compensation is commensurate with the time commitment and
local market rates.

---

## 7. DATA SECURITY AND PRIVACY

### 7.1 Data Collected

| Data Type | Format | Identifiable? |
|-----------|--------|---------------|
| EEG features | 85-dim numeric vectors per 4-sec epoch | No (no raw signal stored) |
| SAM / PANAS scores | Numeric (1–9 / 1–5 scales) | No |
| Questionnaire responses | Text + numeric ratings | Pseudonymized |
| Demographic info | Age, sex, sleep, exercise | Pseudonymized |
| Screening survey | Yes/No eligibility responses | Destroyed after enrollment |

### 7.2 De-identification

Participants are assigned a random 6-character alphanumeric study ID (e.g., "NX4T82")
at enrollment. All data files use this ID only. The master linkage list (ID ↔ real name)
is stored in a separate password-protected file accessible only to the PI.

### 7.3 Storage

- **EEG feature data and questionnaire scores:** Stored in an encrypted PostgreSQL
  database (Neon, SOC 2 Type II certified) accessible only via authenticated API.
- **Master linkage list:** Stored locally on PI's encrypted device (FileVault / BitLocker)
  and backed up to an encrypted cloud drive (Tresorit or equivalent).
- **Session audio/video (if any):** Not collected. Sessions are not recorded.
- **Paper forms:** Stored in a locked drawer; destroyed by shredding after data entry.

### 7.4 Retention and Destruction

Data will be retained for 7 years following study completion per standard research
data retention guidelines, then securely destroyed. De-identified EEG feature data
may be shared publicly as an open dataset (following IRB notification/amendment
if required) to support future research.

### 7.5 Breach Response

In the event of a data breach, affected participants will be notified within 72 hours
via the contact email provided at enrollment. The IRB will be notified within 5 business
days as required.

---

## 8. CONFLICT OF INTEREST

The PI has no financial relationship with InteraXon Inc. (Muse 2 manufacturer) or
any entity with a financial interest in the study outcome. The NeuralDreamWorkshop
software used in this study is developed independently by the PI as an open-source
project.

---

## 9. STATISTICAL ANALYSIS PLAN

### Primary Analysis (Aim 1 — Emotion Validation)

- **Classification accuracy:** Leave-One-Subject-Out Cross-Validation (LOSO-CV)
  comparing EEG-predicted 3-class emotion label (positive/neutral/negative) against
  SAM-derived label. Primary metric: accuracy (%). Secondary: Cohen's κ.
- **Target:** Accuracy > 65%, κ > 0.40 (moderate agreement).

### Secondary Analysis (Aim 2 — Food-Emotion Correlation)

- **Biomarker comparison:** Paired t-tests (pre- vs. post-meal EEG epochs on FAA,
  theta, high-beta). Bonferroni correction for multiple comparisons.
- **Food state classification:** Kappa coefficient between EEG-predicted food state
  and self-reported eating motivation (emotional / physical / habit).
- **Regression:** Hierarchical linear regression with FAA, theta, high-beta as
  predictors of PANAS scores; controlling for sleep, exercise, and caffeine.

### Tertiary Analysis (Aim 3 — Calibration Effect)

- **Paired t-test:** Classification accuracy before vs. after individual baseline
  calibration. Effect size: Cohen's d.

### Missing Data

Sessions with SQI < 60% for more than 30% of the recording will be excluded from
analysis. We expect < 10% of sessions to be excluded based on prior Muse 2 research.

---

## 10. REFERENCES

1. Davidson, R.J. (1992). Anterior cerebral asymmetry and the nature of emotion. *Brain and Cognition, 20*(1), 125–151.
2. Coan, J.A., & Allen, J.J.B. (2004). Frontal EEG asymmetry as a moderator and mediator of emotion. *Biological Psychology, 67*(1–2), 7–49.
3. Giannakakis, G., et al. (2019). Review on psychological stress detection using biosignals. *IEEE Transactions on Affective Computing, 13*(1), 440–460.
4. Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance: a review and analysis. *Brain Research Reviews, 29*(2–3), 169–195.
5. Killgore, W.D., & Yurgelun-Todd, D.A. (2005). Developmental changes in the functional brain responses of adolescents to images of high and low-calorie foods. *Developmental Psychobiology, 47*(4), 377–397.
6. Frank, S., et al. (2010). Processing of food pictures: Influence of hunger, gender and calorie content. *Brain Research, 1food*, 2–11.
7. Cannard, C., et al. (2021). The feasibility of using low-cost electroencephalography (EEG) to study the human brain during middle school instruction. *Frontiers in Human Neuroscience, 15*, 541963.
8. Krigolson, O.E., et al. (2017). Choosing MUSE: Validation of a low-cost, portable EEG system for ERP research. *Frontiers in Neuroscience, 11*, 109.
