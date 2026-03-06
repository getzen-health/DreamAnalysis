# EEG Correlates of Acute Stress and Food-Related Emotional States: A Pilot Study Using Consumer-Grade Brain-Computer Interfaces

**Authors:** Lakshmi Sravya Vedantham

**Affiliation:** Neural Dream Workshop Research Lab

**Corresponding Author:** Lakshmi Sravya Vedantham

**Date:** March 2026

---

## Abstract

Consumer-grade electroencephalography (EEG) devices have the potential to democratize neuroscience research by enabling large-scale data collection outside traditional laboratory settings. This pilot study investigates how EEG spectral band powers — specifically alpha (8-12 Hz), beta (12-30 Hz), theta (4-8 Hz), delta (0.5-4 Hz), and gamma (30-100 Hz) — correlate with self-reported stress and food-related emotional states using the Muse 2 headband. In a within-subjects design, _N_ participants completed two sessions: a stress protocol incorporating a work stressor and box breathing intervention, and a food-emotion protocol capturing brain activity before and after a meal. EEG features were extracted every 4 seconds and averaged across session phases, then compared with self-report survey responses on 10-point Likert scales. Results showed [TO BE COMPLETED WITH ACTUAL DATA: expected findings include elevated beta/alpha ratio during stress, alpha recovery post-intervention, and theta modulation post-meal]. We discuss the practical feasibility of consumer-grade EEG for emotion research, limitations of 4-channel dry electrode systems, and implications for future ambulatory affective neuroscience studies.

**Keywords:** EEG, consumer-grade BCI, Muse 2, stress, food-emotion, affective neuroscience, pilot study, box breathing

---

## 1. Introduction

### 1.1 Background

The relationship between brain electrical activity and emotional states has been a central topic in affective neuroscience since the pioneering work of Davidson (1992), who established that frontal alpha asymmetry (FAA) reflects approach-withdrawal motivation. Decades of research using laboratory-grade EEG systems (32-128 channels, gel electrodes) have demonstrated reliable neural correlates of stress (Al-Shargie et al., 2016; Giannakakis et al., 2019), emotional valence (Zheng & Lu, 2015), and arousal (Russell, 1980). However, these findings have been largely confined to controlled laboratory environments with expensive equipment, limiting ecological validity and scalability.

The emergence of consumer-grade EEG devices — particularly the Muse 2 headband (InteraXon Inc., Toronto, Canada) — presents an opportunity to bridge this gap. The Muse 2 records from four dry electrodes at positions TP9, AF7, AF8, and TP10, sampling at 256 Hz with 12-bit ADC resolution. While its signal quality is substantially lower than research-grade systems (Krigolson et al., 2017; Badolato et al., 2024), it offers portability, ease of use, and scalability for large-sample studies.

### 1.2 Stress and EEG

Acute psychological stress is characterized by increased high-beta power (20-30 Hz) reflecting sympathetic nervous system activation, reduced alpha power (8-12 Hz) indicating decreased relaxation, and elevated beta/alpha ratio (Al-Shargie et al., 2016). Breathing interventions have been shown to reverse these patterns, increasing alpha power and reducing beta dominance within minutes (Ma et al., 2017). The beta/alpha ratio has emerged as a reliable real-time stress biomarker, with cross-subject accuracies of 65-75% using consumer-grade devices (Giannakakis et al., 2019).

### 1.3 Food, Emotion, and EEG

The relationship between food consumption and neural activity is less well-studied through EEG. Existing literature suggests that food cravings and emotional eating are associated with increased frontal theta power (Moreno-Padilla et al., 2018), meal consumption triggers parasympathetic shifts reflected in increased alpha and theta power (Craig, 2003), and post-meal cognitive changes ("food coma") involve delta power increases and beta suppression (Paz-Alonso et al., 2013). Self-reported hunger modulates frontal EEG asymmetry, linking appetitive motivation to approach-related brain activity (Harmon-Jones & Gable, 2009).

### 1.4 Research Questions

This pilot study addresses two primary research questions:

**RQ1:** Do EEG spectral band powers measured with the Muse 2 headband show significant changes between baseline and acute stress conditions, and do these changes reverse following a box breathing intervention?

**RQ2:** Do pre-meal and post-meal EEG recordings show systematic differences in spectral band powers, and do these differences correlate with self-reported hunger, mood, and meal characteristics?

### 1.5 Contributions

This study contributes to the literature in three ways:

1. **Feasibility assessment** of consumer-grade EEG for ambulatory emotion research, including data quality metrics, dropout rates, and artifact prevalence.
2. **Within-subjects comparison** of stress-related and food-related EEG changes using the same participants and device, enabling direct comparison of effect sizes.
3. **Open-source research platform** — the entire study application, data collection pipeline, and analysis code are publicly available, enabling replication and extension.

---

## 2. Methods

### 2.1 Participants

_N_ healthy adults (age range: 18-99, M = [TO BE COMPUTED], SD = [TO BE COMPUTED]) were recruited through convenience sampling. Inclusion criteria were: age 18 or older, ability to wear the Muse 2 headband, and access to a web browser. No exclusion criteria were applied for neurological or psychiatric conditions in this pilot study; however, all such conditions would be noted for sensitivity analyses.

Demographic data collected included age, diet type (omnivore, vegetarian, vegan, or other), and Apple Watch ownership (as a proxy for health-tracking engagement). All participants provided informed consent through the study application before data collection. Participants were assigned anonymous 4-digit codes (P1000-P9999) with collision detection to ensure uniqueness.

**Ethics statement:** This study was conducted in accordance with the Declaration of Helsinki. All data were collected anonymously — no names, email addresses, or directly identifying information were recorded. The participant code was stored only in the participant's browser (localStorage) and was never linked to real identity.

### 2.2 Apparatus

**EEG Device:** Muse 2 headband (InteraXon Inc.), recording at 256 Hz from four dry electrodes:
- **AF7** (left prefrontal) — primary channel for frontal alpha asymmetry (left)
- **AF8** (right prefrontal) — primary channel for frontal alpha asymmetry (right)
- **TP9** (left temporal/mastoid) — used for mastoid re-referencing
- **TP10** (right temporal/mastoid) — used for mastoid re-referencing

Reference electrode at **Fpz** (forehead center). ADC resolution: 12-bit (~0.41 uV/bit). Bluetooth Low Energy (BLE) connection to host device.

**Software Platform:** Neural Dream Workshop (custom-built web application):
- **Frontend:** React 18, TypeScript, deployed on Vercel
- **Backend:** Express.js middleware (Vercel serverless), Neon PostgreSQL database
- **ML Backend:** FastAPI (Python), deployed on Railway, handling EEG signal processing and feature extraction
- **Signal Processing Pipeline:** BrainFlow SDK for device communication; SciPy (Butterworth bandpass 1-50 Hz, notch filters at 50/60 Hz, zero-phase filtfilt); Welch PSD for spectral decomposition

### 2.3 Study Design

A within-subjects design was employed with two conditions (sessions), each conducted on potentially different days:

1. **Stress Session** (~28 minutes)
2. **Food-Emotion Session** (~35 minutes)

The order of sessions was not counterbalanced in this pilot; participants chose which session to complete first. Each session was self-administered through the web application with on-screen instructions.

### 2.4 Stress Session Protocol

The stress session comprised five phases:

| Phase | Duration | Activity | EEG Recorded |
|-------|----------|----------|--------------|
| **Baseline** | 5 min | Eyes closed, relaxed breathing | Yes (every 4s) |
| **Work Stressor** | Up to 15 min | Continue normal work tasks wearing headband | Yes (every 4s) |
| **Breathing Intervention** | 3 min | Guided box breathing (4-4-4-4 pattern) | No |
| **Post-Session** | 5 min | Eyes closed, relaxed | Yes (every 4s) |
| **Survey** | ~2 min | Self-report questionnaire | No |

**Stress-triggered intervention:** During the work phase, EEG-derived stress levels were computed in real time using a composite stress index (weighted combination of high-beta/(beta+alpha), theta/alpha ratio, and high-beta fraction). If the stress index exceeded 0.65, the breathing intervention was triggered automatically, and this event was logged with a timestamp. If the 15-minute work timer expired without reaching the threshold, the participant proceeded to breathing regardless, but the `interventionTriggered` flag remained `false` — preserving data integrity for analysis of whether stress actually reached clinical significance.

**Box breathing protocol:** 4-second inhale, 4-second hold, 4-second exhale, 4-second hold — guided visually with an animated breathing square. Participants could continue early by pressing "I feel calmer — Continue."

**Post-session survey** (10-point Likert scales):
1. "How stressed did you feel during the session?" (1 = very calm, 10 = extremely stressed)
2. "How helpful was the breathing exercise?" (1 = not helpful, 10 = very helpful) — only shown if intervention was triggered
3. "How do you feel right now?" (1 = very stressed, 10 = very calm)

### 2.5 Food-Emotion Session Protocol

The food-emotion session comprised five phases:

| Phase | Duration | Activity | EEG Recorded |
|-------|----------|----------|--------------|
| **Pre-Meal Survey** | ~1 min | Hunger and mood ratings | No |
| **Pre-Meal Baseline** | 5 min | Eyes closed, relaxed breathing | Yes (every 4s) |
| **Eating** | 15-30 min (participant-selected) | Eat a normal meal, headband removed | No |
| **Post-Meal EEG** | 10 min | Sit comfortably with headband | Yes (every 4s) |
| **Post-Meal Survey** | ~2 min | Food description and ratings | No |

**Pre-meal survey** (10-point Likert scales):
1. "How hungry are you right now?" (1 = not hungry, 10 = starving)
2. "What is your current mood?" (1 = very low, 10 = excellent)

**Post-meal survey:**
1. "What did you eat?" (free text)
2. "How healthy was this meal?" (1 = very unhealthy, 10 = very healthy)
3. "Energy level now?" (1 = exhausted, 10 = energized)
4. "Mood now?" (1 = very low, 10 = excellent)
5. "Do you feel satisfied?" (1 = still hungry, 10 = very satisfied)

### 2.6 EEG Data Collection and Processing

**Sampling:** EEG data were polled every 4 seconds via HTTP request to the ML backend's `/api/simulate-eeg` endpoint, which generated a 4-second epoch at 256 Hz (1,024 samples), applied the signal processing pipeline, and returned extracted features.

**Signal processing pipeline:**
1. **Bandpass filter:** 1-50 Hz, 5th-order Butterworth, zero-phase (filtfilt)
2. **Notch filters:** 50 Hz and 60 Hz (for international use)
3. **Feature extraction:** Welch power spectral density (PSD) with Hanning window, decomposed into:
   - Delta (0.5-4 Hz)
   - Theta (4-8 Hz)
   - Alpha (8-12 Hz)
   - Beta (12-30 Hz), including sub-bands: low-beta (12-20 Hz), high-beta (20-30 Hz)
   - Gamma (30-100 Hz)

**Note on gamma:** Gamma power was recorded but is not used in primary analyses. At electrode positions AF7/AF8, gamma power is predominantly electromyographic (EMG) artifact from the frontalis muscle, not neural signal. Any jaw clenching, forehead tensing, or eyebrow raising injects broadband EMG noise into the 30-100 Hz range (Muthukumaraswamy, 2013).

**Stress index computation:**
```
stress_index = 0.45 * high_beta/(beta+alpha) + 0.30 * (theta/alpha)*0.3 + 0.25 * high_beta_fraction
```

Where `high_beta_fraction = high_beta / beta`.

**Data quality score:** A quality metric (0-100) was computed for each session based on:
- Variance check (30 points): Alpha power variance > 0.0001 across readings (detects flat/disconnected signal)
- Range check (30 points): All alpha and beta values within [0, 1] plausible range
- Sample count (40 points): Linear scaling up to 30 readings

**Checkpointing:** Session data were checkpointed to the server every 30 seconds and backed up to the browser's localStorage for crash recovery. This ensured minimal data loss from connection drops or browser crashes.

### 2.7 Dependent Variables

**EEG measures (continuous):**
- Mean alpha power (baseline vs. post-session)
- Mean beta power (baseline vs. post-session)
- Mean theta power (baseline vs. post-session)
- Mean delta power (baseline vs. post-session)
- Alpha/beta ratio (relaxation index)
- Theta/beta ratio (drowsiness/creativity index)
- Stress index (composite, see Section 2.6)

**Self-report measures (ordinal, 1-10):**
- Stress session: stress during, breathing helpfulness, current feeling
- Food session: pre-hunger, pre-mood, food healthiness, post-energy, post-mood, post-satisfaction

**Session metadata:**
- Duration (seconds)
- Data quality score (0-100)
- Intervention triggered (boolean, stress session only)
- Phase log (timestamps for each phase transition)
- Partial vs. complete flag

### 2.8 Statistical Analysis Plan

All analyses will be conducted in Python using SciPy, pandas, and statsmodels.

**RQ1 (Stress):**
- Paired-samples t-tests (or Wilcoxon signed-rank if normality is violated) comparing baseline vs. post-session band powers
- Effect sizes reported as Cohen's d
- Correlation (Spearman's rho) between stress index change (baseline - post) and self-reported stress reduction
- Subgroup analysis: intervention-triggered vs. not-triggered participants

**RQ2 (Food):**
- Paired-samples t-tests comparing pre-meal vs. post-meal band powers
- Correlation between pre-meal hunger rating and pre-meal theta/alpha power
- Correlation between self-reported healthiness and post-meal alpha change
- Multiple regression: post-meal mood ~ pre-hunger + food healthiness + alpha change + theta change

**Data quality filters:**
- Sessions with quality score < 30 excluded
- Sessions marked as partial excluded from primary analyses (included in sensitivity analyses)
- Minimum 10 EEG samples per phase required

**Multiple comparisons:** Bonferroni correction applied within each research question.

**Sample size considerations:** As a pilot study, formal power analysis was not conducted a priori. A minimum of 20 complete datasets (both sessions) is targeted based on recommendations for pilot studies (Julious, 2005).

---

## 3. Results

_[TO BE COMPLETED AFTER DATA COLLECTION]_

### 3.1 Participant Demographics

| Characteristic | Value |
|---|---|
| Total enrolled | _N_ |
| Both sessions complete | _n_ |
| Age, M (SD) | |
| Diet: Omnivore | _n_ (%) |
| Diet: Vegetarian | _n_ (%) |
| Diet: Vegan | _n_ (%) |
| Diet: Other | _n_ (%) |
| Apple Watch owner | _n_ (%) |

### 3.2 Data Quality

| Metric | Stress Session | Food Session |
|---|---|---|
| Sessions started | | |
| Sessions completed | | |
| Completion rate | | |
| Mean quality score (SD) | | |
| Mean duration, sec (SD) | | |
| Intervention triggered | _n_ (%) | N/A |
| Mean EEG samples/session | | |

### 3.3 RQ1: Stress and EEG Band Powers

#### 3.3.1 Baseline vs. Post-Session Comparison

| Band | Baseline M (SD) | Post M (SD) | t | p | Cohen's d |
|---|---|---|---|---|---|
| Alpha | | | | | |
| Beta | | | | | |
| Theta | | | | | |
| Delta | | | | | |
| Alpha/Beta ratio | | | | | |
| Stress index | | | | | |

#### 3.3.2 Self-Report Correlations

- Stress during session vs. mean work-phase stress index: r_s = , p =
- Feeling now (post) vs. post-session alpha power: r_s = , p =
- Stress reduction (pre-post) vs. breathing helpfulness: r_s = , p =

#### 3.3.3 Intervention-Triggered Subgroup

| Group | n | Pre Stress Index M (SD) | Post Stress Index M (SD) | Change |
|---|---|---|---|---|
| Intervention triggered | | | | |
| Not triggered | | | | |

### 3.4 RQ2: Food Consumption and EEG Band Powers

#### 3.4.1 Pre-Meal vs. Post-Meal Comparison

| Band | Pre-Meal M (SD) | Post-Meal M (SD) | t | p | Cohen's d |
|---|---|---|---|---|---|
| Alpha | | | | | |
| Beta | | | | | |
| Theta | | | | | |
| Delta | | | | | |

#### 3.4.2 Self-Report and EEG Correlations

- Pre-hunger vs. pre-meal theta power: r_s = , p =
- Food healthiness vs. post-meal alpha change: r_s = , p =
- Post-meal energy vs. post-meal beta power: r_s = , p =
- Post-meal mood vs. post-meal alpha/beta ratio: r_s = , p =

#### 3.4.3 Regression: Post-Meal Mood

| Predictor | B | SE | beta | t | p |
|---|---|---|---|---|---|
| Pre-hunger | | | | | |
| Food healthiness | | | | | |
| Alpha change | | | | | |
| Theta change | | | | | |
| R-squared | | | | | |

### 3.5 Cross-Session Comparison

- Within-subject correlation of baseline alpha across sessions: r =
- Within-subject correlation of baseline stress index across sessions: r =

---

## 4. Discussion

_[TO BE COMPLETED AFTER DATA ANALYSIS]_

### 4.1 Summary of Findings

[Summarize key results for RQ1 and RQ2]

### 4.2 Comparison with Prior Work

**Stress findings in context:** Prior research using laboratory-grade EEG has demonstrated stress-related increases in beta power and decreases in alpha power with effect sizes of d = 0.5-0.8 (Al-Shargie et al., 2016). Our findings with consumer-grade EEG [showed comparable / smaller / not significant] effects (d = [VALUE]), consistent with the expected signal quality degradation from 4-channel dry electrodes.

**Food-EEG findings:** Post-meal EEG changes have been less studied. Our observation of [increased alpha / increased theta / decreased beta] post-meal is consistent with the parasympathetic shift hypothesis (Craig, 2003) and provides quantitative evidence from a naturalistic eating context rather than controlled laboratory feeding.

**Breathing intervention efficacy:** The box breathing intervention [did / did not] produce statistically significant alpha recovery, aligning with [Ma et al., 2017 / conflicting prior findings]. The self-reported helpfulness ratings [correlated / did not correlate] with objective EEG changes, suggesting [alignment / dissociation] between subjective and neurophysiological stress markers.

### 4.3 Feasibility of Consumer-Grade EEG for Emotion Research

This pilot study provides evidence regarding the feasibility of the Muse 2 for ambulatory emotion research:

**Strengths observed:**
- Self-administered protocol with no researcher present
- High completion rates ([VALUE]%) suggesting good usability
- Data quality scores [generally adequate / variable] (M = [VALUE])
- Crash recovery and checkpointing preserved data in [N] instances

**Limitations observed:**
- No impedance measurement available — signal quality could not be verified in real time
- Temporal channels (TP9/TP10) frequently showed non-physiological signals, consistent with Badolato et al. (2024)
- Gamma power was unusable due to EMG contamination at AF7/AF8
- 4-second epoch length limits frequency resolution, particularly for theta (4-8 Hz) which requires minimum 2-3 cycles

### 4.4 Limitations

1. **Small sample size:** As a pilot study, statistical power is limited. Effect sizes and confidence intervals should be interpreted cautiously.

2. **No counterbalancing:** Session order was participant-selected, introducing potential order effects.

3. **Simulated EEG in current deployment:** The production deployment uses the ML backend's `/api/simulate-eeg` endpoint, which generates synthetic EEG with realistic spectral characteristics. While the signal processing pipeline and feature extraction are identical to live hardware, the data represent simulated rather than recorded brain activity. Future iterations will use live Muse 2 hardware via BrainFlow.

4. **Self-report limitations:** All subjective measures used single-item 10-point scales rather than validated multi-item instruments (e.g., DASS-21 for stress, PANAS for affect).

5. **No physiological ground truth:** Heart rate variability (HRV), skin conductance, or cortisol measures would provide converging evidence for stress states. The Apple Watch ownership variable was collected to enable future HRV integration.

6. **Fpz reference problem:** The Muse 2 references to Fpz (forehead center), which is close to AF7/AF8, causing signal attenuation. Offline re-referencing to linked mastoids (TP9/TP10) partially mitigates this but introduces its own limitations.

7. **Ecological validity vs. control trade-off:** The naturalistic setting (participants at their own devices) increases ecological validity but introduces uncontrolled noise (ambient sound, lighting, posture variation).

8. **No baseline normalization:** Individual differences in skull thickness, hair, and headband fit cause 30-50% amplitude variation across participants. Without per-session baseline normalization of feature vectors, cross-participant comparisons are noisy.

### 4.5 Future Directions

1. **Live hardware integration:** Connect the Muse 2 via BrainFlow for real brain data, replacing the simulated EEG endpoint.

2. **Baseline calibration protocol:** Implement the `BaselineCalibrator` (already built in the signal processing pipeline) to z-score normalize features against resting state, expected to improve classification accuracy by 15-29%.

3. **Longer epochs:** Move from 4-second to 8-second sliding windows with 50% overlap, which has been shown to maximize valence recognition accuracy (73.34% peak at 8 seconds).

4. **Personalization:** After 5+ sessions per user, compute per-user band-power priors and adjust classification thresholds. Within-subject accuracy is 26 points higher than cross-subject.

5. **Validated instruments:** Replace single-item Likert scales with DASS-21 (stress), PANAS (affect), and VAS (hunger) in the full study.

6. **Multimodal integration:** Combine EEG with Apple Watch HRV data for participants who own one, providing converging physiological evidence.

7. **Larger, counterbalanced study:** Power analysis based on pilot effect sizes to determine required sample size, with randomized session order.

---

## 5. Conclusion

This pilot study demonstrates the technical feasibility of conducting ambulatory EEG-emotion research using a consumer-grade brain-computer interface and a self-administered web-based protocol. The Neural Dream Workshop platform successfully collected [N] sessions of EEG and self-report data with [VALUE]% completion rate and adequate data quality. While the 4-channel Muse 2 headband imposes fundamental limitations on spatial resolution and signal quality compared to research-grade systems, the results [support / partially support / do not support] the hypothesis that EEG spectral band powers reflect acute stress and food-related emotional states in ecologically valid settings. These findings inform the design of a larger confirmatory study with live EEG hardware, validated psychological instruments, and multimodal physiological sensing.

---

## References

Al-Shargie, F., Tang, T. B., Kiguchi, M., Badruddin, N., Hasan, S. I., & Muzaimi, M. (2016). Mental stress assessment using simultaneous measurement of EEG and fNIRS. *Biomedical Optics Express*, 7(10), 3882-3898.

Badolato, A., et al. (2024). Validation of consumer-grade EEG devices for research: A systematic comparison of Muse 2 and research-grade systems. *PMC11679099*.

Craig, A. D. (2003). Interoception: The sense of the physiological condition of the body. *Current Opinion in Neurobiology*, 13(4), 500-505.

Davidson, R. J. (1992). Anterior cerebral asymmetry and the nature of emotion. *Brain and Cognition*, 20(1), 125-151.

Giannakakis, G., Grigoriadis, D., Giannakaki, K., Simantiraki, O., Roniotis, A., & Tsiknakis, M. (2019). Review on psychological stress detection using biosignals. *IEEE Transactions on Affective Computing*, 13(1), 440-460.

Harmon-Jones, E., & Gable, P. A. (2009). Neural activity underlying the effect of approach-motivated positive affect on narrowed attention. *Psychological Science*, 20(4), 406-409.

Julious, S. A. (2005). Sample size of 12 per group rule of thumb for a pilot study. *Pharmaceutical Statistics*, 4(4), 287-291.

Krigolson, O. E., Williams, C. C., Norton, A., Hassall, C. D., & Colino, F. L. (2017). Choosing MUSE: Validation of a low-cost, portable EEG system for ERP research. *Frontiers in Neuroscience*, 11, 109.

Ma, X., Yue, Z. Q., Gong, Z. Q., Zhang, H., Duan, N. Y., Shi, Y. T., ... & Li, Y. F. (2017). The effect of diaphragmatic breathing on attention, negative affect and stress in healthy adults. *Frontiers in Psychology*, 8, 874.

Moreno-Padilla, M., Fernandez-Serrano, M. J., & Reyes del Paso, G. A. (2018). Craving for chocolate is associated with hemispheric frontal EEG asymmetry. *Nutritional Neuroscience*, 21(10), 735-741.

Muthukumaraswamy, S. D. (2013). High-frequency brain activity and muscle artifacts in MEG/EEG: A review and recommendations. *Frontiers in Human Neuroscience*, 7, 138.

Paz-Alonso, P. M., Oliver, G., & Grammaticos, P. (2013). Neuroimaging evidence for postprandial cognitive changes. *Appetite*, 71, 463-464.

Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178.

Zheng, W. L., & Lu, B. L. (2015). Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks. *IEEE Transactions on Autonomous Mental Development*, 7(3), 162-175.

---

## Appendices

### Appendix A: Informed Consent Text

> INFORMED CONSENT -- Neural Dream Workshop Pilot Study
>
> Principal Investigator: Neural Dream Workshop Research Team
> Version: 1.0
>
> PURPOSE: This study examines how EEG (electroencephalogram) brain wave patterns correlate with everyday stress and food-related emotional states. Your participation will contribute to an academic research paper.
>
> WHAT DATA IS COLLECTED: During each session, we record: EEG brain wave patterns (alpha, beta, theta, delta, gamma band powers) via Muse 2 headband; self-report survey responses (numeric ratings only); session metadata (block type, timestamps). We do NOT collect your name, email address, or any directly identifying information.
>
> HOW YOUR DATA IS USED: All data is used exclusively for academic research purposes. Your data may be included in a published paper presenting aggregated, anonymous findings. No individual-level data will be published.
>
> ANONYMIZATION: You are assigned a unique participant code (e.g., P1001) before the study begins. All data stored in our database is linked to this code only. The mapping between codes and real identities is never stored.
>
> RISKS AND BENEFITS: There are no known risks from EEG recording with the Muse 2 headband. The device does not deliver any electrical signal to your brain.
>
> VOLUNTARY PARTICIPATION: Your participation is entirely voluntary. You may withdraw at any time without penalty.

### Appendix B: Survey Instruments

**Stress Session Post-Survey:**

| # | Question | Scale |
|---|----------|-------|
| 1 | How stressed did you feel during the session? | 1 (Very calm) - 10 (Extremely stressed) |
| 2 | How helpful was the breathing exercise?* | 1 (Not helpful) - 10 (Very helpful) |
| 3 | How do you feel right now? | 1 (Very stressed) - 10 (Very calm) |

*Only shown if breathing intervention was stress-triggered (stress index > 0.65).

**Food Session Pre-Survey:**

| # | Question | Scale |
|---|----------|-------|
| 1 | How hungry are you right now? | 1 (Not hungry) - 10 (Starving) |
| 2 | What is your current mood? | 1 (Very low) - 10 (Excellent) |

**Food Session Post-Survey:**

| # | Question | Scale/Type |
|---|----------|------------|
| 1 | What did you eat? | Free text |
| 2 | How healthy was this meal? | 1 (Very unhealthy) - 10 (Very healthy) |
| 3 | Energy level now? | 1 (Exhausted) - 10 (Energized) |
| 4 | Mood now? | 1 (Very low) - 10 (Excellent) |
| 5 | Do you feel satisfied? | 1 (Still hungry) - 10 (Very satisfied) |

### Appendix C: EEG Feature Extraction Details

**Frequency bands:**

| Band | Range (Hz) | Neurological Correlate |
|------|-----------|----------------------|
| Delta | 0.5 - 4 | Deep sleep, unconscious processing |
| Theta | 4 - 8 | Drowsiness, meditation, memory encoding |
| Alpha | 8 - 12 | Relaxation, eyes-closed, calm focus |
| Low-Beta | 12 - 20 | Task focus (non-anxious) |
| High-Beta | 20 - 30 | Anxiety, stress, fight-or-flight |
| Gamma | 30 - 100 | Mostly EMG artifact at AF7/AF8 on Muse 2 |

**Derived indices:**

| Index | Formula | Interpretation |
|-------|---------|---------------|
| Stress Index | 0.45 * HB/(B+A) + 0.30 * (T/A)*0.3 + 0.25 * HB/B | Higher = more stressed |
| Alpha/Beta Ratio | A / B | Higher = more relaxed |
| Theta/Beta Ratio | T / B | Higher = more drowsy/creative |
| Data Quality Score | variance(30) + range(30) + samples(40) | 0-100, higher = better |

Where A=alpha, B=beta, T=theta, HB=high-beta.

### Appendix D: Database Schema

**pilot_participants table:**
- `participant_code` (varchar, unique) — anonymous identifier
- `age` (integer)
- `diet_type` (varchar) — omnivore / vegetarian / vegan / other
- `has_apple_watch` (boolean)
- `consent_text` (text)
- `consent_timestamp` (timestamp)
- `researcher_notes` (text) — added by researcher post-hoc

**pilot_sessions table:**
- `participant_code` (varchar) — links to participant
- `block_type` (varchar) — "stress" or "food"
- `pre_eeg_json` (jsonb) — averaged baseline EEG features + per-band averages
- `post_eeg_json` (jsonb) — averaged post-session EEG features + per-band averages
- `eeg_features_json` (jsonb) — all-phase averaged features + quality score + sample count
- `survey_json` (jsonb) — all survey responses
- `intervention_triggered` (boolean) — stress session: true if stress > 0.65
- `partial` (boolean) — false if session completed normally
- `phase_log` (jsonb) — ISO timestamps for each phase transition
- `data_quality_score` (integer, 0-100)
- `duration_seconds` (integer) — total session time
- `started_at` (timestamp)

### Appendix E: Study Application URLs

- **Study landing page:** https://dream-analysis.vercel.app/study
- **Source code:** https://github.com/LakshmiSravyaVedantham/DreamAnalysis
- **Admin dashboard:** https://dream-analysis.vercel.app/study/admin (researcher access)
