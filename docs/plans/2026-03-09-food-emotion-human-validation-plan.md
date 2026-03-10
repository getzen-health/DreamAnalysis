# Food-Emotion Human Validation Plan

**Goal:** Decide whether the current food-emotion module is ready for human testing, define the exact data required, estimate sample size, and lay out the next execution steps for a defensible pilot-to-validation pathway.

**Status:** Ready for a human **feasibility / pilot** study. Not ready for a strong **validation** claim yet.

**Why this status is correct:**
- The current backend route uses a **feature-based biomarker model** rather than a human-labeled, cross-subject validated classifier.
- The current implementation maps four EEG biomarkers to six food states using rule-weighted profiles:
  - `FAA` = frontal alpha asymmetry
  - `high_beta` = 20-30 Hz stress / craving proxy
  - `theta` = prefrontal self-regulation proxy
  - `delta` = satiety / low-arousal proxy
- The active model is implemented in:
  - `ml/models/food_emotion_predictor.py`
  - `ml/api/routes/food_emotion.py`
- The repository itself already acknowledges the validation gap: the food-emotion module is theory-backed but still needs human validation data.

---

## Decision

### What the module is ready for
- Feasibility testing
- Human pilot data collection
- Label quality assessment
- Signal quality assessment
- Within-person effect-size estimation
- Protocol refinement

### What the module is **not** ready for
- Strong accuracy claims
- Clinical or behavioral screening claims
- Publication as a validated classifier without new human data
- Public positioning as a proven 6-class food-state model

---

## Current Technical Reality

### Current food-state outputs
The system predicts six states:
- `craving_carbs`
- `appetite_suppressed`
- `comfort_seeking`
- `balanced`
- `stress_eating`
- `mindful_eating`

### Current model type
- **Model class:** feature-based heuristic scoring with softmax normalization
- **Inputs:** EEG-derived biomarkers
- **Calibration:** resting-state baseline supported
- **Output:** per-state probabilities, confidence, and recommendations

### Main limitation
The current model logic is **scientifically motivated** but not yet **empirically fit or validated** on a dedicated human food-emotion dataset.

---

## What Data Is Needed

The unit of analysis should be a **meal episode**, not just a participant.

Each meal episode should produce one structured record containing the following:

### 1. EEG data
- Pre-meal resting EEG
- Optional food-cue EEG block
- Post-meal EEG
- Raw or minimally processed EEG windows, not only final summary scores
- Signal quality metrics
- Artifact flags
- Device metadata

### 2. Ground-truth label data
- Hunger rating before meal
- Craving intensity before meal
- Craving type
- Stress rating before meal
- Mood / valence before meal
- Eating motive:
  - physical hunger
  - emotional eating
  - habit / convenience
  - reward-seeking / craving
- Eating style:
  - mindful
  - rushed
  - emotional
- Post-meal fullness
- Post-meal satisfaction
- Post-meal mood

### 3. Meal metadata
- Timestamp
- Meal type
- Free-text meal description
- Meal photo if available
- Calories
- Carbs / protein / fat
- Glycemic-load proxy or simple carb density
- Caffeine near meal
- Alcohol near meal
- Supplements taken that day:
  - name
  - dosage
  - time taken
- Medications taken that day:
  - name
  - dosage
  - time taken

### 4. Context covariates
- Sleep duration previous night
- Time of day
- Exercise that day
- Major stressor yes/no
- Medication / appetite-affecting factors if collected

---

## Which Data Answers Which Question

### Question 1: Can EEG detect a food-related state at all?
Needed:
- Pre-meal EEG
- Hunger
- craving intensity
- stress
- mood
- eating motive label

This determines whether the biomarker stack separates any interpretable pre-meal state.

### Question 2: Can EEG distinguish stress-eating from normal hunger?
Needed:
- Pre-meal EEG
- Stress score
- Hunger score
- Eating motive
- Craving score

This is one of the strongest and most testable sub-questions.

### Question 3: Does the meal shift the brain state?
Needed:
- Pre-meal EEG
- Post-meal EEG
- Fullness
- Satisfaction
- Mood after meal

This answers whether the system captures state transition, not just static classification.

### Question 4: Do food and daytime EEG predict later mood or dreams?
Needed:
- Meal metadata
- Daytime EEG
- Evening ratings
- Next-morning dream data

This is the bridge from the food module into the project's larger day-night research program.

---

## Recommended Human Protocol

Humans should **not** be asked to keep the EEG headset on for a full meal in the main protocol. The current app already reflects this practically.

### Recommended per-meal protocol
1. Pre-meal self-report
   - hunger
   - stress
   - mood
   - craving intensity
   - expected meal type
2. Pre-meal resting EEG
   - 2 minutes
3. Optional cue block
   - 60 to 90 seconds
   - meal visual / anticipation period
4. Remove headset
   - participant eats normally
5. Post-meal EEG
   - 2 to 3 minutes
6. Post-meal self-report
   - fullness
   - satisfaction
   - mood
   - whether the meal felt emotional / mindful / rushed

### Why shorten the protocol
The current study-session implementation includes longer intervals:
- 5-minute baseline
- meal break
- 10-minute post-meal EEG

That is workable for a controlled pilot, but likely too burdensome for repeated real-world use. For a human pilot focused on compliance and usable data, the shorter protocol is more realistic.

### Target wear time
- Practical target per meal: **4 to 6 minutes of EEG total**

That is much more likely to retain participants than 15+ minutes per meal.

---

## Sample Size Recommendation

### Feasibility pilot
- **Participants:** 10 to 15
- **Meal episodes per participant:** 3 to 5
- **Purpose:** completion rate, usability, signal quality, label consistency

### Pilot labeling study
- **Participants:** 20 to 30
- **Meal episodes per participant:** 5 to 10
- **Purpose:** estimate separability of states and refine the label taxonomy

### Initial validation study
- **Participants:** 30 to 50
- **Meal episodes per participant:** 8 to 12 usable episodes
- **Purpose:** grouped subject-held-out model evaluation

### Best first target for this repository
- **25 to 30 participants**
- **8 to 10 meal episodes each**
- target **200 to 300 usable meal episodes**

This is a better first validation dataset than a larger number of one-off participants, because the food-emotion problem is highly variable within person.

---

## Analysis Plan

### Phase 1: Feasibility analysis
- completion rate
- usable EEG percentage
- artifact burden
- per-step dropout
- average session duration
- label completion quality

### Phase 2: Exploratory signal analysis
- compare biomarker distributions across:
  - high hunger vs low hunger
  - emotional eating vs hunger-driven eating
  - mindful vs rushed eating
  - high craving vs low craving
- use within-subject plots first

### Phase 3: Supervised modeling
- train on meal episodes with participant grouping
- evaluate with:
  - LOSO-CV or grouped cross-validation
  - macro F1
  - balanced accuracy
  - calibration error
- compare:
  - EEG only
  - self-report only
  - EEG + self-report
  - EEG + meal metadata

### Phase 4: Label simplification if needed
If 6-class classification is weak, collapse to stronger tasks:
- `emotional eating` vs `non-emotional eating`
- `high craving` vs `low craving`
- `stress-linked eating` vs `hunger-linked eating`

This is likely the most pragmatic path if early data are sparse or noisy.

---

## Success Criteria

### Minimum pilot success
- at least 80% session completion
- at least 70% usable EEG windows
- consistent self-report labels
- visible separation in at least one key contrast:
  - stress-eating vs balanced
  - high craving vs low craving

### Minimum validation readiness
- 200+ usable meal episodes
- grouped cross-validation completed
- effect sizes and confidence intervals computed
- clear documentation of class balance
- evidence that the 6-state taxonomy is learnable or should be simplified

---

## Supplement & Medication Tracking (Issue #206)

Extend each meal episode self-report to include:

### Per-session supplement form
```
Supplements taken today:
  [ ] Name: ____________  Dosage: ____  Time taken: ____
  [ ] Name: ____________  Dosage: ____  Time taken: ____

Medications today:
  [ ] Name: ____________  Dosage: ____  Time taken: ____

Caffeine today: _____ mg (or cups)
Alcohol today:  _____ units
```

### Analysis additions
- Correlate supplement timing with pre/post-meal EEG biomarker shifts (FAA, high-beta, theta)
- Track cumulative supplement effects across the 14-day pilot window
- Use `POST /supplements/brain-state` to auto-log EEG context at supplement ingestion time

### Backend integration
- `ml/models/supplement_tracker.py` — already implemented
- `ml/api/routes/supplement_tracker.py` — 7 endpoints live
- Link supplement log entries to meal episode IDs for joint analysis

---

## Immediate Next Steps

1. Freeze the pilot protocol
2. Finalize the label schema before data collection
3. Include supplement and medication entries (see section above) in the evening / meal-adjacent form
4. Shorten the wearable portion to 4-6 minutes total per meal
5. Ensure raw or windowed EEG plus labels are stored per episode
6. Run a 10-15 participant feasibility pilot first
7. Review label balance, ingest completeness, and artifact rate
8. Then expand to 25-30 participants for the first real validation dataset

---

## Bottom Line

The food-emotion module is ready for **human pilot data collection now**, but it is **not yet validated**. The correct next move is to treat the current system as a theory-backed prototype, collect repeated meal episodes, and only then claim model performance.
