# Core Principles — NeuralDreamWorkshop

> Auto-maintained by the research agent. Each principle is distilled from research cycles.
> Format: Principle → Evidence → Implication for our app.

---

## Last updated: 2026-03-20

---

### 1. Consistent Motion Language Across All Pages

**Principle:** Every card and data visualization must use the same entry animation system (staggered fade-up via `cardVariants`) and animated fills for gauges/bars. Static elements in an otherwise animated app feel broken, not minimal.

**Evidence:** Sleep page used plain `<div>` for cards while mood/stress/focus pages all used `motion.div` with `cardVariants`. The sleep stages bar was the only stacked bar in the app without animated segment growth. Users perceive inconsistent animation as lower quality even if the data is identical.

**Implication:** When adding any new card or data visualization, always wrap in `motion.div` with `cardVariants` (staggered `custom={n}`) and animate any fill/bar/gauge from zero to final value. Use the shared `animations.ts` variants -- never inline one-off timing.

### 2. Every Health-Adjacent Endpoint Must Include a Disclaimer

**Principle:** Any API endpoint that suggests, recommends, or scores a health-related action (breathing patterns, music therapy, attention screening, stress levels, sleep quality) must return a `disclaimer` field in its response. Endpoints must never use clinical language ("prescribe", "diagnose", "treat") in user-facing docstrings without qualifying it as a wellness tool.

**Evidence:** FDA classifies software that makes clinical claims as SaMD (Software as a Medical Device). The app's seizure detector is functionally Class II, and attention screening could be classified as SaMD if marketed for ADHD. Other modules (perinatal, psychedelic, voice biomarkers, cognitive) already had disclaimers, but breathing and music therapy did not -- creating inconsistent regulatory posture. The term "prescribe" is regulated in the US and implies clinical authority.

**Implication:** When adding any new health-related endpoint: (1) include a `_WELLNESS_DISCLAIMER` or `_CLINICAL_DISCLAIMER` constant in the route module, (2) return it in every response body, (3) use "recommend" or "suggest" instead of "prescribe" in docstrings and user-facing text. Never claim diagnostic accuracy without citing the specific validation study and its limitations.

### 3. Match Regulation Strategy Type to Arousal Level

**Principle:** Emotion regulation suggestions must be typed by strategy category (cognitive reappraisal, acceptance, distraction, physiological) and the system must prefer distraction/acceptance over reappraisal under high-arousal negative states.

**Evidence:** Gross (2015) process model identifies distinct regulation families with different cognitive costs. Sheppes et al. (2011) demonstrated that cognitive reappraisal requires prefrontal resources that are depleted under high arousal -- making it the wrong strategy when someone is highly anxious or angry. Distraction (attentional deployment) and acceptance (mindfulness-based observation) are more effective under high arousal because they require less top-down cognitive control. Webb et al. (2012) meta-analysis confirmed differential effectiveness across strategy types and arousal levels.

**Implication:** When adding regulation suggestions to any module: (1) label each strategy with its type from the evidence-based taxonomy, (2) order strategies so that high-arousal states lead with distraction/acceptance, (3) reserve cognitive reappraisal for moderate-arousal or low-arousal states where the user has cognitive bandwidth to reframe. Never present a single strategy type for all emotional states.

### 4. One Daily Number Drives Retention — Not Dashboards

**Principle:** A single composite daily score (0-100) that users check every morning is the strongest retention mechanism in consumer health apps. Dashboards with multiple metrics inform but do not create daily habits. The score must be actionable ("expect lower focus today") and occasionally surprising ("your stress recovered faster than usual").

**Evidence:** Oura Ring's Readiness Score (0-100) and Whoop's Recovery Score (green/yellow/red) are the primary retention drivers for their respective platforms. Oura reports ~65% 12-month retention (aided by hardware lock-in, but daily score is the behavioral hook). Headspace with streaks achieves ~30% 12-month retention; without streaks, ~15%. Calm and Headspace — pure software with content but no objective daily score — retain far worse than Oura/Whoop despite larger user bases. The pattern: **objective daily measurement + one actionable number > content library + streaks**.

**Implication:** NDW must synthesize its 16 ML models, sleep staging, HRV, and voice biomarkers into a single Daily Wellness Score (issue #464). The score should be the first thing on the Home page, available via push notification on mobile, and rendered as a shareable image card for social growth. Without this, NDW remains a "dashboard app" — interesting to explore but not habit-forming.

### 5. Loading States Must Mirror Content Layout (Skeleton-First)

**Principle:** Every page loading state must use skeleton placeholders that match the real content layout -- same grid, same card sizes, same spatial hierarchy. A bare spinner on an empty page signals "nothing here yet" and creates layout shift when content arrives. A skeleton signals "your data is coming" and trains the eye to where content will appear.

**Evidence:** The scores dashboard (main health overview) used a centered spinner while dashboard, settings, session-history, and daily-brain-report all used layout-matching skeletons. Oura and Whoop both show shimmer layouts that preview the real content structure during loading. Nielsen Norman Group research on perceived performance shows skeleton screens reduce perceived wait time by 10-20% compared to spinners because they give users spatial context and a sense of progress.

**Implication:** When adding any new page: (1) build the skeleton loading state FIRST, matching the real layout's grid, card count, and section hierarchy, (2) use the existing `Skeleton` component from shadcn/ui, (3) never use a bare spinner as the only loading indicator for a full-page load. Reserve spinners for inline actions (button presses, small data fetches within an already-rendered page).

### 6. Never Map Unvalidated Scores onto Clinical Instrument Scales

**Principle:** When a model produces risk estimates from non-standard inputs (voice, EEG, accelerometry), it must never output scores on a validated clinical instrument's scale (PHQ-9 0-27, GAD-7 0-21, etc.) without explicit, unmissable warnings that the score is not a validated result from that instrument. Borrowing clinical scales creates false authority that can lead to self-diagnosis or treatment decisions.

**Evidence:** The voice depression screener mapped acoustic features onto PHQ-9 (0-27) and GAD-7 (0-21) scales, producing outputs like `phq9_score: 18, severity: moderately_severe`. Meanwhile, the same app has a separate module (`mental_health_questionnaire.py`) that implements the actual validated PHQ-9/GAD-7 self-report instruments. A user seeing both would reasonably assume they are equivalent. The voice-derived scores were never validated against actual PHQ-9/GAD-7 ground truth. The ADHD detector similarly classifies users into DSM-5 subtypes ("inattentive", "hyperactive", "combined") from EEG heuristics alone.

**Implication:** When any model produces risk scores: (1) use a custom scale (e.g., 0-100 "risk index") rather than mapping onto a clinical instrument's scale, OR (2) if using a clinical scale for readability, include `not_validated: true`, a `scale_context` field, and a disclaimer that explicitly names the instrument and says "this is NOT a validated [instrument] result." (3) Never label output categories with DSM/ICD diagnostic subtypes — use descriptive terms like "theta-dominant pattern" instead of "inattentive ADHD profile."

### 7. Sanitize NaN at the Pipeline Entry Point, Not at Each Consumer

**Principle:** In a real-time sensor pipeline, NaN/inf values must be intercepted and repaired at the earliest possible point (before filtering), not individually guarded in every downstream consumer. A single NaN in the input to `filtfilt` poisons the entire output array, and once NaN enters an EMA accumulator it stays NaN forever -- one bad sample can corrupt the rest of the session.

**Evidence:** `preprocess()` in `eeg_processor.py` had zero NaN handling. BrainFlow Bluetooth packet drops inject NaN samples at a 1-5% rate on native BT connections. `scipy.signal.filtfilt` propagates NaN across the entire output, making every band power, FAA, emotion probability, and index NaN. The LGBM paths had partial `np.isfinite` guards but the feature-based heuristic path (the actual live Muse 2 path) had none. The EMA smoothing in `_predict_features()` has no NaN check, so a single corrupted epoch permanently poisons `_ema_probs`, `_ema_valence`, `_ema_stress`, etc.

**Implication:** When building any real-time sensor pipeline: (1) add NaN/inf sanitization at the very first processing step before any filter or transform, (2) use linear interpolation for short gaps (standard in EEG for BT drops), zeros for total disconnection, (3) never assume upstream data is clean -- Bluetooth, USB, and WiFi all produce sporadic NaN/dropout, (4) add `np.isfinite` guards on EMA accumulators as a defense-in-depth measure even after input sanitization.

### 8. Emotion Labels Must Include Plain-Language Explanation and Source Attribution

**Principle:** Whenever the app shows an emotion classification to the user -- whether from voice, EEG, or self-report -- the label must be accompanied by (1) a plain-language explanation of what signals contributed to the classification, (2) a brief insight phrased as an observation rather than a diagnosis, and (3) a disclaimer clarifying the analysis type (acoustic, neural, self-reported) and its non-clinical nature.

**Evidence:** The voice check-in card displayed a bare emotion label ("fear") with a confidence percentage and no context. Users have no way to know what "fear 72% confidence" means -- is the system saying they are afraid? That their voice sounds like fear? That 72% of something matched? Scherer (2003) and Juslin & Laukka (2003) established that each basic emotion produces distinct vocal profiles (e.g., sadness = lower pitch + slower rate, anger = higher energy + faster rate) that can be described accessibly. Without explanation, emotion labels risk being perceived as authoritative judgments rather than pattern observations. This is especially problematic for negatively-valenced labels (fear, anger, sad) which can cause alarm.

**Implication:** When presenting any emotion classification: (1) include a brief plain-language description of the signals (vocal patterns, brain wave patterns, or self-report) that contributed, (2) phrase insights as observations ("your voice sounds...") not assertions ("you are feeling..."), (3) always include a non-clinical disclaimer, (4) for voice-based detection, cite the acoustic features (pitch, rate, energy, spectral tilt). For EEG, cite the neural markers (FAA, alpha/beta ratio). Never show a bare emotion label without context.

<!-- Principles will be appended below by the research agent -->
