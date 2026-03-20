# Research Learnings — NeuralDreamWorkshop

> Auto-maintained by the research agent. Each entry is a specific finding from a research cycle.
> Format: Date | Topic | Finding | Source | Action Taken

---

## Last updated: 2026-03-19

| Date | Topic | Finding | Action |
|---|---|---|---|
| 2026-03-19 | ux-polish | Sleep page SleepDataTab used static divs with no entry animations or animated fills, unlike mood/stress/focus pages which all use motion.div + cardVariants + animated gauges/bars. Inconsistent motion design breaks the premium feel. | Added staggered cardVariants entry animations to all three sleep data cards and animated the sleep stages stacked bar segments (width 0 -> final width with staggered delay). Matches the animation language used across mood-trends, stress-trends, and focus-trends. |
| 2026-03-19 | medical-analytics | Breathing (`/breathing/prescribe`) and music therapy (`/music-therapy/prescribe`) endpoints used the regulated term "prescribe" without any wellness disclaimer, while other clinical-adjacent modules (seizure, perinatal, psychedelic, voice biomarkers, cognitive) all included disclaimers. FDA SaMD risk: seizure detector is Class II, attention screening could be SaMD if marketed for ADHD. PHQ-9/GAD-7 already in clinical_bridge.py but needs clear distinction from EEG-derived mood estimates. | Added `_WELLNESS_DISCLAIMER` to breathing and music therapy route responses. Updated docstrings to say "recommend" instead of "prescribe." Created issue #463 for full medical claims audit covering FDA SaMD, HIPAA, evidence grading, and remaining endpoint disclaimers. |
| 2026-03-19 | eeg-brain | Gamma band upper bound was 100 Hz in BANDS dict, `get_personalized_bands()`, and `meditation_depth.py`, but the preprocessing bandpass filter cuts at 50 Hz. Welch PSD above 50 Hz is near-zero, so integrating 30-100 Hz artificially diluted gamma power estimates. Other modules (graph_emotion, cnn_kan, dsp_mcf, etc.) already used 45 or 50 Hz caps independently. | Capped gamma to (30, 50) Hz in `eeg_processor.py` BANDS, `get_personalized_bands()`, and `meditation_depth.py`. Updated test assertion in `test_dual_epoch_iaf.py`. All 22 IAF tests pass. |
<!-- Entries will be appended below by the research agent -->
