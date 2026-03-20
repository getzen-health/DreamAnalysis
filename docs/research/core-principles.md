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

<!-- Principles will be appended below by the research agent -->
