# Core Principles — NeuralDreamWorkshop

> Auto-maintained by the research agent. Each principle is distilled from research cycles.
> Format: Principle → Evidence → Implication for our app.

---

## Last updated: 2026-03-19

---

### 1. Consistent Motion Language Across All Pages

**Principle:** Every card and data visualization must use the same entry animation system (staggered fade-up via `cardVariants`) and animated fills for gauges/bars. Static elements in an otherwise animated app feel broken, not minimal.

**Evidence:** Sleep page used plain `<div>` for cards while mood/stress/focus pages all used `motion.div` with `cardVariants`. The sleep stages bar was the only stacked bar in the app without animated segment growth. Users perceive inconsistent animation as lower quality even if the data is identical.

**Implication:** When adding any new card or data visualization, always wrap in `motion.div` with `cardVariants` (staggered `custom={n}`) and animate any fill/bar/gauge from zero to final value. Use the shared `animations.ts` variants -- never inline one-off timing.

### 2. Every Health-Adjacent Endpoint Must Include a Disclaimer

**Principle:** Any API endpoint that suggests, recommends, or scores a health-related action (breathing patterns, music therapy, attention screening, stress levels, sleep quality) must return a `disclaimer` field in its response. Endpoints must never use clinical language ("prescribe", "diagnose", "treat") in user-facing docstrings without qualifying it as a wellness tool.

**Evidence:** FDA classifies software that makes clinical claims as SaMD (Software as a Medical Device). The app's seizure detector is functionally Class II, and attention screening could be classified as SaMD if marketed for ADHD. Other modules (perinatal, psychedelic, voice biomarkers, cognitive) already had disclaimers, but breathing and music therapy did not -- creating inconsistent regulatory posture. The term "prescribe" is regulated in the US and implies clinical authority.

**Implication:** When adding any new health-related endpoint: (1) include a `_WELLNESS_DISCLAIMER` or `_CLINICAL_DISCLAIMER` constant in the route module, (2) return it in every response body, (3) use "recommend" or "suggest" instead of "prescribe" in docstrings and user-facing text. Never claim diagnostic accuracy without citing the specific validation study and its limitations.

<!-- Principles will be appended below by the research agent -->
