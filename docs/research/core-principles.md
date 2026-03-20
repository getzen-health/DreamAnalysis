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

<!-- Principles will be appended below by the research agent -->
