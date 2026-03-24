/**
 * Premium animation variants for Framer Motion.
 *
 * Design system: Apple Health + Oura + Calm + How We Feel
 *
 * Spring configs:
 *   snappy  — buttons, taps (stiffness: 400, damping: 25)
 *   smooth  — page transitions (stiffness: 200, damping: 30)
 *   bouncy  — celebrations, mood selection (stiffness: 300, damping: 15)
 *   gentle  — ambient, meditative (stiffness: 100, damping: 20)
 *
 * Easing:
 *   premium — cubic-bezier(0.22, 1, 0.36, 1) — Apple's ease-out
 *   material — cubic-bezier(0.4, 0, 0.2, 1) — Material Design
 */

// ── Spring Configs ──────────────────────────────────────────────────────

export const springs = {
  snappy: { type: "spring" as const, stiffness: 400, damping: 25 },
  smooth: { type: "spring" as const, stiffness: 200, damping: 30 },
  bouncy: { type: "spring" as const, stiffness: 300, damping: 15 },
  gentle: { type: "spring" as const, stiffness: 100, damping: 20 },
};

// ── Easing Curves ───────────────────────────────────────────────────────

export const easings = {
  premium: [0.22, 1, 0.36, 1] as const,
  material: [0.4, 0, 0.2, 1] as const,
  smoothOut: [0, 0, 0.2, 1] as const,
  springOut: [0.34, 1.56, 0.64, 1] as const,
};

// ── Page Transitions ────────────────────────────────────────────────────

export const pageTransition = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -8 },
  transition: { duration: 0.4, ease: easings.premium },
};

// ── Card Variants (staggered entry) ─────────────────────────────────────

export const cardVariants = {
  hidden: { opacity: 0, y: 16, scale: 0.97 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      delay: i * 0.08,
      duration: 0.5,
      ease: easings.premium,
    },
  }),
};

// ── Stagger Container ───────────────────────────────────────────────────

export const staggerContainer = {
  hidden: {},
  show: {
    transition: {
      staggerChildren: 0.06,
      delayChildren: 0.1,
    },
  },
};

export const staggerChild = {
  hidden: { opacity: 0, y: 20 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.4, ease: easings.premium },
  },
};

// ── List Item Variants ──────────────────────────────────────────────────

export const listItemVariants = {
  hidden: { opacity: 0, x: -12 },
  visible: (i: number) => ({
    opacity: 1,
    x: 0,
    transition: {
      delay: i * 0.05,
      duration: 0.35,
      ease: easings.premium,
    },
  }),
};

// ── Scale Variants ──────────────────────────────────────────────────────

export const scaleIn = {
  initial: { opacity: 0, scale: 0.85 },
  animate: { opacity: 1, scale: 1 },
  transition: { duration: 0.4, ease: easings.premium },
};

export const springPop = {
  initial: { scale: 0 },
  animate: { scale: 1 },
  transition: springs.bouncy,
};

// ── Mood Selection Animations ───────────────────────────────────────────

export const moodOrbVariants = {
  idle: {
    scale: 1,
    opacity: 0.8,
  },
  selected: {
    scale: 1.3,
    opacity: 1,
    transition: springs.bouncy,
  },
  unselected: {
    scale: 0.85,
    opacity: 0.35,
    transition: { duration: 0.3, ease: easings.premium },
  },
};

export const moodBackgroundTransition = {
  duration: 0.6,
  ease: easings.material,
};

// ── Tap / Press Feedback ────────────────────────────────────────────────

export const tapScale = {
  whileTap: { scale: 0.95 },
  whileHover: { scale: 1.02 },
  transition: springs.snappy,
};

export const tapBounce = {
  whileTap: { scale: 0.92 },
  transition: springs.bouncy,
};

// ── Emotion Badge Animations ────────────────────────────────────────────

export const badgeGlow = {
  initial: { opacity: 0, scale: 0.8, filter: "blur(4px)" },
  animate: {
    opacity: 1,
    scale: 1,
    filter: "blur(0px)",
    transition: { duration: 0.5, ease: easings.premium },
  },
};

// ── Progress Bar Animation ──────────────────────────────────────────────

export const progressFill = (value: number) => ({
  initial: { width: "0%" },
  animate: {
    width: `${value}%`,
    transition: { duration: 1.2, ease: easings.premium, delay: 0.2 },
  },
});

// ── Counter Animation ───────────────────────────────────────────────────

export const counterConfig = {
  duration: 1.5,
  ease: easings.material,
};

// ── Glow Pulse (for active states) ──────────────────────────────────────

export const glowPulse = {
  animate: {
    boxShadow: [
      "0 0 20px rgba(124, 58, 237, 0.15)",
      "0 0 40px rgba(124, 58, 237, 0.30)",
      "0 0 20px rgba(124, 58, 237, 0.15)",
    ],
  },
  transition: {
    duration: 3,
    repeat: Infinity,
    ease: "easeInOut",
  },
};

// ── Slide Up Reveal ─────────────────────────────────────────────────────

export const slideUp = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: 16 },
  transition: { duration: 0.4, ease: easings.premium },
};

// ── Fade In ─────────────────────────────────────────────────────────────

export const fadeIn = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  transition: { duration: 0.3 },
};

// ── Emotion Wheel Animations ────────────────────────────────────────────

export const wheelNodeVariants = {
  hidden: { scale: 0, opacity: 0 },
  visible: (i: number) => ({
    scale: 1,
    opacity: 1,
    transition: {
      delay: i * 0.1,
      ...springs.bouncy,
    },
  }),
  dominant: {
    scale: 1.2,
    transition: springs.bouncy,
  },
};
