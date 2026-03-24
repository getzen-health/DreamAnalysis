/**
 * useBreathSync — syncs a CSS custom property (--breath-phase) to a
 * sinusoidal breathing rhythm. The entire app's visual layer can
 * reference this variable to "breathe" in sync.
 *
 * Default: 4 second cycle (calm breathing).
 * If actual breathing data is available from EEG/health, it adapts.
 *
 * CSS usage:
 *   border-color: rgba(255, 255, 255, calc(0.06 + var(--breath-phase) * 0.04));
 *   opacity: calc(0.85 + var(--breath-phase) * 0.15);
 */

import { useEffect, useRef } from "react";

const DEFAULT_BREATH_DURATION = 4000; // 4 seconds per cycle

export function useBreathSync() {
  const animRef = useRef<number>(0);
  const startTime = useRef(Date.now());

  useEffect(() => {
    const root = document.documentElement;

    // Check if there's a detected breathing duration from EEG/health
    const getBreathDuration = (): number => {
      const stored = root.style.getPropertyValue("--breathe-duration");
      if (stored) {
        const ms = parseFloat(stored) * 1000;
        if (ms > 1000 && ms < 20000) return ms;
      }
      return DEFAULT_BREATH_DURATION;
    };

    const tick = () => {
      const elapsed = Date.now() - startTime.current;
      const duration = getBreathDuration();
      // Sinusoidal: 0 (exhale) → 1 (inhale peak) → 0 (exhale)
      const phase = (Math.sin((elapsed / duration) * Math.PI * 2 - Math.PI / 2) + 1) / 2;
      root.style.setProperty("--breath-phase", phase.toFixed(3));
      animRef.current = requestAnimationFrame(tick);
    };

    animRef.current = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(animRef.current);
      root.style.removeProperty("--breath-phase");
    };
  }, []);
}
