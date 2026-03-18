/**
 * InlineBreathe — 1-minute breathing exercise right on the Today page.
 * Expanding/contracting circle guides inhale/exhale. No navigation needed.
 */

import { useState, useEffect, useCallback } from "react";
import { hapticLight, hapticMedium } from "@/lib/haptics";

const PHASES = [
  { label: "Breathe in", duration: 4000, scale: 1.4 },
  { label: "Hold", duration: 4000, scale: 1.4 },
  { label: "Breathe out", duration: 6000, scale: 1.0 },
];

const TOTAL_CYCLE = PHASES.reduce((s, p) => s + p.duration, 0); // 14s

export function InlineBreathe({ onClose }: { onClose: () => void }) {
  const [active, setActive] = useState(true);
  const [elapsed, setElapsed] = useState(0);
  const [cycles, setCycles] = useState(0);
  const TARGET_CYCLES = 4; // ~56 seconds

  useEffect(() => {
    if (!active) return;
    const start = Date.now();
    const interval = setInterval(() => {
      const now = Date.now() - start;
      setElapsed(now);

      // Count completed cycles
      const completedCycles = Math.floor(now / TOTAL_CYCLE);
      setCycles(completedCycles);

      if (completedCycles >= TARGET_CYCLES) {
        setActive(false);
        hapticMedium();
      }
    }, 50);
    return () => clearInterval(interval);
  }, [active]);

  // Current phase
  const cyclePos = elapsed % TOTAL_CYCLE;
  let phaseElapsed = 0;
  let currentPhase = PHASES[0];
  for (const phase of PHASES) {
    if (phaseElapsed + phase.duration > cyclePos) {
      currentPhase = phase;
      break;
    }
    phaseElapsed += phase.duration;
  }

  const phaseProgress = (cyclePos - phaseElapsed) / currentPhase.duration;
  const prevScale = PHASES[(PHASES.indexOf(currentPhase) - 1 + PHASES.length) % PHASES.length].scale;
  const scale = prevScale + (currentPhase.scale - prevScale) * phaseProgress;

  // Haptic on phase transitions
  useEffect(() => {
    const phaseStart = phaseElapsed;
    if (cyclePos >= phaseStart && cyclePos < phaseStart + 100) {
      hapticLight();
    }
  }, [currentPhase.label]);

  const done = !active && cycles >= TARGET_CYCLES;
  const progress = Math.min(100, (cycles / TARGET_CYCLES) * 100);

  return (
    <div style={{
      background: "var(--card)", border: "1px solid var(--border)",
      borderRadius: 16, padding: 20, marginBottom: 14, textAlign: "center",
    }}>
      {!done ? (
        <>
          {/* Breathing circle */}
          <div style={{
            width: 100, height: 100, margin: "0 auto 12px",
            borderRadius: "50%",
            background: "radial-gradient(circle, rgba(8,145,178,0.3) 0%, rgba(8,145,178,0.05) 70%)",
            border: "2px solid rgba(8,145,178,0.4)",
            transform: `scale(${scale})`,
            transition: "transform 0.3s ease-out",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: "#0891b2" }}>
              {currentPhase.label}
            </span>
          </div>

          {/* Progress */}
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginBottom: 6 }}>
            Cycle {Math.min(cycles + 1, TARGET_CYCLES)} of {TARGET_CYCLES}
          </div>
          <div style={{ height: 4, background: "var(--border)", borderRadius: 2, overflow: "hidden", marginBottom: 10 }}>
            <div style={{ width: `${progress}%`, height: "100%", background: "#0891b2", borderRadius: 2, transition: "width 0.5s" }} />
          </div>

          <button onClick={() => { setActive(false); onClose(); }} style={{
            background: "transparent", border: "none", color: "var(--muted-foreground)",
            fontSize: 11, cursor: "pointer",
          }}>
            Skip
          </button>
        </>
      ) : (
        <>
          <div style={{ fontSize: 32, marginBottom: 8 }}>✨</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: "var(--foreground)", marginBottom: 4 }}>
            Well done
          </div>
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginBottom: 10 }}>
            1 minute of calm breathing. Your stress response is resetting.
          </div>
          <button onClick={onClose} style={{
            background: "#0891b2", color: "white", border: "none",
            borderRadius: 10, padding: "8px 20px", fontSize: 12, fontWeight: 600, cursor: "pointer",
          }}>
            Done
          </button>
        </>
      )}
    </div>
  );
}
