/**
 * ScoreSplash — shows key scores for 3 seconds when opening Today tab,
 * then fades out to reveal the full dashboard.
 *
 * Shows: Emotion (hero), Readiness, Stress, Focus — animated count-up.
 * Auto-dismisses after 3s or on tap.
 */

import { useState, useEffect, useCallback } from "react";
import { hapticMedium } from "@/lib/haptics";

interface ScoreSplashProps {
  emotion: string;
  readiness: number;
  stress: number;
  focus: number;
  onDismiss: () => void;
}

const EMOTION_EMOJI: Record<string, string> = {
  happy: "😊", sad: "😢", angry: "😠", fear: "😨",
  surprise: "😲", neutral: "😐",
};

const EMOTION_COLOR: Record<string, string> = {
  happy: "#0891b2", sad: "#6366f1", angry: "#ea580c", fear: "#7c3aed",
  surprise: "#d4a017", neutral: "#94a3b8",
};

function AnimatedNumber({ target, duration = 1200 }: { target: number; duration?: number }) {
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    if (target === 0) return;
    const start = Date.now();
    const tick = () => {
      const elapsed = Date.now() - start;
      const progress = Math.min(elapsed / duration, 1);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setCurrent(Math.round(target * eased));
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [target, duration]);

  return <>{current}</>;
}

export function ScoreSplash({ emotion, readiness, stress, focus, onDismiss }: ScoreSplashProps) {
  const [visible, setVisible] = useState(true);
  const [fading, setFading] = useState(false);

  const dismiss = useCallback(() => {
    setFading(true);
    setTimeout(() => {
      setVisible(false);
      onDismiss();
    }, 400);
  }, [onDismiss]);

  // Gentle haptic on splash appearance
  useEffect(() => {
    hapticMedium();
  }, []);

  // Auto-dismiss after 3 seconds
  useEffect(() => {
    const timer = setTimeout(dismiss, 3000);
    return () => clearTimeout(timer);
  }, [dismiss]);

  if (!visible) return null;

  const emoji = EMOTION_EMOJI[emotion] ?? "😐";
  const color = EMOTION_COLOR[emotion] ?? "#94a3b8";
  const stressPercent = Math.round(stress * 100);
  const focusPercent = Math.round(focus * 100);

  return (
    <div
      onClick={dismiss}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 60,
        background: "var(--background)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 24,
        padding: 32,
        opacity: fading ? 0 : 1,
        transition: "opacity 0.4s ease",
        cursor: "pointer",
      }}
    >
      {/* Emotion hero */}
      <div style={{ fontSize: 72, lineHeight: 1, animation: "fadeInUp 0.6s ease" }}>{emoji}</div>
      <div style={{
        fontSize: 28,
        fontWeight: 700,
        color,
        textTransform: "capitalize" as const,
        animation: "fadeInUp 0.6s ease 0.1s both",
      }}>
        {emotion}
      </div>

      {/* Score grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr 1fr",
        gap: 16,
        width: "100%",
        maxWidth: 320,
        marginTop: 16,
        animation: "fadeInUp 0.6s ease 0.3s both",
      }}>
        {/* Readiness */}
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 36, fontWeight: 700, color: "#0891b2" }}>
            <AnimatedNumber target={readiness} />
          </div>
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 2 }}>Readiness</div>
        </div>
        {/* Stress */}
        <div style={{ textAlign: "center" }}>
          <div style={{
            fontSize: 36,
            fontWeight: 700,
            color: stressPercent < 30 ? "#0891b2" : stressPercent < 60 ? "#d4a017" : "#e879a8",
          }}>
            <AnimatedNumber target={stressPercent} />
            <span style={{ fontSize: 16, fontWeight: 400 }}>%</span>
          </div>
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 2 }}>Stress</div>
        </div>
        {/* Focus */}
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 36, fontWeight: 700, color: "#3b82f6" }}>
            <AnimatedNumber target={focusPercent} />
            <span style={{ fontSize: 16, fontWeight: 400 }}>%</span>
          </div>
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 2 }}>Focus</div>
        </div>
      </div>

      {/* Tap hint */}
      <div style={{
        fontSize: 12,
        color: "var(--muted-foreground)",
        marginTop: 24,
        animation: "fadeInUp 0.6s ease 0.5s both",
      }}>
        Tap anywhere to continue
      </div>

      <style>{`
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(16px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
