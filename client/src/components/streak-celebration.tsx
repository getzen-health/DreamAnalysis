/**
 * StreakCelebration — shows a celebration overlay when hitting streak milestones.
 * Appears briefly (2.5s) with confetti-like animation + haptic.
 */

import { useState, useEffect } from "react";
import { hapticSuccess } from "@/lib/haptics";

const MILESTONES = [3, 7, 14, 30, 60, 100];

const MILESTONE_MESSAGES: Record<number, { emoji: string; title: string; subtitle: string }> = {
  3:   { emoji: "🔥", title: "3-Day Streak!", subtitle: "You're building a habit" },
  7:   { emoji: "⭐", title: "1 Week Strong!", subtitle: "Consistency is your superpower" },
  14:  { emoji: "🌟", title: "2 Weeks!", subtitle: "Most people quit by now — not you" },
  30:  { emoji: "🏆", title: "30-Day Champion!", subtitle: "You've built a real wellness routine" },
  60:  { emoji: "💎", title: "60 Days!", subtitle: "You're in the top 5% of users" },
  100: { emoji: "👑", title: "100 Days!", subtitle: "Legendary emotional awareness" },
};

export function StreakCelebration() {
  const [show, setShow] = useState(false);
  const [milestone, setMilestone] = useState<number | null>(null);

  useEffect(() => {
    function check() {
      try {
        const streak = parseInt(localStorage.getItem("ndw_streak_count") || "0", 10);
        const lastCelebrated = parseInt(localStorage.getItem("ndw_streak_celebrated") || "0", 10);

        // Find the highest milestone reached that hasn't been celebrated
        const reached = MILESTONES.filter(m => streak >= m && m > lastCelebrated);
        if (reached.length > 0) {
          const highest = reached[reached.length - 1];
          setMilestone(highest);
          setShow(true);
          hapticSuccess();
          localStorage.setItem("ndw_streak_celebrated", String(highest));

          // Auto-dismiss after 2.5s
          setTimeout(() => setShow(false), 2500);
        }
      } catch { /* ignore */ }
    }

    check();
    window.addEventListener("ndw-voice-updated", check);
    window.addEventListener("ndw-emotion-update", check);
    return () => {
      window.removeEventListener("ndw-voice-updated", check);
      window.removeEventListener("ndw-emotion-update", check);
    };
  }, []);

  if (!show || !milestone) return null;

  const msg = MILESTONE_MESSAGES[milestone] ?? { emoji: "🎉", title: `${milestone}-Day Streak!`, subtitle: "Amazing consistency" };

  return (
    <div
      onClick={() => setShow(false)}
      style={{
        position: "fixed", inset: 0, zIndex: 70,
        background: "rgba(0,0,0,0.6)",
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        gap: 12, padding: 32,
        animation: "celebFadeIn 0.3s ease",
        cursor: "pointer",
      }}
    >
      <div style={{ fontSize: 64, animation: "celebBounce 0.6s ease" }}>{msg.emoji}</div>
      <div style={{ fontSize: 24, fontWeight: 700, color: "#f2ece0", textAlign: "center" }}>{msg.title}</div>
      <div style={{ fontSize: 14, color: "#a09890", textAlign: "center" }}>{msg.subtitle}</div>

      {/* Confetti dots */}
      {Array.from({ length: 20 }).map((_, i) => (
        <div key={i} style={{
          position: "absolute",
          width: 8, height: 8, borderRadius: "50%",
          background: ["#4ade80", "#e8b94a", "#7ba7d9", "#b49ae0", "#e87676", "#0891b2"][i % 6],
          top: `${10 + Math.random() * 80}%`,
          left: `${5 + Math.random() * 90}%`,
          opacity: 0.7,
          animation: `confettiFall ${1 + Math.random()}s ease-out forwards`,
          animationDelay: `${Math.random() * 0.5}s`,
        }} />
      ))}

      <style>{`
        @keyframes celebFadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes celebBounce { 0% { transform: scale(0); } 50% { transform: scale(1.3); } 100% { transform: scale(1); } }
        @keyframes confettiFall {
          0% { transform: translateY(-20px) rotate(0deg); opacity: 0.8; }
          100% { transform: translateY(40px) rotate(180deg); opacity: 0; }
        }
      `}</style>
    </div>
  );
}
