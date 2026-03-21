/**
 * Achievements — standalone gallery page for badges and milestones.
 *
 * Moved out of the You page so achievements get their own dedicated space.
 */

import { motion } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { AchievementBadges } from "@/components/achievements";
import { Trophy } from "lucide-react";

export default function Achievements() {
  return (
    <motion.main
      initial={pageTransition.initial}
      animate={pageTransition.animate}
      transition={pageTransition.transition}
      style={{
        background: "var(--background)",
        padding: 16,
        paddingBottom: 24,
        color: "var(--foreground)",
        fontFamily: "Inter, system-ui, sans-serif",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          marginBottom: 20,
        }}
      >
        <div
          style={{
            width: 40,
            height: 40,
            borderRadius: 12,
            background: "rgba(212, 160, 23, 0.12)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          <Trophy style={{ width: 20, height: 20, color: "#d4a017" }} />
        </div>
        <div>
          <h1
            style={{
              fontSize: 20,
              fontWeight: 700,
              color: "var(--foreground)",
              margin: 0,
              letterSpacing: "-0.3px",
            }}
          >
            Achievements
          </h1>
          <p
            style={{
              fontSize: 12,
              color: "var(--muted-foreground)",
              margin: "2px 0 0",
            }}
          >
            Your badges and milestones
          </p>
        </div>
      </div>

      {/* Full achievement badges gallery */}
      <AchievementBadges />
    </motion.main>
  );
}
