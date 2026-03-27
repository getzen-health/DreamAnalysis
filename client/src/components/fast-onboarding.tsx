/**
 * FastOnboarding — 4 screens, 60 seconds to first value.
 *
 * Screen 1: "Welcome to AntarAI" — app purpose in one sentence, illustration
 * Screen 2: "How it works" — 3 icons: Voice (speak), Brain (EEG), Health (sync)
 * Screen 3: "Quick setup" — choose what to enable: voice, EEG, health sync
 * Screen 4: "You're ready" — CTA to do first voice analysis
 *
 * Uses framer-motion for slide transitions.
 * Skip button visible on every screen.
 */

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic,
  Brain,
  Heart,
  ChevronRight,
  Sparkles,
  Activity,
} from "lucide-react";
import { hapticLight, hapticSuccess } from "@/lib/haptics";
import { sbRemoveSetting, sbSaveSetting } from "../lib/supabase-store";

// ── Types & Constants ─────────────────────────────────────────────────────

type ScreenIndex = 0 | 1 | 2 | 3;

interface Props {
  onComplete: () => void;
}

function markFastComplete() {
  try {
    sbSaveSetting("ndw_onboarding_complete", "true");
    sbRemoveSetting("ndw_onboarding_step");
    sbRemoveSetting("ndw_onboarding_path");
  } catch {
    // ignore quota errors
  }
}

// ── Slide animation variants ──────────────────────────────────────────────

const slideVariants = {
  enter: (direction: number) => ({
    x: direction > 0 ? 200 : -200,
    opacity: 0,
  }),
  center: {
    x: 0,
    opacity: 1,
  },
  exit: (direction: number) => ({
    x: direction > 0 ? -200 : 200,
    opacity: 0,
  }),
};

// ── Dot indicators ────────────────────────────────────────────────────────

function DotIndicators({ current, total }: { current: number; total: number }) {
  return (
    <div style={{ display: "flex", justifyContent: "center", gap: 8 }}>
      {Array.from({ length: total }, (_, i) => (
        <div
          key={i}
          style={{
            width: i === current ? 24 : 8,
            height: 8,
            borderRadius: 4,
            background: i === current
              ? "hsl(var(--primary))"
              : "hsl(var(--muted-foreground) / 0.3)",
            transition: "all 0.3s ease",
          }}
        />
      ))}
    </div>
  );
}

// ── Screen 1: Welcome ─────────────────────────────────────────────────────

function WelcomeScreen() {
  return (
    <div style={{ textAlign: "center" }}>
      {/* Illustration circle */}
      <div
        style={{
          width: 100,
          height: 100,
          borderRadius: "50%",
          background: "linear-gradient(135deg, hsl(270 40% 50% / 0.15), hsl(200 70% 55% / 0.15))",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          margin: "0 auto 24px",
        }}
      >
        <Sparkles style={{ width: 44, height: 44, color: "hsl(var(--primary))" }} />
      </div>

      <h1
        style={{
          fontSize: 28,
          fontWeight: 700,
          color: "var(--foreground)",
          marginBottom: 12,
        }}
      >
        Welcome to AntarAI
      </h1>
      <p
        style={{
          fontSize: 15,
          color: "var(--muted-foreground)",
          lineHeight: 1.5,
          maxWidth: 320,
          margin: "0 auto",
        }}
      >
        Understand your mind and body through voice analysis, brain monitoring,
        and health tracking — all in one place.
      </p>
    </div>
  );
}

// ── Screen 2: How It Works ────────────────────────────────────────────────

const HOW_IT_WORKS = [
  {
    icon: Mic,
    title: "Voice",
    description: "Speak for 10 seconds — AI reads your emotional state",
    color: "hsl(200 70% 55%)",
  },
  {
    icon: Brain,
    title: "Brain",
    description: "Optional EEG headband for real-time neural insights",
    color: "hsl(270 40% 60%)",
  },
  {
    icon: Heart,
    title: "Health",
    description: "Sync steps, heart rate, and sleep from your wearable",
    color: "hsl(0 65% 55%)",
  },
];

function HowItWorksScreen() {
  return (
    <div>
      <h2
        style={{
          fontSize: 24,
          fontWeight: 700,
          color: "var(--foreground)",
          textAlign: "center",
          marginBottom: 24,
        }}
      >
        How it works
      </h2>
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {HOW_IT_WORKS.map((item) => {
          const Icon = item.icon;
          return (
            <div
              key={item.title}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 16,
                background: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: 16,
                padding: "16px",
              }}
            >
              <div
                style={{
                  width: 48,
                  height: 48,
                  borderRadius: 12,
                  background: item.color.replace(")", " / 0.12)"),
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                }}
              >
                <Icon style={{ width: 24, height: 24, color: item.color }} />
              </div>
              <div>
                <div style={{ fontSize: 15, fontWeight: 600, color: "var(--foreground)" }}>
                  {item.title}
                </div>
                <div style={{ fontSize: 12, color: "var(--muted-foreground)", marginTop: 2 }}>
                  {item.description}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Screen 3: Quick Setup ─────────────────────────────────────────────────

interface SetupOption {
  id: string;
  icon: typeof Mic;
  title: string;
  description: string;
  color: string;
}

const SETUP_OPTIONS: SetupOption[] = [
  {
    id: "voice",
    icon: Mic,
    title: "Voice analysis",
    description: "Detect emotions from your voice",
    color: "hsl(200 70% 55%)",
  },
  {
    id: "eeg",
    icon: Brain,
    title: "EEG headband",
    description: "Connect an EEG headband for brain monitoring",
    color: "hsl(270 40% 60%)",
  },
  {
    id: "health",
    icon: Activity,
    title: "Health sync",
    description: "Apple Health or Google Fit",
    color: "hsl(0 65% 55%)",
  },
];

function QuickSetupScreen({
  enabled,
  onToggle,
}: {
  enabled: Set<string>;
  onToggle: (id: string) => void;
}) {
  return (
    <div>
      <h2
        style={{
          fontSize: 24,
          fontWeight: 700,
          color: "var(--foreground)",
          textAlign: "center",
          marginBottom: 6,
        }}
      >
        Quick setup
      </h2>
      <p
        style={{
          fontSize: 13,
          color: "var(--muted-foreground)",
          textAlign: "center",
          marginBottom: 20,
        }}
      >
        Choose what to enable. You can change this later.
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {SETUP_OPTIONS.map((opt) => {
          const Icon = opt.icon;
          const isOn = enabled.has(opt.id);
          return (
            <button
              key={opt.id}
              onClick={() => onToggle(opt.id)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 14,
                width: "100%",
                textAlign: "left",
                background: isOn ? "hsl(var(--primary) / 0.06)" : "var(--card)",
                border: `1.5px solid ${isOn ? "hsl(var(--primary) / 0.4)" : "var(--border)"}`,
                borderRadius: 16,
                padding: "14px 16px",
                cursor: "pointer",
                transition: "all 0.2s ease",
              }}
            >
              <div
                style={{
                  width: 40,
                  height: 40,
                  borderRadius: 10,
                  background: isOn
                    ? opt.color.replace(")", " / 0.15)")
                    : "hsl(var(--muted) / 0.5)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                  transition: "background 0.2s ease",
                }}
              >
                <Icon
                  style={{
                    width: 20,
                    height: 20,
                    color: isOn ? opt.color : "var(--muted-foreground)",
                    transition: "color 0.2s ease",
                  }}
                />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: "var(--foreground)" }}>
                  {opt.title}
                </div>
                <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 1 }}>
                  {opt.description}
                </div>
              </div>
              <div
                style={{
                  width: 22,
                  height: 22,
                  borderRadius: 6,
                  border: `2px solid ${isOn ? "hsl(var(--primary))" : "var(--border)"}`,
                  background: isOn ? "hsl(var(--primary))" : "transparent",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  transition: "all 0.2s ease",
                  flexShrink: 0,
                }}
              >
                {isOn && (
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <path
                      d="M2.5 6L5 8.5L9.5 3.5"
                      stroke="white"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ── Screen 4: You're Ready ────────────────────────────────────────────────

function ReadyScreen() {
  return (
    <div style={{ textAlign: "center" }}>
      <div
        style={{
          width: 80,
          height: 80,
          borderRadius: "50%",
          background: "hsl(165 60% 45% / 0.15)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          margin: "0 auto 24px",
        }}
      >
        <Sparkles style={{ width: 36, height: 36, color: "hsl(165 60% 45%)" }} />
      </div>
      <h2
        style={{
          fontSize: 26,
          fontWeight: 700,
          color: "var(--foreground)",
          marginBottom: 10,
        }}
      >
        You're ready
      </h2>
      <p
        style={{
          fontSize: 14,
          color: "var(--muted-foreground)",
          lineHeight: 1.5,
          maxWidth: 300,
          margin: "0 auto",
        }}
      >
        Start with a quick voice check-in to see how you're feeling right now.
        It takes just 10 seconds.
      </p>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────

export function FastOnboarding({ onComplete }: Props) {
  const [screen, setScreen] = useState<ScreenIndex>(0);
  const [direction, setDirection] = useState(1);
  const [enabled, setEnabled] = useState<Set<string>>(new Set(["voice"]));

  function saveConsent() {
    try {
      sbSaveSetting("ndw_consent_voice", enabled.has("voice") ? "true" : "false");
      sbSaveSetting("ndw_consent_eeg", enabled.has("eeg") ? "true" : "false");
      sbSaveSetting("ndw_consent_health", enabled.has("health") ? "true" : "false");
    } catch {
      // ignore
    }
  }

  function goNext() {
    if (screen < 3) {
      setDirection(1);
      hapticLight();
      setScreen((s) => (s + 1) as ScreenIndex);
    } else {
      finish();
    }
  }

  function finish() {
    saveConsent();
    markFastComplete();
    hapticSuccess();
    onComplete();
  }

  function toggleOption(id: string) {
    hapticLight();
    setEnabled((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const TOTAL = 4;
  const isLast = screen === 3;

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        padding: "24px 20px",
        background: "var(--background)",
        overflow: "hidden",
      }}
    >
      <div style={{ maxWidth: 400, width: "100%", margin: "0 auto" }}>
        {/* Slide content */}
        <AnimatePresence mode="wait" custom={direction}>
          <motion.div
            key={screen}
            custom={direction}
            variants={slideVariants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
            style={{ marginBottom: 32 }}
          >
            {screen === 0 && <WelcomeScreen />}
            {screen === 1 && <HowItWorksScreen />}
            {screen === 2 && (
              <QuickSetupScreen enabled={enabled} onToggle={toggleOption} />
            )}
            {screen === 3 && <ReadyScreen />}
          </motion.div>
        </AnimatePresence>

        {/* Dots */}
        <div style={{ marginBottom: 24 }}>
          <DotIndicators current={screen} total={TOTAL} />
        </div>

        {/* CTA button */}
        <button
          onClick={goNext}
          style={{
            width: "100%",
            padding: "14px",
            borderRadius: 12,
            background: "hsl(var(--primary))",
            color: "hsl(var(--primary-foreground))",
            fontSize: 15,
            fontWeight: 600,
            border: "none",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 8,
          }}
        >
          {isLast ? "Start first voice check-in" : "Continue"}
          <ChevronRight style={{ width: 18, height: 18 }} />
        </button>

        {/* Skip button — visible on every screen */}
        <button
          onClick={finish}
          style={{
            width: "100%",
            padding: "12px",
            marginTop: 10,
            background: "none",
            border: "none",
            color: "var(--muted-foreground)",
            fontSize: 13,
            cursor: "pointer",
          }}
        >
          Skip
        </button>
      </div>
    </div>
  );
}
