/**
 * Chronotype Quiz — 5-question rMEQ assessment.
 *
 * One question per screen, progress bar at top, result screen at end.
 * Saves chronotype to localStorage for baseline adjustment.
 */

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronRight, ChevronLeft, Sun, Moon, Sunrise } from "lucide-react";
import {
  CHRONOTYPE_QUESTIONS,
  scoreChronotype,
  saveChronotype,
  type ChronotypeCategory,
} from "@/lib/chronotype";

// ── Types ─────────────────────────────────────────────────────────────────

interface Props {
  onComplete: (category: ChronotypeCategory) => void;
  onClose?: () => void;
}

// ── Result descriptions ───────────────────────────────────────────────────

const RESULT_INFO: Record<ChronotypeCategory, {
  title: string;
  description: string;
  icon: typeof Sun;
  color: string;
}> = {
  morning: {
    title: "Morning Type",
    description:
      "Your brain is sharpest in the early hours. EEG readings will be adjusted to expect higher alpha power and more positive mood signals in the morning, with natural decline in the evening.",
    icon: Sunrise,
    color: "#d4a017",
  },
  intermediate: {
    title: "Intermediate Type",
    description:
      "You function well across a broad daytime window. Your EEG baselines will get mild adjustments — your brain data is already close to population averages at most times of day.",
    icon: Sun,
    color: "#0891b2",
  },
  evening: {
    title: "Evening Type",
    description:
      "Your peak performance comes later in the day. Morning EEG readings will be adjusted for naturally lower energy, and your evening data will be recognized as your true baseline.",
    icon: Moon,
    color: "#a78bfa",
  },
};

// ── Component ─────────────────────────────────────────────────────────────

export function ChronotypeQuiz({ onComplete, onClose }: Props) {
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState<number[]>([]);
  const [result, setResult] = useState<{ score: number; category: ChronotypeCategory } | null>(null);

  const totalSteps = CHRONOTYPE_QUESTIONS.length;
  const isResult = result !== null;
  const currentQuestion = !isResult ? CHRONOTYPE_QUESTIONS[step] : null;

  const handleSelect = useCallback((score: number) => {
    const updated = [...answers, score];
    setAnswers(updated);

    if (step + 1 < totalSteps) {
      setStep(step + 1);
    } else {
      // All questions answered -- compute result
      const r = scoreChronotype(updated);
      saveChronotype(r.score, r.category);
      setResult(r);
    }
  }, [answers, step, totalSteps]);

  const handleBack = useCallback(() => {
    if (step > 0) {
      setStep(step - 1);
      setAnswers(answers.slice(0, -1));
    }
  }, [step, answers]);

  const handleDone = useCallback(() => {
    if (result) {
      onComplete(result.category);
    }
  }, [result, onComplete]);

  // ── Progress bar ────────────────────────────────────────────────────────

  const progress = isResult ? 1 : step / totalSteps;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 100,
        background: "var(--background)",
        display: "flex",
        flexDirection: "column",
        fontFamily: "Inter, system-ui, sans-serif",
        color: "var(--foreground)",
      }}
    >
      {/* Header with progress */}
      <div style={{ padding: "16px 20px 0" }}>
        {/* Close / Back row */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          {step > 0 && !isResult ? (
            <button
              onClick={handleBack}
              style={{
                background: "transparent",
                border: "none",
                color: "var(--muted-foreground)",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: 4,
                fontSize: 14,
                padding: 0,
              }}
            >
              <ChevronLeft style={{ width: 18, height: 18 }} /> Back
            </button>
          ) : (
            <div />
          )}
          {onClose && !isResult && (
            <button
              onClick={onClose}
              style={{
                background: "transparent",
                border: "none",
                color: "var(--muted-foreground)",
                cursor: "pointer",
                fontSize: 14,
                padding: 0,
              }}
            >
              Skip
            </button>
          )}
        </div>

        {/* Progress bar */}
        <div
          style={{
            height: 4,
            borderRadius: 2,
            background: "var(--border)",
            overflow: "hidden",
          }}
        >
          <motion.div
            initial={false}
            animate={{ width: `${progress * 100}%` }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            style={{
              height: "100%",
              borderRadius: 2,
              background: "linear-gradient(90deg, #7c3aed, #a78bfa)",
            }}
          />
        </div>

        {!isResult && (
          <div style={{ fontSize: 12, color: "var(--muted-foreground)", marginTop: 8, textAlign: "center" }}>
            Question {step + 1} of {totalSteps}
          </div>
        )}
      </div>

      {/* Content area */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", padding: "0 24px" }}>
        <AnimatePresence mode="wait">
          {!isResult && currentQuestion ? (
            <motion.div
              key={currentQuestion.id}
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -30 }}
              transition={{ duration: 0.25 }}
            >
              {/* Question text */}
              <h2
                style={{
                  fontSize: 20,
                  fontWeight: 600,
                  lineHeight: 1.4,
                  marginBottom: 28,
                  textAlign: "center",
                }}
              >
                {currentQuestion.text}
              </h2>

              {/* Options */}
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {currentQuestion.options.map((opt) => (
                  <button
                    key={opt.label}
                    onClick={() => handleSelect(opt.score)}
                    style={{
                      background: "var(--card)",
                      border: "1px solid var(--border)",
                      borderRadius: 14,
                      padding: "14px 18px",
                      fontSize: 15,
                      color: "var(--foreground)",
                      cursor: "pointer",
                      textAlign: "left",
                      transition: "border-color 0.2s, background 0.2s",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                    }}
                  >
                    {opt.label}
                    <ChevronRight style={{ width: 16, height: 16, color: "var(--muted-foreground)", flexShrink: 0 }} />
                  </button>
                ))}
              </div>
            </motion.div>
          ) : result ? (
            <motion.div
              key="result"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              style={{ textAlign: "center" }}
            >
              {(() => {
                const info = RESULT_INFO[result.category];
                const Icon = info.icon;
                return (
                  <>
                    {/* Icon */}
                    <div
                      style={{
                        width: 80,
                        height: 80,
                        borderRadius: "50%",
                        background: `${info.color}18`,
                        border: `2px solid ${info.color}40`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        margin: "0 auto 20px",
                      }}
                    >
                      <Icon style={{ width: 36, height: 36, color: info.color }} />
                    </div>

                    {/* Title */}
                    <div style={{ fontSize: 13, fontWeight: 600, color: info.color, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>
                      Your chronotype
                    </div>
                    <h2 style={{ fontSize: 26, fontWeight: 700, marginBottom: 16 }}>
                      {info.title}
                    </h2>

                    {/* Description */}
                    <p style={{ fontSize: 14, color: "var(--muted-foreground)", lineHeight: 1.6, maxWidth: 360, margin: "0 auto 32px" }}>
                      {info.description}
                    </p>

                    {/* Score badge */}
                    <div
                      style={{
                        display: "inline-block",
                        background: "var(--card)",
                        border: "1px solid var(--border)",
                        borderRadius: 20,
                        padding: "8px 18px",
                        fontSize: 13,
                        color: "var(--muted-foreground)",
                        marginBottom: 32,
                      }}
                    >
                      rMEQ Score: {result.score}/25
                    </div>
                  </>
                );
              })()}
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>

      {/* Bottom action */}
      {isResult && (
        <div style={{ padding: "0 24px 32px" }}>
          <button
            onClick={handleDone}
            style={{
              width: "100%",
              padding: "14px 0",
              borderRadius: 14,
              border: "none",
              background: "linear-gradient(135deg, #7c3aed, #a78bfa)",
              color: "#fff",
              fontSize: 16,
              fontWeight: 600,
              cursor: "pointer",
              transition: "opacity 0.2s",
            }}
          >
            Got it
          </button>
        </div>
      )}
    </div>
  );
}
