/**
 * MoodPicker — 2D Mood Meter inspired by How We Feel (Yale/Marc Brackett).
 *
 * A color-coded 2D field where:
 *   - X axis: Pleasantness (unpleasant ← → pleasant)
 *   - Y axis: Energy (low ↓ → high ↑)
 *
 * Four quadrants, each with its own color family:
 *   - Red (top-left):    High energy + Unpleasant (stressed, angry, anxious)
 *   - Yellow (top-right): High energy + Pleasant (excited, joyful, energized)
 *   - Green (bottom-right): Low energy + Pleasant (calm, content, peaceful)
 *   - Blue (bottom-left):  Low energy + Unpleasant (sad, drained, melancholy)
 *
 * After placing on the grid, user picks a specific emotion word from that zone.
 * Then optional context tags and notes.
 *
 * References:
 *   - How We Feel (howwefeel.org) — 2D Mood Meter, 144 emotion words
 *   - Russell's Circumplex Model of Affect
 *   - Headspace dark mode: #131313, rgba overlays
 *   - Finch: weather metaphors, sparkle micro-interactions
 */

import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { springs, easings } from "@/lib/animations";
import { cn } from "@/lib/utils";

// ── Quadrant emotion words ──────────────────────────────────────────────

const QUADRANT_EMOTIONS: Record<string, { words: string[]; emoji: string }> = {
  "high-unpleasant": {
    emoji: "\u{1F525}",
    words: ["Stressed", "Anxious", "Angry", "Frustrated", "Overwhelmed", "Restless", "Irritable", "Tense"],
  },
  "high-pleasant": {
    emoji: "\u{2728}",
    words: ["Excited", "Joyful", "Energized", "Inspired", "Grateful", "Proud", "Playful", "Hopeful"],
  },
  "low-pleasant": {
    emoji: "\u{1F343}",
    words: ["Calm", "Content", "Peaceful", "Relaxed", "Serene", "Thoughtful", "Cozy", "Grateful"],
  },
  "low-unpleasant": {
    emoji: "\u{1F30A}",
    words: ["Sad", "Drained", "Lonely", "Melancholy", "Tired", "Numb", "Bored", "Defeated"],
  },
};

// ── Quadrant colors ─────────────────────────────────────────────────────

const QUADRANT_COLORS = {
  "high-unpleasant": { bg: "#DC2626", light: "#FCA5A5", accent: "#EF4444", label: "High Energy, Unpleasant" },
  "high-pleasant":   { bg: "#EAB308", light: "#FDE68A", accent: "#FBBF24", label: "High Energy, Pleasant" },
  "low-pleasant":    { bg: "#16A34A", light: "#86EFAC", accent: "#4ADE80", label: "Low Energy, Pleasant" },
  "low-unpleasant":  { bg: "#2563EB", light: "#93C5FD", accent: "#60A5FA", label: "Low Energy, Unpleasant" },
};

// ── Activity tags ───────────────────────────────────────────────────────

const ACTIVITY_TAGS = [
  { emoji: "\u{1F4AA}", label: "Exercise" },
  { emoji: "\u{1F6CC}", label: "Good sleep" },
  { emoji: "\u{1F465}", label: "Social" },
  { emoji: "\u{1F3B5}", label: "Music" },
  { emoji: "\u{1F9D8}", label: "Meditation" },
  { emoji: "\u{1F333}", label: "Nature" },
  { emoji: "\u{1F37D}\uFE0F", label: "Good food" },
  { emoji: "\u{1F4DA}", label: "Learning" },
  { emoji: "\u{2615}", label: "Coffee" },
  { emoji: "\u{2764}\uFE0F", label: "Love" },
];

// ── Types ───────────────────────────────────────────────────────────────

export interface MoodPickerResult {
  quadrant: string;
  emotionWord: string;
  energy: number;       // 0-1 (low to high)
  pleasantness: number; // 0-1 (unpleasant to pleasant)
  tags: string[];
  note: string;
  timestamp: string;
}

interface MoodPickerProps {
  onMoodSelect?: (result: MoodPickerResult) => void;
  userName?: string;
  compact?: boolean;
}

type Step = "grid" | "word" | "tags" | "done";

// ── Component ───────────────────────────────────────────────────────────

export function MoodPicker({ onMoodSelect, userName, compact = false }: MoodPickerProps) {
  const [step, setStep] = useState<Step>("grid");
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const [quadrant, setQuadrant] = useState<string | null>(null);
  const [selectedWord, setSelectedWord] = useState<string | null>(null);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [note, setNote] = useState("");
  const gridRef = useRef<HTMLDivElement>(null);

  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";

  // Determine quadrant from position
  const getQuadrant = (x: number, y: number): string => {
    const isPleasant = x > 0.5;
    const isHighEnergy = y < 0.5; // y=0 is top (high energy)
    if (isHighEnergy && !isPleasant) return "high-unpleasant";
    if (isHighEnergy && isPleasant) return "high-pleasant";
    if (!isHighEnergy && isPleasant) return "low-pleasant";
    return "low-unpleasant";
  };

  // Handle grid tap/touch
  const handleGridInteraction = useCallback((clientX: number, clientY: number) => {
    if (!gridRef.current) return;
    const rect = gridRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));
    setPosition({ x, y });
    setQuadrant(getQuadrant(x, y));
  }, []);

  const handleGridClick = (e: React.MouseEvent) => {
    handleGridInteraction(e.clientX, e.clientY);
  };

  const handleGridTouch = (e: React.TouchEvent) => {
    if (e.touches.length > 0) {
      handleGridInteraction(e.touches[0].clientX, e.touches[0].clientY);
    }
  };

  const handleWordSelect = (word: string) => {
    setSelectedWord(word);
    if (compact) {
      finishSelection(word);
    } else {
      setStep("tags");
    }
  };

  const toggleTag = (tag: string) => {
    setSelectedTags(prev =>
      prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
    );
  };

  const finishSelection = (word?: string) => {
    if (!position || !quadrant) return;
    onMoodSelect?.({
      quadrant,
      emotionWord: word ?? selectedWord ?? "",
      energy: 1 - position.y,
      pleasantness: position.x,
      tags: selectedTags,
      note,
      timestamp: new Date().toISOString(),
    });
    setStep("done");
  };

  const qColors = quadrant ? QUADRANT_COLORS[quadrant as keyof typeof QUADRANT_COLORS] : null;

  return (
    <div className="relative p-5 space-y-4">
      {/* Ambient glow behind grid */}
      {qColors && (
        <motion.div
          className="absolute top-0 left-1/2 -translate-x-1/2 w-48 h-48 rounded-full blur-[80px] pointer-events-none"
          animate={{ background: qColors.bg, opacity: 0.15 }}
          transition={{ duration: 0.8 }}
        />
      )}

      {/* ── Step: Grid ── */}
      <AnimatePresence mode="wait">
        {step === "grid" && (
          <motion.div
            key="grid"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, y: -12 }}
            className="space-y-4 relative z-10"
          >
            {/* Greeting */}
            <div className="text-center space-y-1">
              <p className="text-xs font-medium" style={{ color: "rgba(255,255,255,0.35)" }}>
                {greeting}{userName ? `, ${userName}` : ""}
              </p>
              <h3 className="text-lg font-semibold" style={{ color: "rgba(255,255,255,0.9)" }}>
                How are you feeling?
              </h3>
            </div>

            {/* 2D Mood Meter Grid */}
            <div className="flex flex-col items-center">
              {/* Energy label (top) */}
              <span className="text-[10px] font-medium mb-1.5" style={{ color: "rgba(255,255,255,0.3)" }}>
                HIGH ENERGY
              </span>

              <div className="flex items-center gap-2">
                {/* Unpleasant label (left) */}
                <span className="text-[10px] font-medium writing-mode-vertical"
                  style={{ color: "rgba(255,255,255,0.3)", writingMode: "vertical-rl", transform: "rotate(180deg)" }}>
                  UNPLEASANT
                </span>

                {/* The Grid */}
                <div
                  ref={gridRef}
                  onClick={handleGridClick}
                  onTouchMove={handleGridTouch}
                  onTouchStart={handleGridTouch}
                  className="relative w-[260px] h-[260px] rounded-2xl overflow-hidden cursor-crosshair select-none touch-none"
                  style={{
                    background: `
                      conic-gradient(
                        from 225deg at 50% 50%,
                        #DC2626 0deg,
                        #EAB308 90deg,
                        #16A34A 180deg,
                        #2563EB 270deg,
                        #DC2626 360deg
                      )`,
                    boxShadow: "inset 0 0 60px rgba(0,0,0,0.4)",
                  }}
                >
                  {/* Soft center fade for depth */}
                  <div className="absolute inset-0" style={{
                    background: "radial-gradient(circle at 50% 50%, rgba(0,0,0,0.25) 0%, transparent 70%)",
                  }} />

                  {/* Grid lines — subtle crosshair */}
                  <div className="absolute left-1/2 top-0 bottom-0 w-px" style={{ background: "rgba(255,255,255,0.08)" }} />
                  <div className="absolute top-1/2 left-0 right-0 h-px" style={{ background: "rgba(255,255,255,0.08)" }} />

                  {/* Quadrant emoji hints */}
                  <span className="absolute top-3 left-3 text-lg opacity-30">{QUADRANT_EMOTIONS["high-unpleasant"].emoji}</span>
                  <span className="absolute top-3 right-3 text-lg opacity-30">{QUADRANT_EMOTIONS["high-pleasant"].emoji}</span>
                  <span className="absolute bottom-3 right-3 text-lg opacity-30">{QUADRANT_EMOTIONS["low-pleasant"].emoji}</span>
                  <span className="absolute bottom-3 left-3 text-lg opacity-30">{QUADRANT_EMOTIONS["low-unpleasant"].emoji}</span>

                  {/* Selection indicator */}
                  {position && (
                    <motion.div
                      className="absolute pointer-events-none"
                      animate={{
                        left: `${position.x * 100}%`,
                        top: `${position.y * 100}%`,
                      }}
                      transition={springs.snappy}
                      style={{ transform: "translate(-50%, -50%)" }}
                    >
                      {/* Outer glow */}
                      <motion.div
                        className="absolute -inset-6 rounded-full"
                        animate={{ opacity: [0.3, 0.6, 0.3], scale: [1, 1.1, 1] }}
                        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                        style={{ background: `radial-gradient(circle, ${qColors?.accent ?? "#fff"}60, transparent 70%)` }}
                      />
                      {/* Dot */}
                      <motion.div
                        className="w-6 h-6 rounded-full border-2 border-white relative"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={springs.bouncy}
                        style={{
                          background: qColors?.accent ?? "#fff",
                          boxShadow: `0 0 16px ${qColors?.accent ?? "#fff"}80, 0 0 32px ${qColors?.accent ?? "#fff"}40`,
                        }}
                      />
                    </motion.div>
                  )}
                </div>

                {/* Pleasant label (right) */}
                <span className="text-[10px] font-medium"
                  style={{ color: "rgba(255,255,255,0.3)", writingMode: "vertical-rl" }}>
                  PLEASANT
                </span>
              </div>

              {/* Low energy label (bottom) */}
              <span className="text-[10px] font-medium mt-1.5" style={{ color: "rgba(255,255,255,0.3)" }}>
                LOW ENERGY
              </span>
            </div>

            {/* Quadrant label + Continue */}
            {quadrant && qColors && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col items-center gap-3"
              >
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ background: qColors.accent, boxShadow: `0 0 8px ${qColors.accent}80` }} />
                  <span className="text-sm font-medium" style={{ color: qColors.accent }}>{qColors.label}</span>
                </div>
                <motion.button
                  onClick={() => setStep("word")}
                  whileTap={{ scale: 0.95 }}
                  whileHover={{ scale: 1.02 }}
                  transition={springs.snappy}
                  className="px-6 py-2.5 rounded-xl text-sm font-semibold text-white"
                  style={{
                    background: `linear-gradient(135deg, ${qColors.bg}, ${qColors.accent})`,
                    boxShadow: `0 4px 20px ${qColors.bg}60`,
                  }}
                >
                  Continue
                </motion.button>
              </motion.div>
            )}
          </motion.div>
        )}

        {/* ── Step: Word Selection ── */}
        {step === "word" && quadrant && (
          <motion.div
            key="word"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-4 relative z-10"
          >
            <div className="text-center space-y-1">
              <p className="text-lg font-semibold" style={{ color: "rgba(255,255,255,0.9)" }}>
                What best describes it?
              </p>
              <p className="text-xs" style={{ color: "rgba(255,255,255,0.35)" }}>
                Pick the closest word
              </p>
            </div>

            <div className="flex flex-wrap justify-center gap-2.5">
              {QUADRANT_EMOTIONS[quadrant]?.words.map((word, i) => {
                const isSelected = selectedWord === word;
                const colors = QUADRANT_COLORS[quadrant as keyof typeof QUADRANT_COLORS];
                return (
                  <motion.button
                    key={word}
                    onClick={() => handleWordSelect(word)}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    whileTap={{ scale: 0.9 }}
                    transition={{ delay: i * 0.04, ...springs.snappy }}
                    className={cn(
                      "px-4 py-2 rounded-full text-sm font-medium",
                      "border transition-all duration-200",
                    )}
                    style={{
                      background: isSelected ? `${colors.accent}25` : "rgba(255,255,255,0.04)",
                      borderColor: isSelected ? `${colors.accent}50` : "rgba(255,255,255,0.08)",
                      color: isSelected ? colors.accent : "rgba(255,255,255,0.6)",
                      boxShadow: isSelected ? `0 0 16px ${colors.accent}30` : "none",
                    }}
                  >
                    {word}
                  </motion.button>
                );
              })}
            </div>

            {/* Back button */}
            <div className="flex justify-center">
              <button
                onClick={() => { setStep("grid"); setSelectedWord(null); }}
                className="text-xs font-medium px-3 py-1.5"
                style={{ color: "rgba(255,255,255,0.3)" }}
              >
                Back to grid
              </button>
            </div>
          </motion.div>
        )}

        {/* ── Step: Tags ── */}
        {step === "tags" && quadrant && (
          <motion.div
            key="tags"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-4 relative z-10"
          >
            <div className="text-center space-y-1">
              <p className="text-lg font-semibold" style={{ color: "rgba(255,255,255,0.9)" }}>
                What's going on?
              </p>
              <p className="text-xs" style={{ color: "rgba(255,255,255,0.35)" }}>Optional — tap what applies</p>
            </div>

            <div className="flex flex-wrap justify-center gap-2">
              {ACTIVITY_TAGS.map((tag, i) => {
                const isActive = selectedTags.includes(tag.label);
                const colors = QUADRANT_COLORS[quadrant as keyof typeof QUADRANT_COLORS];
                return (
                  <motion.button
                    key={tag.label}
                    onClick={() => toggleTag(tag.label)}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    whileTap={{ scale: 0.92 }}
                    transition={{ delay: i * 0.03, ...springs.snappy }}
                    className="px-3 py-1.5 rounded-full text-xs font-medium border transition-all duration-200"
                    style={{
                      background: isActive ? `${colors.accent}18` : "rgba(255,255,255,0.03)",
                      borderColor: isActive ? `${colors.accent}40` : "rgba(255,255,255,0.06)",
                      color: isActive ? colors.accent : "rgba(255,255,255,0.45)",
                    }}
                  >
                    <span className="mr-1">{tag.emoji}</span>{tag.label}
                  </motion.button>
                );
              })}
            </div>

            {/* Note input */}
            <textarea
              value={note}
              onChange={e => setNote(e.target.value)}
              placeholder="Anything else on your mind?"
              rows={2}
              className="w-full rounded-xl px-4 py-3 text-sm resize-none transition-all duration-200"
              style={{
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.06)",
                color: "rgba(255,255,255,0.85)",
                outline: "none",
              }}
              onFocus={e => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.15)"; }}
              onBlur={e => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.06)"; }}
            />

            {/* Actions */}
            <div className="flex justify-center gap-3">
              <button
                onClick={() => setStep("word")}
                className="px-4 py-2 rounded-xl text-xs font-medium"
                style={{ color: "rgba(255,255,255,0.35)" }}
              >
                Back
              </button>
              <motion.button
                onClick={() => finishSelection()}
                whileTap={{ scale: 0.95 }}
                transition={springs.snappy}
                className="px-6 py-2.5 rounded-xl text-sm font-semibold text-white"
                style={{
                  background: `linear-gradient(135deg, ${qColors?.bg}, ${qColors?.accent})`,
                  boxShadow: `0 4px 20px ${qColors?.bg}60`,
                }}
              >
                Save
              </motion.button>
            </div>
          </motion.div>
        )}

        {/* ── Step: Done ── */}
        {step === "done" && quadrant && (
          <motion.div
            key="done"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center space-y-3 relative z-10 py-4"
          >
            <motion.div
              className="text-4xl"
              animate={{ scale: [1, 1.3, 1], rotate: [0, 10, -10, 0] }}
              transition={{ duration: 0.6 }}
            >
              {QUADRANT_EMOTIONS[quadrant]?.emoji}
            </motion.div>
            <div>
              <p className="text-base font-semibold" style={{ color: qColors?.accent }}>
                {selectedWord}
              </p>
              <p className="text-xs mt-1" style={{ color: "rgba(255,255,255,0.4)" }}>
                Logged — take care of yourself
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
