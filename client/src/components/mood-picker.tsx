/**
 * MoodPicker — beautiful "How are you feeling?" check-in component.
 *
 * Inspired by: How We Feel (2D color field), Daylio (emoji faces),
 * Headspace (bouncy selection), Calm (gradient backgrounds).
 *
 * Features:
 * - 5 gradient-filled emoji orbs with spring animations
 * - Background color shifts to match selected mood
 * - Progressive disclosure: mood → optional tags → optional note
 * - Glow effects and particle-like ambiance
 * - Haptic-feel press animations
 *
 * Usage:
 *   <MoodPicker onMoodSelect={(mood, tags, note) => { ... }} />
 */

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { springs, easings } from "@/lib/animations";
import { cn } from "@/lib/utils";

interface MoodLevel {
  id: number;
  label: string;
  emoji: string;
  gradient: [string, string];
  glow: string;
  bgTint: string; // subtle page background tint
  description: string;
}

const MOOD_LEVELS: MoodLevel[] = [
  {
    id: 1,
    label: "Terrible",
    emoji: "\u{1F629}",
    gradient: ["#F87171", "#DC2626"],
    glow: "rgba(248, 113, 113, 0.4)",
    bgTint: "rgba(248, 113, 113, 0.06)",
    description: "Really struggling",
  },
  {
    id: 2,
    label: "Bad",
    emoji: "\u{1F61E}",
    gradient: ["#FB923C", "#EA580C"],
    glow: "rgba(251, 146, 60, 0.4)",
    bgTint: "rgba(251, 146, 60, 0.06)",
    description: "Not great",
  },
  {
    id: 3,
    label: "Okay",
    emoji: "\u{1F610}",
    gradient: ["#FACC15", "#EAB308"],
    glow: "rgba(250, 204, 21, 0.4)",
    bgTint: "rgba(250, 204, 21, 0.04)",
    description: "Getting by",
  },
  {
    id: 4,
    label: "Good",
    emoji: "\u{1F60A}",
    gradient: ["#A3E635", "#65A30D"],
    glow: "rgba(163, 230, 53, 0.4)",
    bgTint: "rgba(163, 230, 53, 0.06)",
    description: "Feeling positive",
  },
  {
    id: 5,
    label: "Amazing",
    emoji: "\u{1F929}",
    gradient: ["#4ADE80", "#16A34A"],
    glow: "rgba(74, 222, 128, 0.5)",
    bgTint: "rgba(74, 222, 128, 0.06)",
    description: "On top of the world",
  },
];

const ACTIVITY_TAGS = [
  { emoji: "\u{1F4AA}", label: "Exercise" },
  { emoji: "\u{1F6CC}", label: "Good sleep" },
  { emoji: "\u{1F465}", label: "Social" },
  { emoji: "\u{1F3B5}", label: "Music" },
  { emoji: "\u{2615}", label: "Coffee" },
  { emoji: "\u{1F9D8}", label: "Meditation" },
  { emoji: "\u{1F4DA}", label: "Learning" },
  { emoji: "\u{1F333}", label: "Nature" },
  { emoji: "\u{1F37D}\uFE0F", label: "Good food" },
  { emoji: "\u{2764}\uFE0F", label: "Love" },
];

export interface MoodPickerResult {
  moodLevel: number;
  moodLabel: string;
  tags: string[];
  note: string;
  timestamp: string;
}

interface MoodPickerProps {
  /** Called when user completes mood selection (after any layer) */
  onMoodSelect?: (result: MoodPickerResult) => void;
  /** Show greeting with user's name */
  userName?: string;
  /** Compact mode (no tags/notes, just mood selection) */
  compact?: boolean;
}

export function MoodPicker({ onMoodSelect, userName, compact = false }: MoodPickerProps) {
  const [selected, setSelected] = useState<number | null>(null);
  const [step, setStep] = useState<"mood" | "tags" | "note" | "done">("mood");
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [note, setNote] = useState("");

  const selectedMood = MOOD_LEVELS.find(m => m.id === selected);

  const handleMoodSelect = useCallback((id: number) => {
    setSelected(id);

    if (compact) {
      const mood = MOOD_LEVELS.find(m => m.id === id)!;
      onMoodSelect?.({
        moodLevel: id,
        moodLabel: mood.label,
        tags: [],
        note: "",
        timestamp: new Date().toISOString(),
      });
      setStep("done");
    }
  }, [compact, onMoodSelect]);

  const handleContinue = useCallback(() => {
    if (step === "mood" && selected) {
      setStep("tags");
    } else if (step === "tags") {
      setStep("note");
    } else if (step === "note" && selectedMood) {
      onMoodSelect?.({
        moodLevel: selected!,
        moodLabel: selectedMood.label,
        tags: selectedTags,
        note,
        timestamp: new Date().toISOString(),
      });
      setStep("done");
    }
  }, [step, selected, selectedMood, selectedTags, note, onMoodSelect]);

  const handleSkip = useCallback(() => {
    if (step === "tags") setStep("note");
    else if (step === "note" && selectedMood) {
      onMoodSelect?.({
        moodLevel: selected!,
        moodLabel: selectedMood.label,
        tags: selectedTags,
        note: "",
        timestamp: new Date().toISOString(),
      });
      setStep("done");
    }
  }, [step, selected, selectedMood, selectedTags, onMoodSelect]);

  const toggleTag = (tag: string) => {
    setSelectedTags(prev =>
      prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
    );
  };

  // Determine greeting based on time of day
  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";

  return (
    <motion.div
      className="relative rounded-2xl overflow-hidden"
      animate={{
        background: selectedMood
          ? `linear-gradient(160deg, ${selectedMood.bgTint}, transparent 80%)`
          : "transparent",
      }}
      transition={{ duration: 0.6, ease: easings.material }}
    >
      {/* Ambient glow for selected mood */}
      <AnimatePresence>
        {selectedMood && (
          <motion.div
            key={selectedMood.id}
            className="absolute top-0 left-1/2 -translate-x-1/2 w-40 h-40 rounded-full blur-3xl"
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 0.25, scale: 1.2 }}
            exit={{ opacity: 0, scale: 0.5 }}
            style={{ background: selectedMood.glow }}
            transition={{ duration: 0.6 }}
          />
        )}
      </AnimatePresence>

      <div className="relative z-10 p-5 space-y-5">
        {/* Greeting */}
        <AnimatePresence mode="wait">
          {step === "mood" && (
            <motion.div
              key="greeting"
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="text-center space-y-1"
            >
              <p className="text-sm text-foreground/40 font-medium">
                {greeting}{userName ? `, ${userName}` : ""}
              </p>
              <h3 className="text-lg font-semibold text-foreground/90">
                How are you feeling?
              </h3>
            </motion.div>
          )}

          {step === "tags" && (
            <motion.div
              key="tags-heading"
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="text-center space-y-1"
            >
              <p className="text-lg font-semibold text-foreground/90">
                What's going on?
              </p>
              <p className="text-xs text-foreground/40">Optional — tap to tag</p>
            </motion.div>
          )}

          {step === "note" && (
            <motion.div
              key="note-heading"
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="text-center space-y-1"
            >
              <p className="text-lg font-semibold text-foreground/90">
                Anything else?
              </p>
              <p className="text-xs text-foreground/40">Optional — add a note</p>
            </motion.div>
          )}

          {step === "done" && selectedMood && (
            <motion.div
              key="done"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-center space-y-2"
            >
              <motion.span
                className="text-4xl block"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                {selectedMood.emoji}
              </motion.span>
              <p className="text-sm text-foreground/50">
                Logged as <span className="font-semibold text-foreground/70">{selectedMood.label}</span>
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Mood orbs */}
        {(step === "mood" || step === "tags" || step === "note") && (
          <div className="flex justify-center gap-4">
            {MOOD_LEVELS.map((mood, i) => {
              const isSelected = selected === mood.id;
              const hasSelection = selected !== null;

              return (
                <motion.button
                  key={mood.id}
                  onClick={() => handleMoodSelect(mood.id)}
                  initial={{ opacity: 0, y: 20, scale: 0.5 }}
                  animate={{
                    opacity: hasSelection && !isSelected ? 0.4 : 1,
                    y: 0,
                    scale: isSelected ? 1.25 : hasSelection ? 0.85 : 1,
                  }}
                  whileHover={!isSelected ? { scale: hasSelection ? 0.95 : 1.1 } : {}}
                  whileTap={{ scale: 0.9 }}
                  transition={{
                    ...springs.bouncy,
                    delay: i * 0.06,
                    opacity: { duration: 0.3 },
                  }}
                  className="relative flex flex-col items-center gap-1.5 outline-none"
                >
                  {/* Glow ring behind selected orb */}
                  {isSelected && (
                    <motion.div
                      className="absolute -inset-2 rounded-full"
                      initial={{ opacity: 0, scale: 0 }}
                      animate={{ opacity: 1, scale: 1 }}
                      style={{
                        background: `radial-gradient(circle, ${mood.glow}, transparent 70%)`,
                      }}
                      transition={springs.bouncy}
                    />
                  )}

                  {/* Orb */}
                  <div
                    className={cn(
                      "relative w-12 h-12 rounded-full flex items-center justify-center",
                      "transition-shadow duration-300",
                    )}
                    style={{
                      background: `linear-gradient(135deg, ${mood.gradient[0]}, ${mood.gradient[1]})`,
                      boxShadow: isSelected
                        ? `0 0 24px ${mood.glow}, 0 4px 12px rgba(0,0,0,0.3)`
                        : "0 2px 8px rgba(0,0,0,0.2)",
                    }}
                  >
                    <span className={cn("text-xl", isSelected && "text-2xl")}>
                      {mood.emoji}
                    </span>
                  </div>

                  {/* Label (shown when selected or no selection) */}
                  <AnimatePresence>
                    {(isSelected || !hasSelection) && (
                      <motion.span
                        initial={{ opacity: 0, y: 4 }}
                        animate={{ opacity: isSelected ? 1 : 0.5, y: 0 }}
                        exit={{ opacity: 0, y: 4 }}
                        className={cn(
                          "text-[10px] font-medium",
                          isSelected ? "text-foreground/80" : "text-foreground/40"
                        )}
                      >
                        {mood.label}
                      </motion.span>
                    )}
                  </AnimatePresence>
                </motion.button>
              );
            })}
          </div>
        )}

        {/* Activity tags (step 2) */}
        <AnimatePresence>
          {step === "tags" && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="space-y-3 overflow-hidden"
            >
              <div className="flex flex-wrap justify-center gap-2">
                {ACTIVITY_TAGS.map((tag, i) => {
                  const isActive = selectedTags.includes(tag.label);
                  return (
                    <motion.button
                      key={tag.label}
                      onClick={() => toggleTag(tag.label)}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      whileTap={{ scale: 0.92 }}
                      transition={{ delay: i * 0.04, ...springs.snappy }}
                      className={cn(
                        "px-3 py-1.5 rounded-full text-xs font-medium",
                        "border transition-all duration-200",
                        isActive
                          ? "border-primary/40 bg-primary/15 text-primary"
                          : "border-foreground/8 bg-foreground/[0.03] text-foreground/50 hover:bg-foreground/[0.06]"
                      )}
                    >
                      <span className="mr-1">{tag.emoji}</span>
                      {tag.label}
                    </motion.button>
                  );
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Note input (step 3) */}
        <AnimatePresence>
          {step === "note" && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              <textarea
                value={note}
                onChange={e => setNote(e.target.value)}
                placeholder="How's your day going?"
                rows={2}
                className={cn(
                  "w-full rounded-xl px-4 py-3 text-sm resize-none",
                  "bg-foreground/[0.03] border border-foreground/8",
                  "text-foreground placeholder:text-foreground/30",
                  "focus:outline-none focus:border-primary/30 focus:bg-foreground/[0.05]",
                  "transition-all duration-200"
                )}
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Action buttons */}
        {!compact && step !== "done" && selected && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex justify-center gap-3"
          >
            {step !== "mood" && (
              <motion.button
                onClick={handleSkip}
                whileTap={{ scale: 0.95 }}
                className="px-4 py-2 rounded-xl text-xs font-medium text-foreground/40 hover:text-foreground/60 transition-colors"
              >
                Skip
              </motion.button>
            )}
            <motion.button
              onClick={handleContinue}
              whileTap={{ scale: 0.95 }}
              whileHover={{ scale: 1.02 }}
              transition={springs.snappy}
              className={cn(
                "px-6 py-2 rounded-xl text-xs font-semibold text-white",
                "transition-all duration-200",
              )}
              style={{
                background: selectedMood
                  ? `linear-gradient(135deg, ${selectedMood.gradient[0]}, ${selectedMood.gradient[1]})`
                  : "linear-gradient(135deg, #7C3AED, #6366F1)",
                boxShadow: selectedMood
                  ? `0 4px 16px ${selectedMood.glow}`
                  : "0 4px 16px rgba(124, 58, 237, 0.3)",
              }}
            >
              {step === "mood" ? "Continue" : step === "tags" ? "Next" : "Save"}
            </motion.button>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}
