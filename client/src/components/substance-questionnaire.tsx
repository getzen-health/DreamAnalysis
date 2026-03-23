/**
 * Pre-session substance/medication questionnaire.
 *
 * Shown before EEG sessions to capture context that affects baseline readings.
 * Takes <15 seconds to complete. Stores in localStorage, never sent to server.
 * Only asks once per day.
 */

import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Pill, Coffee, Wine, Leaf, X, Plus, ChevronRight } from "lucide-react";
import {
  saveSubstanceLog,
  hasAnsweredToday,
  type SubstanceLog,
} from "@/lib/substance-context";

// ── Types ────────────────────────────────────────────────────────────────

interface SubstanceQuestionnaireProps {
  onComplete: (log: SubstanceLog | null) => void;
}

type CaffeineOption = SubstanceLog["caffeine"];
type AlcoholOption = SubstanceLog["alcohol"];
type CannabisOption = SubstanceLog["cannabis"];

// ── Option pill button ──────────────────────────────────────────────────

function PillButton({
  label,
  selected,
  onClick,
}: {
  label: string;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`
        px-3 py-1.5 rounded-full text-xs font-medium transition-all
        ${
          selected
            ? "bg-indigo-600 text-white border-indigo-500"
            : "bg-gray-800 text-gray-400 border-gray-700 hover:bg-gray-750 hover:text-gray-300"
        }
        border
      `}
    >
      {label}
    </button>
  );
}

// ── Main component ──────────────────────────────────────────────────────

export default function SubstanceQuestionnaire({
  onComplete,
}: SubstanceQuestionnaireProps) {
  const [caffeine, setCaffeine] = useState<CaffeineOption>("none");
  const [alcohol, setAlcohol] = useState<AlcoholOption>("none");
  const [cannabis, setCannabis] = useState<CannabisOption>("none");
  const [medications, setMedications] = useState<string[]>([]);
  const [medInput, setMedInput] = useState("");

  // If already answered today, don't render
  if (hasAnsweredToday()) {
    return null;
  }

  function addMedication() {
    const trimmed = medInput.trim();
    if (trimmed && !medications.includes(trimmed)) {
      setMedications([...medications, trimmed]);
      setMedInput("");
    }
  }

  function removeMedication(med: string) {
    setMedications(medications.filter((m) => m !== med));
  }

  function handleSubmit() {
    const log: SubstanceLog = {
      timestamp: new Date().toISOString(),
      caffeine,
      alcohol,
      medications,
      cannabis,
    };
    saveSubstanceLog(log);
    onComplete(log);
  }

  function handleSkip() {
    // Save a minimal log so we don't ask again today, but pass null
    // to indicate no adjustments should be applied.
    const skipLog: SubstanceLog = {
      timestamp: new Date().toISOString(),
      caffeine: "none",
      alcohol: "none",
      medications: [],
      cannabis: "none",
    };
    saveSubstanceLog(skipLog);
    onComplete(null);
  }

  return (
    <Card className="p-5 space-y-4 bg-gray-900 border-gray-800">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Pill className="w-4 h-4 text-indigo-400" />
          <span className="text-sm font-semibold text-gray-100">
            Quick pre-session check
          </span>
        </div>
        <button
          type="button"
          onClick={handleSkip}
          className="text-xs text-gray-500 hover:text-gray-400 transition-colors"
        >
          Skip
        </button>
      </div>

      <p className="text-xs text-gray-500">
        Substances affect EEG readings. This helps us adjust your baseline.
      </p>

      {/* Caffeine */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-1.5">
          <Coffee className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-xs text-gray-300">Caffeine</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <PillButton label="None" selected={caffeine === "none"} onClick={() => setCaffeine("none")} />
          <PillButton label="1-2hrs ago" selected={caffeine === "1-2hrs_ago"} onClick={() => setCaffeine("1-2hrs_ago")} />
          <PillButton label="3-6hrs ago" selected={caffeine === "3-6hrs_ago"} onClick={() => setCaffeine("3-6hrs_ago")} />
          <PillButton label="6+ hrs ago" selected={caffeine === "6+hrs_ago"} onClick={() => setCaffeine("6+hrs_ago")} />
        </div>
      </div>

      {/* Alcohol */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-1.5">
          <Wine className="w-3.5 h-3.5 text-rose-400" />
          <span className="text-xs text-gray-300">Alcohol</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <PillButton label="None" selected={alcohol === "none"} onClick={() => setAlcohol("none")} />
          <PillButton label="Last night" selected={alcohol === "last_night"} onClick={() => setAlcohol("last_night")} />
          <PillButton label="Today" selected={alcohol === "today"} onClick={() => setAlcohol("today")} />
        </div>
      </div>

      {/* Cannabis */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-1.5">
          <Leaf className="w-3.5 h-3.5 text-green-400" />
          <span className="text-xs text-gray-300">Cannabis</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <PillButton label="None" selected={cannabis === "none"} onClick={() => setCannabis("none")} />
          <PillButton label="Today" selected={cannabis === "today"} onClick={() => setCannabis("today")} />
          <PillButton label="Yesterday" selected={cannabis === "yesterday"} onClick={() => setCannabis("yesterday")} />
        </div>
      </div>

      {/* Medications */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-1.5">
          <Pill className="w-3.5 h-3.5 text-blue-400" />
          <span className="text-xs text-gray-300">Medications</span>
        </div>
        <div className="flex gap-1.5">
          <Input
            value={medInput}
            onChange={(e) => setMedInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                addMedication();
              }
            }}
            placeholder="e.g., sertraline, melatonin"
            className="h-7 text-xs bg-gray-800 border-gray-700"
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={addMedication}
            className="h-7 w-7 shrink-0"
          >
            <Plus className="w-3 h-3" />
          </Button>
        </div>
        {medications.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {medications.map((med) => (
              <Badge
                key={med}
                variant="secondary"
                className="text-[10px] bg-gray-800 text-gray-300 gap-1"
              >
                {med}
                <button
                  type="button"
                  onClick={() => removeMedication(med)}
                  className="hover:text-gray-100"
                >
                  <X className="w-2.5 h-2.5" />
                </button>
              </Badge>
            ))}
          </div>
        )}
      </div>

      {/* Submit */}
      <Button
        onClick={handleSubmit}
        className="w-full h-8 text-xs bg-indigo-600 hover:bg-indigo-700"
      >
        Continue to session
        <ChevronRight className="w-3 h-3 ml-1" />
      </Button>
    </Card>
  );
}

/**
 * Inline context note shown when substance adjustments are active.
 * Use this below emotion displays.
 */
export function SubstanceContextNote({ note }: { note: string }) {
  if (!note) return null;

  return (
    <div className="flex items-start gap-1.5 px-3 py-2 rounded-md bg-amber-950/30 border border-amber-900/50">
      <Pill className="w-3 h-3 text-amber-400 shrink-0 mt-0.5" />
      <span className="text-[11px] text-amber-300/80">{note}</span>
    </div>
  );
}
