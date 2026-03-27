// client/src/components/emotion-picker.tsx
import { useState, useMemo, KeyboardEvent } from "react";
import { EmotionTaxonomy, type Quadrant } from "@/lib/insight-engine/emotion-taxonomy";

interface Props {
  valence: number;  // 0-1
  arousal: number;  // 0-1
  onSelect: (label: string) => void;
  personalFingerprints?: Array<{ label: string; quadrant: Quadrant }>;
}

const QUADRANT_LABELS: Record<Quadrant, string> = {
  ha_pos: "High Energy Positive",
  ha_neg: "High Energy Negative",
  la_pos: "Low Energy Positive",
  la_neg: "Low Energy Negative",
};

// Taxonomy instance created inside component to avoid module-scope localStorage reads at import time
export function EmotionPicker({ valence, arousal, onSelect, personalFingerprints = [] }: Props) {
  // useMemo ensures one stable instance per component mount, not per render
  const taxonomy = useMemo(() => new EmotionTaxonomy("_picker"), []);
  const defaultQ = taxonomy.getQuadrant(valence, arousal);
  const [activeQ, setActiveQ] = useState<Quadrant>(defaultQ);
  const [custom, setCustom] = useState("");
  const [selected, setSelected] = useState<string[]>([]);

  const presets = taxonomy.getPresetsForQuadrant(activeQ);
  const personal = personalFingerprints.filter(f => f.quadrant === activeQ).map(f => f.label);

  const handleSelect = (label: string) => {
    setSelected(prev => prev.includes(label) ? prev.filter(l => l !== label) : [...prev, label]);
    onSelect(label);
  };

  const handleCustomKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && custom.trim()) {
      onSelect(custom.trim());
      setCustom("");
    }
  };

  return (
    <div className="space-y-3">
      {/* Quadrant tabs */}
      <div className="flex gap-1 flex-wrap" role="tablist">
        {(["ha_pos", "ha_neg", "la_pos", "la_neg"] as Quadrant[]).map(q => (
          <button
            key={q}
            role="tab"
            aria-selected={activeQ === q}
            onClick={() => setActiveQ(q)}
            className={`px-2.5 py-1 text-xs rounded-full transition-colors ${
              activeQ === q
                ? "bg-primary text-primary-foreground"
                : "bg-muted/50 text-muted-foreground hover:bg-muted"
            }`}
          >
            {QUADRANT_LABELS[q]}
          </button>
        ))}
      </div>

      {/* Personal vocabulary first */}
      {personal.length > 0 && (
        <div>
          <p className="text-xs text-muted-foreground mb-1.5">Your words</p>
          <div className="flex flex-wrap gap-1.5">
            {personal.map(label => (
              <button
                key={label}
                onClick={() => handleSelect(label)}
                className={`px-2.5 py-1 text-xs rounded-full border transition-colors ${
                  selected.includes(label)
                    ? "bg-primary text-primary-foreground border-primary"
                    : "border-primary/30 text-primary hover:bg-primary/10"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Preset vocabulary */}
      <div className="flex flex-wrap gap-1.5">
        {presets.map(label => (
          <button
            key={label}
            onClick={() => handleSelect(label)}
            className={`px-2.5 py-1 text-xs rounded-full border transition-colors ${
              selected.includes(label)
                ? "bg-primary/20 border-primary text-primary font-medium"
                : "border-border/40 text-foreground/70 hover:border-primary/40 hover:text-foreground"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Custom label input */}
      <input
        value={custom}
        onChange={e => setCustom(e.target.value)}
        onKeyDown={handleCustomKey}
        placeholder="Type your own word and press Enter..."
        className="w-full px-3 py-2 text-sm rounded-lg bg-muted/30 border border-border/30 text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/40"
      />
    </div>
  );
}
