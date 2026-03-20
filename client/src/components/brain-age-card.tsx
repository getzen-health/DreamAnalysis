/**
 * BrainAgeCard — displays estimated brain age from EEG session data.
 *
 * Shows:
 * - Estimated brain age as a large number
 * - Brain age gap relative to actual age (green = younger, amber = same, rose = older)
 * - Expandable "How it works" section
 * - Wellness disclaimer
 *
 * Reads actual age from localStorage (ndw_actual_age).
 * Reads cached brain age result from localStorage (ndw_brain_age).
 */

import { useState, useEffect } from "react";
import { Brain, ChevronDown, ChevronUp } from "lucide-react";
import { type BrainAgeResult } from "@/lib/brain-age";

// ── localStorage keys ────────────────────────────────────────────────────────

const AGE_KEY = "ndw_actual_age";
const BRAIN_AGE_KEY = "ndw_brain_age";

// ── Helpers ──────────────────────────────────────────────────────────────────

function getStoredAge(): number | null {
  try {
    const raw = localStorage.getItem(AGE_KEY);
    if (!raw) return null;
    const parsed = parseInt(raw, 10);
    return isNaN(parsed) ? null : parsed;
  } catch {
    return null;
  }
}

function getStoredBrainAge(): (BrainAgeResult & { timestamp: number }) | null {
  try {
    const raw = localStorage.getItem(BRAIN_AGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function saveActualAge(age: number): void {
  try {
    localStorage.setItem(AGE_KEY, String(age));
  } catch {
    // storage full or unavailable
  }
}

function getGapColor(gap: number): string {
  if (gap <= -3) return "#22c55e"; // green — younger brain
  if (gap <= 2) return "#d4a017";  // amber — roughly same age
  return "#f43f5e";                // rose — older brain
}

function getGapLabel(gap: number): string {
  const absGap = Math.abs(gap);
  if (gap <= -3) return `${absGap} years younger`;
  if (gap >= 3) return `${absGap} years older`;
  return "about your age";
}

function formatTimeAgo(timestamp: number): string {
  const diffMs = Date.now() - timestamp;
  const diffMin = Math.floor(diffMs / 60_000);
  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  return `${diffDay}d ago`;
}

// ── Component ────────────────────────────────────────────────────────────────

export function BrainAgeCard() {
  const [actualAge, setActualAge] = useState<number | null>(() => getStoredAge());
  const [brainAge, setBrainAge] = useState<(BrainAgeResult & { timestamp: number }) | null>(
    () => getStoredBrainAge(),
  );
  const [ageInput, setAgeInput] = useState("");
  const [showHowItWorks, setShowHowItWorks] = useState(false);
  const [askingAge, setAskingAge] = useState(false);

  // Refresh from localStorage on mount (in case another component wrote it)
  useEffect(() => {
    const stored = getStoredBrainAge();
    if (stored) setBrainAge(stored);
  }, []);

  // ── Age prompt ─────────────────────────────────────────────────────────
  if (actualAge === null && !askingAge) {
    // Show prompt to enter age on first use
    return (
      <div
        style={{
          background: "var(--card)",
          borderRadius: 20,
          border: "1px solid var(--border)",
          padding: 16,
          boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
          marginBottom: 14,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
          <Brain style={{ width: 18, height: 18, color: "#6366f1" }} />
          <span style={{ fontSize: 14, fontWeight: 600, color: "var(--foreground)" }}>
            Brain Age
          </span>
        </div>
        <p style={{ fontSize: 13, color: "var(--muted-foreground)", marginBottom: 12, lineHeight: 1.5 }}>
          Enter your age to see how your brain compares. This stays on your device.
        </p>
        <button
          onClick={() => setAskingAge(true)}
          style={{
            background: "#6366f1",
            color: "#fff",
            border: "none",
            borderRadius: 10,
            padding: "8px 16px",
            fontSize: 13,
            fontWeight: 500,
            cursor: "pointer",
          }}
        >
          Set my age
        </button>
      </div>
    );
  }

  if (askingAge) {
    return (
      <div
        style={{
          background: "var(--card)",
          borderRadius: 20,
          border: "1px solid var(--border)",
          padding: 16,
          boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
          marginBottom: 14,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
          <Brain style={{ width: 18, height: 18, color: "#6366f1" }} />
          <span style={{ fontSize: 14, fontWeight: 600, color: "var(--foreground)" }}>
            Brain Age
          </span>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <input
            type="number"
            min={10}
            max={100}
            placeholder="Your age"
            value={ageInput}
            onChange={(e) => setAgeInput(e.target.value)}
            style={{
              flex: 1,
              background: "var(--background)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              padding: "8px 12px",
              fontSize: 14,
              color: "var(--foreground)",
              outline: "none",
            }}
          />
          <button
            onClick={() => {
              const parsed = parseInt(ageInput, 10);
              if (!isNaN(parsed) && parsed >= 10 && parsed <= 100) {
                saveActualAge(parsed);
                setActualAge(parsed);
                setAskingAge(false);
              }
            }}
            style={{
              background: "#6366f1",
              color: "#fff",
              border: "none",
              borderRadius: 8,
              padding: "8px 14px",
              fontSize: 13,
              fontWeight: 500,
              cursor: "pointer",
              flexShrink: 0,
            }}
          >
            Save
          </button>
        </div>
      </div>
    );
  }

  // ── No session data yet ────────────────────────────────────────────────
  if (!brainAge) {
    return (
      <div
        style={{
          background: "var(--card)",
          borderRadius: 20,
          border: "1px solid var(--border)",
          padding: 16,
          boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
          marginBottom: 14,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
          <Brain style={{ width: 18, height: 18, color: "#6366f1" }} />
          <span style={{ fontSize: 14, fontWeight: 600, color: "var(--foreground)" }}>
            Brain Age
          </span>
        </div>
        <p style={{ fontSize: 13, color: "var(--muted-foreground)", lineHeight: 1.5 }}>
          Complete an EEG session to see your brain age estimate.
        </p>
      </div>
    );
  }

  // ── Full display ───────────────────────────────────────────────────────
  const gapColor = getGapColor(brainAge.brainAgeGap);
  const gapLabel = getGapLabel(brainAge.brainAgeGap);

  return (
    <div
      style={{
        background: "var(--card)",
        borderRadius: 20,
        border: "1px solid var(--border)",
        borderLeft: `3px solid ${gapColor}`,
        padding: 16,
        boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
        marginBottom: 14,
      }}
    >
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Brain style={{ width: 18, height: 18, color: "#6366f1" }} />
          <span style={{ fontSize: 14, fontWeight: 600, color: "var(--foreground)" }}>
            Brain Age
          </span>
        </div>
        {brainAge.timestamp && (
          <span style={{ fontSize: 11, color: "var(--muted-foreground)" }}>
            {formatTimeAgo(brainAge.timestamp)}
          </span>
        )}
      </div>

      {/* Estimated Age */}
      <div style={{ textAlign: "center", marginBottom: 8 }}>
        <div style={{ fontSize: 42, fontWeight: 700, color: gapColor, lineHeight: 1 }}>
          {brainAge.estimatedAge}
        </div>
        <div style={{ fontSize: 12, color: "var(--muted-foreground)", marginTop: 4 }}>
          estimated brain age
        </div>
      </div>

      {/* Gap Label */}
      <div
        style={{
          textAlign: "center",
          fontSize: 13,
          fontWeight: 500,
          color: gapColor,
          marginBottom: 4,
        }}
      >
        Your brain is {gapLabel} than your actual age ({actualAge})
      </div>

      {/* Alpha Peak */}
      <div style={{ textAlign: "center", fontSize: 11, color: "var(--muted-foreground)", marginBottom: 12 }}>
        Alpha peak: {brainAge.alphaPeakHz.toFixed(1)} Hz
      </div>

      {/* How it works (expandable) */}
      <button
        onClick={() => setShowHowItWorks(!showHowItWorks)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 4,
          background: "none",
          border: "none",
          color: "var(--muted-foreground)",
          fontSize: 12,
          cursor: "pointer",
          padding: 0,
          margin: "0 auto 8px",
        }}
      >
        How it works
        {showHowItWorks ? (
          <ChevronUp style={{ width: 14, height: 14 }} />
        ) : (
          <ChevronDown style={{ width: 14, height: 14 }} />
        )}
      </button>

      {showHowItWorks && (
        <div
          style={{
            background: "var(--background)",
            borderRadius: 12,
            padding: 12,
            marginBottom: 8,
            fontSize: 12,
            lineHeight: 1.6,
            color: "var(--muted-foreground)",
          }}
        >
          <div style={{ marginBottom: 6 }}>{brainAge.factors.alphaPeak}</div>
          <div style={{ marginBottom: 6 }}>{brainAge.factors.thetaPower}</div>
          <div>{brainAge.factors.betaRatio}</div>
        </div>
      )}

      {/* Disclaimer */}
      <div
        style={{
          fontSize: 10,
          color: "var(--muted-foreground)",
          opacity: 0.7,
          lineHeight: 1.5,
          textAlign: "center",
        }}
      >
        {brainAge.disclaimer}
      </div>
    </div>
  );
}
