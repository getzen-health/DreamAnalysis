import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { motion } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useDevice } from "@/hooks/use-device";
import { computeBrainSync, type BrainSyncResult } from "@/lib/brain-sync";
import { Heart, Play, Square, Radio, Users } from "lucide-react";

// ── Synthetic partner data generation ─────────────────────────────────────────
// For demo mode: generates band powers that gradually converge toward person 1's
// real data, simulating the synchrony experience without requiring two headbands.

interface BandPowers {
  alpha: number;
  theta: number;
  beta: number;
}

function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}

/**
 * Generate synthetic partner band powers that drift toward target values.
 * Uses exponential smoothing with noise to feel organic.
 */
function generateSyntheticPartner(
  target: BandPowers,
  previous: BandPowers,
  convergenceRate: number,
): BandPowers {
  const noise = () => (Math.random() - 0.5) * 0.06;
  return {
    alpha: clamp(previous.alpha + (target.alpha - previous.alpha) * convergenceRate + noise(), 0, 1),
    theta: clamp(previous.theta + (target.theta - previous.theta) * convergenceRate + noise(), 0, 1),
    beta: clamp(previous.beta + (target.beta - previous.beta) * convergenceRate + noise(), 0, 1),
  };
}

// ── State color mapping ─────────────────────────────────────────────────────
// Maps dominant brain state to visualization color

function getDominantStateColor(bandPowers: BandPowers): string {
  const { alpha, theta, beta } = bandPowers;
  if (alpha >= theta && alpha >= beta) return "#8b5cf6"; // violet = calm
  if (beta >= alpha && beta >= theta) return "#f43f5e";  // rose = stressed/active
  return "#06b6d4"; // cyan = focused/meditative
}

function getDominantStateLabel(bandPowers: BandPowers): string {
  const { alpha, theta, beta } = bandPowers;
  if (alpha >= theta && alpha >= beta) return "Calm";
  if (beta >= alpha && beta >= theta) return "Active";
  return "Focused";
}

// ── Timer formatting ─────────────────────────────────────────────────────────

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

// ── Pulsing circle component ─────────────────────────────────────────────────

interface PersonCircleProps {
  label: string;
  bandPowers: BandPowers;
  syncScore: number;
}

function PersonCircle({ label, bandPowers, syncScore }: PersonCircleProps) {
  const color = getDominantStateColor(bandPowers);
  const stateLabel = getDominantStateLabel(bandPowers);
  // Pulse intensity scales with alpha (calm) — calmer = smoother pulse
  const pulseScale = 1 + bandPowers.alpha * 0.08;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
        textTransform: "uppercase", letterSpacing: "0.5px",
      }}>
        {label}
      </div>
      <div style={{ position: "relative", width: 120, height: 120 }}>
        {/* Outer glow */}
        <div style={{
          position: "absolute", inset: -8,
          borderRadius: "50%",
          background: `radial-gradient(circle, ${color}30 0%, transparent 70%)`,
          animation: "pulse 3s ease-in-out infinite",
          transform: `scale(${pulseScale})`,
          transition: "transform 1s ease, background 1s ease",
        }} />
        {/* Main circle */}
        <div style={{
          position: "absolute", inset: 0,
          borderRadius: "50%",
          background: `radial-gradient(circle at 40% 35%, ${color}60 0%, ${color}20 60%, transparent 100%)`,
          border: `2px solid ${color}50`,
          display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center",
          transition: "all 1s ease",
        }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "var(--foreground)" }}>
            {stateLabel}
          </div>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>
            {`${Math.round(bandPowers.alpha * 100)}% alpha`}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Sync bridge visualization ────────────────────────────────────────────────

interface SyncBridgeProps {
  syncResult: BrainSyncResult;
}

function SyncBridge({ syncResult }: SyncBridgeProps) {
  const { overallSync, phase } = syncResult;
  // Bridge opacity and glow intensity scale with sync
  const opacity = 0.2 + overallSync * 0.8;
  const glowSize = Math.round(4 + overallSync * 20);

  const phaseColor = phase === "deep_sync" ? "#8b5cf6"
    : phase === "in_sync" ? "#06b6d4"
    : phase === "syncing" ? "#d4a017"
    : "#6b7280";

  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", gap: 6, minWidth: 80,
    }}>
      {/* Sync percentage */}
      <div style={{
        fontSize: 28, fontWeight: 800, color: phaseColor,
        transition: "color 1s ease",
        textShadow: `0 0 ${glowSize}px ${phaseColor}40`,
      }}>
        {Math.round(overallSync * 100)}%
      </div>

      {/* Connecting line */}
      <svg width="60" height="8" viewBox="0 0 60 8">
        <defs>
          <linearGradient id="syncGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={phaseColor} stopOpacity={opacity * 0.3} />
            <stop offset="50%" stopColor={phaseColor} stopOpacity={opacity} />
            <stop offset="100%" stopColor={phaseColor} stopOpacity={opacity * 0.3} />
          </linearGradient>
        </defs>
        <rect x="0" y="2" width="60" height="4" rx="2"
          fill="url(#syncGrad)"
          style={{ transition: "all 1s ease" }}
        />
      </svg>

      {/* Phase label */}
      <div style={{
        fontSize: 10, fontWeight: 600, color: phaseColor,
        textTransform: "uppercase", letterSpacing: "0.4px",
        transition: "color 1s ease",
      }}>
        {phase.replace("_", " ")}
      </div>
    </div>
  );
}

// ── Band sync detail bars ────────────────────────────────────────────────────

function BandSyncBars({ bandSync }: { bandSync: BrainSyncResult["bandSync"] }) {
  const bands = [
    { label: "Alpha", value: bandSync.alpha, color: "#8b5cf6", description: "Calm sync" },
    { label: "Theta", value: bandSync.theta, color: "#06b6d4", description: "Deep sync" },
    { label: "Beta", value: bandSync.beta, color: "#f43f5e", description: "Active sync" },
  ];

  return (
    <div style={{
      background: "var(--card)",
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 16, padding: "12px 14px",
    }}>
      <div style={{
        fontSize: 10, fontWeight: 700, color: "var(--muted-foreground)",
        textTransform: "uppercase", letterSpacing: "0.6px", marginBottom: 10,
      }}>
        Band synchrony
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {bands.map((band) => (
          <div key={band.label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 48, fontSize: 10, fontWeight: 600, color: band.color }}>
              {band.label}
            </div>
            <div style={{
              flex: 1, height: 6, borderRadius: 3,
              background: "rgba(255,255,255,0.06)",
              overflow: "hidden",
            }}>
              <div style={{
                height: "100%", borderRadius: 3,
                background: band.color,
                width: `${band.value * 100}%`,
                transition: "width 1s ease",
              }} />
            </div>
            <div style={{ width: 36, fontSize: 10, color: "var(--muted-foreground)", textAlign: "right" }}>
              {Math.round(band.value * 100)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main page component ──────────────────────────────────────────────────────

export default function CouplesMeditation() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const bandPowers = latestFrame?.analysis?.band_powers;

  const [isActive, setIsActive] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Person 1 band powers (from real device or defaults)
  const person1: BandPowers = useMemo(() => ({
    alpha: bandPowers?.alpha ?? 0.3,
    theta: bandPowers?.theta ?? 0.2,
    beta: bandPowers?.beta ?? 0.25,
  }), [bandPowers?.alpha, bandPowers?.theta, bandPowers?.beta]);

  // Person 2 synthetic data
  const [person2, setPerson2] = useState<BandPowers>({
    alpha: 0.15 + Math.random() * 0.2,
    theta: 0.1 + Math.random() * 0.15,
    beta: 0.3 + Math.random() * 0.2,
  });

  // Convergence rate increases over time -- simulates partners syncing up
  const convergenceRate = useMemo(() => {
    if (!isActive) return 0;
    // Starts at 0.02, reaches ~0.15 after 3 minutes
    return Math.min(0.15, 0.02 + elapsed * 0.0007);
  }, [isActive, elapsed]);

  // Update synthetic partner every second
  useEffect(() => {
    if (!isActive) return;
    const interval = setInterval(() => {
      setPerson2((prev) => generateSyntheticPartner(person1, prev, convergenceRate));
    }, 1000);
    return () => clearInterval(interval);
  }, [isActive, person1, convergenceRate]);

  // Timer
  useEffect(() => {
    if (isActive) {
      timerRef.current = setInterval(() => {
        setElapsed((prev) => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isActive]);

  // Compute synchrony
  const syncResult = useMemo(() =>
    computeBrainSync(
      person1.alpha, person1.theta, person1.beta,
      person2.alpha, person2.theta, person2.beta,
    ),
    [person1, person2],
  );

  const handleToggle = useCallback(() => {
    if (isActive) {
      setIsActive(false);
      setElapsed(0);
      // Reset partner to diverged state
      setPerson2({
        alpha: 0.15 + Math.random() * 0.2,
        theta: 0.1 + Math.random() * 0.15,
        beta: 0.3 + Math.random() * 0.2,
      });
    } else {
      setIsActive(true);
    }
  }, [isActive]);

  return (
    <motion.main
      initial={pageTransition.initial}
      animate={pageTransition.animate}
      transition={pageTransition.transition}
      style={{
        background: "var(--background)",
        padding: 16,
        paddingBottom: 100,
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: 16, display: "flex", alignItems: "center", gap: 10 }}>
        <Heart style={{ width: 22, height: 22, color: "#e879a8" }} />
        <div>
          <h1 style={{
            fontSize: 20, fontWeight: 700, color: "var(--foreground)",
            margin: 0, letterSpacing: "-0.3px",
          }}>
            Couples Meditation
          </h1>
          <p style={{ fontSize: 12, color: "var(--muted-foreground)", margin: "2px 0 0" }}>
            Synchronize your brainwaves together
          </p>
        </div>
      </div>

      {/* Demo mode badge */}
      <div style={{ marginBottom: 14, display: "flex", gap: 8, flexWrap: "wrap" }}>
        <Badge variant="outline" style={{
          fontSize: 10, borderColor: "#d4a01740", color: "#d4a017",
          display: "flex", alignItems: "center", gap: 4,
        }}>
          <Users style={{ width: 12, height: 12 }} />
          Demo Mode
        </Badge>
        {isStreaming ? (
          <Badge variant="outline" style={{
            fontSize: 10, borderColor: "#4ade8040", color: "#4ade80",
            display: "flex", alignItems: "center", gap: 4,
          }}>
            <Radio style={{ width: 12, height: 12 }} />
            Muse Connected
          </Badge>
        ) : (
          <Badge variant="outline" style={{
            fontSize: 10, borderColor: "rgba(255,255,255,0.15)", color: "var(--muted-foreground)",
          }}>
            Using simulated EEG
          </Badge>
        )}
      </div>

      {/* Timer */}
      {isActive && (
        <div style={{
          textAlign: "center", marginBottom: 16,
          fontSize: 32, fontWeight: 300, color: "var(--foreground)",
          fontVariantNumeric: "tabular-nums",
          letterSpacing: "2px",
        }}>
          {formatTime(elapsed)}
        </div>
      )}

      {/* Main visualization: two circles with sync bridge */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "center",
        gap: 8, marginBottom: 20, minHeight: 180,
      }}>
        <PersonCircle label="You" bandPowers={person1} syncScore={syncResult.overallSync} />
        <SyncBridge syncResult={syncResult} />
        <PersonCircle label="Partner" bandPowers={person2} syncScore={syncResult.overallSync} />
      </div>

      {/* Sync message */}
      <div style={{
        textAlign: "center", marginBottom: 20,
        fontSize: 14, fontWeight: 500, color: "var(--foreground)",
        fontStyle: "italic",
        minHeight: 24,
        transition: "opacity 0.5s ease",
        opacity: isActive ? 1 : 0.5,
      }}>
        {isActive ? syncResult.message : "Start a session to begin synchronizing"}
      </div>

      {/* Band synchrony detail */}
      {isActive && (
        <div style={{ marginBottom: 20 }}>
          <BandSyncBars bandSync={syncResult.bandSync} />
        </div>
      )}

      {/* Start/Stop button */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
        <Button
          onClick={handleToggle}
          size="lg"
          style={{
            borderRadius: 24, paddingLeft: 32, paddingRight: 32,
            background: isActive
              ? "linear-gradient(135deg, #e879a8 0%, #f43f5e 100%)"
              : "linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)",
            border: "none",
            fontSize: 14, fontWeight: 600,
            display: "flex", alignItems: "center", gap: 8,
          }}
        >
          {isActive ? (
            <><Square style={{ width: 16, height: 16 }} /> End Session</>
          ) : (
            <><Play style={{ width: 16, height: 16 }} /> Begin Together</>
          )}
        </Button>
      </div>

      {/* Science note */}
      <div style={{
        background: "var(--card)",
        border: "1px solid rgba(255,255,255,0.06)",
        borderRadius: 16, padding: "14px 16px",
        marginBottom: 16,
      }}>
        <div style={{
          fontSize: 10, fontWeight: 700, color: "var(--muted-foreground)",
          textTransform: "uppercase", letterSpacing: "0.6px", marginBottom: 6,
        }}>
          The science
        </div>
        <p style={{
          fontSize: 11, color: "var(--muted-foreground)",
          margin: 0, lineHeight: 1.6,
        }}>
          Inter-brain synchrony is measurable with consumer EEG. Romantic couples show the
          strongest neural coupling, especially in alpha (8-12 Hz) and theta (4-8 Hz) bands
          during shared attention. This demo uses power envelope correlation -- a simplified
          but valid hyperscanning metric.
        </p>
      </div>

      {/* Future work note */}
      <div style={{
        background: "rgba(212, 160, 23, 0.05)",
        border: "1px solid rgba(212, 160, 23, 0.15)",
        borderRadius: 12, padding: "10px 14px",
      }}>
        <p style={{
          fontSize: 10, color: "#d4a017", margin: 0, lineHeight: 1.5,
        }}>
          <strong>Demo mode:</strong> Partner data is simulated. Real dual-device support
          requires a WebSocket room for two Muse headbands streaming to the same session.
          This is planned for a future release.
        </p>
      </div>
    </motion.main>
  );
}
