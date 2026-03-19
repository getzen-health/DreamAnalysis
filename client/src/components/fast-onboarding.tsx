/**
 * Fast-track onboarding — 3 screens to first value in under 60 seconds.
 *
 * Screen 1 (intro):   What AntarAI does — 3 value props
 * Screen 2 (checkin): Voice check-in (15 seconds via VoiceCheckinCard)
 * Screen 3 (done):    Result + "You're all set"
 *
 * On completion, writes localStorage keys that the full 5-step flow
 * already treats as "already done":
 *   ndw_onboarding_complete = "true"
 *   ndw_onboarding_step     = removed (or "5")
 */

import { useState } from "react";
import { Check, ChevronRight, Brain, Heart, Mic } from "lucide-react";
import { VoiceCheckinCard } from "@/components/voice-checkin-card";
import type { VoiceWatchCheckinResult } from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";
import { hapticLight, hapticSuccess } from "@/lib/haptics";

type Screen = "intro" | "checkin" | "done";

const VALUE_PROPS = [
  {
    icon: Mic,
    title: "10-Second Voice Check-in",
    description:
      "Speak for 10 seconds — AI detects your emotional state from your voice",
    color: "hsl(200 70% 55%)",
  },
  {
    icon: Brain,
    title: "Personalized Insights",
    description:
      "Track stress, focus, mood patterns and get actionable suggestions",
    color: "hsl(270 40% 60%)",
  },
  {
    icon: Heart,
    title: "Holistic Wellness",
    description:
      "Connect health data, log meals, journal dreams — all in one place",
    color: "hsl(0 65% 55%)",
  },
];

/** Mark onboarding complete in localStorage (compatible with the 5-step flow). */
function markFastComplete() {
  try {
    localStorage.setItem("ndw_onboarding_complete", "true");
    localStorage.removeItem("ndw_onboarding_step");
    localStorage.removeItem("ndw_onboarding_path");
  } catch {
    // ignore quota errors
  }
}

interface Props {
  onComplete: () => void;
}

export function FastOnboarding({ onComplete }: Props) {
  const [screen, setScreen] = useState<Screen>("intro");
  const [emotion, setEmotion] = useState<string | null>(null);
  const userId = getParticipantId();

  function handleCheckinComplete(result: VoiceWatchCheckinResult) {
    setEmotion(result.emotion ?? null);
    hapticSuccess();
    setScreen("done");
  }

  function handleFinish() {
    markFastComplete();
    onComplete();
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        padding: "24px 20px",
        background: "var(--background)",
      }}
    >
      {/* Screen 1: Value Proposition */}
      {screen === "intro" && (
        <div style={{ animation: "fadeIn 0.3s ease-out" }}>
          <div
            style={{
              textAlign: "center",
              marginBottom: 32,
            }}
          >
            <div
              style={{
                fontSize: 28,
                fontWeight: 700,
                color: "var(--foreground)",
                marginBottom: 6,
              }}
            >
              Welcome to AntarAI
            </div>
            <div style={{ fontSize: 13, color: "var(--muted-foreground)" }}>
              Your AI-powered wellness companion
            </div>
          </div>

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 12,
              marginBottom: 32,
            }}
          >
            {VALUE_PROPS.map((prop) => {
              const Icon = prop.icon;
              return (
                <div
                  key={prop.title}
                  style={{
                    background: "var(--card)",
                    border: "1px solid var(--border)",
                    borderRadius: 14,
                    padding: "14px 16px",
                    display: "flex",
                    alignItems: "center",
                    gap: 14,
                  }}
                >
                  <div
                    style={{
                      width: 40,
                      height: 40,
                      borderRadius: 10,
                      background: prop.color.replace(")", " / 0.15)"),
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      flexShrink: 0,
                    }}
                  >
                    <Icon style={{ width: 20, height: 20, color: prop.color }} />
                  </div>
                  <div>
                    <div
                      style={{
                        fontSize: 13,
                        fontWeight: 600,
                        color: "var(--foreground)",
                      }}
                    >
                      {prop.title}
                    </div>
                    <div
                      style={{
                        fontSize: 11,
                        color: "var(--muted-foreground)",
                        marginTop: 2,
                        lineHeight: 1.3,
                      }}
                    >
                      {prop.description}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <button
            onClick={() => {
              hapticLight();
              setScreen("checkin");
            }}
            style={{
              width: "100%",
              padding: "14px",
              borderRadius: 12,
              background: "hsl(165 60% 45%)",
              color: "white",
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
            Try it now{" "}
            <ChevronRight style={{ width: 18, height: 18 }} />
          </button>

          <button
            onClick={handleFinish}
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
            Skip for now
          </button>
        </div>
      )}

      {/* Screen 2: Voice Check-in */}
      {screen === "checkin" && (
        <div style={{ animation: "fadeIn 0.3s ease-out" }}>
          <div
            style={{
              textAlign: "center",
              marginBottom: 16,
            }}
          >
            <div
              style={{
                fontSize: 18,
                fontWeight: 600,
                color: "var(--foreground)",
                marginBottom: 4,
              }}
            >
              Let's try it
            </div>
            <div style={{ fontSize: 12, color: "var(--muted-foreground)" }}>
              Speak for 15 seconds about anything — your day, how you feel, what you had for lunch
            </div>
          </div>
          <VoiceCheckinCard
            userId={userId}
            onComplete={handleCheckinComplete}
            forceShow
          />
          <button
            onClick={handleFinish}
            style={{
              width: "100%",
              padding: "12px",
              marginTop: 12,
              background: "none",
              border: "none",
              color: "var(--muted-foreground)",
              fontSize: 13,
              cursor: "pointer",
            }}
          >
            Skip for now
          </button>
        </div>
      )}

      {/* Screen 3: Done */}
      {screen === "done" && (
        <div
          style={{
            animation: "fadeIn 0.3s ease-out",
            textAlign: "center",
          }}
        >
          <div
            style={{
              width: 60,
              height: 60,
              borderRadius: "50%",
              background: "hsl(165 60% 45% / 0.15)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              margin: "0 auto 16px",
            }}
          >
            <Check
              style={{ width: 28, height: 28, color: "hsl(165 60% 45%)" }}
            />
          </div>
          <div
            style={{
              fontSize: 22,
              fontWeight: 700,
              color: "var(--foreground)",
              marginBottom: 6,
            }}
          >
            You're all set!
          </div>
          {emotion && (
            <div
              style={{
                fontSize: 14,
                color: "var(--muted-foreground)",
                marginBottom: 20,
              }}
            >
              We detected you're feeling{" "}
              <span
                style={{
                  color: "var(--foreground)",
                  fontWeight: 600,
                  textTransform: "capitalize",
                }}
              >
                {emotion}
              </span>
            </div>
          )}
          {!emotion && (
            <div
              style={{
                fontSize: 14,
                color: "var(--muted-foreground)",
                marginBottom: 20,
              }}
            >
              Your first check-in is complete.
            </div>
          )}
          <div
            style={{
              fontSize: 12,
              color: "var(--muted-foreground)",
              marginBottom: 24,
              lineHeight: 1.5,
            }}
          >
            Check in daily to track your emotional patterns. You can connect
            health data and explore more features anytime.
          </div>
          <button
            onClick={handleFinish}
            style={{
              width: "100%",
              padding: "14px",
              borderRadius: 12,
              background: "hsl(165 60% 45%)",
              color: "white",
              fontSize: 15,
              fontWeight: 600,
              border: "none",
              cursor: "pointer",
            }}
          >
            Start Exploring
          </button>
        </div>
      )}
    </div>
  );
}
