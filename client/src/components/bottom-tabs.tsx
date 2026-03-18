import { Link, useLocation } from "wouter";
import { useState, useEffect } from "react";
import {
  Sun,
  Compass,
  UtensilsCrossed,
  CircleUser,
  Mic,
} from "lucide-react";
import { hapticLight, hapticMedium } from "@/lib/haptics";
import { useQueryClient } from "@tanstack/react-query";
import { VoiceCheckinCard } from "@/components/voice-checkin-card";
import { getParticipantId } from "@/lib/participant";

const tabs = [
  { path: "/",          icon: Sun,              label: "Today",     aliases: [] as string[] },
  { path: "/discover",  icon: Compass,          label: "Discover",  aliases: ["/inner-energy", "/ai-companion", "/brain-monitor", "/dreams", "/neurofeedback", "/biofeedback", "/sleep-session", "/insights", "/weekly-summary", "/emotions", "/body-metrics", "/workout", "/habits", "/wellness"] },
  // mic button stays in center (already implemented)
  { path: "/nutrition",  icon: UtensilsCrossed, label: "Nutrition", aliases: ["/food", "/food-log", "/food-emotion"] },
  { path: "/you",        icon: CircleUser,      label: "You",       aliases: ["/settings", "/profile", "/sessions", "/records"] },
];

const LEFT_TABS = tabs.slice(0, 2);
const RIGHT_TABS = tabs.slice(2);

const USER_ID = getParticipantId();

export function BottomTabs() {
  const [location] = useLocation();
  const [showCheckin, setShowCheckin] = useState(false);
  const queryClient = useQueryClient();

  // Allow other components (e.g. EmotionBadge) to open the voice analysis modal
  useEffect(() => {
    const openHandler = () => setShowCheckin(true);
    window.addEventListener("ndw-open-voice-checkin", openHandler);
    return () => window.removeEventListener("ndw-open-voice-checkin", openHandler);
  }, []);

  function handleCheckinComplete() {
    setShowCheckin(false);

    // Update streak counter in localStorage for badges
    try {
      const today = new Date().toISOString().slice(0, 10);
      const lastDate = localStorage.getItem("ndw_streak_last_date");
      const currentStreak = parseInt(localStorage.getItem("ndw_streak_count") || "0", 10);

      if (lastDate === today) {
        // Already checked in today — streak unchanged
      } else {
        const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
        if (lastDate === yesterday) {
          // Consecutive day — increment streak
          localStorage.setItem("ndw_streak_count", String(currentStreak + 1));
        } else {
          // Streak broken — reset to 1
          localStorage.setItem("ndw_streak_count", "1");
        }
        localStorage.setItem("ndw_streak_last_date", today);
      }
    } catch { /* ignore */ }

    // Invalidate everything so all pages reflect the new voice data
    const uid = USER_ID;
    queryClient.invalidateQueries({ queryKey: ["voice-checkins"] });
    queryClient.invalidateQueries({ queryKey: ["streak-status"] });
    queryClient.invalidateQueries({ queryKey: ["sessions"] });
    queryClient.invalidateQueries({ queryKey: ["emotions"] });
    queryClient.invalidateQueries({ queryKey: [`/api/brain/history/${uid}?days=1`] });
    queryClient.invalidateQueries({ queryKey: [`/api/brain/history/${uid}?days=7`] });
    queryClient.invalidateQueries({ queryKey: ["voice-inner-energy", uid] });
    queryClient.invalidateQueries({ queryKey: ["health-brain-report"] });
    // Force Today page to re-read localStorage on next render
    window.dispatchEvent(new Event("ndw-voice-updated"));
  }

  function renderTab(tab: (typeof tabs)[number]) {
    const Icon = tab.icon;
    const isActive =
      location === tab.path ||
      (tab.path !== "/" && location.startsWith(tab.path)) ||
      tab.aliases.some((alias) => location === alias || location.startsWith(alias + "/"));

    return (
      <Link
        key={tab.path}
        href={tab.path}
        onClick={() => hapticLight()}
        aria-current={isActive ? "page" : undefined}
        aria-label={tab.label}
        className="relative flex flex-col items-center justify-center gap-0.5 flex-1 py-1.5 transition-all active:scale-95 focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background rounded-md"
      >
        {/* Active indicator pill above icon */}
        {isActive && (
          <span
            className="absolute top-0 left-1/2 -translate-x-1/2 rounded-full bg-primary"
            style={{ width: 24, height: 2, borderRadius: 1 }}
          />
        )}

        <Icon
          className={`transition-colors ${
            isActive
              ? "text-primary"
              : "text-muted-foreground/55"
          }`}
          style={{ width: 24, height: 24 }}
          aria-hidden="true"
          strokeWidth={isActive ? 2.5 : 1.75}
          fill={isActive ? "currentColor" : "none"}
        />
        <span
          className={`text-[10px] leading-none tracking-tight transition-colors ${
            isActive
              ? "text-primary font-semibold"
              : "text-muted-foreground/55 font-normal"
          }`}
        >
          {tab.label}
        </span>
      </Link>
    );
  }

  return (
    <>
      <nav
        role="navigation"
        aria-label="Main navigation"
        className="fixed bottom-0 left-0 right-0 z-40 md:hidden border-t bg-background/95 border-border/60"
        style={{
          backdropFilter: "blur(24px)",
          WebkitBackdropFilter: "blur(24px)",
          paddingBottom: "env(safe-area-inset-bottom, 0px)",
        }}
      >
        <div className="flex items-stretch justify-around" style={{ height: "56px" }}>
          {/* Left two tabs */}
          {LEFT_TABS.map(renderTab)}

          {/* Center mic button */}
          <div className="relative flex flex-col items-center justify-center flex-1">
            <button
              aria-label="Voice analysis"
              onClick={() => {
                hapticMedium();
                setShowCheckin(true);
              }}
              className="absolute -top-5 flex items-center justify-center rounded-full bg-primary text-primary-foreground shadow-lg active:scale-95 transition-all focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background"
              style={{ width: 44, height: 44 }}
            >
              <Mic style={{ width: 20, height: 20 }} strokeWidth={2} aria-hidden="true" />
            </button>
          </div>

          {/* Right two tabs */}
          {RIGHT_TABS.map(renderTab)}
        </div>
      </nav>

      {/* Voice analysis modal */}
      {showCheckin && (
        <div
          className="fixed inset-0 z-50 flex items-end justify-center bg-black/40"
          onClick={() => setShowCheckin(false)}
        >
          <div
            className="w-full max-w-lg bg-background rounded-t-2xl p-4 pb-safe"
            style={{ paddingBottom: "max(1rem, env(safe-area-inset-bottom, 1rem))" }}
            onClick={(e) => e.stopPropagation()}
          >
            <VoiceCheckinCard userId={USER_ID} onComplete={handleCheckinComplete} forceShow />
          </div>
        </div>
      )}
    </>
  );
}
