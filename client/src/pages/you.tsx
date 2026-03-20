import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { useAuth } from "@/hooks/use-auth";
import { useTheme } from "@/hooks/use-theme";
import { getParticipantId } from "@/lib/participant";
import { useQuery } from "@tanstack/react-query";
import { resolveUrl } from "@/lib/queryClient";
import { listSessions, type SessionSummary } from "@/lib/ml-api";
import { AchievementBadges } from "@/components/achievements";
import {
  Flame, Calendar, Trophy, BarChart3, Heart, Brain, Palette,
  Bell, Download, Lock, HelpCircle, Watch, Sun, type LucideIcon,
} from "lucide-react";
import { ChronotypeQuiz } from "@/components/chronotype-quiz";
import { getStoredChronotype, type ChronotypeCategory } from "@/lib/chronotype";
import { BrainAgeCard } from "@/components/brain-age-card";
import { NotificationPrefsSheet } from "@/components/notification-prefs-sheet";

// ── Types ─────────────────────────────────────────────────────────────────────

interface StreakStatus {
  current_streak: number;
  longest_streak: number;
}

interface SessionCount {
  count: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function getPlatform(): "ios" | "android" | "web" {
  try {
    const cap = (window as unknown as { Capacitor?: { getPlatform?: () => string } }).Capacitor;
    const plat = cap?.getPlatform?.();
    if (plat === "ios") return "ios";
    if (plat === "android") return "android";
  } catch {
    // ignore
  }
  return "web";
}

function getHealthConnectStatus(): boolean {
  try {
    return localStorage.getItem("ndw_health_connect_granted") === "true";
  } catch {
    return false;
  }
}

function getMuseStatus(): boolean {
  try {
    const raw = localStorage.getItem("ndw_muse_connected");
    return raw === "true";
  } catch {
    return false;
  }
}

function getMemberSince(createdAt?: string): string {
  if (!createdAt) return "Mar 2026";
  const d = new Date(createdAt);
  return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
}

// ── Status Badge ──────────────────────────────────────────────────────────────

function StatusBadge({ connected }: { connected: boolean }) {
  return (
    <span
      style={{
        fontSize: 10,
        fontWeight: 600,
        color: connected ? "#0891b2" : "var(--muted-foreground)",
        background: connected ? "rgba(8,145,178,0.12)" : "rgba(139,133,120,0.12)",
        border: `1px solid ${connected ? "rgba(8,145,178,0.3)" : "rgba(139,133,120,0.3)"}`,
        borderRadius: 20,
        padding: "2px 8px",
        marginRight: 6,
        flexShrink: 0,
      }}
    >
      {connected ? "Connected" : "Not connected"}
    </span>
  );
}

// ── List Item ─────────────────────────────────────────────────────────────────

function ListItem({
  icon: Icon,
  iconColor,
  title,
  rightText,
  rightBadge,
  onClick,
  isLast,
}: {
  icon: LucideIcon;
  iconColor?: string;
  title: string;
  rightText?: string;
  rightBadge?: React.ReactNode;
  onClick?: () => void;
  isLast?: boolean;
}) {
  return (
    <div
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        padding: "13px 14px",
        borderBottom: isLast ? "none" : "1px solid var(--border)",
        cursor: onClick ? "pointer" : "default",
        transition: onClick ? "background 0.2s ease" : "none",
      }}
    >
      <Icon style={{ width: 18, height: 18, marginRight: 10, color: iconColor ?? "var(--muted-foreground)", flexShrink: 0 }} />
      <div style={{ flex: 1, fontSize: 14, color: "var(--foreground)" }}>{title}</div>
      {rightBadge}
      {rightText && (
        <span style={{ fontSize: 11, color: "var(--muted-foreground)", marginRight: 6 }}>{rightText}</span>
      )}
      {onClick && <span style={{ color: "var(--muted-foreground)", fontSize: 16 }}>›</span>}
    </div>
  );
}

// ── Section Label ─────────────────────────────────────────────────────────────

function SectionLabel({ children }: { children: string }) {
  return (
    <div
      style={{
        fontSize: 11,
        fontWeight: 600,
        color: "var(--muted-foreground)",
        textTransform: "uppercase",
        letterSpacing: "0.5px",
        marginBottom: 8,
      }}
    >
      {children}
    </div>
  );
}

// ── Grouped List Container ────────────────────────────────────────────────────

function GroupedList({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        background: "var(--card)",
        borderRadius: 20,
        border: "1px solid var(--border)",
        overflow: "hidden",
        boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
        marginBottom: 14,
      }}
    >
      {children}
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function You() {
  const { user, logout } = useAuth();
  const { theme, themeSetting, setTheme } = useTheme();
  const [, setLocation] = useLocation();
  const userId = getParticipantId();

  const [platform] = useState(() => getPlatform());
  const [healthConnected] = useState(() => getHealthConnectStatus());
  const [museConnected] = useState(() => getMuseStatus());
  const [showChronotypeQuiz, setShowChronotypeQuiz] = useState(false);
  const [showNotificationPrefs, setShowNotificationPrefs] = useState(false);
  const [chronotype, setChronotype] = useState<ChronotypeCategory | null>(
    () => getStoredChronotype()?.category ?? null,
  );

  // Sessions from ML backend (Railway) — where voice check-in data actually lives
  const { data: sessionList } = useQuery<SessionSummary[]>({
    queryKey: ["sessions", userId],
    queryFn: () => listSessions(userId),
    staleTime: 30_000,
    retry: false,
  });
  const sessions = Array.isArray(sessionList) ? sessionList.length : 0;

  // Streak: count consecutive days with at least one session
  const streak = (() => {
    if (!sessionList || sessionList.length === 0) return 0;
    const daySet = new Set<string>();
    for (const s of sessionList) {
      if (s.start_time) {
        const d = new Date(s.start_time * 1000);
        daySet.add(`${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`);
      }
    }
    // Count consecutive days backwards from today
    let count = 0;
    const now = new Date();
    for (let i = 0; i < 365; i++) {
      const d = new Date(now);
      d.setDate(d.getDate() - i);
      const key = `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
      if (daySet.has(key)) {
        count++;
      } else if (i > 0) {
        break; // streak broken
      }
    }
    return count;
  })();

  const displayName = user?.username ?? "Dreamer";
  const initial = displayName.charAt(0).toUpperCase();
  const memberSince = getMemberSince(user?.createdAt);

  async function handleSignOut() {
    try {
      await logout();
    } catch {
      // Clear manually on failure
      try { localStorage.removeItem("auth_token"); } catch { /* ok */ }
    }
    setLocation("/welcome");
  }

  function handleExportData() {
    setLocation("/export");
  }

  return (
    <motion.main
      initial={pageTransition.initial}
      animate={pageTransition.animate}
      transition={pageTransition.transition}
      style={{
        background: "var(--background)",
        padding: 16,
        paddingBottom: 16,
        color: "var(--foreground)",
        fontFamily: "Inter, system-ui, sans-serif",
      }}
    >
      {/* Profile Header */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          paddingTop: 8,
          marginBottom: 20,
        }}
      >
        {/* Avatar */}
        <div
          style={{
            width: 72,
            height: 72,
            borderRadius: "50%",
            background: "linear-gradient(135deg, #0891b2, #0e7490)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: 10,
            flexShrink: 0,
          }}
        >
          <span style={{ fontSize: 28, fontWeight: 700, color: "#0a0e17" }}>{initial}</span>
        </div>

        {/* Name */}
        <div style={{ fontSize: 18, fontWeight: 600, color: "var(--foreground)", marginBottom: 4 }}>
          {displayName}
        </div>

        {/* Member since */}
        <div style={{ fontSize: 12, color: "var(--muted-foreground)" }}>Member since {memberSince}</div>
      </div>

      {/* Stats Row */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 10,
          marginBottom: 16,
        }}
      >
        {/* Streak */}
        <div
          style={{
            background: "var(--card)",
            borderRadius: 20,
            border: "1px solid var(--border)",
            padding: 14,
            textAlign: "center",
            boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
          }}
        >
          <div style={{ fontSize: 28, fontWeight: 700, color: "#d4a017" }}>{streak}</div>
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 3, display: "flex", alignItems: "center", justifyContent: "center", gap: 4 }}>Day Streak <Flame style={{ width: 12, height: 12, color: "#d4a017" }} /></div>
        </div>

        {/* Sessions */}
        <div
          style={{
            background: "var(--card)",
            borderRadius: 20,
            border: "1px solid var(--border)",
            padding: 14,
            textAlign: "center",
            boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
          }}
        >
          <div style={{ fontSize: 28, fontWeight: 700, color: "#0891b2" }}>{sessions}</div>
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 3 }}>Sessions Total</div>
        </div>
      </div>

      {/* Achievements */}
      <AchievementBadges />

      {/* Brain Health Section */}
      <SectionLabel>Brain Health</SectionLabel>
      <BrainAgeCard />

      {/* Activity Section */}
      <SectionLabel>Activity</SectionLabel>
      <GroupedList>
        <ListItem
          icon={Calendar}
          iconColor="#0891b2"
          title="Session History"
          onClick={() => setLocation("/sessions")}
        />
        <ListItem
          icon={Trophy}
          iconColor="#d4a017"
          title="Personal Records"
          onClick={() => setLocation("/records")}
        />
        <ListItem
          icon={BarChart3}
          iconColor="#6366f1"
          title="Weekly Summary"
          onClick={() => setLocation("/weekly-summary")}
          isLast
        />
      </GroupedList>

      {/* Connected Assets */}
      <SectionLabel>Connected Assets</SectionLabel>
      <GroupedList>
        <ListItem
          icon={Heart}
          iconColor="#e879a8"
          title={platform === "android" ? "Google Health Connect" : platform === "ios" ? "Apple HealthKit" : "Health Connect"}
          rightBadge={<StatusBadge connected={healthConnected} />}
          onClick={() => setLocation("/connected-assets")}
        />
        <ListItem
          icon={Brain}
          iconColor="#6366f1"
          title="BCI / EEG"
          rightBadge={<StatusBadge connected={museConnected} />}
          onClick={() => setLocation("/connected-assets")}
        />
        <ListItem
          icon={Watch}
          iconColor="#d4a017"
          title="Wearables"
          onClick={() => setLocation("/connected-assets")}
          isLast
        />
      </GroupedList>

      {/* Settings Section */}
      <SectionLabel>Settings</SectionLabel>
      <GroupedList>
        <ListItem
          icon={Sun}
          iconColor="#d4a017"
          title="Chronotype"
          rightText={chronotype ? chronotype.charAt(0).toUpperCase() + chronotype.slice(1) : "Not set"}
          onClick={() => setShowChronotypeQuiz(true)}
        />
        <ListItem
          icon={Palette}
          iconColor="#a78bfa"
          title="Appearance"
          rightText={themeSetting === "auto" ? "Auto" : themeSetting === "dark" ? "Dark" : "Light"}
          onClick={() => {
            // Cycle: dark → light → auto → dark
            const next = themeSetting === "dark" ? "light" : themeSetting === "light" ? "auto" : "dark";
            setTheme(next);
          }}
        />
        <ListItem
          icon={Bell}
          iconColor="#d4a017"
          title="Notifications"
          onClick={() => setShowNotificationPrefs(true)}
        />
        <ListItem
          icon={Download}
          iconColor="#0891b2"
          title="Export Data"
          onClick={handleExportData}
        />
        <ListItem
          icon={Lock}
          iconColor="#e879a8"
          title="Privacy & Data"
          onClick={() => setLocation("/privacy")}
        />
        <ListItem
          icon={HelpCircle}
          iconColor="#94a3b8"
          title="Help & Feedback"
          onClick={() => setLocation("/help")}
          isLast
        />
      </GroupedList>

      {/* Sign Out */}
      <button
        onClick={handleSignOut}
        style={{
          width: "100%",
          background: "transparent",
          border: "1px solid var(--border)",
          borderRadius: 12,
          padding: 12,
          color: "#e879a8",
          fontSize: 14,
          fontWeight: 500,
          cursor: "pointer",
          transition: "transform 0.2s ease, box-shadow 0.2s ease",
          marginTop: 4,
        }}
      >
        Sign Out
      </button>

      {/* Chronotype Quiz Overlay */}
      {showChronotypeQuiz && (
        <ChronotypeQuiz
          onComplete={(category) => {
            setChronotype(category);
            setShowChronotypeQuiz(false);
          }}
          onClose={() => setShowChronotypeQuiz(false)}
        />
      )}

      {/* Notification Preferences Sheet */}
      <NotificationPrefsSheet
        open={showNotificationPrefs}
        onOpenChange={setShowNotificationPrefs}
      />
    </motion.main>
  );
}
