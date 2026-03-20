import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { useAuth } from "@/hooks/use-auth";
import { useTheme } from "@/hooks/use-theme";
import { getParticipantId } from "@/lib/participant";
import { useQuery } from "@tanstack/react-query";
import { resolveUrl } from "@/lib/queryClient";
import { AchievementBadges } from "@/components/achievements";
import {
  Flame, Calendar, Trophy, BarChart3, Heart, Brain, Palette,
  Bell, Share, Lock, HelpCircle, type LucideIcon,
} from "lucide-react";

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

  // Streak from localStorage (updated by bottom-tabs on every voice check-in)
  const streak = (() => {
    try { return parseInt(localStorage.getItem("ndw_streak_count") || "0", 10); } catch { return 0; }
  })();

  // Session count from brain history API (each voice analysis = 1 session)
  const { data: historyData } = useQuery<Array<{ timestamp: string }>>({
    queryKey: [`/api/brain/history/${userId}?days=30`],
    retry: false,
    staleTime: 60_000,
  });
  const sessions = Array.isArray(historyData) ? historyData.length : 0;

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
    // Navigate to export page if it exists, otherwise show a toast-style alert
    setLocation("/sessions");
  }

  return (
    <motion.main
      initial={pageTransition.initial}
      animate={pageTransition.animate}
      transition={pageTransition.transition}
      style={{
        background: "var(--background)",
        minHeight: "100vh",
        padding: 16,
        paddingBottom: 100,
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

      {/* Connected Section */}
      <SectionLabel>Connected</SectionLabel>
      <GroupedList>
        {platform === "android" && (
          <ListItem
            icon={Heart}
            iconColor="#e879a8"
            title="Google Health Connect"
            rightBadge={<StatusBadge connected={healthConnected} />}
            onClick={() => setLocation("/settings")}
          />
        )}
        {platform === "ios" && (
          <ListItem
            icon={Heart}
            iconColor="#e879a8"
            title="Apple Health"
            rightBadge={<StatusBadge connected={healthConnected} />}
            onClick={() => setLocation("/settings")}
          />
        )}
        {platform === "web" && (
          <ListItem
            icon={Heart}
            iconColor="#e879a8"
            title="Health Connect"
            rightBadge={<StatusBadge connected={healthConnected} />}
            onClick={() => setLocation("/settings")}
          />
        )}
        <ListItem
          icon={Brain}
          iconColor="#6366f1"
          title="Muse 2 EEG"
          rightBadge={<StatusBadge connected={museConnected} />}
          onClick={() => setLocation("/device-setup")}
          isLast
        />
      </GroupedList>

      {/* Settings Section */}
      <SectionLabel>Settings</SectionLabel>
      <GroupedList>
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
          onClick={() => setLocation("/settings")}
        />
        <ListItem
          icon={Share}
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
          onClick={() => setLocation("/settings")}
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
    </motion.main>
  );
}
