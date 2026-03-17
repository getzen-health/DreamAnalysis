import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { useAuth } from "@/hooks/use-auth";
import { useTheme } from "@/hooks/use-theme";
import { getParticipantId } from "@/lib/participant";
import { useQuery } from "@tanstack/react-query";
import { resolveUrl } from "@/lib/queryClient";

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
        color: connected ? "#2dd4a0" : "var(--muted-foreground)",
        background: connected ? "rgba(45,212,160,0.12)" : "rgba(139,133,120,0.12)",
        border: `1px solid ${connected ? "rgba(45,212,160,0.3)" : "rgba(139,133,120,0.3)"}`,
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
  emoji,
  title,
  rightText,
  rightBadge,
  onClick,
  isLast,
}: {
  emoji: string;
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
      }}
    >
      <span style={{ fontSize: 18, marginRight: 10 }}>{emoji}</span>
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
        borderRadius: 14,
        border: "1px solid var(--border)",
        overflow: "hidden",
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
  const { theme, setTheme } = useTheme();
  const [, setLocation] = useLocation();
  const userId = getParticipantId();

  const [platform] = useState(() => getPlatform());
  const [healthConnected] = useState(() => getHealthConnectStatus());
  const [museConnected] = useState(() => getMuseStatus());

  const { data: streakData } = useQuery<StreakStatus>({
    queryKey: ["/api/streaks/status", userId],
    queryFn: async () => {
      try {
        const res = await fetch(resolveUrl(`/api/streaks/status/${userId}`));
        if (!res.ok) return { current_streak: 0, longest_streak: 0 };
        return res.json();
      } catch {
        return { current_streak: 0, longest_streak: 0 };
      }
    },
    retry: false,
  });

  const { data: sessionData } = useQuery<SessionCount>({
    queryKey: ["/api/sessions/count", userId],
    queryFn: async () => {
      try {
        const res = await fetch(resolveUrl(`/api/sessions/count/${userId}`));
        if (!res.ok) return { count: 0 };
        return res.json();
      } catch {
        return { count: 0 };
      }
    },
    retry: false,
  });

  const streak = streakData?.current_streak ?? 0;
  const sessions = sessionData?.count ?? 0;

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
    <main
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
            background: "linear-gradient(135deg, #2dd4a0, #059669)",
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
            borderRadius: 14,
            border: "1px solid var(--border)",
            padding: 14,
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: 24, fontWeight: 700, color: "#f59e0b" }}>{streak}</div>
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 3 }}>Day Streak 🔥</div>
        </div>

        {/* Sessions */}
        <div
          style={{
            background: "var(--card)",
            borderRadius: 14,
            border: "1px solid var(--border)",
            padding: 14,
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: 24, fontWeight: 700, color: "#2dd4a0" }}>{sessions}</div>
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 3 }}>Sessions Total</div>
        </div>
      </div>

      {/* Activity Section */}
      <SectionLabel>Activity</SectionLabel>
      <GroupedList>
        <ListItem
          emoji="📅"
          title="Session History"
          onClick={() => setLocation("/sessions")}
        />
        <ListItem
          emoji="🏆"
          title="Personal Records"
          onClick={() => setLocation("/records")}
        />
        <ListItem
          emoji="📊"
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
            emoji="❤️"
            title="Google Health Connect"
            rightBadge={<StatusBadge connected={healthConnected} />}
            onClick={() => setLocation("/settings")}
          />
        )}
        {platform === "ios" && (
          <ListItem
            emoji="❤️"
            title="Apple Health"
            rightBadge={<StatusBadge connected={healthConnected} />}
            onClick={() => setLocation("/settings")}
          />
        )}
        {platform === "web" && (
          <ListItem
            emoji="❤️"
            title="Health Connect"
            rightBadge={<StatusBadge connected={healthConnected} />}
            onClick={() => setLocation("/settings")}
          />
        )}
        <ListItem
          emoji="🧠"
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
          emoji="🎨"
          title="Appearance"
          rightText={theme === "dark" ? "Dark" : "Light"}
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        />
        <ListItem
          emoji="🔔"
          title="Notifications"
          onClick={() => setLocation("/settings")}
        />
        <ListItem
          emoji="📤"
          title="Export Data"
          onClick={handleExportData}
        />
        <ListItem
          emoji="🔒"
          title="Privacy & Data"
          onClick={() => setLocation("/privacy")}
        />
        <ListItem
          emoji="❓"
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
          color: "#f87171",
          fontSize: 14,
          fontWeight: 500,
          cursor: "pointer",
          marginTop: 4,
        }}
      >
        Sign Out
      </button>
    </main>
  );
}
