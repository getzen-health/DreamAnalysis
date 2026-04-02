import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import { useAuth } from "@/hooks/use-auth";
import { useTheme } from "@/hooks/use-theme";
import { getParticipantId } from "@/lib/participant";
import { useQuery } from "@tanstack/react-query";
import { listSessions, type SessionSummary } from "@/lib/ml-api";
import {
  Flame, Calendar, Trophy, BarChart3, Heart, Brain, Palette,
  Bell, Download, Lock, HelpCircle, Watch, Sun, ChevronRight, Link2, type LucideIcon,
} from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { ChronotypeQuiz } from "@/components/chronotype-quiz";
import { getStoredChronotype, type ChronotypeCategory } from "@/lib/chronotype";
import { loadPersonalAdapter, resetPersonalAdapter, getPersonalizationStats } from "@/lib/personal-adapter";
import { getModalityAccuracies, type ModalityAccuracy } from "@/lib/multimodal-fusion";
import { NotificationPrefsSheet } from "@/components/notification-prefs-sheet";
import { Zap } from "lucide-react";
import { getCorrectionCount, getRetrainingStatus } from "@/lib/feedback-sync";
import { InterventionTriggerSettings } from "@/components/intervention-trigger-settings";
import { sbGetSetting } from "../lib/supabase-store";

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
    return sbGetSetting("ndw_health_connect_granted") === "true";
  } catch {
    return false;
  }
}

function getMuseStatus(): boolean {
  try {
    const raw = sbGetSetting("ndw_muse_connected");
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

/** Total number of device slots on the connected-assets page. */
const TOTAL_DEVICE_SLOTS = 7;

/** Count how many devices are currently connected (from localStorage). */
function getConnectedDeviceCount(): number {
  let count = 0;
  try {
    if (sbGetSetting("ndw_health_connect_granted") === "true"
        || sbGetSetting("ndw_apple_health_granted") === "true") count++;
    if (sbGetSetting("ndw_muse_connected") === "true") count++;
    // Wearables: oura, whoop, garmin, fitbit, samsung
    for (const key of ["ndw_oura_connected", "ndw_whoop_connected", "ndw_garmin_connected",
                        "ndw_fitbit_connected", "ndw_samsung_connected"]) {
      if (sbGetSetting(key) === "true") count++;
    }
  } catch { /* ignore */ }
  return count;
}

// ── Status Badge ──────────────────────────────────────────────────────────────

function StatusBadge({ connected }: { connected: boolean }) {
  return (
    <span
      className={`rounded-full px-2 py-0.5 text-[10px] font-medium mr-1.5 shrink-0 border ${
        connected
          ? "text-cyan-600 bg-cyan-600/[0.12] border-cyan-600/30"
          : "text-muted-foreground bg-foreground/[0.06] border-foreground/[0.12]"
      }`}
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
      className={`flex items-center px-3.5 py-[13px] transition-colors ${
        onClick ? "cursor-pointer hover:bg-foreground/[0.03]" : "cursor-default"
      } ${isLast ? "" : "border-b border-foreground/[0.04]"}`}
    >
      <Icon className="w-[18px] h-[18px] mr-2.5 shrink-0" style={{ color: iconColor ?? "var(--muted-foreground)" }} />
      <div className="flex-1 text-sm text-foreground">{title}</div>
      {rightBadge}
      {rightText && (
        <span className="text-[11px] text-muted-foreground mr-1.5">{rightText}</span>
      )}
      {onClick && <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />}
    </div>
  );
}

// ── Section Label ─────────────────────────────────────────────────────────────

function SectionLabel({ children }: { children: string }) {
  return (
    <div className="text-[11px] font-semibold uppercase tracking-wider text-foreground/35 mb-2">
      {children}
    </div>
  );
}

// ── Grouped List Container ────────────────────────────────────────────────────

function GroupedList({ children }: { children: React.ReactNode }) {
  return (
    <div className="glass-card rounded-2xl overflow-hidden mb-3.5">
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
  const [connectedDeviceCount] = useState(() => getConnectedDeviceCount());
  const [showChronotypeQuiz, setShowChronotypeQuiz] = useState(false);
  const [showNotificationPrefs, setShowNotificationPrefs] = useState(false);

  // Personal adapter stats
  const [adapterStats, setAdapterStats] = useState(() => {
    const adapter = loadPersonalAdapter();
    return getPersonalizationStats(adapter);
  });
  const handleResetAdapter = () => {
    resetPersonalAdapter();
    const adapter = loadPersonalAdapter();
    setAdapterStats(getPersonalizationStats(adapter));
  };

  const [chronotype, setChronotype] = useState<ChronotypeCategory | null>(
    () => getStoredChronotype()?.category ?? null,
  );

  // Supabase correction count for training pipeline display
  const [supabaseCorrectionCount, setSupabaseCorrectionCount] = useState(0);
  useEffect(() => {
    getCorrectionCount(userId).then(setSupabaseCorrectionCount).catch(() => {});
  }, [userId]);

  // Retraining status from local tracking
  const retrainStatus = getRetrainingStatus();

  // Modality accuracy stats
  const [modalityAcc, setModalityAcc] = useState<ModalityAccuracy>(() => getModalityAccuracies());
  // Refresh when corrections count changes (proxy for new corrections)
  useEffect(() => {
    setModalityAcc(getModalityAccuracies());
  }, [supabaseCorrectionCount]);

  // Sessions from ML backend (Railway) — where voice check-in data actually lives
  const { data: sessionList, isLoading: sessionsLoading } = useQuery<SessionSummary[]>({
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
      className="bg-background p-4 pb-4 text-foreground font-[Inter,system-ui,sans-serif]"
    >
      {/* Profile Header */}
      <motion.div
        custom={0}
        initial="hidden"
        animate="visible"
        variants={cardVariants}
        className="flex flex-col items-center pt-2 mb-5"
      >
        {/* Avatar — 80px, gradient primary-to-secondary, glow shadow */}
        <div
          className="w-20 h-20 rounded-full flex items-center justify-center mb-2.5 shrink-0 shadow-[0_0_24px_rgba(8,145,178,0.35)]"
          style={{ background: "linear-gradient(135deg, #0891b2, #6366f1)" }}
        >
          <span className="text-[30px] font-bold text-white">{initial}</span>
        </div>

        {/* Name */}
        <div className="text-lg font-semibold text-foreground mb-1">
          {displayName}
        </div>

        {/* Member since */}
        <div className="text-xs text-muted-foreground">Member since {memberSince}</div>
      </motion.div>

      {/* Stats Row */}
      <motion.div
        custom={1}
        initial="hidden"
        animate="visible"
        variants={cardVariants}
        className="grid grid-cols-2 gap-2.5 mb-4"
      >
        {/* Streak */}
        <div className="glass-card rounded-2xl p-3.5 text-center">
          {sessionsLoading ? (
            <Skeleton className="h-8 w-10 rounded mx-auto mb-1" />
          ) : (
            <div className="text-[28px] font-bold bg-gradient-to-r from-yellow-500 to-amber-500 bg-clip-text text-transparent">{streak}</div>
          )}
          <div className="text-[11px] text-muted-foreground mt-0.5 flex items-center justify-center gap-1">Day Streak <Flame className="w-3 h-3 text-amber-500" /></div>
        </div>

        {/* Sessions */}
        <div className="glass-card rounded-2xl p-3.5 text-center">
          {sessionsLoading ? (
            <Skeleton className="h-8 w-10 rounded mx-auto mb-1" />
          ) : (
            <div className="text-[28px] font-bold bg-gradient-to-r from-cyan-500 to-cyan-600 bg-clip-text text-transparent">{sessions}</div>
          )}
          <div className="text-[11px] text-muted-foreground mt-0.5">Sessions Total</div>
        </div>
      </motion.div>

      {/* Achievements Link */}
      <motion.div
        custom={2}
        initial="hidden"
        animate="visible"
        variants={cardVariants}
        onClick={() => setLocation("/achievements")}
        className="glass-card rounded-2xl px-4 py-3.5 mb-4 cursor-pointer flex items-center gap-3 border-amber-500/20 hover:bg-foreground/[0.03] transition-colors"
        style={{ background: "linear-gradient(135deg, var(--card) 0%, rgba(212,160,23,0.05) 100%)" }}
      >
        <div className="w-9 h-9 rounded-[10px] bg-amber-500/[0.12] flex items-center justify-center shrink-0">
          <Trophy className="w-[18px] h-[18px] text-amber-500" />
        </div>
        <div className="flex-1">
          <div className="text-sm font-semibold text-foreground">
            Achievements
          </div>
          <div className="text-[11px] text-muted-foreground mt-0.5">
            View your badges and milestones
          </div>
        </div>
        <ChevronRight className="w-[18px] h-[18px] text-muted-foreground shrink-0" />
      </motion.div>

      {/* Brain Health Section */}
      <SectionLabel>Brain Health</SectionLabel>

      <motion.div
        custom={3}
        initial="hidden"
        animate="visible"
        variants={cardVariants}
      >
        {/* EEG Personalization Status */}
        <div className="glass-card rounded-xl px-4 py-2.5 mb-2 flex items-center justify-between">
          <div>
            <div className="text-[13px] font-medium text-foreground">
              Personalization: {adapterStats.confidenceLevel === "learning"
                ? `Learning (${adapterStats.sessionsProcessed} sessions)`
                : adapterStats.confidenceLevel === "calibrating"
                  ? `Calibrating (${adapterStats.sessionsProcessed} sessions)`
                  : `Personalized (${adapterStats.sessionsProcessed} sessions)`}
            </div>
            {adapterStats.correctionsApplied > 0 && (
              <div className="text-[11px] text-muted-foreground mt-0.5">
                {adapterStats.correctionsApplied} correction{adapterStats.correctionsApplied !== 1 ? "s" : ""} applied
              </div>
            )}
          </div>
          {adapterStats.sessionsProcessed > 0 && (
            <button
              onClick={handleResetAdapter}
              className="text-[11px] text-muted-foreground bg-transparent border border-border rounded-md px-2 py-1 cursor-pointer hover:bg-foreground/[0.03] transition-colors"
            >
              Reset
            </button>
          )}
        </div>

        {/* Training Pipeline Stats */}
        {supabaseCorrectionCount > 0 && (
          <div className="glass-card rounded-xl px-4 py-2.5 mb-2">
            <div className="text-[13px] font-medium text-foreground">
              {supabaseCorrectionCount} correction{supabaseCorrectionCount !== 1 ? "s" : ""} submitted
            </div>
            <div className="text-[11px] text-muted-foreground mt-0.5">
              {supabaseCorrectionCount >= 5
                ? "Model improving with your feedback"
                : `${5 - supabaseCorrectionCount} more correction${5 - supabaseCorrectionCount !== 1 ? "s" : ""} until model retraining`}
            </div>
          </div>
        )}

        {/* Retraining Status */}
        {retrainStatus.correctionsCount > 0 && (
          <div className="glass-card rounded-xl px-4 py-2.5 mb-2">
            <div className="text-[13px] font-medium text-foreground">
              Model Retraining
            </div>
            <div className="text-[11px] text-muted-foreground mt-0.5">
              {retrainStatus.lastRetrained
                ? `Last updated: ${(() => {
                    const diff = Date.now() - new Date(retrainStatus.lastRetrained).getTime();
                    const hours = Math.floor(diff / 3600000);
                    if (hours < 1) return "just now";
                    if (hours < 24) return `${hours} hour${hours !== 1 ? "s" : ""} ago`;
                    const days = Math.floor(hours / 24);
                    return `${days} day${days !== 1 ? "s" : ""} ago`;
                  })()} (based on ${retrainStatus.correctionsCount} corrections)`
                : `${retrainStatus.nextRetrainAt - retrainStatus.correctionsCount} more correction${
                    retrainStatus.nextRetrainAt - retrainStatus.correctionsCount !== 1 ? "s" : ""
                  } until next model update`}
            </div>
          </div>
        )}

        {/* Per-Modality Accuracy */}
        {(modalityAcc.eeg.total > 0 || modalityAcc.voice.total > 0 || modalityAcc.health.total > 0) && (
          <div className="glass-card rounded-xl px-4 py-2.5 mb-2">
            <div className="text-[13px] font-medium text-foreground mb-1.5">
              Modality Accuracy
            </div>
            <div className="text-[11px] text-muted-foreground flex gap-3 flex-wrap">
              {modalityAcc.eeg.total > 0 && (
                <span>EEG: {Math.round(modalityAcc.eeg.accuracy * 100)}% ({modalityAcc.eeg.correct}/{modalityAcc.eeg.total})</span>
              )}
              {modalityAcc.voice.total > 0 && (
                <span>Voice: {Math.round(modalityAcc.voice.accuracy * 100)}% ({modalityAcc.voice.correct}/{modalityAcc.voice.total})</span>
              )}
              {modalityAcc.health.total > 0 && (
                <span>Health: {Math.round(modalityAcc.health.accuracy * 100)}% ({modalityAcc.health.correct}/{modalityAcc.health.total})</span>
              )}
            </div>
          </div>
        )}
      </motion.div>

      {/* Activity Section */}
      <motion.div custom={4} initial="hidden" animate="visible" variants={cardVariants}>
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
      </motion.div>

      {/* Connected Assets */}
      <motion.div custom={5} initial="hidden" animate="visible" variants={cardVariants}>
        <SectionLabel>Connected Assets</SectionLabel>
        <GroupedList>
          <ListItem
            icon={Link2}
            iconColor="#0891b2"
            title="Connected Devices"
            rightText={`${connectedDeviceCount} of ${TOTAL_DEVICE_SLOTS} connected`}
            onClick={() => setLocation("/connected-assets")}
            isLast
          />
        </GroupedList>
      </motion.div>

      {/* Smart Interventions Section (#504) */}
      <motion.div custom={6} initial="hidden" animate="visible" variants={cardVariants}>
        <SectionLabel>Smart Interventions</SectionLabel>
        <GroupedList>
          <div className="px-3.5 pt-3.5 pb-2.5">
            <InterventionTriggerSettings />
          </div>
        </GroupedList>
      </motion.div>

      {/* Settings Section */}
      <motion.div custom={7} initial="hidden" animate="visible" variants={cardVariants}>
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
              // Cycle: dark -> light -> auto -> dark
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
      </motion.div>

      {/* Sign Out */}
      <motion.div custom={8} initial="hidden" animate="visible" variants={cardVariants}>
        <button
          onClick={handleSignOut}
          className="w-full bg-transparent border border-border rounded-xl p-3 text-destructive text-sm font-medium cursor-pointer transition-colors hover:bg-destructive/10 mt-1"
        >
          Sign Out
        </button>
      </motion.div>

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
