import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Bell,
  Mic,
  Flame,
  Brain,
  Trophy,
  Pill,
  Calendar,
  BarChart3,
  Trash2,
  BellOff,
} from "lucide-react";

/* ── Types ──────────────────────────────────────────────────── */

export type NotificationType =
  | "voice_result"
  | "streak"
  | "reminder"
  | "eeg_summary"
  | "achievement"
  | "supplement"
  | "weekly_summary";

export interface AppNotification {
  id: string;
  type: NotificationType;
  title: string;
  body: string;
  timestamp: number; // epoch ms
  read: boolean;
}

/* ── Storage helpers (exported for testing and external use) ── */

const STORAGE_KEY = "ndw_notifications";

export function loadNotifications(): AppNotification[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed as AppNotification[];
  } catch {
    return [];
  }
}

export function saveNotifications(notifications: AppNotification[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(notifications));
  } catch {
    // localStorage full or unavailable — silently ignore
  }
}

export function addNotification(n: Omit<AppNotification, "id" | "timestamp" | "read">): void {
  const all = loadNotifications();
  const entry: AppNotification = {
    ...n,
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    timestamp: Date.now(),
    read: false,
  };
  // Keep max 100 notifications
  all.unshift(entry);
  if (all.length > 100) all.length = 100;
  saveNotifications(all);
}

export function clearNotifications(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}

/* ── Icon mapping ──────────────────────────────────────────── */

const ICON_MAP: Record<NotificationType, React.ComponentType<{ className?: string }>> = {
  voice_result: Mic,
  streak: Flame,
  reminder: Calendar,
  eeg_summary: Brain,
  achievement: Trophy,
  supplement: Pill,
  weekly_summary: BarChart3,
};

const COLOR_MAP: Record<NotificationType, string> = {
  voice_result: "text-cyan-400 bg-cyan-400/10",
  streak: "text-yellow-400 bg-yellow-400/10",
  reminder: "text-indigo-400 bg-indigo-400/10",
  eeg_summary: "text-violet-400 bg-violet-400/10",
  achievement: "text-amber-400 bg-amber-400/10",
  supplement: "text-rose-400 bg-rose-400/10",
  weekly_summary: "text-emerald-400 bg-emerald-400/10",
};

/* ── Time formatting ───────────────────────────────────────── */

function formatTime(ts: number): string {
  const now = Date.now();
  const diff = now - ts;
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(ts).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

/* ── Notification Item ─────────────────────────────────────── */

function NotificationItem({
  notification,
  onMarkRead,
}: {
  notification: AppNotification;
  onMarkRead: (id: string) => void;
}) {
  const Icon = ICON_MAP[notification.type] ?? Bell;
  const colorClasses = COLOR_MAP[notification.type] ?? "text-muted-foreground bg-muted/30";

  return (
    <div
      className={`flex gap-3 p-3 rounded-lg transition-colors ${
        notification.read ? "opacity-60" : "bg-muted/20"
      }`}
      onClick={() => !notification.read && onMarkRead(notification.id)}
      style={{ cursor: notification.read ? "default" : "pointer" }}
    >
      <div
        className={`w-9 h-9 rounded-full flex items-center justify-center shrink-0 ${colorClasses}`}
      >
        <Icon className="h-4 w-4" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2">
          <p className="text-sm font-medium leading-tight">{notification.title}</p>
          <span className="text-[10px] text-muted-foreground whitespace-nowrap shrink-0">
            {formatTime(notification.timestamp)}
          </span>
        </div>
        <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">
          {notification.body}
        </p>
      </div>
      {!notification.read && (
        <div className="w-2 h-2 rounded-full bg-primary shrink-0 mt-1.5" />
      )}
    </div>
  );
}

/* ── Empty State ───────────────────────────────────────────── */

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-4">
      <div className="w-16 h-16 rounded-full bg-muted/30 flex items-center justify-center mb-4">
        <BellOff className="h-7 w-7 text-muted-foreground" />
      </div>
      <p className="text-sm font-medium text-muted-foreground mb-1">
        No notifications yet
      </p>
      <p className="text-xs text-muted-foreground/70 text-center max-w-[280px]">
        Complete a voice analysis to get started. Notifications for mood results, streaks,
        reminders, and achievements will appear here.
      </p>
    </div>
  );
}

/* ── Main Page ─────────────────────────────────────────────── */

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState<AppNotification[]>([]);

  useEffect(() => {
    setNotifications(loadNotifications());
  }, []);

  function handleMarkRead(id: string) {
    setNotifications((prev) => {
      const updated = prev.map((n) =>
        n.id === id ? { ...n, read: true } : n
      );
      saveNotifications(updated);
      return updated;
    });
  }

  function handleMarkAllRead() {
    setNotifications((prev) => {
      const updated = prev.map((n) => ({ ...n, read: true }));
      saveNotifications(updated);
      return updated;
    });
  }

  function handleClearAll() {
    clearNotifications();
    setNotifications([]);
  }

  const unreadCount = notifications.filter((n) => !n.read).length;

  return (
    <main className="p-4 md:p-6 pb-24 space-y-4 max-w-3xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Bell className="h-6 w-6 text-primary" />
          <div>
            <h1 className="text-xl font-semibold">Notifications</h1>
            <p className="text-xs text-muted-foreground">
              {unreadCount > 0
                ? `${unreadCount} unread`
                : "All caught up"}
            </p>
          </div>
        </div>
        {notifications.length > 0 && (
          <div className="flex items-center gap-2">
            {unreadCount > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleMarkAllRead}
                className="text-xs"
              >
                Mark all read
              </Button>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={handleClearAll}
              className="text-xs text-destructive hover:text-destructive"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>
        )}
      </div>

      {/* Notification list */}
      {notifications.length === 0 ? (
        <EmptyState />
      ) : (
        <Card className="glass-card p-2 divide-y divide-border/20">
          {notifications.map((n) => (
            <NotificationItem
              key={n.id}
              notification={n}
              onMarkRead={handleMarkRead}
            />
          ))}
        </Card>
      )}
    </main>
  );
}
