import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import NotificationsPage, {
  loadNotifications,
  saveNotifications,
  addNotification,
  clearNotifications,
  type AppNotification,
} from "@/pages/notifications";

vi.mock("wouter", () => ({
  useLocation: () => ["/notifications", vi.fn()],
}));

describe("Notifications page", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("renders without crashing", () => {
    renderWithProviders(<NotificationsPage />);
    expect(document.body).toBeTruthy();
  });

  it("shows the page heading", () => {
    renderWithProviders(<NotificationsPage />);
    expect(screen.getByText("Notifications")).toBeInTheDocument();
  });

  it("shows empty state when no notifications exist", () => {
    renderWithProviders(<NotificationsPage />);
    expect(screen.getByText("No notifications yet")).toBeInTheDocument();
    expect(
      screen.getByText(/Complete a voice analysis to get started/)
    ).toBeInTheDocument();
  });

  it("shows 'All caught up' subtitle when no unread notifications", () => {
    renderWithProviders(<NotificationsPage />);
    expect(screen.getByText("All caught up")).toBeInTheDocument();
  });

  it("shows notifications from localStorage", () => {
    const notifications: AppNotification[] = [
      {
        id: "test-1",
        type: "voice_result",
        title: "Voice Analysis Complete",
        body: "Your mood was detected as Happy",
        timestamp: Date.now(),
        read: false,
      },
    ];
    saveNotifications(notifications);

    renderWithProviders(<NotificationsPage />);
    expect(screen.getByText("Voice Analysis Complete")).toBeInTheDocument();
    expect(screen.getByText("Your mood was detected as Happy")).toBeInTheDocument();
  });

  it("shows unread count", () => {
    const notifications: AppNotification[] = [
      {
        id: "test-1",
        type: "streak",
        title: "3-day streak!",
        body: "Keep going",
        timestamp: Date.now(),
        read: false,
      },
      {
        id: "test-2",
        type: "reminder",
        title: "Meditation reminder",
        body: "Time to check in",
        timestamp: Date.now() - 60000,
        read: true,
      },
    ];
    saveNotifications(notifications);

    renderWithProviders(<NotificationsPage />);
    expect(screen.getByText("1 unread")).toBeInTheDocument();
  });

  it("shows Mark all read button when unread notifications exist", () => {
    saveNotifications([
      {
        id: "test-1",
        type: "achievement",
        title: "Achievement Unlocked",
        body: "First check-in complete",
        timestamp: Date.now(),
        read: false,
      },
    ]);

    renderWithProviders(<NotificationsPage />);
    expect(screen.getByText("Mark all read")).toBeInTheDocument();
  });

  it("marks all notifications as read", () => {
    saveNotifications([
      {
        id: "test-1",
        type: "voice_result",
        title: "Result 1",
        body: "Body 1",
        timestamp: Date.now(),
        read: false,
      },
      {
        id: "test-2",
        type: "streak",
        title: "Result 2",
        body: "Body 2",
        timestamp: Date.now(),
        read: false,
      },
    ]);

    renderWithProviders(<NotificationsPage />);
    fireEvent.click(screen.getByText("Mark all read"));
    expect(screen.getByText("All caught up")).toBeInTheDocument();
  });

  it("clears all notifications", () => {
    saveNotifications([
      {
        id: "test-1",
        type: "weekly_summary",
        title: "Weekly Summary",
        body: "Your week in review",
        timestamp: Date.now(),
        read: false,
      },
    ]);

    renderWithProviders(<NotificationsPage />);
    expect(screen.getByText("Weekly Summary")).toBeInTheDocument();

    // Click the clear/trash button
    const trashButton = screen.getByRole("button", { name: "" });
    // Find the button with Trash2 icon (it's the one without text)
    const buttons = screen.getAllByRole("button");
    const clearBtn = buttons.find((btn) => btn.querySelector("svg"));
    if (clearBtn) fireEvent.click(clearBtn);

    // After clearing, notifications gone — but may need re-render
    // Verify localStorage is cleared
    expect(loadNotifications()).toEqual([]);
  });
});

describe("Notification storage helpers", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("loadNotifications returns empty array when localStorage is empty", () => {
    expect(loadNotifications()).toEqual([]);
  });

  it("saveNotifications and loadNotifications round-trip correctly", () => {
    const notifications: AppNotification[] = [
      {
        id: "n1",
        type: "voice_result",
        title: "Test",
        body: "Test body",
        timestamp: 1000,
        read: false,
      },
    ];
    saveNotifications(notifications);
    const loaded = loadNotifications();
    expect(loaded).toEqual(notifications);
  });

  it("addNotification adds to the front of the list", () => {
    addNotification({
      type: "streak",
      title: "First",
      body: "First notification",
    });
    addNotification({
      type: "achievement",
      title: "Second",
      body: "Second notification",
    });

    const loaded = loadNotifications();
    expect(loaded).toHaveLength(2);
    expect(loaded[0].title).toBe("Second");
    expect(loaded[1].title).toBe("First");
  });

  it("addNotification sets read to false", () => {
    addNotification({
      type: "reminder",
      title: "Test",
      body: "Reminder",
    });
    const loaded = loadNotifications();
    expect(loaded[0].read).toBe(false);
  });

  it("addNotification generates unique ids", () => {
    addNotification({ type: "streak", title: "A", body: "a" });
    addNotification({ type: "streak", title: "B", body: "b" });
    const loaded = loadNotifications();
    expect(loaded[0].id).not.toBe(loaded[1].id);
  });

  it("clearNotifications removes all notifications", () => {
    addNotification({ type: "voice_result", title: "Test", body: "Body" });
    expect(loadNotifications()).toHaveLength(1);
    clearNotifications();
    expect(loadNotifications()).toEqual([]);
  });

  it("loadNotifications returns empty array for invalid JSON", () => {
    localStorage.setItem("ndw_notifications", "not-json");
    expect(loadNotifications()).toEqual([]);
  });

  it("loadNotifications returns empty array for non-array JSON", () => {
    localStorage.setItem("ndw_notifications", '{"foo": "bar"}');
    expect(loadNotifications()).toEqual([]);
  });

  it("addNotification caps at 100 notifications", () => {
    // Add 105 notifications
    for (let i = 0; i < 105; i++) {
      addNotification({
        type: "streak",
        title: `Notification ${i}`,
        body: `Body ${i}`,
      });
    }
    const loaded = loadNotifications();
    expect(loaded).toHaveLength(100);
    // Most recent should be first
    expect(loaded[0].title).toBe("Notification 104");
  });
});
