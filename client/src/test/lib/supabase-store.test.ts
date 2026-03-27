import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

// Mock supabase-browser before importing supabase-store
const mockInsert = vi.fn().mockResolvedValue({ error: null });
const mockUpdate = vi.fn().mockReturnValue({ eq: vi.fn().mockResolvedValue({ error: null }) });
const mockDeleteFn = vi.fn().mockReturnValue({ eq: vi.fn().mockResolvedValue({ error: null }) });
const mockThrowOnError = vi.fn().mockResolvedValue({ error: null });
const mockSelect = vi.fn();
const mockFrom = vi.fn().mockReturnValue({
  insert: mockInsert,
  update: mockUpdate,
  delete: mockDeleteFn,
  select: mockSelect,
});

const mockSupabase = { from: mockFrom };

vi.mock("@/lib/supabase-browser", () => ({
  getSupabase: vi.fn().mockResolvedValue(null), // Default: no Supabase
}));

import { getSupabase } from "@/lib/supabase-browser";
import {
  saveMoodLog,
  getMoodLogs,
  saveVoiceHistory,
  getVoiceHistory,
  saveEmotionHistory,
  getEmotionHistory,
  saveFoodLog,
  getFoodLogs,
  saveCycleData,
  getCycleData,
  saveBrainAge,
  getBrainAge,
  saveGlp1Injection,
  getGlp1Injections,
  saveNotification,
  getNotifications,
  markNotificationRead,
  markAllNotificationsRead,
  clearAllNotifications,
  syncLocalToSupabase,
} from "@/lib/supabase-store";

beforeEach(() => {
  localStorage.clear();
  vi.clearAllMocks();
  // Default: no Supabase
  vi.mocked(getSupabase).mockResolvedValue(null);
});

afterEach(() => {
  localStorage.clear();
});

// ── Mood Logs ────────────────────────────────────────────────────────────────

describe("saveMoodLog", () => {
  it("writes to localStorage when Supabase is unavailable", async () => {
    await saveMoodLog("user1", { mood: 7, energy: 6, notes: "good day" });
    const stored = JSON.parse(localStorage.getItem("ndw_mood_logs") || "[]");
    expect(stored.length).toBe(1);
    expect(stored[0].moodScore).toBe("7");
    expect(stored[0].notes).toBe("good day");
  });

  it("writes to both localStorage and Supabase when available", async () => {
    vi.mocked(getSupabase).mockResolvedValue(mockSupabase);
    mockInsert.mockResolvedValue({ error: null });

    await saveMoodLog("user1", { mood: 8, energy: 5 });

    // localStorage
    const stored = JSON.parse(localStorage.getItem("ndw_mood_logs") || "[]");
    expect(stored.length).toBe(1);
    // Supabase
    expect(mockFrom).toHaveBeenCalledWith("mood_logs");
    expect(mockInsert).toHaveBeenCalledWith(
      expect.objectContaining({ user_id: "user1", mood: 8, energy: 5 })
    );
  });

  it("caps localStorage at 100 entries", async () => {
    const existing = Array.from({ length: 100 }, (_, i) => ({
      id: `old_${i}`,
      moodScore: "5",
      loggedAt: new Date().toISOString(),
    }));
    localStorage.setItem("ndw_mood_logs", JSON.stringify(existing));

    await saveMoodLog("user1", { mood: 9 });
    const stored = JSON.parse(localStorage.getItem("ndw_mood_logs") || "[]");
    expect(stored.length).toBe(100);
    // newest at front
    expect(stored[0].moodScore).toBe("9");
  });
});

describe("getMoodLogs", () => {
  it("returns localStorage data when Supabase unavailable", async () => {
    localStorage.setItem("ndw_mood_logs", JSON.stringify([
      { id: "1", moodScore: "7", loggedAt: "2026-03-20T10:00:00Z" },
    ]));
    const logs = await getMoodLogs("user1");
    expect(logs.length).toBe(1);
    expect(logs[0].moodScore).toBe("7");
  });

  it("returns Supabase data when available and updates localStorage cache", async () => {
    const mockData = [
      { id: "sb-1", user_id: "user1", mood: 8, energy: 6, notes: null, created_at: "2026-03-20T10:00:00Z" },
    ];
    mockSelect.mockReturnValue({
      eq: vi.fn().mockReturnValue({
        order: vi.fn().mockReturnValue({
          limit: vi.fn().mockResolvedValue({ data: mockData, error: null }),
        }),
      }),
    });
    vi.mocked(getSupabase).mockResolvedValue(mockSupabase);

    const logs = await getMoodLogs("user1");
    expect(logs.length).toBe(1);
    expect(logs[0].moodScore).toBe("8");
    // Should also update localStorage
    const cached = JSON.parse(localStorage.getItem("ndw_mood_logs") || "[]");
    expect(cached.length).toBe(1);
  });
});

// ── Voice History ────────────────────────────────────────────────────────────

describe("saveVoiceHistory", () => {
  it("writes to localStorage when Supabase unavailable", async () => {
    await saveVoiceHistory("user1", { emotion: "happy", stress: 0.2, focus: 0.8, valence: 0.6 });
    const stored = JSON.parse(localStorage.getItem("ndw_voice_history") || "[]");
    expect(stored.length).toBe(1);
    expect(stored[0].emotion).toBe("happy");
  });
});

describe("getVoiceHistory", () => {
  it("returns localStorage data when Supabase unavailable", async () => {
    localStorage.setItem("ndw_voice_history", JSON.stringify([
      { emotion: "sad", timestamp: Date.now() },
    ]));
    const history = await getVoiceHistory("user1");
    expect(history.length).toBe(1);
    expect(history[0].emotion).toBe("sad");
  });
});

// ── Emotion History ──────────────────────────────────────────────────────────

describe("saveEmotionHistory", () => {
  it("writes to localStorage and prunes old entries", async () => {
    // Add an old entry that should be pruned (>30 days retention)
    const oldDate = new Date(Date.now() - 31 * 24 * 60 * 60 * 1000).toISOString();
    localStorage.setItem("ndw_emotion_history", JSON.stringify([
      { stress: 0.5, happiness: 0.5, focus: 0.5, dominantEmotion: "neutral", timestamp: oldDate },
    ]));

    await saveEmotionHistory("user1", { stress: 0.3, focus: 0.7, mood: 0.8, source: "voice" });

    const stored = JSON.parse(localStorage.getItem("ndw_emotion_history") || "[]");
    // Old entry should be pruned (>30 days), only new entry remains
    expect(stored.length).toBe(1);
    expect(stored[0].stress).toBe(0.3);
  });
});

describe("getEmotionHistory", () => {
  it("returns entries from localStorage within day range", async () => {
    const recentDate = new Date().toISOString();
    localStorage.setItem("ndw_emotion_history", JSON.stringify([
      { stress: 0.4, happiness: 0.6, focus: 0.5, dominantEmotion: "happy", timestamp: recentDate },
    ]));
    const history = await getEmotionHistory("user1", 7);
    expect(history.length).toBe(1);
    expect(history[0].dominantEmotion).toBe("happy");
  });
});

// ── Food Logs ────────────────────────────────────────────────────────────────

describe("saveFoodLog", () => {
  it("writes to user-specific localStorage key", async () => {
    await saveFoodLog("user1", { summary: "salad", calories: 350 });
    const stored = JSON.parse(localStorage.getItem("ndw_food_logs_user1") || "[]");
    expect(stored.length).toBe(1);
    expect(stored[0].summary).toBe("salad");
    expect(stored[0].calories).toBe(350);
  });
});

describe("getFoodLogs", () => {
  it("returns localStorage data when Supabase unavailable", async () => {
    localStorage.setItem("ndw_food_logs_user1", JSON.stringify([
      { id: "f1", summary: "pizza", calories: 800 },
    ]));
    const logs = await getFoodLogs("user1");
    expect(logs.length).toBe(1);
    expect(logs[0].summary).toBe("pizza");
  });
});

// ── Cycle Data ───────────────────────────────────────────────────────────────

describe("saveCycleData", () => {
  it("writes cycle data to localStorage in expected format", async () => {
    await saveCycleData("user1", {
      last_period_start: "2026-03-01",
      cycle_length: 30,
      period_length: 6,
    });
    const stored = JSON.parse(localStorage.getItem("ndw_cycle_data") || "null");
    expect(stored).not.toBeNull();
    expect(stored.lastPeriodStart).toBe("2026-03-01");
    expect(stored.cycleLength).toBe(30);
    expect(stored.periodLength).toBe(6);
  });
});

describe("getCycleData", () => {
  it("returns localStorage data when Supabase unavailable", async () => {
    localStorage.setItem("ndw_cycle_data", JSON.stringify({
      lastPeriodStart: "2026-03-05",
      cycleLength: 28,
      periodLength: 5,
    }));
    const data = await getCycleData("user1");
    expect(data).not.toBeNull();
    expect(data!.last_period_start).toBe("2026-03-05");
    expect(data!.cycle_length).toBe(28);
  });

  it("returns null when no data exists", async () => {
    const data = await getCycleData("user1");
    expect(data).toBeNull();
  });
});

// ── Brain Age ────────────────────────────────────────────────────────────────

describe("saveBrainAge", () => {
  it("writes brain age to localStorage", async () => {
    await saveBrainAge("user1", { estimated_age: 28, actual_age: 30, gap: -2 });
    const stored = JSON.parse(localStorage.getItem("ndw_brain_age") || "null");
    expect(stored).not.toBeNull();
    expect(stored.estimatedAge).toBe(28);
    expect(stored.brainAgeGap).toBe(-2);
  });
});

describe("getBrainAge", () => {
  it("returns localStorage data when Supabase unavailable", async () => {
    localStorage.setItem("ndw_brain_age", JSON.stringify({
      estimatedAge: 25,
      actualAge: 30,
      brainAgeGap: -5,
      timestamp: Date.now(),
    }));
    const result = await getBrainAge("user1");
    expect(result).not.toBeNull();
    expect(result!.estimated_age).toBe(25);
    expect(result!.gap).toBe(-5);
  });

  it("returns null when no data exists", async () => {
    const result = await getBrainAge("user1");
    expect(result).toBeNull();
  });
});

// ── GLP-1 Injections ─────────────────────────────────────────────────────────

describe("saveGlp1Injection", () => {
  it("writes injection to localStorage", async () => {
    await saveGlp1Injection("user1", { medication: "Ozempic", dose: 0.5 });
    const stored = JSON.parse(localStorage.getItem("ndw_glp1_injections") || "[]");
    expect(stored.length).toBe(1);
    expect(stored[0].medication).toBe("Ozempic");
    expect(stored[0].dose).toBe(0.5);
  });
});

describe("getGlp1Injections", () => {
  it("returns localStorage data when Supabase unavailable", async () => {
    localStorage.setItem("ndw_glp1_injections", JSON.stringify([
      { id: "g1", medication: "Wegovy", dose: 1.7, date: "2026-03-20" },
    ]));
    const injections = await getGlp1Injections("user1");
    expect(injections.length).toBe(1);
    expect(injections[0].medication).toBe("Wegovy");
  });
});

// ── Notifications ────────────────────────────────────────────────────────────

describe("saveNotification", () => {
  it("writes notification to localStorage", async () => {
    await saveNotification("user1", { type: "streak", title: "3-day streak!", body: "Keep going" });
    const stored = JSON.parse(localStorage.getItem("ndw_notifications") || "[]");
    expect(stored.length).toBe(1);
    expect(stored[0].type).toBe("streak");
    expect(stored[0].title).toBe("3-day streak!");
    expect(stored[0].read).toBe(false);
  });
});

describe("getNotifications", () => {
  it("returns localStorage data when Supabase unavailable", async () => {
    localStorage.setItem("ndw_notifications", JSON.stringify([
      { id: "n1", type: "reminder", title: "Check in", body: "", timestamp: Date.now(), read: false },
    ]));
    const notifs = await getNotifications("user1");
    expect(notifs.length).toBe(1);
    expect(notifs[0].title).toBe("Check in");
  });
});

describe("markNotificationRead", () => {
  it("marks a single notification as read in localStorage", async () => {
    localStorage.setItem("ndw_notifications", JSON.stringify([
      { id: "n1", type: "streak", title: "Streak", read: false },
      { id: "n2", type: "reminder", title: "Remind", read: false },
    ]));
    await markNotificationRead("user1", "n1");
    const stored = JSON.parse(localStorage.getItem("ndw_notifications") || "[]");
    expect(stored[0].read).toBe(true);
    expect(stored[1].read).toBe(false);
  });
});

describe("markAllNotificationsRead", () => {
  it("marks all notifications as read in localStorage", async () => {
    localStorage.setItem("ndw_notifications", JSON.stringify([
      { id: "n1", read: false },
      { id: "n2", read: false },
    ]));
    await markAllNotificationsRead("user1");
    const stored = JSON.parse(localStorage.getItem("ndw_notifications") || "[]");
    expect(stored.every((n: any) => n.read === true)).toBe(true);
  });
});

describe("clearAllNotifications", () => {
  it("clears all notifications from localStorage", async () => {
    localStorage.setItem("ndw_notifications", JSON.stringify([{ id: "n1" }]));
    await clearAllNotifications("user1");
    const stored = JSON.parse(localStorage.getItem("ndw_notifications") || "[]");
    expect(stored.length).toBe(0);
  });
});

// ── syncLocalToSupabase ──────────────────────────────────────────────────────

describe("syncLocalToSupabase", () => {
  it("does nothing when Supabase is unavailable", async () => {
    localStorage.setItem("ndw_mood_logs", JSON.stringify([{ moodScore: "5" }]));
    await syncLocalToSupabase("user1");
    expect(mockFrom).not.toHaveBeenCalled();
    // Should not set sync flag
    expect(localStorage.getItem("ndw_supabase_synced")).toBeNull();
  });

  it("migrates localStorage data to Supabase when available", async () => {
    vi.mocked(getSupabase).mockResolvedValue(mockSupabase);
    mockInsert.mockReturnValue({ throwOnError: mockThrowOnError });

    localStorage.setItem("ndw_mood_logs", JSON.stringify([
      { moodScore: "7", energyLevel: "6", notes: "test", loggedAt: "2026-03-20T10:00:00Z" },
    ]));
    localStorage.setItem("ndw_voice_history", JSON.stringify([
      { emotion: "happy", stress_index: 0.2, timestamp: 1000 },
    ]));

    await syncLocalToSupabase("user1");

    // Should have called insert for mood_logs and voice_history
    expect(mockFrom).toHaveBeenCalledWith("mood_logs");
    expect(mockFrom).toHaveBeenCalledWith("voice_history");
    // Should set sync flag
    expect(JSON.parse(localStorage.getItem("ndw_supabase_synced") || "false")).toBe(true);
  });

  it("does not re-sync if already synced", async () => {
    vi.mocked(getSupabase).mockResolvedValue(mockSupabase);
    localStorage.setItem("ndw_supabase_synced", "true");
    localStorage.setItem("ndw_mood_logs", JSON.stringify([{ moodScore: "5" }]));

    await syncLocalToSupabase("user1");
    expect(mockFrom).not.toHaveBeenCalled();
  });
});
