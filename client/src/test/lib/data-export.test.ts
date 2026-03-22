import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  exportSessionsCSV,
  exportVoiceCheckinsCSV,
  exportAccountJSON,
  downloadFile,
  deleteAllLocalData,
  type CheckinData,
} from "@/lib/data-export";
import type { SessionSummary } from "@/lib/ml-api";

// ── Test data ────────────────────────────────────────────────────────────────

const MOCK_SESSIONS: SessionSummary[] = [
  {
    session_id: "sess-001",
    user_id: "user-42",
    session_type: "meditation",
    start_time: 1774300000, // some unix timestamp
    status: "completed",
    summary: {
      duration_sec: 300,
      n_frames: 150,
      avg_stress: 0.3,
      avg_focus: 0.7,
      avg_relaxation: 0.6,
      avg_flow: 0.5,
      avg_creativity: 0.4,
      avg_valence: 0.2,
      avg_arousal: 0.35,
      dominant_emotion: "neutral",
    },
  },
  {
    session_id: "sess-002",
    user_id: "user-42",
    session_type: "focus",
    start_time: 1774400000,
    status: "completed",
    summary: {
      duration_sec: 600,
      n_frames: 300,
      avg_stress: 0.5,
      avg_focus: 0.85,
      avg_relaxation: 0.3,
      avg_flow: 0.7,
      avg_creativity: 0.2,
      avg_valence: 0.1,
      avg_arousal: 0.6,
      dominant_emotion: "happy, excited",
    },
  },
];

const MOCK_CHECKINS: CheckinData[] = [
  {
    id: "checkin-001",
    timestamp: "2026-03-21T14:30:00Z",
    emotion: "calm",
    intensity: 0.6,
    notes: "Feeling good after meditation",
    voiceBiomarkers: {
      energy: 0.4,
      stress: 0.2,
      valence: 0.7,
    },
  },
  {
    id: "checkin-002",
    timestamp: "2026-03-21T18:00:00Z",
    emotion: "anxious, stressed",
    intensity: 0.8,
    notes: 'Had a "difficult" meeting',
    voiceBiomarkers: {
      energy: 0.7,
      stress: 0.85,
      valence: -0.3,
    },
  },
];

// ── exportSessionsCSV ────────────────────────────────────────────────────────

describe("exportSessionsCSV", () => {
  it("has correct headers in the first line", () => {
    const csv = exportSessionsCSV(MOCK_SESSIONS);
    const firstLine = csv.split("\n")[0];
    expect(firstLine).toContain("session_id");
    expect(firstLine).toContain("user_id");
    expect(firstLine).toContain("session_type");
    expect(firstLine).toContain("start_time");
    expect(firstLine).toContain("status");
    expect(firstLine).toContain("duration_sec");
    expect(firstLine).toContain("avg_stress");
    expect(firstLine).toContain("avg_focus");
    expect(firstLine).toContain("dominant_emotion");
  });

  it("produces the correct number of rows (header + data)", () => {
    const csv = exportSessionsCSV(MOCK_SESSIONS);
    const lines = csv.split("\n");
    expect(lines.length).toBe(3); // 1 header + 2 data rows
  });

  it("handles special characters in dominant_emotion by quoting", () => {
    const csv = exportSessionsCSV(MOCK_SESSIONS);
    const lines = csv.split("\n");
    // The second session has dominant_emotion = "happy, excited" which contains a comma
    const lastRow = lines[2];
    // Should be wrapped in quotes
    expect(lastRow).toContain('"happy, excited"');
  });

  it("returns only the header for empty sessions array", () => {
    const csv = exportSessionsCSV([]);
    const lines = csv.split("\n");
    expect(lines.length).toBe(1);
    expect(lines[0]).toContain("session_id");
  });
});

// ── exportVoiceCheckinsCSV ───────────────────────────────────────────────────

describe("exportVoiceCheckinsCSV", () => {
  it("has correct headers", () => {
    const csv = exportVoiceCheckinsCSV(MOCK_CHECKINS);
    const firstLine = csv.split("\n")[0];
    expect(firstLine).toContain("id");
    expect(firstLine).toContain("timestamp");
    expect(firstLine).toContain("emotion");
    expect(firstLine).toContain("intensity");
    expect(firstLine).toContain("notes");
    expect(firstLine).toContain("voice_energy");
    expect(firstLine).toContain("voice_stress");
    expect(firstLine).toContain("voice_valence");
  });

  it("escapes commas in emotion field", () => {
    const csv = exportVoiceCheckinsCSV(MOCK_CHECKINS);
    // "anxious, stressed" has a comma
    expect(csv).toContain('"anxious, stressed"');
  });

  it("escapes quotes in notes field", () => {
    const csv = exportVoiceCheckinsCSV(MOCK_CHECKINS);
    // notes: 'Had a "difficult" meeting' should become: "Had a ""difficult"" meeting"
    expect(csv).toContain('"Had a ""difficult"" meeting"');
  });
});

// ── exportAccountJSON ────────────────────────────────────────────────────────

describe("exportAccountJSON", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("includes all expected top-level keys", () => {
    const json = exportAccountJSON("user-42", MOCK_SESSIONS, MOCK_CHECKINS);
    const parsed = JSON.parse(json);
    expect(parsed).toHaveProperty("export_version");
    expect(parsed).toHaveProperty("export_date");
    expect(parsed).toHaveProperty("user_id");
    expect(parsed).toHaveProperty("sessions");
    expect(parsed).toHaveProperty("voice_checkins");
    expect(parsed).toHaveProperty("preferences");
  });

  it("includes the correct user_id", () => {
    const json = exportAccountJSON("user-42");
    const parsed = JSON.parse(json);
    expect(parsed.user_id).toBe("user-42");
  });

  it("includes sessions and checkins data", () => {
    const json = exportAccountJSON("user-42", MOCK_SESSIONS, MOCK_CHECKINS);
    const parsed = JSON.parse(json);
    expect(parsed.sessions).toHaveLength(2);
    expect(parsed.voice_checkins).toHaveLength(2);
  });

  it("collects ndw_ prefixed localStorage keys", () => {
    localStorage.setItem("ndw_theme", "dark");
    localStorage.setItem("ndw_muse_connected", "true");
    localStorage.setItem("other_key", "should-not-appear");

    const json = exportAccountJSON("user-42");
    const parsed = JSON.parse(json);

    expect(parsed.preferences).toHaveProperty("ndw_theme", "dark");
    expect(parsed.preferences).toHaveProperty("ndw_muse_connected", "true");
    expect(parsed.preferences).not.toHaveProperty("other_key");
  });
});

// ── downloadFile ─────────────────────────────────────────────────────────────

describe("downloadFile", () => {
  it("creates and clicks an anchor element to trigger download", () => {
    // Mock URL.createObjectURL and URL.revokeObjectURL
    const mockUrl = "blob:http://localhost/fake-blob-url";
    const createObjectURLSpy = vi.spyOn(URL, "createObjectURL").mockReturnValue(mockUrl);
    const revokeObjectURLSpy = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});

    // Track anchor element creation and click
    const clickSpy = vi.fn();
    const appendChildSpy = vi.spyOn(document.body, "appendChild").mockImplementation((node) => {
      // Capture the click spy on the anchor
      if (node instanceof HTMLAnchorElement) {
        node.click = clickSpy;
      }
      return node;
    });
    const removeChildSpy = vi.spyOn(document.body, "removeChild").mockImplementation((node) => node);

    downloadFile("test content", "export.csv", "text/csv");

    expect(createObjectURLSpy).toHaveBeenCalledOnce();
    expect(clickSpy).toHaveBeenCalledOnce();
    expect(removeChildSpy).toHaveBeenCalledOnce();
    expect(revokeObjectURLSpy).toHaveBeenCalledWith(mockUrl);

    // Clean up
    createObjectURLSpy.mockRestore();
    revokeObjectURLSpy.mockRestore();
    appendChildSpy.mockRestore();
    removeChildSpy.mockRestore();
  });

  it("handles ArrayBuffer content", () => {
    const mockUrl = "blob:http://localhost/fake-blob-url";
    const createObjectURLSpy = vi.spyOn(URL, "createObjectURL").mockReturnValue(mockUrl);
    const revokeObjectURLSpy = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const appendChildSpy = vi.spyOn(document.body, "appendChild").mockImplementation((node) => {
      if (node instanceof HTMLAnchorElement) {
        node.click = vi.fn();
      }
      return node;
    });
    const removeChildSpy = vi.spyOn(document.body, "removeChild").mockImplementation((node) => node);

    const buffer = new ArrayBuffer(16);
    downloadFile(buffer, "data.edf", "application/octet-stream");

    // Should have been called with a Blob wrapping the ArrayBuffer
    expect(createObjectURLSpy).toHaveBeenCalledOnce();
    const blobArg = createObjectURLSpy.mock.calls[0][0];
    expect(blobArg).toBeInstanceOf(Blob);

    createObjectURLSpy.mockRestore();
    revokeObjectURLSpy.mockRestore();
    appendChildSpy.mockRestore();
    removeChildSpy.mockRestore();
  });
});

// ── deleteAllLocalData ───────────────────────────────────────────────────────

describe("deleteAllLocalData", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("removes ndw_ prefixed keys and auth_token", () => {
    localStorage.setItem("ndw_theme", "dark");
    localStorage.setItem("ndw_muse_connected", "true");
    localStorage.setItem("auth_token", "some-token");
    localStorage.setItem("unrelated_key", "keep-this");

    const removed = deleteAllLocalData();

    expect(removed).toBe(3); // ndw_theme, ndw_muse_connected, auth_token
    expect(localStorage.getItem("ndw_theme")).toBeNull();
    expect(localStorage.getItem("ndw_muse_connected")).toBeNull();
    expect(localStorage.getItem("auth_token")).toBeNull();
    expect(localStorage.getItem("unrelated_key")).toBe("keep-this");
  });

  it("returns 0 when no matching keys exist", () => {
    localStorage.setItem("unrelated", "value");
    const removed = deleteAllLocalData();
    expect(removed).toBe(0);
  });
});
