import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import Onboarding from "@/pages/onboarding";

// ── Shared mocks ─────────────────────────────────────────────────────────────

vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...rest }: any) => <div {...rest}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

vi.mock("@/hooks/use-device", () => ({
  useDevice: () => ({
    latestFrame: null,
    state: "disconnected",
    deviceStatus: null,
  }),
}));

vi.mock("@/hooks/use-auth", () => ({
  useAuth: () => ({ user: null, isLoading: false }),
}));

vi.mock("@/hooks/use-voice-emotion", () => ({
  useVoiceEmotion: () => ({
    lastResult: null,
    isRecording: false,
    isAnalyzing: false,
    error: null,
    startRecording: vi.fn(),
  }),
}));

vi.mock("wouter", () => ({
  useLocation: () => ["/onboarding", vi.fn()],
  Link: (props: any) => <a href={props.href}>{props.children}</a>,
}));

vi.mock("@/lib/ml-api", () => ({
  getBaselineStatus: vi.fn().mockResolvedValue({ ready: false, n_frames: 0 }),
  addBaselineFrame: vi.fn().mockResolvedValue({}),
  simulateEEG: vi.fn().mockResolvedValue({ signals: [[]] }),
}));

vi.mock("@/lib/participant", () => ({
  getParticipantId: () => "test-user",
}));

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    })
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

// ── Step 1: Welcome ───────────────────────────────────────────────────────────

describe("Onboarding — Fast Track (new users)", () => {
  it("renders without crashing", () => {
    renderWithProviders(<Onboarding />);
  });

  it("shows the Welcome to AntarAI heading for fresh users", () => {
    renderWithProviders(<Onboarding />);
    expect(
      screen.getByText("Welcome to AntarAI")
    ).toBeInTheDocument();
  });

  it("shows a Try it now button", () => {
    renderWithProviders(<Onboarding />);
    expect(
      screen.getByText(/Try it now/i)
    ).toBeInTheDocument();
  });

  it("shows a Skip for now option", () => {
    renderWithProviders(<Onboarding />);
    expect(screen.getByText(/Skip for now/i)).toBeInTheDocument();
  });

  it("shows 3 value proposition cards", () => {
    renderWithProviders(<Onboarding />);
    expect(screen.getByText("10-Second Voice Check-in")).toBeInTheDocument();
    expect(screen.getByText("Personalized Insights")).toBeInTheDocument();
    expect(screen.getByText("Holistic Wellness")).toBeInTheDocument();
  });

  it("falls back to step 2 flow when step is already set", () => {
    localStorage.setItem("ndw_onboarding_step", "2");
    renderWithProviders(<Onboarding />);
    expect(screen.getByText(/How do you want to start/i)).toBeInTheDocument();
    expect(screen.getByText(/Step 2 of 5/i)).toBeInTheDocument();
  });
});

// ── Step 2: Choose path ────────────────────────────────────────────────────────

describe("Onboarding — Step 2 (Choose path)", () => {
  function renderStep2() {
    localStorage.setItem("ndw_onboarding_step", "2");
    renderWithProviders(<Onboarding />);
  }

  it("shows the choose path heading", () => {
    renderStep2();
    expect(screen.getByText(/How do you want to start/i)).toBeInTheDocument();
  });

  it("shows Quick Start and Full Setup options", () => {
    renderStep2();
    expect(screen.getByRole("button", { name: /Quick Start \(Voice\)/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Full Setup \(EEG\)/i })).toBeInTheDocument();
  });

  it("advances to voice step (3a) when Quick Start is chosen", () => {
    renderStep2();
    fireEvent.click(screen.getByRole("button", { name: /Quick Start \(Voice\)/i }));
    // Use heading role to avoid matching the button with similar text
    expect(screen.getByRole("heading", { name: "Voice Analysis" })).toBeInTheDocument();
    expect(screen.getByText(/Step 3 of 5/i)).toBeInTheDocument();
  });

  it("advances to EEG step (3b) when Full Setup is chosen", () => {
    renderStep2();
    fireEvent.click(screen.getByRole("button", { name: /Full Setup \(EEG\)/i }));
    expect(screen.getByText(/EEG Baseline Calibration/i)).toBeInTheDocument();
    expect(screen.getByText(/Step 3 of 5/i)).toBeInTheDocument();
  });
});

// ── Step 3a: Voice check-in ────────────────────────────────────────────────────

describe("Onboarding — Step 3a (Voice check-in)", () => {
  function renderVoiceStep() {
    localStorage.setItem("ndw_onboarding_step", "3");
    localStorage.setItem("ndw_onboarding_path", "voice");
    renderWithProviders(<Onboarding />);
  }

  it("shows the Voice Analysis heading", () => {
    renderVoiceStep();
    expect(screen.getByRole("heading", { name: "Voice Analysis" })).toBeInTheDocument();
  });

  it("shows the start recording button", () => {
    renderVoiceStep();
    expect(
      screen.getByRole("button", { name: /Start 10-second voice analysis/i })
    ).toBeInTheDocument();
  });

  it("shows a Skip for now option", () => {
    renderVoiceStep();
    expect(
      screen.getByRole("button", { name: /Skip for now/i })
    ).toBeInTheDocument();
  });

  it("advances to step 4 when Skip is clicked", () => {
    renderVoiceStep();
    fireEvent.click(screen.getByRole("button", { name: /Skip for now/i }));
    // Use heading role to avoid matching the "Connect Health Data" button
    expect(screen.getByRole("heading", { name: "Connect Health Data" })).toBeInTheDocument();
    expect(screen.getByText(/Step 4 of 5/i)).toBeInTheDocument();
  });
});

// ── Step 3b: EEG calibration ───────────────────────────────────────────────────

describe("Onboarding — Step 3b (EEG calibration)", () => {
  function renderEegStep() {
    localStorage.setItem("ndw_onboarding_step", "3");
    localStorage.setItem("ndw_onboarding_path", "eeg");
    renderWithProviders(<Onboarding />);
  }

  it("shows the EEG Baseline Calibration heading", () => {
    renderEegStep();
    expect(screen.getByText("EEG Baseline Calibration")).toBeInTheDocument();
  });

  it("shows the 2 minutes tagline", () => {
    renderEegStep();
    expect(screen.getByText(/Muse 2 only · 2 minutes · Done once/)).toBeInTheDocument();
  });

  it("shows the no-headset simulation notice when device not connected", () => {
    renderEegStep();
    expect(
      screen.getByText(/No headset detected — simulation mode will preview the flow/)
    ).toBeInTheDocument();
  });

  it("shows the Start calibration button", () => {
    renderEegStep();
    expect(
      screen.getByRole("button", { name: /Start calibration/i })
    ).toBeInTheDocument();
  });

  it("shows a Skip for now button", () => {
    renderEegStep();
    expect(
      screen.getByRole("button", { name: /Skip for now/i })
    ).toBeInTheDocument();
  });

  it("transitions to recording phase when Start calibration is clicked", () => {
    renderEegStep();
    fireEvent.click(screen.getByRole("button", { name: /Start calibration/i }));
    expect(screen.getByText("Recording your baseline\u2026")).toBeInTheDocument();
  });

  it("shows simulation mode notice in recording phase when no device", () => {
    renderEegStep();
    fireEvent.click(screen.getByRole("button", { name: /Start calibration/i }));
    expect(
      screen.getByText(/Simulation mode — no headset needed/)
    ).toBeInTheDocument();
  });

  it("shows the frame progress counter in recording phase", () => {
    renderEegStep();
    fireEvent.click(screen.getByRole("button", { name: /Start calibration/i }));
    expect(screen.getByText(/0 \/ 120 frames/)).toBeInTheDocument();
  });

  it("advances to step 4 when Skip for now is clicked", () => {
    renderEegStep();
    fireEvent.click(screen.getByRole("button", { name: /Skip for now/i }));
    expect(screen.getByRole("heading", { name: "Connect Health Data" })).toBeInTheDocument();
    expect(screen.getByText(/Step 4 of 5/i)).toBeInTheDocument();
  });
});

// ── Step 4: Health sync ────────────────────────────────────────────────────────

describe("Onboarding — Step 4 (Health sync)", () => {
  function renderHealthStep() {
    localStorage.setItem("ndw_onboarding_step", "4");
    localStorage.setItem("ndw_onboarding_path", "voice");
    renderWithProviders(<Onboarding />);
  }

  it("shows the Connect Health Data heading", () => {
    renderHealthStep();
    expect(screen.getByRole("heading", { name: "Connect Health Data" })).toBeInTheDocument();
  });

  it("shows the health data list items", () => {
    renderHealthStep();
    expect(screen.getByText("Steps")).toBeInTheDocument();
    expect(screen.getByText("Heart rate")).toBeInTheDocument();
  });

  it("shows a Skip for now option", () => {
    renderHealthStep();
    expect(
      screen.getByRole("button", { name: /Skip for now/i })
    ).toBeInTheDocument();
  });

  it("advances to step 5 when Skip is clicked", () => {
    renderHealthStep();
    fireEvent.click(screen.getByRole("button", { name: /Skip for now/i }));
    expect(screen.getByText(/You're all set/i)).toBeInTheDocument();
    expect(screen.getByText(/Step 5 of 5/i)).toBeInTheDocument();
  });
});

// ── Step 5: Done ──────────────────────────────────────────────────────────────

describe("Onboarding — Step 5 (Done)", () => {
  function renderDoneStep() {
    localStorage.setItem("ndw_onboarding_step", "5");
    localStorage.setItem("ndw_onboarding_path", "voice");
    renderWithProviders(<Onboarding />);
  }

  it("shows the You're all set heading", () => {
    renderDoneStep();
    expect(screen.getByText("You're all set")).toBeInTheDocument();
  });

  it("shows a Go to dashboard button", () => {
    renderDoneStep();
    expect(
      screen.getByRole("button", { name: /Go to dashboard/i })
    ).toBeInTheDocument();
  });

  it("shows step 5 of 5 in progress bar", () => {
    renderDoneStep();
    expect(screen.getByText(/Step 5 of 5/i)).toBeInTheDocument();
  });
});

// ── localStorage persistence ──────────────────────────────────────────────────

describe("Onboarding — localStorage persistence", () => {
  it("saves onboarding complete when Skip for now is clicked", () => {
    localStorage.clear();
    renderWithProviders(<Onboarding />);
    fireEvent.click(screen.getByText(/Skip for now/i));
    expect(localStorage.getItem("ndw_onboarding_complete")).toBe("true");
  });

  it("resumes from step 2 when ndw_onboarding_step is set to 2", () => {
    localStorage.setItem("ndw_onboarding_step", "2");
    renderWithProviders(<Onboarding />);
    expect(screen.getByText(/How do you want to start/i)).toBeInTheDocument();
  });

  it("saves path choice to localStorage", () => {
    localStorage.setItem("ndw_onboarding_step", "2");
    renderWithProviders(<Onboarding />);
    fireEvent.click(screen.getByRole("button", { name: /Quick Start \(Voice\)/i }));
    expect(localStorage.getItem("ndw_onboarding_path")).toBe("voice");
  });
});
