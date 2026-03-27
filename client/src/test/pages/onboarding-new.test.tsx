import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import OnboardingNew from "@/pages/onboarding-new";

const navigate = vi.fn();
const startRecording = vi.fn();

const mockVoiceEmotion = vi.hoisted(() => ({
  isRecording: false,
  isAnalyzing: false,
  lastResult: null as null | {
    emotion: string;
    confidence: number;
    valence: number;
    model_type: string;
    arousal: number;
    probabilities: Record<string, number>;
  },
  error: null as string | null,
}));

vi.mock("wouter", () => ({
  useLocation: () => ["/onboarding-new", navigate],
  Link: (props: any) => <a href={props.href}>{props.children}</a>,
}));

vi.mock("@/hooks/use-auth", () => ({
  useAuth: () => ({ user: null }),
}));

vi.mock("@/hooks/use-voice-emotion", () => ({
  useVoiceEmotion: () => ({
    startRecording,
    stopRecording: vi.fn(),
    ...mockVoiceEmotion,
  }),
}));

describe("OnboardingNew page", () => {
  beforeEach(() => {
    navigate.mockReset();
    startRecording.mockReset();
    mockVoiceEmotion.isRecording = false;
    mockVoiceEmotion.isAnalyzing = false;
    mockVoiceEmotion.lastResult = null;
    mockVoiceEmotion.error = null;
  });

  it("shows the two-path onboarding choice", () => {
    renderWithProviders(<OnboardingNew />);
    expect(screen.getByText("Choose your setup path")).toBeInTheDocument();
    expect(screen.getByText("Voice + Health")).toBeInTheDocument();
    expect(screen.getByText("EEG Headband")).toBeInTheDocument();
  });

  it("opens the voice stage and starts recording", () => {
    renderWithProviders(<OnboardingNew />);
    fireEvent.click(screen.getByRole("button", { name: /Start with voice/i }));
    expect(screen.getByText("Voice + Health setup")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Start 10-second voice analysis/i }));
    expect(startRecording).toHaveBeenCalledTimes(1);
  });

  it("shows the result stage when a voice result exists", () => {
    mockVoiceEmotion.lastResult = {
      emotion: "happy",
      confidence: 0.82,
      valence: 0.4,
      model_type: "voice_emotion2vec_large",
      arousal: 0.5,
      probabilities: {},
    };
    renderWithProviders(<OnboardingNew />);
    expect(screen.getByText("Your first state read is ready")).toBeInTheDocument();
    expect(screen.getByText("happy")).toBeInTheDocument();
    expect(screen.getByText("82%")).toBeInTheDocument();
  });
});
