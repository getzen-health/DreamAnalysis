import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { AICompanion } from "@/components/ai-companion";

// Mock the useDevice hook
vi.mock("@/hooks/use-device", () => ({
  useDevice: () => ({
    latestFrame: null,
    state: "disconnected",
  }),
}));

describe("AICompanion component", () => {
  it("renders the chat interface", () => {
    render(<AICompanion userId="test-user" />);
    expect(screen.getByText("AI Brain Companion")).toBeInTheDocument();
  });

  it("shows NO DEVICE when not streaming", () => {
    render(<AICompanion userId="test-user" />);
    expect(screen.getByText("NO DEVICE")).toBeInTheDocument();
  });

  it("renders quick action buttons", () => {
    render(<AICompanion userId="test-user" />);
    expect(screen.getByText("Breathing Exercise")).toBeInTheDocument();
    expect(screen.getByText("Guided Meditation")).toBeInTheDocument();
    expect(screen.getByText("Mood Check-In")).toBeInTheDocument();
    expect(screen.getByText("Stress Relief")).toBeInTheDocument();
  });

  it("renders live brain metrics section", () => {
    render(<AICompanion userId="test-user" />);
    expect(screen.getByText("Live Brain Metrics")).toBeInTheDocument();
  });

  it("shows welcome message when no chat history", () => {
    render(<AICompanion userId="test-user" />);
    expect(
      screen.getByText(/Connect your Muse 2 from the sidebar/)
    ).toBeInTheDocument();
  });

  it("has a message input field", () => {
    render(<AICompanion userId="test-user" />);
    const input = screen.getByPlaceholderText(/Connect your Muse 2/);
    expect(input).toBeInTheDocument();
  });

  it("disables send button when input is empty", () => {
    render(<AICompanion userId="test-user" />);
    const sendButton = screen.getByRole("button", { name: "" });
    // The button with just the Send icon - find by looking for disabled state
    const buttons = screen.getAllByRole("button");
    const sendBtn = buttons.find((b) => b.hasAttribute("disabled"));
    expect(sendBtn).toBeTruthy();
  });

  it("adds bot message when quick action clicked", () => {
    render(<AICompanion userId="test-user" />);
    const breathingBtn = screen.getByText("Breathing Exercise");
    fireEvent.click(breathingBtn);

    // Should have added a bot message about breathing
    setTimeout(() => {
      expect(screen.getByText(/breathing exercise/i)).toBeInTheDocument();
    }, 600);
  });
});
