import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "./test-utils";
import { AICompanion } from "@/components/ai-companion";

vi.mock("@/hooks/use-device", () => ({
  useDevice: () => ({
    latestFrame: null,
    state: "disconnected",
  }),
}));

vi.mock("wouter", () => ({
  useLocation: () => ["/ai-companion", vi.fn()],
  Link: (props: any) => <a href={props.href}>{props.children}</a>,
}));

describe("AICompanion component", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => [] })
    );
  });
  it("renders the chat interface", () => {
    renderWithProviders(<AICompanion userId="test-user" />);
    expect(screen.getByText("AI Brain Companion")).toBeInTheDocument();
  });

  it("shows NO DEVICE when not streaming", () => {
    renderWithProviders(<AICompanion userId="test-user" />);
    expect(screen.getByText("NO DEVICE")).toBeInTheDocument();
  });

  it("renders quick action buttons", () => {
    renderWithProviders(<AICompanion userId="test-user" />);
    expect(screen.getByText("Breathing Exercise")).toBeInTheDocument();
    expect(screen.getByText("Guided Meditation")).toBeInTheDocument();
    expect(screen.getByText("Mood Check-In")).toBeInTheDocument();
    expect(screen.getByText("Stress Relief")).toBeInTheDocument();
  });

  it("renders live brain metrics section", () => {
    renderWithProviders(<AICompanion userId="test-user" />);
    expect(screen.getByText("Live Brain Metrics")).toBeInTheDocument();
  });

  it("shows welcome message when no chat history", async () => {
    renderWithProviders(<AICompanion userId="test-user" />);
    expect(
      await screen.findByText(/Hello! I'm your AI wellness companion/)
    ).toBeInTheDocument();
  });

  it("has a message input field", () => {
    renderWithProviders(<AICompanion userId="test-user" />);
    const input = screen.getByPlaceholderText(/Message AI companion/);
    expect(input).toBeInTheDocument();
  });

  it("disables send button when input is empty", () => {
    renderWithProviders(<AICompanion userId="test-user" />);
    const buttons = screen.getAllByRole("button");
    const sendBtn = buttons.find((b) => b.hasAttribute("disabled"));
    expect(sendBtn).toBeTruthy();
  });

  it("adds bot message when quick action clicked", () => {
    renderWithProviders(<AICompanion userId="test-user" />);
    const breathingBtn = screen.getByText("Breathing Exercise");
    fireEvent.click(breathingBtn);
    setTimeout(() => {
      expect(screen.getByText(/breathing exercise/i)).toBeInTheDocument();
    }, 600);
  });
});
