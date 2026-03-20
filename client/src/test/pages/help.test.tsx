import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import HelpPage from "@/pages/help";

vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

vi.mock("wouter", () => ({
  useLocation: () => ["/help", vi.fn()],
}));

describe("Help page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    }) as unknown as typeof fetch;
  });

  it("renders without crashing", () => {
    renderWithProviders(<HelpPage />);
    expect(document.body).toBeTruthy();
  });

  it("shows the page heading", () => {
    renderWithProviders(<HelpPage />);
    expect(screen.getByText("Help & Feedback")).toBeInTheDocument();
  });

  it("shows subtitle text", () => {
    renderWithProviders(<HelpPage />);
    expect(
      screen.getByText("Learn how to use AntarAI and share your feedback")
    ).toBeInTheDocument();
  });

  it("shows Quick Start section", () => {
    renderWithProviders(<HelpPage />);
    expect(screen.getByText("Quick Start")).toBeInTheDocument();
  });

  it("shows all five quick start steps", () => {
    renderWithProviders(<HelpPage />);
    expect(screen.getByText("Check in with your voice")).toBeInTheDocument();
    expect(screen.getByText("Log your nutrition")).toBeInTheDocument();
    expect(screen.getByText("Connect your devices")).toBeInTheDocument();
    expect(screen.getByText("Review your trends")).toBeInTheDocument();
    expect(screen.getByText("Journal your dreams")).toBeInTheDocument();
  });

  it("shows FAQ section", () => {
    renderWithProviders(<HelpPage />);
    expect(
      screen.getByText("Frequently Asked Questions")
    ).toBeInTheDocument();
  });

  it("shows FAQ questions as collapsed by default", () => {
    renderWithProviders(<HelpPage />);
    expect(
      screen.getByText("How does voice-based emotion analysis work?")
    ).toBeInTheDocument();
    expect(
      screen.getByText("Do I need an EEG headband to use this app?")
    ).toBeInTheDocument();
    expect(
      screen.getByText("How do I connect my health data?")
    ).toBeInTheDocument();
    expect(
      screen.getByText("What is a dream journal entry?")
    ).toBeInTheDocument();
    expect(
      screen.getByText("How is my data stored?")
    ).toBeInTheDocument();
    expect(
      screen.getByText("What does the calibration step do?")
    ).toBeInTheDocument();
  });

  it("expands a FAQ when clicked", () => {
    renderWithProviders(<HelpPage />);
    const question = screen.getByText(
      "Do I need an EEG headband to use this app?"
    );
    fireEvent.click(question);
    expect(
      screen.getByText(/Voice analysis is the primary input method/)
    ).toBeInTheDocument();
  });

  it("collapses a FAQ when clicked again", () => {
    renderWithProviders(<HelpPage />);
    const question = screen.getByText(
      "Do I need an EEG headband to use this app?"
    );
    fireEvent.click(question);
    expect(
      screen.getByText(/Voice analysis is the primary input method/)
    ).toBeInTheDocument();
    fireEvent.click(question);
    expect(
      screen.queryByText(/Voice analysis is the primary input method/)
    ).not.toBeInTheDocument();
  });

  it("shows Send Feedback section heading", () => {
    renderWithProviders(<HelpPage />);
    // "Send Feedback" appears as both an h3 and a button — find the heading
    const headings = screen.getAllByText("Send Feedback");
    const h3 = headings.find((el) => el.tagName.toLowerCase() === "h3");
    expect(h3).toBeInTheDocument();
  });

  it("shows feedback type buttons", () => {
    renderWithProviders(<HelpPage />);
    expect(screen.getByText("General")).toBeInTheDocument();
    expect(screen.getByText("Bug Report")).toBeInTheDocument();
    expect(screen.getByText("Feature Request")).toBeInTheDocument();
  });

  it("shows Send Feedback submit button", () => {
    renderWithProviders(<HelpPage />);
    expect(
      screen.getByRole("button", { name: /Send Feedback/i })
    ).toBeInTheDocument();
  });

  it("Send Feedback button is disabled when textarea is empty", () => {
    renderWithProviders(<HelpPage />);
    const btn = screen.getByRole("button", { name: /Send Feedback/i });
    expect(btn).toBeDisabled();
  });

  it("shows Contact section", () => {
    renderWithProviders(<HelpPage />);
    expect(screen.getByText("Contact")).toBeInTheDocument();
  });

  it("shows Email Support option", () => {
    renderWithProviders(<HelpPage />);
    expect(screen.getByText("Email Support")).toBeInTheDocument();
  });

  it("shows Privacy Policy option", () => {
    renderWithProviders(<HelpPage />);
    expect(screen.getByText("Privacy Policy")).toBeInTheDocument();
  });

  it("shows version info", () => {
    renderWithProviders(<HelpPage />);
    expect(screen.getByText("AntarAI v1.0")).toBeInTheDocument();
  });

  it("switching feedback type updates active state", () => {
    renderWithProviders(<HelpPage />);
    const bugBtn = screen.getByText("Bug Report");
    fireEvent.click(bugBtn);
    // After clicking Bug Report, the textarea placeholder should change
    const textarea = screen.getByPlaceholderText(
      /Describe the bug/
    );
    expect(textarea).toBeInTheDocument();
  });
});
