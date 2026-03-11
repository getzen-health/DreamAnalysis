import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import WelcomeIntro from "@/pages/welcome-intro";

const navigateMock = vi.fn();
vi.mock("wouter", () => ({
  useLocation: () => ["/welcome-intro", navigateMock],
}));

describe("WelcomeIntro", () => {
  it("renders without crashing", () => {
    renderWithProviders(<WelcomeIntro />);
  });

  it("renders the first card title", () => {
    renderWithProviders(<WelcomeIntro />);
    expect(screen.getByText("Welcome to AntarAI")).toBeInTheDocument();
  });

  it("renders the first card description", () => {
    renderWithProviders(<WelcomeIntro />);
    expect(
      screen.getByText(
        /Track your emotions, stress, and brain activity/
      )
    ).toBeInTheDocument();
  });

  it("shows 3 dot indicators", () => {
    renderWithProviders(<WelcomeIntro />);
    const dots = [
      screen.getByLabelText("Go to slide 1"),
      screen.getByLabelText("Go to slide 2"),
      screen.getByLabelText("Go to slide 3"),
    ];
    expect(dots).toHaveLength(3);
  });

  it("shows Next button on the first card", () => {
    renderWithProviders(<WelcomeIntro />);
    expect(screen.getByText("Next")).toBeInTheDocument();
  });

  it("shows Skip button", () => {
    renderWithProviders(<WelcomeIntro />);
    expect(screen.getByText("Skip")).toBeInTheDocument();
  });

  it("Back button is disabled on the first card", () => {
    renderWithProviders(<WelcomeIntro />);
    const backBtn = screen.getByText("Back").closest("button");
    expect(backBtn).toBeDisabled();
  });

  it("Next button advances to card 2", () => {
    renderWithProviders(<WelcomeIntro />);
    fireEvent.click(screen.getByText("Next"));
    expect(screen.getByText("Voice, EEG, or Both")).toBeInTheDocument();
    expect(
      screen.getByText(
        /Start with just your voice for mood and stress analysis/
      )
    ).toBeInTheDocument();
  });

  it("Back button is enabled on card 2", () => {
    renderWithProviders(<WelcomeIntro />);
    fireEvent.click(screen.getByText("Next"));
    const backBtn = screen.getByText("Back").closest("button");
    expect(backBtn).not.toBeDisabled();
  });

  it("Back button returns to card 1 from card 2", () => {
    renderWithProviders(<WelcomeIntro />);
    fireEvent.click(screen.getByText("Next"));
    expect(screen.getByText("Voice, EEG, or Both")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Back"));
    expect(screen.getByText("Welcome to AntarAI")).toBeInTheDocument();
  });

  it("shows Get Started on the last card", () => {
    renderWithProviders(<WelcomeIntro />);
    // Advance to card 2
    fireEvent.click(screen.getByText("Next"));
    // Advance to card 3
    fireEvent.click(screen.getByText("Next"));
    expect(screen.getByText("Let's Explore")).toBeInTheDocument();
    expect(screen.getByText("Get Started")).toBeInTheDocument();
    // "Next" should no longer appear
    expect(screen.queryByText("Next")).not.toBeInTheDocument();
  });

  it("clicking a dot indicator navigates to that card", () => {
    renderWithProviders(<WelcomeIntro />);
    fireEvent.click(screen.getByLabelText("Go to slide 3"));
    expect(screen.getByText("Let's Explore")).toBeInTheDocument();
    expect(screen.getByText("Get Started")).toBeInTheDocument();
  });

  it("Get Started sets localStorage and navigates to /intent", () => {
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem");
    renderWithProviders(<WelcomeIntro />);
    // Navigate to last card
    fireEvent.click(screen.getByText("Next"));
    fireEvent.click(screen.getByText("Next"));
    // Click Get Started
    fireEvent.click(screen.getByText("Get Started"));
    expect(setItemSpy).toHaveBeenCalledWith("onboarding_complete", "true");
    expect(navigateMock).toHaveBeenCalledWith("/intent");
    setItemSpy.mockRestore();
  });

  it("Skip sets localStorage and navigates to /intent", () => {
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem");
    renderWithProviders(<WelcomeIntro />);
    fireEvent.click(screen.getByText("Skip"));
    expect(setItemSpy).toHaveBeenCalledWith("onboarding_complete", "true");
    expect(navigateMock).toHaveBeenCalledWith("/intent");
    setItemSpy.mockRestore();
  });
});
