import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import WelcomeIntro from "@/pages/welcome-intro";

const navigateMock = vi.fn();
vi.mock("wouter", () => ({
  useLocation: () => ["/welcome-intro", navigateMock],
}));

beforeEach(() => {
  navigateMock.mockClear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("WelcomeIntro — Step 1: Health Connect", () => {
  it("renders without crashing", () => {
    renderWithProviders(<WelcomeIntro />);
  });

  it("shows the Health Connect heading on step 1", () => {
    renderWithProviders(<WelcomeIntro />);
    expect(
      screen.getByText("Connect Your Health Data")
    ).toBeInTheDocument();
  });

  it("shows the health sync description", () => {
    renderWithProviders(<WelcomeIntro />);
    expect(
      screen.getByText(/Sync steps, heart rate, and sleep/)
    ).toBeInTheDocument();
  });

  it("shows the 'What we sync' list items", () => {
    renderWithProviders(<WelcomeIntro />);
    expect(screen.getByText("Steps")).toBeInTheDocument();
    expect(screen.getByText("Heart rate")).toBeInTheDocument();
    expect(screen.getByText("Sleep stages")).toBeInTheDocument();
    expect(screen.getByText("Active energy")).toBeInTheDocument();
  });

  it("shows the Connect Health Data button", () => {
    renderWithProviders(<WelcomeIntro />);
    expect(
      screen.getByRole("button", { name: "Connect Health Data" })
    ).toBeInTheDocument();
  });
});

describe("WelcomeIntro — dot indicators", () => {
  it("shows 3 dot indicators with correct aria labels", () => {
    renderWithProviders(<WelcomeIntro />);
    expect(screen.getByLabelText("Go to Health")).toBeInTheDocument();
    expect(screen.getByLabelText("Go to Voice")).toBeInTheDocument();
    expect(screen.getByLabelText("Go to Devices")).toBeInTheDocument();
  });

  it("clicking a dot indicator navigates to that step", () => {
    renderWithProviders(<WelcomeIntro />);
    fireEvent.click(screen.getByLabelText("Go to Devices"));
    expect(
      screen.getByText("Do You Have Any Health Devices?")
    ).toBeInTheDocument();
  });
});

describe("WelcomeIntro — navigation buttons", () => {
  it("shows Next button on the first step", () => {
    renderWithProviders(<WelcomeIntro />);
    expect(
      screen.getByRole("button", { name: /Next/ })
    ).toBeInTheDocument();
  });

  it("Back button is disabled on the first step", () => {
    renderWithProviders(<WelcomeIntro />);
    const backBtn = screen.getByRole("button", { name: /Back/ });
    expect(backBtn).toBeDisabled();
  });

  it("Next button advances to step 2 (Voice)", () => {
    renderWithProviders(<WelcomeIntro />);
    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
    expect(
      screen.getByText("Enable Voice Check-ins")
    ).toBeInTheDocument();
  });

  it("Back button is enabled on step 2", () => {
    renderWithProviders(<WelcomeIntro />);
    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
    const backBtn = screen.getByRole("button", { name: /Back/ });
    expect(backBtn).not.toBeDisabled();
  });

  it("Back button returns to step 1 from step 2", () => {
    renderWithProviders(<WelcomeIntro />);
    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
    expect(screen.getByText("Enable Voice Check-ins")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Back/ }));
    expect(screen.getByText("Connect Your Health Data")).toBeInTheDocument();
  });

  it("shows 'Continue to Dashboard' on the last step instead of Next", () => {
    renderWithProviders(<WelcomeIntro />);
    // Advance to step 2
    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
    // Advance to step 3
    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
    expect(
      screen.getByRole("button", { name: /Continue to Dashboard/ })
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Next$/ })).not.toBeInTheDocument();
  });
});

describe("WelcomeIntro — Step 2: Voice Check-in", () => {
  it("shows Voice step heading and description", () => {
    renderWithProviders(<WelcomeIntro />);
    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
    expect(screen.getByText("Enable Voice Check-ins")).toBeInTheDocument();
    expect(
      screen.getByText(/A quick 3-second test to make sure your microphone works/)
    ).toBeInTheDocument();
  });

  it("shows Test Microphone button", () => {
    renderWithProviders(<WelcomeIntro />);
    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
    expect(
      screen.getByRole("button", { name: "Test Microphone" })
    ).toBeInTheDocument();
  });
});

describe("WelcomeIntro — Step 3: Device selection", () => {
  function goToStep3() {
    renderWithProviders(<WelcomeIntro />);
    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
  }

  it("shows device step heading", () => {
    goToStep3();
    expect(
      screen.getByText("Do You Have Any Health Devices?")
    ).toBeInTheDocument();
  });

  it("shows device step description", () => {
    goToStep3();
    expect(
      screen.getByText(/Select any devices you own/)
    ).toBeInTheDocument();
  });

  it("shows Muse and Fitness Tracker device options", () => {
    goToStep3();
    expect(screen.getByText("Muse 2 / Muse S")).toBeInTheDocument();
    expect(screen.getByText("Fitness Tracker / Smartwatch")).toBeInTheDocument();
  });

  it("shows no-device reassurance when nothing is selected", () => {
    goToStep3();
    expect(
      screen.getByText(
        /Voice check-ins and manual logging work great on their own/
      )
    ).toBeInTheDocument();
  });

  it("shows Muse info card when Muse is selected", () => {
    goToStep3();
    fireEvent.click(screen.getByText("Muse 2 / Muse S"));
    expect(
      screen.getByText(/connect your Muse headband from the Settings page/)
    ).toBeInTheDocument();
  });

  it("hides no-device reassurance when a device is selected", () => {
    goToStep3();
    fireEvent.click(screen.getByText("Muse 2 / Muse S"));
    expect(
      screen.queryByText(
        /Voice check-ins and manual logging work great on their own/
      )
    ).not.toBeInTheDocument();
  });

  it("toggles device selection off when clicked again", () => {
    goToStep3();
    fireEvent.click(screen.getByText("Muse 2 / Muse S"));
    // Muse info card visible
    expect(
      screen.getByText(/connect your Muse headband from the Settings page/)
    ).toBeInTheDocument();
    // Deselect
    fireEvent.click(screen.getByText("Muse 2 / Muse S"));
    // Back to no-device reassurance
    expect(
      screen.getByText(
        /Voice check-ins and manual logging work great on their own/
      )
    ).toBeInTheDocument();
  });
});

describe("WelcomeIntro — Skip for now", () => {
  it("shows 'Skip for now' button", () => {
    renderWithProviders(<WelcomeIntro />);
    expect(screen.getByText("Skip for now")).toBeInTheDocument();
  });

  it("Skip sets localStorage and navigates to /", () => {
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem");
    renderWithProviders(<WelcomeIntro />);
    fireEvent.click(screen.getByText("Skip for now"));
    expect(setItemSpy).toHaveBeenCalledWith("onboarding_complete", "true");
    expect(navigateMock).toHaveBeenCalledWith("/");
  });
});

describe("WelcomeIntro — finish flow", () => {
  it("'Continue to Dashboard' on last step sets localStorage and navigates to /", () => {
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem");
    renderWithProviders(<WelcomeIntro />);
    // Navigate to last step
    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
    // Click finish
    fireEvent.click(
      screen.getByRole("button", { name: /Continue to Dashboard/ })
    );
    expect(setItemSpy).toHaveBeenCalledWith("onboarding_complete", "true");
    expect(navigateMock).toHaveBeenCalledWith("/");
  });

  it("Skip for now works from step 2 as well", () => {
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem");
    renderWithProviders(<WelcomeIntro />);
    fireEvent.click(screen.getByRole("button", { name: /Next/ }));
    fireEvent.click(screen.getByText("Skip for now"));
    expect(setItemSpy).toHaveBeenCalledWith("onboarding_complete", "true");
    expect(navigateMock).toHaveBeenCalledWith("/");
  });
});
