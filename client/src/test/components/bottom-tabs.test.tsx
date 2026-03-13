import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, cleanup } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { BottomTabs } from "@/components/bottom-tabs";

vi.mock("@/lib/haptics", () => ({
  hapticLight: vi.fn(),
}));

/** Mutable location that the hoisted mock reads at call time. */
let currentLocation = "/";

vi.mock("wouter", () => ({
  useLocation: () => [currentLocation, vi.fn()],
  Link: (props: any) => (
    <a href={props.href} onClick={props.onClick} className={props.className}>
      {props.children}
    </a>
  ),
}));

beforeEach(() => {
  cleanup();
  currentLocation = "/";
});

describe("BottomTabs", () => {
  it("renders without crashing", () => {
    renderWithProviders(<BottomTabs />);
  });

  it("renders all 5 tab labels", () => {
    renderWithProviders(<BottomTabs />);
    expect(screen.getByText("Home")).toBeInTheDocument();
    expect(screen.getByText("Brain")).toBeInTheDocument();
    expect(screen.getByText("Emotions")).toBeInTheDocument();
    expect(screen.getByText("Health")).toBeInTheDocument();
    expect(screen.getByText("Profile")).toBeInTheDocument();
  });

  it("renders correct href for each tab", () => {
    renderWithProviders(<BottomTabs />);
    const links = screen.getAllByRole("link");
    const hrefs = links.map((a) => a.getAttribute("href"));
    expect(hrefs).toContain("/");
    expect(hrefs).toContain("/brain-monitor");
    expect(hrefs).toContain("/emotions");
    expect(hrefs).toContain("/health-analytics");
    expect(hrefs).toContain("/settings");
  });

  it("renders a nav element", () => {
    renderWithProviders(<BottomTabs />);
    expect(screen.getByRole("navigation")).toBeInTheDocument();
  });

  it("highlights Home tab when location is /", () => {
    currentLocation = "/";
    renderWithProviders(<BottomTabs />);
    // The text-primary class is on the label span, not the link
    const homeLabel = screen.getByText("Home");
    expect(homeLabel.className).toContain("text-primary");
  });

  it("does not highlight non-active tabs when on Home", () => {
    currentLocation = "/";
    renderWithProviders(<BottomTabs />);
    const brainLabel = screen.getByText("Brain");
    expect(brainLabel.className).not.toContain("text-primary");
  });

  it("highlights Brain tab when location is /brain-monitor", () => {
    currentLocation = "/brain-monitor";
    renderWithProviders(<BottomTabs />);
    const brainLabel = screen.getByText("Brain");
    expect(brainLabel.className).toContain("text-primary");
    const homeLabel = screen.getByText("Home");
    expect(homeLabel.className).not.toContain("text-primary");
  });

  it("highlights Emotions tab when location is /emotions", () => {
    currentLocation = "/emotions";
    renderWithProviders(<BottomTabs />);
    const emotionsLabel = screen.getByText("Emotions");
    expect(emotionsLabel.className).toContain("text-primary");
  });

  it("highlights Health tab when location is /health-analytics", () => {
    currentLocation = "/health-analytics";
    renderWithProviders(<BottomTabs />);
    const healthLabel = screen.getByText("Health");
    expect(healthLabel.className).toContain("text-primary");
  });

  it("highlights Profile tab when location is /settings", () => {
    currentLocation = "/settings";
    renderWithProviders(<BottomTabs />);
    const profileLabel = screen.getByText("Profile");
    expect(profileLabel.className).toContain("text-primary");
  });
});
