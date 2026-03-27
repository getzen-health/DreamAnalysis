import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, cleanup } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { BottomTabs } from "@/components/bottom-tabs";

vi.mock("@/lib/haptics", () => ({
  hapticLight: vi.fn(),
  hapticMedium: vi.fn(),
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

  it("renders all 4 tab labels", () => {
    renderWithProviders(<BottomTabs />);
    expect(screen.getByText("Today")).toBeInTheDocument();
    expect(screen.getByText("Discover")).toBeInTheDocument();
    expect(screen.getByText("AI Chat")).toBeInTheDocument();
    expect(screen.getByText("You")).toBeInTheDocument();
  });

  it("renders exactly 4 tabs (not 5)", () => {
    renderWithProviders(<BottomTabs />);
    const links = screen.getAllByRole("link");
    expect(links).toHaveLength(4);
  });

  it("renders correct href for each tab", () => {
    renderWithProviders(<BottomTabs />);
    const links = screen.getAllByRole("link");
    const hrefs = links.map((a) => a.getAttribute("href"));
    expect(hrefs).toContain("/");
    expect(hrefs).toContain("/discover");
    expect(hrefs).toContain("/ai-companion");
    expect(hrefs).toContain("/you");
  });

  it("renders a nav element", () => {
    renderWithProviders(<BottomTabs />);
    expect(screen.getByRole("navigation")).toBeInTheDocument();
  });

  it("highlights Today tab when location is /", () => {
    currentLocation = "/";
    renderWithProviders(<BottomTabs />);
    const todayLabel = screen.getByText("Today");
    expect(todayLabel.className).toContain("text-emerald-500");
  });

  it("does not highlight non-active tabs when on Today", () => {
    currentLocation = "/";
    renderWithProviders(<BottomTabs />);
    const discoverLabel = screen.getByText("Discover");
    expect(discoverLabel.className).not.toContain("text-emerald-500");
  });

  it("highlights Discover tab when location is /discover", () => {
    currentLocation = "/discover";
    renderWithProviders(<BottomTabs />);
    const discoverLabel = screen.getByText("Discover");
    expect(discoverLabel.className).toContain("text-emerald-500");
    const todayLabel = screen.getByText("Today");
    expect(todayLabel.className).not.toContain("text-emerald-500");
  });

  it("highlights Discover tab when location is /nutrition (alias)", () => {
    currentLocation = "/nutrition";
    renderWithProviders(<BottomTabs />);
    const discoverLabel = screen.getByText("Discover");
    expect(discoverLabel.className).toContain("text-emerald-500");
  });

  it("highlights You tab when location is /you", () => {
    currentLocation = "/you";
    renderWithProviders(<BottomTabs />);
    const youLabel = screen.getByText("You");
    expect(youLabel.className).toContain("text-emerald-500");
  });

  it("highlights You tab when location is /settings (alias path)", () => {
    currentLocation = "/settings";
    renderWithProviders(<BottomTabs />);
    const youLabel = screen.getByText("You");
    expect(youLabel.className).toContain("text-emerald-500");
  });
});
