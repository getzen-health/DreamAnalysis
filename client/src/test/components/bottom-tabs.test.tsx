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
    expect(screen.getByText("Emotions")).toBeInTheDocument();
    expect(screen.getByText("Dreams")).toBeInTheDocument();
    expect(screen.getByText("AI Chat")).toBeInTheDocument();
    expect(screen.getByText("Settings")).toBeInTheDocument();
  });

  it("renders correct href for each tab", () => {
    renderWithProviders(<BottomTabs />);
    const links = screen.getAllByRole("link");
    const hrefs = links.map((a) => a.getAttribute("href"));
    expect(hrefs).toContain("/");
    expect(hrefs).toContain("/emotions");
    expect(hrefs).toContain("/dreams");
    expect(hrefs).toContain("/ai-companion");
    expect(hrefs).toContain("/settings");
  });

  it("renders a nav element", () => {
    renderWithProviders(<BottomTabs />);
    expect(screen.getByRole("navigation")).toBeInTheDocument();
  });

  it("highlights Home tab when location is /", () => {
    currentLocation = "/";
    renderWithProviders(<BottomTabs />);
    const homeLink = screen.getByText("Home").closest("a");
    expect(homeLink?.className).toContain("text-primary");
  });

  it("does not highlight non-active tabs when on Home", () => {
    currentLocation = "/";
    renderWithProviders(<BottomTabs />);
    const emotionsLink = screen.getByText("Emotions").closest("a");
    expect(emotionsLink?.className).not.toContain("text-primary");
  });

  it("highlights Emotions tab when location is /emotions", () => {
    currentLocation = "/emotions";
    renderWithProviders(<BottomTabs />);
    const emotionsLink = screen.getByText("Emotions").closest("a");
    expect(emotionsLink?.className).toContain("text-primary");
    const homeLink = screen.getByText("Home").closest("a");
    expect(homeLink?.className).not.toContain("text-primary");
  });

  it("highlights Dreams tab when location is /dreams", () => {
    currentLocation = "/dreams";
    renderWithProviders(<BottomTabs />);
    const dreamsLink = screen.getByText("Dreams").closest("a");
    expect(dreamsLink?.className).toContain("text-primary");
  });

  it("highlights AI Chat tab when location is /ai-companion", () => {
    currentLocation = "/ai-companion";
    renderWithProviders(<BottomTabs />);
    const chatLink = screen.getByText("AI Chat").closest("a");
    expect(chatLink?.className).toContain("text-primary");
  });

  it("highlights Settings tab when location is /settings", () => {
    currentLocation = "/settings";
    renderWithProviders(<BottomTabs />);
    const settingsLink = screen.getByText("Settings").closest("a");
    expect(settingsLink?.className).toContain("text-primary");
  });
});
