import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import ArchitectureGuide from "@/pages/architecture-guide";

vi.mock("@/hooks/use-device", () => ({
  useDevice: () => ({
    latestFrame: null,
    state: "disconnected",
    deviceStatus: null,
  }),
}));

vi.mock("@/hooks/use-theme", () => ({
  useTheme: () => ({ theme: "dark", setTheme: vi.fn() }),
}));

describe("ArchitectureGuide page", () => {
  it("renders the project heading", () => {
    renderWithProviders(<ArchitectureGuide />);

    expect(
      screen.getByText(/AntarAI is a wearable EEG research and product platform/i)
    ).toBeInTheDocument();
  });

  it("renders the GitHub snapshot section", () => {
    renderWithProviders(<ArchitectureGuide />);

    expect(screen.getByText(/Snapshot: March 9, 2026/i)).toBeInTheDocument();
  });
});
