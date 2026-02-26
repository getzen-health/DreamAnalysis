import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import ResearchHub from "@/pages/research-hub";

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

vi.mock("wouter", () => ({
  useLocation: () => ["/research", vi.fn()],
  Link: (props: any) => <a href={props.href}>{props.children}</a>,
}));

vi.mock("@/lib/participant", () => ({
  getParticipantId: () => "test-participant-uuid",
  getStudyCode: () => null,
  saveStudyCode: vi.fn(),
  clearParticipantIdentity: vi.fn(),
}));

// Stub fetch: status query returns not-enrolled, correlation query returns []
beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockImplementation((url: string) => {
      if (String(url).includes("/api/research/correlation")) {
        return Promise.resolve({
          ok: true,
          json: async () => [],
        });
      }
      // Default: study status — not enrolled
      return Promise.resolve({
        ok: true,
        json: async () => ({ enrolled: false, study_code: null }),
      });
    })
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ResearchHub page — not enrolled", () => {
  it("renders without crashing", async () => {
    renderWithProviders(<ResearchHub />);
  });

  it("shows the study headline after loading", async () => {
    renderWithProviders(<ResearchHub />);
    // findByText waits for the async query to resolve and the content to appear
    expect(
      await screen.findByText("The Emotional Day-Night Cycle", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows the 30-day study description", async () => {
    renderWithProviders(<ResearchHub />);
    expect(
      await screen.findByText(/30-day EEG \+ dream study/, {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows volunteer study bullet", async () => {
    renderWithProviders(<ResearchHub />);
    expect(
      await screen.findByText(/Volunteer study — contribute to dream science/, {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows the Muse 2 requirement bullet", async () => {
    renderWithProviders(<ResearchHub />);
    expect(
      await screen.findByText(/Requires your own Muse 2 EEG headset/, {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows the food-mood-dream tracking bullet", async () => {
    renderWithProviders(<ResearchHub />);
    expect(
      await screen.findByText(/Photo your meals/, {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows a Join the Study button", async () => {
    renderWithProviders(<ResearchHub />);
    expect(
      await screen.findByRole("button", { name: /Join the Study/i }, { timeout: 3000 })
    ).toBeInTheDocument();
  });
});
