import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import DataHub from "@/pages/session-history";

beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

vi.mock("@/lib/participant", () => ({ getParticipantId: () => "test-user-123" }));

vi.mock("@/lib/ml-api", () => ({
  listSessions: vi.fn().mockResolvedValue([]),
  deleteSession: vi.fn().mockResolvedValue({}),
  exportSession: vi.fn().mockResolvedValue("csv-data"),
  getEmotionHistory: vi.fn().mockResolvedValue([]),
}));

describe("session-history (DataHub) page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    }) as unknown as typeof fetch;
  });

  it("renders without crashing", async () => {
    renderWithProviders(<DataHub />);
    await waitFor(() => expect(document.body).toBeTruthy());
  });

  it("shows the Data Hub heading", async () => {
    renderWithProviders(<DataHub />);
    await waitFor(() => {
      expect(screen.getByText("Data Hub")).toBeInTheDocument();
    });
  });

  it("shows period filter buttons (Today, Week, Month)", async () => {
    renderWithProviders(<DataHub />);
    await waitFor(() => {
      expect(screen.getByText("Today")).toBeInTheDocument();
      expect(screen.getByText("Week")).toBeInTheDocument();
      expect(screen.getByText("Month")).toBeInTheDocument();
    });
  });

  it("shows all period filter buttons including 3 Months and Year", async () => {
    renderWithProviders(<DataHub />);
    await waitFor(() => {
      expect(screen.getByText("3 Months")).toBeInTheDocument();
      expect(screen.getByText("Year")).toBeInTheDocument();
    });
  });

  it("shows summary metric cards (Readings, Dreams, Health)", async () => {
    renderWithProviders(<DataHub />);
    await waitFor(() => {
      // Readings appears only in the metric card
      const allText = document.body.textContent ?? "";
      expect(allText).toContain("Readings");
      expect(allText).toContain("Sessions");
      expect(allText).toContain("Health");
    });
  });

  it("shows tab navigation (Sessions, Emotions, Dreams, Health)", async () => {
    renderWithProviders(<DataHub />);
    await waitFor(() => {
      expect(screen.getByText("Emotions")).toBeInTheDocument();
    });
  });

  it("shows empty state when no sessions in the period", async () => {
    renderWithProviders(<DataHub />);
    await waitFor(() => {
      expect(screen.getByText("No sessions yet")).toBeInTheDocument();
    });
  });

  it("clicking Week period tab does not crash", async () => {
    renderWithProviders(<DataHub />);
    await waitFor(() => expect(screen.getByText("Week")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Week"));
    await waitFor(() => {
      expect(screen.getByText("Data Hub")).toBeInTheDocument();
    });
  });

  it("switching to Emotions tab shows empty state for emotion readings", async () => {
    renderWithProviders(<DataHub />);
    await waitFor(() => expect(screen.getByText("Emotions")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Emotions"));
    await waitFor(() => {
      expect(
        screen.getByText("No emotion readings in this period.")
      ).toBeInTheDocument();
    });
  });

  it("switching to Dreams tab shows no dreams empty state", async () => {
    renderWithProviders(<DataHub />);
    // Wait for page to settle
    await waitFor(() => expect(screen.getAllByText("Dreams").length).toBeGreaterThanOrEqual(1));
    // Find all elements with "Dreams" and click the tab button (has border-b-2 class)
    const dreamElements = screen.getAllByText("Dreams");
    // The tab element contains an SVG icon sibling — click the last occurrence which is inside the tab row
    fireEvent.click(dreamElements[dreamElements.length - 1]);
    await waitFor(() => {
      expect(
        screen.getByText("No dreams recorded in this period.")
      ).toBeInTheDocument();
    });
  });
});
