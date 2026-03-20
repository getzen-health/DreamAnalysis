import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import ExportPage from "@/pages/export";

beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

vi.mock("@/lib/participant", () => ({
  getParticipantId: () => "test-user-123",
}));

vi.mock("@/lib/ml-api", () => ({
  listSessions: vi.fn().mockResolvedValue([]),
  exportSession: vi.fn().mockResolvedValue("header\nrow1"),
}));

vi.mock("@/lib/queryClient", () => ({
  resolveUrl: (path: string) => `http://localhost:5000${path}`,
}));

describe("Export page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
      blob: async () => new Blob(["csv-data"], { type: "text/csv" }),
    }) as unknown as typeof fetch;
  });

  it("renders without crashing", async () => {
    renderWithProviders(<ExportPage />);
    await waitFor(() => {
      expect(document.body).toBeTruthy();
    });
  });

  it("shows the page heading", async () => {
    renderWithProviders(<ExportPage />);
    await waitFor(() => {
      expect(screen.getByText("Export Data")).toBeInTheDocument();
    });
  });

  it("shows subtitle text", async () => {
    renderWithProviders(<ExportPage />);
    await waitFor(() => {
      expect(
        screen.getByText("Download your data in CSV format")
      ).toBeInTheDocument();
    });
  });

  it("shows Bulk Downloads section", async () => {
    renderWithProviders(<ExportPage />);
    await waitFor(() => {
      expect(screen.getByText("Bulk Downloads")).toBeInTheDocument();
    });
  });

  it("shows Per-Session Downloads section", async () => {
    renderWithProviders(<ExportPage />);
    await waitFor(() => {
      expect(screen.getByText("Per-Session Downloads")).toBeInTheDocument();
    });
  });

  it("shows Download All Sessions button", async () => {
    renderWithProviders(<ExportPage />);
    await waitFor(() => {
      expect(screen.getByText("Download All Sessions")).toBeInTheDocument();
    });
  });

  it("shows Download Health Data button", async () => {
    renderWithProviders(<ExportPage />);
    await waitFor(() => {
      expect(screen.getByText("Download Health Data")).toBeInTheDocument();
    });
  });

  it("shows Download Dream Analysis button", async () => {
    renderWithProviders(<ExportPage />);
    await waitFor(() => {
      expect(screen.getByText("Download Dream Analysis")).toBeInTheDocument();
    });
  });

  it("shows empty state when no sessions exist", async () => {
    renderWithProviders(<ExportPage />);
    await waitFor(() => {
      expect(
        screen.getByText("No sessions to export yet.")
      ).toBeInTheDocument();
    });
  });

  it("shows empty state description", async () => {
    renderWithProviders(<ExportPage />);
    await waitFor(() => {
      expect(
        screen.getByText(
          "Complete a voice analysis or EEG session to generate data."
        )
      ).toBeInTheDocument();
    });
  });

  it("Download All Sessions button is disabled when no sessions", async () => {
    renderWithProviders(<ExportPage />);
    await waitFor(() => {
      const btn = screen.getByText("Download All Sessions").closest("button");
      expect(btn).toBeDisabled();
    });
  });

  it("Download Health Data button is not disabled", async () => {
    renderWithProviders(<ExportPage />);
    await waitFor(() => {
      const btn = screen.getByText("Download Health Data").closest("button");
      expect(btn).not.toBeDisabled();
    });
  });

  it("Download Dream Analysis button is not disabled", async () => {
    renderWithProviders(<ExportPage />);
    await waitFor(() => {
      const btn = screen
        .getByText("Download Dream Analysis")
        .closest("button");
      expect(btn).not.toBeDisabled();
    });
  });

  it("shows health data description text", async () => {
    renderWithProviders(<ExportPage />);
    await waitFor(() => {
      expect(
        screen.getByText("Heart rate, sleep, steps, stress levels")
      ).toBeInTheDocument();
    });
  });

  it("shows dream analysis description text", async () => {
    renderWithProviders(<ExportPage />);
    await waitFor(() => {
      expect(
        screen.getByText("Dream journal entries with AI analysis")
      ).toBeInTheDocument();
    });
  });
});
