import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { EFSInsightBanner } from "@/components/efs-insight-banner";

// ── Clipboard mock ────────────────────────────────────────────────────────────

const writeTextMock = vi.fn().mockResolvedValue(undefined);

beforeEach(() => {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: writeTextMock },
  });
  // Ensure navigator.share is not available so clipboard path is used
  Object.defineProperty(navigator, "share", {
    configurable: true,
    value: undefined,
  });
});

afterEach(() => {
  vi.clearAllMocks();
});

// ── Fixtures ──────────────────────────────────────────────────────────────────

const INSIGHT = {
  text: "You're a Suppressor",
  type: "awareness_gap",
  actionNudge: "Try naming emotions",
};

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("EFSInsightBanner", () => {
  it("renders insight text and action nudge", () => {
    renderWithProviders(<EFSInsightBanner insight={INSIGHT} />);
    expect(screen.getByText("You're a Suppressor")).toBeInTheDocument();
    expect(screen.getByText("Try naming emotions")).toBeInTheDocument();
  });

  it("does not render when insight is null", () => {
    const { container } = renderWithProviders(<EFSInsightBanner insight={null} />);
    expect(container.innerHTML).toBe("");
  });

  it("has a share button with correct aria-label", () => {
    renderWithProviders(<EFSInsightBanner insight={INSIGHT} />);
    expect(screen.getByLabelText("Share insight")).toBeInTheDocument();
  });

  it("copies text to clipboard when share button is clicked", async () => {
    renderWithProviders(<EFSInsightBanner insight={INSIGHT} />);

    const shareBtn = screen.getByLabelText("Share insight");
    fireEvent.click(shareBtn);

    expect(writeTextMock).toHaveBeenCalledOnce();
    const clipboardText: string = writeTextMock.mock.calls[0][0];
    expect(clipboardText).toContain("You're a Suppressor");
    expect(clipboardText).toContain("Try naming emotions");
  });

  it("renders without action nudge when actionNudge is empty", () => {
    const noNudge = { ...INSIGHT, actionNudge: "" };
    renderWithProviders(<EFSInsightBanner insight={noNudge} />);
    expect(screen.getByText("You're a Suppressor")).toBeInTheDocument();
    expect(screen.queryByText("Try naming emotions")).not.toBeInTheDocument();
  });
});
