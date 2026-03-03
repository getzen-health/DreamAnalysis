import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import SleepSession from "@/pages/sleep-session";

vi.mock("@/hooks/use-device", () => ({
  useDevice: () => ({
    latestFrame: null,
    state: "disconnected",
    deviceStatus: null,
  }),
}));

vi.mock("wouter", () => ({
  useLocation: () => ["/sleep-session", vi.fn()],
  Link: (props: any) => <a href={props.href}>{props.children}</a>,
}));

vi.mock("@/lib/ml-api", () => ({
  startSession: vi.fn().mockResolvedValue({}),
  stopSession: vi.fn().mockResolvedValue({}),
}));

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    })
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("SleepSession page — idle state", () => {
  it("renders without crashing", () => {
    renderWithProviders(<SleepSession />);
  });

  it("shows the Sleep Session heading", () => {
    renderWithProviders(<SleepSession />);
    expect(screen.getByText("Sleep Session")).toBeInTheDocument();
  });

  it("shows the overnight EEG recording description", () => {
    renderWithProviders(<SleepSession />);
    expect(
      screen.getByText(/Overnight EEG recording/)
    ).toBeInTheDocument();
  });

  it("shows no-device simulation notice when device is not connected", () => {
    renderWithProviders(<SleepSession />);
    expect(
      screen.getByText(/No device connected — simulation mode will cycle through/)
    ).toBeInTheDocument();
  });

  it("shows the Recent Sleep section with stats", () => {
    renderWithProviders(<SleepSession />);
    expect(screen.getByText("Recent Sleep")).toBeInTheDocument();
    expect(screen.getByText("Last session")).toBeInTheDocument();
    expect(screen.getByText("Sleep quality")).toBeInTheDocument();
  });

  it("shows the 7-day avg stat in recent sleep section", () => {
    renderWithProviders(<SleepSession />);
    expect(screen.getByText("7-day avg")).toBeInTheDocument();
  });

  it("shows the Avg quality stat in recent sleep section", () => {
    renderWithProviders(<SleepSession />);
    expect(screen.getByText("Avg quality")).toBeInTheDocument();
  });

  it("shows the Start Sleep Session button", () => {
    renderWithProviders(<SleepSession />);
    expect(
      screen.getByRole("button", { name: /Start Sleep Session/i })
    ).toBeInTheDocument();
  });

  it("transitions to recording phase when Start is clicked", () => {
    renderWithProviders(<SleepSession />);
    const startBtn = screen.getByRole("button", { name: /Start Sleep Session/i });
    fireEvent.click(startBtn);
    expect(screen.getByText("Recording in progress")).toBeInTheDocument();
  });

  it("shows Wake Up button once recording begins", () => {
    renderWithProviders(<SleepSession />);
    const startBtn = screen.getByRole("button", { name: /Start Sleep Session/i });
    fireEvent.click(startBtn);
    expect(
      screen.getByRole("button", { name: /Wake up/i })
    ).toBeInTheDocument();
  });

  it("shows Recording in progress label after start", () => {
    renderWithProviders(<SleepSession />);
    const startBtn = screen.getByRole("button", { name: /Start Sleep Session/i });
    fireEvent.click(startBtn);
    expect(screen.getByText("Recording in progress")).toBeInTheDocument();
  });

  it("shows the Dreams detected card while recording", () => {
    renderWithProviders(<SleepSession />);
    const startBtn = screen.getByRole("button", { name: /Start Sleep Session/i });
    fireEvent.click(startBtn);
    expect(
      screen.getByText("Dreams detected this session")
    ).toBeInTheDocument();
  });
});
