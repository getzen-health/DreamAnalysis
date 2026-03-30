import React from "react";
import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import SmartAlarmSettings from "@/components/smart-alarm-settings";

const defaults = {
  enabled: true,
  onEnabledChange: vi.fn(),
  targetWakeTime: "07:30",
  onTargetWakeTimeChange: vi.fn(),
  windowMinutes: 30,
  onWindowMinutesChange: vi.fn(),
};

describe("SmartAlarmSettings", () => {
  it("renders with data-testid", () => {
    renderWithProviders(<SmartAlarmSettings {...defaults} />);
    expect(screen.getByTestId("smart-alarm-settings")).toBeInTheDocument();
  });

  it("shows toggle in 'On' state when enabled", () => {
    renderWithProviders(<SmartAlarmSettings {...defaults} enabled={true} />);
    expect(screen.getByText("On")).toBeInTheDocument();
  });

  it("shows toggle in 'Off' state when disabled", () => {
    renderWithProviders(<SmartAlarmSettings {...defaults} enabled={false} />);
    expect(screen.getByText("Off")).toBeInTheDocument();
  });

  it("renders time picker with correct value", () => {
    renderWithProviders(<SmartAlarmSettings {...defaults} targetWakeTime="06:45" />);
    const input = screen.getByTestId("smart-alarm-time") as HTMLInputElement;
    expect(input.value).toBe("06:45");
  });

  it("calls onTargetWakeTimeChange when time is changed", () => {
    const onTimeChange = vi.fn();
    renderWithProviders(
      <SmartAlarmSettings {...defaults} onTargetWakeTimeChange={onTimeChange} />,
    );
    const input = screen.getByTestId("smart-alarm-time");
    fireEvent.change(input, { target: { value: "08:00" } });
    expect(onTimeChange).toHaveBeenCalledWith("08:00");
  });

  it("renders all three window options", () => {
    renderWithProviders(<SmartAlarmSettings {...defaults} />);
    expect(screen.getByTestId("smart-alarm-window-15")).toBeInTheDocument();
    expect(screen.getByTestId("smart-alarm-window-30")).toBeInTheDocument();
    expect(screen.getByTestId("smart-alarm-window-45")).toBeInTheDocument();
  });

  it("highlights the selected window option", () => {
    renderWithProviders(<SmartAlarmSettings {...defaults} windowMinutes={45} />);
    const btn45 = screen.getByTestId("smart-alarm-window-45");
    expect(btn45.className).toContain("border-primary");
  });

  it("calls onWindowMinutesChange when a window button is clicked", () => {
    const onWindowChange = vi.fn();
    renderWithProviders(
      <SmartAlarmSettings {...defaults} onWindowMinutesChange={onWindowChange} />,
    );
    fireEvent.click(screen.getByTestId("smart-alarm-window-15"));
    expect(onWindowChange).toHaveBeenCalledWith(15);
  });

  it("dims controls when disabled", () => {
    const { container } = renderWithProviders(
      <SmartAlarmSettings {...defaults} enabled={false} />,
    );
    // The controls wrapper should have opacity-50
    const dimmedDiv = container.querySelector(".opacity-50");
    expect(dimmedDiv).toBeTruthy();
  });
});
