/**
 * Tests for HealthSyncStatusBar component.
 *
 * Covers:
 *  1. Renders "Sync Now" button
 *  2. Shows last sync time when synced
 *  3. Shows data summary (steps, HR) when data is present
 *  4. Shows empty-data guidance when no data on Android
 *  5. Shows empty-data guidance when no data on iOS
 *  6. Sync button shows spinner when syncing
 *  7. Shows "No data synced" when payload is empty
 */
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { HealthSyncStatusBar } from "@/components/health-sync-status-bar";

describe("HealthSyncStatusBar", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-03-20T12:00:00Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders Sync Now button", () => {
    renderWithProviders(
      <HealthSyncStatusBar
        status="idle"
        lastSyncAt={null}
        latestPayload={null}
        onSyncNow={() => {}}
        platform="android"
      />,
    );
    expect(screen.getByText("Sync Now")).toBeInTheDocument();
  });

  it("shows last sync time when synced", () => {
    const fiveMinAgo = new Date(Date.now() - 5 * 60 * 1000);
    renderWithProviders(
      <HealthSyncStatusBar
        status="ok"
        lastSyncAt={fiveMinAgo}
        latestPayload={{ user_id: "test", steps_today: 3000 }}
        onSyncNow={() => {}}
        platform="android"
      />,
    );
    expect(screen.getByText(/5 min ago/i)).toBeInTheDocument();
  });

  it("shows data summary when data is present", () => {
    renderWithProviders(
      <HealthSyncStatusBar
        status="ok"
        lastSyncAt={new Date()}
        latestPayload={{
          user_id: "test",
          steps_today: 5234,
          resting_heart_rate: 62,
        }}
        onSyncNow={() => {}}
        platform="android"
      />,
    );
    expect(screen.getByText(/5,234 steps/)).toBeInTheDocument();
    expect(screen.getByText(/62 bpm resting HR/)).toBeInTheDocument();
  });

  it("shows empty-data guidance on Android when no data", () => {
    renderWithProviders(
      <HealthSyncStatusBar
        status="ok"
        lastSyncAt={new Date()}
        latestPayload={{ user_id: "test" }}
        onSyncNow={() => {}}
        platform="android"
      />,
    );
    expect(screen.getByText(/Withings/)).toBeInTheDocument();
    expect(screen.getByText(/Health Connect/)).toBeInTheDocument();
  });

  it("shows empty-data guidance on iOS when no data", () => {
    renderWithProviders(
      <HealthSyncStatusBar
        status="ok"
        lastSyncAt={new Date()}
        latestPayload={{ user_id: "test" }}
        onSyncNow={() => {}}
        platform="ios"
      />,
    );
    expect(screen.getByText(/Apple Health/)).toBeInTheDocument();
  });

  it("shows spinner on Sync Now button when syncing", () => {
    renderWithProviders(
      <HealthSyncStatusBar
        status="syncing"
        lastSyncAt={null}
        latestPayload={null}
        onSyncNow={() => {}}
        platform="android"
      />,
    );
    expect(screen.getByText("Syncing...")).toBeInTheDocument();
  });

  it("shows 'No data synced' when payload has no usable data", () => {
    renderWithProviders(
      <HealthSyncStatusBar
        status="ok"
        lastSyncAt={new Date()}
        latestPayload={{ user_id: "test" }}
        onSyncNow={() => {}}
        platform="android"
      />,
    );
    expect(screen.getByText("No data synced")).toBeInTheDocument();
  });

  it("calls onSyncNow when button is clicked", () => {
    const onSync = vi.fn();
    renderWithProviders(
      <HealthSyncStatusBar
        status="idle"
        lastSyncAt={null}
        latestPayload={null}
        onSyncNow={onSync}
        platform="android"
      />,
    );
    fireEvent.click(screen.getByText("Sync Now"));
    expect(onSync).toHaveBeenCalledTimes(1);
  });
});
