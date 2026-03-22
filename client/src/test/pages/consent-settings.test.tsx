/**
 * Tests for the granular biometric consent settings page.
 *
 * Requirements:
 * - Shows per-modality toggles: EEG, Voice, Health, Nutrition, Location
 * - All toggles default to OFF for new users
 * - Accept All and Reject All buttons with equal visual weight
 * - Persists consent state
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import ConsentSettings from "@/pages/consent-settings";
import { getConsentState, saveConsentState, DEFAULT_CONSENT_STATE } from "@/lib/consent-store";

vi.mock("wouter", () => ({
  useLocation: () => ["/consent-settings", vi.fn()],
}));

describe("ConsentSettings page", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("renders without crashing", () => {
    renderWithProviders(<ConsentSettings />);
    expect(document.body).toBeTruthy();
  });

  it("shows the page heading", () => {
    renderWithProviders(<ConsentSettings />);
    expect(screen.getByText("Biometric Consent")).toBeInTheDocument();
  });

  it("shows all 5 modality toggles", () => {
    renderWithProviders(<ConsentSettings />);
    expect(screen.getByText("EEG Brain Data")).toBeInTheDocument();
    expect(screen.getByText("Voice Analysis")).toBeInTheDocument();
    expect(screen.getByText("Health Data Sync")).toBeInTheDocument();
    expect(screen.getByText("Nutrition Tracking")).toBeInTheDocument();
    expect(screen.getByText("Location")).toBeInTheDocument();
  });

  it("all toggles default to OFF for new users", () => {
    renderWithProviders(<ConsentSettings />);
    const state = getConsentState();
    expect(state.eeg).toBe(false);
    expect(state.voice).toBe(false);
    expect(state.health).toBe(false);
    expect(state.nutrition).toBe(false);
    expect(state.location).toBe(false);
  });

  it("shows Accept All button", () => {
    renderWithProviders(<ConsentSettings />);
    expect(screen.getByText("Accept All")).toBeInTheDocument();
  });

  it("shows Reject All button", () => {
    renderWithProviders(<ConsentSettings />);
    expect(screen.getByText("Reject All")).toBeInTheDocument();
  });

  it("Accept All enables all modalities", () => {
    renderWithProviders(<ConsentSettings />);
    fireEvent.click(screen.getByText("Accept All"));
    const state = getConsentState();
    expect(state.eeg).toBe(true);
    expect(state.voice).toBe(true);
    expect(state.health).toBe(true);
    expect(state.nutrition).toBe(true);
    expect(state.location).toBe(true);
  });

  it("Reject All disables all modalities", () => {
    // First enable all
    saveConsentState({
      eeg: true, voice: true, health: true, nutrition: true, location: true,
    });
    renderWithProviders(<ConsentSettings />);
    fireEvent.click(screen.getByText("Reject All"));
    const state = getConsentState();
    expect(state.eeg).toBe(false);
    expect(state.voice).toBe(false);
    expect(state.health).toBe(false);
    expect(state.nutrition).toBe(false);
    expect(state.location).toBe(false);
  });

  it("shows description text for each modality", () => {
    renderWithProviders(<ConsentSettings />);
    expect(screen.getByText(/Brainwave recordings/)).toBeInTheDocument();
    expect(screen.getByText(/Voice recordings/)).toBeInTheDocument();
    expect(screen.getByText(/Heart rate, HRV/)).toBeInTheDocument();
    expect(screen.getByText(/Meal logs/)).toBeInTheDocument();
    expect(screen.getByText(/Location data/)).toBeInTheDocument();
  });
});
