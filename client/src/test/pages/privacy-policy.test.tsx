import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import PrivacyPolicy from "@/pages/privacy-policy";

describe("Privacy Policy page", () => {
  it("renders without crashing", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(document.body).toBeTruthy();
  });

  it("shows the page heading", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(screen.getByText("Privacy Policy")).toBeInTheDocument();
  });

  it("shows effective date", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(screen.getByText(/effective March 2026/)).toBeInTheDocument();
  });

  it("shows What We Collect section", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(screen.getByText("What We Collect")).toBeInTheDocument();
  });

  it("mentions EEG data collection", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText(/EEG brainwave data recorded during sessions/)
    ).toBeInTheDocument();
  });

  it("mentions voice analysis", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(screen.getByText("Voice Analysis")).toBeInTheDocument();
  });

  it("mentions voice data is processed on-device", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText(/Voice emotion analysis is processed on-device/)
    ).toBeInTheDocument();
  });

  it("mentions nutrition and GLP-1 data", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText("Nutrition and GLP-1 Data")
    ).toBeInTheDocument();
  });

  it("mentions GLP-1 injection records", () => {
    renderWithProviders(<PrivacyPolicy />);
    const matches = screen.getAllByText(/GLP-1 injection/);
    expect(matches.length).toBeGreaterThan(0);
  });

  it("shows Data Storage and Security section", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText("Data Storage and Security")
    ).toBeInTheDocument();
  });

  it("mentions AES-256 encryption", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText(/AES-256/)
    ).toBeInTheDocument();
  });

  it("shows Third-Party Integrations section", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText("Third-Party Integrations")
    ).toBeInTheDocument();
  });

  it("mentions Apple HealthKit", () => {
    renderWithProviders(<PrivacyPolicy />);
    const matches = screen.getAllByText(/Apple HealthKit/);
    expect(matches.length).toBeGreaterThan(0);
  });

  it("mentions Google Health Connect", () => {
    renderWithProviders(<PrivacyPolicy />);
    const matches = screen.getAllByText(/Google Health Connect/);
    expect(matches.length).toBeGreaterThan(0);
  });

  it("mentions Oura, WHOOP, Garmin", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText(/Oura, WHOOP, Garmin/)
    ).toBeInTheDocument();
  });

  it("shows Data Export section", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(screen.getByText("Data Export")).toBeInTheDocument();
  });

  it("mentions GDPR Article 20", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText(/GDPR Article 20/)
    ).toBeInTheDocument();
  });

  it("shows Data Deletion section", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(screen.getByText("Data Deletion")).toBeInTheDocument();
  });

  it("mentions GDPR Article 17", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText(/GDPR Article 17/)
    ).toBeInTheDocument();
  });

  it("mentions 30-day grace period", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText(/30-day grace period/)
    ).toBeInTheDocument();
  });

  it("shows Medical Disclaimer section", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(screen.getByText("Medical Disclaimer")).toBeInTheDocument();
  });

  it("states app is not a medical device", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText(/NOT a medical device/)
    ).toBeInTheDocument();
  });

  it("mentions no selling of data", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText(/do not sell, rent, or share your personal data/)
    ).toBeInTheDocument();
  });

  it("shows Contact section", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText("Contact and Data Requests")
    ).toBeInTheDocument();
  });

  it("mentions CCPA compliance", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText(/CCPA/)
    ).toBeInTheDocument();
  });

  it("shows last updated notice", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText(/Last updated: March 2026/)
    ).toBeInTheDocument();
  });

  it("shows EEG Data section", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(screen.getByText("EEG Data")).toBeInTheDocument();
  });

  it("mentions Research Participation", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(screen.getByText("Research Participation")).toBeInTheDocument();
  });

  it("shows EU AI Act Notice section", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(screen.getByText("EU AI Act Notice")).toBeInTheDocument();
  });

  it("mentions high-risk AI classification under EU AI Act", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText(/classified as high-risk AI under EU AI Act Annex III/)
    ).toBeInTheDocument();
  });

  it("mentions users can disable emotion recognition", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText(/disable emotion recognition at any time via Biometric Consent/)
    ).toBeInTheDocument();
  });

  it("states system is for personal wellness only", () => {
    renderWithProviders(<PrivacyPolicy />);
    expect(
      screen.getByText(/designed for personal wellness use only/)
    ).toBeInTheDocument();
  });
});
