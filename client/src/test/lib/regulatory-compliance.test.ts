import { describe, it, expect } from "vitest";
import {
  FDA_DISCLAIMER,
  FDA_CLASSIFICATION,
  EU_AI_ACT_NOTICE,
  EU_AI_ACT_CLASSIFICATION,
  GDPR_COMPLIANCE,
  CCPA_COMPLIANCE,
  BIPA_COMPLIANCE,
  GOOGLE_PLAY_HEALTH_DECLARATION,
  APPLE_HEALTH_GUIDELINES,
} from "@/lib/regulatory-compliance";

describe("regulatory-compliance", () => {
  describe("FDA_DISCLAIMER", () => {
    it("contains required 'not a medical device' language", () => {
      expect(FDA_DISCLAIMER).toContain("NOT a medical device");
    });

    it("states not intended to diagnose, treat, cure, or prevent", () => {
      expect(FDA_DISCLAIMER).toContain("not intended to diagnose, treat, cure, or prevent");
    });

    it("states not FDA cleared", () => {
      expect(FDA_DISCLAIMER).toContain("Not FDA cleared");
    });

    it("mentions wellness", () => {
      expect(FDA_DISCLAIMER).toContain("wellness");
    });
  });

  describe("FDA_CLASSIFICATION", () => {
    it("has wellness-exempt status", () => {
      expect(FDA_CLASSIFICATION.status).toContain("Wellness");
      expect(FDA_CLASSIFICATION.status).toContain("exempt");
    });

    it("lists permitted claims", () => {
      expect(FDA_CLASSIFICATION.permittedClaims.length).toBeGreaterThan(0);
    });

    it("lists prohibited claims", () => {
      expect(FDA_CLASSIFICATION.prohibitedClaims.length).toBeGreaterThan(0);
      expect(FDA_CLASSIFICATION.prohibitedClaims.some(c => c.includes("Diagnose"))).toBe(true);
    });
  });

  describe("EU_AI_ACT_NOTICE", () => {
    it("mentions Annex III high-risk classification", () => {
      expect(EU_AI_ACT_NOTICE).toContain("high-risk AI");
      expect(EU_AI_ACT_NOTICE).toContain("Annex III");
    });

    it("states personal wellness use only", () => {
      expect(EU_AI_ACT_NOTICE).toContain("personal wellness use only");
    });

    it("mentions users can disable emotion recognition", () => {
      expect(EU_AI_ACT_NOTICE).toContain("disable emotion recognition");
    });

    it("states not for workplace or educational settings", () => {
      expect(EU_AI_ACT_NOTICE).toContain("not deployed in workplace or educational");
    });
  });

  describe("EU_AI_ACT_CLASSIFICATION", () => {
    it("is classified as high-risk", () => {
      expect(EU_AI_ACT_CLASSIFICATION.riskLevel).toBe("high-risk");
    });

    it("lists mitigations", () => {
      expect(EU_AI_ACT_CLASSIFICATION.mitigations.length).toBeGreaterThan(0);
    });

    it("lists transparency obligations", () => {
      expect(EU_AI_ACT_CLASSIFICATION.transparencyObligations.length).toBeGreaterThan(0);
    });
  });

  describe("GDPR_COMPLIANCE", () => {
    it("lists all key data subject rights", () => {
      const rights = GDPR_COMPLIANCE.dataSubjectRights;
      expect(rights.some(r => r.includes("Article 15"))).toBe(true); // access
      expect(rights.some(r => r.includes("Article 17"))).toBe(true); // erasure
      expect(rights.some(r => r.includes("Article 20"))).toBe(true); // portability
    });

    it("specifies legal basis for biometric data", () => {
      expect(GDPR_COMPLIANCE.legalBasis).toContain("Article 9");
    });

    it("has a contact email", () => {
      expect(GDPR_COMPLIANCE.contactEmail).toContain("@");
    });
  });

  describe("CCPA_COMPLIANCE", () => {
    it("does not sell data", () => {
      expect(CCPA_COMPLIANCE.saleOfData).toBe(false);
    });

    it("lists consumer rights", () => {
      expect(CCPA_COMPLIANCE.consumerRights.length).toBeGreaterThan(0);
    });

    it("includes biometric information in data categories", () => {
      expect(CCPA_COMPLIANCE.dataCategories.some(c => c.includes("Biometric"))).toBe(true);
    });
  });

  describe("BIPA_COMPLIANCE", () => {
    it("notes EEG and voice biometric coverage", () => {
      expect(BIPA_COMPLIANCE.applicability).toContain("EEG");
      expect(BIPA_COMPLIANCE.applicability).toContain("voice");
    });

    it("lists obligations", () => {
      expect(BIPA_COMPLIANCE.obligations.length).toBeGreaterThan(0);
    });
  });

  describe("GOOGLE_PLAY_HEALTH_DECLARATION", () => {
    it("does not make medical device claims", () => {
      expect(GOOGLE_PLAY_HEALTH_DECLARATION.medicalDeviceClaim).toBe(false);
    });

    it("declares health data collection", () => {
      expect(GOOGLE_PLAY_HEALTH_DECLARATION.dataHandling.collectsHealthData).toBe(true);
    });

    it("supports data deletion", () => {
      expect(GOOGLE_PLAY_HEALTH_DECLARATION.dataHandling.userCanRequestDeletion).toBe(true);
    });

    it("supports data export", () => {
      expect(GOOGLE_PLAY_HEALTH_DECLARATION.dataHandling.userCanExportData).toBe(true);
    });

    it("uses FDA disclaimer", () => {
      expect(GOOGLE_PLAY_HEALTH_DECLARATION.disclaimer).toBe(FDA_DISCLAIMER);
    });
  });

  describe("APPLE_HEALTH_GUIDELINES", () => {
    it("lists HealthKit data types", () => {
      expect(APPLE_HEALTH_GUIDELINES.healthKitDataTypes.length).toBeGreaterThan(0);
    });

    it("has purpose strings for required permissions", () => {
      expect(APPLE_HEALTH_GUIDELINES.purposeStrings.healthKit).toBeTruthy();
      expect(APPLE_HEALTH_GUIDELINES.purposeStrings.microphone).toBeTruthy();
      expect(APPLE_HEALTH_GUIDELINES.purposeStrings.bluetooth).toBeTruthy();
    });
  });
});
